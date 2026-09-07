// The VLMK numeric kernel shared by llama-llm-kld and llama-vlm-kld:
// token reading, the record layout, the float64 metric kernel and its
// parallel driver, the INDEPENDENT naive reference, and the self-test.
// Moved VERBATIM from the two byte-identical copies (588-line md5 match).
// NUMERICAL_CONTRACT.md #1/#2: accumulation order and the naive path's
// independence are frozen -- do not reorder, fuse, or unify.
#pragma once

#include "skymizer-common.h"

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#else
#    include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// metric kernel
// ---------------------------------------------------------------------------

// One scored answer position. Field order/types are the on-disk record; keep
// in sync with kld_metrics_io.py's KLD_RECORD_DT.
struct kld_record {
    float   kld;            // KL(p_ref || p_cand), nats
    float   reversed_kld;   // KL(p_cand || p_ref), nats
    float   js_kld;         // Jensen-Shannon divergence, nats
    float   nll_ref;        // -log p_ref(target)
    float   nll_cand;       // -log p_cand(target)
    float   entropy_ref;    // -sum p_ref * log p_ref
    float   entropy_cand;   // -sum p_cand * log p_cand
    float   ear;            // sum min(p_ref, p_cand) = 1 - TV (Expected Acceptance Rate)
    float   ear_20;         // sum over the reference's top-20 slots of min(p_ref, p_cand) (v4)
    float   ear_10;         // ... top-10 (v4)
    float   ear_5;          // ... top-5 (v4)
    float   ear_20_normalized;  // same slots, both rows renormalized on them first (v4)
    float   ear_10_normalized;  // ... top-10 (v4)
    float   ear_5_normalized;   // ... top-5 (v4)
    int32_t target;         // teacher-forced target token id
    int32_t argmax_ref;     // reference argmax token id
    int32_t argmax_cand;    // candidate argmax token id
    float   ear_64;         // reference top-64 share of EAR (v5, appended after the v4 record)
    float   ear_64_normalized; // both rows renormalized on the reference top-64 (v5)
};
static_assert(offsetof(kld_record, ear_64) == 68, "v5 must preserve the v4 record prefix");
static_assert(sizeof(kld_record) == 76, "kld_record must be 76 packed bytes (on-disk layout)");

static constexpr uint32_t VLMK_MAGIC   = 0x564C4D4B; // "VLMK"
static constexpr uint32_t VLMK_VERSION = 5;          // v5: 76-byte records (+ EAR_64); v4 = 68, v3 = 56, v2 = 44, v1 = 40

// Write to a temporary file and rename after the complete file is synced.
static bool write_vlmk_file(
        const std::string & path,
        uint32_t n_vocab,
        uint32_t n_prefill,
        uint32_t n_past_actual,
        const std::vector<kld_record> & records,
        stderr_prefix * lp) {
    const std::string tmp_path = path + ".tmp";
    FILE * f = fopen(tmp_path.c_str(), "wb");
    if (f == nullptr) {
        prefixed_fprintf(lp, "failed to open output %s\n", tmp_path.c_str());
        return false;
    }
    bool ok = true;
    // Sixth word records the caller's consumed position count.
    const uint32_t header[6] = {
        VLMK_MAGIC, VLMK_VERSION, n_vocab,
        (uint32_t) records.size(), n_prefill, n_past_actual,
    };
    ok = ok && fwrite(header, sizeof(header), 1, f) == 1;
    ok = ok && (records.empty() ||
                fwrite(records.data(), sizeof(kld_record), records.size(), f) == records.size());
    ok = ok && fflush(f) == 0;
#if defined(_WIN32)
    ok = ok && _commit(_fileno(f)) == 0;
#else
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        prefixed_fprintf(lp, "failed writing %s: %s\n", tmp_path.c_str(), strerror(errno));
        std::remove(tmp_path.c_str());
        return false;
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        prefixed_fprintf(lp, "failed to rename %s -> %s: %s\n",
                         tmp_path.c_str(), path.c_str(), strerror(errno));
        std::remove(tmp_path.c_str());
        return false;
    }
    return true;
}

static bool kld_record_all_finite(const kld_record & rec) {
    const float values[] = {
        rec.kld, rec.reversed_kld, rec.js_kld, rec.nll_ref,
        rec.nll_cand, rec.entropy_ref, rec.entropy_cand, rec.ear,
        rec.ear_20, rec.ear_10, rec.ear_5,
        rec.ear_20_normalized, rec.ear_10_normalized, rec.ear_5_normalized,
        rec.ear_64, rec.ear_64_normalized,
    };
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool validate_kld_record_finite(
        const kld_record & rec, size_t position, stderr_prefix * lp) {
    const float values[] = {
        rec.kld, rec.reversed_kld, rec.js_kld, rec.nll_ref,
        rec.nll_cand, rec.entropy_ref, rec.entropy_cand, rec.ear,
        rec.ear_20, rec.ear_10, rec.ear_5,
        rec.ear_20_normalized, rec.ear_10_normalized, rec.ear_5_normalized,
        rec.ear_64, rec.ear_64_normalized,
    };
    const char * names[] = {
        "kld", "reversed_kld", "js_kld", "nll_ref",
        "nll_cand", "entropy_ref", "entropy_cand", "ear",
        "ear_20", "ear_10", "ear_5",
        "ear_20_normalized", "ear_10_normalized", "ear_5_normalized",
        "ear_64", "ear_64_normalized",
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        if (!std::isfinite(values[i])) {
            prefixed_fprintf(lp, "non-finite metric %s at answer position %zu; refusing VLMK output\n",
                             names[i], position);
            return false;
        }
    }
    return true;
}

// The same metrics in DOUBLE, i.e. exactly what the kernel accumulates before
// kld_record's float32 store. compute_kld_record is a thin cast of this, and
// the self-test compares HERE so that a regression smaller than the stored
// float's own resolution is still visible: swapping `ear`'s accumulator from
// double to float moves it by ~3e-7, which is under 5 ulp of a float32 near
// 0.99 and therefore invisible in the stored record.
struct kld_values {
    double  kld          = 0.0;
    double  reversed_kld = 0.0;
    double  js_kld       = 0.0;
    double  nll_ref      = 0.0;
    double  nll_cand     = 0.0;
    double  entropy_ref  = 0.0;
    double  entropy_cand = 0.0;
    double  ear          = 0.0;
    double  ear_20       = 0.0;
    double  ear_10       = 0.0;
    double  ear_5        = 0.0;
    double  ear_20_normalized = 0.0;
    double  ear_10_normalized = 0.0;
    double  ear_5_normalized  = 0.0;
    double  ear_64       = 0.0;
    double  ear_64_normalized = 0.0;
    int32_t target       = 0;
    int32_t argmax_ref   = 0;
    int32_t argmax_cand  = 0;
};

static kld_record to_kld_record(const kld_values & v) {
    kld_record rec;
    rec.kld          = (float) v.kld;
    rec.reversed_kld = (float) v.reversed_kld;
    rec.js_kld       = (float) v.js_kld;
    rec.nll_ref      = (float) v.nll_ref;
    rec.nll_cand     = (float) v.nll_cand;
    rec.entropy_ref  = (float) v.entropy_ref;
    rec.entropy_cand = (float) v.entropy_cand;
    rec.ear          = (float) v.ear;
    rec.ear_20       = (float) v.ear_20;
    rec.ear_10       = (float) v.ear_10;
    rec.ear_5        = (float) v.ear_5;
    rec.ear_20_normalized = (float) v.ear_20_normalized;
    rec.ear_10_normalized = (float) v.ear_10_normalized;
    rec.ear_5_normalized  = (float) v.ear_5_normalized;
    rec.target       = v.target;
    rec.argmax_ref   = v.argmax_ref;
    rec.argmax_cand  = v.argmax_cand;
    rec.ear_64       = (float) v.ear_64;
    rec.ear_64_normalized = (float) v.ear_64_normalized;
    return rec;
}

struct row_stats {
    double  max   = 0.0;   // max logit (log-softmax shift)
    double  log_z = 0.0;   // log sum exp(logit), i.e. lp(i) = logit(i) - log_z
    int32_t argmax = 0;
};

static row_stats compute_row_stats(const float * x, int n) {
    row_stats s;
    s.max = x[0];
    s.argmax = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > s.max) {
            s.max = x[i];
            s.argmax = i;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp((double) x[i] - s.max);
    }
    s.log_z = s.max + std::log(sum);
    return s;
}

// EAR_K family (v4: K=5/10/20; v5 adds K=64), on the REFERENCE's K most likely tokens (the K vocab
// slots with the largest reference logits, ties -> lower index; K clamped to
// n_vocab):
//   ear_K            = sum over those slots of min(p_ref, p_cand) with the
//                      FULL-vocab probabilities: the part of `ear` that the
//                      reference's top-K contributes. Monotone in K and
//                      <= ear; bounded by the reference's own top-K mass.
//   ear_K_normalized = both rows renormalized over exactly those K slots
//                      (a softmax over the K logits), then sum of mins: how
//                      well the candidate reproduces the reference's RELATIVE
//                      preferences among its K most likely tokens. 1.0 =
//                      identical shape on the set; candidate mass outside the
//                      set is ignored, so it is NOT monotone in K. 0 when the
//                      candidate has no mass on any of the K slots.
// For n_vocab <= K both equal `ear` up to rounding.
static constexpr int EAR_TOPK_MAX = 64;

// One pass with a sorted insertion buffer: once K slots are filled the common
// case is a single failed compare per vocab entry. Strict `>` on entry and
// strict `<` while shifting keep ties in ascending-index order.
static int select_ref_topk(const float * ref, int n_vocab, int32_t * top_idx, float * top_val) {
    const int k_max = std::min(EAR_TOPK_MAX, n_vocab);
    int n_top = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float x = ref[i];
        if (n_top == k_max && !(x > top_val[n_top - 1])) {
            continue;
        }
        int j = (n_top == k_max) ? n_top - 1 : n_top;
        while (j > 0 && top_val[j - 1] < x) {
            top_val[j] = top_val[j - 1];
            top_idx[j] = top_idx[j - 1];
            --j;
        }
        top_val[j] = x;
        top_idx[j] = i;
        if (n_top < k_max) {
            ++n_top;
        }
    }
    return n_top;
}

struct ear_topk_pair {
    double mass       = 0.0;   // ear_K
    double normalized = 0.0;   // ear_K_normalized
};

// Both EAR_K variants over the first k entries of a descending top list (all in
// double; fixed ascending-j summation order). log_z_r / log_z_c are the full
// rows' log-partition functions (row_stats::log_z).
static ear_topk_pair ear_topk_values(const float * cand, const int32_t * top_idx, const float * top_val,
                                     int k, double log_z_r, double log_z_c) {
    const double neg_inf = -std::numeric_limits<double>::infinity();
    ear_topk_pair out;
    if (k <= 0) {
        return out;
    }
    // ear_K: full-vocab probabilities, min selected via the log-probs like the
    // main loop does.
    for (int j = 0; j < k; ++j) {
        const double lp_r = (double) top_val[j] - log_z_r;
        const double lp_c = (double) cand[top_idx[j]] - log_z_c;
        out.mass += std::exp(std::min(lp_r, lp_c));
    }
    // ear_K_normalized: softmax over the k selected logits on each side.
    const double r_max = top_val[0];
    double c_max = neg_inf;
    for (int j = 0; j < k; ++j) {
        c_max = std::max(c_max, (double) cand[top_idx[j]]);
    }
    if (!(r_max > neg_inf) || !(c_max > neg_inf)) {
        return out;   // a fully masked side has no distribution on the set
    }
    double z_r = 0.0, z_c = 0.0;
    for (int j = 0; j < k; ++j) {
        z_r += std::exp((double) top_val[j] - r_max);
        z_c += std::exp((double) cand[top_idx[j]] - c_max);
    }
    for (int j = 0; j < k; ++j) {
        const double p_r = std::exp((double) top_val[j] - r_max) / z_r;
        const double p_c = std::exp((double) cand[top_idx[j]] - c_max) / z_c;
        out.normalized += std::min(p_r, p_c);
    }
    return out;
}

// Compute all metrics for one position from two full-vocab fp32 logit rows.
// Everything is accumulated in double; the record stores float32. The JSD
// mixture is computed in log space (logaddexp) so vanishing tail probabilities
// never produce 0*log(0) — mirrors paired_compare.py's dense math.
//
// Zero-probability entries (a -inf logit, e.g. a masked vocab slot, or exp
// underflow) contribute exactly 0 to their own side's sums — the p > 0 guards
// keep a -inf logit from turning 0 * inf into NaN. For finite logits the
// guards never change the result (an underflowed p contributed exactly 0.0
// already). KL is still legitimately +inf when the OTHER side has mass where
// this side has none.
static kld_values compute_kld_values(const float * ref, const float * cand, int n_vocab, int32_t target) {
    static const double LOG2 = std::log(2.0);

    const row_stats sr = compute_row_stats(ref,  n_vocab);
    const row_stats sc = compute_row_stats(cand, n_vocab);

    double kld = 0.0, rkld = 0.0, jsd = 0.0, ent_r = 0.0, ent_c = 0.0, ear = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double lp_r = (double) ref[i]  - sr.log_z;
        const double lp_c = (double) cand[i] - sc.log_z;
        const double p_r  = std::exp(lp_r);
        const double p_c  = std::exp(lp_c);
        if (p_r <= 0.0 && p_c <= 0.0) {
            continue;
        }
        if (p_r > 0.0) {
            kld   += p_r * (lp_r - lp_c);
            ent_r -= p_r * lp_r;
        }
        if (p_c > 0.0) {
            rkld  += p_c * (lp_c - lp_r);
            ent_c -= p_c * lp_c;
        }
        // min(p_r, p_c), selected via the log-probs so no second exp() runs
        // in this ~vocab-sized inner loop: exp is monotone, and a one-sided
        // masked slot (lp = -inf -> p = 0) contributes exactly 0.
        ear += (lp_c < lp_r) ? p_c : p_r;
        // log m(i) = logaddexp(lp_r, lp_c) - log 2, evaluated stably. At least
        // one lp is finite here, so hi is finite and lo - hi never yields
        // inf - inf.
        const double hi    = std::max(lp_r, lp_c);
        const double lo    = std::min(lp_r, lp_c);
        const double log_m = hi + std::log1p(std::exp(lo - hi)) - LOG2;
        if (p_r > 0.0) jsd += 0.5 * p_r * (lp_r - log_m);
        if (p_c > 0.0) jsd += 0.5 * p_c * (lp_c - log_m);
    }

    kld_values v;
    v.kld          = kld;
    v.reversed_kld = rkld;
    v.js_kld       = jsd;
    v.nll_ref      = sr.log_z - (double) ref[target];
    v.nll_cand     = sc.log_z - (double) cand[target];
    v.entropy_ref  = ent_r;
    v.entropy_cand = ent_c;
    v.ear          = ear;
    v.target       = target;
    v.argmax_ref   = sr.argmax;
    v.argmax_cand  = sc.argmax;
    // v4: the EAR_K family from a separate top-K pass; the frozen loop above
    // is untouched.
    {
        int32_t top_idx[EAR_TOPK_MAX];
        float   top_val[EAR_TOPK_MAX];
        const int n_top = select_ref_topk(ref, n_vocab, top_idx, top_val);
        const ear_topk_pair e20 = ear_topk_values(cand, top_idx, top_val, std::min(20, n_top), sr.log_z, sc.log_z);
        const ear_topk_pair e10 = ear_topk_values(cand, top_idx, top_val, std::min(10, n_top), sr.log_z, sc.log_z);
        const ear_topk_pair e5  = ear_topk_values(cand, top_idx, top_val, std::min(5,  n_top), sr.log_z, sc.log_z);
        v.ear_20 = e20.mass;  v.ear_20_normalized = e20.normalized;
        v.ear_10 = e10.mass;  v.ear_10_normalized = e10.normalized;
        v.ear_5  = e5.mass;   v.ear_5_normalized  = e5.normalized;
        const ear_topk_pair e64 = ear_topk_values(cand, top_idx, top_val, n_top, sr.log_z, sc.log_z);
        v.ear_64 = e64.mass;  v.ear_64_normalized = e64.normalized;
    }
    return v;
}

static kld_record compute_kld_record(const float * ref, const float * cand, int n_vocab, int32_t target) {
    return to_kld_record(compute_kld_values(ref, cand, n_vocab, target));
}

// Compute records for a chunk of positions in parallel. Each position is
// independent (two const rows in, one record out), so any thread assignment
// yields identical bits. Row pointers are gathered by the caller on the main
// thread; workers only read plain float arrays.
static void compute_records_parallel(
        const std::vector<const float *> & ref_rows,
        const std::vector<const float *> & cand_rows,
        const std::vector<int32_t> & targets,
        int n_vocab,
        int n_threads,
        kld_record * out) {
    const int n = (int) ref_rows.size();
    n_threads = std::max(1, std::min(n_threads, n));
    if (n_threads == 1) {
        for (int i = 0; i < n; ++i) {
            out[i] = compute_kld_record(ref_rows[i], cand_rows[i], n_vocab, targets[i]);
        }
        return;
    }
    std::atomic<int> next(0);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
                out[i] = compute_kld_record(ref_rows[i], cand_rows[i], n_vocab, targets[i]);
            }
        });
    }
    for (auto & w : workers) {
        w.join();
    }
}

static int resolve_metric_threads(int requested) {
    if (requested >= 1) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int) hc : 4;
}

// ---------------------------------------------------------------------------
// metric kernel self test (--self-test; no models needed)
// ---------------------------------------------------------------------------

// Straightforward reference implementation: materialize both softmaxes, then
// apply the textbook formulas directly. Deliberately structured differently
// from compute_kld_record (probability-space mixture vs log-space) so the two
// can cross-check each other.
static kld_values naive_kld_values(const std::vector<float> & ref, const std::vector<float> & cand, int32_t target) {
    const int n = (int) ref.size();
    auto softmax = [n](const std::vector<float> & x) {
        std::vector<double> p(n);
        double mx = x[0];
        for (int i = 1; i < n; ++i) mx = std::max(mx, (double) x[i]);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) { p[i] = std::exp((double) x[i] - mx); sum += p[i]; }
        for (int i = 0; i < n; ++i) p[i] /= sum;
        return p;
    };
    const std::vector<double> p = softmax(ref);
    const std::vector<double> q = softmax(cand);
    double kld = 0.0, rkld = 0.0, jsd = 0.0, ent_r = 0.0, ent_c = 0.0, ear = 0.0;
    for (int i = 0; i < n; ++i) {
        const double m = 0.5 * (p[i] + q[i]);
        ear += std::min(p[i], q[i]);
        if (p[i] > 0.0) {
            kld   += p[i] * std::log(p[i] / q[i]);
            jsd   += 0.5 * p[i] * std::log(p[i] / m);
            ent_r -= p[i] * std::log(p[i]);
        }
        if (q[i] > 0.0) {
            rkld  += q[i] * std::log(q[i] / p[i]);
            jsd   += 0.5 * q[i] * std::log(q[i] / m);
            ent_c -= q[i] * std::log(q[i]);
        }
    }
    kld_values v;
    v.kld          = kld;
    v.reversed_kld = rkld;
    v.js_kld       = jsd;
    v.nll_ref      = -std::log(p[target]);
    v.nll_cand     = -std::log(q[target]);
    v.entropy_ref  = ent_r;
    v.entropy_cand = ent_c;
    v.ear          = ear;
    v.target       = target;
    v.argmax_ref   = (int32_t) (std::max_element(p.begin(), p.end()) - p.begin());
    v.argmax_cand  = (int32_t) (std::max_element(q.begin(), q.end()) - q.begin());
    // EAR_K the textbook way: rank slots by the materialized reference
    // probability (stable sort -> lower index first on ties), renormalize by
    // the partial sums, sum the mins. Deliberately probability-space, unlike
    // the kernel's log-space softmax over the selected logits.
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&p](int a, int b) { return p[a] > p[b]; });
    auto ear_k_mass = [&](int k) {
        k = std::min(k, n);
        double e = 0.0;
        for (int j = 0; j < k; ++j) e += std::min(p[order[j]], q[order[j]]);
        return e;
    };
    auto ear_k_normalized = [&](int k) {
        k = std::min(k, n);
        double sp = 0.0, sq = 0.0;
        for (int j = 0; j < k; ++j) { sp += p[order[j]]; sq += q[order[j]]; }
        if (!(sp > 0.0) || !(sq > 0.0)) return 0.0;
        double e = 0.0;
        for (int j = 0; j < k; ++j) e += std::min(p[order[j]] / sp, q[order[j]] / sq);
        return e;
    };
    v.ear_20 = ear_k_mass(20);
    v.ear_10 = ear_k_mass(10);
    v.ear_5  = ear_k_mass(5);
    v.ear_20_normalized = ear_k_normalized(20);
    v.ear_10_normalized = ear_k_normalized(10);
    v.ear_5_normalized  = ear_k_normalized(5);
    v.ear_64 = ear_k_mass(64);
    v.ear_64_normalized = ear_k_normalized(64);
    return v;
}

static kld_record naive_kld_record(const std::vector<float> & ref, const std::vector<float> & cand, int32_t target) {
    return to_kld_record(naive_kld_values(ref, cand, target));
}

static bool self_test_close(const char * what, double got, double want, double tol) {
    if (std::abs(got - want) <= tol) {
        return true;
    }
    fprintf(stderr, "self-test FAIL: %s: got %.9g want %.9g (tol %.1g)\n", what, got, want, tol);
    return false;
}

// Relative form, for the production-regime trial: an absolute tolerance
// picked at one test point says nothing at another. `want` is the naive
// double value, so a zero would demand exactness -- no metric checked with
// this is ever zero at that test point (asserted there).
static bool self_test_close_rel(const char * what, double got, double want, double rtol) {
    const double err = std::abs(got - want);
    if (err <= rtol * std::abs(want)) {
        return true;
    }
    fprintf(stderr, "self-test FAIL: %s: got %.17g want %.17g "
                    "(relative %.3g > rtol %.1g)\n",
            what, got, want, err / std::abs(want), rtol);
    return false;
}

static bool run_self_test() {
    bool ok = true;

    // 1) closed-form 2-token case: p_ref = [1/2, 1/2], p_cand = [1/4, 3/4]
    {
        const std::vector<float> r = {0.0f, 0.0f};
        const std::vector<float> c = {0.0f, (float) std::log(3.0)};
        const kld_record rec = compute_kld_record(r.data(), c.data(), 2, 1);
        const double kld  = 0.5 * std::log(0.5 / 0.25) + 0.5 * std::log(0.5 / 0.75);
        const double rkld = 0.25 * std::log(0.25 / 0.5) + 0.75 * std::log(0.75 / 0.5);
        const double jsd  = 0.5 * (0.5  * std::log(0.5  / 0.375) + 0.5  * std::log(0.5  / 0.625))
                          + 0.5 * (0.25 * std::log(0.25 / 0.375) + 0.75 * std::log(0.75 / 0.625));
        // EAR = min(1/2, 1/4) + min(1/2, 3/4) = 3/4
        ok &= self_test_close("2-token kld",          rec.kld,          kld,            1e-6);
        ok &= self_test_close("2-token reversed_kld", rec.reversed_kld, rkld,           1e-6);
        ok &= self_test_close("2-token js_kld",       rec.js_kld,       jsd,            1e-6);
        ok &= self_test_close("2-token ear",          rec.ear,          0.75,           1e-6);
        // K clamps to the 2-slot vocabulary, so the whole EAR_K family equals EAR here
        ok &= self_test_close("2-token ear_20",       rec.ear_20,       0.75,           1e-6);
        ok &= self_test_close("2-token ear_10",       rec.ear_10,       0.75,           1e-6);
        ok &= self_test_close("2-token ear_5",        rec.ear_5,        0.75,           1e-6);
        ok &= self_test_close("2-token ear_20_normalized", rec.ear_20_normalized, 0.75, 1e-6);
        ok &= self_test_close("2-token ear_10_normalized", rec.ear_10_normalized, 0.75, 1e-6);
        ok &= self_test_close("2-token ear_5_normalized",  rec.ear_5_normalized,  0.75, 1e-6);
        ok &= self_test_close("2-token ear_64", rec.ear_64, 0.75, 1e-6);
        ok &= self_test_close("2-token ear_64_normalized", rec.ear_64_normalized, 0.75, 1e-6);
        ok &= self_test_close("2-token nll_ref",      rec.nll_ref,      std::log(2.0),  1e-6);
        ok &= self_test_close("2-token nll_cand",     rec.nll_cand,     std::log(4.0 / 3.0), 1e-6);
        ok &= self_test_close("2-token entropy_ref",  rec.entropy_ref,  std::log(2.0),  1e-6);
        ok &= (rec.argmax_ref == 0 || rec.argmax_ref == 1);
        ok &= rec.argmax_cand == 1;
        ok &= rec.target == 1;
    }

    // 2) identity: same row on both sides -> all divergences exactly ~0
    // 3) random rows vs the naive implementation + invariants. The 2e-6
    //    tolerance is deliberately tight: both paths accumulate in double and
    //    store float32, so they agree to ~1 ulp — a regression to float
    //    accumulators (~1e-5 error even at vocab 257) must trip it.
    {
        uint64_t state = 0x243F6A8885A308D3ull;   // deterministic LCG
        auto next_logit = [&state]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            // map the top bits to [-15, 15] — spans realistic logit ranges
            return (float) (((state >> 33) % 30000) / 1000.0 - 15.0);
        };
        auto check_vs_naive = [&ok](const char * tag, const kld_record & got,
                                    const kld_record & want, double tol) {
            ok &= self_test_close(tag, got.kld,          want.kld,          tol);
            ok &= self_test_close(tag, got.reversed_kld, want.reversed_kld, tol);
            ok &= self_test_close(tag, got.js_kld,       want.js_kld,       tol);
            ok &= self_test_close(tag, got.ear,          want.ear,          tol);
            ok &= self_test_close(tag, got.ear_20,       want.ear_20,       tol);
            ok &= self_test_close(tag, got.ear_10,       want.ear_10,       tol);
            ok &= self_test_close(tag, got.ear_5,        want.ear_5,        tol);
            ok &= self_test_close(tag, got.ear_20_normalized, want.ear_20_normalized, tol);
            ok &= self_test_close(tag, got.ear_10_normalized, want.ear_10_normalized, tol);
            ok &= self_test_close(tag, got.ear_5_normalized,  want.ear_5_normalized,  tol);
            ok &= self_test_close(tag, got.ear_64, want.ear_64, tol);
            ok &= self_test_close(tag, got.ear_64_normalized, want.ear_64_normalized, tol);
            ok &= self_test_close(tag, got.nll_ref,      want.nll_ref,      tol);
            ok &= self_test_close(tag, got.nll_cand,     want.nll_cand,     tol);
            ok &= self_test_close(tag, got.entropy_ref,  want.entropy_ref,  tol);
            ok &= self_test_close(tag, got.entropy_cand, want.entropy_cand, tol);
            if (got.argmax_ref != want.argmax_ref || got.argmax_cand != want.argmax_cand ||
                got.target != want.target) {
                fprintf(stderr, "self-test FAIL: %s: argmax/target mismatch\n", tag);
                ok = false;
            }
        };
        const int n_vocab = 257;   // odd size: catches stride/edge bugs
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<float> r(n_vocab), c(n_vocab);
            for (int i = 0; i < n_vocab; ++i) { r[i] = next_logit(); c[i] = next_logit(); }
            if (trial == 0) {
                // identity case (element loop dodges a GCC 11 -Wstringop-overflow
                // false positive on the vector copy assignment)
                for (int i = 0; i < n_vocab; ++i) c[i] = r[i];
            }
            if (trial == 1) {
                c[40] = 100.0f;   // extreme spike: stresses log-space stability
            }
            // every third trial targets the last vocab slot, so an off-by-one
            // at the boundary is a caught mismatch, not an OOB read
            const int32_t target = (trial % 3 == 2) ? (int32_t) (n_vocab - 1)
                                                    : (int32_t) (trial % n_vocab);
            const kld_record got  = compute_kld_record(r.data(), c.data(), n_vocab, target);
            const kld_record want = naive_kld_record(r, c, target);
            check_vs_naive("rand", got, want, 2e-6);
            // invariants: non-negative divergences, JSD bounded by ln 2, EAR in
            // [0, 1] and above the Pinsker floor 1 - sqrt(kld/2) (EAR = 1 - TV,
            // TV <= sqrt(KL/2)).
            if (got.kld < -1e-6f || got.reversed_kld < -1e-6f ||
                got.js_kld < -1e-6f || got.js_kld > (float) std::log(2.0) + 1e-6f ||
                got.ear < -1e-6f || got.ear > 1.0f + 1e-6f ||
                // ear_K (full-vocab mass on the reference top-K) is monotone in K
                // and never above the full ear; the normalized variants are in [0, 1]
                got.ear_5 < -1e-6f || got.ear_5 > got.ear_10 + 1e-6f ||
                got.ear_10 > got.ear_20 + 1e-6f || got.ear_20 > got.ear_64 + 1e-6f ||
                got.ear_64 > got.ear + 1e-6f ||
                got.ear_64_normalized < -1e-6f || got.ear_64_normalized > 1.0f + 1e-6f ||
                got.ear_20_normalized < -1e-6f || got.ear_20_normalized > 1.0f + 1e-6f ||
                got.ear_10_normalized < -1e-6f || got.ear_10_normalized > 1.0f + 1e-6f ||
                got.ear_5_normalized  < -1e-6f || got.ear_5_normalized  > 1.0f + 1e-6f ||
                (double) got.ear < 1.0 - std::sqrt(std::max(0.0, (double) got.kld) / 2.0) - 1e-6) {
                fprintf(stderr, "self-test FAIL: invariant violated on trial %d "
                                "(kld=%g rkld=%g jsd=%g ear=%g)\n",
                        trial, got.kld, got.reversed_kld, got.js_kld, got.ear);
                ok = false;
            }
            if (trial == 0 &&
                (std::abs(got.kld) > 1e-9f || std::abs(got.reversed_kld) > 1e-9f ||
                 std::abs(got.js_kld) > 1e-9f || std::abs(got.ear - 1.0f) > 1e-6f ||
                 // identity: the normalized variants are exactly 1; the mass
                 // variants equal the reference's own top-K mass (checked vs naive)
                 std::abs(got.ear_64_normalized - 1.0f) > 1e-6f ||
                 std::abs(got.ear_20_normalized - 1.0f) > 1e-6f ||
                 std::abs(got.ear_10_normalized - 1.0f) > 1e-6f ||
                 std::abs(got.ear_5_normalized - 1.0f) > 1e-6f)) {
                fprintf(stderr, "self-test FAIL: identity divergences not ~0 "
                                "(kld=%g rkld=%g jsd=%g ear=%g)\n",
                        got.kld, got.reversed_kld, got.js_kld, got.ear);
                ok = false;
            }
        }

        // 3b) realistic-vocab trial: at vocab 50k a float-accumulator
        //     regression lands ~1e-4 off the double-accumulated naive value,
        //     far beyond this tolerance (vocab 257 alone cannot separate
        //     pairwise-float from double accumulation). NOTE the test point:
        //     logits uniform on [-15, 15] over independent rows give
        //     KLD ~ 13.8 nats and EAR ~ 0.07, i.e. two unrelated
        //     distributions. That is a fine place to catch a gross error and
        //     a useless one for a precision guard -- see 3e.
        {
            const int big_vocab = 50000;
            std::vector<float> r(big_vocab), c(big_vocab);
            for (int i = 0; i < big_vocab; ++i) { r[i] = next_logit(); c[i] = next_logit(); }
            const int32_t target = (int32_t) (big_vocab - 1);
            const kld_record got  = compute_kld_record(r.data(), c.data(), big_vocab, target);
            const kld_record want = naive_kld_record(r, c, target);
            check_vs_naive("big-vocab", got, want, 4e-6);
        }

        // 3e) PRODUCTION-REGIME trial: the full Qwen3-VL vocabulary against a
        //     NEAR-IDENTICAL candidate, which is the only regime real runs
        //     live in (KLD ~ 1e-3 nats, EAR ~ 0.99). 3b's test point sits at
        //     KLD ~ 13.8 / EAR ~ 0.07, four orders of magnitude away, so its
        //     absolute 4e-6 tolerance is not load-bearing here: a float
        //     accumulator lands ~2.8e-6 off a 6e-4 KLD, which PASSES a 4e-6
        //     absolute check, and moves `ear` by ~2.9e-7, which is under
        //     5 ulp of a float32 near 0.99 and so cannot be seen in the
        //     stored record at all.
        //
        //     Two changes make the guard bite: compare in DOUBLE (the
        //     kld_values the kernel accumulates, before kld_record's float32
        //     store), and use a RELATIVE tolerance so it scales with the
        //     metric instead of with the test point. Both implementations
        //     accumulate in double by different routes (log-space vs
        //     probability-space), which agree to ~1e-11 relative; 1e-9
        //     leaves two orders of headroom and still fails a float
        //     accumulator by ~10^6 on kld and ~10^2 on ear.
        {
            const int prod_vocab = 151936;                  // Qwen3-VL
            std::vector<float> r(prod_vocab), c(prod_vocab);
            for (int i = 0; i < prod_vocab; ++i) {
                r[i] = next_logit();
                // candidate = reference + a Q8-class perturbation. next_logit
                // is uniform on [-15, 15]; /15*0.0866 makes it uniform on
                // +/-0.0866, i.e. sd 0.05.
                c[i] = r[i] + (float) ((next_logit() / 15.0) * 0.0866);
            }
            const int32_t target = (int32_t) (prod_vocab - 1);
            const kld_values got  = compute_kld_values(r.data(), c.data(), prod_vocab, target);
            const kld_values want = naive_kld_values(r, c, target);
            // Pin the REGIME itself: if the generator ever drifts, the
            // tolerance below stops meaning what this comment says.
            if (!(got.kld > 1e-5 && got.kld < 1e-2 && got.ear > 0.95)) {
                fprintf(stderr, "self-test FAIL: production-regime trial left the "
                                "production regime (kld=%g ear=%g); expected "
                                "kld in (1e-5, 1e-2) and ear > 0.95\n",
                        got.kld, got.ear);
                ok = false;
            }
            const double rtol = 1e-9;
            ok &= self_test_close_rel("prod kld",          got.kld,          want.kld,          rtol);
            ok &= self_test_close_rel("prod reversed_kld", got.reversed_kld, want.reversed_kld, rtol);
            ok &= self_test_close_rel("prod js_kld",       got.js_kld,       want.js_kld,       rtol);
            ok &= self_test_close_rel("prod ear",          got.ear,          want.ear,          rtol);
            ok &= self_test_close_rel("prod ear_20",       got.ear_20,       want.ear_20,       rtol);
            ok &= self_test_close_rel("prod ear_10",       got.ear_10,       want.ear_10,       rtol);
            ok &= self_test_close_rel("prod ear_5",        got.ear_5,        want.ear_5,        rtol);
            ok &= self_test_close_rel("prod ear_20_normalized", got.ear_20_normalized, want.ear_20_normalized, rtol);
            ok &= self_test_close_rel("prod ear_10_normalized", got.ear_10_normalized, want.ear_10_normalized, rtol);
            ok &= self_test_close_rel("prod ear_5_normalized",  got.ear_5_normalized,  want.ear_5_normalized,  rtol);
            ok &= self_test_close_rel("prod ear_64", got.ear_64, want.ear_64, rtol);
            ok &= self_test_close_rel("prod ear_64_normalized", got.ear_64_normalized, want.ear_64_normalized, rtol);
            ok &= self_test_close_rel("prod nll_ref",      got.nll_ref,      want.nll_ref,      rtol);
            ok &= self_test_close_rel("prod nll_cand",     got.nll_cand,     want.nll_cand,     rtol);
            ok &= self_test_close_rel("prod entropy_ref",  got.entropy_ref,  want.entropy_ref,  rtol);
            ok &= self_test_close_rel("prod entropy_cand", got.entropy_cand, want.entropy_cand, rtol);
            if (got.target != want.target || got.argmax_ref != want.argmax_ref ||
                got.argmax_cand != want.argmax_cand) {
                fprintf(stderr, "self-test FAIL: production-regime argmax/target mismatch\n");
                ok = false;
            }
        }

        // 3f) EAR_K family: the candidate agrees with the reference EXACTLY on
        //     the reference's five most likely slots and is shifted everywhere
        //     else, so ear_5_normalized must be 1 while ear_10/20_normalized
        //     (which see the shifted slots) and the full-vocab ear must not.
        //     The mass variants must not exceed the reference's own top-K
        //     mass. Ties in the tail (equal logits) exercise the
        //     lower-index-first rule against the naive stable sort.
        {
            std::vector<float> r(n_vocab), c(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                r[i] = (i < 30) ? (float) (20 - i) : -5.0f;   // 30 distinct leaders, then a flat tail
                c[i] = (i < 5)  ? r[i] : r[i] + 0.5f;
            }
            const kld_record got  = compute_kld_record(r.data(), c.data(), n_vocab, 0);
            const kld_record want = naive_kld_record(r, c, 0);
            check_vs_naive("topk", got, want, 2e-6);
            double ref_top5_mass = 0.0;   // sum of the reference's five largest probabilities
            {
                double mx = r[0], z = 0.0;
                for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, (double) r[i]);
                for (int i = 0; i < n_vocab; ++i) z += std::exp((double) r[i] - mx);
                for (int i = 0; i < 5; ++i) ref_top5_mass += std::exp((double) r[i] - mx) / z;
            }
            if (std::abs(got.ear_5_normalized - 1.0f) > 1e-6f ||
                got.ear_10_normalized > 1.0f - 1e-4f || got.ear_20_normalized > 1.0f - 1e-4f ||
                got.ear > 1.0f - 1e-4f || (double) got.ear_5 > ref_top5_mass + 1e-6) {
                fprintf(stderr, "self-test FAIL: EAR_K top-5 agreement case "
                                "(ear_5=%g/%g ear_10n=%g ear_20n=%g ear=%g ref_top5_mass=%g)\n",
                        got.ear_5, got.ear_5_normalized, got.ear_10_normalized,
                        got.ear_20_normalized, got.ear, ref_top5_mass);
                ok = false;
            }
        }

        // 3c) masked-vocab trial: -inf logits at the SAME slots on both sides
        //     (p == 0 on both) must contribute exactly nothing — without the
        //     p > 0 guards this is 0 * (inf - inf) = NaN. The naive
        //     implementation skips zero-probability terms by construction.
        {
            std::vector<float> r(n_vocab), c(n_vocab);
            for (int i = 0; i < n_vocab; ++i) { r[i] = next_logit(); c[i] = next_logit(); }
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int i = 5; i < n_vocab; i += 7) { r[i] = neg_inf; c[i] = neg_inf; }
            const int32_t target = 0;   // not a masked slot
            const kld_record got  = compute_kld_record(r.data(), c.data(), n_vocab, target);
            const kld_record want = naive_kld_record(r, c, target);
            if (!std::isfinite(got.kld) || !std::isfinite(got.reversed_kld) ||
                !std::isfinite(got.js_kld) || !std::isfinite(got.entropy_ref) ||
                !std::isfinite(got.entropy_cand) || !std::isfinite(got.ear) ||
                !std::isfinite(got.ear_20) || !std::isfinite(got.ear_10) || !std::isfinite(got.ear_5) ||
                !std::isfinite(got.ear_20_normalized) || !std::isfinite(got.ear_10_normalized) ||
                !std::isfinite(got.ear_5_normalized) ||
                !std::isfinite(got.ear_64) || !std::isfinite(got.ear_64_normalized)) {
                fprintf(stderr, "self-test FAIL: masked-vocab metrics not finite "
                                "(kld=%g rkld=%g jsd=%g ent_r=%g ent_c=%g ear=%g)\n",
                        got.kld, got.reversed_kld, got.js_kld,
                        got.entropy_ref, got.entropy_cand, got.ear);
                ok = false;
            } else {
                check_vs_naive("masked", got, want, 2e-6);
            }
        }

        // 3d) ONE-SIDED masked slot: -inf on the candidate side only. The
        //     EAR term exp(min(lp_r, -inf)) must contribute exactly 0 and the
        //     record's ear stay finite/naive-consistent; forward KL is then
        //     legitimately +inf (ref has mass where cand has none) while
        //     reverse KL, JSD and the entropies stay finite. check_vs_naive
        //     is unusable here (|inf - inf| = NaN on the kld field), so the
        //     finite fields are checked individually.
        {
            std::vector<float> r(n_vocab), c(n_vocab);
            for (int i = 0; i < n_vocab; ++i) { r[i] = next_logit(); c[i] = next_logit(); }
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int i = 3; i < n_vocab; i += 11) { c[i] = neg_inf; }
            const int32_t target = 0;   // not a masked slot
            const kld_record got  = compute_kld_record(r.data(), c.data(), n_vocab, target);
            const kld_record want = naive_kld_record(r, c, target);
            if (!(std::isinf(got.kld) && got.kld > 0 &&
                  std::isinf(want.kld) && want.kld > 0)) {
                fprintf(stderr, "self-test FAIL: one-sided mask kld not +inf "
                                "(got %g, naive %g)\n", got.kld, want.kld);
                ok = false;
            }
            if (kld_record_all_finite(got)) {
                fprintf(stderr, "self-test FAIL: non-finite policy gate accepted a +inf record\n");
                ok = false;
            }
            if (!std::isfinite(got.ear) || !std::isfinite(got.reversed_kld) ||
                !std::isfinite(got.js_kld) || !std::isfinite(got.entropy_ref) ||
                !std::isfinite(got.entropy_cand)) {
                fprintf(stderr, "self-test FAIL: one-sided mask finite fields "
                                "(ear=%g rkld=%g jsd=%g ent_r=%g ent_c=%g)\n",
                        got.ear, got.reversed_kld, got.js_kld,
                        got.entropy_ref, got.entropy_cand);
                ok = false;
            } else {
                ok &= self_test_close("one-sided ear",          got.ear,          want.ear,          2e-6);
                ok &= self_test_close("one-sided ear_64", got.ear_64, want.ear_64, 2e-6);
                ok &= self_test_close("one-sided ear_64_normalized", got.ear_64_normalized, want.ear_64_normalized, 2e-6);
                ok &= self_test_close("one-sided reversed_kld", got.reversed_kld, want.reversed_kld, 2e-6);
                ok &= self_test_close("one-sided js_kld",       got.js_kld,       want.js_kld,       2e-6);
                ok &= self_test_close("one-sided nll_ref",      got.nll_ref,      want.nll_ref,      2e-6);
                ok &= self_test_close("one-sided nll_cand",     got.nll_cand,     want.nll_cand,     2e-6);
                ok &= self_test_close("one-sided entropy_cand", got.entropy_cand, want.entropy_cand, 2e-6);
                ok &= self_test_close("one-sided entropy_ref",  got.entropy_ref,  want.entropy_ref,  2e-6);
                if (got.target != want.target || got.argmax_ref != want.argmax_ref ||
                    got.argmax_cand != want.argmax_cand) {
                    fprintf(stderr, "self-test FAIL: one-sided mask: argmax/target mismatch\n");
                    ok = false;
                }
            }
        }
    }

    // EAR_64 boundaries and selected-support normalization, without changing earlier RNG trials.
    for (const int n_vocab : {1, 2, 63, 64, 65, 128}) {
        std::vector<float> r(n_vocab), c(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            r[i] = (float) ((i * 17) % 23) / 4.0f;
            c[i] = r[i] + (float) ((i * 7) % 11) / 8.0f;
        }
        const kld_values got = compute_kld_values(r.data(), c.data(), n_vocab, n_vocab - 1);
        const kld_values want = naive_kld_values(r, c, n_vocab - 1);
        ok &= self_test_close("boundary ear_64", got.ear_64, want.ear_64, 1e-12);
        ok &= self_test_close("boundary ear_64_normalized", got.ear_64_normalized, want.ear_64_normalized, 1e-12);
        if (n_vocab <= 64) {
            ok &= self_test_close("clamped ear_64", got.ear_64, got.ear, 1e-12);
            ok &= self_test_close("clamped ear_64_normalized", got.ear_64_normalized, got.ear, 1e-12);
        }
    }
    {
        std::vector<float> r(65, 0.0f), c(65, 0.0f);
        c[64] = (float) std::log(64.0);
        const kld_values got = compute_kld_values(r.data(), c.data(), 65, 0);
        const double selected_mass = 64.0 / (64.0 + std::exp((double) c[64]));
        ok &= self_test_close("cutoff tie ear_64", got.ear_64, selected_mass, 1e-12);
        ok &= self_test_close("outside mass ear_64_normalized", got.ear_64_normalized, 1.0, 1e-12);
        int32_t ids[EAR_TOPK_MAX];
        float logits[EAR_TOPK_MAX];
        const int count = select_ref_topk(r.data(), 65, ids, logits);
        ok &= self_test_close("cutoff tie count", count, 64, 0);
        for (int i = 0; i < count; ++i) {
            ok &= self_test_close("cutoff tie token id", ids[i], i, 0);
        }
        for (const float selected : {-std::numeric_limits<float>::infinity(), -1000.0f}) {
            std::fill(c.begin(), c.end(), selected);
            c[64] = 0.0f;
            const kld_values masked = compute_kld_values(r.data(), c.data(), 65, 64);
            ok &= self_test_close("zero full-row selected mass", masked.ear_64, 0.0, 0);
            ok &= self_test_close("local selected softmax", masked.ear_64_normalized,
                                  std::isfinite(selected) ? 1.0 : 0.0, 1e-12);
        }
        for (float kld_record::* field : {&kld_record::ear_64, &kld_record::ear_64_normalized}) {
            kld_record rec = to_kld_record(got);
            rec.*field = std::numeric_limits<float>::quiet_NaN();
            if (kld_record_all_finite(rec) || validate_kld_record_finite(rec, 0, nullptr)) {
                fprintf(stderr, "self-test FAIL: finite gate ignored an EAR_64 NaN\n");
                ok = false;
            }
        }
    }

    // 4) the parallel driver must produce the same bits as the serial loop
    {
        const int n_vocab = 64, n_rows = 7;
        std::vector<std::vector<float>> rs(n_rows, std::vector<float>(n_vocab));
        std::vector<std::vector<float>> cs(n_rows, std::vector<float>(n_vocab));
        uint64_t state = 0x13198A2E03707344ull;
        for (int p = 0; p < n_rows; ++p) {
            for (int i = 0; i < n_vocab; ++i) {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                rs[p][i] = (float) (((state >> 33) % 20000) / 1000.0 - 10.0);
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                cs[p][i] = (float) (((state >> 33) % 20000) / 1000.0 - 10.0);
            }
        }
        std::vector<const float *> rp, cp;
        std::vector<int32_t> tgts;
        for (int p = 0; p < n_rows; ++p) {
            rp.push_back(rs[p].data());
            cp.push_back(cs[p].data());
            tgts.push_back((int32_t) (p % n_vocab));
        }
        std::vector<kld_record> serial(n_rows), parallel(n_rows);
        compute_records_parallel(rp, cp, tgts, n_vocab, 1, serial.data());
        compute_records_parallel(rp, cp, tgts, n_vocab, 4, parallel.data());
        if (std::memcmp(serial.data(), parallel.data(), n_rows * sizeof(kld_record)) != 0) {
            fprintf(stderr, "self-test FAIL: parallel records differ from serial\n");
            ok = false;
        }
    }

    fprintf(stderr, "self-test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


// Shared teacher forcing; each caller supplies its model state and output writer.
template <typename ARGS, typename SIDE, typename WRITE_FN>
static bool kld_teacher_force_and_write(
        const ARGS & args,
        SIDE & ref,
        SIDE & cand,
        int32_t n_vocab,
        int32_t n_batch,
        int n_eval,
        const std::vector<int32_t> & tokens_full,
        stderr_prefix * lp,
        WRITE_FN && write_vlmk_file) {
    // Snapshot the POST-PREFILL position now: the teacher-forcing loop below
    // advances ref.n_past by one per fed token, so reading it at write time
    // would record prefill + n_feed and no longer be comparable with
    // vlm-score's VLMS header (which records the prefill count).
    const llama_pos n_past_after_prefill = ref.n_past;

    const int metric_threads = resolve_metric_threads(args.metric_threads);
    std::vector<kld_record> records;
    records.reserve(n_eval);

    // Slot 0: both sides' last-prefill logits (they predict answer token 0).
    // The pointers stay valid until the NEXT decode on the SAME context, and
    // cand's prefill does not touch ref's output buffer.
    {
        const float * r0 = llama_get_logits_ith(ref.lctx.get(),  -1);
        const float * c0 = llama_get_logits_ith(cand.lctx.get(), -1);
        if (!r0 || !c0) {
            prefixed_fprintf(lp, "llama_get_logits_ith(-1) returned NULL after prefill\n");
            return false;
        }
        const kld_record rec = compute_kld_record(
                r0, c0, n_vocab, tokens_full[args.n_prefill]);
        if (!validate_kld_record_finite(rec, 0, lp)) {
            return false;
        }
        records.push_back(rec);
    }

    // Teacher-force the shared answer tokens through BOTH sides in chunks (one
    // llama_decode per side per chunk), then compute this chunk's records from
    // the two output buffers before the next decode invalidates them. Feeding
    // answer token i (tokens_full[n_prefill + i]) produces slot i + 1; slot
    // i's target is tokens_full[n_prefill + i]. Same chunked-decode numerics
    // caveat as vlm-score applies (--tf-chunk 1 = per-token bit-compat).
    const int n_feed    = n_eval - 1;
    const int tf_req    = (args.tf_chunk > 0) ? args.tf_chunk : args.n_ubatch;
    const int chunk_max = std::max(1, std::min(tf_req, (int) n_batch));

    llama_batch ref_batch  = llama_batch_init(chunk_max, 0, 1);
    llama_batch cand_batch = llama_batch_init(chunk_max, 0, 1);
    bool ok = true;
    std::vector<const float *> ref_rows, cand_rows;
    std::vector<int32_t> targets;
    // Pre-size everything the loop touches: with `records` already reserved to
    // n_eval, no allocation can throw between llama_batch_init and
    // llama_batch_free below.
    ref_rows.reserve(chunk_max);
    cand_rows.reserve(chunk_max);
    targets.reserve(chunk_max);
    for (int base = 0; base < n_feed && ok; base += chunk_max) {
        const int chunk = std::min(chunk_max, n_feed - base);
        common_batch_clear(ref_batch);
        common_batch_clear(cand_batch);
        for (int j = 0; j < chunk; ++j) {
            const llama_token tok = (llama_token) tokens_full[args.n_prefill + base + j];
            common_batch_add(ref_batch,  tok, ref.n_past  + j, {0}, /*logits=*/true);
            common_batch_add(cand_batch, tok, cand.n_past + j, {0}, /*logits=*/true);
        }
        if (llama_decode(ref.lctx.get(), ref_batch) != 0 ||
            llama_decode(cand.lctx.get(), cand_batch) != 0) {
            prefixed_fprintf(lp, "llama_decode failed at slots %d..%d (chunk=%d)\n",
                             base + 1, base + chunk, chunk);
            ok = false;
            break;
        }
        ref_rows.clear(); cand_rows.clear(); targets.clear();
        for (int j = 0; j < chunk; ++j) {
            const float * rr = llama_get_logits_ith(ref.lctx.get(),  j);
            const float * cr = llama_get_logits_ith(cand.lctx.get(), j);
            if (!rr || !cr) {
                prefixed_fprintf(lp, "llama_get_logits_ith(%d) returned NULL\n", j);
                ok = false;
                break;
            }
            ref_rows.push_back(rr);
            cand_rows.push_back(cr);
            targets.push_back(tokens_full[args.n_prefill + base + j + 1]);
        }
        if (!ok) {
            break;
        }
        const size_t prev = records.size();
        records.resize(prev + chunk);
        compute_records_parallel(ref_rows, cand_rows, targets, n_vocab,
                                 metric_threads, records.data() + prev);
        for (int j = 0; j < chunk; ++j) {
            if (!validate_kld_record_finite(records[prev + j], prev + j, lp)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
        ref.n_past  += chunk;
        cand.n_past += chunk;
        const int done = base + chunk;
        // Throttle progress to ~every 100 slots (so --tf-chunk 1 does not spam).
        if (done == n_feed || done / 100 != (done - chunk) / 100) {
            prefixed_fprintf(lp, "scored slots %d / %d\n", done, n_feed);
        }
    }
    llama_batch_free(ref_batch);
    llama_batch_free(cand_batch);
    if (!ok) {
        return false;
    }

    if (!write_vlmk_file(args.output_metrics_path, (uint32_t) n_vocab,
                         (uint32_t) args.n_prefill,
                         (uint32_t) n_past_after_prefill,
                         records, lp)) {
        return false;
    }
    prefixed_fprintf(lp, "wrote %d positions x %zu-byte records to %s\n",
                     n_eval, sizeof(kld_record), args.output_metrics_path.c_str());
    return true;
}
