// llama-llm-kld — on-the-fly paired fidelity metrics for text-only LLM evaluation.
//
// Loads TWO GGUF models — a reference (e.g. F16) and a candidate
// (e.g. Q4_K_M) — runs the SAME teacher-forced prompt+answer token stream through both, and
// computes per-answer-token divergence metrics from the two full-vocab logit
// rows while they are still in memory. Only the metrics are written to disk:
// 76 bytes/position instead of vocab * 4 bytes/position of dense fp32 logits
// (~8,000x smaller at vocab ~152k). Top-K metrics select reference tokens.
//
// Deliberately a STANDALONE module, independent of vlm-score.cpp (which dumps
// logits for offline comparison): the two tools answer different questions and
// reviewing/maintaining them separately keeps each contract small. The small
// helpers shared with vlm-score.cpp (stderr prefixing, file slurping, token
// reading) are duplicated here on purpose.
//
// Does NOT sample. Does NOT generate. Pure dual forward + metric extraction.
//
// CLI:
//   --ref-model <gguf>                 Reference LLM weights
//   --cand-model <gguf>                Candidate LLM weights
//   --tokens-in <path>                 Binary file: int32[L] of full input_ids
//   --n-prefill <int>                  First answer-token index (must match HF stored value)
//   --output-metrics <path>            Output binary file (VLMK records)
//   --manifest <jsonl>                 Batch mode: JSONL entries with tokens_in,
//                                      n_prefill, output_metrics. Loads both models once.
//                                      Takes precedence: the per-row flags
//                                      above are ignored when given.
//   --num-eval-tokens <int>            Stop teacher-forcing after this many answer
//                                      positions (-1 = all, default; clamped to
//                                      L - n_prefill)
//   --perplexity-window                Decode one full even-length context window per side, like classic np1 perplexity.
//                                      Requires L = -c = -b = -ub and n_prefill = L/2 + 1; scores all L/2 - 1 targets.
//                                      Input tokens must already include the caller's BOS policy; no tokens are changed.
//   -ngl <int>                         GPU layers to offload, per model (default 99)
//   -c <int>                           Context size, per model (default 32768)
//   -b <int>                           Logical batch size for prefill (default 2048)
//   -ub <int>                          Physical micro-batch for prefill; also the
//                                      default teacher-forcing chunk size (default 2048)
//   --tf-chunk <int>                   Teacher-forcing chunk size, decoupled from -ub
//                                      (default -1 = follow -ub). Same numerics caveat
//                                      as vlm-score: batched chunks shift logits by FP
//                                      non-associativity, so runs that must be
//                                      comparable (e.g. ref-vs-A and ref-vs-B sharing
//                                      the same reference) MUST use the same --tf-chunk.
//   -t <int>                           CPU threads for llama decode
//                                      (default -1 = ggml's compiled default;
//                                      irrelevant for fully-offloaded GPU runs)
//   --metric-threads <int>             Threads for the per-position metric kernel
//                                      (default -1 = hardware concurrency)
//   --flash-attn                       Require flash attention
//   --no-flash-attn                    Disable flash attention (default: auto)
//   --swa-full                         Full-size KV cache for sliding-window layers
//                                      (the llama_context default). Off by default:
//                                      SWA layers allocate only their window, like
//                                      the common llama.cpp CLI. Same attention math
//                                      either way; results differ by FP reordering
//                                      only, so runs that must be comparable use the
//                                      same setting (recorded in collect_meta.json).
//   --allow-vocab-attr-mismatch        Accept per-token ATTRIBUTE differences
//                                      (token type / EOG flags) between the two
//                                      vocabs; token texts must still match id
//                                      by id. For same-base-model conversions
//                                      whose metadata disagrees (e.g. which id
//                                      is <eos>). Every mismatch is logged.
//   --self-test                        Run the metric-kernel self test (no models
//                                      needed) and exit 0/1
//   --vlmk-version                     Print the VLMK format version this binary
//                                      writes (stdout, integer) and exit 0; the
//                                      collectors preflight it against their
//                                      kld_metrics_io.VLMK_VERSION so a stale
//                                      build is refused before any GPU work
//
// NOTE: the answer region tokens[n_prefill:] is teacher-forced from the stored
// token ids through BOTH models, so position i of the metrics aligns by answer
// index. Each side keeps its own KV cache / n_past, which is always equal after
// text prefill. The two models MUST share a vocabulary; size, vocabulary type,
// and every token ID's text and attributes are checked at startup
// (--allow-vocab-attr-mismatch downgrades only the attribute check to a
// logged warning; texts remain strict).
//
// Output binary layout (LE on x86_64), magic "VLMK":
//   uint32_t magic = 0x564C4D4B                ("VLMK")
//   uint32_t version = 5                       (v1 = 40-byte records without ear)
//   uint32_t vocab_size                        (== llama_vocab_n_tokens, both models)
//   uint32_t n_positions                       (# answer tokens scored)
//   uint32_t n_prefill                         (HF ground-truth sequential
//                                              length, echoed from the manifest)
//   uint32_t n_past_actual                     (llama.cpp's OWN position count
//                                              after prefill -- the number that
//                                              actually moves when the vision
//                                              budget changes; 0 = not recorded,
//                                              i.e. written before this field)
//                                              With --perplexity-window, this is L because the whole window was decoded.
// followed by n_positions packed records of 76 bytes:
//   float32 kld                                KL(p_ref || p_cand), nats
//   float32 reversed_kld                       KL(p_cand || p_ref), nats
//   float32 js_kld                             Jensen-Shannon divergence, nats
//   float32 nll_ref                            -log p_ref(target)
//   float32 nll_cand                           -log p_cand(target)
//   float32 entropy_ref                        -sum p_ref * log p_ref
//   float32 entropy_cand                       -sum p_cand * log p_cand
//   float32 ear                                sum min(p_ref, p_cand) = 1 - TV distance
//   float32 ear_20                             sum of min(p_ref, p_cand) over the reference's
//                                              top-20 slots, full-vocab probabilities (v4)
//   float32 ear_10 / ear_5                     same, top-10 / top-5 (v4)
//   float32 ear_20_normalized                  same slots, both rows renormalized on them
//                                              first (v4)
//   float32 ear_10_normalized / ear_5_normalized  same, top-10 / top-5 (v4)
//   int32   target                             teacher-forced target token id
//   int32   argmax_ref                         reference argmax token id
//   int32   argmax_cand                        candidate argmax token id
//   float32 ear_64, ear_64_normalized (v5, after the v4 prefix)
// All metrics use float64 accumulation and float32 storage. Derivable downstream: same_top = (argmax_ref == argmax_cand),
// p(target) = exp(-nll), delta-p at target, perplexities = exp(mean nll).
// `ear` is the per-position Expected Acceptance Rate (arXiv:2605.02404): the
// probability-mass overlap sum_i min(p_ref(i), p_cand(i)) = 1 - d_TV, i.e. the
// max probability that optimally-coupled samples from the two distributions
// agree. Computed over the full vocabulary.
//
// Consumed by tools/skymizer/kld_metrics_io.py (load + lossless .npz
// conversion) and orchestrated over a dataset by tools/skymizer/collect_llm_kld.py.
//
// Build registered in tools/skymizer/CMakeLists.txt.

#include "skymizer-identity.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"

#include "nlohmann/json.hpp"

#include "skymizer-common.h"
#include "skymizer-vlmk-kernel.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#else
#    include <unistd.h>
#endif

static constexpr uint32_t VLMK_MAGIC   = 0x564C4D4B; // "VLMK"
static constexpr uint32_t VLMK_VERSION = 5;          // v5: 76-byte records (+ EAR_64); v4 = 68, v3 = 56, v2 = 44, v1 = 40


// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct vlm_kld_args {
    std::string ref_model_path;
    std::string cand_model_path;
    std::string tokens_in_path;
    std::string output_metrics_path;
    std::string manifest_path;
    int n_prefill = -1;
    int num_eval_tokens = -1;    // -1 = score all answer positions
    int n_gpu_layers = 99;
    int n_ctx = 32768;
    int n_batch = 2048;
    int n_ubatch = 2048;
    int tf_chunk = -1;           // teacher-forcing chunk size; -1 = follow n_ubatch
    int n_threads = -1;
    int metric_threads = -1;     // -1 = hardware concurrency
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool swa_full = false;             // see load_side()
    bool allow_vocab_attr_mismatch = false;  // see validate_vocab_id_mapping()
    bool perplexity_window = false;
    bool self_test = false;
    bool print_vlmk_version = false;
};


static bool parse_args(int argc, char ** argv, vlm_kld_args & a) {
    bool saw_flash_attn = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        auto need_int = [&](const char * what, int & out) {
            const char * v = need(what);
            return v != nullptr && parse_int_arg(what, v, out);
        };
        if      (k == "--ref-model")        { const char * v = need("--ref-model");        if (!v) return false; a.ref_model_path = v; }
        else if (k == "--cand-model")       { const char * v = need("--cand-model");       if (!v) return false; a.cand_model_path = v; }
        else if (k == "--tokens-in")        { const char * v = need("--tokens-in");        if (!v) return false; a.tokens_in_path = v; }
        else if (k == "--n-prefill")        { if (!need_int("--n-prefill",        a.n_prefill))        return false; }
        else if (k == "--output-metrics")   { const char * v = need("--output-metrics");   if (!v) return false; a.output_metrics_path = v; }
        else if (k == "--manifest")         { const char * v = need("--manifest");         if (!v) return false; a.manifest_path = v; }
        else if (k == "--num-eval-tokens")  { if (!need_int("--num-eval-tokens",  a.num_eval_tokens))  return false; }
        else if (k == "-ngl")               { if (!need_int("-ngl",               a.n_gpu_layers))     return false; }
        else if (k == "-c")                 { if (!need_int("-c",                 a.n_ctx))            return false; }
        else if (k == "-b")                 { if (!need_int("-b",                 a.n_batch))          return false; }
        else if (k == "-ub")                { if (!need_int("-ub",                a.n_ubatch))         return false; }
        else if (k == "--tf-chunk")         { if (!need_int("--tf-chunk",         a.tf_chunk))         return false; }
        else if (k == "-t")                 { if (!need_int("-t",                 a.n_threads))        return false; }
        else if (k == "--metric-threads")   { if (!need_int("--metric-threads",   a.metric_threads))   return false; }
        else if (k == "--flash-attn" || k == "--no-flash-attn") {
            if (saw_flash_attn) {
                fprintf(stderr, "flash-attention mode specified more than once\n");
                return false;
            }
            saw_flash_attn = true;
            a.flash_attn_type = k == "--flash-attn"
                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        else if (k == "--swa-full")         { a.swa_full = true; }
        else if (k == "--perplexity-window") { a.perplexity_window = true; }
        else if (k == "--allow-vocab-attr-mismatch") { a.allow_vocab_attr_mismatch = true; }
        else if (k == "--self-test")        { a.self_test = true; }
        else if (k == "--vlmk-version")     { a.print_vlmk_version = true; }
        else if (k == "--help" || k == "-h") {
            fprintf(stderr, "see header of llm-kld.cpp for CLI documentation\n");
            return false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", k.c_str());
            return false;
        }
    }
    if (a.self_test || a.print_vlmk_version) {
        return true;   // no other arguments required
    }
    if (a.ref_model_path.empty() || a.cand_model_path.empty()) {
        fprintf(stderr, "missing required arguments (run with --help)\n");
        return false;
    }
    if (a.manifest_path.empty()) {
        if (a.tokens_in_path.empty() || a.output_metrics_path.empty() ||
            a.n_prefill < 0) {
            fprintf(stderr, "missing required arguments (run with --help)\n");
            return false;
        }
    }
    if (a.n_batch < 1 || a.n_ubatch < 1) {
        fprintf(stderr, "-b and -ub must be >= 1, got -b %d -ub %d\n", a.n_batch, a.n_ubatch);
        return false;
    }
    if (a.n_ubatch > a.n_batch) {
        fprintf(stderr, "-ub (%d) must be <= -b (%d)\n", a.n_ubatch, a.n_batch);
        return false;
    }
    if (a.tf_chunk != -1 && a.tf_chunk < 1) {
        fprintf(stderr, "--tf-chunk must be -1 (follow -ub) or >= 1, got %d\n", a.tf_chunk);
        return false;
    }
    if (a.num_eval_tokens != -1 && a.num_eval_tokens < 1) {
        fprintf(stderr, "num-eval-tokens must be -1 (all) or >= 1, got %d\n", a.num_eval_tokens);
        return false;
    }
    if (a.metric_threads != -1 && a.metric_threads < 1) {
        fprintf(stderr, "--metric-threads must be -1 (auto) or >= 1, got %d\n", a.metric_threads);
        return false;
    }
    if (a.perplexity_window &&
            (a.n_ctx < 4 || a.n_ctx % 2 != 0 || a.n_batch != a.n_ctx || a.n_ubatch != a.n_ctx ||
             (a.tf_chunk != -1 && a.tf_chunk != a.n_ctx) ||
             (a.num_eval_tokens != -1 && a.num_eval_tokens != a.n_ctx / 2 - 1))) {
        fprintf(stderr, "--perplexity-window requires even -c >= 4, -b = -ub = -c, --tf-chunk -1 or -c, and all -c/2 - 1 targets\n");
        return false;
    }
    return true;
}


static bool validate_perplexity_window_input(const vlm_kld_args & args, size_t n_tokens, stderr_prefix * lp) {
    if (n_tokens != (size_t) args.n_ctx || args.n_prefill != args.n_ctx / 2 + 1) {
        prefixed_fprintf(lp, "--perplexity-window requires exactly %d tokens and n_prefill=%d, got %zu tokens and n_prefill=%d\n",
                         args.n_ctx, args.n_ctx / 2 + 1, n_tokens, args.n_prefill);
        return false;
    }
    return true;
}

static bool validate_perplexity_window_vocab(const llama_vocab * ref, const llama_vocab * cand) {
    if (llama_vocab_get_add_eos(ref) || llama_vocab_get_add_eos(cand)) {
        fprintf(stderr, "--perplexity-window requires add_eos=false for both vocabularies\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// output writer
// ---------------------------------------------------------------------------

// Write the whole VLMK file at once (header + records), to <path>.tmp first,
// fsync, then atomically rename. Metrics are tiny (76 bytes/position), so
// unlike vlm-score's streaming vlms_writer there is no need to stream — a
// buffered single-shot write keeps the commit logic trivially reviewable and
// a crashed/partial run never leaves a complete-looking output behind.
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
    // Sixth word: llama.cpp's OWN position count after prefill. For a
    // text-only model that equals n_prefill, or the whole window length in perplexity-window mode.
    // The field exists so both lanes' headers carry the same
    // contract, and 0 still means "not recorded".
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

// ---------------------------------------------------------------------------
// manifest
// ---------------------------------------------------------------------------

struct manifest_entry {
    nlohmann::ordered_json reference_vocabulary;
    bool ok = false;
    std::string error;
    std::string tokens_in_path;
    std::string output_metrics_path;
    int n_prefill = -1;
};

static bool read_manifest(const std::string & path, std::vector<manifest_entry> & entries) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "failed to open manifest %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        manifest_entry entry;
        try {
            if (line.empty()) {
                throw std::runtime_error("empty manifest line");
            }
            const auto j = nlohmann::ordered_json::parse(line);
            if (!j.is_object()) {
                throw std::runtime_error("manifest line is not a JSON object");
            }
            if (!j.contains("tokens_in") || !j.at("tokens_in").is_string()) {
                throw std::runtime_error("missing or invalid tokens_in");
            }
            if (!j.contains("n_prefill") || !j.at("n_prefill").is_number_integer()) {
                throw std::runtime_error("missing or invalid n_prefill");
            }
            if (!j.contains("output_metrics") || !j.at("output_metrics").is_string()) {
                throw std::runtime_error("missing or invalid output_metrics");
            }

            entry.reference_vocabulary = j.value("reference_vocabulary", nlohmann::ordered_json());
            entry.tokens_in_path       = j.at("tokens_in").get<std::string>();
            entry.n_prefill            = j.at("n_prefill").get<int>();
            entry.output_metrics_path  = j.at("output_metrics").get<std::string>();

            if (entry.tokens_in_path.empty() || entry.output_metrics_path.empty()) {
                throw std::runtime_error("tokens_in and output_metrics must be non-empty");
            }
            entry.ok = true;
        } catch (const std::exception & e) {
            entry.error = e.what();
        }
        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        fprintf(stderr, "manifest contains no entries: %s\n", path.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

// One loaded model plus its decode state for the current item.
struct model_side {
    const char *      tag;   // "ref" / "cand" (log labels)
    llama_model_ptr   model;
    llama_context_ptr lctx;
    llama_pos         n_past = 0;
};

static bool load_side(model_side & s, const char * tag,
                      const std::string & model_path,
                      const vlm_kld_args & args) {
    s.tag = tag;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    s.model.reset(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!s.model) {
        fprintf(stderr, "[%s] failed to load model %s\n", tag, model_path.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = args.n_ctx;
    cparams.n_batch   = args.n_batch;
    cparams.n_ubatch  = args.n_ubatch;
    cparams.n_seq_max = 1;
    // Set both thread knobs: multi-token ubatches (prefill + chunked teacher
    // forcing) use n_threads_batch, which would otherwise silently stay at
    // ggml's compiled default and ignore -t on CPU runs.
    cparams.n_threads       = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    cparams.flash_attn_type = args.flash_attn_type;
    // Default false matches the common llama.cpp CLI: the low-level context
    // default is full SWA, which allocates n_ctx cells for every sliding-window
    // layer, and this tool always uses a single sequential sequence. --swa-full
    // restores it for runs that must stay comparable with earlier collections.
    cparams.swa_full        = args.swa_full;
    s.lctx.reset(llama_init_from_model(s.model.get(), cparams));
    if (!s.lctx) {
        fprintf(stderr, "[%s] failed to create llama_context\n", tag);
        return false;
    }

    return true;
}

// Prefill one side with tokens[0:n_prefill] on sequence 0 of its own context.
// The final prefill token keeps logits, which predict answer token 0.
static bool prefill_side(model_side & s,
                         const std::vector<int32_t> & tokens,
                         int n_prefill,
                         int n_batch,
                         stderr_prefix * lp) {
    const int chunk_max = std::max(1, n_batch);
    llama_batch batch = llama_batch_init(chunk_max, 0, 1);
    bool ok = true;
    for (int base = 0; base < n_prefill && ok; base += chunk_max) {
        const int chunk = std::min(chunk_max, n_prefill - base);
        common_batch_clear(batch);
        for (int j = 0; j < chunk; ++j) {
            const int pos = base + j;
            common_batch_add(batch, (llama_token) tokens[(size_t) pos],
                             pos, {0}, /*logits=*/pos == n_prefill - 1);
        }
        if (llama_decode(s.lctx.get(), batch) != 0) {
            prefixed_fprintf(lp, "[%s] llama_decode failed during prefill at tokens %d..%d\n",
                             s.tag, base, base + chunk - 1);
            ok = false;
            break;
        }
    }
    llama_batch_free(batch);
    if (!ok) {
        return false;
    }
    s.n_past = (llama_pos) n_prefill;
    prefixed_fprintf(lp, "[%s] prefill done, n_past=%d\n", s.tag, (int) s.n_past);
    return true;
}

static bool score_perplexity_window(
        const vlm_kld_args & args,
        model_side & ref,
        model_side & cand,
        int32_t n_vocab,
        int32_t n_batch,
        const std::vector<int32_t> & tokens,
        stderr_prefix * lp) {
    const int length = (int) tokens.size();
    if (n_batch != length || llama_n_ctx(ref.lctx.get()) != (uint32_t) length ||
            llama_n_ctx(cand.lctx.get()) != (uint32_t) length ||
            llama_n_ubatch(ref.lctx.get()) != (uint32_t) length ||
            llama_n_ubatch(cand.lctx.get()) != (uint32_t) length) {
        prefixed_fprintf(lp, "--perplexity-window effective context and batch sizes must equal the window length\n");
        return false;
    }
    const int first = length / 2;
    const int n_eval = length - first - 1;
    std::vector<kld_record> records(n_eval);
    std::vector<const float *> ref_rows, cand_rows;
    std::vector<int32_t> targets;
    ref_rows.reserve(n_eval);
    cand_rows.reserve(n_eval);
    targets.reserve(n_eval);

    llama_batch batch = llama_batch_init(length, 0, 1);
    common_batch_clear(batch);
    for (int pos = 0; pos < length; ++pos) {
        common_batch_add(batch, tokens[pos], pos, {0}, pos >= first);
    }
    const bool decoded = llama_decode(ref.lctx.get(), batch) == 0 && llama_decode(cand.lctx.get(), batch) == 0;
    llama_batch_free(batch);
    if (!decoded) {
        prefixed_fprintf(lp, "llama_decode failed for the full perplexity window\n");
        return false;
    }
    ref.n_past = cand.n_past = length;
    for (int pos = first; pos < length - 1; ++pos) {
        const float * r = llama_get_logits_ith(ref.lctx.get(), pos);
        const float * c = llama_get_logits_ith(cand.lctx.get(), pos);
        if (!r || !c) {
            prefixed_fprintf(lp, "missing logits for perplexity window position %d\n", pos);
            return false;
        }
        ref_rows.push_back(r);
        cand_rows.push_back(c);
        targets.push_back(tokens[pos + 1]);
    }
    compute_records_parallel(ref_rows, cand_rows, targets, n_vocab,
                             resolve_metric_threads(args.metric_threads), records.data());
    for (int i = 0; i < n_eval; ++i) {
        if (!validate_kld_record_finite(records[i], i, lp)) {
            return false;
        }
    }
    return write_vlmk_file(args.output_metrics_path, n_vocab, args.n_prefill, length, records, lp);
}

// Score one item: prefill both sides, teacher-force the shared answer tokens
// through both in chunks, and compute one kld_record per answer position from
// the two in-memory logit rows. Records are buffered (76 B/position) and the
// output is committed atomically at the end.
static bool score_one(
        const vlm_kld_args & args,
        model_side & ref,
        model_side & cand,
        int32_t n_vocab,
        int32_t n_batch,
        stderr_prefix * lp) {
    // Every item starts from clean KV on both sides (no-op on first use).
    llama_memory_clear(llama_get_memory(ref.lctx.get()),  args.perplexity_window);
    llama_memory_clear(llama_get_memory(cand.lctx.get()), args.perplexity_window);

    // --- read tokens ---
    std::vector<int32_t> tokens_full = read_tokens_bin(args.tokens_in_path, lp);
    if (tokens_full.empty()) {
        prefixed_fprintf(lp, "tokens-in is empty or unreadable\n");
        return false;
    }
    if (args.perplexity_window && !validate_perplexity_window_input(args, tokens_full.size(), lp)) {
        return false;
    }
    const int L = (int) tokens_full.size();
    if (args.n_prefill <= 0 || args.n_prefill >= L) {
        prefixed_fprintf(lp, "invalid n_prefill=%d for L=%d\n", args.n_prefill, L);
        return false;
    }
    if (!validate_tokens_in_vocab(tokens_full, n_vocab, lp)) {
        return false;
    }
    const int n_answer = L - args.n_prefill;
    const int n_eval = (args.num_eval_tokens > 0)
                       ? std::min(args.num_eval_tokens, n_answer)
                       : n_answer;
    prefixed_fprintf(lp, "tokens_full=%d, n_prefill=%d, n_answer=%d, n_eval=%d, vocab=%d\n",
                     L, args.n_prefill, n_answer, n_eval, n_vocab);

    if (args.perplexity_window) {
        return score_perplexity_window(args, ref, cand, n_vocab, n_batch, tokens_full, lp);
    }

    // --- prefill both sides ---
    if (!prefill_side(ref,  tokens_full, args.n_prefill, n_batch, lp) ||
        !prefill_side(cand, tokens_full, args.n_prefill, n_batch, lp)) {
        return false;
    }
    if (ref.n_past != cand.n_past || ref.n_past != args.n_prefill) {
        prefixed_fprintf(lp, "internal error: ref n_past=%d cand n_past=%d n_prefill=%d\n",
                         (int) ref.n_past, (int) cand.n_past, args.n_prefill);
        return false;
    }

    return kld_teacher_force_and_write(args, ref, cand, n_vocab, n_batch, n_eval,
                                       tokens_full, lp, write_vlmk_file);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const int identity_command = skymizer_identity::command(argc, argv);
    if (identity_command >= 0) { return identity_command; }
    vlm_kld_args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }
    if (args.print_vlmk_version) {
        printf("%u\n", VLMK_VERSION);
        return 0;
    }
    if (args.self_test) {
        return run_self_test() ? 0 : 1;
    }

    std::vector<manifest_entry> manifest_entries;
    if (!args.manifest_path.empty() && !read_manifest(args.manifest_path, manifest_entries)) {
        return 1;
    }
    if (args.perplexity_window) {
        const auto valid_input = [](const vlm_kld_args & row) {
            return validate_perplexity_window_input(row, read_tokens_bin(row.tokens_in_path, nullptr).size(), nullptr);
        };
        if (args.manifest_path.empty()) {
            if (!valid_input(args)) { return 1; }
        } else {
            for (const auto & entry : manifest_entries) {
                if (!entry.ok) {
                    fprintf(stderr, "%s\n", entry.error.c_str());
                    return 1;
                }
                vlm_kld_args row = args;
                row.tokens_in_path = entry.tokens_in_path;
                row.n_prefill = entry.n_prefill;
                if (!valid_input(row)) { return 1; }
            }
        }
    }

    try {
        std::vector<skymizer_identity::json> expected;
        for (const auto & entry : manifest_entries) {
            if (entry.ok && !entry.reference_vocabulary.is_null()) {
                expected.push_back(entry.reference_vocabulary);
            }
        }
        skymizer_identity::check_manifest(expected, args.ref_model_path, args.cand_model_path,
                                          args.allow_vocab_attr_mismatch);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    ggml_backend_load_all();

    model_side ref, cand;
    if (!load_side(ref,  "ref",  args.ref_model_path,  args) ||
        !load_side(cand, "cand", args.cand_model_path, args)) {
        return 1;
    }

    // The metrics compare distributions index-by-index, so both models must
    // share one vocabulary. (Quantizations of the same base model do.)
    const llama_vocab * ref_vocab  = llama_model_get_vocab(ref.model.get());
    const llama_vocab * cand_vocab = llama_model_get_vocab(cand.model.get());
    if (args.perplexity_window && !validate_perplexity_window_vocab(ref_vocab, cand_vocab)) {
        return 1;
    }
    if (!validate_vocab_id_mapping(ref_vocab, cand_vocab, nullptr,
                                   args.allow_vocab_attr_mismatch)) {
        return 1;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(ref_vocab);

    // Both contexts were created with the same n_batch/n_ctx, so their
    // effective (clamped) batch sizes agree; reconcile via min() anyway.
    const int eff_n_batch = (int) std::min(llama_n_batch(ref.lctx.get()),
                                           llama_n_batch(cand.lctx.get()));

    if (args.manifest_path.empty()) {
        try {
            return score_one(args, ref, cand, n_vocab, eff_n_batch, nullptr) ? 0 : 1;
        } catch (const std::exception & ex) {
            fprintf(stderr, "unhandled exception: %s\n", ex.what());
            return 1;
        }
    }

    bool any_failed = false;
    ggml_log_callback saved_llama_log_callback = nullptr;
    void * saved_llama_log_user_data = nullptr;
    llama_log_get(&saved_llama_log_callback, &saved_llama_log_user_data);

    const size_t n_entries = manifest_entries.size();
    for (size_t i = 0; i < n_entries; ++i) {
        const manifest_entry & e = manifest_entries[i];
        if (!e.ok) {
            fprintf(stderr, "[manifest entry %zu/%zu] %s\n", i + 1, n_entries, e.error.c_str());
            any_failed = true;
            continue;
        }

        stderr_prefix prefix;
        prefix.text = "[row " + std::to_string(i + 1) + "/" + std::to_string(n_entries) + "] ";
        llama_log_set(prefixed_log_callback, &prefix);

        vlm_kld_args ea = args;
        ea.tokens_in_path       = e.tokens_in_path;
        ea.n_prefill            = e.n_prefill;
        ea.output_metrics_path  = e.output_metrics_path;
        ea.manifest_path.clear();

        bool row_ok = false;
        const auto t_start = std::chrono::steady_clock::now();
        try {
            row_ok = score_one(ea, ref, cand, n_vocab, eff_n_batch, &prefix);
        } catch (const std::exception & ex) {
            prefixed_fprintf(&prefix, "unhandled exception: %s\n", ex.what());
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

        llama_log_set(saved_llama_log_callback, saved_llama_log_user_data);

        if (row_ok) {
            fprintf(stderr, "[row] DONE output_metrics=%s wall_s=%.3f\n",
                    e.output_metrics_path.c_str(), wall_s);
        } else {
            any_failed = true;
        }
    }

    return any_failed ? 1 : 0;
}
