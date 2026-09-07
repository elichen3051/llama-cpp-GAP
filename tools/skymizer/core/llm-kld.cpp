// Compare two text models on the same teacher-forced prompt and answer tokens.
// Write per-answer-token metrics; metric computation and the VLMK layout are in skymizer-vlmk-kernel.h.
//
// CLI:
//   --ref-model <gguf>                 Reference LLM weights
//   --cand-model <gguf>                Candidate LLM weights
//   --tokens-in <path>                 Binary file: int32[L] of full input_ids
//   --n-prefill <int>                  First answer-token index (must match the stored value)
//   --output-metrics <path>            Output binary file (VLMK records)
//   --manifest <jsonl>                 Batch JSONL input takes precedence over per-row CLI flags. Load both models once.
//   --num-eval-tokens <int>            Answer positions to score (-1 = all, default; clamped to L - n_prefill)
//   --perplexity-window                Decode one full context window per side, like classic np1 perplexity.
//                                      Requires even L = -c = -b = -ub and n_prefill = L/2 + 1; score all L/2 - 1 targets.
//                                      Input tokens already include the caller's BOS policy; no tokens are changed.
//   -ngl <int>                         GPU layers to offload, per model (default 99)
//   -c <int>                           Context size, per model (default 32768)
//   -b <int>                           Logical batch size for prefill (default 2048)
//   -ub <int>                          Physical micro-batch and default teacher-forcing chunk size (default 2048)
//   --tf-chunk <int>                   Teacher-forcing chunk size (-1 = follow -ub, default).
//                                      Keep chunk size fixed across comparable runs; batch shape can change floating-point rounding.
//   -t <int>                           CPU threads for llama decode (default -1 = ggml's compiled default; irrelevant for fully-offloaded GPU runs)
//   --metric-threads <int>             Threads for the metric kernel (-1 = hardware concurrency, default)
//   --flash-attn                       Require flash attention
//   --no-flash-attn                    Disable flash attention (default: auto)
//   --swa-full                         Use full-size SWA KV caches (default: window-sized).
//                                      Keep this setting fixed across comparable runs; cache layout can change floating-point rounding.
//   --allow-vocab-attr-mismatch        Allow and log token-attribute differences; token text must match at every ID.
//   --self-test                        Run the metric-kernel self test without models and exit 0/1
//   --vlmk-version                     Print the VLMK format version and exit 0

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

struct llm_kld_args {
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

static bool parse_args(int argc, char ** argv, llm_kld_args & args) {
    bool saw_flash_attn = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        auto need_int = [&](const char * what, int & out) {
            const char * value = need(what);
            return value != nullptr && parse_int_arg(what, value, out);
        };
        if (key == "--ref-model") {
            const char * value = need("--ref-model");
            if (!value) {
                return false;
            }
            args.ref_model_path = value;
        } else if (key == "--cand-model") {
            const char * value = need("--cand-model");
            if (!value) {
                return false;
            }
            args.cand_model_path = value;
        } else if (key == "--tokens-in") {
            const char * value = need("--tokens-in");
            if (!value) {
                return false;
            }
            args.tokens_in_path = value;
        } else if (key == "--n-prefill") {
            if (!need_int("--n-prefill", args.n_prefill)) {
                return false;
            }
        } else if (key == "--output-metrics") {
            const char * value = need("--output-metrics");
            if (!value) {
                return false;
            }
            args.output_metrics_path = value;
        } else if (key == "--manifest") {
            const char * value = need("--manifest");
            if (!value) {
                return false;
            }
            args.manifest_path = value;
        } else if (key == "--num-eval-tokens") {
            if (!need_int("--num-eval-tokens", args.num_eval_tokens)) {
                return false;
            }
        } else if (key == "-ngl") {
            if (!need_int("-ngl", args.n_gpu_layers)) {
                return false;
            }
        } else if (key == "-c") {
            if (!need_int("-c", args.n_ctx)) {
                return false;
            }
        } else if (key == "-b") {
            if (!need_int("-b", args.n_batch)) {
                return false;
            }
        } else if (key == "-ub") {
            if (!need_int("-ub", args.n_ubatch)) {
                return false;
            }
        } else if (key == "--tf-chunk") {
            if (!need_int("--tf-chunk", args.tf_chunk)) {
                return false;
            }
        } else if (key == "-t") {
            if (!need_int("-t", args.n_threads)) {
                return false;
            }
        } else if (key == "--metric-threads") {
            if (!need_int("--metric-threads", args.metric_threads)) {
                return false;
            }
        } else if (key == "--flash-attn" || key == "--no-flash-attn") {
            if (saw_flash_attn) {
                fprintf(stderr, "flash-attention mode specified more than once\n");
                return false;
            }
            saw_flash_attn = true;
            args.flash_attn_type = key == "--flash-attn"
                ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else if (key == "--swa-full") {
            args.swa_full = true;
        } else if (key == "--perplexity-window") {
            args.perplexity_window = true;
        } else if (key == "--allow-vocab-attr-mismatch") {
            args.allow_vocab_attr_mismatch = true;
        } else if (key == "--self-test") {
            args.self_test = true;
        } else if (key == "--vlmk-version") {
            args.print_vlmk_version = true;
        } else if (key == "--help" || key == "-h") {
            fprintf(stderr, "see header of llm-kld.cpp for CLI documentation\n");
            return false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", key.c_str());
            return false;
        }
    }
    if (args.self_test || args.print_vlmk_version) {
        return true;   // no other arguments required
    }
    if (args.ref_model_path.empty() || args.cand_model_path.empty()) {
        fprintf(stderr, "missing required arguments (run with --help)\n");
        return false;
    }
    if (args.manifest_path.empty()) {
        if (args.tokens_in_path.empty() || args.output_metrics_path.empty() ||
            args.n_prefill < 0) {
            fprintf(stderr, "missing required arguments (run with --help)\n");
            return false;
        }
    }
    if (args.n_batch < 1 || args.n_ubatch < 1) {
        fprintf(stderr, "-b and -ub must be >= 1, got -b %d -ub %d\n", args.n_batch, args.n_ubatch);
        return false;
    }
    if (args.n_ubatch > args.n_batch) {
        fprintf(stderr, "-ub (%d) must be <= -b (%d)\n", args.n_ubatch, args.n_batch);
        return false;
    }
    if (args.tf_chunk != -1 && args.tf_chunk < 1) {
        fprintf(stderr, "--tf-chunk must be -1 (follow -ub) or >= 1, got %d\n", args.tf_chunk);
        return false;
    }
    if (args.num_eval_tokens != -1 && args.num_eval_tokens < 1) {
        fprintf(stderr, "num-eval-tokens must be -1 (all) or >= 1, got %d\n", args.num_eval_tokens);
        return false;
    }
    if (args.metric_threads != -1 && args.metric_threads < 1) {
        fprintf(stderr, "--metric-threads must be -1 (auto) or >= 1, got %d\n", args.metric_threads);
        return false;
    }
    if (args.perplexity_window &&
            (args.n_ctx < 4 || args.n_ctx % 2 != 0 || args.n_batch != args.n_ctx || args.n_ubatch != args.n_ctx ||
             (args.tf_chunk != -1 && args.tf_chunk != args.n_ctx) ||
             (args.num_eval_tokens != -1 && args.num_eval_tokens != args.n_ctx / 2 - 1))) {
        fprintf(stderr, "--perplexity-window requires even -c >= 4, -b = -ub = -c, --tf-chunk -1 or -c, and all -c/2 - 1 targets\n");
        return false;
    }
    return true;
}

static bool validate_perplexity_window_input(const llm_kld_args & args, size_t n_tokens, stderr_prefix * lp) {
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

// One loaded model plus its decode state for the current item.
struct model_side {
    const char *      tag;   // "ref" / "cand" (log labels)
    llama_model_ptr   model;
    llama_context_ptr lctx;
    llama_pos         n_past = 0;
};

static bool load_side(model_side & s, const char * tag,
                      const std::string & model_path,
                      const llm_kld_args & args) {
    s.tag = tag;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    s.model.reset(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!s.model) {
        fprintf(stderr, "[%s] failed to load model %s\n", tag, model_path.c_str());
        return false;
    }

    llama_context_params cparams = kld_context_params(args);
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
        const llm_kld_args & args,
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

// Prefill both sides, then score the shared answer tokens in chunks.
// Buffer records until the complete output can be written atomically.
static bool score_one(
        const llm_kld_args & args,
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

static int score_manifest(
        const llm_kld_args & args,
        const std::vector<manifest_entry> & manifest_entries,
        model_side & ref,
        model_side & cand,
        int32_t n_vocab,
        int eff_n_batch) {
    bool any_failed = false;
    ggml_log_callback saved_llama_log_callback = nullptr;
    void * saved_llama_log_user_data = nullptr;
    llama_log_get(&saved_llama_log_callback, &saved_llama_log_user_data);

    const size_t n_entries = manifest_entries.size();
    for (size_t i = 0; i < n_entries; ++i) {
        const manifest_entry & entry = manifest_entries[i];
        if (!entry.ok) {
            fprintf(stderr, "[manifest entry %zu/%zu] %s\n", i + 1, n_entries, entry.error.c_str());
            any_failed = true;
            continue;
        }

        stderr_prefix prefix;
        prefix.text = "[row " + std::to_string(i + 1) + "/" + std::to_string(n_entries) + "] ";
        llama_log_set(prefixed_log_callback, &prefix);

        llm_kld_args row_args = args;
        row_args.tokens_in_path       = entry.tokens_in_path;
        row_args.n_prefill            = entry.n_prefill;
        row_args.output_metrics_path  = entry.output_metrics_path;
        row_args.manifest_path.clear();

        bool row_ok = false;
        const auto t_start = std::chrono::steady_clock::now();
        try {
            row_ok = score_one(row_args, ref, cand, n_vocab, eff_n_batch, &prefix);
        } catch (const std::exception & ex) {
            prefixed_fprintf(&prefix, "unhandled exception: %s\n", ex.what());
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

        llama_log_set(saved_llama_log_callback, saved_llama_log_user_data);

        if (row_ok) {
            fprintf(stderr, "[row] DONE output_metrics=%s wall_s=%.3f\n",
                    entry.output_metrics_path.c_str(), wall_s);
        } else {
            any_failed = true;
        }
    }

    return any_failed ? 1 : 0;
}

int main(int argc, char ** argv) {
    const int identity_command = skymizer_identity::command(argc, argv);
    if (identity_command >= 0) {
        return identity_command;
    }
    llm_kld_args args;
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
        const auto valid_input = [](const llm_kld_args & row) {
            return validate_perplexity_window_input(row, read_tokens_bin(row.tokens_in_path, nullptr).size(), nullptr);
        };
        if (args.manifest_path.empty()) {
            if (!valid_input(args)) {
                return 1;
            }
        } else {
            for (const auto & entry : manifest_entries) {
                if (!entry.ok) {
                    fprintf(stderr, "%s\n", entry.error.c_str());
                    return 1;
                }
                llm_kld_args row = args;
                row.tokens_in_path = entry.tokens_in_path;
                row.n_prefill = entry.n_prefill;
                if (!valid_input(row)) {
                    return 1;
                }
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

    // Both distributions must use the same token IDs.
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

    // Use the smaller effective batch size after context setup.
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

    return score_manifest(args, manifest_entries, ref, cand, n_vocab, eff_n_batch);
}
