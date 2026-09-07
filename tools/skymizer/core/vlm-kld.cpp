// Compare two model/projector pairs on the same teacher-forced prompt and answer.
// Write per-answer-token metrics; metric computation and the VLMK layout are in skymizer-vlmk-kernel.h.
//
// CLI:
//   --ref-model <gguf>                 Reference LLM weights
//   --ref-mmproj <gguf>                Reference vision projector
//   --cand-model <gguf>                Candidate LLM weights
//   --cand-mmproj <gguf>               Candidate vision projector
//   --image <path>                     Image file (repeatable; one --image per <__media__> marker in --formatted-chat)
//   --formatted-chat <path>            File containing the rendered chat string
//   --tokens-in <path>                 Binary file: int32[L] of full input_ids
//   --n-prefill <int>                  First answer-token index (must match the stored value)
//   --output-metrics <path>            Output binary file (VLMK records)
//   --manifest <jsonl>                 Batch JSONL input takes precedence over per-row CLI flags. Load both models once.
//   --image-min-tokens <int>           Lower bound; -1 = use model metadata (default -1)
//   --image-max-tokens <int>           Upper bound; -1 = use model metadata (default -1)
//   --num-eval-tokens <int>            Answer positions to score (-1 = all, default; clamped to L - n_prefill)
//   -ngl <int>                         GPU layers to offload, per model (default 99)
//   -c <int>                           Context size, per model (default 32768)
//   -b <int>                           Logical batch size for prefill (default 2048)
//   -ub <int>                          Physical micro-batch and default teacher-forcing chunk size (default 2048)
//   --tf-chunk <int>                   Teacher-forcing chunk size (-1 = follow -ub, default).
//                                      Keep chunk size fixed across comparable runs; batch shape can change floating-point rounding.
//   -t <int>                           CPU threads for llama decode and mtmd image encode (default -1 = ggml's compiled default; irrelevant for fully-offloaded GPU runs)
//   --metric-threads <int>             Threads for the metric kernel (-1 = hardware concurrency, default)
//   --flash-attn                       Require flash attention
//   --no-flash-attn                    Disable flash attention (default: auto)
//   --swa-full                         Use full-size SWA KV caches (default: window-sized).
//                                      Keep this setting fixed across comparable runs; cache layout can change floating-point rounding.
//   --add-special                      mtmd add_special=true for the prefix; the manifest key "add_special" sets it per row
//   --allow-prefix-drift               Image-span drift vs tokens_in only warns (default: error, like any text-token mismatch); --require-prefix-match restores the default
//   --allow-n-past-drift               Allow different prefill positions on the two sides.
//                                      This changes answer conditioning; use only when the drift itself is being measured.
//   --allow-vocab-attr-mismatch        Allow and log token-attribute differences; token text must match at every ID.
//   --self-test                        Run the metric-kernel self test without models and exit 0/1
//   --vlmk-version                     Print the VLMK format version and exit 0

#include "skymizer-identity.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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

struct vlm_kld_args {
    std::string ref_model_path;
    std::string ref_mmproj_path;
    std::string cand_model_path;
    std::string cand_mmproj_path;
    std::vector<std::string> image_paths;
    std::string formatted_chat_path;
    std::string tokens_in_path;
    std::string output_metrics_path;
    std::string manifest_path;
    int n_prefill = -1;
    int image_min_tokens = -1;   // -1 = use model metadata
    int image_max_tokens = -1;
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
    bool allow_n_past_drift = false;   // see the drift guard in score_one()
    bool add_special = false;          // Replay the generator's mtmd_input_text.add_special setting.
    bool require_prefix_match = true;  // Reject image-span drift; text tokens must always match.
    bool allow_vocab_attr_mismatch = false;  // see validate_vocab_id_mapping()
    bool self_test = false;
    bool print_vlmk_version = false;
};

static bool parse_args(int argc, char ** argv, vlm_kld_args & args) {
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
        } else if (key == "--ref-mmproj") {
            const char * value = need("--ref-mmproj");
            if (!value) {
                return false;
            }
            args.ref_mmproj_path = value;
        } else if (key == "--cand-model") {
            const char * value = need("--cand-model");
            if (!value) {
                return false;
            }
            args.cand_model_path = value;
        } else if (key == "--cand-mmproj") {
            const char * value = need("--cand-mmproj");
            if (!value) {
                return false;
            }
            args.cand_mmproj_path = value;
        } else if (key == "--image") {
            const char * value = need("--image");
            if (!value) {
                return false;
            }
            args.image_paths.emplace_back(value);
        } else if (key == "--formatted-chat") {
            const char * value = need("--formatted-chat");
            if (!value) {
                return false;
            }
            args.formatted_chat_path = value;
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
        } else if (key == "--image-min-tokens") {
            if (!need_int("--image-min-tokens", args.image_min_tokens)) {
                return false;
            }
        } else if (key == "--image-max-tokens") {
            if (!need_int("--image-max-tokens", args.image_max_tokens)) {
                return false;
            }
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
        } else if (key == "--allow-n-past-drift") {
            args.allow_n_past_drift = true;
        } else if (key == "--add-special") {
            args.add_special = true;
        } else if (key == "--require-prefix-match") {
            args.require_prefix_match = true;
        } else if (key == "--allow-prefix-drift") {
            args.require_prefix_match = false;
        } else if (key == "--allow-vocab-attr-mismatch") {
            args.allow_vocab_attr_mismatch = true;
        } else if (key == "--self-test") {
            args.self_test = true;
        } else if (key == "--vlmk-version") {
            args.print_vlmk_version = true;
        } else if (key == "--help" || key == "-h") {
            fprintf(stderr, "see header of vlm-kld.cpp for CLI documentation\n");
            return false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", key.c_str());
            return false;
        }
    }
    if (args.self_test || args.print_vlmk_version) {
        return true;   // no other arguments required
    }
    if (args.ref_model_path.empty() || args.ref_mmproj_path.empty() ||
        args.cand_model_path.empty() || args.cand_mmproj_path.empty()) {
        fprintf(stderr, "missing required arguments (run with --help)\n");
        return false;
    }
    if (args.manifest_path.empty()) {
        if (args.image_paths.empty() || args.formatted_chat_path.empty() ||
            args.tokens_in_path.empty() || args.output_metrics_path.empty() ||
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
    return true;
}

struct manifest_entry {
    nlohmann::ordered_json reference_vocabulary;
    bool ok = false;
    std::string error;
    std::vector<std::string> image_paths;
    std::string formatted_chat_path;
    std::string tokens_in_path;
    std::string output_metrics_path;
    int n_prefill = -1;
    bool add_special = false;   // optional manifest key; see vlm_kld_args::add_special
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
            if (!j.contains("images") || !j.at("images").is_array()) {
                throw std::runtime_error("missing or invalid images");
            }
            if (!j.contains("formatted_chat") || !j.at("formatted_chat").is_string()) {
                throw std::runtime_error("missing or invalid formatted_chat");
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

            entry.image_paths          = j.at("images").get<std::vector<std::string>>();
            entry.formatted_chat_path  = j.at("formatted_chat").get<std::string>();
            entry.reference_vocabulary = j.value("reference_vocabulary", nlohmann::ordered_json());
            entry.tokens_in_path       = j.at("tokens_in").get<std::string>();
            entry.n_prefill            = j.at("n_prefill").get<int>();
            entry.output_metrics_path  = j.at("output_metrics").get<std::string>();
            if (j.contains("add_special")) {
                if (!j.at("add_special").is_boolean()) {
                    throw std::runtime_error("add_special must be a boolean");
                }
                entry.add_special = j.at("add_special").get<bool>();
            }

            if (entry.formatted_chat_path.empty() || entry.tokens_in_path.empty() ||
                entry.output_metrics_path.empty()) {
                throw std::runtime_error("formatted_chat, tokens_in, and output_metrics must be non-empty");
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

// One loaded (model, mmproj) pair plus its decode state for the current item.
struct model_side {
    const char *      tag;   // "ref" / "cand" (log labels)
    llama_model_ptr   model;
    llama_context_ptr lctx;
    mtmd::context_ptr vctx;
    llama_pos         n_past = 0;
};

static bool load_side(model_side & s, const char * tag,
                      const std::string & model_path, const std::string & mmproj_path,
                      const vlm_kld_args & args) {
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

    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu          = true;
    vparams.print_timings    = true;
    vparams.n_threads        = args.n_threads;
    vparams.flash_attn_type  = cparams.flash_attn_type;
    vparams.warmup           = true;
    vparams.image_min_tokens = args.image_min_tokens;
    vparams.image_max_tokens = args.image_max_tokens;
    s.vctx.reset(mtmd_init_from_file(mmproj_path.c_str(), s.model.get(), vparams));
    if (!s.vctx) {
        fprintf(stderr, "[%s] failed to load mmproj %s\n", tag, mmproj_path.c_str());
        return false;
    }
    return true;
}

// Match text tokens and contiguous media spans against the stored prefix.
// Adjacent tiles share one placeholder run; only span length drift can be allowed.
// Text tokens and the total stored prefix must still match.
static bool check_prefix_against_tokens(const mtmd::input_chunks & chunks,
                                        const std::vector<int32_t> & tokens,
                                        int n_prefill,
                                        bool strict,
                                        const char * tag,
                                        stderr_prefix * lp) {
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    size_t pos = 0;
    size_t img_idx = 0;
    size_t n_drift = 0;
    for (size_t ci = 0; ci < n_chunks; ++ci) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), ci);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
            for (size_t i = 0; i < n; ++i, ++pos) {
                const int32_t stored = pos < (size_t) n_prefill ? tokens[pos] : -1;
                if (stored != (int32_t) t[i]) {
                    prefixed_fprintf(lp,
                        "[%s] prefix mismatch at index %zu: stored=%d, this mtmd=%d (text chunk %zu). "
                        "The stored prompt is not what this scorer tokenized: check formatted_chat, "
                        "add_special (manifest \"add_special\" / --add-special) and that the GGUF "
                        "vocab matches the generator's.\n",
                        tag, pos, stored, (int) t[i], ci);
                    return false;
                }
            }
            continue;
        }
        size_t n = mtmd_input_chunk_get_n_tokens(c);
        while (ci + 1 < n_chunks) {
            const auto * next = mtmd_input_chunks_get(chunks.ptr.get(), ci + 1);
            if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                break;
            }
            n += mtmd_input_chunk_get_n_tokens(next);
            ++ci;
        }
        if (pos >= (size_t) n_prefill) {
            prefixed_fprintf(lp, "[%s] prefix mismatch: media chunk %zu starts at %zu, past n_prefill=%d\n",
                             tag, ci, pos, n_prefill);
            return false;
        }
        const int32_t pad = tokens[pos];
        size_t run = 0;
        while (pos + run < (size_t) n_prefill && tokens[pos + run] == pad) {
            ++run;
        }
        if (run != n) {
            prefixed_fprintf(lp,
                "[%s] %s: image %zu spans %zu embedding tokens in this scorer but %zu stored "
                "placeholder tokens (vision preprocessing drift between generator and scorer: "
                "different llama.cpp build or --image-min/max-tokens budget)\n",
                tag, strict ? "prefix mismatch" : "WARNING", img_idx, n, run);
            if (strict) {
                return false;
            }
            // Realign on the STORED run so the text after the image (and any
            // later image) is still checked; only the span length drifted.
            n_drift++;
            pos += run;
            ++img_idx;
            continue;
        }
        pos += n;
        ++img_idx;
    }
    if (pos != (size_t) n_prefill) {
        prefixed_fprintf(lp, "[%s] prefix mismatch: this mtmd produced %zu prefix tokens, stored n_prefill=%d\n",
                         tag, pos, n_prefill);
        return false;
    }
    if (n_drift > 0) {
        prefixed_fprintf(lp, "[%s] prefix check: text tokens match stored input_ids[:n_prefill]; %zu of %zu image "
                             "span%s drifted (allowed by --allow-prefix-drift)\n",
                         tag, n_drift, img_idx, img_idx == 1 ? "" : "s");
        return true;
    }
    prefixed_fprintf(lp, "[%s] prefix check: %zu tokens (%zu image span%s) match stored input_ids[:n_prefill]\n",
                     tag, pos, img_idx, img_idx == 1 ? "" : "s");
    return true;
}

static bool prefill_side(model_side & s,
                         const std::vector<std::string> & image_paths,
                         const std::string & formatted,
                         bool add_special,
                         const std::vector<int32_t> & tokens_full,
                         int n_prefill,
                         bool require_prefix_match,
                         int n_batch,
                         stderr_prefix * lp) {
    mtmd::bitmaps bmps_owned;
    bmps_owned.entries.reserve(image_paths.size());
    for (size_t i = 0; i < image_paths.size(); ++i) {
        auto media = mtmd_helper_bitmap_init_from_file(
            s.vctx.get(), image_paths[i].c_str(), false, mtmd_helper_init_opt_default());
        mtmd::bitmap b(media.bitmap);
        mtmd_helper::video_ptr video(media.video_ctx);
        if (video) {
            prefixed_fprintf(lp, "[%s] --image does not support video: %s\n", s.tag, image_paths[i].c_str());
            return false;
        }
        if (!b.ptr) {
            prefixed_fprintf(lp, "[%s] failed to load image[%zu] %s\n", s.tag, i, image_paths[i].c_str());
            return false;
        }
        bmps_owned.entries.push_back(std::move(b));
    }

    mtmd_input_text text{};
    text.text          = formatted.c_str();
    text.text_len      = formatted.size();
    text.add_special   = add_special;  // must replay the generator's setting (see vlm_kld_args::add_special)
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bmps = bmps_owned.c_ptr();
    int32_t rc = mtmd_tokenize(s.vctx.get(), chunks.ptr.get(), &text, bmps.data(), bmps.size());
    if (rc != 0) {
        prefixed_fprintf(lp,
                         "[%s] mtmd_tokenize failed (rc=%d); common cause: <__media__> marker count "
                         "in --formatted-chat must equal --image count (got %zu)\n",
                         s.tag, rc, bmps.size());
        return false;
    }
    if (!check_prefix_against_tokens(chunks, tokens_full, n_prefill, require_prefix_match, s.tag, lp)) {
        return false;
    }

    llama_pos new_n_past = 0;
    if (mtmd_helper_eval_chunks(
            s.vctx.get(), s.lctx.get(), chunks.ptr.get(), /*n_past=*/0, /*seq_id=*/0,
            /*n_batch=*/n_batch, /*logits_last=*/true, &new_n_past) != 0) {
        prefixed_fprintf(lp, "[%s] mtmd_helper_eval_chunks failed\n", s.tag);
        return false;
    }
    s.n_past = new_n_past;
    prefixed_fprintf(lp, "[%s] prefill done, n_past=%d\n", s.tag, (int) s.n_past);
    return true;
}

// Prefill both sides, then score the shared answer tokens in chunks.
// Buffer records until the complete output can be written atomically.
static bool score_one(
        const vlm_kld_args & args,
        model_side & ref,
        model_side & cand,
        int32_t n_vocab,
        int32_t n_batch,
        stderr_prefix * lp) {
    // Every item starts from clean KV on both sides (no-op on first use).
    llama_memory_clear(llama_get_memory(ref.lctx.get()),  false);
    llama_memory_clear(llama_get_memory(cand.lctx.get()), false);

    // --- read tokens ---
    std::vector<int32_t> tokens_full = read_tokens_bin(args.tokens_in_path, lp);
    if (tokens_full.empty()) {
        prefixed_fprintf(lp, "tokens-in is empty or unreadable\n");
        return false;
    }
    const int L = (int) tokens_full.size();
    if (args.n_prefill <= 0 || args.n_prefill >= L) {
        prefixed_fprintf(lp, "invalid n_prefill=%d for L=%d\n", args.n_prefill, L);
        return false;
    }
    for (int i = args.n_prefill; i < L; ++i) {
        if (tokens_full[i] < 0 || tokens_full[i] >= n_vocab) {
            prefixed_fprintf(lp, "answer token id %d at index %d out of vocab range [0, %d)\n",
                             tokens_full[i], i, n_vocab);
            return false;
        }
    }
    const int n_answer = L - args.n_prefill;
    const int n_eval = (args.num_eval_tokens > 0)
                       ? std::min(args.num_eval_tokens, n_answer)
                       : n_answer;
    prefixed_fprintf(lp, "tokens_full=%d, n_prefill=%d, n_answer=%d, n_eval=%d, vocab=%d, n_images=%zu\n",
                     L, args.n_prefill, n_answer, n_eval, n_vocab, args.image_paths.size());

    // --- prefill both sides ---
    const std::string formatted = slurp_file(args.formatted_chat_path, lp);
    if (formatted.empty()) {
        return false;
    }
    if (!prefill_side(ref,  args.image_paths, formatted, args.add_special, tokens_full, args.n_prefill,
                      args.require_prefix_match, n_batch, lp) ||
        !prefill_side(cand, args.image_paths, formatted, args.add_special, tokens_full, args.n_prefill,
                      args.require_prefix_match, n_batch, lp)) {
        return false;
    }

    // Different prefill lengths change the conditioning of every answer token.
    // Allow this only when the drift itself is being measured.
    if (ref.n_past != cand.n_past) {
        if (!args.allow_n_past_drift) {
            prefixed_fprintf(lp,
                "prefill drift: ref n_past=%d != cand n_past=%d. The two sides "
                "turned the same images into different numbers of embeddings, "
                "so every scored position is conditioned on a different "
                "prefix and the paired metrics would measure that, not the "
                "quantization. Check --image-min-tokens/--image-max-tokens and "
                "the two mmproj files' vision metadata. Pass "
                "--allow-n-past-drift only if the drift IS the measurement.\n",
                (int) ref.n_past, (int) cand.n_past);
            return false;
        }
        prefixed_fprintf(lp,
            "WARNING: prefill drift ref n_past=%d != cand n_past=%d, allowed "
            "by --allow-n-past-drift; the header records the REFERENCE side's "
            "count\n", (int) ref.n_past, (int) cand.n_past);
    }

    return kld_teacher_force_and_write(args, ref, cand, n_vocab, n_batch, n_eval,
                                       tokens_full, lp, write_vlmk_file);
}

static int score_manifest(
        const vlm_kld_args & args,
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
        mtmd_helper_log_set(prefixed_log_callback, &prefix);

        vlm_kld_args row_args = args;
        row_args.image_paths          = entry.image_paths;
        row_args.formatted_chat_path  = entry.formatted_chat_path;
        row_args.tokens_in_path       = entry.tokens_in_path;
        row_args.n_prefill            = entry.n_prefill;
        row_args.add_special          = entry.add_special || args.add_special;
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
        mtmd_helper_log_set(nullptr, nullptr);

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

    // Same advisory as vlm-score: the teacher-forced region must be pure text.
    fprintf(stderr, "note: answer-region scoring assumes a pure-text continuation; "
                    "vision tokens after n_prefill (e.g. multi-turn with images in "
                    "later turns) are unsupported and would yield incorrect metrics.\n");

    std::vector<manifest_entry> manifest_entries;
    if (!args.manifest_path.empty() && !read_manifest(args.manifest_path, manifest_entries)) {
        return 1;
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
    if (!load_side(ref,  "ref",  args.ref_model_path,  args.ref_mmproj_path,  args) ||
        !load_side(cand, "cand", args.cand_model_path, args.cand_mmproj_path, args)) {
        return 1;
    }

    // Both distributions must use the same token IDs.
    const llama_vocab * ref_vocab  = llama_model_get_vocab(ref.model.get());
    const llama_vocab * cand_vocab = llama_model_get_vocab(cand.model.get());
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
