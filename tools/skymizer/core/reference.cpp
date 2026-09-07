#include "arg.h"
#include "skymizer-identity.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "skymizer-repetition.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static json sampling_json(const common_params_sampling & p, const common_sampler * sampler) {
    json names = json::array();
    for (auto type : p.samplers) {
        names.push_back(common_sampler_type_to_str(type));
    }
    json biases = json::array();
    for (const auto & b : p.logit_bias) {
        biases.push_back({{"token", b.token}, {"bias", std::isfinite(b.bias) ? json(b.bias) : json(b.bias < 0 ? "-inf" : "inf")}});
    }
    return {
        {"seed", common_sampler_get_seed(sampler)}, {"temperature", p.temp},
        {"top_k", p.top_k}, {"top_p", p.top_p}, {"min_p", p.min_p}, {"typical_p", p.typ_p},
        {"min_keep", p.min_keep}, {"n_prev", p.n_prev}, {"samplers", names},
        {"sampler_chain", common_sampler_print(sampler)},
        {"repeat_last_n", p.penalty_last_n}, {"repeat_penalty", p.penalty_repeat},
        {"frequency_penalty", p.penalty_freq}, {"presence_penalty", p.penalty_present},
        {"penalize_prompt", true}, {"dry_multiplier", p.dry_multiplier}, {"dry_base", p.dry_base},
        {"dry_allowed_length", p.dry_allowed_length}, {"dry_penalty_last_n", p.dry_penalty_last_n},
        {"dry_sequence_breakers", p.dry_sequence_breakers},
        {"xtc_probability", p.xtc_probability}, {"xtc_threshold", p.xtc_threshold},
        {"dynatemp_range", p.dynatemp_range}, {"dynatemp_exponent", p.dynatemp_exponent},
        {"top_n_sigma", p.top_n_sigma}, {"mirostat", p.mirostat},
        {"mirostat_tau", p.mirostat_tau}, {"mirostat_eta", p.mirostat_eta},
        {"adaptive_target", p.adaptive_target}, {"adaptive_decay", p.adaptive_decay},
        {"ignore_eos", p.ignore_eos}, {"logit_bias", biases},
        {"grammar", p.grammar.grammar}, {"grammar_lazy", p.grammar_lazy},
        {"backend_sampling", p.backend_sampling}, {"user_sampling_config", p.user_sampling_config},
    };
}

using sampler_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

struct reference_row {
    llama_seq_id sequence = 0;
    llama_pos evaluated = 0;
    size_t n_prefill = 0;
    llama_tokens tokens;
    llama_tokens processed_tokens;
    common_params_sampling sampling;
    sampler_ptr sampler{nullptr, common_sampler_free};
    common_speculative_ptr spec;
    std::vector<double> logprobs;
    std::vector<std::string> stops;
    std::string content;
    std::string finish = "length";
    std::string stop_type = "limit";
    std::string stopping_word;
    json repetition = nullptr;
    json result;
    json stats = {{"drafted_tokens", 0}, {"accepted_draft_tokens", 0}, {"verification_batches", 0},
                  {"rollback_batches", 0}, {"checkpoint_replays", 0}};
};

struct reference_context {
    common_params params;
    common_init_result_ptr loaded;
    common_speculative_init_result_ptr draft_loaded;
    bool repetition_stop = true;
    bool checkpoint_target = false;
    bool checkpoint_draft = false;
    mtmd::context_ptr vision;
    common_chat_templates_ptr templates;
    json metadata;

    explicit reference_context(common_params p, bool stop_repetition) : params(std::move(p)), loaded(common_init_from_params(params)), repetition_stop(stop_repetition) {
        require(loaded && loaded->model() && loaded->context(), "failed to load model/context");
        require(!params.sampling.backend_sampling, "backend sampling is unsupported: raw logits must be available");
        require(params.sampling.reasoning_budget_tokens == -1, "reasoning budgets are unsupported; use a generation token cap");
        if (params.speculative.types == std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP}) {
            if (!params.speculative.has_dft()) {
                require(llama_model_n_layer_nextn(loaded->model()) > 0, "target GGUF has no embedded MTP head; supply a matching local --model-draft");
            }
            if (params.speculative.has_dft()) {
                auto header_params = llama_model_default_params();
                header_params.vocab_only = true;
                llama_model_ptr head(llama_model_load_from_file(params.speculative.draft.mparams.path.c_str(), header_params));
                require(bool(head), "cannot read MTP sidecar metadata");
                skymizer_identity::check(skymizer_identity::vocabulary(llama_model_get_vocab(loaded->model())),
                                        skymizer_identity::vocabulary(llama_model_get_vocab(head.get())), true);
            }
            auto target_rm = common_context_can_seq_rm(loaded->context());
            require(target_rm != COMMON_CONTEXT_SEQ_RM_TYPE_NO, "target context does not support MTP state management");
            checkpoint_target = target_rm != COMMON_CONTEXT_SEQ_RM_TYPE_PART;
            auto draft_params = common_base_params_to_speculative(params);
            draft_loaded = common_speculative_init_from_params(draft_params, loaded->model(), loaded->context());
            require(draft_loaded && draft_loaded->context(), "failed to initialize MTP head/context");
            auto * draft_model = llama_get_model(draft_loaded->context());
            require(llama_model_n_layer_nextn(draft_model) > 0, "draft GGUF has no MTP layers");
            require(llama_model_n_embd_out(draft_model) == llama_model_n_embd_out(loaded->model()), "MTP head hidden width differs from target");
            params.speculative.draft.ctx_tgt = loaded->context();
            params.speculative.draft.ctx_dft = draft_loaded->context();
            if (llama_get_memory(draft_loaded->context())) {
                checkpoint_draft = common_context_can_seq_rm(draft_loaded->context()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
            }
        }
        templates = common_chat_templates_init(loaded->model(), params.chat_template);
        require(llama_model_chat_template(loaded->model(), nullptr) || !params.chat_template.empty(),
                "model has no chat template; supply --chat-template or --chat-template-file");
        if (!params.mmproj.path.empty()) {
            auto vp = mtmd_context_params_default();
            vp.use_gpu = params.mmproj_use_gpu;
            vp.device = params.mmproj_device;
            vp.n_threads = params.cpuparams.n_threads;
            vp.flash_attn_type = params.flash_attn_type;
            vp.warmup = params.warmup;
            vp.image_min_tokens = params.image_min_tokens;
            vp.image_max_tokens = params.image_max_tokens;
            vp.media_marker = mtmd_default_marker();
            vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), loaded->model(), vp));
            require(vision && mtmd_support_vision(vision.get()), "mmproj does not support image input");
        }
        sampler_ptr sampler(common_sampler_init(loaded->model(), params.sampling), common_sampler_free);
        require(bool(sampler), "failed to initialize sampler");
        params.sampling.seed = common_sampler_get_seed(sampler.get());
        auto * ctx = loaded->context();
        auto * vocab = llama_model_get_vocab(loaded->model());
        json model_metadata = json::object();
        for (int32_t i = 0; i < llama_model_meta_count(loaded->model()); ++i) {
            char key[512];
            int len = llama_model_meta_val_str_by_index(loaded->model(), i, nullptr, 0);
            if (len < 0) {
                continue;
            }
            std::vector<char> value(size_t(len) + 1);
            llama_model_meta_key_by_index(loaded->model(), i, key, sizeof(key));
            llama_model_meta_val_str_by_index(loaded->model(), i, value.data(), value.size());
            model_metadata[key] = value.data();
        }
        bool default_thinking = params.enable_reasoning != 0;
        auto thinking = params.default_template_kwargs.find("enable_thinking");
        if (thinking != params.default_template_kwargs.end()) {
            require(thinking->second == "true" || thinking->second == "false", "enable_thinking must be boolean");
            default_thinking = thinking->second == "true";
        }
        metadata = {
            {"schema_version", "skymizer-reference-v2"}, {"generation_engine", "llama.cpp"},
            {"producer", "llama-reference"}, {"build_info", llama_build_info()},
            {"decoding", {{"method", "autoregressive"}, {"logprob_source", "target_raw_logits"}, {"token_source", "target_accepted"}}},
            {"model_path", fs::absolute(params.model.path).string()},
            {"mmproj_path", params.mmproj.path.empty() ? "" : fs::absolute(params.mmproj.path).string()},
            {"modalities", {{"text", true}, {"vision", bool(vision)}}},
            {"n_ctx", llama_n_ctx(ctx)}, {"n_ctx_per_seq", llama_n_ctx_seq(ctx)},
            {"n_batch", llama_n_batch(ctx)}, {"n_ubatch", llama_n_ubatch(ctx)},
            {"n_threads", llama_n_threads(ctx)}, {"n_threads_batch", llama_n_threads_batch(ctx)},
            {"total_slots", params.n_parallel}, {"n_predict", params.n_predict},
            {"image_min_tokens", params.image_min_tokens}, {"image_max_tokens", params.image_max_tokens},
            {"image_token_budget_source", "mtmd_init_params"},
            {"media_marker", mtmd_default_marker()}, {"image_placeholder_id", LLAMA_TOKEN_NULL},
            {"vocabulary", skymizer_identity::vocabulary(vocab)},
            {"vocab_size", llama_vocab_n_tokens(vocab)}, {"bos_token_id", llama_vocab_bos(vocab)},
            {"eos_token_id", llama_vocab_eos(vocab)}, {"eot_token_id", llama_vocab_eot(vocab)},
            {"add_bos", llama_vocab_get_add_bos(vocab)},
            {"chat_template_source", params.chat_template.empty() ? "gguf" : "cli_override"},
            {"chat_template", common_chat_templates_source(templates.get())},
            {"chat_template_kwargs", params.default_template_kwargs}, {"jinja", params.use_jinja},
            {"system_prompt", params.system_prompt},
            {"supports_enable_thinking", common_chat_templates_support_enable_thinking(templates.get())},
            {"default_enable_thinking", default_thinking},
            {"sampling", sampling_json(params.sampling, sampler.get())},
            {"model_metadata", model_metadata},
        };
        if (draft_loaded) {
            const auto & d = params.speculative.draft;
            metadata["decoding"] = {
                {"method", "mtp"}, {"logprob_source", "target_raw_logits"}, {"token_source", "target_accepted"},
                {"mtp", {{"head_source", params.speculative.has_dft() ? "sidecar" : "embedded"},
                         {"head_path", params.speculative.has_dft() ? fs::absolute(d.mparams.path).string() : metadata["model_path"].get<std::string>()},
                         {"settings", {{"n_max", d.n_max}, {"n_min", d.n_min}, {"p_min", d.p_min},
                                       {"backend_sampling", d.backend_sampling}, {"n_gpu_layers", d.n_gpu_layers},
                                       {"cache_type_k", ggml_type_name(d.cache_type_k)}, {"cache_type_v", ggml_type_name(d.cache_type_v)},
                                       {"n_ctx", llama_n_ctx(draft_loaded->context())}, {"acceptance", "target_sample_match"}}}}},
            };
        }
    }

    std::unique_ptr<reference_row> prepare(const json & request, llama_seq_id sequence) {
        auto row = std::make_unique<reference_row>();
        row->sequence = sequence;
        auto & tokens = row->tokens;
        auto & evaluated = row->evaluated;
        auto & spec = row->spec;
        auto & sampler = row->sampler;
        auto & sampling = row->sampling;
        auto * ctx = loaded->context();
        auto * vocab = llama_model_get_vocab(loaded->model());
        if (params.n_parallel == 1) {
            llama_memory_clear(llama_get_memory(ctx), true);
        } else {
            require(llama_memory_seq_rm(llama_get_memory(ctx), sequence, -1, -1), "cannot clear reference sequence");
        }
        if (draft_loaded) {
            llama_memory_clear(llama_get_memory(draft_loaded->context()), true);
            spec.reset(common_speculative_init(params.speculative, 1));
            require(bool(spec), "failed to initialize MTP driver");
        }
        const std::string question = request.at("question").get<std::string>();
        const auto images = request.value("images", std::vector<std::string>{});
        const std::string marker = mtmd_default_marker();
        require(question.find(marker) == std::string::npos, "question contains reserved media marker");
        require(images.empty() || bool(vision), "images require --mmproj");

        common_chat_templates_inputs input;
        input.use_jinja = params.use_jinja;
        input.enable_thinking = params.enable_reasoning != 0;
        input.chat_template_kwargs = params.default_template_kwargs;
        if (request.contains("enable_thinking") && !request["enable_thinking"].is_null()) {
            require(request["enable_thinking"].is_boolean(), "enable_thinking must be boolean");
            input.chat_template_kwargs["enable_thinking"] = request["enable_thinking"].dump();
        }
        auto thinking = input.chat_template_kwargs.find("enable_thinking");
        if (thinking != input.chat_template_kwargs.end()) {
            require(thinking->second == "true" || thinking->second == "false", "enable_thinking must be boolean");
            input.enable_thinking = thinking->second == "true";
        }
        if (request.contains("system_prompt") || !params.system_prompt.empty()) {
            common_chat_msg system;
            system.role = "system";
            system.content = request.value("system_prompt", params.system_prompt);
            require(system.content.find(marker) == std::string::npos, "system prompt contains reserved media marker");
            input.messages.push_back(std::move(system));
        }
        common_chat_msg user;
        user.role = "user";
        for (size_t i = 0; i < images.size(); ++i) {
            user.content_parts.push_back({"media_marker", marker});
        }
        user.content_parts.push_back({"text", question});
        input.messages.push_back(std::move(user));
        auto chat = common_chat_templates_apply(templates.get(), input);
        std::string prompt = chat.prompt;
        bool stripped_bos = false;
        if (llama_vocab_get_add_bos(vocab)) {
            auto bos = common_token_to_piece(ctx, llama_vocab_bos(vocab), true);
            if (!bos.empty() && prompt.compare(0, bos.size(), bos) == 0) {
                prompt.erase(0, bos.size());
                stripped_bos = true;
            }
        }
        size_t markers = 0;
        for (size_t pos = 0; (pos = prompt.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            ++markers;
        }
        require(markers == images.size(), "template changed the number of media markers");

        mtmd::bitmaps bitmaps;
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        json layout_chunks = json::array();
        llama_pos n_pos = 0;
        if (vision) {
            for (const auto & path : images) {
                auto image = mtmd_helper_bitmap_init_from_file(vision.get(), path.c_str(), false, mtmd_helper_init_opt_default());
                require(image.bitmap != nullptr && image.video_ctx == nullptr, "cannot read image: " + path);
                bitmaps.entries.emplace_back(image.bitmap);
            }
            auto ptrs = bitmaps.c_ptr();
            mtmd_input_text text{prompt.data(), prompt.size(), true, true};
            require(mtmd_tokenize(vision.get(), chunks.ptr.get(), &text, ptrs.data(), ptrs.size()) == 0,
                    "mtmd tokenization failed");
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
                const auto * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
                size_t n = mtmd_input_chunk_get_n_tokens(c);
                json item = {{"start", tokens.size()}, {"n_tokens", n}, {"n_pos", mtmd_input_chunk_get_n_pos(c)}};
                if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    const auto * ids = mtmd_input_chunk_get_tokens_text(c, &n);
                    std::vector<llama_token> text_ids(ids, ids + n);
                    item["type"] = "text";
                    item["tokens"] = text_ids;
                    tokens.insert(tokens.end(), text_ids.begin(), text_ids.end());
                } else {
                    require(mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE, "only text and images are supported");
                    if (mtmd_decode_use_non_causal(vision.get(), c)) {
                        require(n <= llama_n_batch(ctx) && n <= llama_n_ubatch(ctx),
                                "non-causal image exceeds batch/ubatch capacity: need at least " + std::to_string(n) +
                                " tokens in both -b and -ub");
                    }
                    item["type"] = "image";
                    const auto * img = mtmd_input_chunk_get_tokens_image(c);
                    auto pos = mtmd_image_tokens_get_decoder_pos(img, 0, n - 1);
                    item["grid_x"] = pos.x + 1;
                    item["grid_y"] = pos.y + 1;
                    item["grid_t"] = pos.t + 1;
                    tokens.insert(tokens.end(), n, LLAMA_TOKEN_NULL);
                }
                layout_chunks.push_back(std::move(item));
            }
            n_pos = mtmd_helper_get_n_pos(chunks.ptr.get());
        } else {
            tokens = common_tokenize(ctx, prompt, true, true);
            n_pos = tokens.size();
            layout_chunks.push_back({{"type", "text"}, {"start", 0}, {"n_tokens", tokens.size()},
                                     {"n_pos", n_pos}, {"tokens", tokens}});
        }
        require(!tokens.empty(), "template produced an empty prompt");
        require(std::max(size_t(n_pos), tokens.size()) + params.n_predict <= llama_n_ctx_seq(ctx),
                "prompt plus generation cap exceeds context; no truncation or context shift is allowed");
        auto prefill_text = [&](const llama_token * ids, size_t count) {
            const int capacity = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
            llama_batch part = llama_batch_init(capacity, 0, 1);
            try {
                for (size_t start = 0; start < count; start += capacity) {
                    common_batch_clear(part);
                    const size_t n = std::min(size_t(capacity), count - start);
                    for (size_t i = 0; i < n; ++i) {
                        common_batch_add(part, ids[start + i], evaluated + i, {sequence}, i + 1 == n);
                    }
                    require(llama_decode(ctx, part) == 0, "MTP text prefill failed");
                    require(common_speculative_process(spec.get(), part), "MTP prompt processing failed");
                    evaluated += n;
                }
            } catch (...) {
                llama_batch_free(part);
                throw;
            }
            llama_batch_free(part);
        };
        if (spec && vision) {
            require(layout_chunks.back()["type"] == "text", "MTP requires a text suffix after the final image");
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
                const auto * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t n;
                    const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &n);
                    prefill_text(ids, n);
                } else {
                    require(mtmd_helper_eval_chunk_single(vision.get(), ctx, chunk, evaluated, sequence, llama_n_batch(ctx), false, &evaluated) == 0,
                            "MTP image prefill failed");
                }
            }
        } else if (spec) {
            prefill_text(tokens.data(), tokens.size());
        } else if (vision) {
            require(mtmd_helper_eval_chunks(vision.get(), ctx, chunks.ptr.get(), 0, sequence, llama_n_batch(ctx), true, &evaluated) == 0,
                    "mtmd prefill failed");
        } else {
            llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);
            try {
                for (size_t i = 0; i < tokens.size(); i += llama_n_batch(ctx)) {
                    int32_t n = std::min(size_t(llama_n_batch(ctx)), tokens.size() - i);
                    common_batch_clear(batch);
                    for (int32_t j = 0; j < n; ++j) {
                        common_batch_add(batch, tokens[i + j], evaluated + j, {sequence}, j + 1 == n);
                    }
                    require(llama_decode(ctx, batch) == 0, "text prefill failed");
                    evaluated += n;
                }
            } catch (...) {
                llama_batch_free(batch);
                throw;
            }
            llama_batch_free(batch);
        }
        require(evaluated == n_pos, "evaluated prompt positions differ from tokenized layout");
        row->n_prefill = tokens.size();
        sampling = params.sampling;
        if (request.contains("seed")) {
            const int64_t seed = request.at("seed").get<int64_t>();
            require(seed >= 0 && seed < UINT32_MAX, "seed must be between 0 and UINT32_MAX-1");
            sampling.seed = seed;
        }
        sampler.reset(common_sampler_init(loaded->model(), sampling));
        require(bool(sampler), "failed to initialize row sampler");
        for (auto token : tokens) {
            if (token != LLAMA_TOKEN_NULL) {
                common_sampler_accept(sampler.get(), token, false);
            }
        }
        row->stops = chat.additional_stops;
        for (auto token : tokens) {
            if (token != LLAMA_TOKEN_NULL) {
                row->processed_tokens.push_back(token);
            }
        }
        if (spec) {
            common_speculative_begin(spec.get(), sequence, row->processed_tokens);
        }
        row->result = {
            {"id", request.at("id")}, {"prompt", prompt},
            {"n_prefill_tokens", row->n_prefill}, {"n_past_prefill", n_pos},
            {"prompt_layout", {{"n_tokens", row->n_prefill}, {"n_pos", n_pos}, {"chunks", layout_chunks}}},
            {"sampling", sampling_json(sampling, sampler.get())}, {"add_special", true},
            {"stripped_leading_bos", stripped_bos}, {"enable_thinking", input.enable_thinking},
            {"chat_template_kwargs", input.chat_template_kwargs},
        };
        return row;
    }

    std::pair<llama_token, double> sample(reference_row & row, int index) {
        auto * ctx = loaded->context();
        auto * vocab = llama_model_get_vocab(loaded->model());
        const float * logits = llama_get_logits_ith(ctx, index);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        require(logits != nullptr, "target logits are missing");
        std::vector<float> raw(logits, logits + n_vocab);
        double max_logit = *std::max_element(raw.begin(), raw.end());
        double sum = 0;
        for (float value : raw) {
            sum += std::exp(double(value) - max_logit);
        }
        require(std::isfinite(max_logit) && sum > 0 && std::isfinite(sum), "non-finite model logits");
        llama_token token = common_sampler_sample(row.sampler.get(), ctx, index);
        require(token >= 0 && token < n_vocab, "sampler returned an invalid token");
        double logprob = double(raw[token]) - max_logit - std::log(sum);
        require(std::isfinite(logprob), "non-finite sampled-token logprob");
        return std::make_pair(token, logprob);
    }

    void emit(reference_row & row, llama_token token, double logprob) {
        auto * ctx = loaded->context();
        auto * vocab = llama_model_get_vocab(loaded->model());
        row.tokens.push_back(token);
        row.logprobs.push_back(logprob);
        common_sampler_accept(row.sampler.get(), token, true);
        if (llama_vocab_is_eog(vocab, token) && !row.sampling.ignore_eos) {
            row.finish = "stop";
            row.stop_type = "eos";
            return;
        }
        row.content += common_token_to_piece(ctx, token, true);
        for (const auto & stop : row.stops) {
            if (!stop.empty() && row.content.size() >= stop.size() && row.content.compare(row.content.size() - stop.size(), stop.size(), stop) == 0) {
                row.stopping_word = stop;
                row.content.resize(row.content.size() - stop.size());
                row.finish = "stop";
                row.stop_type = "word";
                break;
            }
        }
        if (repetition_stop && row.logprobs.size() % skymizer_repetition::interval == 0) {
            record_repetition(row, skymizer_repetition::tail(row.tokens.data() + row.n_prefill, row.logprobs.size()), "online");
            if (!row.repetition.is_null()) {
                row.finish = "stop";
                row.stop_type = "repetition";
            }
        }
    }

    void record_repetition(reference_row & row, skymizer_repetition::match match, const char * stage) {
        if (match) {
            row.repetition = {{"start", match.start}, {"unit_len", match.unit_len}, {"repeats", match.repeats},
                              {"repeated_len", match.repeated_len}, {"detected_at_generated_token", row.logprobs.size()}, {"stage", stage}};
        }
    }

    bool finished(const reference_row & row) const {
        return row.finish == "stop" || row.logprobs.size() >= size_t(params.n_predict);
    }

    json finish_row(reference_row & row) {
        if (repetition_stop && row.repetition.is_null()) {
            record_repetition(row, skymizer_repetition::full(row.tokens.data() + row.n_prefill, row.logprobs.size()), "final");
        }
        row.result.update({
            {"input_ids", row.tokens}, {"content", row.content}, {"finish_reason", row.finish},
            {"stop_type", row.stop_type}, {"stopping_word", row.stopping_word},
            {"repetition", row.repetition}, {"token_logprobs", row.logprobs}, {"decoding_stats", row.stats},
        });
        return std::move(row.result);
    }

    void decode(std::vector<std::unique_ptr<reference_row>> & slots) {
        auto * ctx = loaded->context();
        llama_batch batch = llama_batch_init(params.n_parallel, 0, 1);
        try {
            for (const auto & row : slots) {
                if (row) {
                    common_batch_add(batch, row->tokens.back(), row->evaluated, {row->sequence}, true);
                }
            }
            require(batch.n_tokens > 0 && llama_decode(ctx, batch) == 0, "parallel generation decode failed");
            int index = 0;
            for (auto & row : slots) {
                if (row) {
                    auto next = sample(*row, index++);
                    emit(*row, next.first, next.second);
                    ++row->evaluated;
                }
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
    }

    json generate(const json & request) {
        auto prepared = prepare(request, 0);
        auto & row = *prepared;
        auto * ctx = loaded->context();
        auto & tokens = row.tokens;
        auto & evaluated = row.evaluated;
        auto & spec = row.spec;
        auto & processed_tokens = row.processed_tokens;
        auto & logprobs = row.logprobs;
        auto & finish = row.finish;
        auto & stats = row.stats;
        llama_batch batch = llama_batch_init(spec ? std::min(llama_n_batch(ctx), llama_n_ubatch(ctx)) : 1, 0, 1);
        try {
            auto first = sample(row, -1);
            emit(row, first.first, first.second);
            while (finish != "stop" && logprobs.size() < size_t(params.n_predict)) {
                const llama_pos start = evaluated;
                llama_tokens draft;
                common_prompt_checkpoint checkpoint;
                if (spec) {
                    const int limit = std::min({params.speculative.draft.n_max,
                            int(params.n_predict - logprobs.size()) - 1,
                            int(std::min(llama_n_batch(ctx), llama_n_ubatch(ctx))) - 1});
                    if (limit > 0) {
                        if (checkpoint_draft) {
                            checkpoint.update_dft(draft_loaded->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
                        common_speculative_get_draft_params(spec.get(), 0) = {true, limit, start, tokens.back(), &processed_tokens, &draft};
                        common_speculative_draft(spec.get());
                        if (checkpoint_draft) {
                            checkpoint.load_dft(draft_loaded->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
                        require(llama_memory_seq_rm(llama_get_memory(draft_loaded->context()), 0, start, -1), "cannot restore draft prefix");
                    }
                    if (checkpoint_target && !draft.empty()) {
                        checkpoint.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                    stats["drafted_tokens"] = stats["drafted_tokens"].get<int>() + draft.size();
                }
                common_batch_clear(batch);
                common_batch_add(batch, tokens.back(), start, {0}, true);
                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch, draft[i], start + 1 + i, {0}, true);
                }
                require(llama_decode(ctx, batch) == 0, "generation verification decode failed");
                int accepted = 0;
                for (size_t i = 0; i <= draft.size(); ++i) {
                    auto next = sample(row, i);
                    emit(row, next.first, next.second);
                    ++accepted;
                    if (finish == "stop" || logprobs.size() == size_t(params.n_predict) || i == draft.size() || next.first != draft[i]) {
                        break;
                    }
                }
                if (spec) {
                    stats["verification_batches"] = stats["verification_batches"].get<int>() + 1;
                    stats["accepted_draft_tokens"] = stats["accepted_draft_tokens"].get<int>() + accepted - 1;
                    const bool rollback = accepted < batch.n_tokens;
                    if (rollback) {
                        stats["rollback_batches"] = stats["rollback_batches"].get<int>() + 1;
                        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, start + accepted, -1)) {
                            require(checkpoint_target, "target cannot remove rejected draft tokens");
                            checkpoint.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            require(llama_memory_seq_rm(llama_get_memory(ctx), 0, start, -1), "cannot restore target checkpoint");
                            batch.n_tokens = accepted;
                            require(llama_decode(ctx, batch) == 0, "accepted prefix replay failed");
                            stats["checkpoint_replays"] = stats["checkpoint_replays"].get<int>() + 1;
                        }
                    }
                    // Process only the committed prefix; rejected rows never update the MTP carry state.
                    batch.n_tokens = accepted;
                    require(common_speculative_process(spec.get(), batch), "MTP accepted-prefix processing failed");
                    common_speculative_accept(spec.get(), 0, accepted - 1);
                    processed_tokens.insert(processed_tokens.end(), batch.token, batch.token + accepted);
                }
                evaluated += accepted;
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
        return finish_row(row);
    }

};

static void usage(int, char ** argv) {
    fprintf(stderr, "\n%s -m MODEL [--mmproj PROJECTOR] --requests rows.jsonl --out-dir DIR -n TOKENS [llama.cpp options]\n"
                    "  -np N: batch N autoregressive sequences; -c is total context (MTP requires -np 1)\n"
                    "  --describe: load the model and print effective metadata, without generating\n"
                    "  rows: {id, question, images: [local paths], enable_thinking?: bool, seed?: int, system_prompt?: str}\n"
                    "  output: metadata.json and generations.jsonl; text/image support, one model load, no HTTP or HF tokenizer\n", argv[0]);
}

int main(int argc, char ** argv) {
    const int identity_command = skymizer_identity::command(argc, argv);
    if (identity_command >= 0) { return identity_command; }
    try {
        common_params params;
        params.n_predict = 128;
        params.n_ctx = 8192;
        params.use_jinja = true;
        std::string requests_path;
        std::string out_dir;
        bool describe = false;
        bool repetition_stop = true;
        bool continue_on_error = false;
        std::vector<char *> common_args{argv[0]};
        json command = json::array();
        for (int i = 0; i < argc; ++i) {
            command.push_back(argv[i]);
        }
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--requests" || key == "--out-dir") {
                require(i + 1 < argc, "missing value for " + key);
                (key == "--requests" ? requests_path : out_dir) = argv[++i];
            } else if (key == "--describe") {
                describe = true;
            } else if (key == "--repetition-stop" || key == "--no-repetition-stop") {
                repetition_stop = key == "--repetition-stop";
            } else if (key == "--continue-on-error") {
                continue_on_error = true;
            } else {
                common_args.push_back(argv[i]);
            }
        }
        common_init();
        require(common_params_parse(common_args.size(), common_args.data(), params, LLAMA_EXAMPLE_CLI, usage), "invalid arguments");
        require(!params.model.path.empty() && fs::is_regular_file(params.model.path), "provide an existing local GGUF with -m");
        require(params.model.hf_repo.empty() && params.model.url.empty() && params.mmproj.hf_repo.empty() && params.mmproj.url.empty(),
                "reference generation requires local GGUF files");
        const bool mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
        require(std::all_of(params.speculative.types.begin(), params.speculative.types.end(), [](auto type) {
                    return type == COMMON_SPECULATIVE_TYPE_NONE || type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
                }) && (mtp || !params.speculative.has_dft()),
                "reference generation supports only autoregressive or --spec-type draft-mtp");
        params.speculative.types = {mtp ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP : COMMON_SPECULATIVE_TYPE_NONE};
        require(!params.speculative.has_synth(), "synthetic acceptance is not valid for reference generation");
        require(params.lora_adapters.empty(), "LoRA is not supported for native reference generation");
        if (mtp) {
            const auto & d = params.speculative.draft;
            require(d.n_max > 0 && d.n_max < UINT16_MAX && d.n_min >= 0 && d.n_min <= d.n_max, "invalid MTP draft bounds");
            require(std::isfinite(d.p_min) && d.p_min >= 0 && d.p_min <= 1, "invalid MTP confidence threshold");
            require(d.mparams.hf_repo.empty() && d.mparams.url.empty(), "MTP requires local head files");
            require(!params.speculative.has_dft() || fs::is_regular_file(d.mparams.path), "MTP sidecar does not exist");
            require(d.n_max < std::min(params.n_batch, params.n_ubatch), "MTP draft maximum must be smaller than both batch and ubatch sizes");
            const auto limits = common_speculative_get_output_limits(params.n_batch, 1, d.n_max);
            params.n_outputs_max = limits.total;
            params.n_outputs_max_per_seq = limits.per_seq;
        }
        require(params.n_parallel > 0 && params.n_parallel <= std::min(params.n_batch, params.n_ubatch),
                "parallel sequences must fit in batch and ubatch");
        require(!mtp || params.n_parallel == 1, "MTP reference generation requires one sequence");
        require(params.n_predict > 0, "-n must be positive");
        require(params.prompt.empty() && params.image.empty(), "use --requests for prompts and images");
        require(params.antiprompt.empty(), "custom reverse prompts are unsupported; model EOG and template stops are used");
        require(params.n_predict < INT32_MAX, "generation cap is too large");
        require(describe || (!requests_path.empty() && !out_dir.empty()), "provide --requests and --out-dir, or --describe");
        std::ifstream requests;
        if (!describe) {
            requests.open(requests_path);
            require(bool(requests), "cannot read requests file");
            require(!fs::exists(out_dir), "output directory already exists: " + out_dir);
        }
        ggml_backend_load_all();
        mtmd_helper_log_set(common_log_default_callback, nullptr);
        reference_context context(std::move(params), repetition_stop);
        context.metadata["repetition_detector"] = {{"enabled", repetition_stop}, {"version", "exact-token-repeat-v1"},
            {"source_commit", "d4163fc1c328fe39310465680647b465cf96c4af"},
            {"source_sha256", "8c0ede4aa476d7ceea39e57ac4bd3e462569d45683c5672746fee1d65940bda9"},
            {"min_repeated_tokens", skymizer_repetition::min_span}, {"min_repeats", skymizer_repetition::min_repeats},
            {"online_window", skymizer_repetition::window}, {"check_interval", skymizer_repetition::interval},
            {"input", "generated_target_accepted_token_ids"}, {"final_scan", "exact_consecutive_blocks_anywhere"}};
        context.metadata["command"] = command;
        if (describe) {
            std::cout << context.metadata.dump(2) << '\n';
            return 0;
        }
        fs::create_directories(out_dir);
        std::ofstream metadata(fs::path(out_dir) / "metadata.json");
        metadata << context.metadata.dump(2) << '\n';
        metadata.flush();
        require(bool(metadata), "failed to write metadata");
        metadata.close();
        std::ofstream output(fs::path(out_dir) / "generations.jsonl");
        std::ofstream events(fs::path(out_dir) / "events.jsonl");
        auto event = [&](json value) {
            events << value.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
            events.flush();
            require(bool(events), "failed to write row journal");
        };
        auto write_result = [&](json result) {
            auto id = result.at("id").get<std::string>();
            output << result.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
            output.flush();
            require(bool(output), "failed to write generation");
            event({{"event", "row_succeeded"}, {"id", id}});
            std::cerr << "[reference] id=" << id << " tokens=" << result["token_logprobs"].size() << '\n';
        };
        auto row_error = [&](const std::string & id, const std::exception & error) {
            const std::string message = error.what();
            const bool data_error = message.find("exceeds context") != std::string::npos ||
                message.find("non-causal image exceeds") != std::string::npos ||
                message.find("cannot read image") != std::string::npos || message.find("template") != std::string::npos ||
                message.find("question") != std::string::npos;
            event({{"event", "row_failed"}, {"id", id}, {"error", message}, {"retryable", !data_error}});
            std::cerr << "[reference] failed id=" << id << " error=" << message << '\n';
            return continue_on_error && data_error;
        };
        std::vector<std::unique_ptr<reference_row>> slots(context.params.n_parallel);
        std::set<std::string> seen;
        std::string line;
        size_t count = 0;
        bool exhausted = false;
        while (true) {
            for (size_t sequence = 0; sequence < slots.size(); ++sequence) {
                auto & slot = slots[sequence];
                while (!slot && !exhausted) {
                    if (!std::getline(requests, line)) {
                        exhausted = true;
                        break;
                    }
                    if (line.empty()) {
                        continue;
                    }
                    auto request = json::parse(line);
                    auto id = request.at("id").get<std::string>();
                    require(!id.empty(), "request id must not be empty");
                    require(seen.insert(id).second, "duplicate request id: " + id);
                    ++count;
                    event({{"event", "row_started"}, {"id", id}, {"index", count - 1}, {"sequence", sequence}});
                    try {
                        if (slots.size() == 1) {
                            write_result(context.generate(request));
                        } else {
                            slot = context.prepare(request, sequence);
                            auto first = context.sample(*slot, -1);
                            context.emit(*slot, first.first, first.second);
                            if (context.finished(*slot)) {
                                write_result(context.finish_row(*slot));
                                slot.reset();
                            }
                        }
                    } catch (const std::exception & error) {
                        slot.reset();
                        if (!row_error(id, error)) { return 2; }
                    }
                }
            }
            if (std::none_of(slots.begin(), slots.end(), [](const auto & row) { return bool(row); })) {
                break;
            }
            context.decode(slots);
            for (auto & row : slots) {
                if (row && context.finished(*row)) {
                    write_result(context.finish_row(*row));
                    row.reset();
                }
            }
        }
        require(count > 0, "requests file is empty");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama-reference: " << error.what() << '\n';
        return 1;
    }
}
