#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

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

struct reference_context {
    common_params params;
    common_init_result_ptr loaded;
    mtmd::context_ptr vision;
    common_chat_templates_ptr templates;
    json metadata;

    explicit reference_context(common_params p) : params(std::move(p)), loaded(common_init_from_params(params)) {
        require(loaded && loaded->model() && loaded->context(), "failed to load model/context");
        require(!params.sampling.backend_sampling, "backend sampling is unsupported: raw logits must be available");
        require(params.sampling.reasoning_budget_tokens == -1, "reasoning budgets are unsupported; use a generation token cap");
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
            {"schema_version", "skymizer-reference-v1"}, {"generation_engine", "llama.cpp"},
            {"producer", "llama-reference"}, {"build_info", llama_build_info()},
            {"model_path", fs::absolute(params.model.path).string()},
            {"mmproj_path", params.mmproj.path.empty() ? "" : fs::absolute(params.mmproj.path).string()},
            {"modalities", {{"text", true}, {"vision", bool(vision)}}},
            {"n_ctx", llama_n_ctx(ctx)}, {"n_ctx_per_seq", llama_n_ctx_seq(ctx)},
            {"n_batch", llama_n_batch(ctx)}, {"n_ubatch", llama_n_ubatch(ctx)},
            {"n_threads", llama_n_threads(ctx)}, {"n_threads_batch", llama_n_threads_batch(ctx)},
            {"total_slots", 1}, {"n_predict", params.n_predict},
            {"image_min_tokens", params.image_min_tokens}, {"image_max_tokens", params.image_max_tokens},
            {"image_token_budget_source", "mtmd_init_params"},
            {"media_marker", mtmd_default_marker()}, {"image_placeholder_id", LLAMA_TOKEN_NULL},
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
    }

    json generate(const json & request) {
        auto * ctx = loaded->context();
        auto * vocab = llama_model_get_vocab(loaded->model());
        llama_memory_clear(llama_get_memory(ctx), true);
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
        std::vector<llama_token> tokens;
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
        llama_pos evaluated = 0;
        if (vision) {
            require(mtmd_helper_eval_chunks(vision.get(), ctx, chunks.ptr.get(), 0, 0, llama_n_batch(ctx), true, &evaluated) == 0,
                    "mtmd prefill failed");
        } else {
            for (size_t i = 0; i < tokens.size(); i += llama_n_batch(ctx)) {
                int32_t n = std::min(size_t(llama_n_batch(ctx)), tokens.size() - i);
                auto batch = llama_batch_get_one(tokens.data() + i, n);
                require(llama_decode(ctx, batch) == 0, "text prefill failed");
                evaluated += n;
            }
        }
        require(evaluated == n_pos, "evaluated prompt positions differ from tokenized layout");
        const size_t n_prefill = tokens.size();
        auto sampling = params.sampling;
        if (request.contains("seed")) {
            const int64_t seed = request.at("seed").get<int64_t>();
            require(seed >= 0 && seed < UINT32_MAX, "seed must be between 0 and UINT32_MAX-1");
            sampling.seed = seed;
        }
        sampler_ptr sampler(common_sampler_init(loaded->model(), sampling), common_sampler_free);
        require(bool(sampler), "failed to initialize row sampler");
        for (auto token : tokens) {
            if (token != LLAMA_TOKEN_NULL) {
                common_sampler_accept(sampler.get(), token, false);
            }
        }
        json row_sampling = sampling_json(sampling, sampler.get());
        std::vector<double> logprobs;
        std::string content;
        std::string finish = "length";
        std::string stop_type = "limit";
        std::string stopping_word;
        llama_batch batch = llama_batch_init(1, 0, 1);
        try {
            for (int i = 0; i < params.n_predict; ++i) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                const int n_vocab = llama_vocab_n_tokens(vocab);
                std::vector<float> raw(logits, logits + n_vocab);
                double max_logit = *std::max_element(raw.begin(), raw.end());
                double sum = 0;
                for (float value : raw) {
                    sum += std::exp(double(value) - max_logit);
                }
                require(std::isfinite(max_logit) && sum > 0 && std::isfinite(sum), "non-finite model logits");
                llama_token token = common_sampler_sample(sampler.get(), ctx, -1);
                require(token >= 0 && token < n_vocab, "sampler returned an invalid token");
                double logprob = double(raw[token]) - max_logit - std::log(sum);
                require(std::isfinite(logprob), "non-finite sampled-token logprob");
                tokens.push_back(token);
                logprobs.push_back(logprob);
                common_sampler_accept(sampler.get(), token, true);
                if (llama_vocab_is_eog(vocab, token) && !sampling.ignore_eos) {
                    finish = "stop";
                    stop_type = "eos";
                    break;
                }
                content += common_token_to_piece(ctx, token, true);
                for (const auto & stop : chat.additional_stops) {
                    if (!stop.empty() && content.size() >= stop.size() && content.compare(content.size() - stop.size(), stop.size(), stop) == 0) {
                        stopping_word = stop;
                        content.resize(content.size() - stop.size());
                        finish = "stop";
                        stop_type = "word";
                        break;
                    }
                }
                if (finish == "stop" || i + 1 == params.n_predict) {
                    break;
                }
                common_batch_clear(batch);
                common_batch_add(batch, token, evaluated++, {0}, true);
                require(llama_decode(ctx, batch) == 0, "generation decode failed");
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
        return {
            {"id", request.at("id")}, {"prompt", prompt}, {"input_ids", tokens},
            {"n_prefill_tokens", n_prefill}, {"n_past_prefill", n_pos},
            {"prompt_layout", {{"n_tokens", n_prefill}, {"n_pos", n_pos}, {"chunks", layout_chunks}}},
            {"content", content}, {"finish_reason", finish}, {"stop_type", stop_type}, {"stopping_word", stopping_word},
            {"token_logprobs", logprobs}, {"sampling", row_sampling}, {"add_special", true},
            {"stripped_leading_bos", stripped_bos}, {"enable_thinking", input.enable_thinking},
            {"chat_template_kwargs", input.chat_template_kwargs},
        };
    }
};

static void usage(int, char ** argv) {
    fprintf(stderr, "\n%s -m MODEL [--mmproj PROJECTOR] --requests rows.jsonl --out-dir DIR -n TOKENS [llama.cpp options]\n"
                    "  --describe: load the model and print effective metadata, without generating\n"
                    "  rows: {id, question, images: [local paths], enable_thinking?: bool, seed?: int, system_prompt?: str}\n"
                    "  output: metadata.json and generations.jsonl; text/image support, one model load, no HTTP or HF tokenizer\n", argv[0]);
}

int main(int argc, char ** argv) {
    try {
        common_params params;
        params.n_predict = 128;
        params.n_ctx = 8192;
        params.use_jinja = true;
        std::string requests_path;
        std::string out_dir;
        bool describe = false;
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
            } else {
                common_args.push_back(argv[i]);
            }
        }
        common_init();
        require(common_params_parse(common_args.size(), common_args.data(), params, LLAMA_EXAMPLE_CLI, usage), "invalid arguments");
        require(!params.model.path.empty() && fs::is_regular_file(params.model.path), "provide an existing local GGUF with -m");
        require(params.model.hf_repo.empty() && params.model.url.empty() && params.mmproj.hf_repo.empty() && params.mmproj.url.empty(),
                "reference generation requires local GGUF files");
        require(params.n_parallel == 1, "reference generation requires one sequence");
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
        reference_context context(std::move(params));
        context.metadata["command"] = command;
        if (describe) {
            std::cout << context.metadata.dump(2) << '\n';
            return 0;
        }
        fs::create_directories(out_dir);
        std::ofstream metadata(fs::path(out_dir) / "metadata.json");
        metadata << context.metadata.dump(2) << '\n';
        require(bool(metadata), "failed to write metadata");
        std::ofstream output(fs::path(out_dir) / "generations.jsonl");
        std::set<std::string> seen;
        std::string line;
        size_t count = 0;
        while (std::getline(requests, line)) {
            if (line.empty()) {
                continue;
            }
            auto request = json::parse(line);
            auto id = request.at("id").get<std::string>();
            require(!id.empty(), "request id must not be empty");
            require(seen.insert(id).second, "duplicate request id: " + id);
            auto result = context.generate(request);
            output << result.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
            output.flush();
            require(bool(output), "failed to write generation");
            std::cerr << "[reference] row " << ++count << " id=" << id << " tokens=" << result["token_logprobs"].size() << '\n';
        }
        require(count > 0, "requests file is empty");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama-reference: " << error.what() << '\n';
        return 1;
    }
}
