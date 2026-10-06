// Chat-template and tokenizer helpers for derived teacher-forcing corpora.
// Loads only the GGUF vocabulary and chat template (no weights, CPU only).
//
//   llama-trajectory render-prompt     MODEL REQUESTS OUT  native user prompt (+ text chunk tokens)
//   llama-trajectory render-trajectory MODEL REQUESTS OUT  full assistant turn through the native template
//   llama-trajectory tokenize          MODEL REQUESTS OUT  segmented continuation tokens
//
// REQUESTS and OUT are JSON lines; OUT has one line per request, in order.

#include "chat.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static const std::string MEDIA_MARKER = "<__media__>";

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::vector<llama_token> encode(const llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> tokens(n < 0 ? -n : n);
    n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_special, parse_special);
    require(n >= 0, "tokenization failed");
    tokens.resize(n);
    return tokens;
}

static std::string decode(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    for (auto token : tokens) {
        int n = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
        std::string piece(n < 0 ? -n : n, '\0');
        n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        require(n >= 0, "token piece decoding failed");
        piece.resize(n);
        result += piece;
    }
    return result;
}

// The generator stores prompts without the BOS text that the template may add.
static void strip_bos(const llama_vocab * vocab, std::string & prompt) {
    if (!llama_vocab_get_add_bos(vocab)) {
        return;
    }
    const std::string bos = decode(vocab, {llama_vocab_bos(vocab)});
    if (!bos.empty() && prompt.rfind(bos, 0) == 0) {
        prompt.erase(0, bos.size());
    }
}

static common_chat_msg user_message(const json & req) {
    common_chat_msg user;
    user.role = "user";
    for (int i = 0; i < req.at("num_images").get<int>(); i++) {
        user.content_parts.push_back({"media_marker", MEDIA_MARKER});
    }
    user.content_parts.push_back({"text", req.at("question").get<std::string>()});
    return user;
}

static void add_system_message(const json & req, common_chat_templates_inputs & args) {
    if (req.contains("system_prompt") && !req["system_prompt"].is_null()) {
        common_chat_msg system;
        system.role = "system";
        system.content = req["system_prompt"].get<std::string>();
        args.messages.push_back(system);
    }
}

static json render_prompt(const llama_vocab * vocab, const common_chat_templates * templates, const json & req) {
    common_chat_templates_inputs args;
    args.use_jinja = true;
    args.enable_thinking = req.at("enable_thinking").get<bool>();
    args.chat_template_kwargs = req.at("chat_template_kwargs").get<std::map<std::string, std::string>>();
    args.chat_template_kwargs["enable_thinking"] = req.at("enable_thinking").dump();
    add_system_message(req, args);
    args.messages.push_back(user_message(req));
    auto prompt = common_chat_templates_apply(templates, args).prompt;
    strip_bos(vocab, prompt);
    // Text between media markers, tokenized as the generator did (BOS only on the first part).
    json parts = json::array();
    size_t start = 0;
    bool first = true;
    while (true) {
        auto pos = prompt.find(MEDIA_MARKER, start);
        auto text = prompt.substr(start, pos == std::string::npos ? pos : pos - start);
        parts.push_back(encode(vocab, text, first && req.at("add_special").get<bool>(), true));
        if (pos == std::string::npos) {
            break;
        }
        start = pos + MEDIA_MARKER.size();
        first = false;
    }
    return {{"id", req.at("id")}, {"prompt", prompt}, {"text_parts_tokens", parts},
            {"template", common_chat_templates_source(templates)}};
}

static json render_trajectory(const llama_vocab * vocab, const common_chat_templates * templates, const json & req) {
    common_chat_templates_inputs args;
    args.use_jinja = true;
    args.enable_thinking = true;
    args.add_generation_prompt = false;
    args.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    args.chat_template_kwargs = req.at("chat_template_kwargs").get<std::map<std::string, std::string>>();
    args.chat_template_kwargs["enable_thinking"] = "true";
    add_system_message(req, args);
    args.messages.push_back(user_message(req));
    common_chat_msg assistant;
    assistant.role = "assistant";
    assistant.reasoning_content = req.at("reasoning").get<std::string>();
    assistant.content = req.at("answer").get<std::string>();
    args.messages.push_back(assistant);
    auto rendered = common_chat_templates_apply(templates, args).prompt;
    strip_bos(vocab, rendered);
    const auto prefix = req.at("native_prompt").get<std::string>();
    require(rendered.compare(0, prefix.size(), prefix) == 0,
            "completed template changed native prefix: " + req.at("id").get<std::string>());
    const auto suffix = rendered.substr(prefix.size());
    return {{"id", req.at("id")}, {"rendered", rendered}, {"suffix", suffix},
            {"suffix_tokens", encode(vocab, suffix, false, true)}, {"template", common_chat_templates_source(templates)}};
}

// Encode content with the literal Qwen "</think>" split into ordinary pieces. Only reviewed rows use this.
static std::vector<llama_token> encode_literal_content(const llama_vocab * vocab, const std::string & text) {
    static const std::string marker = "</think>";
    std::vector<llama_token> result;
    size_t begin = 0;
    while (true) {
        size_t at = text.find(marker, begin);
        size_t end = at == std::string::npos ? text.size() : at + 1;
        auto part = encode(vocab, text.substr(begin, end - begin), false, false);
        result.insert(result.end(), part.begin(), part.end());
        if (at == std::string::npos) {
            break;
        }
        begin = end;
    }
    for (auto token : result) {
        auto piece = decode(vocab, {token});
        require(piece != "<think>" && piece != "</think>" && piece != "<|im_end|>" && piece != "<|im_start|>",
                "structural Qwen token in literal content");
    }
    return result;
}

static json tokenize(const llama_vocab * vocab, const json & req) {
    const bool literal = req.value("allow_literal_special_text", false);
    if (literal) {
        require(llama_vocab_n_tokens(vocab) > 248069 && decode(vocab, {248069}) == "</think>",
                "literal exception requires a verified Qwen vocabulary");
    }
    std::vector<llama_token> tokens;
    json segments = json::array();
    std::string text;
    for (const auto & segment : req.at("segments")) {
        auto content = segment.at("text").get<std::string>();
        bool special = segment.at("parse_special").get<bool>();
        auto part = special || !literal ? encode(vocab, content, false, special) : encode_literal_content(vocab, content);
        require(decode(vocab, part) == content, "segment roundtrip changed bytes");
        tokens.insert(tokens.end(), part.begin(), part.end());
        text += content;
        segments.push_back({{"text", content}, {"parse_special", special}, {"tokens", part}});
    }
    // Without the literal exception, segmented and whole-text tokenization must agree.
    const bool canonical = encode(vocab, text, false, true) == tokens;
    require(canonical != literal, "unexpected canonical tokenization result: " + req.at("id").get<std::string>());
    require(decode(vocab, tokens) == text, "tokenizer roundtrip changed trajectory bytes");
    return {{"id", req.at("id")}, {"tokens", tokens}, {"segments", segments}, {"roundtrip_exact", true},
            {"canonical_tokenization_exact", canonical}, {"vocab_size", llama_vocab_n_tokens(vocab)}};
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: llama-trajectory render-prompt|render-trajectory|tokenize MODEL REQUESTS OUT");
        const std::string command = argv[1];
        require(command == "render-prompt" || command == "render-trajectory" || command == "tokenize",
                "unknown command: " + command);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[2], params);
        require(model != nullptr, "model vocabulary load failed");
        const auto * vocab = llama_model_get_vocab(model);
        common_chat_templates_ptr templates;
        if (command != "tokenize") {
            templates = common_chat_templates_init(model, "");
        }
        std::ifstream input(argv[3]);
        std::ofstream output(argv[4]);
        require(input && output, "cannot open input/output");
        std::string line;
        while (std::getline(input, line)) {
            const auto req = json::parse(line);
            json result;
            if (command == "render-prompt") {
                result = render_prompt(vocab, templates.get(), req);
            } else if (command == "render-trajectory") {
                result = render_trajectory(vocab, templates.get(), req);
            } else {
                result = tokenize(vocab, req);
            }
            output << result.dump() << '\n';
        }
        output.flush();
        require(bool(output), "output write failed");
        templates.reset();
        llama_model_free(model);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
