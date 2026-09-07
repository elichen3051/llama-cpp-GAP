#pragma once

#include "llama.h"
#include "ggml-backend.h"
#ifdef __linux__
#include <link.h>
#endif
#include "build-info.h"
#include "hash/hash.h"
#include "nlohmann/json.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace skymizer_identity {
using json = nlohmann::ordered_json;

inline void integer(std::string & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(char((value >> (8 * i)) & 255));
    }
}

inline json vocabulary(const llama_vocab * vocab) {
    std::string mapping, attributes;
    const int n = llama_vocab_n_tokens(vocab);
    integer(mapping, llama_vocab_type(vocab));
    integer(mapping, n);
    for (llama_token id = 0; id < n; ++id) {
        const char * text = llama_vocab_get_text(vocab, id);
        if (!text) {
            throw std::runtime_error("vocabulary contains a null token text");
        }
        const size_t len = std::strlen(text);
        integer(mapping, len);
        mapping.append(text, len);
        integer(attributes, llama_vocab_get_attr(vocab, id));
    }
    return {{"scheme", "llama-vocabulary-sha256-v1"}, {"size", n},
            {"type", int(llama_vocab_type(vocab))},
            {"mapping", hash_sha256_hex(mapping.data(), mapping.size())},
            {"attributes", hash_sha256_hex(attributes.data(), attributes.size())}};
}

inline json from_file(const std::string & path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), params), llama_model_free);
    if (!model) {
        throw std::runtime_error("cannot read GGUF vocabulary: " + path);
    }
    return vocabulary(llama_model_get_vocab(model.get()));
}

inline void check(const json & expected, const json & actual, bool allow_attributes = false) {
    for (const char * key : {"scheme", "size", "type", "mapping", "attributes"}) {
        if (allow_attributes && std::string(key) == "attributes") {
            continue;
        }
        if (!expected.is_object() || !expected.contains(key) || expected.at(key) != actual.at(key)) {
            throw std::runtime_error(std::string("reference dataset vocabulary mismatch: ") + key);
        }
    }
}

inline void check_manifest(const std::vector<json> & expected, const std::string & ref,
                           const std::string & cand, bool allow_attributes) {
    if (expected.empty()) {
        return;
    }
    const auto ref_vocab = from_file(ref);
    const auto cand_vocab = from_file(cand);
    for (const auto & entry : expected) {
        check(entry, ref_vocab);
        check(entry, cand_vocab, allow_attributes);
    }
}

inline int command(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--build-info") {
            ggml_backend_load_all();
            json libraries = json::array();
#ifdef __linux__
            dl_iterate_phdr([](struct dl_phdr_info * info, size_t, void * output) {
                if (info->dlpi_name && info->dlpi_name[0]) {
                    static_cast<json *>(output)->push_back(info->dlpi_name);
                }
                return 0;
            }, &libraries);
#endif
            std::cout << json({{"build", llama_build_info()}, {"loaded_libraries", libraries},
                               {"contracts", {"reference-vocabulary-v1"}}}).dump() << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--vocab-identity") {
            std::cout << from_file(argv[2]).dump() << '\n';
            return 0;
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return -1;
}
}
