// Shared scaffolding for the four skymizer C++ tools (llm/vlm x score/kld):
// the stderr prefix logger, the reporting integer parser, and the
// guarded small-file helpers (slurp_file with the f.bad() check,
// read_tokens_bin with the short-read check, validate_tokens_in_vocab). Bodies moved
// VERBATIM from the per-tool copies (they were byte-identical); `static`
// keeps per-TU internal linkage, so codegen is what textual inclusion gave.
#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdarg>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct stderr_prefix {
    std::string text;
    bool at_line_start = true;
};

static void prefixed_write(stderr_prefix * prefix, const char * text) {
    if (prefix == nullptr) {
        fputs(text, stderr);
        return;
    }
    for (const char * p = text; *p != '\0'; ++p) {
        if (prefix->at_line_start) {
            fputs(prefix->text.c_str(), stderr);
            prefix->at_line_start = false;
        }
        fputc(*p, stderr);
        if (*p == '\n') {
            prefix->at_line_start = true;
        }
    }
    fflush(stderr);
}

static void prefixed_fprintf(stderr_prefix * prefix, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (prefix == nullptr) {
        vfprintf(stderr, fmt, args);
        va_end(args);
        return;
    }
    const int n = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    if (n < 0) {
        return;
    }
    std::vector<char> buf((size_t) n + 1);
    va_start(args, fmt);
    vsnprintf(buf.data(), buf.size(), fmt, args);
    va_end(args);
    prefixed_write(prefix, buf.data());
}

static void prefixed_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    prefixed_write(static_cast<stderr_prefix *>(user_data), text);
}

// std::stoi that reports instead of throwing (a typo like "-ngl 9o" should be
// a usage error, not std::terminate).
static bool parse_int_arg(const char * what, const char * v, int & out) {
    try {
        size_t pos = 0;
        const int parsed = std::stoi(v, &pos);
        if (pos != std::strlen(v)) {
            throw std::invalid_argument("trailing characters");
        }
        out = parsed;
        return true;
    } catch (const std::exception &) {
        fprintf(stderr, "invalid integer for %s: '%s'\n", what, v);
        return false;
    }
}

static std::string slurp_file(const std::string & path, stderr_prefix * log_prefix = nullptr) {
    std::ifstream f(path);
    if (!f.is_open()) {
        prefixed_fprintf(log_prefix, "failed to open %s\n", path.c_str());
        return "";
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // A stream error mid-read returns a silently-truncated prefix that would
    // prefill a wrong (shorter) conditioning and still "succeed" downstream.
    if (f.bad()) {
        prefixed_fprintf(log_prefix, "read error on %s\n", path.c_str());
        return "";
    }
    return s;
}

static std::vector<int32_t> read_tokens_bin(const std::string & path, stderr_prefix * log_prefix = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        prefixed_fprintf(log_prefix, "failed to open %s\n", path.c_str());
        return {};
    }
    auto size = f.tellg();
    if (size % 4 != 0) {
        prefixed_fprintf(log_prefix, "tokens-in size %lld is not a multiple of 4\n", (long long) size);
        return {};
    }
    f.seekg(0);
    std::vector<int32_t> out(size / 4);
    f.read(reinterpret_cast<char *>(out.data()), size);
    // A short read would leave a zero-filled tail: token id 0 is in-vocab, so
    // it would otherwise be teacher-forced and scored as if it were real data.
    if (!f) {
        prefixed_fprintf(log_prefix, "short read on %s (got %lld of %lld bytes)\n",
                         path.c_str(), (long long) f.gcount(), (long long) size);
        return {};
    }
    return out;
}

static bool validate_tokens_in_vocab(
        const std::vector<int32_t> & tokens,
        int32_t n_vocab,
        stderr_prefix * lp) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || tokens[i] >= n_vocab) {
            prefixed_fprintf(lp, "token id %d at index %zu out of vocab range [0, %d)\n",
                             tokens[i], i, n_vocab);
            return false;
        }
    }
    return true;
}

// KLD compares the two output distributions index by index, so equal vocab
// sizes are insufficient: every token id must denote the same token on both
// sides. Check the full mapping once after model load, before inference.
//
// allow_attr_mismatch relaxes ONLY the per-token attribute comparison (token
// type / EOG flags), which vendors sometimes disagree on for the same base
// model (e.g. gemma-4-E4B-it: bartowski types token 1 <eos> CONTROL with
// eos_token_id=1, unsloth types it NORMAL with eos_token_id=106). Attributes
// affect special-token parsing and generation stopping, neither of which the
// teacher-forced scorers use; the index-by-index distribution comparison only
// needs the id->text mapping, which stays strictly enforced. Every mismatch
// is still reported so the run log records the exact divergence.
static bool validate_vocab_id_mapping(
        const llama_vocab * ref_vocab,
        const llama_vocab * cand_vocab,
        stderr_prefix * lp = nullptr,
        bool allow_attr_mismatch = false) {
    const int32_t n_ref  = llama_vocab_n_tokens(ref_vocab);
    const int32_t n_cand = llama_vocab_n_tokens(cand_vocab);
    if (n_ref != n_cand) {
        prefixed_fprintf(lp,
                "vocab mismatch: ref=%d cand=%d — the two models must share a tokenizer\n",
                n_ref, n_cand);
        return false;
    }
    if (llama_vocab_type(ref_vocab) != llama_vocab_type(cand_vocab)) {
        prefixed_fprintf(lp,
                "vocab mismatch: tokenizer types differ (ref=%d cand=%d)\n",
                (int) llama_vocab_type(ref_vocab),
                (int) llama_vocab_type(cand_vocab));
        return false;
    }
    int64_t n_attr_mismatch = 0;
    for (llama_token id = 0; id < n_ref; ++id) {
        const char * ref_text  = llama_vocab_get_text(ref_vocab, id);
        const char * cand_text = llama_vocab_get_text(cand_vocab, id);
        const bool text_matches =
                ref_text != nullptr && cand_text != nullptr &&
                std::strcmp(ref_text, cand_text) == 0;
        if (!text_matches) {
            prefixed_fprintf(lp,
                    "vocab id mapping mismatch at token id %d: token text differs\n",
                    (int) id);
            return false;
        }
        const llama_token_attr ref_attr  = llama_vocab_get_attr(ref_vocab, id);
        const llama_token_attr cand_attr = llama_vocab_get_attr(cand_vocab, id);
        if (ref_attr != cand_attr) {
            if (!allow_attr_mismatch) {
                prefixed_fprintf(lp,
                        "vocab id mapping mismatch at token id %d: token attributes differ "
                        "(--allow-vocab-attr-mismatch to accept)\n",
                        (int) id);
                return false;
            }
            ++n_attr_mismatch;
            prefixed_fprintf(lp,
                    "warning: token id %d ('%s') attributes differ (ref=0x%x cand=0x%x); "
                    "accepted via --allow-vocab-attr-mismatch\n",
                    (int) id, ref_text, (unsigned) ref_attr, (unsigned) cand_attr);
        }
    }
    if (n_attr_mismatch > 0) {
        prefixed_fprintf(lp,
                "warning: %lld token attribute mismatch(es) accepted; token texts all match\n",
                (long long) n_attr_mismatch);
    }
    return true;
}
