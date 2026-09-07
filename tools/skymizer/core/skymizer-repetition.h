#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// Port of repetition_curse_detector d4163fc1: reversed KMP and consecutive-block matching.
// Apply a minimum repeated span so short punctuation/math repetitions do not trigger.
namespace skymizer_repetition {
constexpr size_t min_span = 96;
constexpr size_t min_repeats = 3;
constexpr size_t window = 2048;
constexpr size_t interval = 32;

struct match {
    size_t start = 0;
    size_t unit_len = 0;
    size_t repeats = 0;
    size_t repeated_len = 0;
    explicit operator bool() const { return repeated_len != 0; }
};

inline match tail(const int32_t * tokens, size_t count) {
    const size_t n = std::min(count, window);
    if (n < min_span) { return {}; }
    std::vector<size_t> pi(n);
    size_t j = 0;
    match best;
    for (size_t i = 1; i < n; ++i) {
        while (j && tokens[count - 1 - i] != tokens[count - 1 - j]) { j = pi[j - 1]; }
        if (tokens[count - 1 - i] == tokens[count - 1 - j]) { ++j; }
        pi[i] = j;
        const size_t length = i + 1;
        const size_t period = length - j;
        if (length >= min_span && length % period == 0 && length / period >= min_repeats) {
            best = {count - length, period, length / period, length};
        }
    }
    return best;
}

inline match full(const int32_t * tokens, size_t n) {
    if (n < min_span) { return {}; }
    for (size_t period = 1; period <= n / min_repeats; ++period) {
        const size_t repeats = std::max(min_repeats, (min_span + period - 1) / period);
        const size_t required = period * (repeats - 1);
        size_t run = 0;
        for (size_t i = 0; i + period < n; ++i) {
            run = tokens[i] == tokens[i + period] ? run + 1 : 0;
            if (run >= required) {
                return {i + 1 - run, period, repeats, period * repeats};
            }
        }
    }
    return {};
}
} // namespace skymizer_repetition
