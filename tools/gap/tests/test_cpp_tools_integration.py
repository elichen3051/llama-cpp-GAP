# Behaviour tests for the two C++ KLD scorers, replacing the source-text tests
# that used to grep .cpp files for `cparams.n_threads_batch = ...`, the
# absence of `slurp_file`, and the absence of `--top-k`. They run the built
# binaries; each is marked `integration` and skips cleanly when the build
# or a C compiler is absent.
#
# The context probe is an LD_PRELOAD shim that stubs model loading and
# intercepts llama_init_from_model to print the llama_context_params fields
# the tool actually passes (threads, swa_full) -- no model file needed,
# milliseconds per tool.
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

COMPANY = Path(__file__).resolve().parents[1]
REPO = COMPANY.parents[1]
BIN = Path(os.environ.get("COMPANY_TEST_BIN", REPO / "build" / "bin"))
TOOLS = ("llama-llm-kld", "llama-vlm-kld")
# Each tool's required arguments (values need not exist: every probe exits
# before any file is opened, and the usage checks only test for presence).
REQUIRED = {
    "llama-llm-kld": ["--ref-model", "m.gguf", "--cand-model", "c.gguf",
                      "--tokens-in", "t.bin", "--n-prefill", "1", "--output-metrics", "o.bin"],
    "llama-vlm-kld": ["--ref-model", "m.gguf", "--ref-mmproj", "p.gguf",
                      "--cand-model", "c.gguf", "--cand-mmproj", "p.gguf",
                      "--image", "i.png", "--formatted-chat", "c.txt",
                      "--tokens-in", "t.bin", "--n-prefill", "1", "--output-metrics", "o.bin"],
}

pytestmark = pytest.mark.integration


def _binary(tool):
    p = BIN / tool
    if not p.exists():
        pytest.skip(f"{p} not built")
    return p


@pytest.fixture(scope="module")
def context_probe(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available to build the LD_PRELOAD probe")
    d = tmp_path_factory.mktemp("probe")
    src = d / "probe.c"
    src.write_text(textwrap.dedent('''
        #include <stdio.h>
        #include <stdlib.h>
        #include "llama.h"
        struct llama_model * llama_model_load_from_file(const char * path, struct llama_model_params params) {
            (void) path; (void) params; return (struct llama_model *) 0x1; }
        const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
            (void) model; return (const struct llama_vocab *) 0x1; }
        int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) { (void) vocab; return 1000; }
        struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params) {
            (void) model;
            fprintf(stderr, "[probe] n_threads=%d n_threads_batch=%d\\n",
                    (int) params.n_threads, (int) params.n_threads_batch);
            fprintf(stderr, "[probe] swa_full=%d\\n", (int) params.swa_full);
            exit(42); }
    '''))
    so = d / "probe.so"
    subprocess.run(["gcc", "-shared", "-fPIC", str(src), "-o", str(so),
                    "-I", str(REPO / "include"), "-I", str(REPO / "ggml" / "include")],
                   check=True)
    return so


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_passes_t_to_both_thread_knobs(tool, context_probe):
    """-t must reach n_threads AND n_threads_batch (prefill + chunked
    teacher forcing run as multi-token ubatches, which use the batch knob;
    llama-perplexity's -tb defaults to -t the same way). vlm-score used to
    set only n_threads -- the drift fixed in 81ac2984 -- and this is the
    behaviour test that would have caught it, on all four lanes."""
    r = subprocess.run([str(_binary(tool)), *REQUIRED[tool], "-t", "6"],
                       env={**os.environ, "LD_PRELOAD": str(context_probe)},
                       capture_output=True, text=True)
    assert r.returncode == 42, r.stderr
    assert "[probe] n_threads=6 n_threads_batch=6" in r.stderr


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("flags,expected", [([], 0), (["--swa-full"], 1)])
def test_kld_tools_swa_full_default_off_and_flag_on(tool, context_probe, flags, expected):
    """llama_context_default_params() sets swa_full = true, which allocates
    n_ctx KV cells for every sliding-window layer (gemma-4 has them). Both
    scorers run one sequential sequence, so by default they pass swa_full =
    false like the common llama.cpp CLI; --swa-full restores the full-size
    cache for runs that must match earlier collections."""
    r = subprocess.run([str(_binary(tool)), *REQUIRED[tool], *flags],
                       env={**os.environ, "LD_PRELOAD": str(context_probe)},
                       capture_output=True, text=True)
    assert r.returncode == 42, r.stderr
    assert f"[probe] swa_full={expected}" in r.stderr


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_rejects_a_malformed_integer_cleanly(tool):
    """Every integer flag goes through parse_int_arg: a typo is a usage error
    with the flag named, never std::terminate (vlm-score used bare std::stoi
    until 388e49a1)."""
    r = subprocess.run([str(_binary(tool)), *REQUIRED[tool], "-ngl", "9o"],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "invalid integer for -ngl: '9o'" in r.stderr


@pytest.mark.parametrize("tool", ("llama-llm-kld", "llama-vlm-kld"))
@pytest.mark.parametrize("flags", (("--flash-attn", "--no-flash-attn"),
                                   ("--flash-attn", "--flash-attn")))
def test_kld_tools_reject_multiple_flash_attention_modes(tool, flags):
    r = subprocess.run([str(_binary(tool)), *REQUIRED[tool], *flags],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "flash-attention mode specified more than once" in r.stderr


@pytest.fixture(scope="module")
def vocabulary_probe(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    directory = tmp_path_factory.mktemp("vocabulary-probe")
    source = directory / "probe.c"
    source.write_text(r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"
struct llama_model * llama_model_load_from_file(const char * path, struct llama_model_params p) {
    if (!p.vocab_only) { fprintf(stderr, "WEIGHTS_LOADED\n"); }
    return (struct llama_model *) (strstr(path, "permuted") ? 2 : strstr(path, "larger") ? 3 : 1);
}
void llama_model_free(struct llama_model * m) { (void) m; }
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * m) { return (const struct llama_vocab *) m; }
int32_t llama_vocab_n_tokens(const struct llama_vocab * v) { return (size_t)v == 3 ? 3 : 2; }
enum llama_vocab_type llama_vocab_type(const struct llama_vocab * v) { (void)v; return LLAMA_VOCAB_TYPE_BPE; }
const char * llama_vocab_get_text(const struct llama_vocab * v, llama_token id) {
    const char * texts[] = {"a", "b", "c"};
    return texts[(size_t)v == 2 ? 1 - id : id];
}
enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * v, llama_token id) { (void)v; (void)id; return LLAMA_TOKEN_ATTR_NORMAL; }
struct llama_context * llama_init_from_model(struct llama_model * m, struct llama_context_params p) {
    (void)m; (void)p; fprintf(stderr, "INFERENCE_REACHED\n"); exit(42);
}
""")
    so = directory / "probe.so"
    subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(so),
                    "-I", str(REPO / "include"), "-I", str(REPO / "ggml/include")], check=True)
    return so


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("candidate,expected", [("quantized.gguf", 42), ("permuted.gguf", 1), ("larger.gguf", 1)])
def test_native_vocabulary_is_checked_before_loading_weights(tool, candidate, expected, vocabulary_probe, tmp_path):
    import json
    env = {**os.environ, "LD_PRELOAD": str(vocabulary_probe)}
    binary = _binary(tool)
    vocab = subprocess.run([str(binary), "--vocab-identity", "reference.gguf"],
                           capture_output=True, text=True, env=env, check=True)
    entry = {"tokens_in": "unused.tokens", "n_prefill": 1, "output_metrics": "unused.bin",
             "reference_vocabulary": json.loads(vocab.stdout)}
    if tool == "llama-vlm-kld":
        entry.update(images=[], formatted_chat="unused.txt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    args = [str(binary), "--ref-model", "reference.gguf", "--cand-model", candidate,
            "--manifest", str(manifest)]
    if tool == "llama-vlm-kld":
        args += ["--ref-mmproj", "unused.mmproj", "--cand-mmproj", "unused.mmproj"]
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    if expected == 1:
        assert "vocabulary mismatch" in result.stderr
        assert "WEIGHTS_LOADED" not in result.stderr
    else:
        assert "INFERENCE_REACHED" in result.stderr


@pytest.mark.parametrize("flags,message", [
    (["--spec-type", "draft-mtp", "--spec-draft-n-max", "0"], "invalid MTP draft bounds"),
    (["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-n-min", "3"], "invalid MTP draft bounds"),
    (["--spec-type", "draft-mtp", "-b", "2", "-ub", "2"], "MTP draft maximum"),
    (["--spec-type", "ngram-simple"], "supports only autoregressive"),
    (["--spec-type", "draft-mtp", "--spec-synth-len", "2"], "synthetic acceptance"),
    (["--spec-type", "draft-mtp", "--model-draft", "/missing/mtp.gguf"], "MTP sidecar does not exist"),
])
def test_reference_rejects_invalid_mtp_setup_before_loading_weights(tmp_path, flags, message):
    model = tmp_path / "target.gguf"
    model.write_bytes(b"not a model; validation must run before loading")
    result = subprocess.run([str(_binary("llama-reference")), "-m", str(model), "--describe", *flags],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert message in result.stderr


def test_reference_repetition_stops_only_long_generated_loops(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    source = tmp_path / "repeat.cpp"
    source.write_text(r'''
#include "company-repetition.h"
#include <cassert>
#include <vector>
using namespace company_repetition;
int main() {
    std::vector<int32_t> x = {1, 1, 1, 4, 5, 4, 5, 4, 5};
    assert(!tail(x.data(), x.size()));
    assert(!full(x.data(), x.size()));
    x.clear();
    for (int i = 0; i < 500; ++i) { x.push_back(i); }
    assert(!full(x.data(), x.size()));
    for (int i = 0; i < 96; ++i) { x.push_back(1000 + i % 4); }
    auto m = tail(x.data(), x.size());
    assert(m && m.start == 500 && m.unit_len == 4 && m.repeated_len == 96);
    x.push_back(9000); x.push_back(9001);
    assert(!tail(x.data(), x.size()));
    m = full(x.data(), x.size());
    assert(m && m.start == 500 && m.repeated_len == 96);
    x.clear();
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 800; ++i) { x.push_back(2000 + i); }
    }
    assert(!tail(x.data(), x.size()));
    assert(full(x.data(), x.size()).unit_len == 800);
    // Prompt repetition is not part of the generated suffix passed to the detector.
    assert(!full(x.data() + x.size() - 10, 10));
}
''')
    exe = tmp_path / "repeat"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(COMPANY / "core"), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize("tool", TOOLS)
def test_kld_metric_self_test_and_current_format(tool):
    import lib.kld_metrics_io as kio
    result = subprocess.run([str(_binary(tool)), "--self-test"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "self-test PASS" in result.stderr
    version = subprocess.run([str(_binary(tool)), "--vlmk-version"],
                             capture_output=True, text=True, check=True)
    assert version.stdout.strip() == str(kio.VLMK_VERSION)


def test_vlm_prefix_checks_adjacent_tiles_and_preserves_strict_validation(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    source = tmp_path / "prefix.cpp"
    source.write_text(r"""
#define main company_vlm_kld_main
#include "vlm-kld.cpp"
#undef main
#include <cassert>
struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> text;
    size_t n;
};
struct mtmd_input_chunks { std::vector<mtmd_input_chunk> entries; };
size_t mtmd_input_chunks_size(const mtmd_input_chunks * c) { return c->entries.size(); }
const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * c, size_t i) { return &c->entries.at(i); }
void mtmd_input_chunks_free(mtmd_input_chunks * c) { delete c; }
mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * c) { return c->type; }
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * c) { return c->n; }
const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * c, size_t * n) {
    *n = c->text.size();
    return c->text.data();
}
int main() {
    for (int tiles : {1, 3, 13}) {
        mtmd::input_chunks chunks(new mtmd_input_chunks);
        auto & entries = chunks.ptr->entries;
        entries.push_back({MTMD_INPUT_CHUNK_TYPE_TEXT, {1, 2}, 2});
        for (int i = 0; i < tiles; ++i) {
            entries.push_back({MTMD_INPUT_CHUNK_TYPE_IMAGE, {}, 256});
        }
        entries.push_back({MTMD_INPUT_CHUNK_TYPE_TEXT, {3, 4}, 2});
        entries.push_back({MTMD_INPUT_CHUNK_TYPE_IMAGE, {}, 256});
        entries.push_back({MTMD_INPUT_CHUNK_TYPE_TEXT, {5}, 1});
        std::vector<int32_t> tokens = {1, 2};
        tokens.insert(tokens.end(), tiles * 256, -1);
        tokens.insert(tokens.end(), {3, 4});
        tokens.insert(tokens.end(), 256, -1);
        tokens.push_back(5);
        auto check = [&](const std::vector<int32_t> & input, bool strict) {
            return check_prefix_against_tokens(chunks, input, input.size(), strict, "test", nullptr);
        };
        assert(check(tokens, true));
        auto changed = tokens;
        changed[tiles * 256 + 2] = 9;
        assert(!check(changed, true));
        assert(!check(changed, false));
        changed = tokens;
        changed.erase(changed.begin() + 2);
        assert(!check(changed, true));
        assert(check(changed, false));
        changed.back() = 9;
        assert(!check(changed, false));
        changed = tokens;
        changed.push_back(6);
        assert(!check(changed, true));
        changed = tokens;
        changed[2 + tiles * 128] = -2;
        assert(!check(changed, true));
    }
}
""")
    exe = tmp_path / "prefix"
    includes = [COMPANY / "core", REPO / "include", REPO / "ggml/include", REPO / "common", REPO / "vendor", REPO / "tools/mtmd"]
    command = ["g++", "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]
    for directory in includes:
        command += ["-I", str(directory)]
    subprocess.run([*command, str(source), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_llm_perplexity_window_matches_full_batch_and_preserves_default(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    source = tmp_path / "window.cpp"
    source.write_text(r"""
#define main company_llm_kld_main
#include "llm-kld.cpp"
#undef main
#include <cassert>
struct llama_context {
    int length, side, calls = 0, clears = 0;
    bool window;
    std::vector<std::vector<float>> rows;
};
struct llama_vocab { bool add_eos; };
static std::vector<int32_t> input;
static bool poison = false;
void llama_free(llama_context * ctx) { delete ctx; }
void llama_model_free(llama_model *) {}
uint32_t llama_n_ctx(const llama_context * ctx) { return ctx->length; }
uint32_t llama_n_ubatch(const llama_context * ctx) { return ctx->length; }
bool llama_vocab_get_add_eos(const llama_vocab * v) { return v->add_eos; }
llama_memory_t llama_get_memory(const llama_context * ctx) { return (llama_memory_t) ctx; }
void llama_memory_clear(llama_memory_t memory, bool data) {
    auto * ctx = (llama_context *) memory;
    assert(data == ctx->window);
    ++ctx->clears;
    ctx->calls = 0;
}
llama_batch llama_batch_init(int32_t n, int32_t embd, int32_t seq) {
    assert(embd == 0 && seq == 1);
    llama_batch b{};
    b.token = new llama_token[n]; b.pos = new llama_pos[n]; b.n_seq_id = new int32_t[n];
    b.seq_id = new llama_seq_id *[n]; b.seq_id[0] = new llama_seq_id[n];
    for (int i = 1; i < n; ++i) { b.seq_id[i] = b.seq_id[0] + i; }
    b.logits = new int8_t[n];
    return b;
}
void llama_batch_free(llama_batch b) {
    delete[] b.token; delete[] b.pos; delete[] b.n_seq_id;
    delete[] b.seq_id[0]; delete[] b.seq_id; delete[] b.logits;
}
void common_batch_clear(llama_batch & b) { b.n_tokens = 0; }
void common_batch_add(llama_batch & b, llama_token t, llama_pos p, const std::vector<llama_seq_id> & seq, bool logits) {
    assert(seq == std::vector<llama_seq_id>{0});
    int i = b.n_tokens++;
    b.token[i] = t; b.pos[i] = p; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = logits;
}
int32_t llama_decode(llama_context * ctx, llama_batch b) {
    const int half = ctx->length / 2;
    const int base = ctx->window || ctx->calls == 0 ? 0 : half + 1;
    const int count = ctx->window ? ctx->length : ctx->calls == 0 ? half + 1 : half - 2;
    assert(b.n_tokens == count);
    ctx->rows.assign(count, {});
    for (int i = 0; i < count; ++i) {
        const int pos = base + i;
        const bool keep = ctx->window ? pos >= half : ctx->calls == 0 ? pos == half : true;
        assert(b.pos[i] == pos && b.token[i] == input[pos]);
        assert(b.n_seq_id[i] == 1 && b.seq_id[i][0] == 0 && bool(b.logits[i]) == keep);
        if (keep) {
            ctx->rows[i].assign(16, 0.0f);
            if (pos == ctx->length - 1 || poison) {
                ctx->rows[i][0] = std::numeric_limits<float>::quiet_NaN();
            } else {
                ctx->rows[i][input[pos + 1]] = ctx->side == 0 ? 2.0f : 1.0f;
            }
        }
    }
    ++ctx->calls;
    return 0;
}
float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    if (i == -1) { i = (int) ctx->rows.size() - 1; }
    return i >= 0 && i < (int) ctx->rows.size() && !ctx->rows[i].empty() ? ctx->rows[i].data() : nullptr;
}
int main(int argc, char ** argv) {
    assert(argc == 2);
    llm_kld_args a;
    a.metric_threads = 2;
    a.tokens_in_path = std::string(argv[1]) + "/tokens.bin";
    a.output_metrics_path = std::string(argv[1]) + "/metrics.bin";
    for (int length : {8, 512}) {
        a.n_ctx = a.n_batch = a.n_ubatch = length;
        a.n_prefill = length / 2 + 1;
        input.resize(length);
        for (int i = 0; i < length; ++i) { input[i] = (i * 3 + 7) % 16; }
        std::ofstream f(a.tokens_in_path, std::ios::binary);
        f.write((const char *) input.data(), input.size() * sizeof(int32_t)); f.close();
        for (bool window : {false, true}) {
            a.perplexity_window = window;
            model_side ref, cand;
            ref.tag = "ref"; cand.tag = "cand";
            ref.lctx.reset(new llama_context{length, 0, 0, 0, window, {}});
            cand.lctx.reset(new llama_context{length, 1, 0, 0, window, {}});
            for (int repeat = 0; repeat < 2; ++repeat) {
                assert(score_one(a, ref, cand, 16, length, nullptr));
                assert(ref.lctx->calls == (window ? 1 : 2) && cand.lctx->calls == ref.lctx->calls);
                assert(ref.lctx->clears == repeat + 1 && cand.lctx->clears == repeat + 1);
                std::ifstream out(a.output_metrics_path, std::ios::binary);
                uint32_t h[6]; out.read((char *) h, sizeof(h));
                assert(h[0] == VLMK_MAGIC && h[1] == VLMK_VERSION && h[2] == 16);
                assert(h[3] == length / 2 - 1 && h[4] == a.n_prefill && h[5] == (window ? length : a.n_prefill));
                for (int i = 0; i < length / 2 - 1; ++i) {
                    kld_record rec; out.read((char *) &rec, sizeof(rec));
                    assert(out && rec.target == input[a.n_prefill + i]);
                    assert(std::abs(rec.nll_ref - (std::log(std::exp(2.0) + 15) - 2)) < 1e-6);
                    assert(std::abs(rec.nll_cand - (std::log(std::exp(1.0) + 15) - 1)) < 1e-6);
                    assert(validate_kld_record_finite(rec, i, nullptr));
                }
                assert(out.peek() == EOF);
            }
            std::remove(a.output_metrics_path.c_str());
            poison = true;
            assert(!score_one(a, ref, cand, 16, length, nullptr));
            assert(!std::ifstream(a.output_metrics_path));
            poison = false;
        }
        assert(validate_perplexity_window_input(a, length, nullptr));
        assert(!validate_perplexity_window_input(a, length - 1, nullptr));
        --a.n_prefill;
        assert(!validate_perplexity_window_input(a, length, nullptr));
    }
    const auto parse = [](std::vector<std::string> extra) {
        std::vector<std::string> flags = {"test", "--ref-model", "ref", "--cand-model", "cand", "--tokens-in", "tokens", "--output-metrics", "metrics", "--n-prefill", "5", "--perplexity-window", "-c", "8", "-b", "8", "-ub", "8"};
        flags.insert(flags.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto & flag : flags) { argv.push_back(flag.data()); }
        llm_kld_args parsed;
        return parse_args(argv.size(), argv.data(), parsed);
    };
    assert(parse({}) && parse({"--num-eval-tokens", "3"}) && parse({"--tf-chunk", "8"}));
    for (const auto & flags : std::vector<std::vector<std::string>>{
            {"-c", "7"}, {"-c", "2", "-b", "2", "-ub", "2"}, {"-b", "16"}, {"-ub", "4"},
            {"--num-eval-tokens", "2"}, {"--num-eval-tokens", "4"}, {"--tf-chunk", "1"}}) {
        assert(!parse(flags));
    }
    llama_vocab no{false}, yes{true};
    assert(validate_perplexity_window_vocab(&no, &no));
    assert(!validate_perplexity_window_vocab(&yes, &no));
    assert(!validate_perplexity_window_vocab(&no, &yes));
}
""")
    exe = tmp_path / "window"
    includes = [COMPANY / "core", REPO / "include", REPO / "ggml/include", REPO / "common", REPO / "vendor"]
    command = ["g++", "-std=c++17", "-O1", "-pthread", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]
    for directory in includes:
        command += ["-I", str(directory)]
    subprocess.run([*command, str(source), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_metric_worker_counts_preserve_record_bytes(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    source = tmp_path / "metric-workers.cpp"
    source.write_text(r'''
#include "company-vlmk-kernel.h"
#include <cassert>
int main() {
    const int rows = 31, vocab = 257;
    std::vector<std::vector<float>> ref(rows, std::vector<float>(vocab));
    std::vector<std::vector<float>> cand = ref;
    std::vector<const float *> rp, cp;
    std::vector<int32_t> targets;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < vocab; ++j) {
            ref[i][j] = float((i * 19 + j * 7) % 101 - 50) / 9;
            cand[i][j] = ref[i][j] + float((i + j) % 11 - 5) / 100;
        }
        rp.push_back(ref[i].data());
        cp.push_back(cand[i].data());
        targets.push_back(i % vocab);
    }
    std::vector<kld_record> serial(rows), parallel(rows);
    compute_records_parallel(rp, cp, targets, vocab, 1, serial.data());
    for (int threads : {4, 8, 12, 64}) {
        compute_records_parallel(rp, cp, targets, vocab, threads, parallel.data());
        assert(std::memcmp(serial.data(), parallel.data(), rows * sizeof(kld_record)) == 0);
    }
}
''')
    exe = tmp_path / "metric-workers"
    command = ["g++", "-std=c++17", "-O2", "-pthread", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections"]
    for directory in (COMPANY / "core", REPO / "include", REPO / "ggml/include", REPO / "common", REPO / "vendor"):
        command += ["-I", str(directory)]
    subprocess.run([*command, str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
