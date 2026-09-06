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

SKYMIZER = Path(__file__).resolve().parents[1]
REPO = SKYMIZER.parents[1]
BIN = REPO / "build" / "bin"
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
