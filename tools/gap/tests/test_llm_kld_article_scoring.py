"""Run the real generic scorer with deterministic CPU logits and article rows."""

import shutil
import subprocess
import sys

import numpy as np
import pytest

from article_fakes import GAP, REPO, VOCABULARY, prepared_articles
from lib.kld_metrics_io import KLD_RECORD_DT, load_kld_metrics


pytestmark = pytest.mark.integration

SOURCE = r'''
#define main company_llm_kld_main
#include "llm-kld.cpp"
#undef main
#include <cassert>
struct llama_context {
    int length, side, cursor = 0;
    std::vector<std::vector<float>> rows;
};
static std::vector<int32_t> input;
void llama_free(llama_context * ctx) { delete ctx; }
void llama_model_free(llama_model *) {}
uint32_t llama_n_ctx(const llama_context * ctx) { return ctx->length; }
uint32_t llama_n_ubatch(const llama_context * ctx) { return ctx->length; }
llama_memory_t llama_get_memory(const llama_context * ctx) { return (llama_memory_t) ctx; }
void llama_memory_clear(llama_memory_t memory, bool) {
    auto * ctx = (llama_context *) memory;
    ctx->cursor = 0;
    ctx->rows.clear();
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
    ctx->rows.assign(b.n_tokens, {});
    for (int i = 0; i < b.n_tokens; ++i) {
        int pos = ctx->cursor++;
        assert(pos < (int) input.size() - 1);
        assert(b.pos[i] == pos && b.token[i] == input[pos]);
        assert(b.n_seq_id[i] == 1 && b.seq_id[i][0] == 0 && b.logits[i]);
        ctx->rows[i].assign(512, 0.0f);
        ctx->rows[i][input[pos + 1]] = 1.0f + float(pos % 7) / 4.0f + float(ctx->side) / 8.0f;
    }
    return 0;
}
float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    if (i == -1) { i = (int) ctx->rows.size() - 1; }
    return i >= 0 && i < (int) ctx->rows.size() ? ctx->rows[i].data() : nullptr;
}
int main(int argc, char ** argv) {
    llm_kld_args args;
    if (!parse_args(argc, argv, args)) { return 2; }
    input = read_tokens_bin(args.tokens_in_path, nullptr);
    model_side ref, cand;
    ref.tag = "ref"; cand.tag = "cand";
    ref.lctx.reset(new llama_context{args.n_ctx, 0, 0, {}});
    cand.lctx.reset(new llama_context{args.n_ctx, 1, 0, {}});
    return score_one(args, ref, cand, 512, args.n_batch, nullptr) ? 0 : 1;
}
'''


def compile_scorer(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ unavailable")
    source = tmp_path / "article-scorer.cpp"
    source.write_text(SOURCE)
    exe = tmp_path / "article-scorer"
    linker = "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
    command = [compiler, "-std=c++17", "-O1", "-pthread", "-ffunction-sections", "-fdata-sections", linker]
    for directory in (GAP / "core", REPO / "include", REPO / "ggml/include", REPO / "common", REPO / "vendor"):
        command += ["-I", str(directory)]
    result = subprocess.run([*command, str(source), "-o", str(exe)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return exe


def score_row(exe, tmp_path, row, chunk, suffix):
    tokens = tmp_path / f"tokens-{suffix}.bin"
    np.asarray(row["input_ids"], dtype="<i4").tofile(tokens)
    output = tmp_path / f"metrics-{suffix}.bin"
    result = subprocess.run([
        str(exe), "--ref-model", "ref.gguf", "--cand-model", "cand.gguf", "--tokens-in", str(tokens),
        "--n-prefill", str(row["n_prefill_tokens"]), "--output-metrics", str(output), "--num-eval-tokens", "-1",
        "-c", "32768", "-b", "2048", "-ub", "512", "--tf-chunk", str(chunk), "--metric-threads", "2",
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return output


@pytest.mark.parametrize("mode", ["bos", "no-bos"])
def test_article_scores_every_target_from_last_prefill_logits(tmp_path, mode):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    articles = (b"x" if mode == "bos" else b"xy", b"A longer article.\n")
    _, _, rows, _ = prepared_articles(tmp_path, mode, articles)
    exe = compile_scorer(tmp_path)
    for i, row in enumerate(rows):
        output = score_row(exe, tmp_path, row, 2048, str(i))
        metrics, header = load_kld_metrics(output)
        count = len(row["input_ids"]) - 1
        assert header["npos"] == count
        assert header["n_prefill"] == 1
        assert header["vocab"] == VOCABULARY["size"]
        assert metrics["target"].tolist() == row["input_ids"][1:]
        assert all(len(values) == count for values in metrics.values())
        for side in ("ref", "cand"):
            boost = 1.0 + (np.arange(count) % 7) / 4.0 + (0 if side == "ref" else 0.125)
            expected = np.log(np.exp(boost) + VOCABULARY["size"] - 1) - boost
            np.testing.assert_allclose(metrics[f"nll_{side}"], expected, rtol=0, atol=1e-6)
            assert metrics[f"argmax_{side}"].tolist() == row["input_ids"][1:]
        assert all(np.isfinite(metrics[key]).all() for key in KLD_RECORD_DT.names if KLD_RECORD_DT[key].kind == "f")


def test_article_teacher_forcing_chunks_preserve_all_record_bytes(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    _, _, rows, _ = prepared_articles(tmp_path, articles=(b"a" * 1024 + b"\n", b"A second article.\n"))
    exe = compile_scorer(tmp_path)
    for i, row in enumerate(rows):
        full = score_row(exe, tmp_path, row, 2048, f"{i}-full")
        for chunk in (1, 3, 7):
            assert chunk < len(row["input_ids"])
            chunked = score_row(exe, tmp_path, row, chunk, f"{i}-{chunk}")
            assert chunked.read_bytes() == full.read_bytes()
