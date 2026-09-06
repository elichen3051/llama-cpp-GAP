# Upstream port and local smoke tests

The port copies the complete tracked `tools/skymizer` tree from `llama-cpp-for-GAP` commit `03a50eb8ef7569d283a79ba8052a6f65289f8f74` onto upstream llama.cpp commit `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`. This upstream revision already has text KL divergence in `llama-perplexity`, but lacks these VLM/LLM collectors and the saved-metrics paired comparison pipeline.

## Integration

- Register both executable targets after `mtmd` and honor `LLAMA_TOOLS_INSTALL`.
- Adapt image loading to the upstream bitmap/video wrapper and pass the required `mtmd_input_text.text_len`. Leaving this length at zero makes upstream tokenize an empty prompt even when the media marker is present.
- Resolve collector binaries and Git provenance from the repository root after the Python entry points moved into `cli/` and `lib/`.
- Use upstream's OpenSSL build option in `env_setup.sh`, and let the numbered scripts select the repository's `.venv` by default. Explicit `PYTHON` overrides retain their existing behavior.

The metric kernel, statistical engine, VLMK v2 format, and dataset fingerprint algorithm are unchanged. The complete source tests, real binary dataset fixtures, documentation, knowledge files, and optional power-analysis tools are included. Collections from different llama.cpp builds must not be mixed into one paired comparison; re-collect both candidates with the same binary and settings.

## Build and regression checks

Run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-vlm-kld llama-llm-kld -j
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project tools/skymizer --python 3.12 --group dev
build/bin/llama-vlm-kld --self-test
build/bin/llama-llm-kld --self-test
.venv/bin/python -m pytest -q tools/skymizer
```

For a CPU-only build, configure a separate build directory with `-DGGML_CUDA=OFF`. Both executables build and pass their numerical self-tests without a GPU.

## Reproduce the VLM smoke

The runner collects candidate A, candidate B, and an independent repeat of A. It checks every manifest row, VLMK version, non-finite values, metric bounds, identical repeat columns, strict shared-reference alignment, and zero self-pair deltas and confidence intervals. It emits the normal paired Markdown/JSON reports and `summary.json`. The default is three rows and 64 answer tokens per row. Use a fresh output directory on each run; existing collections are refused.

```bash
q35="$HOME/models/qwen3.5-4b/bartowski"
.venv/bin/python tools/skymizer/review-functionality/smoke_kld.py \
    --ref-model "$q35/Qwen_Qwen3.5-4B-bf16.gguf" \
    --cand-a-model "$q35/Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
    --cand-b-model "$q35/Qwen_Qwen3.5-4B-Q4_1.gguf" \
    --mmproj "$q35/mmproj-Qwen_Qwen3.5-4B-bf16.gguf" \
    --dataset elichen-skymizer/qwen3.5-4b-500-ref \
    --subset mmstar-subsample-500-ins-2048 \
    --image-min-tokens 64 --image-max-tokens 16384 \
    --out tools/skymizer/outputs/smoke-qwen35-vlm

q36="$HOME/models/qwen3.6-35b-a3b/bartowski"
.venv/bin/python tools/skymizer/review-functionality/smoke_kld.py \
    --ref-model "$q36/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
    --cand-a-model "$q36/Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf" \
    --cand-b-model "$q36/Qwen_Qwen3.6-35B-A3B-Q4_K_S.gguf" \
    --mmproj "$q36/mmproj-Qwen_Qwen3.6-35B-A3B-bf16.gguf" \
    --dataset elichen-skymizer/qwen3.6-35b-a3b-full-ref-text \
    --subset mmstar-subsample-100-ins \
    --image-min-tokens 64 --image-max-tokens 16384 \
    --out tools/skymizer/outputs/smoke-qwen36-vlm
```

Both datasets contain actual images. `full-ref-text` in the second dataset name does not make it text-only. These smokes run both vision encoders and score text distributions conditioned on image embeddings. Qwen3.6 uses the available Q4_K_M as its reference; its results are a functionality check, not a measurement against full-precision Qwen3.6.

## Reproduce the LLM smoke

The supplied datasets are multimodal. For the text lane, take frozen answer tokens from the Qwen3.5 dataset, use the first 16 as a text prompt, and score the next 64. The helper writes a local parquet dataset and a provenance sidecar. It does not feed image placeholder tokens into the text model or re-tokenize the answer. This fixture checks the LLM pipeline, not the original multimodal task accuracy.

```bash
q35="$HOME/models/qwen3.5-4b/bartowski"
text_data="$PWD/tools/skymizer/outputs/smoke-inputs/llm-answer-continuations"
.venv/bin/python tools/skymizer/review-functionality/prepare_text_smoke.py --out "$text_data"
.venv/bin/python tools/skymizer/review-functionality/smoke_kld.py --lane llm \
    --ref-model "$q35/Qwen_Qwen3.5-4B-bf16.gguf" \
    --cand-a-model "$q35/Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
    --cand-b-model "$q35/Qwen_Qwen3.5-4B-Q4_1.gguf" \
    --dataset "$text_data" \
    --out tools/skymizer/outputs/smoke-qwen35-llm
```

For normal runs, the numbered `scripts/01_save_vlm_kld.sh`, `02_save_llm_kld.sh`, and `03_paired_test_kld.sh` remain the collect/compare entry points. The smoke runner additionally verifies independent repeat collections. `--verify-only` with the same arguments checks the saved smoke results without rescoring.

## Validation record (2026-09-05)

Hardware: NVIDIA RTX PRO 6000 Blackwell, CUDA 13.2.51; Python 3.12, NumPy 2.5.2. CUDA and CPU-only builds both pass both numerical self-tests. The complete test suite passes: **545 passed, 0 skipped**.

The image rows are `mmstar-01052`, `mmstar-00897`, and `mmstar-00191`. Every collection contains 192 scored positions. The VLM settings explicitly match the dataset's configured vision bounds (64 to 16384 tokens per image), with context 8192, batch/microbatch 512, teacher-forcing chunks of 16, eight CPU/metric threads, and flash attention enabled.

| Smoke | Reference | Candidate A mean KLD | Candidate B mean KLD |
| --- | --- | ---: | ---: |
| Qwen3.5 VLM | BF16 | Q4_K_M: 0.0195740207 | Q4_1: 0.0271832627 |
| Qwen3.5 LLM text continuation | BF16 | Q4_K_M: 0.0126955860 | Q4_1: 0.0173912537 |
| Qwen3.6 VLM | Q4_K_M | Q3_K_M: 0.0430904734 | Q4_K_S: 0.0086603570 |

For these completed runs, every metric column in A and its independent repeat is bit-identical, paired reference drift is zero, and all self-pair deltas are exactly zero. Three-row smokes establish pipeline functionality; they do not establish a statistically reliable quantization ranking.

Build logs, collection manifests, prepared images/tokens, per-token NPZ metrics, paired reports, and summaries are retained under `tools/skymizer/outputs/`. This directory and the model files are excluded from the commit. Collection Git provenance records the upstream base HEAD because validation ran before the port commit.
