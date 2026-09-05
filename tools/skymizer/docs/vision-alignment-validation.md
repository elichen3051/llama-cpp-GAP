# Vision preprocessing alignment validation

Validated on 2026-09-05 in the `llama.cpp` checkout based on `f41f902cf`, using
local GGUF models and `elichen-skymizer/*-pivot` datasets. The implementation
prepares RGB geometry once and makes mtmd preserve it. Stored reference IDs
and reference text are unchanged. See [collection usage](collect.md#aligned-image-preprocessing).

## Dataset and GPU coverage

The implemented geometry matched every image's reference placeholder run,
as well as the recorded total and maximum vision-token counts, in 1,292 rows
containing 1,496 images. This included 124 rows with multiple images, across
13 dataset subsets.

| Model | Rows checked on CPU | GPU scoring cases |
| --- | ---: | ---: |
| Gemma 4 31B | 100 | 1 |
| Gemma 4 E4B | 300 | 4 |
| GLM-4.6V-Flash | 100 | 1 |
| Kimi-VL-A3B-Instruct | 195 | 3 |
| Kimi-VL-A3B-Thinking-2506 | 198 | 3 |
| Qwen3-VL-4B-Instruct | 100 | 1 |
| Qwen3-VL-4B-Thinking | 100 | 1 |
| Qwen3.5-4B | 100 | 1 |
| Qwen3.6-35B-A3B | 99 | 1 |

All 16 GPU cases passed the scorer's checks for PNG dimensions, per-image
embedding counts, reference text IDs, image placeholder IDs, and M-RoPE grids
where applicable. Each case scored eight answer tokens. Actual decoder
positions matched the expected positions in every case.

The GPU was an RTX PRO 6000 Blackwell Server Edition. Gemma E4B compared BF16
against Q4_K_M; the remaining variants compared the same Q4 model on both
sides and returned zero KLD. These are input-contract and execution checks,
not full quantization evaluations or proof of vLLM/llama.cpp logit equality.

Selected boundary cases:

- Gemma E4B single-image and four-image prefills matched 488 and 1,052
  reference tokens respectively. Narrow OCR images also passed.
- Kimi Instruct preserved 1,100 image tokens and Kimi Thinking preserved
  4,240, even with the scorer's native maximum set to 1,024.
- The GLM four-image case preserved its 1,321-token sequential prefix and
  correctly advanced the M-RoPE decoder position to 97.
- Gemma and Kimi inputs that change geometry when the HF resize step is
  applied twice were included; mtmd preserved the first prepared geometry.

## Other checks

- The C++ `llama-vlm-kld` target built successfully.
- All 571 Python tests passed. Geometry regressions cover rounding, Gemma
  pooling, Kimi padding and distinct budgets, GLM's temporal factor, nested
  overrides, and swapped per-image counts with unchanged total counts.
- For three examples each from Gemma, Qwen, and GLM, the inspected HF
  processor produced exactly equal pixel tensors and position/grid tensors
  when comparing original-image processing with prepared-image processing
  using `do_resize=False`.
- Six intentionally invalid scorer inputs were rejected without metrics:
  wrong PNG dimensions, inconsistent grid metadata, changed text IDs,
  changed image placeholder IDs, missing metadata, and an unsupported
  preprocessing version. A valid row after these failures still scored.
- A real Gemma E4B row completed the full `collect_kld.py` workflow. The
  temporary prep files were removed and the preprocessing metadata remained
  beside the metrics. The existing collector regression test also verifies
  that metadata survives binary-to-NPZ conversion.

- Both collector modes were also exercised with Qwen3.6-35B-A3B BF16 on a
  thinking sample, scoring 2,048 reference tokens per mode. Native and aligned
  mode each reproduced the corresponding direct-scorer run exactly, including
  every BF16 NLL value. Native mode preserved original RGB pixels and omitted
  the HF geometry contract; aligned mode enforced it.

Local run artifacts are under
`/tmp/vision-token-investigation-20260905/implemented/`, including
`geometry-validation.json`, `gpu-results.json`, per-model scorer logs,
`negative-contracts/`, and `collector-smoke/`.

## Reproducibility limits

The original model revisions and exact generation backend versions were not
recorded. The inspected generation environment has vLLM 0.20.0 and
Transformers 5.6.2. The implementation defaults to torchvision resizing with
torch 2.11.0 and torchvision 0.26.0; Kimi uses its source processor's Pillow
path. CPU wheels and Pillow 12.3.0 were used by the final collector smoke run.

The dataset's per-image token counts and stored prefix IDs provide direct
alignment checks. They cannot establish which interpolation backend produced
the original reference. Backend choice and installed package versions are
therefore explicit metadata, and incompatible or legacy collections cannot
be mixed. Re-score existing references into a fresh output directory.
