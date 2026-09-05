# Collecting KLD metrics

Per-script reference for the collection half of the pipeline (the README
keeps the TL;DR and the index). See `compare.md` for the paired-test tool
and `formats.md` for the on-disk layout.

### `prep_vlm_score_from_hf.py` — one HF row → 4 intermediate files

Per-row utility; in a sweep `collect_kld.py` calls
`prep_row()` / `load_dataset_sorted()` / `load_tokenizer()` directly
(in-process, dataset + tokenizer loaded once for the whole run), but the
CLI form here can be run standalone to inspect a single sample.

```bash
python3 tools/skymizer/cli/prep_vlm_score_from_hf.py \
    --out tmp/ds-0 \
    (--row N | --item-id ITEM_ID) \
    [--dataset elichen-skymizer/GAP-mmmu-pro-standard-10] \
    [--subset  qwen3.5-4b-ins-gen-2048] \
    [--split   train] \
    [--sort-by num_images]
```

Writes `img_{i}.png` ×N, `tokens.bin` (int32 input_ids), `formatted_chat.txt`
(chat string with single `<__media__>` markers per image), `meta.json`
(`n_prefill`, `n_answer`, `num_images`, plus the full
`image_processor_config` and a derived `image_token_limits` block with
`image_min_tokens`/`image_max_tokens` for mtmd, …). Examples: `--row 0` =
smallest `num_images`, `--row -1` = largest (under default `--sort-by`).

`--row` and `--item-id` are mutually exclusive; exactly one must be supplied.
`--row` accepts negative indices (Python semantics).

#### Aligned image preprocessing

With the default `--image-preprocessing`, VLM prep resizes each original image once using the dataset row's effective HF settings, including nested `vllm_mm_processor_kwargs.size` overrides. It writes lossless RGB PNGs and verifies every image's token count against the corresponding run of reference placeholder IDs. The reference text and `tokens.bin` are unchanged.

Install the default resize backend with `uv sync --project tools/skymizer --extra vision`. The default is `--image-resize-backend torchvision`, matching the inspected Transformers generation environment (torch 2.11.0, torchvision 0.26.0). This is an explicit assumption for older datasets that did not record their backend. Use `--image-resize-backend pillow` when generation used the PIL backend. Kimi always uses its source processor's Pillow bicubic resize and bottom/right black padding before normalization. Backend and package versions are recorded; there is no automatic fallback.

When enabled, `collect_kld.py` passes `--images-preprocessed` and each row's `image_metadata` path to the scorer. In this mode mtmd validates patch alignment and directly normalizes/encodes the RGB image: it does not resize or pad it again. `--image-min-tokens` and `--image-max-tokens` do not alter prepared images, including Kimi outputs above the nominal cap. Direct scorer calls must pass both `--images-preprocessed` and `--image-metadata DIR/meta.json`.

For each side, before its forward pass, the scorer verifies the PNG dimensions, per-image token counts, M-RoPE grid coordinates where applicable, and every non-image prefix token against the stored reference IDs. Kimi's complete `image<|media_content|>` prelude is restored by mtmd. GLM-4.6V is registered with its own wrapper and temporal factor: the recorded default size 12544/9633792 corresponds to 8/6144 image tokens, not 16/12288.

Each aligned `meta.json` includes `image_preprocessing` (version, backend, package versions, source/output RGB hashes, original/resized/final sizes, merged grids and token counts) and `n_past_expected`. Successful collections retain that metadata in `metrics/NNN_ITEM.preprocess.json` even when temporary prep directories are removed. `collect_meta.json.image_preprocessing` is a shard and comparison identity field. Use a fresh output directory to re-score existing vLLM references; legacy and aligned results cannot be mixed. Do not feed prepared images through the HF resize step again: Gemma and Kimi preprocessing is deterministic but not generally idempotent.

Both `collect_kld.py` and standalone `prep_vlm_score_from_hf.py` accept the same switch:

| Option | Images passed to mtmd | Resize and padding |
| --- | --- | --- |
| `--image-preprocessing` (default) | HF-aligned RGB images with geometry metadata | Prepared once in Python; mtmd preserves geometry |
| `--no-image-preprocessing` | Original dataset images saved losslessly as RGB PNGs | Native mtmd processing, controlled by `--image-min-tokens` / `--image-max-tokens` |

Disabled mode skips HF geometry checks and the aligned `n_past_expected` check. The collector omits both `--images-preprocessed` and the manifest's `image_metadata` field. Reference text, IDs, and thinking mode are unchanged. `--image-resize-backend` is ignored in this mode, and the vision extra is not required for image preparation. Always start from dataset originals; removing the scorer flag from already resized inputs would process those inputs again.

Native collections record `image_preprocessing.resize_backend="native"`, Pillow's version, original sizes and RGB hashes. They also retain `.preprocess.json` sidecars, without claiming an HF grid or expected decoder position. Use separate output directories for native and aligned collections. Native mode uses the current mtmd build, including existing model fixes; it does not restore an older binary. Historical collections with no preprocessing identity remain separate from either newly recorded mode.

See [validation coverage and reproducibility limits](vision-alignment-validation.md) for the tested models and datasets.

#### `meta.json` schema

```json
{
  "item_id": "test_Physics_353",
  "source": "mmmu-pro-standard-10",
  "category": "Physics",
  "n_prefill": 218,
  "n_answer": 698,
  "input_tokens_len": 916,
  "num_images": 1,
  "sum_vision_tokens": 176,
  "max_vision_tokens": 176,
  "model": "Qwen/Qwen3.5-4B",
  "image_processor_config_hash": "...",
  "image_processor_config": { /* full processor config from the dataset row */ },
  "image_token_limits": {
    "image_min_tokens": 4,
    "image_max_tokens": 16384
  },
  "ref_answer": "C",
  "generated_texts_preview": "We are given a circuit ...",
  "dataset_seed": 42
}
```

`collect_kld.py` reads `n_prefill` and `n_past_expected`. The full `image_processor_config` and derived `image_token_limits` describe generation-time settings. Prepared image geometry is enforced by `image_preprocessing`; the global scorer token bounds do not resize prepared images.

#### Validation (prep aborts non-zero on any of these)

- `len(input_ids) != input_tokens_len`
- `n_prefill_tokens` out of range for `input_tokens_len`
- `n_prefill + n_answer != input_tokens_len`
- labels disagree with `input_ids`: `labels[:n_prefill]` must all be `-100`
  and `labels[n_prefill:]` must equal `input_ids[n_prefill:]`
- `<__media__>` marker count after decode-and-collapse != `num_images`
- model family not registered in `MODEL_FAMILIES` (longest-prefix match;
  currently `Qwen/Qwen3-VL`, `Qwen/Qwen3.5`, `Qwen/Qwen3.6`, `google/gemma-4`,
  `moonshotai/Kimi-VL`, and `zai-org/GLM-4.6V`)

#### How `formatted_chat.txt` is constructed (decode-and-collapse)

```python
raw = tokenizer.decode(input_ids[:n_prefill], skip_special_tokens=False)
formatted = re.sub(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>",
                   "<__media__>", raw)
```

(The image-block regex is per-family; the Qwen form is shown.) This is
lossless w.r.t. the HF tokenizer and avoids needing to reconstruct the
original chat-template messages list (which the dataset does not store).

The `<|vision_start|>`/`<|vision_end|>` wrappers are swallowed together with
the pad run because mtmd owns them at eval time: `add_media` injects its own
wrapper pair around the image embeddings, so wrappers left in the text would
be duplicated (2 extra tokens per image vs the HF/vLLM conditioning). Dirs
collected before this strip are recorded as `media_wrapper: "doubled"`
(missing field = legacy = doubled); such a dir can neither be extended with
a new shard nor compared against current `"stripped"` dirs.

### What `llama-vlm-kld` actually consumes — the prep artifacts, not the HF dataset

`tools/skymizer/vlm-kld.cpp` does **not** read the HuggingFace dataset
directly. Every row is first translated by `prep_vlm_score_from_hf.py` into
the four files above; the scorer then consumes those artifacts plus one
integer (`n_prefill`) — once per side, through both (model, mmproj) pairs.

| Scorer input | Source from prep | Role |
|---|---|---|
| `--image img_{i}.png` (repeatable, N images) | `img_0.png` … `img_{N-1}.png` | One PNG per `<__media__>` marker; each side's mtmd embeds them via its own projector (`--ref-mmproj` / `--cand-mmproj`) |
| `--formatted-chat formatted_chat.txt` | `formatted_chat.txt` | Conditioning prompt with `<__media__>` markers; mtmd re-tokenizes it and segments it into text + image chunks |
| `--tokens-in tokens.bin` | `tokens.bin` (vLLM's `input_ids` as `int32[L]`) | Source of the **answer tokens** that are teacher-forced |
| `--n-prefill <int>` | `meta.json["n_prefill"]` (forwarded by `collect_kld.py`) | Slice index — answer tokens are `tokens_full[n_prefill:]` |
| `--image-metadata meta.json` with `--images-preprocessed` | `meta.json` | Required geometry and reference-prefix checks for aligned input |
| `--output-metrics <path>` | (output) | VLMK binary metric dump |

In `--manifest` mode each row's inputs and output path are packed into one JSONL
entry; the contract is identical, just batched so the two models + mmprojs
load once for the whole sweep.

**Two-phase forward pass**, executed for BOTH sides per row:

1. **Prefill** - mtmd tokenizes the saved prompt and inserts image embeddings. Prepared RGB geometry is preserved and every text/image chunk is checked against the reference prefix before evaluation. Gemma and Kimi use sequential positions, so `n_past_actual == n_prefill`. Qwen and GLM use M-RoPE, so their expected position count is the number of text tokens plus the sum of each image's longest merged-grid side; equal position counts alone do not prove equal grids.

2. **Teacher-forced answer scoring** — the scorer takes
   `tokens_full[n_prefill:]` as the answer tokens, `llama_decode`s them
   through both models in `--tf-chunk`-sized chunks, and computes the
   per-position metric record from the two full-vocab logit rows while both
   are in memory. Capped to `--num-eval-tokens` if set.

**Why the split:** the answer tokens come straight from the dataset's
`input_ids` — vLLM's tokenization, frozen at ground-truth generation time —
so the candidate is scored against the **same** target sequence as the
reference. In aligned mode, prefix tokenization and image-grid mismatches
abort scoring. This removes those input differences from the quantization
comparison; it does not establish numerical equivalence between llama.cpp and
vLLM. `meta.json.image_preprocessing` supplies the scorer's geometry contract,
while the original processor settings and answer metadata remain available
for downstream analysis.

**Performance / teacher-forcing batching.** The answer tokens are known
ground-truth, so they are teacher-forced in chunks (one `llama_decode` per
`--tf-chunk` tokens, default `n_ubatch`) instead of one decode per token — ~5–6×
faster on decode-heavy rows. Because batched GPU kernels (M>1) accumulate in a
different order than the single-token path (M=1), the **batched logits differ
from per-token by floating-point non-associativity** (~1.5% of near-tie argmaxes
flip; on K=1024 KLD this perturbation is ~6% of the q4-vs-f16 signal). It is not
a bug, but it means **runs whose metrics are compared against each other must be
collected with the same `--tf-chunk`.** Use `--tf-chunk 1` to reproduce
pre-batching goldens bit-for-bit; otherwise collect every pair of a sweep with
the same (batched) setting so the comparison stays self-consistent.

## `collect_kld.py` + `llama-vlm-kld` — the metrics collector

`llama-vlm-kld` loads BOTH a reference (model, mmproj) and a candidate
(model, mmproj) at once, teacher-forces each row's answer tokens through
both, and computes per-answer-token metrics from the two full-vocab logit
rows while they are in memory. Only the metrics hit the disk: one packed
44-byte record per position, ~44 KiB/row at `npos=1024` — every metric is
full-vocabulary by construction, and no logits are ever stored.

```
dataset ─collect_kld (ref=BF16, cand=Q4_K_M·mmF16)─► <out_a>/metrics/*.npz
dataset ─collect_kld (ref=BF16, cand=Q4_K_M·mmQ80)─► <out_b>/metrics/*.npz
                                  └──► numpy / torch postprocessing
```

```bash
python3 tools/skymizer/cli/collect_kld.py \
    --ref-model   models/bf16/Qwen_Qwen3.5-4B-bf16.gguf \
    --ref-mmproj  models/bf16/mmproj-Qwen_Qwen3.5-4B-bf16.gguf \
    --cand-model  models/q4km/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
    --cand-mmproj models/bf16/mmproj-Qwen_Qwen3.5-4B-bf16.gguf \
    --out tmp/kld-q4km/ \
    --dataset elichen-skymizer/GAP-mmmu-pro-standard-10 \
    --subset qwen3.5-4b-ins-gen-2048 --split train \
    [--dataset-limit 50] [--num-eval-tokens 1024] [--tf-chunk 16] \
    [--start 0] [--end N|-1] [--n-batch 2048] [--n-ubatch 2048] \
    [--image-min-tokens -1] [--image-max-tokens -1] [--n-ctx 32768] \
    [--n-gpu-layers 99] [--n-threads -1] [--metric-threads -1] \
    [--flash-attn | --no-flash-attn] [--swa-full] [--llama-vlm-kld build/bin/llama-vlm-kld] [--keep-prep]
```

The text-only twin is `collect_llm_kld.py` + `llama-llm-kld`: same
lifecycle, same output layout, no images/mmproj/chat reconstruction — it
consumes the dataset's `input_ids`/`labels`/`n_prefill_tokens` directly.

**How it scores (manifest mode):** the script preps every requested row
in-process via `prep_vlm_score_from_hf.prep_row()` (dataset + tokenizer
loaded once for the whole sweep, not per row), writes a single
`_manifest.jsonl`, then calls `llama-vlm-kld` **once** with `--manifest`. The
scorer loads the two models + mmprojs a single time and iterates the rows,
clearing the KV caches between them — so the per-row model-reload cost is
paid once per sweep instead of once per row. After scoring, each row's dump
is validated (header, on-disk size, target IDs, and every stored metric
column — a scorer killed mid-write is recorded FAIL and its dump kept as
`.rejected` evidence rather than recorded as done), then losslessly
converted to `.npz`.

Per-token record fields (all derivations downstream are trivial):

| Field | Type | Meaning |
|---|---|---|
| `kld` | f32 | Forward `KL(p_ref ‖ p_cand)`, nats |
| `reversed_kld` | f32 | Reverse `KL(p_cand ‖ p_ref)`, nats |
| `js_kld` | f32 | Jensen-Shannon divergence, nats |
| `nll_ref` / `nll_cand` | f32 | `−log p(target)` per side; `p(target) = exp(−nll)` |
| `entropy_ref` / `entropy_cand` | f32 | Self entropy per side, nats |
| `ear` | f32 | Expected Acceptance Rate: `Σ_v min(p_ref, p_cand)` = `1 − TV` ([arXiv:2605.02404](https://arxiv.org/abs/2605.02404)); VLMK v2 dumps only |
| `target` | i32 | Teacher-forced target token id |
| `argmax_ref` / `argmax_cand` | i32 | Per-side argmax; `same_top = (argmax_ref == argmax_cand)` |

All metrics are accumulated in float64 over the full vocab inside the C++
scorer (`--metric-threads` parallelises the kernel across a chunk's
positions) and stored as float32.

Each row lands as `metrics/<idx>_<item_id>.npz` — directly `np.load`-able,
no repo imports needed:

```python
import numpy as np
z = np.load("tmp/kld-q4km/metrics/000_test_X.npz")
z["kld"].mean()                                   # mean forward KLD
(z["argmax_ref"] == z["argmax_cand"]).mean()      # same_top rate
np.exp(-z["nll_cand"]) - np.exp(-z["nll_ref"])    # delta-p at target
# header fields: int(z["vocab"]), int(z["npos"]), int(z["n_prefill"])
```

Output layout under `--out`:

```
collect_meta.json                  pair/dataset identity (read by
                                   downstream tools; shard identity)
manifest.csv                       row_idx, item_id, num_images, n_prefill,
                                   n_answer, n_eval, vocab, metrics_bytes,
                                   wall_s, status
metrics/<idx>_<item_id>.npz        VLMK metric dump (lossless .npz form)
logs/kld_run.log                   combined prep + scorer stderr for the run
```

**Pairing is fixed at collection time.** A metrics dir describes ONE
(reference, candidate) pair. For an A-vs-B verdict, run `collect_kld.py`
once per candidate against the same reference, then:

```bash
python3 tools/skymizer/cli/saved_metrics_paired_compare.py \
    --candidate-a tmp/kld-q4km-mmf16/ \
    --candidate-b tmp/kld-q4km-mmq80/ \
    --out tmp/paired-q4km.md \
    [--label-a ...] [--label-b ...] [--output-json tmp/paired-q4km.json]
```

It hard-fails unless the two dirs share a **bit-identical reference**: the
per-item `nll_ref`/`entropy_ref`/`argmax_ref` columns must match exactly
between the dirs (this is what "same reference run configuration" actually
buys; `--allow-ref-drift` downgrades the check to a warning and records the
drift magnitude in the report).

## Output-root policy (no resume)

Every requested row is written exactly once; the collectors never skip,
delete or overwrite. Before prep or scoring the run takes an exclusive lock
on `--out` (a second collector on the same root is refused), requires
`manifest.csv` (if present) to carry exactly the current column schema (no
older-layout tolerance; a torn last line refuses), and scans the requested
`--start/--end` window for ANY prior output: a `manifest.csv` record (`OK`,
`SKIP_OVER_BUDGET`, or a `FAIL_*` whose row left files behind — a `FAIL_*`
row with NO artifact at all is a transient failure and is retried in place
instead), or any entry under `metrics/` or `_prep/` whose name starts with
the row's `{idx:03d}_` stem prefix — dumps, sidecars, `.tmp` leftovers, prep
dirs, whatever item_id they carry. One hit anywhere in the window refuses
the whole run with a listing ("Nothing was deleted or overwritten"); the
remedies are a fresh `--out` or a window fully disjoint from the rows
already collected there.

Disjoint windows may be collected sequentially into one `--out`. The first
shard writes `collect_meta.json` (identity + provenance + the dataset
content hash and model fingerprints); a later shard must match the dir's
identity fields exactly — `kind`, `vlmk_version`, `ref_model`, `ref_mmproj`,
`cand_model`, `cand_mmproj`, `dataset`, `subset`, `split`, `sort_by`,
`num_eval_tokens`, `max_total_tokens`, `image_min_tokens`,
`image_max_tokens`, `tf_chunk`, `n_ctx`, `n_batch`, `n_ubatch`,
`n_gpu_layers`, `n_threads`, `metric_threads`, `flash_attn`,
`media_wrapper` (no legacy defaults: a dir whose meta lacks a field cannot
be extended). Every shard also re-derives and verifies the CONTENT tiers
against shard 1's record: the whole-dataset content hash (it pins the HF
dataset/subset/split by content and does not depend on which
`--start/--end` window a shard collects) and the model-file fingerprints —
so a model file replaced IN PLACE at the same path, or a dataset
regenerated under the same name, refuses the shard. A meta that predates
those fields warns and proceeds (legacy dirs). An empty requested window
(`--start` past the end, or `--start >= --end`) is refused rather than
reported as a completed run. A crashed run leaves its partial rows as
collisions: clear that row's files and manifest line by hand, or use a
fresh `--out`. A `FAIL_*` row that left no artifact (a transient prep
failure: a truncated image in the HF cache, an OOM) is the one retryable
case — re-running its window re-collects it, appending a second record that
the last-wins readers resolve; a `FAIL_*` row that did leave files (e.g. a
rejected KLD dump kept as evidence) still collides until an operator clears
it.

Two consequences worth knowing, both by design:

- **A lost manifest line is not repaired.** The artifact is committed
  (atomic rename) before its manifest row is appended, so a kill in that
  window leaves a valid dump with no manifest record: the glob-driven
  comparator (`saved_metrics_paired_compare.py`) still sees the item, while
  its manifest-driven budget-skip check does not. Re-collecting that row is
  refused, so fix it by hand (remove the row's artifacts and its manifest
  line, then re-collect the single-row window) or use a fresh `--out`.
- **A rejected KLD dump is renamed, not deleted.** Validation failures leave
  `metrics/<stem>.bin.rejected`, which no `*.bin`/`*.npz` glob picks up, so a
  rejected dump can never be consumed as data or promoted to an
  indistinguishable `.npz`; the row still counts as a collision. Wrapper
  scripts that used to "increase END and rerun" must instead pass a
  disjoint `--start`.

## Numerics / comparability

The reference side's logits depend on decode batching (FP
non-associativity), so runs whose metrics are compared against each other
MUST share `--tf-chunk` / `--n-ubatch` / `--n-batch`, context size,
offload/thread settings, flash-attention mode, and the SWA cache mode
(`--swa-full`) (enforced per-dir via
`collect_meta.json` identity fields when a later shard is added). Both
models must share one tokenizer: startup validates vocabulary size and
type, then checks every token ID's text and attributes for an exact mapping
match. `--allow-vocab-attr-mismatch` (scorer flag, forwarded by
`collect_kld.py`) downgrades only the attribute half to a logged warning,
for same-base-model conversions whose metadata disagrees on token types /
which id is `<eos>` (e.g. bartowski vs unsloth gemma-4-E4B-it differ solely
at token id 1); token texts remain strictly enforced. KV/VRAM cost is two
models + two mmprojs + two contexts resident at once.

## Validation

`llama-vlm-kld --self-test` checks the metric kernel against closed-form
and naive-reference values (no models needed), including a
**production-regime** case: the full 151,936-slot vocabulary against a
near-identical candidate (σ = 0.05), which lands at KLD ≈ 1.2e-3 nats and
EAR ≈ 0.98 — where real runs live. That case compares the kernel's `double`
accumulators against the naive implementation with a **relative** tolerance
(1e-9), because the older absolute tolerances were picked at a test point with
KLD ≈ 13.8 nats and EAR ≈ 0.07 (two unrelated distributions) and are 10³–10⁴×
looser than the production numbers; and because a regression in `ear` moves it
by less than 5 ulp of the stored float32, so it is invisible in the record.
Measured: the two double implementations agree to ≤ 1e-12 relative, while a
float accumulator lands 2.4e-6 (kld) to 8.6e-5 (entropy) off.
