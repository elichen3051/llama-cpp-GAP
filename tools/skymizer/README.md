# Skymizer — VLM/LLM KLD metrics & paired comparison

Generate reference datasets and evaluate quantization fidelity with llama.cpp.
Operator handovers: [RunPod reference generation](docs/reference-runpod-handover.md), [AWS VLM KLD collection](docs/kld-aws-handover.md), and [AWS llama-perplexity / LLM KLD collection](docs/perplexity-llm-kld-aws-handover.md). InternVL, GLM and Muse use the [final-evaluation reference handover](docs/reference-runpod-final-handover.md).

The text bridge prepares native 512-token corpus windows, aligns full-window PPL/LLM-KLD execution, verifies original and quantized likelihoods, and supports paired window/article/contiguous-block reports. See the [text collection protocol](docs/perplexity-llm-kld-aws-handover.md) and [grouped statistical assumptions](docs/compare.md#fixed-text-corpora-and-grouped-inference).
`llama-reference` renders GGUF chat templates, tokenizes with the GGUF vocabulary,
and generates text/image trajectories directly with llama.cpp and mtmd.
[Native reference generation](docs/reference.md) needs neither llama-server nor an HF tokenizer.
Reference generation supports autoregressive decoding and upstream MTP heads (`--spec-type draft-mtp`, with `--model-draft` for a sidecar). KLD collectors use ordinary teacher forcing and consume the resulting local dataset or a legacy Hub dataset.

> [!NOTE]
> **What this measures, precisely: text-token fidelity CONDITIONED on an
> image** — not per-vision-token fidelity. Every record sits at a
> post-prefill *answer* position, because a decoder-only VLM has no
> well-defined target distribution at an image-embedding position, so an
> upstream-style per-image-token KLD is not available even in principle.
>
> The vision encoder is genuinely run: each side builds a real mtmd context
> and its image embeddings condition every scored logit. So a paired A/B
> that differs **only** in the projector does isolate the projector's
> contribution — see [What gets compared](docs/compare.md#what-gets-compared) — and that is
> a measurement `llama-perplexity` cannot make at all.

## Directory layout

```
tools/skymizer/
├── scripts/                the numbered pipeline (start here)
│   ├── 00_env.sh           shared config: models, datasets, knobs
│   ├── 01_save_vlm_kld.sh  collect VLM metrics (ref-vs-A, ref-vs-B)
│   ├── 02_save_llm_kld.sh  collect LLM metrics (optional text lane)
│   ├── 03_paired_test_kld.sh [vlm|llm]   the paired A-vs-B report
│   └── 04_power_analysis.sh  [vlm|llm]   optional sample-size planning
├── cli/                    the Python entry points the scripts drive
│   ├── generate_reference.py     native GGUF reference dataset generator
│   ├── generate_model_reference.py  pinned model/source/mode launcher
│   ├── run_reference_campaign.py    shared GPU queue and archived attempts
│   ├── upload_reference.py          audited atomic Hub publication
│   ├── restore_reference_models.py  verified BF16 model restoration
│   ├── collect_kld.py            VLM collector (one (ref, cand) pair)
│   ├── collect_llm_kld.py        LLM collector
│   ├── saved_metrics_paired_compare.py   two metric dirs → report
│   ├── prep_vlm_score_from_hf.py         one dataset row → scorer inputs
│   ├── prep_llm_score_from_hf.py
│   ├── power_analysis.py                 prospective N × token-cap planning
│   ├── random_subsample_power.py         observed-effect / reproducibility diagnostic
│   └── variance_decomposition.py         empirical var/N-vs-cap curve; tokens-vs-items
├── lib/                    shared support modules
│   ├── collect_core.py           the one collection lifecycle (spec-driven)
│   ├── collect_common.py         lock / manifest / collision / identity guards
│   ├── collect_vision.py         vision-token budget reporter
│   ├── collect_meta_provenance.py  build/GPU provenance block
│   ├── dataset_fingerprint.py    ds-v3 content hash + model fingerprints
│   └── kld_metrics_io.py         VLMK reader/writer (.bin/.npz)
├── compare/                the statistics/report engine
│   (contracts, student_t, inference, tokens, engine, render, cli_common)
├── reference.cpp                native batch producer and --describe
├── vlm-kld.cpp llm-kld.cpp skymizer-common.h skymizer-vlmk-kernel.h
├── tests/                  hermetic pytest suite + vendored real GAP rows
├── review-functionality/   manual GPU smokes & integration checks
├── docs/  knowledge/       reference docs; teaching notes + the SOP
└── outputs/                gitignored run outputs
```

## The pipeline, end to end

One experiment answers: *is quantization A statistically closer to the
reference than quantization B?* It is two mandatory steps (collect, then
compare) plus optional planning, all driven by `scripts/`:

```
                 ┌──────────────────────────────┐
   GAP dataset ─►│ 01  collect_kld  ref-vs-A    │─► outputs/vlm-kld-ref-vs-a/metrics/*.npz
   + GGUF pairs  │     collect_kld  ref-vs-B    │─► outputs/vlm-kld-ref-vs-b/metrics/*.npz
                 └──────────────────────────────┘
                                 │  (bit-identical reference columns required)
                                 ▼
                 ┌──────────────────────────────┐
                 │ 03  saved_metrics_paired_    │─► paired-…-A-vs-B.md / .json
                 │     compare  A vs B          │    (verdict + CIs + tails)
                 └──────────────────────────────┘
                                 │
                                 ▼ (optional)
                 ┌──────────────────────────────┐
                 │ 04  sequential-prefix power │─► N × token-cap power / MDE surface
                 │     + cap diagnostics        │
                 └──────────────────────────────┘
```

`llama-vlm-kld` loads BOTH (model, mmproj) pairs at once, teacher-forces the
same answer tokens through both, computes full-vocabulary per-token metrics
while the two logit rows are in memory, and stores ONLY the metrics
(~44 B/position — ~15 MiB per dir on a 500-row dataset). No logits ever hit
the disk. The text lane (`02`, `llama-llm-kld`) is the same shape without
images/mmproj.

### Step 0 — build & environment (once per machine)

```bash
# everything at once: apt deps, uv, CUDA/CPU build, venv, smoke checks
tools/skymizer/env_setup.sh

# or by hand:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-reference llama-vlm-kld llama-llm-kld -j
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project tools/skymizer --python 3.12 --group dev
source .venv/bin/activate
```

Requirements: Python 3.12+, GGUF weights (+ mmproj for the VLM lane), and a
ground-truth dataset from the `elichen-skymizer` GAP collection
(`GAP-mmmu-pro-standard-10`, `GAP-mmstar`, `GAP-mmmu-pro-vision`,
`GAP-ocrbench-v1/v2` — auto-cached from the hub on first run). The dataset
config's generating model family must match the GGUFs; registered families:
Qwen3-VL, Qwen3.5, Qwen3.6, Gemma-4, Kimi-VL (`cli/prep_vlm_score_from_hf.py
MODEL_FAMILIES`). Kimi-VL's tokenizer is repository code (`tiktoken` +
`blobfile`; prep passes `trust_remote_code` for that family only). Legacy
HF/vLLM dataset prep needs `uv sync --extra hf-tokenizer`. Native reference
rows use their saved GGUF prompt and raw image bytes, without this extra or a
registered HF model family.

### Step 1 — configure the experiment

Edit `scripts/00_env.sh` (or override per invocation; every value is an env
var): reference (model, mmproj), the two candidates, dataset config, and the
knobs. **Hold constant whatever you are not measuring** — same mmproj
everywhere isolates LLM quantization; same LLM GGUF with varying mmproj
isolates the projector (docs/compare.md "What gets compared").

### Step 2 — collect metrics (GPU)

```bash
./tools/skymizer/scripts/01_save_vlm_kld.sh          # VLM lane
./tools/skymizer/scripts/02_save_llm_kld.sh          # LLM lane (optional)
```

Each run of the collector sweeps ONE (reference, candidate) pair over the
dataset: per-row prep (images + `tokens.bin` + `formatted_chat.txt` from the
frozen `input_ids`), one scorer process over a JSONL manifest (models load
once), strict per-row validation, lossless `.npz` conversion. Both A and B
runs must share the same reference and knobs — `collect_meta.json` records
the full identity and later shards / the comparator enforce it. Collectors
never resume, skip, or overwrite; re-running an already-collected window is
refused (extend with a disjoint `KLD_START`, or use a fresh `OUT_ROOT`).

Direct CLI equivalent (what the script runs):

```bash
python3 tools/skymizer/cli/collect_kld.py \
    --ref-model   .../Qwen_Qwen3.5-4B-bf16.gguf \
    --ref-mmproj  .../mmproj-Qwen_Qwen3.5-4B-bf16.gguf \
    --cand-model  .../Qwen_Qwen3.5-4B-Q4_K_M.gguf \
    --cand-mmproj .../mmproj-Qwen_Qwen3.5-4B-bf16.gguf \
    --dataset elichen-skymizer/GAP-mmmu-pro-standard-10 \
    --subset  qwen3.5-4b-ins-gen-2048 \
    --num-eval-tokens 2048 \
    --out outputs/vlm-kld-ref-vs-a
```

### Step 3 — the paired report (CPU, seconds)

```bash
./tools/skymizer/scripts/03_paired_test_kld.sh vlm    # or: llm
```

Loads the two metric dirs, verifies the stored reference columns are
bit-identical per item (the shared-reference premise; `--allow-ref-drift`
downgrades to a recorded warning), aggregates per-item scores, and emits the
markdown + JSON report: one pre-registered confirmatory endpoint (KLD ×
item-weighted, paired Student-t CI by default), every other cell exploratory
with Holm-adjusted p-values, per-item p99/p99.9/max KLD tails with witness
positions, pooled per-token ladders, and the full execution/identity
metadata. Use the JSON, not markdown scraping, for downstream tables.

`--num-eval-tokens N` at compare time derives prefix reports from an
existing collection without rescoring (collection cap must be ≥ compare cap).

### Step 4 (optional) — planning / power

```bash
./tools/skymizer/scripts/04_power_analysis.sh vlm
```

`power_analysis.py` treats each `--num-eval-tokens` value as a distinct
ordered-prefix estimand. It resamples whole paired items, never tokens. Supply
a signed, externally chosen `SESOI` for prospective power; without one it emits
precision and MDE only. `variance_decomposition.py` is an observed cap-profile
diagnostic, and `random_subsample_power.py --mode reproducibility` measures
finite-pilot verdict stability rather than power. See
[`knowledge/seq-power-analysis.md`](knowledge/seq-power-analysis.md).

### Verifying a setup

- `build/bin/llama-vlm-kld --self-test` / `llama-llm-kld --self-test` —
  metric-kernel closed forms + independent naive reference; no models needed
  (step 01/02 run it automatically).
- `review-functionality/smoke_vlm_gemma4.sh` — GPU end-to-end on real GAP
  rows: double collection, metric-column determinism, self-pair report with
  exact-zero deltas.
- `review-functionality/check_manifest_mode.py` — GPU check that
  `--manifest` mode is byte-identical to single-row scoring.
- `python3 -m pytest -q` from `tools/skymizer/` — the hermetic suite,
  including the bit-exact engine golden (see `NUMERICAL_CONTRACT.md`).

## Documentation

- [`docs/upstream-port.md`](docs/upstream-port.md) - upstream integration and reproducible local GPU smoke tests
- [`docs/collect.md`](docs/collect.md) — prep, the scorer contract, `cli/collect_kld.py`, the output-root policy
- [`docs/compare.md`](docs/compare.md) — the report: metrics, interval construction, multiplicity, strata, per-token ladders, what gets compared
- [`docs/formats.md`](docs/formats.md) — the VLMK on-disk layout
- [`docs/gotchas.md`](docs/gotchas.md), [`docs/troubleshooting.md`](docs/troubleshooting.md)
- [`NUMERICAL_CONTRACT.md`](NUMERICAL_CONTRACT.md) — what a refactor must not change
- `knowledge/` — teaching notes and the pre-registered evaluation SOP

## Limitations / out of scope

- **Registered model families only.** Qwen3-VL, Qwen3.5, Qwen3.6, Gemma-4,
  and Kimi-VL image wrappers are wired into `cli/prep_vlm_score_from_hf.py`; a new family
  needs its own image-pad regex plus end-to-end validation on real
  collection rows (docs/troubleshooting.md walks through it).
- **No audio.** The prep script and datasets assume images.
- **`image_processor_config_hash` is recorded but not enforced.**
  llama.cpp's `clip` preprocessor may produce a different vision-token
  count than HF's processor; this drift is accepted. Same-model KLD
  measurements stay valid because both sides use the same llama.cpp
  preprocessor.
- **Pipeline assumes a deterministic backend.** Same-build double-runs are
  bit-exact on CUDA; `--n-seq-max > 1` and different `--tf-chunk` values
  legitimately perturb logits (FP non-associativity) and are identity-guarded.

## Tests

```bash
cd tools/skymizer
python3 -m pytest -q                       # full suite
python3 -m pytest -q -m "not statistical"  # fast iteration loop
```

Markers: `statistical` (Monte-Carlo validation), `integration` (runs the
built C++ tools, self-skips without the build), `external_model` (needs a
downloadable tokenizer). The deterministic tests need no GPU or models.

Test data policy: fixtures that stand for dataset rows are REAL collection
samples vendored under `tests/data/` (currently the first two rows of
`GAP-mmmu-pro-standard-10 :: qwen3.5-4b-ins-gen-2048` and three rows of
`kimi-vl-a3b-ins-full-ref-text-rec-dec :: ocrbench-v1-subsample-100-ins`
(the two smallest images and one over the derived vision cap), saved verbatim with
`datasets.Dataset.save_to_disk`) — not script-generated look-alikes. Binary
metric fixtures (VLMK records) are synthesized in-memory by the tests
themselves.

The dev environment pins `numpy>=2.5.2`: the bit-exact goldens were
generated under numpy 2.x and fail under numpy 1.x promotion rules.
