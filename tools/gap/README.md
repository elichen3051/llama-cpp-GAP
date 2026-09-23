# GAP: Reference Trajectories and VLM Fidelity Metrics

This guide describes how to generate reference response trajectories and collect token-level fidelity metrics for a candidate vision-language model (VLM). A reference model first generates a response to each question and its images. The collector then feeds the saved response tokens to both the reference and candidate models, comparing their next-token distributions at each response position.

The workflow ends with a completed metric collection for downstream use. It uses the direct local-data entry points in this source tree; model profiles and dataset publication are not required.

```text
Prepared questions and images + reference GGUF and projector
  -> generate_reference.py -> llama-reference
  -> Frozen reference dataset with response token IDs
  -> collect_kld.py -> llama-vlm-kld (+ candidate GGUF and projector)
  -> Per-response-token metrics, manifest and provenance
```

## 1. Environment and build

Run all commands from the repository root, in the same Bash session. The commands below target a Linux CUDA host with a compatible NVIDIA driver and CUDA toolkit, a C++17 compiler, CMake, and `uv`. Python dependencies are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock); the Python version is 3.12.3.

`GAP_WORK` is the directory for the Python environment, compiled binaries, caches and experiment outputs. The example below uses `$HOME/gap-work`, a directory under your own home directory, and creates it automatically. Keep it outside the source tree. If your home disk has limited space, use a writable directory on a larger data disk, such as `/mnt/data/gap-work`. Replace the `/path/to/...` dataset and model paths in the later steps with your actual inputs.

```bash
export GAP_WORK="$HOME/gap-work"
export TMPDIR="$GAP_WORK/tmp"
export UV_CACHE_DIR="$GAP_WORK/uv-cache"
export UV_PROJECT_ENVIRONMENT="$GAP_WORK/venv"
export HF_HOME="$GAP_WORK/hf"
export CUDA_CACHE_PATH="$GAP_WORK/cuda-cache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" "$CUDA_CACHE_PATH"

uv sync --project tools/gap --python 3.12.3 --locked
export GAP_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export GAP_BIN="$GAP_WORK/build/bin"

cmake -S . -B "$GAP_WORK/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build "$GAP_WORK/build" --target llama-reference llama-vlm-kld -j8
```

The two executables are defined in [CMakeLists.txt](CMakeLists.txt). Keep their shared libraries with the build. Collection records the executed binary and loaded-library hashes, so use the same build throughout a candidate comparison.

The metric-kernel self-test can check the scorer without loading model weights:

```bash
"$GAP_BIN/llama-vlm-kld" --self-test
```

Generation loads the reference model and its projector. Collection loads the reference and candidate models, both projector instances, and their context buffers together. Choose a GPU with enough memory for that pair. Run one collection process per GPU.

## 2. Prepare source data and model files

Use a local Hugging Face Datasets directory created with `Dataset.save_to_disk()` or `DatasetDict.save_to_disk()`. For a `DatasetDict`, the commands below select its `train` split. The source contains questions and images; the next step adds the reference responses.

| Column | Expected contents |
| --- | --- |
| `question` | A string containing one user question. It may be empty when images are present. |
| `images` | An ordered list of encoded images, represented as `Sequence(Image(decode=False))`. Embed the original image bytes for a portable dataset. |
| `item_id` | Recommended: a unique string using letters, digits, `.`, `_` and `-`, excluding `.` and `..` as entire IDs. |

The generator uses `item_id`, then `id`, then a generated row index as its identifier. It accepts `--question-column`, `--images-column` and `--id-column` for other column names. Rows with neither question text nor images are rejected. Images precede the question in the rendered user message; preserve their order and original encoded bytes.

Set the actual dataset and model paths:

```bash
export SOURCE_DATASET=/path/to/prepared-source-dataset
export REF_MODEL=/path/to/reference-bf16.gguf
export REF_MMPROJ=/path/to/reference-mmproj.gguf
export CAND_MODEL=/path/to/candidate-quantized.gguf
export CAND_MMPROJ=/path/to/candidate-mmproj.gguf
export CUDA_VISIBLE_DEVICES=0
```

If a prepared dataset is not available, the following example creates a one-question dataset at `SOURCE_DATASET`, which must be a new directory. Replace the image path and question, and extend `rows` with the intended evaluation inputs. Skip this block when using an existing dataset.

```bash
"$GAP_PYTHON" - <<'PY'
import os
from pathlib import Path
from datasets import Dataset, Features, Image, Sequence, Value

rows = [{
    "item_id": "sample-0001",
    "question": "What is shown in the image?",
    "images": [{"bytes": Path("/path/to/image.png").read_bytes(), "path": None}],
}]
features = Features({
    "item_id": Value("string"),
    "question": Value("string"),
    "images": Sequence(Image(decode=False)),
})
Dataset.from_list(rows, features=features).save_to_disk(os.environ["SOURCE_DATASET"])
PY
```

Supply a reference GGUF supported by this checkout and its matching vision projector, plus a candidate derived from the same base checkpoint and the candidate's own matching projector. Use `REF_MODEL` and `REF_MMPROJ` for reference generation and the reference side of collection. Use `CAND_MODEL` and `CAND_MMPROJ` for the candidate side of collection. For split GGUF models, supply the first shard and keep all shards together. Preserve complete checksums and source versions for all model files and the input dataset.

The models must share the vocabulary type, size and token-ID-to-text mapping. The scorer also checks token attributes. An equal vocabulary size alone is insufficient. These native reference trajectories use GGUF tokenization and the saved prompt; the optional `hf-tokenizer` dependency is unnecessary for this workflow.

## 3. Generate the reference dataset

Choose the context, batch sizes, generation cap and sampling settings for the selected model and evaluation protocol. The following values are an example configuration, not model-specific reproduction settings. The context must fit each multimodal prompt plus its generation allowance. Complete non-causal image chunks must fit both batch and microbatch sizes; increase these values when the model's image layout requires it.

```bash
export N_CTX=32768
export N_BATCH=2048
export N_UBATCH=2048
export N_THREADS=8
export GEN_TOKENS=1024
export REFERENCE_RUN="$GAP_WORK/reference/run-001"

"$GAP_PYTHON" tools/gap/cli/generate_reference.py \
  --dataset "$SOURCE_DATASET" --split train \
  --out "$REFERENCE_RUN" \
  --llama-reference "$GAP_BIN/llama-reference" \
  --no-enable-thinking -- \
  -m "$REF_MODEL" --mmproj "$REF_MMPROJ" \
  -ngl all -c "$N_CTX" -b "$N_BATCH" -ub "$N_UBATCH" \
  -t "$N_THREADS" -tb "$N_THREADS" -np 1 -fa on --fit off \
  -ctk f16 -ctv f16 -n "$GEN_TOKENS" \
  --seed 1234 --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0
```

`REFERENCE_RUN` must not already exist. Python driver options precede `--`; native model and sampling options follow it. This example uses one sequence and autoregressive generation. Adapt sampling settings to the protocol and retain the recorded effective sampler settings. For a model/template with thinking support, replace `--no-enable-thinking` with `--enable-thinking` when required; thinking tokens remain part of the trajectory and count toward the generation cap. An optional `--system-prompt` belongs before `--`.

For an initial small run, add `--num-samples 2` before `--` and use a separate output directory. It selects the first two source rows. Create a fresh directory for the full run.

| Output under `REFERENCE_RUN` | Contents |
| --- | --- |
| `dataset/` | Eligible reference rows, including the prompt, original images and full token sequence. Absent if no rows are eligible. |
| `metadata.json` | Effective runtime, sampling, model identities and cohort counts. |
| `complete.json`, `run_state.json` | Completion status and eligible/excluded/failed row accounting. |
| `requests.jsonl`, `inputs/` | Prepared requests and original encoded input images. |
| `excluded.jsonl`, `failures.jsonl` | Repetition exclusions and generation/preparation failures. |
| `attempts/`, `scripts/` | Native attempt logs/results and a snapshot of the generator sources. |

Check `complete.json` before collecting metrics. Require `status` to be `complete`, `rows` to be positive, and `dataset/` to exist. The driver can finish with `complete_with_failures`; a successful process exit or the presence of `complete.json` alone does not prove every requested row succeeded. Resolve failures before a full-cohort collection and retain their original records.

Repetition detection excludes an entire row and records the reason. Failed and excluded rows are not replaced automatically. Non-repetitive responses that reach the generation cap remain eligible and are marked `truncated_by_cap`. Preserve these counts and the selected row identities with the experiment.

The native schema is defined in [lib/reference_dataset.py](lib/reference_dataset.py) and [lib/reference_contract.py](lib/reference_contract.py). `input_ids[n_prefill_tokens:]` is the frozen response trajectory, including any generated end-of-generation token. The dataset also retains the exact rendered prompt, image layout, vocabulary identity and raw reference token log-probabilities. Keep the full generated dataset intact for collection.

## 4. Collect VLM KLD along the response trajectory

For each saved response position, both models receive the same prompt, images and preceding reference tokens. The collector compares their full-vocabulary next-token distributions using teacher forcing. The candidate does not generate a separate response. Forward KLD is `KL(p_reference || p_candidate)` in nats at each scored response position.

```bash
export REFERENCE_DATASET="$REFERENCE_RUN/dataset"
export COLLECTION_OUT="$GAP_WORK/metrics/candidate-001"
export SCORING_CAP=-1
export TF_CHUNK=2048

"$GAP_PYTHON" tools/gap/cli/collect_kld.py \
  --dataset "$REFERENCE_DATASET" --subset '' --split train \
  --ref-model "$REF_MODEL" --ref-mmproj "$REF_MMPROJ" \
  --cand-model "$CAND_MODEL" --cand-mmproj "$CAND_MMPROJ" \
  --llama-vlm-kld "$GAP_BIN/llama-vlm-kld" \
  --out "$COLLECTION_OUT" --sort-by num_images \
  --n-ctx "$N_CTX" --n-batch "$N_BATCH" --n-ubatch "$N_UBATCH" \
  --tf-chunk "$TF_CHUNK" --n-threads "$N_THREADS" \
  --metric-threads "$N_THREADS" --n-gpu-layers -2 --flash-attn \
  --num-eval-tokens "$SCORING_CAP"
```

`SCORING_CAP=-1` scores all saved response tokens. A positive value scores at most that many response tokens per item. Shorter responses retain their actual lengths. Collection cannot extend a saved trajectory. Keep the same scoring horizon and teacher-forcing chunk size for every candidate; changing batch shapes can change floating-point results.

The command uses one sequence, F16 KV caches and all supported GPU layers (`--n-gpu-layers -2`). Image-token limits are omitted so the collector adopts the reference dataset's recorded image policy. Prefix/image-position mismatches and vocabulary mismatches are rejected by default. Keep the same context, batch sizes, threads, image policy, flash-attention setting and native build across candidates. If a runtime change is necessary, collect all affected candidates into new output directories using the revised configuration.

For an examined conversion that changes only vocabulary attributes, `--allow-vocab-attr-mismatch` permits those attribute differences while retaining the exact token-text mapping checks. Record and apply that choice consistently across the candidate family. It cannot make different tokenizations compatible.

Use a fresh `COLLECTION_OUT` for each run. For a small collection check, add `--dataset-limit 2` and use a separate output directory. Row selection occurs after the requested sorting. For each additional candidate, set its `CAND_MODEL`, matching `CAND_MMPROJ` and a new `COLLECTION_OUT`, then repeat the command with the same frozen reference inputs and runtime. Use a new output directory after a failed or interrupted collection.

### Completion and saved metrics

A completed collection has a successful collector exit, a reconciled manifest, and `.attempts/*/state.json` records in the `completed` state for the declared work. The command above requests all eligible reference rows without a budget filter: every manifest row should have `status=OK`, the row count should match `complete.json`'s `rows`, and each row should have a corresponding metric file. `FAIL_*` statuses, unfinished attempts, or `.bin.rejected` files mean the collection is incomplete. Review `logs/kld_run.log` and retain failed output for diagnosis.

```text
COLLECTION_OUT/
  collect_meta.json
  manifest.csv
  metrics/<row-index>_<item-id>.npz
  logs/kld_run.log
  .attempts/<attempt-id>/
    request.json
    state.json
    statuses.jsonl
    references.jsonl
    generators/
```

The collector validates the native metric version, vocabulary size, prefill and target counts, exact target token IDs, and finite metric values before converting each native `.bin` file to `.npz`. Successful conversion preserves the arrays and removes the intermediate `.bin`. Keep the entire collection, including hidden `.attempts/` records and `collect_meta.json`; metric files alone lose the completion and provenance evidence.

The current VLMK v5 output contains one value per scored response token for each metric:

| NPZ fields | Meaning |
| --- | --- |
| `kld`, `reversed_kld`, `js_kld` | Forward KL, reverse KL and Jensen-Shannon divergence, in nats. |
| `nll_ref`, `nll_cand` | Negative log-probability of the saved target token under each model. |
| `entropy_ref`, `entropy_cand` | Full-vocabulary entropy of each distribution, in nats. |
| `ear` | Expected acceptance rate, `sum_v min(p_reference(v), p_candidate(v))`. |
| `ear_5`, `ear_10`, `ear_20`, `ear_64` | EAR mass restricted to the reference's top-K tokens. |
| `ear_5_normalized`, `ear_10_normalized`, `ear_20_normalized`, `ear_64_normalized` | EAR after renormalizing both distributions over those reference top-K tokens. |
| `target`, `argmax_ref`, `argmax_cand` | Saved target and each model's most likely token ID. |

Header fields include `version`, `vocab`, `npos`, `n_prefill` and `n_past_actual`. Metric arrays have shape `(npos,)`; `npos` is the number of scored response positions. The calculations are in [core/company-vlmk-kernel.h](core/company-vlmk-kernel.h), and the format reader is [lib/kld_metrics_io.py](lib/kld_metrics_io.py).

To inspect one completed file without aggregating the collection:

```bash
"$GAP_PYTHON" - <<'PY'
import os
from pathlib import Path
import numpy as np

path = sorted((Path(os.environ["COLLECTION_OUT"]) / "metrics").glob("*.npz"))[0]
with np.load(path, allow_pickle=False) as metrics:
    print(path.name)
    print("positions:", int(metrics["npos"]))
    print("KLD shape:", metrics["kld"].shape)
    print("first target IDs:", metrics["target"][:5])
    print("first KLD values:", metrics["kld"][:5])
PY
```

These values measure fidelity to the reference model along its saved response trajectory. They do not measure task-answer accuracy. At this point, the complete metric collection is ready for downstream use.
