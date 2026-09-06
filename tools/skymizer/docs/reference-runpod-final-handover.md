# RunPod references: InternVL, GLM and Muse

Status: bounded GPU acceptance, portable deployment checks and independent gpt-6-astra max review passed on 2026-09-06. Use the accepted source commit in DELIVERY.json or the verified base-plus-patch route below.

This package generates reference datasets only. It covers both pilot100 and collect500 for InternVL3.5-30B-A3B, GLM-4.6V-Flash and Muse-Glimmer-30B. It does not replace the accepted small-model or remaining-SNR packages. For each size, finish the SNR group before starting this final-evaluation group. Candidate validation and KLD collection have separate acceptance requirements.

## Frozen protocol

The upstream implementation base is `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`. These models use its existing architecture, graph, kernels and vision preprocessing. Only Skymizer tooling is changed. Do not add model support or alter upstream preprocessing to make a checkpoint pass.

| Checkpoint | Modes | Model files |
| --- | --- | --- |
| internvl3.5-30b-a3b | instruct; R1 thinking | 2 bartowski BF16 shards + BF16 mmproj |
| glm-4.6v-flash | instruct; thinking | bartowski BF16 main + BF16 mmproj |
| muse-glimmer-30b | thinking/high only | 2 bartowski BF16 shards + BF16 mmproj |

InternVL thinking requires the exact R1 system prompt in `sources/internvl-r1-system-prompt.txt`; the profile supplies it. Ordinary instruct uses temperature 0.8; R1 uses 0.6. GLM uses temperature 0.8, top_p 0.6, top_k 2, min_p 0.05 and repeat_penalty 1.1. Muse uses temperature 1.0, top_p 0.95, top_k 64, min_p 0.05, `reasoning_strength=high` and fixed `current_date=2026-09-06`. Muse's template has no thinking-off branch. Exact sampler dictionaries, official source revisions and all artifact hashes are in the profiles; parameters not recommended by a model card are recorded as retained native defaults.

All lanes use one RTX PRO 6000 Blackwell Server Edition 96 GiB, one generator process, `np=1`, per-sequence context 32768, batch 2048, ubatch 512, threads/threads-batch 8, all GPU layers, Flash Attention on, f16 K/V, fit off, seed 1234 and autoregressive decoding. `swa_full=false` leaves architectural SWA behavior intact. Omit both image-token flags to use mmproj defaults. These large-model checks do not establish safe `np=4` operation.

| Profile | Requested items per source | Instruct cap | Thinking cap |
| --- | ---: | ---: | ---: |
| reference-model-profiles-pilot100.json | 100 | 2048 | 8192 |
| reference-model-profiles-collect500.json | 500 | 1024 | 4096 |

There are 5 modes x 7 sources = 35 jobs per size. Use the matching profile for generation and publication. Profile `kld_eval_tokens` fields are reference-handover placeholders, not approval of the KLD study.

Prepared data is `elichen-skymizer/vlm-prepared-dataset@6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b`, split `train`. Sources: `blink-val`, `mathvision-test`, `mmmu-pro-standard-10`, `mmmu-pro-vision`, `mmstar`, `ocrbench-v1`, `ocrbench-v2`. Their exact configs end in `-subsample-100` or `-subsample-500`. Do not use `main` or override image limits. Pilot may overlap collect500; it is not an independent holdout.

## Setup and source verification

Carry this entire directory to `~/to_runpod_supplements/skymizer-final-reference-20260906`. It contains protocols, checks and source patch, not weights, datasets, credentials or binaries. Use the accepted commit from `source-provenance.json` after the user pushes it. Alternatively, checkout its `base_commit` and apply the bundled patch with the verifier below. Both routes must reproduce the sealed source hashes before building.

Required: Python >=3.12, uv, CMake/C++/CUDA toolchain, AWS CLI, readable prepared HF dataset and S3 objects. HF publication also needs a write token. Confirm the actual NVMe mount and free space with `findmnt` and `df`; the directory name alone does not establish its storage type. Eight model files require 141,960,284,608 bytes, plus caches, build, inputs, Arrow/Parquet copies and outputs. Successful publication does not automatically delete local files.

```bash
set -euo pipefail
cd ~/projects/llama.cpp
export FINAL_BUNDLE="$HOME/to_runpod_supplements/skymizer-final-reference-20260906"
export FINAL_ROOT=/opt/dlami/nvme/skymizer-final-reference-a01
export TMPDIR="$FINAL_ROOT/tmp"
export HF_HOME="$FINAL_ROOT/cache/hf"
export HF_DATASETS_CACHE="$FINAL_ROOT/cache/datasets"
export HF_HUB_CACHE="$FINAL_ROOT/cache/hub"
export HF_XET_CACHE="$FINAL_ROOT/cache/xet"
export HF_ASSETS_CACHE="$FINAL_ROOT/cache/assets"
export UV_CACHE_DIR="$FINAL_ROOT/cache/uv"
export CUDA_CACHE_PATH="$FINAL_ROOT/cache/cuda"
export UV_PROJECT_ENVIRONMENT="$FINAL_ROOT/venv"
export PYTHONDONTWRITEBYTECODE=1
export AWS_DEFAULT_REGION=us-east-2
export AWS_REGION=us-east-2
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$UV_CACHE_DIR" "$CUDA_CACHE_PATH"
(cd "$FINAL_BUNDLE" && sha256sum -c SHA256SUMS)
python3 "$FINAL_BUNDLE/verify_source.py" --repo "$PWD" --apply-patch
cmake -S . -B "$FINAL_ROOT/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build "$FINAL_ROOT/build" -j 8 --target llama-reference llama-vlm-kld
uv sync --project tools/skymizer --locked --no-dev
export FINAL_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export FINAL_REFERENCE_BINARY="$FINAL_ROOT/build/bin/llama-reference"
export FINAL_MODELS="$FINAL_ROOT/models"
export FINAL_RUNS="$FINAL_ROOT/runs"
mkdir -p "$FINAL_ROOT/provenance" "$FINAL_ROOT/frozen-tools" "$FINAL_RUNS"
test ! -e "$FINAL_ROOT/frozen-tools/skymizer"
cp -a tools/skymizer "$FINAL_ROOT/frozen-tools/skymizer"
cp -a "$FINAL_BUNDLE" "$FINAL_ROOT/provenance/bundle"
export FINAL_TOOLS="$FINAL_ROOT/frozen-tools/skymizer"
git rev-parse HEAD > "$FINAL_ROOT/provenance/git-head.txt"
git diff --binary HEAD > "$FINAL_ROOT/provenance/working-tree.patch"
sha256sum "$FINAL_REFERENCE_BINARY" "$FINAL_ROOT/build/bin/llama-vlm-kld" > "$FINAL_ROOT/provenance/binaries.sha256"
nvidia-smi > "$FINAL_ROOT/provenance/nvidia-smi.txt"
```

`HF_HOME` above is new. Run `"$UV_PROJECT_ENVIRONMENT/bin/hf" auth login` if credentials are needed; do not assume the old home token was copied and do not put credentials in the bundle or logs. Use a fresh `FINAL_ROOT` for a new build/attempt, retaining prior artifacts.

## Restore by exact S3 key

```bash
"$FINAL_PY" "$FINAL_TOOLS/cli/restore_reference_models.py" \
  --profiles "$FINAL_BUNDLE/protocol/reference-model-profiles-pilot100.json" --models-dir "$FINAL_MODELS"
"$FINAL_PY" "$FINAL_TOOLS/cli/restore_reference_models.py" \
  --profiles "$FINAL_BUNDLE/protocol/reference-model-profiles-pilot100.json" --models-dir "$FINAL_MODELS" --download \
  | tee "$FINAL_ROOT/provenance/model-restore.jsonl"
```

The first command shows the plan. The second downloads using individual `aws s3 cp` object URIs, then checks complete size/SHA256; conflicting existing files cause a stop without replacement. No `ls`, `sync`, recursive download or ListBucket permission is needed. Region is explicitly `us-east-2`; objects use AES256 and have no KMS key requirement. `protocol/restore-map.json` contains all eight keys. A fresh eight-file RunPod download was not performed during preparation. Existing remote identity checks are in `audit/reference-identity.json`; the InternVL projector has a prior streamed full SHA match plus unchanged current object identity, not a new HeadObject checksum.

## Generate one cohort

Set `FINAL_SIZE=100` after the SNR pilot is complete. Later repeat with `FINAL_SIZE=500` after the SNR collect500 is complete. The loop runs 35 jobs for the selected size and creates a fresh directory per job. A process failure is recorded and blocks the final success status; completed runs with excluded/failed rows retain their full cohort partition.

```bash
set -euo pipefail
FINAL_SIZE=100
case "$FINAL_SIZE" in
  100) cohort=pilot100 ;;
  500) cohort=collect500 ;;
  *) exit 1 ;;
esac
profile="$FINAL_BUNDLE/protocol/reference-model-profiles-$cohort.json"
jobs="$FINAL_ROOT/provenance/jobs-$cohort.tsv"
"$FINAL_PY" - "$profile" > "$jobs" <<'JOBS'
import json, sys
p = json.load(open(sys.argv[1]))
for model, settings in p['models'].items():
    for mode in settings['semantic_modes']:
        for source in p['sources']:
            print(model, mode, source, sep='\t')
JOBS
failed=0
while IFS=$'\t' read -r model mode source; do
  run="$FINAL_RUNS/$cohort/$model/$mode/$source"
  test ! -e "$run"
  mkdir -p "$(dirname "$run")"
  if "$FINAL_PY" "$FINAL_TOOLS/cli/generate_model_reference.py" \
    --profiles "$profile" --model "$model" --mode "$mode" --source "$source" \
    --size "$FINAL_SIZE" --hardware pro6000 --gpu 0 --models-dir "$FINAL_MODELS" \
    --parallel 1 --out "$run" --llama-reference "$FINAL_REFERENCE_BINARY" > "$run.log" 2>&1; then
    cp "$profile" "$run/selected-profile.json"
  else
    mkdir -p "$run"
    cp "$profile" "$run/selected-profile.json"
    printf '%s\n' "$run" >> "$FINAL_ROOT/provenance/failed-$cohort.txt"
    failed=$((failed + 1))
    continue
  fi
  "$FINAL_PY" "$FINAL_BUNDLE/verify_generation_profile.py" \
    --tools "$FINAL_TOOLS" --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
  "$FINAL_PY" "$FINAL_TOOLS/cli/upload_reference.py" --private --dry-run \
    --run "$run" --profiles "$profile" --model "$model" --mode "$mode" > "$run.prepare.log" 2>&1
done < "$jobs"
test "$failed" -eq 0
```

Do not add `--num-samples` for production. A 2-row acceptance smoke cannot be published as a complete 100/500 cohort. Preserve `dataset/`, metadata, inputs, native output, scripts and the requested/eligible/excluded/failed IDs. Repetition excludes the whole row; nonrepeating cap-limited answers remain. No replacement sampling is performed. If prompt tokens/positions plus the requested cap exceed context, the native tool rejects that row; it does not silently truncate the image/prompt or shrink the cap. Default row watchdog is 1800 seconds. Cross-invocation resume is not supported; keep failed attempts and use a new root for any authorized rerun.

## Publish privately

Use this for each prepared run, setting `model`, `mode`, `source` and `cohort` to that run. The example selects the InternVL instruct pilot. Dry-run exports locally. The next block explicitly creates or changes the destination to private before upload; `--private` alone only controls newly created repositories in this code version.

```bash
set -euo pipefail
model=internvl3.5-30b-a3b
mode=instruct
source=blink-val
cohort=pilot100
profile="$FINAL_BUNDLE/protocol/reference-model-profiles-$cohort.json"
run="$FINAL_RUNS/$cohort/$model/$mode/$source"
"$FINAL_PY" "$FINAL_BUNDLE/verify_generation_profile.py" \
  --tools "$FINAL_TOOLS" --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
"$FINAL_PY" "$FINAL_TOOLS/cli/upload_reference.py" --private --dry-run \
  --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
"$FINAL_PY" - "$run/upload/manifest.json" <<'PRIVATE'
import json, sys
from huggingface_hub import HfApi
repo = json.load(open(sys.argv[1]))['repo']
api = HfApi()
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
if not api.repo_info(repo, repo_type='dataset').private:
    api.update_repo_settings(repo, repo_type='dataset', private=True)
if api.repo_info(repo, repo_type='dataset').private is not True:
    raise SystemExit('Dataset is not private; upload refused')
PRIVATE
"$FINAL_PY" "$FINAL_TOOLS/cli/upload_reference.py" --private \
  --run "$run" --profiles "$profile" --model "$model" --mode "$mode"
```

Repositories: `elichen-skymizer/<checkpoint>-pilot` or `elichen-skymizer/<checkpoint>-collect-500`. Configs: `<source>-subsample-<100|500>-<ins|think>`, split `train`; Muse has only `-think`. Each config has an audit manifest. Conflicting remote content is rejected, not overwritten. Preserve the returned HF commit and upload receipt. No HF write was performed during this preparation.

## Acceptance and limits

`DELIVERY.json` and `evidence/` record actual tests, profile/source hashes and independent review. Five semantic lanes generated 10 new-pin rows and 7,512 tokens; all stopped naturally. All 341 reference/capacity/strict-replay checks passed, plus five expected short-cohort publication refusals. The 70 production commands were checked without executing the full cohorts.

| Model / mode | Generated lengths | Peak VRAM MiB |
| --- | --- | ---: |
| InternVL instruct | 37, 85 | 62541 |
| InternVL R1 thinking | 1126, 1463 | 62541 |
| GLM instruct | 109, 91 | 22199 |
| GLM thinking | 701, 1640 | 22199 |
| Muse high | 649, 1611 | 58167 |

Muse's 21-row capacity probe peaked at 58,241 MiB; maximum prompt demand was 10,588 tokens/positions, leaving 13,988 context slots after reserving the pilot thinking cap. This arithmetic reserve is not a measurement of every input generating 8,192 tokens. InternVL strict VLM-KLD replay passed 3 original regression rows / 48 real targets at 84,905 MiB. Its instruct native generation preceded the Python checker correction; the existing rows were rebuilt and revalidated on CPU with identical native IDs, logprobs and image bytes. Other modes passed the updated high-level producer directly.

 Small samples establish that these workflows execute; they do not establish full-cohort success, candidate quality, native SNR saturation or publication acceptance. Muse capacity used three heuristic high-demand rows from each new-pin 500 subset, not an exhaustive proof of the largest native token count.

InternVL uses strict replay after correcting the Skymizer checker to compare a complete adjacent-tile placeholder span. The Python schema's legacy `per_image_*` fields describe chunks, while media markers and image hashes describe source images; they need not have equal counts. Text IDs, each tile span, position totals and original image SHA256 remain checked. Upstream model code was not changed.
