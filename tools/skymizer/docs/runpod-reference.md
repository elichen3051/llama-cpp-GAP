# Reference generation on a GPU host

Use the [shared setup](workflows.md) and a reviewed cohort profile. The same commands run on RunPod or another host with the required local model files, CUDA build and authenticated input access. Paths to a previous host's audit directory, source patches and external supplement bundles are not needed by this checkout.

| Profile group | Checkpoints | Files in `profiles/` |
| --- | --- | --- |
| Small SNR | Qwen3.5-4B, Gemma E4B | `small-pilot100.json`, `small-collect500.json` |
| Remaining SNR | Gemma 31B, Qwen3.6-35B-A3B, Kimi Instruct, Kimi Thinking-2506 | `snr-pilot100.json`, `snr-collect500.json` |
| Final evaluation | InternVL3.5-30B-A3B, GLM-4.6V-Flash, Muse Glimmer | `final-pilot100.json`, `final-collect500.json` |

These profiles retain the accepted dataset revision, file identities, sampling, semantic modes and measured runtimes. See [profile provenance](../profiles/README.md). Pilot100 uses instruct/thinking caps of 2048/8192; collect500 uses 1024/4096. The size is part of the profile contract. For each cohort, complete the SNR group before the final-evaluation group. The operator coordinates this order across hosts.

Small-model autoregressive generation uses four sequences in one model/context. The accepted large-model profiles use one; Gemma 31B uses its Q8 MTP sidecar and requires one. Do not copy parallelism or microbatch settings between checkpoints. Kimi Instruct and Thinking are separate checkpoints with their own modes. InternVL thinking requires the profile's R1 system prompt. Muse has thinking/high only and a frozen template date. The profile is the source of these values; model-card advice is not read automatically at runtime.

## Restore and select one job

Set absolute paths after the shared setup. Exact S3 restore requires AWS CLI credentials and the bucket's region. It reads the profile's object keys and checks complete file hashes; existing mismatches are refused.

```bash
export COHORT_PROFILE="$PWD/tools/skymizer/profiles/small-pilot100.json"
export MODELS_DIR=/path/to/model-volume/models
export CHECKPOINT=qwen3.5-4b
export SOURCE=mmmu-pro-vision
export MODE=instruct
export COHORT_SIZE=100
export GPU=0
export REFERENCE_OUT="$SKYMIZER_WORK/reference/qwen35-pilot-mmmu-vision-ins"

"$SKYMIZER_PYTHON" tools/skymizer/cli/restore_reference_models.py \
  --profiles "$COHORT_PROFILE" --models-dir "$MODELS_DIR" --models "$CHECKPOINT"
```

The restore command prints a plan. Add `--download` to fetch missing files. An existing independently verified local model tree can be used directly.

```bash
tools/skymizer/scripts/01_generate_reference.sh --dry-run
tools/skymizer/scripts/01_generate_reference.sh
```

The output directory must be new. A diagnostic `--num-samples 2` run must use a separate output directory and cannot be published as a complete 100/500 cohort. Changing generation caps or parallelism changes the experiment; use a matching reviewed profile for formal runs.

## Validate and optionally publish

Retain the entire run: `dataset/`, `metadata.json`, `complete.json`, native outputs, source snapshots, exact requests/images, logs and excluded/failed IDs. Reconcile every requested ID. Repetition excludes the whole row; nonrepeating cap-limited answers remain marked as truncated. Context or image failures do not trigger replacements or silent input truncation. Inspect effective context, batch/microbatch, sequence count, sampling, model/projector/MTP hashes and image policy against the selected profile. Publisher checks do not replace this runtime review or a native replay check.

```bash
tools/skymizer/scripts/02_upload_reference.sh --dry-run
```

The dry run validates and stages local Parquet and audit files. To publish the same completed run privately:

```bash
tools/skymizer/scripts/02_upload_reference.sh
```

The current uploader creates or changes the destination to private and verifies visibility before accepting a receipt. It publishes to `elichen-skymizer/<checkpoint>-pilot` or `<checkpoint>-collect-500`; configs are `<source>-subsample-<100|500>-<ins|think>`. Conflicting existing content is refused. Preserve the returned exact Hub commit and receipt. Publication is optional: a local `REFERENCE_OUT/dataset` is already a valid collector input.

## Multiple jobs and recovery

The campaign CLI archives the selected profile, scripts and provenance before dispatching one job at a time per selected GPU. This example generates every supported source/mode for one small checkpoint without publication:

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/run_reference_campaign.py \
  --profiles "$COHORT_PROFILE" --models "$CHECKPOINT" --size "$COHORT_SIZE" \
  --models-dir "$MODELS_DIR" --gpus "$GPU" --hardware pro6000 \
  --llama-reference "$SKYMIZER_WORK/build/bin/llama-reference" \
  --out "$SKYMIZER_WORK/reference-campaign" --no-upload
```

Use `--dry-run` to inspect a campaign. Explicit `--upload` enables its private upload workers. A campaign resume checks the frozen plan, immutable archive, execution identity and completed artifacts. Old archives remain immutable; a directory layout change does not migrate their evidence. A current uploader that differs from an archived one cannot automatically publish that old campaign. See [native reference recovery and publication](reference.md#production-profiles-shared-gpu-queue-and-publication).

Direct `generate_model_reference.py` jobs do not resume across Python invocations. Internal row retries preserve completed native rows within that invocation. Preserve a failed attempt and choose a new output root for a rerun. A successful short acceptance run demonstrates only the exercised runtime and rows; it does not establish full-cohort completion or candidate quality.
