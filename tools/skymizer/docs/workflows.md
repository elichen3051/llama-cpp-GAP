# Workflow scripts

Run from the repository root. Set `SKYMIZER_WORK` to a writable work volume; on a GPU host, use the actual NVMe mount for builds, environments, caches and outputs. Confirm the mount and available space before a large run. Keep model files and frozen input data at explicit paths.

```bash
export SKYMIZER_WORK=/path/to/work-volume/skymizer
export TMPDIR="$SKYMIZER_WORK/tmp"
export UV_CACHE_DIR="$SKYMIZER_WORK/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_WORK/venv"
export HF_HOME="$SKYMIZER_WORK/hf"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME"
uv sync --project tools/skymizer --python 3.12.3 --locked --group dev
export SKYMIZER_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
```

A new `HF_HOME` has its own token/cache location. Authenticate the intended account when private inputs or publication require it. Do not copy credentials into output archives.

The optional `scripts/setup.sh` installs missing host prerequisites, builds all five native tools, syncs the locked environment and runs CPU checks. Inspect it before use. It requires `SKYMIZER_WORK`; `FORCE_CPU=1` selects a CPU build, `SKIP_APT=1` skips apt installation, and `SKIP_BUILD=1` or `SKIP_TESTS=1` skips those stages. `BUILD_DIR`, `VENV_DIR`, `TMPDIR`, `UV_CACHE_DIR`, `JOBS` and `PYTHON_VERSION` override its defaults. A prepared host can use:

```bash
SKIP_APT=1 tools/skymizer/scripts/setup.sh
```

`00_env.sh` is a sourced settings helper, not a stage to execute. Wrappers select `PYTHON`, then `SKYMIZER_PYTHON`, then the configured environment. `BUILD_DIR` defaults to `$SKYMIZER_WORK/build`; `OUT_ROOT` defaults to `$SKYMIZER_WORK/outputs`. Relative model, dataset, build and output overrides are resolved from `tools/skymizer`; use absolute paths when invoking wrappers from another directory. Relative `SKYMIZER_WORK` and `PYTHON` paths are resolved before the helper changes directory.

| Script | Purpose | Inputs and result |
| --- | --- | --- |
| `01_generate_reference.sh` | One checkpoint/source/mode generation job | Requires `COHORT_PROFILE`, `CHECKPOINT`, `SOURCE`, `MODE`, `COHORT_SIZE`; writes `REFERENCE_OUT` or `$OUT_ROOT/reference`. Supports `--dry-run`. |
| `02_upload_reference.sh` | Optional private publication | Uses `COHORT_PROFILE`, `CHECKPOINT`, `MODE`, `REFERENCE_RUN` or `REFERENCE_OUT`. `--dry-run` validates and stages locally; execution without it writes to the Hub. |
| `03_collect_vlm_kld.sh` | Collect VLM candidates A and B against one reference | Requires `VLM_DATASET` and exact `VLM_REF_MODEL`, `VLM_REF_MMPROJ`, `VLM_CAND_A_MODEL`, `VLM_CAND_B_MODEL`. |
| `04_collect_text_bridge.sh` | Prepare and verify the classic PPL/full-logit text bridge | Requires `CORPUS`, `CORPUS_NAME`, `LLM_REF_MODEL`, `LLM_CAND_A_MODEL`, `LLM_CAND_B_MODEL`. Writes a new `TEXT_WORK` directory. |
| `05_compare.sh vlm` or `text` | Paired statistics from completed collections | Forwards trailing comparator options, for example `--unit article`; writes Markdown and JSON. |
| `06_power_analysis.sh vlm` or `llm` | Optional pilot design and diagnostics | Requires `TOKEN_CAPS`; runs sequential-prefix planning, supported variance diagnostics and finite-pilot verdict reproducibility. |

Choose a branch. Reference generation can feed VLM collection locally, so publication is optional. VLM and classic text collection are alternatives and both feed the comparator. Reference generation can also produce ordinary text trajectories; the descriptive `collect_llm_trajectory.sh` wrapper collects those and `05_compare.sh llm` compares them. It does not implement the classic 512-token corpus protocol.

The numbered scripts delegate to existing CLIs. There is no global GPU reservation or automatic experiment group gate. Use one scorer per GPU and preserve each checkpoint's runtime across candidates. The [reference guide](runpod-reference.md), [VLM guide](vlm-kld.md) and [text bridge](text-bridge.md) specify their inputs and execution contracts.

The planning wrapper uses `DESIGN_SIZES` for prospective sample sizes and `REPRO_SIZES` for subsets of the observed pilot. Sizes above the finite pilot are skipped by the reproducibility CLI with a warning. `TOKEN_CAPS` must not exceed the stored scoring horizon. Set `SESOI` to an externally chosen B-minus-A effect for power; without it the planner reports precision/MDE. `EFFECT_PROFILE=pilot` also requires `REFERENCE_CAP`. Variance decomposition supports only `kld`, `reversed_kld`, `js_kld` and `nll`; other planner metrics skip that diagnostic. The reproducibility report is a separate multi-metric diagnostic, not prospective power for `SESOI`. Prefix planning assumes prompt/answer items; choose grouped corpus inference in the text guide instead of treating adjacent windows as independent pilot questions.

`scripts/dataset_maintenance/` is an optional prepared-dataset audit and replacement workflow. Its scanner is read-only; `replace_subsamples.py publish` changes sampled source data after its documented independent review. It is not a required evaluation stage. Functional and numerical checks live in [verify_and_validation_scripts](../verify_and_validation_scripts/README.md).
