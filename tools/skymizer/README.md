# Skymizer: reference generation, KLD collection and paired statistics

Skymizer generates reference answers with llama.cpp, replays their exact tokens through a reference and a quantized candidate, and compares saved candidate metrics. Supported models and CUDA kernels come from this checkout's upstream base. Skymizer does not implement additional model architectures.

Reference generation, VLM-KLD and the text PPL bridge are separate workflows. Statistics runs on completed metric collections without loading a model or using a GPU.

## Source layout

```text
tools/skymizer/
  core/                          C++ generation, teacher forcing, metrics and repetition detection
  cli/                           Python reference and collection entry points
  lib/                           Shared datasets, metric I/O, provenance and collection state
  stats/                         Statistical inference, power planning and report rendering
    cli/                         Command-line analysis tools
  profiles/                      Explicit model/runtime profiles for pilot100 and collect500
  scripts/                       Numbered workflow helpers and optional dataset maintenance
  verify_and_validation_scripts/ Functional, numerical and GPU validation helpers
  tests/                         Automated tests and fixed numerical fixtures
  docs/                          Workflow guides, formats and troubleshooting
  knowledge/                     Statistical theory and explicitly historical research notes
  CMakeLists.txt
  pyproject.toml
  uv.lock
```

See [numbered workflows](docs/workflows.md) for the optional shell wrappers.

The native targets remain `llama-reference`, `llama-vlm-kld` and `llama-llm-kld`, built into the selected build directory's `bin/`. C++ sources are together in `core/`; collection and analysis share `lib/kld_metrics_io.py` rather than defining the metric format twice.

## Reproduce the tested environment

Run commands from the llama.cpp repository root. Put new environments, builds, caches, logs and datasets on a work volume with sufficient space. Existing model files may remain elsewhere.

```bash
export SKYMIZER_WORK=/opt/dlami/nvme/skymizer
export TMPDIR="$SKYMIZER_WORK/tmp"
export UV_CACHE_DIR="$SKYMIZER_WORK/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_WORK/venv"
export HF_HOME="$SKYMIZER_WORK/hf"
export CCACHE_DIR="$SKYMIZER_WORK/ccache"
export CCACHE_TEMPDIR="$TMPDIR/ccache"
export CUDA_CACHE_PATH="$SKYMIZER_WORK/cuda-cache"
export XDG_CACHE_HOME="$SKYMIZER_WORK/xdg-cache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$CCACHE_TEMPDIR"
uv sync --project tools/skymizer --python 3.12.3 --locked --group dev
export SKYMIZER_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"

cmake -S . -B "$SKYMIZER_WORK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build "$SKYMIZER_WORK/build" --target \
  llama-reference llama-vlm-kld llama-llm-kld llama-perplexity llama-tokenize -j8
```

The validated Python environment uses Python 3.12.3, NumPy 2.5.2, datasets 5.0.1, Pillow 12.3.0, huggingface-hub 1.30.0 and PyArrow 25.0.1. SciPy 1.18.1 is a runtime dependency for statistics. Development checks use pytest 9.1.1 and pytest-timeout 2.4.0. Direct dependencies are pinned in `pyproject.toml`; `uv.lock` pins transitive dependencies. `--locked` refuses a lockfile that no longer matches the project rather than silently changing its resolution ([uv documentation](https://docs.astral.sh/uv/concepts/projects/sync/)).

For local CPU statistics and tests, run `uv sync --python 3.12 --locked --group dev` from `tools/skymizer`. With `UV_PROJECT_ENVIRONMENT` unset, this creates `.venv`; use `.venv/bin/python` or activate it with `source .venv/bin/activate`.

Student-t tests, t distribution functions and Wilson intervals use SciPy. Function docstrings carry API references; [statistical APIs and verification](docs/statistical-apis.md) records MATLAB, Julia and Wolfram correspondences and independent checks. Token-weighted results are descriptive only.

The KLD acceptance environment used CUDA 13.2.51 and an RTX PRO 6000 Blackwell Server Edition. Python pins do not pin CUDA, compiler flags, model files or GPU behavior: retain native binary/backend hashes and runtime metadata with every study. The optional `hf-tokenizer` extra is for legacy HF/vLLM references and was not used by the native reference/KLD run; its lockfile entries are not evidence of native-run validation. `scripts/setup.sh` automates setup and checks; inspect its options before running it.

## 1. Generate reference answers

Choose the matching profile explicitly; there is no implicit six-model production profile. The [profile index](profiles/README.md) lists model groups, supported modes and cohort limits. `pilot100` and `collect500` use different generation caps. Image-token bounds are omitted so generation uses the mmproj defaults. Generation parallelism comes from each validated profile; KLD uses one sequence.

This command generates one Qwen3.5-4B pilot subset locally. Models must already be present under `--models-dir` with the profile's relative filenames.

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/run_reference_campaign.py \
  --profiles tools/skymizer/profiles/small-pilot100.json \
  --models qwen3.5-4b --modes instruct --sources mmstar \
  --size 100 --gpus 0 --models-dir "$HOME/models" \
  --llama-reference "$SKYMIZER_WORK/build/bin/llama-reference" \
  --out "$SKYMIZER_WORK/reference/qwen35-pilot-v1" --no-upload
```

Remove the model/mode/source filters to collect the selected profile's complete supported set. Add `--dry-run` to inspect the plan first. For 500 questions, choose `small-collect500.json`, `--size 500` and a new study directory. Run the SNR decision groups before the final-evaluation group within each cohort. Pilot rows may overlap the 500-question cohort; they are not an independent confirmation set.

Each campaign freezes scripts, profiles, source provenance and its plan. Successful attempts contain `dataset/`, `metadata.json`, `complete.json`, `run_state.json`, `requests.jsonl`, `excluded.jsonl` and `failures.jsonl`. Preserve the whole attempt. Repetition removes an entire row; context failures are recorded without replacement. Non-repetitive answers reaching the generation cap remain eligible and are marked as capped.

Publication uses `cli/upload_reference.py` or the campaign's upload option, verifies the generated data, and sets the HF dataset **private**. The namespace is `<model>-pilot` or `<model>-collect-500`; configs are `<source>-subsample-<100|500>-<ins|think>`, split `train`. Do not reuse a remote config for different content. See [reference generation](docs/reference.md) for upload, MTP, resumption and single-job commands.

## 2. Collect VLM-KLD on frozen answers

All candidates for the same base checkpoint use the same reference dataset and scoring runtime. The scorer loads reference and candidate together, prefills the same prompt/images, then teacher-forces the saved continuation tokens. It computes full-vocabulary metrics while logits are in memory and saves metric records, not full logits. Native reference token IDs, original image content and vocabulary mapping are checked before scoring.

Use a separate KLD study and a stable candidate label. `REFERENCE_DATASET` points to a completed attempt's local `dataset/` directory; `CANDIDATE_GGUF` points to the candidate file, or the first shard for a split model.

```bash
export REFERENCE_DATASET=/path/to/completed/attempt-0001/dataset
export CANDIDATE_GGUF=/path/to/qwen35-q4_0.gguf
"$SKYMIZER_PYTHON" tools/skymizer/cli/collect_model_kld.py \
  --profiles tools/skymizer/profiles/small-pilot100.json \
  --study "$SKYMIZER_WORK/vlm-kld/qwen35-pilot-v1" --size 100 \
  --model qwen3.5-4b --source mmstar --mode instruct \
  --candidate bartowski-Q4_0 --cand-model "$CANDIDATE_GGUF" \
  --dataset "$REFERENCE_DATASET" --models-dir "$HOME/models" --gpu 0 \
  --llama-vlm-kld "$SKYMIZER_WORK/build/bin/llama-vlm-kld"
```

Repeat with the other candidate path and label in the same KLD study. The helper chooses the recorded profile runtime and reference BF16 projector. Native KLD remains autoregressive even if reference generation used MTP. Threads and metric threads are 8, flash attention is enabled, all layers are offloaded and `swa_full` is false; architecture-defined sliding-window attention is not removed. Context and ubatch are model-specific. See [collection](docs/collect.md) and [VLM-KLD handover](docs/vlm-kld.md) for direct flags, capacity and permitted vocabulary-attribute differences.

## 3. Collect the text PPL / LLM-KLD bridge

This branch uses a frozen text corpus, not VLM-generated answers. Follow [the text bridge guide](docs/text-bridge.md) for WikiText-2 or the full corpus from `scripts/get-pg.sh` and the numbered text-bridge wrapper.

Freeze corpus bytes and article boundaries before tokenization. Both tools use the same native tokenizer, BOS handling, 512-token windows and target positions 257 through 511 (255 targets). Use `-c 512 -b 512 -ub 512`, one sequence, matching thread counts and runtime. The saved llama-perplexity base is quantized; full-distribution LLM-KLD metrics provide the precision bridge. Save both PPL logs, the reference logits base, the prepared corpus and each candidate's complete metric collection. VLM context settings do not need to match this 512-token protocol.

## Keep studies ready for analysis

Use a new study ID when changing the reference, profile, corpus, native build, runtime or collection horizon. Keep each base checkpoint, reference mode/strength, cohort and subset identifiable. Candidate labels must include the provider when filenames alone are ambiguous.

```text
$SKYMIZER_WORK/
  reference/<study-id>/
    plan.json, study.json, status.json
    scripts/                                  Frozen code, profile and provenance
    artifacts/<model>/<subset>/attempt-0001/   Complete reference attempt
  vlm-kld/<study-id>/
    scripts/, plan.json, study.json
    artifacts/<model>/<subset>/kld/<candidate>/
      collect_meta.json
      manifest.csv
      metrics/<item-key>.npz
  text/<corpus-revision>/<base-checkpoint>/
    prepared/                                 Corpus, tokens, windows and article mapping
    ppl/                                      Reference logits and per-candidate logs
    kld/<candidate>/                          Complete LLM metric collection
  analysis/<comparison-id>/
    paired-window.md, paired-window.json
    paired-article.md, paired-article.json
    power.md, power.json
```

The reference and VLM paths above are launcher output layouts. The text and analysis parents are a naming convention you choose through output arguments. Retain all collection contents: copying only `metrics/*.npz` loses the provenance and completion checks needed by the statistical tools. Do not rename metric stems or infer pairings from filesystem order. Pairing uses exact stored identities and target alignment; directory names are for navigation.

Record source revision, subset/split, model/provider hashes, semantic mode, generation cap, scoring horizon, runtime and exclusions with the study. Keep article/window maps and related-source IDs. Article or block aggregation changes the unit of analysis, but does not prove independent sampling. Do not pool different strengths or unrelated corpora into one comparison merely because the directories have similar names.

## 4. Run statistics

`A` and `B` must be completed collection directories for two candidates scored against the same reference. Example directories from the VLM command above:

```bash
export A="$SKYMIZER_WORK/vlm-kld/qwen35-pilot-v1/artifacts/qwen3.5-4b/mmstar-subsample-100-ins/kld/bartowski-Q4_0"
export B="$SKYMIZER_WORK/vlm-kld/qwen35-pilot-v1/artifacts/qwen3.5-4b/mmstar-subsample-100-ins/kld/bartowski-Q4_1"
export REPORTS="$SKYMIZER_WORK/analysis/qwen35-mmstar-pilot-q4_0-vs-q4_1"
mkdir -p "$REPORTS"
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/saved_metrics_paired_compare.py \
  --candidate-a "$A" --candidate-b "$B" --ci-method t \
  --out "$REPORTS/paired.md" --output-json "$REPORTS/paired.json"
```

The default primary endpoint is item-weighted forward KLD with a paired Student-t confidence interval. For KLD, a negative mean difference B - A favors B. NLL, reverse KLD, JSD, EAR and other metrics have their own direction and interpretation; a lower KLD is fidelity to the reference, not answer accuracy.

| Tool | Purpose | Key options |
| --- | --- | --- |
| `saved_metrics_paired_compare.py` | Paired differences, intervals, equivalence decisions and exploratory tails | `--metrics`, `--weighting`, `--ci-method`, `--equivalence-margin`, `--num-eval-tokens` |
| Same comparison tool | Text-window, article or contiguous-block aggregation | `--unit window`, `--unit article`, `--unit block --block-windows 8` |
| `power_analysis.py` | Prospective sample-size/token-cap planning; plug-in precision and Gaussian directional MDE without an assumed effect | `--token-caps`, `--sample-sizes`, `--sesoi`, `--reps`, `--outer-reps` |
| `variance_decomposition.py` | Observed prefix length, effect and variance diagnostics | `--metric kld`, `--num-eval-tokens`, `--output-json` |
| `random_subsample_power.py` | Verdict stability within an already observed finite pilot | `--mode reproducibility`, `--sizes`, `--reps` |

For an LLM corpus comparison, use the same comparison command with the two text collection paths and `--unit article`, or `--unit block --block-windows 8`. Article/block analysis needs complete original windows; do not truncate those windows with an answer-prefix flag.

A prospective planning example for a study whose saved horizon is at least 128 tokens:

```bash
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/power_analysis.py \
  --candidate-a "$A" --candidate-b "$B" --metric kld --weighting item \
  --token-caps 32 64 128 --sample-sizes 50 100 200 \
  --sesoi=-0.0001 --reps 2000 --outer-reps 200 \
  --out "$REPORTS/power.md" --output-json "$REPORTS/power.json"
```

The example SESOI is illustrative: choose a scientifically meaningful effect before interpreting prospective power. Omitting `--sesoi` gives precision/MDE planning. Finite-pilot reproducibility and observed-effect variance diagnostics do not estimate fresh-sample power. Prefixes only use existing positions; early EOS does not become zero-padded data. Read [statistics](docs/compare.md) for metrics, weighting, confidence methods, Holm adjustment and grouping, and [power planning](docs/power-analysis.md) for assumptions and limitations.

## Validate and migrate

```bash
CUDA_VISIBLE_DEVICES='' SKYMIZER_TEST_BIN="$SKYMIZER_WORK/build/bin" \
  "$SKYMIZER_PYTHON" -m pytest tools/skymizer/tests \
  -q -m 'not external_model' --basetemp "$SKYMIZER_WORK/pytest-check"
```

Use a dedicated `--basetemp`: pytest clears that directory. Native integration checks require built binaries; external-model checks require model/data assets. Functional GPU helpers live in `verify_and_validation_scripts/`. Preserve numerical reduction order, RNG order, metric layout and report semantics during refactoring; the independent oracle and exact fixtures enforce [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md).

Version 0.2.0 moves C++ sources into `core/`, `compare/` into `stats/`, the four analysis commands into `stats/cli/`, and `review-functionality/` into `verify_and_validation_scripts/`. Workflow names and profile paths changed too. Existing reference datasets keep their format. Existing campaign archives remain immutable: resume them with their original archived code, or start a new study using their validated local dataset. Do not rewrite old archives to resemble a new release.

For implementation review, follow the [architecture and invariant map](docs/architecture.md).
