# Skymizer: reference generation and quantization evaluation

Skymizer uses llama.cpp to generate reference trajectories, compare a reference and candidate at the same target tokens, and test paired differences between candidates. It computes full-vocabulary KLD, reverse KLD, JSD, NLL, EAR and reference top-K EAR while logits are in memory. Collectors save 76-byte metric records per target rather than full logits.

Model execution uses the architectures, quantization kernels and vision support already present in this checkout's upstream base. Skymizer does not add model implementations. Native reference rows retain GGUF tokenization and original images; legacy HF/vLLM rows use a separate compatibility path. VLM metrics measure answer-token fidelity conditioned on images. They do not measure distributions over vision embeddings or task accuracy.

## Choose a workflow

| Task | Entry point | Guide |
| --- | --- | --- |
| Generate and privately publish reference datasets | `cli/run_reference_campaign.py` | [Reference generation](docs/reference.md), [RunPod handover](docs/reference-runpod-handover.md), [final-evaluation models](docs/reference-runpod-final-handover.md) |
| Collect VLM metrics on frozen answers | `cli/collect_kld.py` | [Collection contract](docs/collect.md), [AWS VLM handover](docs/kld-aws-handover.md) |
| Compare classic corpus PPL with full-logit LLM KLD | `cli/prepare_perplexity_corpus.py`, `cli/collect_llm_kld.py`, `cli/verify_perplexity_bridge.py` | [WikiText-2 and full PG bridge](docs/perplexity-llm-kld-aws-handover.md) |
| Compare two completed metric collections | `cli/saved_metrics_paired_compare.py` | [Statistics and grouping](docs/compare.md) |
| Plan sample size or inspect pilot stability | `cli/power_analysis.py`, `cli/variance_decomposition.py`, `cli/random_subsample_power.py` | [Sequential-prefix planning](knowledge/seq-power-analysis.md) |

The 100-question cohort is the **pilot**. Pilot and 500-question collection profiles have different generation caps; always select the matching profile. Model-specific sampling, MTP, image capacity and execution order belong to the handovers, not a shared model-independent default.

## Environment and build

Run these commands from the repository root. Change `SKYMIZER_WORK` to a writable work volume on another host. Python requires 3.12 or newer; this is a repository-local script collection, not an installed Python package.

```bash
export SKYMIZER_WORK=/opt/dlami/nvme/skymizer-demo
export TMPDIR="$SKYMIZER_WORK/tmp"
export UV_CACHE_DIR="$SKYMIZER_WORK/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_WORK/venv"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
uv sync --project tools/skymizer --python 3.12 --frozen --group dev
export SKYMIZER_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
```

CPU reports need no model or CUDA build. To generate references or collect metrics:

```bash
cmake -S . -B "$SKYMIZER_WORK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build "$SKYMIZER_WORK/build" --target   llama-reference llama-vlm-kld llama-llm-kld llama-perplexity llama-tokenize -j8
"$SKYMIZER_WORK/build/bin/llama-vlm-kld" --self-test
"$SKYMIZER_WORK/build/bin/llama-llm-kld" --self-test
```

Use `-DGGML_CUDA=OFF` for a CPU build. The optional `env_setup.sh` automates host dependency installation, build and checks; inspect its documented environment overrides before running it. Legacy HF/vLLM preparation additionally needs `uv sync --project tools/skymizer --frozen --extra hf-tokenizer`; native rows do not need that extra.

## CPU demonstration

This example uses four explicit synthetic paired scores to demonstrate the report API. Its output is not a model evaluation. Real studies should use the collection CLI, which checks provenance, completed work, target alignment and the shared reference before invoking this engine.

```bash
"$SKYMIZER_PYTHON" - <<'PYTHON'
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tools/skymizer').resolve()))
from compare.engine import compare_items
from compare.render import format_comparison_table

scores_a = [{'kld': x} for x in [0.01, 0.03, 0.02, 0.04]]
scores_b = [{'kld': x} for x in [0.03, 0.04, 0.06, 0.07]]
result = compare_items(
    scores_a, scores_b, [100, 200, 150, 80], metrics=['kld'],
    confidence_level=0.95, bootstrap_iters=1000, seed=42,
    model_a_label='Demo A', model_b_label='Demo B', ci_method='t')
result['demo'] = {'source': 'synthetic scores', 'model_evaluation': False}
out = Path(os.environ['SKYMIZER_WORK']) / 'cpu-demo'
out.mkdir(parents=True, exist_ok=True)
(out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
(out / 'report.md').write_text(format_comparison_table(result, reference_label='Demo reference'))
print(out)
PYTHON
```

## Collect and compare real data

Use the lane-specific handover to freeze model files, reference dataset, binary/backend identity and runtime before collection. Both models remain loaded together. All quantizations of one base checkpoint share the same runtime and reference trajectories. With image-token flags omitted, collectors preserve a native dataset's recorded image policy; when that policy is also default, bounds come from the chosen mmproj. KLD uses one sequence; reference generation may use validated parallel generation.

The numbered scripts are optional wrappers around the same CLIs:

```bash
VLM_DATASET=/path/to/frozen/reference/dataset VLM_SUBSET=''   OUT_ROOT="$SKYMIZER_WORK/metrics" tools/skymizer/scripts/01_save_vlm_kld.sh
OUT_ROOT="$SKYMIZER_WORK/metrics" tools/skymizer/scripts/03_paired_test_kld.sh vlm
```

Set the exact reference/candidate model and projector variables in `scripts/00_env.sh` or through environment overrides before running the example. Use a fresh output root for a new comparison. Direct commands and the text-corpus protocol are in the guides above.

A paired report's default primary endpoint is item-weighted forward KLD with a paired Student-t interval. Token weighting answers a different question. Other cells are exploratory; their p-values have Holm adjustment within the documented families. Corpus comparisons can use windows, articles or contiguous blocks. Grouping does not by itself establish independence; correlated source material can invalidate nominal CI coverage.

Prefix analysis reuses saved metrics and avoids another GPU pass. Compare all candidates at the same stored scoring horizon before taking prefixes. Prospective power requires an externally chosen effect; finite-pilot verdict stability is a separate diagnostic.

## Validation and maintenance

```bash
CUDA_VISIBLE_DEVICES='' SKYMIZER_TEST_BIN="$SKYMIZER_WORK/build/bin" \
  "$SKYMIZER_PYTHON" -m pytest tools/skymizer/tests   -q -p no:cacheprovider -m 'not external_model' --basetemp "$SKYMIZER_WORK/pytest-check"
```

Use a dedicated `--basetemp`: pytest clears that directory before running. The suite includes independent numerical oracles, exact engine goldens, collection failure handling and protocol checks. Native integration tests need the relevant binaries and otherwise skip. `external_model` tests require external assets; `statistical` tests run Monte Carlo checks. Manual GPU checks live in `review-functionality/`.

Keep the optimized metric kernel and its naive oracle independent. Preserve reduction order, RNG order, record layout and report values during code motion; see [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md). Schema or numerical changes require separate review and regression evidence. A successful load, finite short run or sparse parity check does not establish full-corpus model quality.

Source layout: `cli/` contains entry points, `lib/` owns preparation and collection support, `compare/` owns statistics/reporting, and the C++ tools use shared headers beside their sources. Data formats are defined in [formats.md](docs/formats.md); operational failures are covered in [troubleshooting.md](docs/troubleshooting.md). The [experiment checklist](docs/gotchas.md) links the relevant contracts without duplicating them.
