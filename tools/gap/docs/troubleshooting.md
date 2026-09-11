# Verifying outputs and troubleshooting

Start with the collector's native log, `manifest.csv`, `collect_meta.json` and attempt records. A process exit of zero is not sufficient evidence of a completed metric collection or PPL pass. Preserve failed attempts and use a fresh output directory when retrying.

## Header and record inspection

From the repository root:

```bash
"$COMPANY_PYTHON" - "$METRIC_FILE" <<'PYTHON'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tools/gap').resolve()))
from lib.kld_metrics_io import load_kld_metrics
metrics, header = load_kld_metrics(Path(sys.argv[1]))
print(header)
print({name: (str(values.dtype), values.shape) for name, values in metrics.items()})
PYTHON
```

Use the [format definition](formats.md) to interpret version, vocabulary, target count and prefill fields. Native VLM positions and the frozen sequential prefill length may differ. A classic 512-token PPL row instead has `n_prefill=257` and `n_past_actual=512`.

## Reference drift

The comparator requires exact reference metric columns on the same target IDs. Check both model/projector fingerprints, the reference dataset, binary and loaded libraries, GPU/backend, context, batching, threads, image bounds and scoring horizon. Equal command flags do not establish equality if one of those inputs changed.

To check reproducibility, collect the same pair and row range into two fresh directories with identical inputs and runtime. Compare the metric records, then run `saved_metrics_paired_compare.py` on those directories; a self-pair should produce zero deltas. `verify_and_validation_scripts/smoke_vlm_gemma4.sh` demonstrates this on GPU. Do not assume every backend or build is deterministic without checking it.

`--allow-ref-drift` records an approximate comparison. It cannot override different binaries, unfinished work, wrong targets or incompatible corpus windows. Keep strict pairing for the primary result.

## Image and prefix failures

For `marker count != num_images`, inspect the prepared prompt's `<__media__>` count and ordered image paths. Native rows use their saved prompt and original images. Legacy HF rows also require the declared tokenizer and supported wrapper reconstruction. A wrong dataset/tokenizer pairing can fail before model loading.

One source image can produce multiple tiles. A strict prefix check must compare the complete source-image tile span against the matching placeholder span. Do not reduce the image resolution or disable a strict check merely to hide a mismatch. Inspect the original row and native chunk log first. InternVL adjacent tiles must be checked as a complete image group; this is a replay check, not a new upstream model implementation.

If the complete non-causal image chunk exceeds batch or microbatch capacity, use the family runtime that was validated for that projector's defaults. Any runtime change applies to all quantizations of the same checkpoint and creates a new collection identity. A single-model load does not prove reference-plus-candidate capacity.

`unsupported model family` during legacy preparation refers to its HF wrapper registry. Prefer native reference rows for upstream-supported models. Models unsupported by the checkout's upstream base remain outside scope; do not add a model implementation to bypass this check.

## Incomplete collections or collisions

An active writer, interrupted attempt, missing terminal record or one-sided output gap blocks formal comparison. Keep the failed directory intact and repeat the affected work in a new output root. Disjoint appends are allowed only after completed work and with unchanged collection identity. See [completion rules](collect.md#completion-and-artifacts).

For too few paired groups, collect more usable rows or choose the intended complete corpus. At least two groups are necessary to estimate a sampling variance, but that minimum does not imply adequate power or independence.

## PPL or KLD anomalies

For the corpus bridge, first verify exact tokenization, BOS replacement, 512-token windows and targets with `verify_perplexity_bridge.py`. Candidate PPL needs both `--kl-divergence` and `--kl-divergence-base FILE`; the latter alone can select a writer path and overwrite the base. Preserve and recheck the original base SHA.

Original reference PPL differs from PPL reconstructed from the clipped uint16 base. Report both. A high candidate PPL with an equally high uncompressed reference PPL is not by itself evidence of damaged candidate bytes. Repeated candidate-only degradation should be checked against the same reference, protocol and a healthy quantization. Loading failures, nonfinite metrics, OOM and ordinary low-bit loss require different diagnoses.

## Build or dependency failures

Use the [README build commands](workflows.md), including a fresh CMake configuration if a target is absent. Build directories, environments and caches can live outside the checkout. Native rows need the normal Python dependencies; only legacy HF preparation needs the `hf-tokenizer` extra. Never replace binaries or backend libraries underneath an active collection.
