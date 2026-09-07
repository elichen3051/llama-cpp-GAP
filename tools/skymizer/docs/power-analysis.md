# Power planning and pilot diagnostics

These CPU tools consume two complete, aligned metric collections. Read [paired comparison](compare.md) first. They share the reference, runtime, target and completion guards used by the comparison CLI. Do not copy only NPZ files into a new directory to bypass the collection contract.

Run examples from the repository root after setting `SKYMIZER_PYTHON`, `A`, `B` and `REPORTS` as shown in the [README](../README.md). The numerical values below illustrate syntax; they are not a recommended effect size for every model.

## Prospective power for an externally chosen effect

```bash
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/power_analysis.py \
  --candidate-a "$A" --candidate-b "$B" \
  --metric kld --weighting item \
  --token-caps 32 64 128 --sample-sizes 50 100 200 \
  --sesoi=-0.0001 --reps 2000 --outer-reps 200 \
  --out "$REPORTS/power.md" --output-json "$REPORTS/power.json"
```

SESOI is the smallest effect of scientific interest, expressed as candidate B minus A in the metric's units. Choose it independently of the observed pilot difference. For KLD, a negative value describes a lower candidate-B KLD. Without `--sesoi`, the tool reports precision and minimum detectable effect (MDE), not power under an assumed alternative.

The planner resamples whole paired items, preserving within-item prefix dependence. It evaluates the production paired Student-t endpoint across the requested sample-size and token-cap grid. `--effect-profile flat` is the default; `--effect-profile pilot` uses the pilot effect shape and requires an explicit `--reference-cap` present in `--token-caps`.

Caps must be positive and supported by the saved collection horizon. A short answer contributes only its available prefix; it is not padded with zeros. Sample sizes describe a hypothetical future sample and may exceed the pilot size, subject to the pilot representativeness assumption.

Reports include null-calibration diagnostics, Monte Carlo Wilson intervals and outer pilot-bootstrap sensitivity. These are distinct sources of uncertainty. Current outputs are exploratory design aids: null-calibration diagnostics do not automatically veto a design, zero-variance cells require inspection, outer intervals do not provide simultaneous guarantees for the whole grid, and the documented limitations of correct-direction outer bands remain. A layout refactor does not resolve these statistical limitations. See the implementation's module notes and [method description](../knowledge/seq-power-analysis.md).

## Observed prefix variance

```bash
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/variance_decomposition.py \
  --candidate-a "$A" --candidate-b "$B" \
  --metric kld --num-eval-tokens 128 \
  --output-json "$REPORTS/variance.json" > "$REPORTS/variance.md"
```

Supported metrics are `kld`, `reversed_kld`, `js_kld` and `nll`. This tool writes its human-readable report to stdout; it has no `--out` option. It needs at least three paired items and two scored positions per item.

Use the observed effect, variance and prefix curves to understand the pilot. The between/within-item decomposition is approximate and does not assume that longer continuations are stationary or independent. An observed-effect sample-size calculation is a diagnostic, not prospective power. A flat curve can also reflect early EOS; it is not proof that per-token information has saturated. See [nonstationary variance](../knowledge/variance-estimation-nonstationarity-theory.md).

## Stability within the finite pilot

```bash
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/random_subsample_power.py \
  --candidate-a "$A" --candidate-b "$B" \
  --mode reproducibility --sizes 25 50 75 --reps 100 \
  --out "$REPORTS/stability.md" --output-json "$REPORTS/stability.json"
```

Use sizes that fit the actual aligned pilot. The default samples without replacement and measures agreement with the full pilot's verdict. It uses all saved positions, mutually available base metrics and Student-t inference. It has no token-cap, metric-selection or alternative CI-method option. The legacy `--mode power` is conditional replication using the observed pilot effect; use `power_analysis.py --sesoi` for prospective planning. The retained `--bootstrap-iters` compatibility option does not change Student-t inference.

## Plan with the correct sampling unit

Saved text-corpus windows can be compared as windows, articles or contiguous blocks using the paired comparison tool. Grouping is not currently a selectable planning unit in these three planning CLIs. Do not claim article-level power by running a window-level planner on related windows. Likewise, multiple VLM presentations of one underlying problem are not automatically independent items.

For a fair cap comparison, retain a fixed common eligible cohort and show how many cases actually reach each cap. Record repetition exclusions, generation-cap hits, model/corpus identity, seed and candidate pair. The tools do not turn one pilot or one quantization pair into a general model-family conclusion.

The optional [numbered planning wrapper](workflows.md) keeps prospective sample sizes separate from finite-pilot subset sizes and skips unsupported variance metrics. Direct CLI commands remain available when only one diagnostic is needed.
