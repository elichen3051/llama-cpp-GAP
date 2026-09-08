# Paired comparison

The report compares paired candidate measurements after validating their collection and reference identities.

### `saved_metrics_paired_compare.py` — item-level paired verdict

The right tool when you have **two quantizations** of the same FP base
and want to decide which is statistically closer to FP. Takes two
`collect_kld.py` metric dirs that share the SAME reference, aggregates the
stored per-token metrics to one score per item, then tests the per-item
deltas (the item is the correct statistical unit, because tokens within
one answer are correlated).

Run from the repository root after the [shared setup](workflows.md). Set `CANDIDATE_A` and `CANDIDATE_B` to completed metric directories.

```bash
"$SKYMIZER_PYTHON" tools/skymizer/stats/cli/saved_metrics_paired_compare.py \
    --candidate-a "$CANDIDATE_A" --candidate-b "$CANDIDATE_B" \
    --label-a Q4_K_M --label-b Q4_1 --ci-method t \
    --out "$SKYMIZER_WORK/paired.md" --output-json "$SKYMIZER_WORK/paired.json"
```

Use `--metrics` to select endpoints, `--confidence-level` to change coverage, `--ci-method studentized|bca|percentile` for a bootstrap method, and `--bootstrap-iters`/`--seed` for its resampling. `--weighting item|token|both` controls reported weightings. `--start`/`--end` select paired rows, `--show-diagnostic-metrics` adds diagnostics, and `--help` lists the full interface. These are optional flags, not literal shell bracket notation.

`--num-eval-tokens N` scores only the first `min(N, stored_npos)` answer
positions per item (applied at read time after the per-item npos guard,
uniformly to both dirs; the stored files are unchanged). `-1`
(default) uses all stored positions. This is distinct from the
collection-time `num_eval_tokens` guard, which still requires the two
dirs to have been *collected* with the same cap.

For each base metric the report shows both candidate means and the item-weighted and token-weighted delta `b - a`. Only item-weighted endpoints have a paired CI, p-value, directional verdict or equivalence decision. Token-weighted blocks are descriptive only, including derived PPL and RMS values. The default item interval is Student-t without bootstrap resampling; optional bootstrap methods resample the same paired item indices across metrics.

JSON schema `vlm-paired-compare-v5` retains the v4 policy, which removes inference fields from token-weighted blocks and removes `weighting_consensus`. Descriptive blocks have `role: descriptive`. `--weighting token` controls display only and does not enable token-weighted inference.

> [!IMPORTANT]
> `inconclusive` means *the interval contains 0* — **not** that the two
> candidates are equivalent. Set beside `A closer` and `B closer` the old
> `no sig. diff.` wording read as a third finding; at the n = 25–75 sizes this
> tooling is used at it much more often meant *underpowered*. Every such row's
> `reason` now carries `equivalence_bound` = the tightest margin its own
> interval actually rules out. Pass `--equivalence-margin M` (in the primary
> metric's own units) to have the primary cell tested for equivalence
> properly: a CI lying entirely inside ±M is an equivalence result at that
> margin. A CI can exclude zero and still fit inside that practical margin: the directional verdict is retained and `equivalence_established=true` reports equivalence independently. If the CI also contains zero, the categorical verdict is `EQUIVALENT`. The JSON records the actual interval confidence, margin, bound and one-sided `equivalence_alpha=(1-confidence_level)/2`. A 95% interval therefore corresponds to one-sided alpha 0.025; it is more conservative than conventional alpha 0.05 TOST using a 90% interval.

### Fixed text corpora and grouped inference

The same entry point accepts full-window `collect_llm_kld.py` results prepared by `prepare_perplexity_corpus.py`. Use `--unit window`, `--unit article`, or `--unit block --block-windows 8`. Article and block modes require all original corpus windows and targets. They validate each paired window before merging records, preserve all scored targets, and use the resulting groups for degrees of freedom and bootstrap resampling. Window-level `item` mode remains the compatibility default for corpus data. See the [text bridge handover](text-bridge.md) for source bytes, native tokenization, BOS and boundary-token assignment.

For G groups, paired differences d_g and target counts w_g:

- Equal-group estimate: `mean(d_g)`; standard error: `sd(d_g, ddof=1)/sqrt(G)`.
- Token-weighted descriptive estimate: `sum(w_g*d_g)/sum(w_g)`; no paired inference.
- Pooled PPL is `exp(pooled mean NLL)`, never the arithmetic mean of per-group PPL values.

Fixed WikiText/PG windows and adjacent articles or blocks can remain dependent. CI and test calculations treat groups as independent sampling units; residual dependence can invalidate coverage and p-values. Grouping does not prove independence, and these are conditional comparisons of a fixed corpus, not automatically confirmatory inference to a population of independently sampled questions. Choose grouping, block size and the item-weighted primary metric before inspecting outcomes. Position buckets are disabled after article/block concatenation because a concatenated group position is not an original answer position.

The engine rejects fewer than two groups, nonfinite scores, invalid weights and unrepresentable derived PPL values instead of producing a verdict. The legacy random-subsampling and variance loaders use the main collection/runtime/pairing guards and refuse failed or nonfinite rows. Their planning results remain conditional on the observed panel. The variance decomposition omits token autocovariance; its sample-size calculation is a t-quantile approximation, not exact noncentral-t power inversion. Zero effect with zero variance has no defined SNR or finite detection sample size.

#### Interval construction (`--ci-method`, default `t`)

Per-item KLD deltas are right-skewed, and the plain percentile bootstrap
interval is only first-order accurate, so it under-covers in exactly the
sample range this tooling operates in. Four constructions are available:

- **`t`** (default) — the classical paired Student-t interval on the per-item deltas,
  `θ ± t_{n−1, 1−α/2}·SE`, with `p = P(|T_{n−1}| ≥ |θ/SE|)`. The SE is the
  closed form Efron & Tibshirani (1986) cite as the case where resampling is
  unnecessary (`s/sqrt(n)` item-weighted). **No bootstrap**: the result is a
  deterministic function of the deltas — no seed, no replicate count
  (`--bootstrap-iters` is ignored and the JSON records `0`). The paired test uses [SciPy `ttest_rel` and its confidence interval API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html). Shared distribution helpers use [SciPy `t.ppf` and `t.sf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html). See [statistical API references and independent verification](statistical-apis.md) for numerical limits and MATLAB, Julia and Wolfram correspondences.
- **`studentized`** — the bootstrap-t interval. Every replicate is
  divided by *its own* analytic standard error, so the interval is built from
  the pivotal quantity `t* = (θ* − θ)/SE*` and inverted:
  `[θ − t*_(1−α/2)·SE, θ − t*_(α/2)·SE]`. Only item weighting is supported.
- **`bca`** — bias-corrected and accelerated: corrects the median bias of the
  bootstrap distribution (`z0`) and its skew (the acceleration `a`, from the
  leave-one-**item**-out jackknife — the same exchangeable unit the bootstrap
  resamples).
- **`percentile`** — the first-order interval. Kept because the cross-repo
  golden fixtures pin it.

**Measured coverage.** All four methods share the point estimate and (for
`t` and `studentized`) the same analytic SE; they differ only in the
critical value. The study below drives the production code path
(`_build_weighting_block`) on right-skewed paired deltas with a **true delta
of zero** — per item a lognormal difficulty times a mean-zero lognormal
factor, so the delta's right tail comes from a few hard items, the mechanism
behind a KLD comparison — and is regenerated by
`verify_and_validation_scripts/measure_ci_coverage.py` (deterministic; the exact
command is printed under the table).

Nominal 95% interval, true delta 0, 1000 simulations, `--bootstrap-iters 4000` for the bootstrap methods, item weighting (`coverage (misses above / below the truth)`; width relative to `t`):

| n | skew | t | studentized | bca | percentile |
|---|---|---|---|---|---|
| 25 | 2.4 | 0.932 (0.009/0.059) | 0.928 (0.020/0.052) ×1.11 | 0.910 (0.030/0.060) ×0.97 | 0.914 (0.018/0.068) ×0.92 |
| 50 | 2.4 | 0.927 (0.013/0.060) | 0.924 (0.030/0.046) ×1.06 | 0.904 (0.040/0.056) ×0.99 | 0.913 (0.024/0.063) ×0.96 |
| 100 | 2.4 | 0.937 (0.013/0.050) | 0.937 (0.023/0.040) ×1.03 | 0.927 (0.029/0.044) ×1.00 | 0.930 (0.020/0.050) ×0.98 |
| 25 | 6.3 | 0.927 (0.002/0.071) | 0.890 (0.018/0.092) ×1.33 | 0.855 (0.035/0.110) ×1.02 | 0.889 (0.012/0.099) ×0.92 |
| 50 | 6.3 | 0.908 (0.004/0.088) | 0.886 (0.026/0.088) ×1.19 | 0.865 (0.035/0.100) ×1.04 | 0.890 (0.016/0.094) ×0.96 |
| 100 | 6.3 | 0.933 (0.006/0.061) | 0.911 (0.020/0.069) ×1.12 | 0.898 (0.031/0.071) ×1.05 | 0.924 (0.009/0.067) ×0.98 |

Generated by `verify_and_validation_scripts/measure_ci_coverage.py --sims 1000 --iters 4000 --seed 0`.

What to read off it:

- **No method reaches 0.95 at n = 25.** The failure is the same for all of
  them and it is one-sided: the misses are almost all *below* the truth. A
  sample that happens to hold few of the rare hard items has both a low mean
  and a low SD, so its interval is narrow *and* sits low. In a KLD report
  that reads as "the model with the catastrophic items looks better than it
  is" — a small sample systematically understates rare catastrophic damage,
  and no interval construction can see items that were not drawn. More
  independently sampled items can improve representation of rare events; changing the interval method cannot recover unobserved events.
- **The t interval is not worse than the bootstrap-t here** — equal or
  slightly higher coverage at every (n, skew), with a 3–33% narrower interval,
  and it has no Monte-Carlo noise at all: a borderline verdict cannot flip
  between seeds, which is why it is the default. (An earlier, uncommitted
  simulation had rated the bootstrap-t best; its generator was never
  checked in and could not be reproduced. The table above is the study of
  record.)
- The bootstrap-t buys its skew-awareness with replicate-level SEs that are
  themselves unstable in small skewed samples, which is where its extra
  width comes from without extra coverage. It remains the right tool when
  the sampling distribution of the statistic is far from t — e.g. per-item
  *tail* statistics — and BCa / percentile are kept for parity checks.

Bootstrap uses SciPy for percentile and BCa intervals. BCa uses SciPy's midrank tie handling. Bootstrap-t remains explicit because SciPy does not expose this method; any unusable pivot aborts instead of silently deleting a tail. Nonconstant two-item samples fail. Nominal resample floors are only prefilters: the reported endpoints each need ten strict-outside draws. Constant differences retain an explicitly labeled empirical point-interval convention, which does not establish zero population variance.

Bootstrap p-values invert the same fixed empirical CI family. Outside the supported confidence range, JSON records censoring and a conservative upper bound; markdown shows `<=`. This is not a Monte Carlo confidence bound and does not prove population calibration. Decisions use strict `p < alpha`, consistent with a closed CI containing its boundary. Unsupported optional buckets and tails retain descriptive estimates and a reason, with no CI, p or verdict. They still count in their declared Holm family through internal p=1 placeholders; invalid or nonfinite data remain fatal.

The coverage table above records the former bootstrap implementation. It does not validate the corrected implementation or its new unavailable-result policy. Repeat the coverage study with unavailable outcomes counted before relying on those historical bootstrap rates.

#### Multiplicity: one confirmatory endpoint, Holm over the rest

Every base metric has an item-weighted inference endpoint and a token-weighted descriptive summary. A complete VLMK v5 report has 15 tested endpoints: one primary and 14 exploratory endpoints (13 and 12 for v4).

The report designates exactly one primary endpoint with `--primary-metric` (default `kld`). `--primary-weighting` accepts only `item`. The other item-weighted endpoints carry Holm-Bonferroni-adjusted p-values across their family. Token-weighted summaries do not enter that family. The metric endpoints share observations and can be dependent; Holm controls family-wise error under arbitrary dependence when the input p-values are valid.

Adding EAR_64 to the default report expands the exploratory Holm family, so
adjusted p-values can change. Existing metric values, means, raw p-values and
CIs retain their definitions. Use the same explicit `--metrics` list for
reports with the same multiplicity family. Old dumps omit unavailable metrics
with a saved warning; explicitly requesting a missing metric requires recollection.

Each p-value is obtained by **inverting the interval that was actually built**
— solving for the confidence level whose endpoint lands on 0 — so `p ≤ α` and
"the CI excludes 0" cannot disagree, whichever `--ci-method` is in use.

**Weighting and the sampling unit.** Item weighting gives equal weight to each selected item or corpus group. Token weighting estimates the ratio of total metric sum to total target count, so longer groups contribute more. Under an appropriate sampling design, these estimate different population quantities: the expected group-average difference and the expected group-total difference divided by expected group size. Token weighting also reproduces corpus-pooled NLL/PPL aggregation. Neither weighting guarantees generalization or CI coverage.

Tokens within an answer are correlated, and questions may share images or source material. The t calculation and bootstrap treat the selected items or groups as independent units; that assumption must be assessed for the study. A fixed corpus or a pilot overlapping the final cohort does not become an independent validation set through resampling. Select the unit and item-weighted primary endpoint before inspecting candidate results. `--primary-weighting token` is rejected; token-weighted rows remain available for descriptive interpretation.

The report opens with a **`## Verdict summary`** listing just the confirmatory
endpoint and the exploratory cells that survive Holm, and a
**`## Reference corpus likelihood`** section carrying `nll_ref` / PPL(reference)
— `llama-perplexity`'s `PPL(base)` — outside the paired table, since there is
nothing paired about it.

| Base metric | Direction | What it measures |
|---|---|---|
| `nll` | lower better | Teacher-forced NLL of the candidate at FP target tokens |
| `kld` | lower better | Forward `KL(p_ref ‖ p_cand)` — mass-covering |
| `reversed_kld` | lower better | Reverse `KL(p_cand ‖ p_ref)` — mode-seeking |
| `js_kld` | lower better | Jensen-Shannon divergence (symmetric, bounded by ln 2) |
| `ear` | higher better | Expected Acceptance Rate ([arXiv:2605.02404](https://arxiv.org/abs/2605.02404)): per position `Σ_v min(p_ref, p_cand)` = `1 − TV distance`, summed over the full vocabulary and then averaged over scored positions. `EAR 0.99` ⇒ the two models emit the same token 99% of the time under optimal coupling — the speculative-decoding acceptance probability |
| `ear_64` / `ear_20` / `ear_10` / `ear_5` | higher better | The **reference's top-K share of EAR**: `Σ_{v ∈ top-K_ref} min(p_ref, p_cand)` with full-vocab probabilities, top-K_ref = the K ids with the largest reference logits (ties → lower id). Decomposes `ear` (`ear_5 <= ear_10 <= ear_20 <= ear_64 <= ear`); its ceiling is the reference's own top-K mass, so a flat reference caps it regardless of the candidate. K=64 requires VLMK v5+; K=5/10/20 requires v4+ |
| `ear_64_normalized` / `ear_20_normalized` / `ear_10_normalized` / `ear_5_normalized` | higher better | Same K ids, but both rows renormalized over exactly those ids (softmax of the K logits) before `Σ_k min(p̃_ref, p̃_cand)`. Answers "how well does the candidate reproduce the reference's relative preferences among its K most likely tokens" — `1.0` = identical shape on that set. Candidate mass outside the reference's top-K is ignored by design, so it is not monotone in K and should be read together with `ear`. K=64 requires VLMK v5+; normalized K=5/10/20 is present in v3+ |
| `same_top_rate` | higher better | Fraction of positions where `argmax_ref == argmax_cand` |
| `mse_dp` | lower better | `mean((100 * (p_cand(target) - p_ref(target)))^2)` in squared percentage points |

Derived (always shown unless hidden):

- `ppl` = `exp(nll)` per model (absolute perplexity; the `a`/`b` columns are
  `exp` of each side's mean `nll`). Always shown; `ΔPPL` is display-only — the
  statistically-tested PPL change is `ppl_ratio` (so its CI cell points there)
  and its verdict links to `nll`.
- `ppl_ratio` = `exp(Δnll)`, CI = `exp(Δnll CI)`, verdict links to `nll`.
- `rms_dp` = `sqrt(mse_dp)` (pp). Hidden by default; pass
  `--show-diagnostic-metrics` to reveal.

Reference- and candidate-only rows (no paired verdict — nothing to compare):

- `nll_ref` / `ppl_ref` — the **shared FP reference's own** teacher-forced NLL
  and perplexity at the same target tokens, i.e. `llama-perplexity`'s
  `PPL(base)`. Candidates A and B each score the reference during their own collection. The comparator checks their reference columns and reports `max_abs_side_difference`; strict mode rejects drift. `--allow-ref-drift` retains a diagnostic comparison but weakens the shared-reference interpretation.
- `mean_dp` (pp, **signed**) — each candidate's mean
  `p_cand(target) − p_ref(target)`. `mse_dp` and `rms_dp` are unsigned by
  construction, so a systematically over-confident candidate and a
  systematically under-confident one look identical in them. The sign is the
  whole point, and the pooled `Δp` ladder below (median included) shows the
  distribution behind it.
### Answer-position strata (exploratory)

The comparator emits an **`## Answer-position strata`** table for KLD and EAR using per-item means within each `--position-buckets` range (default boundaries `0 32 256`). The paired calculation uses the selected `--ci-method`; `t` is the default. An item contributes only when it has positions in that bucket, and absent buckets are counted rather than zero-filled. Article/block grouping disables these strata.

Position ranges can reveal effects that a whole-answer average hides. Their usefulness depends on the checkpoint, dataset, image policy and scoring horizon. The default boundaries are descriptive diagnostic choices; they do not establish a universal first-32-token effect. The comparison-time cap defaults to `-1`, meaning all stored positions.

Every stratum is exploratory. Holm adjustment applies within the strata family, separately from the main table, and does not turn data-chosen boundaries into a confirmatory analysis.

### Per-item KLD tails (exploratory — own Holm family)

The comparator emits a **`## Per-item KLD tails`** section: each item's OWN
p99 / p99.9 / maximum of its per-token `kld`, with the answer position (and,
when the producer supplied the `target` column, the target token id) it
occurred at — so a bad tail is traceable to the token that produced it. The
JSON (`per_item_tails.metrics.kld.items[*]`) lists every item with both
candidates' three levels, their witness positions, and for the interpolated
levels the two bracketing positions and the interpolation weight; the
markdown shows the ten worst items (by either candidate's max).

The quantile convention is the pooled ladder's — linear interpolation between
neighbouring order statistics, witness = the nearest one, ties to the earliest
position — applied to one item at a time, so a per-item p99 and the pooled p99
are one definition at two scopes. A short item cannot interpolate anywhere but
between its two largest tokens (p99 needs > 100 positions, p99.9 > 1000):
those items are flagged per level (`rests_on_top_two`) and counted in the
cell's `on top-2` column — for them the level is the max in all but name, so
read p99.9 next to the item lengths.

Across items each level is a per-item scalar and gets one paired test with
the report's `--ci-method` (unit: the item). The three cells are
**exploratory** and form their **own Holm family**, separate from the
answer-position strata and from the main verdict family (which keeps its size
and its power): a per-item maximum is a single-token statistic, far
heavier-tailed than a mean, so its interval reaches nominal coverage later
than the mean's, and the same right-skew argument as in the coverage table
applies with more force. These rows never carry the confirmatory endpoint's
α; they are the "which items / which tokens" companion to the pooled ladders
below, which remain the descriptive, corpus-scale view.

### Per-token distribution ladders (descriptive — no bootstrap, no CI)

After the verdict table the comparator emits a **`## Per-token distributions`**
section: for `kld`, `ear` and the signed `dp`, the full quantile ladder

```
max / 99.9% / 99.0% / 95.0% / 90.0% / median / 10.0% / 5.0% / 1.0% / 0.1% / min
```

computed over **all scored tokens of all items flattened into one multiset** —
the same scale and the same convention `llama-perplexity --kl-divergence` uses
for its `99.0% KLD` / `Maximum KLD` lines (linear interpolation between
neighbouring order statistics; upstream does it in float32, this tool in
float64). Each level is reported for both candidates with the raw `b − a`
difference and full provenance:

- the **witness** shown in the markdown table — `item@position`, the nearest
  of the two order statistics, where `position` is 0-based among that item's
  scored answer positions after any compare-time `--num-eval-tokens` cap;
- in the JSON, **both** interpolation neighbours (`lower`/`upper`) with their
  flattened ranks plus `fractional_rank` and `interpolation_weight_upper`, so
  the printed number is reconstructible — an interpolated quantile lies
  *between* two tokens and no single one holds its value.

Below each `kld` and `ear` ladder the report also prints a **tail composition**
table: which items hold the tokens at or beyond that candidate's own p99 /
p99.9 (or p1 / p0.1 for EAR) threshold, how many, what fraction of the item
that is, and — for the KLD tail — how bad EAR got at exactly those tokens. A
quantile says how large the tail is; this says where it came from, which in
practice is usually a handful of items.

> [!IMPORTANT]
> These rows carry **no confidence interval, no bootstrap and no verdict**, and
> the JSON records `"inference": "none"`. The pooled **maximum** has no
> consistent nonparametric bootstrap: a resample can only contain a subset of
> the observed items, so the replicate distribution is degenerate (a point mass
> at the observed value plus a one-sided tail). Measured coverage of a true zero
> delta under the exact null was **0.887 at n=30 and 0.860 at n=150** against a
> nominal 0.95 — about three times the intended Type-I rate, and *worsening* as
> items are added, so it is not fixable by collecting more data. The far
> quantiles (99%, 99.9%) rest on a handful of tokens in a handful of items and
> cannot support a verdict either. Confirmatory inference lives on the mean
> metrics above.

**Direction matters for reading the tails.** `kld` is lower-is-better, so its
degradation tail is the HIGH end (`99.0%` / `99.9%` / `Maximum`). `ear` is
higher-is-better (`EAR = 1 − TV`), so its degradation tail is the LOW end
(`1.0%` / `0.1%` / `Minimum`) — a literal "maximum EAR" sits at ~1.0 for any
candidate worth measuring and carries no information. The report bolds the
degradation tail of each ladder.

`dp` is signed, so both ends of its ladder are marked: an over-confident and
an under-confident candidate are different failures, and the **median** says
which way the mass moved. This is `llama-perplexity`'s Δp table, with the
witness column added.

The `ear` ladder needs the per-token `ear` column, which VLMK v1 dumps do not
have; on such a dir the `kld` and `dp` ladders are still emitted.

**Hard guards** (abort before paired inference and emit no verdict):

- Both directories must agree on kind, reference model, dataset/subset/split, sorting, stored evaluation cap, total-token budget, teacher-forcing chunk, context, batch/microbatch, SWA cache policy, GPU offload, inference/metric threads and flash attention. Executed binary/backend identity is also mandatory.
- VLM collections additionally require the same reference projector, media wrapper and effective image bounds. Native rows retain the recorded GGUF layout; legacy HF rows use their frozen sequential token layout. `n_prefill` identifies the first target in that frozen stream and does not by itself establish the same image embeddings or native positions.
- LLM collections additionally match `perplexity_window`, `corpus_protocol` and `corpus_windows_sha256`. Corpus article/block inference requires the full window set and all stored targets. See [text grouping](text-bridge.md#paired-windows-articles-and-blocks).
- The stored reference columns (`nll_ref`/`entropy_ref`/`argmax_ref`) must
  be bit-identical per item across the two dirs — the shared-reference
  premise. `--allow-ref-drift` downgrades this to a recorded warning and should not be used for a primary result. It does not bypass completion, runtime or execution-identity checks.
- Per-item `vocab`/`npos`/`target` must match across the two dirs.
- The available artifact-key sets must agree across inputs. Without an explicit
  override, a one-sided missing item is treated as missing data and hard-fails.
- The `SKIP_OVER_BUDGET` row-index sets recorded in every available manifest
  must be identical. A mismatch always hard-fails and cannot be overridden by
  `--allow-interaction`.
- Any non-finite metric, rejected/incomplete artifact, or `FAIL_*` collection
  row aborts; numerical and collection failures are never excluded as data.

**Soft warnings** (printed to stderr, run continues):

- Same-model candidates remain a warning. Missing collection or execution metadata is now a hard failure.
- `candidate-a` and `candidate-b` are the same model.
- Either candidate equals the reference (should only differ in quant).

**Explicit intersection mode**:

- By default, any one-sided missing artifact key hard-fails and no report or
  verdict is produced.
- `--allow-interaction` (alias `--allow-intersection`) explicitly permits a
  paired test on the clean key intersection; the missing keys and override are
  recorded in the report's `## Alignment` block.
- A common `SKIP_OVER_BUDGET` set is an intentional shared corpus exclusion and
  is also recorded in `## Alignment`; unequal skip sets still hard-fail.

Fails closed if fewer than 2 usable items remain (a single item yields a
zero-width bootstrap CI that would report spurious "significant" verdicts).

## What gets compared

VLM metrics compare answer-token distributions conditioned on the supplied prompt and images. LLM weights, projector weights, preprocessing, runtime and their interactions can all change those distributions; their effects do not form a general additive decomposition.

| Experiment | Hold constant across the compared arms | Vary |
| --- | --- | --- |
| LLM quantization | Reference, mmproj, prompt/images, target tokens, runtime and backend | Candidate LLM quantization |
| Projector quantization | Reference, candidate LLM, prompt/images, target tokens, runtime and backend | Candidate mmproj |
| Whole-package comparison | Reference, dataset/targets and declared execution protocol | The specified LLM/projector package |

A projector comparison measures its effect on answer-token fidelity conditional on the chosen LLM. It does not isolate a model-independent projector effect or establish task accuracy. NLL can also change with the projector; it is not inherently image-insensitive. Native reference rows avoid legacy HF prompt reconstruction, but still require exact generation and replay provenance.

Targets are the frozen trajectory tokens, which can be sampled rather than greedy, or the observed text-corpus tokens. KLD uses the full vocabulary; NLL and target-probability displacement use the target token. `same_top_rate` compares the two argmax IDs regardless of whether either equals the target. Candidate entropy is descriptive and is not a reference distance.

Every stored VLM record belongs to an answer-token position. The image/prompt prefill is conditioning, and this tool defines no metric over image embeddings. Both candidates must have identical target IDs and compatible native prefill positions. `--num-eval-tokens` derives a shared prefix from those saved records; it cannot create missing targets.

### Collection and executable verification

Saved-metrics comparisons require completed attempt records and a known, equal
`execution_identity` on both sides. That identity fingerprints the executed
scorer and loaded libraries; matching checkout HEAD values are insufficient.
The reader holds shared collection locks until all files are consumed and the
report is written. Missing rows from declared work, interrupted attempts, active
writers, and different executable/backend identities abort before statistics.
Legacy directories missing these records must be re-collected. `--allow-ref-drift`
does not bypass executable identity or completion checks. MTP provenance describes
the reference generator and does not enable MTP in the KLD scorer.
