# NUMERICAL CONTRACT — tools/skymizer

Rules a refactor must NOT change. Every stored metric, CI, verdict and
report byte depends on them. A PR that moves code must leave all of these
bit-identical; a PR that deliberately changes one is a SEMANTIC change —
separate commit, re-baselined goldens, and this file updated in the same
diff. "Only a tiny difference" is not a refactor outcome; it is an
unreviewed algorithm change.

## 0. What "identical" means here

Numerical identity is assessed on the same inputs, build/backend, GPU and execution shape. The KLD scorers use one sequence. Reference generation can use parallel sequences, and generation/replay shapes can differ. Neither equal seeds nor equal CLI flags alone prove bit identity; compare recorded values and executable/backend identities.

Two gates, never confused:

| Change type | Gate | Bit drift |
|---|---|---|
| Refactor PR | dumps/reports byte-identical (`cmp`/sha256) | any = reject |
| llama.cpp upstream sync | `--self-test` ×2, same-build double-run `cmp`-equal, full pytest | expected; legs of one comparison re-collected on one build, cross-build dirs never compared |

## 1. The rules

| # | Rule (do not change) | Where | Guarded by |
|---|---|---|---|
| 1 | C++ KLD kernel accumulation order: one ascending-`i` loop, six `double` accumulators; `jsd` takes TWO separate `+=` per iteration (ref term before cand); the both-zero `continue` defines `ear`'s summation order; `ent_r`/`ent_c` use `-=`; argmax tie-break is first-index (`>`); the EAR_K family (v4: K=5/10/20; v5 adds K=64 without changing those prefixes) runs AFTER that loop as a separate pass (`select_ref_topk`: one ascending-`i` insertion pass, strict `>` on entry and strict `<` while shifting so ties keep ascending-index order; `ear_topk_values`: `ear_K` = ascending-`j` sum of `exp(min(lp_r, lp_c))` with the full-row `log_z`, `ear_K_normalized` = double softmax over exactly the K selected slots per side, ascending-`j` sums) | `core/skymizer-vlmk-kernel.h` (`compute_kld_values`) | `--self-test` (production-regime case, rel 1e-9); dump sha256 protocol |
| 2 | `naive_kld_values` stays an INDEPENDENT implementation (probability space, no early-out) — never unify with `compute_kld_values`; their independence is the self-test | `core/skymizer-vlmk-kernel.h` | `--self-test` cross-check |
| 3 | Metric definitions frozen: forward `KL(p_ref‖p_cand)`, reverse, JSD via log-space mixture, `EAR = Σ min(p_ref,p_cand) = 1−TV`, NLL at the dataset target trajectory; genuine zero-probability ∞ KLD is never clipped. Score-dict derivations from stored columns (`same_top_rate`, `mse_dp`/`mean_dp` in pp, `ppl = exp(nll)`) live in `saved_metrics_paired_compare._side_scores` + the engine's derived blocks | C++ kernels; `stats/cli/saved_metrics_paired_compare.py`; `stats/inference.py` | `--self-test` (closed forms); `test_saved_metrics_paired_compare.py`; engine golden |
| 4 | Bootstrap RNG recipe: `default_rng(seed)`, ONE `rng.integers(0, n, size=n)` per iteration, yielded in order. Every bootstrap CI in every report consumes this stream | `stats/inference.py` `_bootstrap_item_indices` | shared-index test in `test_paired_compare.py`; engine golden |
| 5 | Studentized endpoints are crossed: `[theta - t*_(1-alpha/2)*SE, theta - t*_(alpha/2)*SE]`. The default paired t-test and CI use SciPy `ttest_rel` on the item differences, with SE `s/sqrt(n)`, df `n-1` and a two-sided tail. Input and variance guards precede the package call; zero-spread policy is explicit. | `stats/inference.py` `_student_t_delta`; `stats/student_t.py` | `test_ci_methods.py`; `test_student_t.py`; engine golden |
| 6 | Pooled-ladder ranking uses `np.argsort(pooled, kind="stable")` — witness determinism | `stats/tokens.py` `_pooled_ladder` | ladder witness/tie tests in `test_ear_and_kld_tails.py`; engine golden |
| 7 | `dataset_fingerprint` hash-input order (`ds-v3`, including native generation provenance): baked into every stored `collect_meta.json`; reordering invalidates every `dataset_content_hash` | `lib/dataset_fingerprint.py` | `test_dataset_fingerprint.py`; `test_gap_sample_prep.py` (real rows) |
| 8 | VLMK metric columns stored `<f4`, indices `<i4` (v1 40-byte / v2 44-byte / v3 56-byte / v4 68-byte / v5 76-byte records); float32 storage is the dominant error term — widening upstream accumulators changes bytes for no gain | `lib/kld_metrics_io.py` | `test_kld_metrics_io.py` layout + dtype-strict tests |
| 9 | Float dtype ladder: float32 stored metric columns in, float64 accumulators for per-item means (`kld_metrics_io.item_means`, `_side_scores`) and every engine aggregation, float32 per-token report columns. No reduction axis or order change | `lib/kld_metrics_io.py`; `stats/cli/saved_metrics_paired_compare.py`; `stats/engine.py` | `test_kld_metrics_io.py` (`item_means`); engine golden |
| 10 | Paired-statistics frame: the ITEM is the inference unit (the t test's n, the bootstrap's resampling unit); item-weighted endpoints support inference; token-weighted means and differences are descriptive only, with no CI, p-value, verdict, equivalence or weighting consensus; primary = kld x item by default (only item primary weighting is accepted); default CI = paired Student-t (`--ci-method t`, no bootstrap, seed-free), the three bootstrap constructions optional; Holm over the exploratory family; p-values obtained by inverting the interval actually built | `stats/inference.py`, `stats/contracts.py` | `test_multiplicity.py`; `test_ci_methods.py`; engine golden |
| 11 | Per-item tails (`per_item_tails`): each item's p99 / p99.9 / max of its per-token `kld` via `_pooled_ladder` applied to that item alone (same interpolation + witness rule as #6; `max` exact); tested item-weighted with the report's CI method in their OWN Holm family; never a per-item score key, never in the primary family. The `target` annotation column rides beside the metric columns (int32, one id per scored position) | `stats/tokens.py` `_per_item_tail_blocks`; `stats/cli/saved_metrics_paired_compare.py` (target column) | `test_per_item_tails.py`; engine golden |
| 12 | SciPy is pinned as a runtime dependency. Shared Student-t quantiles and tails call `t.ppf` and `t.sf`; Wilson intervals call `binomtest(...).proportion_ci(method="wilson")`. Library migrations are semantic changes: independently check formulas and published values before updating goldens. API references belong in function docstrings. | `stats/student_t.py`; `stats/inference.py`; `stats/power.py`; [references](docs/statistical-apis.md) | Closed-form t and Wilson tests, published MATLAB values, normal-null simulation; engine golden |

## 2. Acceptance ladder for equivalence claims

| Comparison | Gate |
|---|---|
| Pure code motion, same CPU/numpy | bit-exact: array equality, `float.hex()` golden, byte-diff of reports |
| VLMK `.bin` → `.npz` conversion | dtype/shape/header/value exact; fixture hash unchanged |
| C++ common-code extraction | record/dump bytes exact on same inputs (`sha256sum`) + `--self-test` |
| Report/renderer refactor | golden markdown exact (host-volatile fields excluded via `--omit-host-metadata`) |

## 3. The oracles

1. `tests/test_paired_compare_engine_golden.py` — full `compare_items`
   surface, 4 CI methods, `float.hex()` exact (generator:
   `tests/data/gen_paired_compare_golden.py`).
2. `tests/test_paired_compare_golden.py` — legacy percentile-path pin (1e-12).
3. `--omit-host-metadata` byte-diff: same command twice → `cmp` clean
   (`SKYMIZER_REPRODUCIBLE_REPORT=1`).
4. `llama-{llm,vlm}-kld --self-test` (metric kernel closed forms + the
   independent naive reference, no models needed).
5. Collector artifacts: `sha256sum` of `metrics/`, `collect_meta.json`,
   `manifest.csv` (no timestamps inside).


## 4. Collection identity and completion

The native-reference v2 integration changes identity/status contracts, not metric
formulas or VLMK record bytes. `ds-v3` adds native generation metadata and
request/sampling fields. `gguf-shards-sampled-v1` fingerprints every ordered
model shard. Native manifests bind the producer's target vocabulary to both
scorer vocabularies before weight loading. MTP heads belong to generation
provenance; KLD is main-model teacher forcing.

A formal saved-metrics comparison requires equal known execution identities
(actual binary, loaded libraries, and environment) and completed declared work.
An unfinished append, including a surviving valid NPZ with no terminal status,
is not a complete collection. Shared reader locks exclude concurrent writers.
Old directories without the new provenance and attempt records are not formal
comparison inputs and must be re-collected.


## 5. EAR_64 extension (VLMK v5)

`ear_64` sums `min(p_ref, p_cand)` over the reference's top `min(64, vocab)`
logits, ties by lower token ID, using full-vocabulary probabilities.
`ear_64_normalized` independently normalizes both rows over that same set
before summing minima. It returns 0 for a side with no selected support;
finite selected logits use local softmax even if their full-row mass underflows.
Only the unnormalized family is monotone: `ear_5 <= ear_10 <= ear_20 <= ear_64 <= ear`.
For `vocab <= 64`, both new metrics equal full EAR up to rounding.

V5 appends two float32 fields after the complete 68-byte v4 prefix, including
the integer IDs. Old field offsets and computations remain unchanged.
Readers retain v1-v4 layouts and never fabricate missing EAR_64 columns.
Collectors write v5 only; older outputs need recollection for the new metrics.

The default comparison now has 15 item-weighted inference endpoints, of which 14 are exploratory (VLMK v4: 13 and 12). The 15 token-weighted summaries are descriptive. This deliberately changes the Holm adjustment family, not existing
raw metrics, means, raw p-values or CIs. Explicit-metric golden reports retain
their old metric lists and remain unchanged.

## 6. Classic PPL corpus bridge and grouped comparisons

`--perplexity-window` is an opt-in llm-kld protocol. It decodes the full even-sized window in one batch, including the final unused logit row, and scores positions `L/2 + 1 .. L - 1`. The 512-token protocol has 255 targets, `n_prefill=257` and `n_past_actual=512`. It preserves native full-stream tokenization, per-window BOS replacement and the dropped incomplete tail. The ordinary teacher-forcing path retains its original decode shape.

Corpus protocol and ordered window maps are immutable collection identity fields. Article and contiguous-block comparisons concatenate the original per-target records before computing group means; grouping cannot change the corpus, BOS handling, targets or pooled token estimates. Only articles with scored targets form groups. The statistical unit is the selected window, article or block, and inference is conditional on treating those groups as independent sampling units. Residual dependence can invalidate coverage and p-values.

`llama-perplexity` saved references use clipped uint16 log probabilities. Compare exact tokens/windows/targets and uncompressed mean NLL across tools; report saved-reference quantization effects and both KLD values separately. The bridge verifier holds the collection reader lock for its complete read.

The statistical validation update preserves existing finite-input numerical results. Directional evidence and interval-inclusion equivalence are separate output facts; a 95% interval wholly inside a margin corresponds to TOST with each one-sided test at alpha 0.025. Invalid or non-finite input and unrepresentable derived values fail explicitly. Approximate observed-effect sample-size diagnostics do not replace prospective power analysis with an externally chosen effect.


## 7. Item-only inference (comparison schema v4)

Token-weighted rows retain their means, differences and derived PPL/RMS values, but have `role: descriptive` and no inference fields. There is no weighting consensus. Only item-weighted endpoints enter Holm correction, primary selection, equivalence decisions and paired-test power planning. Direct token-weighted inference calls and token primary/power CLI options fail explicitly.

This deliberately changes report structure and the Holm family. Item-weighted raw statistics and bootstrap resampling retain their definitions. The legacy percentile fixture still checks item intervals and both point estimates; the full-engine golden is re-baselined after independent policy and numerical checks.

## 8. SciPy statistical primitives

The handwritten incomplete-beta/Student-t distribution and variance planner quantile approximation are replaced by SciPy. Production item paired inference uses `ttest_rel` and its confidence interval API. Scalar power intervals reuse that guarded implementation; Wilson intervals share an explicit `method="wilson"` call. Bootstrap constructions and Holm adjustment are unchanged.

This is an intentional numerical migration. Student-t endpoints, p-values and their downstream transforms can differ in their final bits; item and token point estimates and bootstrap results must remain unchanged. Closed-form checks and published MATLAB quantiles validate the replacement independently before re-baselining the engine golden. The exact package APIs, Julia correspondences and numerical tolerances are recorded in function docstrings and [statistical API verification](docs/statistical-apis.md).

## 9. Prospective power schema v2

Empirical item resampling remains the primary power model. Gaussian nuisance sensitivity uses SciPy noncentral-t power in the assumed direction only; its MDE uses dimensionless SciPy root finding. The pilot-SD interval width is labeled as a plug-in quantity. Public counts and grids reject non-integers before simulation. Original and outer pilot samples share the same numerical-resolution rule; unresolved original variance aborts planning, and unresolved outer draws disable the nuisance band.

Detected null inflation excludes a cell from all required-N crossings. Remaining cells have no detected inflation, without a calibration guarantee. The nuisance crossing also requires the ordinary MC lower-bound condition. Monte Carlo intervals and crossings are pointwise, not simultaneous over the grid. Reports preserve the selected item keys, their digest, row selection and metric direction.
