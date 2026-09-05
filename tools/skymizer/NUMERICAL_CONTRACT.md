# NUMERICAL CONTRACT — tools/skymizer

Rules a refactor must NOT change. Every stored metric, CI, verdict and
report byte depends on them. A PR that moves code must leave all of these
bit-identical; a PR that deliberately changes one is a SEMANTIC change —
separate commit, re-baselined goldens, and this file updated in the same
diff. "Only a tiny difference" is not a refactor outcome; it is an
unreviewed algorithm change.

## 0. What "identical" means here

Numerical identity holds PER BUILD/BACKEND. Two divergences are by design
and are NOT violations: GPU run-to-run across different builds/backends,
and `--n-seq-max > 1` (GEMV vs GEMM decode kernels; measured ~1e-5/1e-6 —
recorded in docs/gotchas.md). Same-build, same-GPU, `--n-seq-max 1`
double-runs are bit-exact and the smokes assert it.

Two gates, never confused:

| Change type | Gate | Bit drift |
|---|---|---|
| Refactor PR | dumps/reports byte-identical (`cmp`/sha256) | any = reject |
| llama.cpp upstream sync | `--self-test` ×2, same-build double-run `cmp`-equal, full pytest | expected; legs of one comparison re-collected on one build, cross-build dirs never compared |

## 1. The rules

| # | Rule (do not change) | Where | Guarded by |
|---|---|---|---|
| 1 | C++ KLD kernel accumulation order: one ascending-`i` loop, six `double` accumulators; `jsd` takes TWO separate `+=` per iteration (ref term before cand); the both-zero `continue` defines `ear`'s summation order; `ent_r`/`ent_c` use `-=`; argmax tie-break is first-index (`>`) | `skymizer-vlmk-kernel.h` (`compute_kld_values`) | `--self-test` (production-regime case, rel 1e-9); dump sha256 protocol |
| 2 | `naive_kld_values` stays an INDEPENDENT implementation (probability space, no early-out) — never unify with `compute_kld_values`; their independence is the self-test | `skymizer-vlmk-kernel.h` | `--self-test` cross-check |
| 3 | Metric definitions frozen: forward `KL(p_ref‖p_cand)`, reverse, JSD via log-space mixture, `EAR = Σ min(p_ref,p_cand) = 1−TV`, NLL at the dataset target trajectory; genuine zero-probability ∞ KLD is never clipped. Score-dict derivations from stored columns (`same_top_rate`, `mse_dp`/`mean_dp` in pp, `ppl = exp(nll)`) live in `saved_metrics_paired_compare._side_scores` + the engine's derived blocks | C++ kernels; `cli/saved_metrics_paired_compare.py`; `compare/inference.py` | `--self-test` (closed forms); `test_saved_metrics_paired_compare.py`; engine golden |
| 4 | Bootstrap RNG recipe: `default_rng(seed)`, ONE `rng.integers(0, n, size=n)` per iteration, yielded in order. Every bootstrap CI in every report consumes this stream | `compare/inference.py` `_bootstrap_item_indices` | shared-index test in `test_paired_compare.py`; engine golden |
| 5 | Studentized endpoints are CROSSED: `[θ − t*_(1−α/2)·SE, θ − t*_(α/2)·SE]`; the "obvious" order yields a reflected interval. The (default) t interval is `θ ± t_{n−1,1−α/2}·SE` with `p = I_{ν/(ν+T²)}(ν/2, ½)` (two-sided tail taken directly, no `1 − F` cancellation) on the SAME analytic SE (`_statistic_and_se`: `s/√n` item-weighted, ratio linearization token-weighted) | `compare/inference.py` `_student_t_delta`; `compare/student_t.py` | `test_ci_methods.py`; `test_student_t.py`; engine golden |
| 6 | Pooled-ladder ranking uses `np.argsort(pooled, kind="stable")` — witness determinism | `compare/tokens.py` `_pooled_ladder` | ladder witness/tie tests in `test_ear_and_kld_tails.py`; engine golden |
| 7 | `dataset_fingerprint` hash-input order (`ds-v2`): baked into every stored `collect_meta.json`; reordering invalidates every `dataset_content_hash` | `lib/dataset_fingerprint.py` | `test_dataset_fingerprint.py`; `test_gap_sample_prep.py` (real rows) |
| 8 | VLMK metric columns stored `<f4`, indices `<i4` (v1 40-byte / v2 44-byte records); float32 storage is the dominant error term — widening upstream accumulators changes bytes for no gain | `lib/kld_metrics_io.py` | `test_kld_metrics_io.py` layout + dtype-strict tests |
| 9 | Float dtype ladder: float32 stored metric columns in, float64 accumulators for per-item means (`kld_metrics_io.item_means`, `_side_scores`) and every engine aggregation, float32 per-token report columns. No reduction axis or order change | `lib/kld_metrics_io.py`; `cli/saved_metrics_paired_compare.py`; `compare/engine.py` | `test_kld_metrics_io.py` (`item_means`); engine golden |
| 10 | Paired-statistics frame: the ITEM is the inference unit (the t test's n, the bootstrap's resampling unit); item- and token-weighted are different estimands, both always reported; primary = kld × item (pre-registered, `knowledge/quantization-eval-sop.md`); default CI = paired Student-t (`--ci-method t`, no bootstrap, seed-free), the three bootstrap constructions optional; Holm over the exploratory family; p-values obtained by inverting the interval actually built | `compare/inference.py`, `compare/contracts.py` | `test_multiplicity.py`; `test_ci_methods.py`; engine golden |
| 11 | Per-item tails (`per_item_tails`): each item's p99 / p99.9 / max of its per-token `kld` via `_pooled_ladder` applied to that item alone (same interpolation + witness rule as #6; `max` exact); tested item-weighted with the report's CI method in their OWN Holm family; never a per-item score key, never in the primary family. The `target` annotation column rides beside the metric columns (int32, one id per scored position) | `compare/tokens.py` `_per_item_tail_blocks`; `cli/saved_metrics_paired_compare.py` (target column) | `test_per_item_tails.py`; engine golden |
| 12 | `compare/student_t.py` stays scipy-free and in the tail form: `I_x` by Lentz's continued fraction with `1 − x` formed directly by the caller, the quantile by bracketed Newton in the tail domain. Every t endpoint and p in every report comes from it; swapping in a library changes the last bits of every golden | `compare/student_t.py` | `test_student_t.py` (closed forms + scipy oracle when installed); engine golden |

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
