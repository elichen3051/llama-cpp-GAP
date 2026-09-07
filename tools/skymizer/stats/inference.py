# stats/inference.py -- The paired statistics engine: decision table, SE/BCa/studentized interval
# construction, two-sided p by interval inversion, Holm, the shared bootstrap
# index stream, weighting blocks and the derived ppl/rms blocks.
# NUMERICAL_CONTRACT.md #5/#6.
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ttest_rel

from stats.contracts import CI_METHODS, DEFAULT_CI_METHOD, NonFiniteMetricError
from stats.student_t import t_two_sided_p

# Paired statistics engine (ported from llm_quant_fidelity/paired_compare.py)
# --------------------------------------------------------------------------- #
def _format_confidence_level(confidence_level: float) -> str:
    """"95%" / "99%" / "99.5%" -- the level as it appears in every verdict
    reason string, table header and note, so a report produced at a
    non-default --confidence-level can never contradict itself. Trailing
    zeros are trimmed (0.95 -> "95%", not "95.0%")."""
    pct = confidence_level * 100.0
    text = f"{pct:.10g}"
    return f"{text}%"


def _compute_decision(
    *,
    delta_estimate: float,
    ci_lower: float,
    ci_upper: float,
    null_value: float,
    score_direction: str,
    confidence_level: float = 0.95,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    """Single source of truth for the verdict truth table.

    `null_value` is 0.0 for additive delta metrics or 1.0 for ratio metrics.
    The CI is treated as a closed interval; `lower <= null <= upper` counts
    as containing null (boundary equality conservative).

    `confidence_level` only labels the reason string -- the interval itself
    was already built at that level by the caller. It is a keyword with a
    default so the JSON `reason` can never disagree with the summary bullet
    (they used to, at any --confidence-level other than 0.95).

    A CI containing the null is reported as "inconclusive", NOT as "no sig.
    diff.". Set symmetrically beside "A closer" and "B closer" the old
    wording read as a third finding -- "these two are the same" -- which it
    never was: at the n = 25..75 sizes this tooling is used at it much more
    often meant "underpowered". Absence of evidence is not evidence of
    absence, and nothing here tested for it.

    What the data DO support is reported instead: `equivalence_bound` =
    max(|lower|, |upper|), the tightest margin the interval rules out. By the
    interval-inclusion form of TOST, a CI contained in (-m, m) IS an
    equivalence result at margin m; pass `equivalence_margin` and a cell that
    clears it is reported as EQUIVALENT rather than merely inconclusive.
    Statistical difference and equivalence are separate facts. Directional
    verdicts retain priority when the CI excludes the null. Interval inclusion
    uses this CI's confidence level, with one-sided alpha = (1 - level) / 2.
    """
    if score_direction not in ("lower_is_better", "higher_is_better"):
        raise ValueError(
            "score_direction must be 'lower_is_better' or 'higher_is_better'; "
            f"got {score_direction!r}"
        )

    if not all(math.isfinite(v) for v in (delta_estimate, ci_lower, ci_upper, null_value)):
        raise NonFiniteMetricError("non-finite estimate or confidence interval; paired comparison aborted")
    if ci_lower > ci_upper:
        raise ValueError("confidence interval lower endpoint exceeds upper endpoint")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if equivalence_margin is not None and (not math.isfinite(equivalence_margin) or equivalence_margin <= 0.0):
        raise ValueError("equivalence_margin must be finite and positive")
    contains_null = ci_lower <= null_value <= ci_upper
    is_delta = null_value == 0.0
    fmt = (lambda v: f"{v:+.6f}") if is_delta else (lambda v: f"{v:.6f}")
    null_str = "0" if is_delta else f"{null_value:g}"
    bounds_str = f"[{fmt(ci_lower)}, {fmt(ci_upper)}]"
    level = _format_confidence_level(confidence_level)
    bound = max(abs(ci_lower - null_value), abs(ci_upper - null_value))
    equivalence = {
        "equivalence_bound": bound,
        "equivalence_margin": equivalence_margin,
        "equivalence_established": None if equivalence_margin is None else bound < equivalence_margin,
        "equivalence_alpha": None if equivalence_margin is None else (1.0 - confidence_level) / 2.0,
    }

    if contains_null:
        out = {
            "verdict": "inconclusive",
            "statistically_distinguishable_from_null": False,
            "null_value": null_value,
            "confidence_level": confidence_level,
            **equivalence,
            "reason": (f"{level} CI {bounds_str} contains {null_str}; the data "
                       f"bound |Δ| ≤ {bound:.6g} and establish nothing "
                       "tighter"),
        }
        if equivalence_margin is not None and bound < equivalence_margin:
            out["verdict"] = "EQUIVALENT"
            out["reason"] = (
                f"{level} CI {bounds_str} lies inside ±{equivalence_margin:g} "
                "(TOST by interval inclusion): equivalent at that margin")
        return out

    b_smaller = delta_estimate < null_value
    if score_direction == "lower_is_better":
        verdict = "B closer" if b_smaller else "A closer"
    else:
        verdict = "A closer" if b_smaller else "B closer"
    reason = f"{level} CI {bounds_str} excludes {null_str}"
    if equivalence["equivalence_established"]:
        reason += f"; CI also lies inside +/-{equivalence_margin:g} (equivalent at that margin)"
    return {
        "verdict": verdict,
        "statistically_distinguishable_from_null": True,
        "null_value": null_value,
        "confidence_level": confidence_level,
        **equivalence,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Interval construction
# --------------------------------------------------------------------------- #


def _statistic_and_se(diffs: np.ndarray, weights: np.ndarray, weighting: str):
    """Item mean and analytic SE; reject invalid arithmetic before zero-spread handling.

    Require normal SE^2 for nonconstant input: dividing subnormal variance before sqrt loses precision.
    Paired t SE definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    if weighting != "item":
        raise ValueError("paired inference requires weighting='item'; token weighting is descriptive only")
    n = diffs.size
    if n < 2:
        raise ValueError("paired comparison requires at least two items")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta = float(diffs.mean())
        se = float(diffs.std(ddof=1) / math.sqrt(n))
    if not math.isfinite(theta) or not math.isfinite(se):
        raise NonFiniteMetricError("non-finite estimate or standard error; paired comparison aborted")
    if se < math.sqrt(np.finfo(float).tiny) and np.any(diffs != diffs[0]):
        raise NonFiniteMetricError("standard error underflow for nonconstant differences; paired comparison aborted")
    return theta, se


def _bca_shape(boot: np.ndarray, estimate: float, jackknife: np.ndarray):
    """(z0, a) -- BCa's bias correction and acceleration, or None when either
    is undefined. Shared by the interval and by the p-value that inverts it,
    so the two can never be built from different shape parameters."""
    if boot.size < 2 or jackknife.size < 2:
        return None
    below = float(np.count_nonzero(boot < estimate)) / boot.size
    if not (0.0 < below < 1.0):
        return None                      # z0 = +/-inf: no usable bias correction
    z0 = NormalDist().inv_cdf(below)
    centered = jackknife.mean() - jackknife
    try:
        denom = float((centered ** 2).sum()) ** 1.5
    except OverflowError as error:
        raise NonFiniteMetricError("non-finite BCa acceleration; paired comparison aborted") from error
    if not np.isfinite(denom):
        raise NonFiniteMetricError("non-finite BCa acceleration; paired comparison aborted")
    if denom <= 0.0:
        return None                      # constant jackknife: no acceleration
    a = float((centered ** 3).sum()) / (6.0 * denom)
    if not np.isfinite(a):
        raise NonFiniteMetricError("non-finite BCa acceleration; paired comparison aborted")
    return z0, a


def _bca_endpoints(boot: np.ndarray, estimate: float, jackknife: np.ndarray,
                   confidence_level: float):
    """Bias-corrected and accelerated (BCa) percentile levels, or None when the
    sample is degenerate and BCa is undefined.

    The plain percentile interval is only FIRST-order accurate and carries two
    biases the per-item KLD deltas actually exhibit:

      * median bias -- the bootstrap distribution is not centred on the
        estimate; corrected by z0 = Phi^-1(fraction of replicates below it).
      * skew -- per-item deltas are right-skewed, so the two tails need
        different lengths; corrected by the acceleration `a`, estimated from
        the leave-one-ITEM-out jackknife (the exchangeable unit, same as the
        resample unit).

    Measured coverage of a nominal 95% interval at n=25 on right-skewed
    per-item deltas (1000 sims, B=4000, the same run quoted in this file's
    header): 0.905 at skew 2.3 and 0.869 at skew 6.2 with the percentile
    method -- a real alpha of 9-13%, not 5%, exactly in the n=25..75 range the
    power tooling recommends. BCa moves those to 0.908 / 0.891; the
    studentized interval to 0.936 / 0.923. (That run's generator was never
    committed; the reproducible study in
    verify_and_validation_scripts/measure_ci_coverage.py, tabulated in
    docs/compare.md, is what the current default -- the t interval -- rests
    on.)

    Returns (alpha_lo, alpha_hi) in (0, 1), or None if z0 or `a` cannot be
    formed (all replicates on one side of the estimate, or a jackknife with no
    spread). The caller then falls back to the percentile interval and records
    that it did.
    """
    shape = _bca_shape(boot, estimate, jackknife)
    if shape is None:
        return None
    z0, a = shape
    alpha = 1.0 - confidence_level
    out = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = NormalDist().inv_cdf(q)
        adj = z0 + z
        denom_a = 1.0 - a * adj
        if denom_a <= 0.0 or not np.isfinite(denom_a):
            return None                  # the acceleration blew the map up
        level = NormalDist().cdf(z0 + adj / denom_a)
        if not (0.0 < level < 1.0):
            return None
        out.append(level)
    lo, hi = out
    if not lo < hi:
        return None
    return lo, hi


def _add_one_two_sided(values, pivot) -> float:
    """Symmetric "add-one" two-sided achieved significance level of `pivot`
    against a replicate sample:

        p = 2 min( (1 + #{v <= pivot}) / (B+1), (1 + #{v >= pivot}) / (B+1) )

    capped at 1. The +1/(B+1) form is the standard bootstrap ASL: it can
    never return 0 (a B-replicate bootstrap cannot resolve a p below its own
    2/(B+1) floor, and Holm needs a usable number), and counting BOTH tails
    inclusively keeps it exactly symmetric under negation -- a one-sided
    `<=` count alone makes an all-negative replicate set report half the p of
    an all-positive one."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 1.0
    lo = (float(np.count_nonzero(v <= pivot)) + 1.0) / (n + 1.0)
    hi = (float(np.count_nonzero(v >= pivot)) + 1.0) / (n + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _two_sided_p(boot, estimate, *, ci_method, z0=None, accel=None,
                 t_star=None, delta_se=None, df=None):
    """Two-sided achieved significance level for H0: delta == 0, obtained by
    INVERTING the very interval the verdict reads -- find the confidence level
    at which the interval's endpoint lands exactly on 0. So "p <= alpha" and
    "the CI excludes 0" can never disagree, whichever --ci-method is in use.

    t:           p = P(|T_df| >= |estimate / SE|), the classical paired t
                 test with df = n - 1 -- exactly the level at which
                 estimate -+ t_{df, 1-p/2} SE touches 0.
    percentile:  p = 2 min(F(0), 1 - F(0)) with F the replicate ECDF.
    studentized: the same against the PIVOT: p = 2 min(G(T0), 1 - G(T0)) with
                 G the ECDF of t* and T0 = estimate / SE.
    bca:         the percentile form pushed through BCa's level transform,
                 solved for the alpha whose endpoint hits 0:
                     w = Phi^-1(F(0)),  y = (w - z0) / (1 + a (w - z0)),
                     p = 2 min(Phi(y - z0), 1 - Phi(y - z0)).
                 With z0 = 0 and a = 0 this collapses to the percentile form,
                 as it must.

    Returns None when the pieces are unavailable."""
    if ci_method == "t":
        if df is None or df < 1 or not delta_se:
            return None
        return t_two_sided_p(estimate / delta_se, df)
    boot = np.asarray(boot, dtype=float)
    if ci_method == "studentized":
        if t_star is None or not delta_se:
            return None
        return _add_one_two_sided(t_star, estimate / delta_se)
    if boot.size == 0:
        return None
    if ci_method != "bca" or z0 is None or accel is None:
        return _add_one_two_sided(boot, 0.0)
    # BCa: the transform needs a CDF value, so use the mid-rank ECDF at 0 --
    # symmetric under negation and never exactly 0 or 1.
    n = boot.size
    u = (float(np.count_nonzero(boot < 0.0))
         + 0.5 * float(np.count_nonzero(boot == 0.0)) + 0.5) / (n + 1.0)
    w = NormalDist().inv_cdf(u)
    denom = 1.0 + accel * (w - z0)
    if denom == 0.0 or not np.isfinite(denom):
        return None
    q = NormalDist().cdf((w - z0) / denom - z0)
    p = 2.0 * min(q, 1.0 - q)
    # Floor at the same 2/(B+1) resolution the add-one form has, so the three
    # methods' p-values are on one scale and Holm never sees a 0.
    return float(min(1.0, max(p, 2.0 / (n + 1.0))))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returned in input order.

    Controls the FAMILY-WISE error rate under ARBITRARY dependence, which is
    what this family needs: kld / reversed_kld / js_kld are three
    functionals of the same pair of distributions. Nothing here is
    independent, so Benjamini-Hochberg's assumptions do not hold and
    Holm's do.

    p_adj_(k) = max_{j <= k} (m - j) * p_(j), clipped to 1 -- the running max
    enforces monotonicity, so a cell can never be adjusted below one that had
    a smaller raw p."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        out[i] = min(1.0, running)
    return out


def _bootstrap_item_indices(seed: int | None, n_items: int, iters: int):
    """THE item-resample stream every bootstrap in one report consumes:
    default_rng(seed), then one rng.integers(0, n_items, size=n_items) draw
    per iteration, yielded in order. Every bootstrapped statistic in one
    report iterates this generator, so "all statistics resample identical item
    sets" holds by construction rather than by call sites agreeing on the rng
    recipe.
    Do not change the recipe: the golden-fixture parity tests pin it."""
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        yield rng.integers(0, n_items, size=n_items)


def _student_t_delta(diffs: np.ndarray, weights: np.ndarray, *,
                     weighting: str, confidence_level: float) -> dict[str, Any]:
    """SciPy paired t-test on B-A differences versus zero, with explicit zero-spread policy.

    Uses ttest_rel(diffs, zeros).pvalue and confidence_interval(confidence_level).
    SciPy API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    MATLAB ttest(B, A, 'Alpha', 1-confidence_level): https://www.mathworks.com/help/stats/ttest.html
    Julia OneSampleTTest(B, A): https://juliastats.org/HypothesisTests.jl/stable/parametric/#t-test
    Constant differences use a point CI and p=1 for zero, p=0 otherwise, recorded in fallback.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    n = int(diffs.size)
    df = n - 1
    delta, delta_se = _statistic_and_se(diffs, weights, weighting)
    fallback = None
    if delta_se > 0.0:
        test = ttest_rel(diffs, np.zeros_like(diffs), alternative="two-sided", nan_policy="raise")
        if not math.isfinite(test.statistic):
            raise NonFiniteMetricError("non-finite paired t statistic; paired comparison aborted")
        lower, upper = test.confidence_interval(confidence_level)
        p_value = float(test.pvalue)
    else:
        lower = upper = delta
        p_value = 1.0 if delta == 0.0 else 0.0
        fallback = ("the sample has no spread (identical deltas), "
                    "so the t interval is the point estimate and p is its "
                    "SE -> 0 limit")
    if not all(math.isfinite(v) for v in (lower, upper, p_value)):
        raise NonFiniteMetricError("non-finite confidence interval or p-value; paired comparison aborted")
    ci = {
        "method": f"paired_{weighting}_weighted_student_t",
        "ci_method": "t",
        "confidence_level": confidence_level,
        "lower": float(lower),
        "upper": float(upper),
        "standard_error": delta_se,
        "degrees_of_freedom": df,
        "bootstrap_iters": 0,
        "contains_zero": bool(lower <= 0.0 <= upper),
    }
    if fallback:
        ci["fallback"] = fallback
    return {"estimate": delta, "ci": ci, "p_value": p_value,
            "bootstrap_std": None}


def _paired_bootstrap_delta(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    weights: np.ndarray,
    *,
    weighting: str,
    confidence_level: float,
    bootstrap_iters: int,
    seed: int | None,
    ci_method: str = DEFAULT_CI_METHOD,
) -> dict[str, Any]:
    """Estimate candidate-minus-baseline mean delta and its CI.

    Only item weighting supports paired inference. The bootstrap unit is the item index.

    `ci_method`:
      "t" -- the classical paired Student-t interval (see _student_t_delta):
          no bootstrap, `bootstrap_iters` and `seed` are ignored.
      "studentized" -- the bootstrap-t interval. Each replicate is
          standardized by ITS OWN analytic SE, so the interval is built from
          the pivotal quantity t* = (theta* - theta)/SE* rather than from the
          replicate distribution directly. This is the construction that
          actually repairs coverage for the mean of right-skewed data, which
          is what per-item KLD deltas are.
      "bca" -- bias-corrected and accelerated (see _bca_endpoints):
          second-order accurate and skew-aware, and it rebalances the
          one-sided error rates, but on this data it does not move total
          coverage much.
      "percentile" -- the first-order interval, kept because the cross-repo
          golden fixtures pin it.

    Both corrected methods fall back to percentile on a degenerate sample and
    say so in ci["method"] / ci["ci_method"] / ci["fallback"].
    """
    if weighting != "item":
        raise ValueError("paired inference requires weighting='item'; token weighting is descriptive only")
    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method must be one of {CI_METHODS}; got {ci_method!r}")
    if baseline_values.shape != candidate_values.shape:
        raise ValueError("baseline_values and candidate_values must have the same shape")
    if baseline_values.ndim != 1 or baseline_values.size < 2:
        raise ValueError("paired comparison requires at least two items in one dimension")
    if weights.shape != baseline_values.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must align with scores and contain finite positive values")
    if not np.all(np.isfinite(baseline_values)) or not np.all(np.isfinite(candidate_values)):
        raise NonFiniteMetricError("non-finite metric scores; paired comparison aborted")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if not isinstance(bootstrap_iters, (int, np.integer)) or bootstrap_iters < (0 if ci_method == "t" else 2):
        raise ValueError("bootstrap_iters must be a nonnegative integer, with at least two draws for bootstrap intervals")

    with np.errstate(over="ignore", invalid="ignore"):
        diffs = candidate_values - baseline_values
    if not np.all(np.isfinite(diffs)):
        raise NonFiniteMetricError("non-finite paired differences; paired comparison aborted")

    if ci_method == "t":
        return _student_t_delta(diffs, weights, weighting=weighting,
                                confidence_level=confidence_level)

    delta, delta_se = _statistic_and_se(diffs, weights, weighting)

    boot = np.empty(bootstrap_iters, dtype=float)
    # Replicate SEs are only needed by the studentized interval; computing
    # them for every method would slow the (much more common) mean-only path.
    boot_se = np.empty(bootstrap_iters, dtype=float) if ci_method == "studentized" else None
    for i, idx in enumerate(_bootstrap_item_indices(seed, diffs.size, bootstrap_iters)):
        if boot_se is None:
            boot[i] = diffs[idx].mean()
        else:
            boot[i], boot_se[i] = _statistic_and_se(diffs[idx], weights[idx],
                                                    weighting)
    if not np.all(np.isfinite(boot)):
        raise NonFiniteMetricError("non-finite bootstrap estimates; paired comparison aborted")

    alpha = 1.0 - confidence_level
    levels = (alpha / 2.0, 1.0 - alpha / 2.0)
    fallback = None
    studentized = None
    bca_shape = None
    t_star = None
    if ci_method == "studentized":
        # t* = (theta* - theta) / SE*. Replicates whose own SE is zero (every
        # resampled item identical) have no pivot and are dropped; if too few
        # survive the interval is not trustworthy and we fall back.
        usable = np.isfinite(boot_se) & (boot_se > 0.0)
        if delta_se <= 0.0 or not np.isfinite(delta_se):
            fallback = ("the sample has no spread, so there is no SE to "
                        "studentize by; percentile interval used instead")
        elif int(usable.sum()) < max(2, bootstrap_iters // 2):
            fallback = (f"only {int(usable.sum())} of {bootstrap_iters} "
                        "replicates had a usable standard error; percentile "
                        "interval used instead")
        else:
            t_star = (boot[usable] - delta) / boot_se[usable]
            if not np.all(np.isfinite(t_star)):
                raise NonFiniteMetricError("non-finite studentized bootstrap pivots; paired comparison aborted")
            t_hi, t_lo = np.quantile(t_star, [1.0 - alpha / 2.0, alpha / 2.0])
            # NOTE the crossed order: the bootstrap-t interval is
            # [theta - t_(1-a/2) * SE, theta - t_(a/2) * SE]. Writing it the
            # "obvious" way round silently produces a percentile-like
            # interval reflected through the estimate.
            studentized = (delta - float(t_hi) * delta_se,
                           delta - float(t_lo) * delta_se)
    if ci_method == "bca":
        # Leave-one-ITEM-out jackknife of the SAME statistic (the item is the
        # exchangeable unit and the resample unit, so the acceleration must be
        # estimated on it too). Both forms are closed-form here, so this costs
        # one vector op rather than n re-evaluations.
        n = diffs.size
        total = diffs.sum()
        jack = (total - diffs) / (n - 1)
        jack = np.asarray(jack, dtype=float)
        bca_shape = _bca_shape(boot, delta, jack)
        endpoints = _bca_endpoints(boot, delta, jack, confidence_level)
        if endpoints is None:
            fallback = ("degenerate sample (no usable bias correction or "
                        "acceleration); percentile interval used instead")
        else:
            levels = endpoints
    if studentized is not None:
        lower, upper = studentized
        used = "studentized"
    else:
        lower, upper = np.quantile(boot, list(levels))
        used = ci_method if not fallback else "percentile"
        t_star = None
    # The p-value must invert the interval that was ACTUALLY built (after any
    # fallback), so "p <= alpha" and "the CI excludes 0" always agree.
    p_value = _two_sided_p(
        boot, delta, ci_method=used,
        z0=(bca_shape or (None, None))[0], accel=(bca_shape or (None, None))[1],
        t_star=t_star, delta_se=delta_se)
    bootstrap_std = float(boot.std(ddof=1)) if bootstrap_iters > 1 else 0.0
    if p_value is None or not all(math.isfinite(v) for v in (lower, upper, p_value, bootstrap_std)):
        raise NonFiniteMetricError("non-finite bootstrap interval, p-value, or standard deviation; paired comparison aborted")
    ci = {
        "method": f"paired_{weighting}_weighted_bootstrap_{used}",
        "ci_method": used,
        "confidence_level": confidence_level,
        "lower": float(lower),
        "upper": float(upper),
        "standard_error": delta_se,
        "bootstrap_iters": bootstrap_iters,
        "contains_zero": bool(lower <= 0.0 <= upper),
    }
    if fallback:
        ci["fallback"] = fallback
    return {
        "estimate": delta,
        "ci": ci,
        "p_value": p_value,
        "bootstrap_std": bootstrap_std,
    }


def _build_weighting_block(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    weights: np.ndarray,
    *,
    weighting: str,
    score_direction: str,
    confidence_level: float,
    bootstrap_iters: int,
    seed: int | None,
    ci_method: str = DEFAULT_CI_METHOD,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    if weighting == "token":
        if baseline_values.ndim != 1 or baseline_values.size == 0 or baseline_values.shape != candidate_values.shape:
            raise ValueError("scores must be nonempty, aligned 1-D arrays")
        if weights.shape != baseline_values.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must align with scores and contain finite positive values")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weight_sum = float(weights.sum())
            baseline_mean = float((weights * baseline_values).sum() / weight_sum)
            candidate_mean = float((weights * candidate_values).sum() / weight_sum)
            delta = float((weights * (candidate_values - baseline_values)).sum() / weight_sum)
        if not all(math.isfinite(v) for v in (weight_sum, baseline_mean, candidate_mean, delta)):
            raise NonFiniteMetricError("non-finite token-weighted descriptive statistic; comparison aborted")
        return {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta_candidate_minus_baseline": delta,
            "role": "descriptive",
        }
    delta_result = _paired_bootstrap_delta(
        baseline_values, candidate_values, weights,
        weighting=weighting, confidence_level=confidence_level,
        bootstrap_iters=bootstrap_iters, seed=seed, ci_method=ci_method,
    )
    baseline_mean = float(baseline_values.mean())
    candidate_mean = float(candidate_values.mean())
    if not math.isfinite(baseline_mean) or not math.isfinite(candidate_mean):
        raise NonFiniteMetricError("non-finite metric mean; paired comparison aborted")

    ci = delta_result["ci"]
    decision = _compute_decision(
        delta_estimate=delta_result["estimate"],
        ci_lower=ci["lower"], ci_upper=ci["upper"],
        null_value=0.0, score_direction=score_direction,
        confidence_level=confidence_level,
        equivalence_margin=equivalence_margin,
    )
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta_candidate_minus_baseline": delta_result["estimate"],
        "ci_delta": ci,
        "bootstrap_std": delta_result["bootstrap_std"],
        "p_value": delta_result["p_value"],
        "decision": decision,
    }


def _exp_nll(value: float) -> float:
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise NonFiniteMetricError(f"PPL exceeds floating-point range for log-scale NLL {value:g}; comparison aborted") from error
    if not math.isfinite(result):
        raise NonFiniteMetricError(f"PPL exceeds floating-point range for log-scale NLL {value:g}; comparison aborted")
    return result


def _ppl_ratio_block(nll_weighting_block: Mapping[str, Any], linked_to: str) -> dict[str, Any]:
    delta = nll_weighting_block["delta_candidate_minus_baseline"]
    if nll_weighting_block.get("role") == "descriptive":
        return {"estimate": _exp_nll(delta), "role": "descriptive"}
    nll_ci = nll_weighting_block["ci_delta"]
    return {
        "estimate": _exp_nll(delta),
        "ci": {
            "method": nll_ci["method"],
            "ci_method": nll_ci.get("ci_method"),
            "confidence_level": nll_ci["confidence_level"],
            "lower": _exp_nll(nll_ci["lower"]),
            "upper": _exp_nll(nll_ci["upper"]),
            "bootstrap_iters": nll_ci["bootstrap_iters"],
            "contains_one": bool(nll_ci["lower"] <= 0.0 <= nll_ci["upper"]),
        },
        "decision": {"verdict": f"(linked to {linked_to} above)", "linked_to": linked_to},
    }


def _ppl_block(nll_weighting_block: Mapping[str, Any], linked_to: str) -> dict[str, Any]:
    """Absolute perplexity per model: PPL = exp(mean nll) (the corpus PPL, NOT a
    mean of per-token PPLs). Display-only -- exp is monotonic, so the verdict is
    identical to nll's; the statistically-tested PPL change is ppl_ratio (b / a).
    """
    baseline_ppl = _exp_nll(nll_weighting_block["baseline_mean"])
    candidate_ppl = _exp_nll(nll_weighting_block["candidate_mean"])
    return {
        "baseline_ppl": baseline_ppl,
        "candidate_ppl": candidate_ppl,
        "delta_ppl_b_minus_a": candidate_ppl - baseline_ppl,
        **({"role": "descriptive"} if nll_weighting_block.get("role") == "descriptive" else
           {"decision": {"verdict": f"(linked to {linked_to} above)", "linked_to": linked_to}}),
    }


def _rms_dp_block(mse_weighting_block: Mapping[str, Any], linked_to: str) -> dict[str, Any]:
    baseline_rms = math.sqrt(mse_weighting_block["baseline_mean"])
    candidate_rms = math.sqrt(mse_weighting_block["candidate_mean"])
    return {
        "baseline_rms": baseline_rms,
        "candidate_rms": candidate_rms,
        "delta_rms_b_minus_a": candidate_rms - baseline_rms,
        **({"role": "descriptive"} if mse_weighting_block.get("role") == "descriptive" else
           {"decision": {"verdict": f"(linked to {linked_to} above)", "linked_to": linked_to}}),
    }


# --------------------------------------------------------------------------- #
