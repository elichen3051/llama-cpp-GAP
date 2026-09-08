# stats/inference.py -- The paired statistics engine: decision table, SE/BCa/studentized interval
# construction, two-sided p by interval inversion, Holm, the shared bootstrap
# index stream, weighting blocks and the derived ppl/rms blocks.
# NUMERICAL_CONTRACT.md #5/#6.
import math
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import DegenerateDataWarning, bootstrap, ttest_rel

from stats.contracts import BOOTSTRAP_MIN_TAIL_DRAWS, CI_METHODS, DEFAULT_CI_METHOD, InferenceUnavailableError, NonFiniteMetricError, min_bootstrap_iters

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

    b_smaller = ci_upper < null_value
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


def _bootstrap_tail_counts(samples, lower, upper):
    """Count strict outside draws, excluding endpoint differences within float roundoff."""
    tolerance = 8.0 * np.finfo(float).eps * float(np.max(np.abs(samples)))
    return int(np.count_nonzero(samples < lower - tolerance)), int(np.count_nonzero(samples > upper + tolerance))


def _bootstrap_ci_inference(interval_at, confidence_level: float, bootstrap_iters: int):
    """Invert a nested CI family only where both tails have enough resamples.

    The returned p is the upper numerical bracket, or a conservative upper bound when the null lies outside the supported confidence range. Strict outside counts intentionally reject unresolved tied endpoints. This checks Monte Carlo support, not population coverage or independent sampling.
    SciPy permits CI reuse without new draws: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    """
    evaluated = {}

    def checked(level):
        if level in evaluated:
            return evaluated[level]
        lower, upper, below, above = interval_at(level)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise NonFiniteMetricError("non-finite bootstrap confidence interval; paired comparison aborted")
        if lower > upper:
            raise ValueError("bootstrap confidence interval is reversed; paired comparison aborted")
        for other_level, (other_lower, other_upper, other_below, other_above) in evaluated.items():
            narrow, wide = ((lower, upper), (other_lower, other_upper)) if level < other_level else ((other_lower, other_upper), (lower, upper))
            narrow_counts, wide_counts = ((below, above), (other_below, other_above)) if level < other_level else ((other_below, other_above), (below, above))
            tolerance = 8.0 * np.finfo(float).eps * max(abs(v) for v in (*narrow, *wide))
            if narrow[0] < wide[0] - tolerance or narrow[1] > wide[1] + tolerance or any(a < b for a, b in zip(narrow_counts, wide_counts)):
                raise ValueError("bootstrap confidence intervals or tail counts are not monotone; paired comparison aborted")
        evaluated[level] = (lower, upper, below, above)
        return evaluated[level]

    def supported(interval):
        return min(interval[2:]) >= BOOTSTRAP_MIN_TAIL_DRAWS

    def contains_zero(interval):
        return interval[0] <= 0.0 <= interval[1]

    reported = checked(confidence_level)
    if not supported(reported):
        raise InferenceUnavailableError(
            f"bootstrap CI has insufficient endpoint support: {reported[2]} and {reported[3]} draws strictly outside its endpoints; require {BOOTSTRAP_MIN_TAIL_DRAWS} per tail. Increase --bootstrap-iters; tied or discrete samples may require a different method such as --ci-method t, with its assumptions assessed.")
    central = checked(0.0)
    support_low, support_high = confidence_level, 1.0 - 2.0 / (bootstrap_iters + 1.0)
    if support_high <= support_low:
        raise InferenceUnavailableError("bootstrap confidence level exceeds finite-resample resolution; increase --bootstrap-iters")
    if supported(checked(support_high)):
        support_low = support_high
    else:
        for _ in range(52):
            midpoint = (support_low + support_high) / 2.0
            if midpoint in (support_low, support_high):
                break
            if supported(checked(midpoint)):
                support_low = midpoint
            else:
                support_high = midpoint
    supported_alpha_min = 1.0 - support_low
    censored = not contains_zero(checked(support_low))
    if censored:
        p_lower, p_upper = 0.0, supported_alpha_min
    elif contains_zero(central):
        p_lower = p_upper = 1.0
    else:
        low, high = 0.0, support_low
        if contains_zero(reported):
            high = confidence_level
        else:
            low = confidence_level
        for _ in range(52):
            midpoint = (low + high) / 2.0
            if midpoint in (low, high):
                break
            if contains_zero(checked(midpoint)):
                high = midpoint
            else:
                low = midpoint
        p_lower, p_upper = 1.0 - high, 1.0 - low
    if (p_upper < 1.0 - confidence_level) != (not contains_zero(reported)):
        raise InferenceUnavailableError("bootstrap p-value is unresolved at the requested confidence boundary; increase --bootstrap-iters or use --ci-method t with its assumptions assessed")
    return reported, p_upper, {
        "method": "bootstrap_ci_inversion",
        "censored": censored,
        "lower_bound": p_lower,
        "upper_bound": p_upper,
        "supported_alpha_min": supported_alpha_min,
        "minimum_tail_draws": BOOTSTRAP_MIN_TAIL_DRAWS,
        "tail_roundoff_rtol": 8.0 * np.finfo(float).eps,
        "rejection_rule": "p_value < alpha; p_value is a conservative upper bound when censored",
    }


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
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("Holm p-values must be finite and in [0, 1]")
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
    """Reference item stream: default_rng(seed), then n_items integers per draw.

    Bootstrap-t consumes this generator. SciPy percentile/BCa uses the same stream in batches; parity tests check its distribution against this recipe.
    """
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
    """Paired item-mean inference, with token weighting excluded.

    SciPy percentile/BCa bootstrap resamples the B-A differences with the shared index stream: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    Bootstrap-t uses every finite nonzero-SE pivot and crossed NumPy quantiles: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    Nonconstant two-item bootstrap samples and unsupported tails fail explicitly. Constant differences retain an empirical point-interval convention, without claiming zero population variance. Bootstrap intervals and their inverted p-values remain approximations under independent item sampling.
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
    floor = min_bootstrap_iters(confidence_level, ci_method)
    if bootstrap_iters < floor:
        raise ValueError(f"bootstrap_iters must be >= {floor} for {ci_method} at confidence_level={confidence_level:g}")
    constant = bool(np.all(diffs == diffs[0]))
    if not constant and diffs.size == 2:
        raise InferenceUnavailableError("nonconstant two-item bootstrap inference is unsupported; collect more independent items or use --ci-method t with its assumptions assessed")

    fallback = None
    used = ci_method
    if constant:
        delta = float(diffs[0])
        delta_se = 0.0
        lower = upper = delta
        used = "percentile"
        fallback = "constant differences (no spread): empirical percentile point-interval convention; this does not establish zero population variance"
        bootstrap_std = 0.0
        resolution = 2.0 / (bootstrap_iters + 1.0)
        p_value = 1.0 if delta == 0.0 else resolution
        p_metadata = {
            "method": "constant_empirical_distribution",
            "censored": delta != 0.0,
            "lower_bound": 1.0 if delta == 0.0 else 0.0,
            "upper_bound": p_value,
            "supported_alpha_min": None,
            "resolution": resolution,
            "rejection_rule": "p_value < alpha; constant-input convention, not a population variance guarantee",
        }
        tail_support = {"minimum": BOOTSTRAP_MIN_TAIL_DRAWS, "lower": 0, "upper": 0, "exemption": "constant_empirical_distribution"}
    else:
        if ci_method == "studentized":
            boot = np.empty(bootstrap_iters, dtype=float)
            pivots = np.empty(bootstrap_iters, dtype=float)
            for i, idx in enumerate(_bootstrap_item_indices(seed, diffs.size, bootstrap_iters)):
                draw = diffs[idx]
                if np.all(draw == draw[0]):
                    raise InferenceUnavailableError("studentized bootstrap has a zero-SE resample; no pivots were discarded. Use another method such as --ci-method t with its assumptions assessed")
                boot[i], se = _statistic_and_se(draw, weights[idx], weighting)
                if se <= 0.0:
                    raise InferenceUnavailableError("studentized bootstrap has an unusable resample SE; paired comparison aborted")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pivots[i] = (boot[i] - delta) / se
            if not np.all(np.isfinite(pivots)):
                raise NonFiniteMetricError("non-finite studentized bootstrap pivots; paired comparison aborted")

            def interval_at(level):
                alpha = 1.0 - level
                low_pivot, high_pivot = np.quantile(pivots, [alpha / 2.0, 1.0 - alpha / 2.0])
                below, above = _bootstrap_tail_counts(pivots, low_pivot, high_pivot)
                return (delta - float(high_pivot) * delta_se, delta - float(low_pivot) * delta_se,
                        above, below)
        else:
            def scipy_bootstrap(level, previous=None):
                try:
                    with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise"):
                        warnings.simplefilter("error", DegenerateDataWarning)
                        return bootstrap(
                            (diffs,), np.mean, paired=True, vectorized=True, batch=128,
                            method=ci_method, confidence_level=level,
                            n_resamples=bootstrap_iters if previous is None else 0,
                            rng=np.random.default_rng(seed), bootstrap_result=previous)
                except (DegenerateDataWarning, FloatingPointError) as error:
                    raise NonFiniteMetricError("SciPy bootstrap interval is undefined for this sample; use another method with its assumptions assessed") from error

            resampled = scipy_bootstrap(confidence_level)
            boot = resampled.bootstrap_distribution

            def interval_at(level):
                result = resampled if level == confidence_level else scipy_bootstrap(level, resampled)
                low, high = result.confidence_interval
                return (float(low), float(high), *_bootstrap_tail_counts(boot, low, high))
        if not np.all(np.isfinite(boot)):
            raise NonFiniteMetricError("non-finite bootstrap estimates; paired comparison aborted")
        reported, p_value, p_metadata = _bootstrap_ci_inference(interval_at, confidence_level, bootstrap_iters)
        lower, upper, below, above = reported
        tail_support = {"minimum": BOOTSTRAP_MIN_TAIL_DRAWS, "lower": below, "upper": above}
        bootstrap_std = float(boot.std(ddof=1))
    if not all(math.isfinite(v) for v in (lower, upper, p_value, bootstrap_std)):
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
        "tail_support": tail_support,
    }
    if fallback:
        ci["fallback"] = fallback
    return {
        "estimate": delta,
        "ci": ci,
        "p_value": p_value,
        "p_value_metadata": p_metadata,
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
        **({"p_value_metadata": delta_result["p_value_metadata"]} if "p_value_metadata" in delta_result else {}),
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
