"""Prospective power planning for ordered, dependent token prefixes.

The exchangeable unit is the item.  Tokens stay inside their item and are
never resampled.  A token cap therefore selects an item-level prefix
estimand; it is not treated as a count of IID replications.

This module deliberately implements the production default paired Student-t
endpoint only.  Keeping that restriction explicit is safer than silently
approximating one of the bootstrap CI methods with a different test.
"""

from __future__ import annotations

import math
import warnings
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binomtest, nct, t as student_t

from stats.contracts import NonFiniteMetricError
from stats.inference import _statistic_and_se, _student_t_delta


POWER_SCHEMA_VERSION = "skymizer-sequential-power-v2"


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _arrays(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be same-length 1-D arrays")
    if values.size < 2:
        raise ValueError("power planning needs at least two pilot items")
    if not np.all(np.isfinite(values)):
        raise ValueError("pilot differences must all be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("pilot weights must all be finite and positive")
    return values, weights


def estimate_and_se(values, weights, weighting: str):
    """Production paired-t statistic and analytic SE for one item sample.

    Mean and SE follow the paired test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    """
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    values, weights = _arrays(values, weights)
    return _statistic_and_se(values, weights, weighting)


def paired_t_interval(values, weights, weighting: str, confidence_level: float):
    """The production item paired-t interval, including its zero-spread policy.

    Delegates to _student_t_delta and scipy.stats.ttest_rel(...).confidence_interval(confidence_level).
    SciPy API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    MATLAB correspondence: https://www.mathworks.com/help/stats/ttest.html
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    values, weights = _arrays(values, weights)
    result = _student_t_delta(values, weights, weighting=weighting, confidence_level=confidence_level)
    ci = result["ci"]
    return {
        "estimate": result["estimate"],
        "standard_error": ci["standard_error"],
        "lower": ci["lower"],
        "upper": ci["upper"],
        "degrees_of_freedom": ci["degrees_of_freedom"],
    }


def wilson_interval(hits: int, total: int, confidence_level: float = 0.95):
    """Wilson interval without continuity correction.

    Uses binomtest(hits, total).proportion_ci(confidence_level=..., method="wilson").
    SciPy binomtest: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
    SciPy proportion_ci: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
    This is a two-sided interval; SciPy's default method is different (Clopper-Pearson).
    """
    if not isinstance(hits, (int, np.integer)) or not isinstance(total, (int, np.integer)):
        raise ValueError("hits and total must be integers")
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= hits <= total:
        raise ValueError("hits must lie in [0, total]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if 0.5 + confidence_level / 2.0 == 1.0:
        raise ValueError("confidence_level is too close to 1 for the Wilson interval")
    ci = binomtest(hits, total).proportion_ci(confidence_level=confidence_level, method="wilson")
    if not all(math.isfinite(bound) for bound in ci) or not 0.0 <= ci.low <= ci.high <= 1.0:
        raise NonFiniteMetricError("invalid Wilson confidence interval")
    return float(ci.low), float(ci.high)


def _batch_item_estimate_and_se(values):
    """Equal-item estimates and standard errors for Monte Carlo draws."""
    n = values.shape[1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        estimate = values.mean(axis=1)
        se = values.std(axis=1, ddof=1) / math.sqrt(n)
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(se)):
        raise NonFiniteMetricError("non-finite estimate or standard error; power simulation aborted")
    nonconstant = np.any(values != values[:, :1], axis=1)
    if np.any((se < math.sqrt(np.finfo(float).tiny)) & nonconstant):
        raise NonFiniteMetricError("standard error underflow for nonconstant differences; power simulation aborted")
    return estimate, se


def _pilot_profiles(
    differences_by_cap: Mapping[int, np.ndarray],
    weights_by_cap: Mapping[int, np.ndarray],
    weighting: str,
    sesoi: float | None,
    effect_profile: str,
    reference_cap: int | None,
):
    caps = tuple(sorted(differences_by_cap))
    pilot_effect = {}
    residuals = {}
    for cap in caps:
        values, weights = _arrays(differences_by_cap[cap], weights_by_cap[cap])
        estimate, _ = estimate_and_se(values, weights, weighting)
        pilot_effect[cap] = estimate
        residuals[cap] = values - estimate

    if sesoi is None:
        return pilot_effect, residuals, None
    if not math.isfinite(sesoi) or sesoi == 0.0:
        raise ValueError("sesoi must be finite and non-zero")
    if effect_profile == "flat":
        effects = {cap: float(sesoi) for cap in caps}
    elif effect_profile == "pilot":
        if reference_cap is None:
            raise ValueError("reference_cap is required for the pilot effect profile")
        if reference_cap not in pilot_effect:
            raise ValueError(f"reference_cap {reference_cap} is not one of {list(caps)}")
        shift = float(sesoi) - pilot_effect[reference_cap]
        effects = {cap: pilot_effect[cap] + shift for cap in caps}
    else:
        raise ValueError(
            f"effect_profile must be 'flat' or 'pilot'; got {effect_profile!r}"
        )
    return pilot_effect, residuals, effects


def _simulate_t_surface(
    residuals_by_cap: Mapping[int, np.ndarray],
    effects_by_cap: Mapping[int, float],
    sample_sizes: Sequence[int],
    confidence_level: float,
    reps: int,
    seed: int,
    mc_confidence_level: float,
    chunk_size: int = 256,
):
    """Equal-item future simulation with a two-sided t rejection threshold.

    SciPy t.isf uses alpha/2 directly: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
    caps = tuple(sorted(residuals_by_cap))
    pilot_n = len(residuals_by_cap[caps[0]])
    if any(len(residuals_by_cap[cap]) != pilot_n for cap in caps):
        raise ValueError("every cap must contain the same aligned pilot items")
    rng = np.random.default_rng(seed)
    cells = []

    for sample_size in sample_sizes:
        critical = float(student_t.isf((1.0 - confidence_level) / 2.0, sample_size - 1))
        correct = {cap: 0 for cap in caps}
        wrong = {cap: 0 for cap in caps}
        null_reject = {cap: 0 for cap in caps}
        remaining = reps
        while remaining:
            count = min(chunk_size, remaining)
            indices = rng.integers(0, pilot_n, size=(count, sample_size))
            for cap in caps:
                values = residuals_by_cap[cap][indices]
                null_estimate, se = _batch_item_estimate_and_se(values)
                half_width = critical * se
                effect = effects_by_cap[cap]
                alternative_estimate = null_estimate + effect
                lower = alternative_estimate - half_width
                upper = alternative_estimate + half_width
                if effect > 0.0:
                    correct[cap] += int(np.count_nonzero(lower > 0.0))
                    wrong[cap] += int(np.count_nonzero(upper < 0.0))
                elif effect < 0.0:
                    correct[cap] += int(np.count_nonzero(upper < 0.0))
                    wrong[cap] += int(np.count_nonzero(lower > 0.0))
                else:
                    correct[cap] += int(
                        np.count_nonzero((lower > 0.0) | (upper < 0.0))
                    )
                null_reject[cap] += int(
                    np.count_nonzero(
                        (null_estimate - half_width > 0.0)
                        | (null_estimate + half_width < 0.0)
                    )
                )
            remaining -= count

        for cap in caps:
            null_interval = wilson_interval(null_reject[cap], reps, mc_confidence_level)
            cell = {
                "token_cap": int(cap),
                "sample_size": int(sample_size),
                "assumed_effect": float(effects_by_cap[cap]),
                "null_rejection_rate": null_reject[cap] / reps,
                "null_rejection_mc_lower": null_interval[0],
                "null_rejection_mc_upper": null_interval[1],
                "directional_power_defined": bool(effects_by_cap[cap] != 0.0),
            }
            if effects_by_cap[cap] == 0.0:
                cell.update({
                    "power": None,
                    "power_mc_lower": None,
                    "power_mc_upper": None,
                    "wrong_sign_rate": None,
                    "wrong_sign_mc_lower": None,
                    "wrong_sign_mc_upper": None,
                })
            else:
                power_interval = wilson_interval(
                    correct[cap], reps, mc_confidence_level
                )
                wrong_interval = wilson_interval(
                    wrong[cap], reps, mc_confidence_level
                )
                cell.update({
                    "power": correct[cap] / reps,
                    "power_mc_lower": power_interval[0],
                    "power_mc_upper": power_interval[1],
                    "wrong_sign_rate": wrong[cap] / reps,
                    "wrong_sign_mc_lower": wrong_interval[0],
                    "wrong_sign_mc_upper": wrong_interval[1],
                })
            cells.append(cell)
    return cells


def _effective_sd(values, weights, weighting: str):
    """Pilot SD; classify spread within 10 eps of input magnitude as numerically unresolved."""
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    values, weights = _arrays(values, weights)
    if np.ptp(values) <= 10.0 * np.finfo(float).eps * float(np.max(np.abs(values))):
        return 0.0
    return estimate_and_se(values, weights, weighting)[1] * math.sqrt(values.size)


def _bootstrap_effective_sd(
    differences_by_cap: Mapping[int, np.ndarray],
    weights_by_cap: Mapping[int, np.ndarray],
    weighting: str,
    outer_reps: int,
    seed: int,
):
    if outer_reps == 0:
        return None
    caps = tuple(sorted(differences_by_cap))
    pilot_n = len(differences_by_cap[caps[0]])
    rng = np.random.default_rng(seed)
    draws = {cap: np.empty(outer_reps, dtype=float) for cap in caps}
    for rep in range(outer_reps):
        indices = rng.integers(0, pilot_n, size=pilot_n)
        for cap in caps:
            try:
                draws[cap][rep] = _effective_sd(differences_by_cap[cap][indices], weights_by_cap[cap][indices], weighting)
            except NonFiniteMetricError:
                draws[cap][rep] = 0.0
    return draws


def _nct_sf_checked(critical, df, noncentrality):
    """Use SciPy's noncentral-t upper tail only when its numeric evaluation succeeds.

    API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    Record warnings rather than converting them to exceptions inside the SciPy ufunc.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        value = float(nct.sf(critical, df, noncentrality))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0 or any(issubclass(item.category, RuntimeWarning) for item in caught):
        raise NonFiniteMetricError("noncentral-t tail failed numerical evaluation")
    return value


def _normal_model_power(effect, effective_sd, sample_size, confidence_level):
    """Correct-direction rejection probability of a two-sided t test under IID normal deltas.

    Uses nct.sf(t.isf(alpha/2, df), df, abs(effect)*sqrt(n)/sd); the opposite tail is not power.
    SciPy nct: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    SciPy t.isf: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    This is a Gaussian nuisance projection, separate from the empirical item simulation.
    """
    if not math.isfinite(effective_sd) or effective_sd <= 0.0:
        raise ValueError("normal-model power needs a finite positive SD")
    critical = float(student_t.isf((1.0 - confidence_level) / 2.0, sample_size - 1))
    noncentrality = abs(effect) * math.sqrt(sample_size) / effective_sd
    return _nct_sf_checked(critical, sample_size - 1, noncentrality)


def _normal_model_mde_per_sd(sample_size, confidence_level, target_power):
    """Solve directional normal-model power in dimensionless noncentrality, then divide by sqrt(n).

    SciPy nct.sf: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    SciPy t.isf: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    SciPy brentq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
    Solving in noncentrality avoids a root tolerance tied to the metric's units.
    """
    critical = float(student_t.isf((1.0 - confidence_level) / 2.0, sample_size - 1))
    def residual(noncentrality):
        return _nct_sf_checked(critical, sample_size - 1, noncentrality) - target_power
    high = max(1.0, critical)
    for _ in range(100):
        if residual(high) >= 0.0:
            break
        high *= 2.0
    else:
        raise ValueError("normal-model MDE could not be bracketed")
    root = brentq(residual, 0.0, high, xtol=1e-12)
    if root <= 0.0 or abs(residual(root)) > 1e-10:
        raise ValueError("normal-model MDE inversion did not converge")
    return float(root / math.sqrt(sample_size))


def build_design(
    differences_by_cap: Mapping[int, Sequence[float]],
    weights_by_cap: Mapping[int, Sequence[float]],
    sample_sizes: Sequence[int],
    *,
    weighting: str = "item",
    confidence_level: float = 0.95,
    target_power: float = 0.80,
    sesoi: float | None = None,
    effect_profile: str = "flat",
    reference_cap: int | None = None,
    reps: int = 2000,
    outer_reps: int = 200,
    seed: int = 7,
    mc_confidence_level: float = 0.95,
):
    """Build a cap-by-sample-size prospective design surface.

    ``differences_by_cap`` contains candidate-minus-baseline item prefix
    scores.  The empirical residual vector at each cap is centered and shifted
    to the declared alternative before future items are sampled with
    replacement.  The same sampled item indices are used for every cap.
    Plug-in critical values use scipy.stats.t.isf: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
    caps = tuple(sorted(_integer(cap, "token cap", 1) for cap in differences_by_cap))
    if not caps:
        raise ValueError("token caps must be nonempty")
    if set(caps) != set(weights_by_cap):
        raise ValueError("differences_by_cap and weights_by_cap need identical caps")
    for cap in weights_by_cap:
        _integer(cap, "weight token cap", 1)
    sizes = tuple(sorted({_integer(n, "sample size", 2) for n in sample_sizes}))
    if not sizes:
        raise ValueError("sample sizes must be nonempty")
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if 0.5 + confidence_level / 2.0 == 1.0:
        raise ValueError("confidence_level is too close to 1 for the production paired-t interval")
    if not 0.5 < target_power < 1.0:
        raise ValueError("target_power must be in (0.5, 1)")
    reps = _integer(reps, "reps", 1)
    outer_reps = _integer(outer_reps, "outer_reps", 0)
    seed = _integer(seed, "seed", 0)
    if not 0.0 < mc_confidence_level < 1.0:
        raise ValueError("mc_confidence_level must be in (0, 1)")
    if 0.5 + mc_confidence_level / 2.0 == 1.0:
        raise ValueError("mc_confidence_level is too close to 1 for the Wilson interval")
    if effect_profile not in ("flat", "pilot"):
        raise ValueError("effect_profile must be flat or pilot")
    if reference_cap is not None:
        reference_cap = _integer(reference_cap, "reference_cap", 1)
    if effect_profile == "pilot" and (sesoi is None or reference_cap not in caps):
        raise ValueError("pilot effect profile needs sesoi and a reference_cap in the cap grid")
    if effect_profile == "flat" and reference_cap is not None:
        raise ValueError("reference_cap is only meaningful for the pilot effect profile")

    differences = {}
    weights = {}
    pilot_n = None
    for cap in caps:
        differences[cap], weights[cap] = _arrays(
            differences_by_cap[cap], weights_by_cap[cap]
        )
        if pilot_n is None:
            pilot_n = differences[cap].size
        elif differences[cap].size != pilot_n:
            raise ValueError("every cap must contain the same aligned pilot items")

    pilot_effect, residuals, effects = _pilot_profiles(
        differences, weights, weighting, sesoi, effect_profile, reference_cap
    )
    effective_sd = {
        cap: _effective_sd(differences[cap], weights[cap], weighting)
        for cap in caps
    }
    for cap in caps:
        if effective_sd[cap] == 0.0:
            raise ValueError(f"token cap {cap}: constant or numerically unresolved pilot variance; power and precision are unidentified")
    if effects is not None and not all(math.isfinite(effect) for effect in effects.values()):
        raise NonFiniteMetricError("non-finite assumed effect profile")
    sd_draws = _bootstrap_effective_sd(
        differences, weights, weighting, outer_reps, seed + 104729
    )

    cells_by_key = {}
    if effects is not None:
        simulated = _simulate_t_surface(
            residuals, effects, sizes, confidence_level,
            reps, seed, mc_confidence_level,
        )
        cells_by_key = {
            (cell["token_cap"], cell["sample_size"]): cell
            for cell in simulated
        }

    mde_factors = {n: _normal_model_mde_per_sd(n, confidence_level, target_power) for n in sizes}
    cells = []
    for cap in caps:
        for sample_size in sizes:
            critical = float(student_t.isf((1.0 - confidence_level) / 2.0, sample_size - 1))
            scale = effective_sd[cap] / math.sqrt(sample_size)
            cell = cells_by_key.get((cap, sample_size), {
                "token_cap": int(cap),
                "sample_size": int(sample_size),
                "assumed_effect": None,
            })
            cell.update({
                "plugin_ci_half_width": float(critical * scale),
                "normal_model_mde": float(mde_factors[sample_size] * effective_sd[cap]),
                "expected_evaluated_tokens": float(
                    sample_size * float(weights[cap].mean())
                ),
            })
            if effects is not None:
                inflated = cell["null_rejection_mc_lower"] > 1.0 - confidence_level
                cell["null_calibration_status"] = "detected_inflation" if inflated else "no_detected_inflation"
                cell["crossing_eligible"] = not inflated and cell["directional_power_defined"]
            if sd_draws is not None:
                half_widths = critical * sd_draws[cap] / math.sqrt(sample_size)
                mdes = mde_factors[sample_size] * sd_draws[cap]
                uncertainty = {
                    "method": "outer_item_bootstrap_plus_directional_normal_model",
                    "ci_half_width_p10": float(np.quantile(half_widths, 0.10)),
                    "ci_half_width_p50": float(np.quantile(half_widths, 0.50)),
                    "ci_half_width_p90": float(np.quantile(half_widths, 0.90)),
                    "mde_p10": float(np.quantile(mdes, 0.10)),
                    "mde_p50": float(np.quantile(mdes, 0.50)),
                    "mde_p90": float(np.quantile(mdes, 0.90)),
                }
                degenerate = int(np.count_nonzero(sd_draws[cap] == 0.0))
                uncertainty["degenerate_draws"] = degenerate
                uncertainty["n_draws"] = outer_reps
                if degenerate:
                    uncertainty.update({key: None for key in uncertainty if key.endswith(("_p10", "_p50", "_p90"))})
                    uncertainty["status"] = "unavailable_degenerate_resamples"
                else:
                    uncertainty["status"] = "available"
                if effects is not None and effects[cap] != 0.0 and not degenerate:
                    approx_power = np.array([
                        _normal_model_power(
                            effects[cap], sd, sample_size, confidence_level
                        )
                        for sd in sd_draws[cap]
                    ])
                    uncertainty.update({
                        "power_p10": float(np.quantile(approx_power, 0.10)),
                        "power_p50": float(np.quantile(approx_power, 0.50)),
                        "power_p90": float(np.quantile(approx_power, 0.90)),
                        "mean_power": float(approx_power.mean()),
                        "probability_power_at_least_target": float(
                            np.mean(approx_power >= target_power)
                        ),
                    })
                cell["pilot_uncertainty"] = uncertainty
            cells.append(cell)

    pilot_caps = []
    for cap in caps:
        entry = {
            "token_cap": int(cap),
            "pilot_effect": float(pilot_effect[cap]),
            "effective_sd": float(effective_sd[cap]),
            "mean_tokens_per_item": float(weights[cap].mean()),
            "fraction_reaching_cap": float(np.mean(weights[cap] >= cap)),
        }
        if sd_draws is not None:
            entry["unresolved_outer_draws"] = int(np.count_nonzero(sd_draws[cap] == 0.0))
            for suffix, quantile in (("p10", .1), ("p50", .5), ("p90", .9)):
                entry[f"effective_sd_outer_{suffix}"] = None if entry["unresolved_outer_draws"] else float(np.quantile(sd_draws[cap], quantile))
        pilot_caps.append(entry)

    required = []
    for cap in caps:
        cap_cells = [cell for cell in cells if cell["token_cap"] == cap]
        point = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("crossing_eligible") and cell.get("power") is not None
             and cell["power"] >= target_power),
            None,
        )
        conservative = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("crossing_eligible") and cell.get("power_mc_lower") is not None
             and cell["power_mc_lower"] >= target_power),
            None,
        )
        nuisance = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("crossing_eligible") and cell.get("power_mc_lower", 0.0) >= target_power
             and cell.get("pilot_uncertainty", {}).get("power_p10") is not None
             and cell["pilot_uncertainty"]["power_p10"] >= target_power),
            None,
        )
        required.append({
            "token_cap": int(cap),
            "first_evaluated_n_point_estimate": point,
            "first_evaluated_n_mc_lower_bound": conservative,
            "first_evaluated_n_pilot_power_p10": nuisance,
        })

    return {
        "schema_version": POWER_SCHEMA_VERSION,
        "mode": "prospective_power" if sesoi is not None else "precision_only",
        "pilot_n_items": int(pilot_n),
        "token_caps": list(caps),
        "sample_sizes": list(sizes),
        "weighting": weighting,
        "confidence_level": confidence_level,
        "target_power": target_power,
        "sesoi": sesoi,
        "effect_profile": effect_profile if sesoi is not None else None,
        "reference_cap": reference_cap if sesoi is not None else None,
        "test": "paired_student_t",
        "reps": reps if sesoi is not None else 0,
        "outer_reps": outer_reps,
        "seed": seed,
        "mc_confidence_level": mc_confidence_level,
        "mc_interval_scope": "pointwise_per_cell; no simultaneous grid coverage",
        "crossing_rule": "exclude cells with null MC lower bound above nominal alpha; no detected inflation is not proof of calibration",
        "normal_model_scope": "directional Gaussian nuisance sensitivity and MDE; empirical MC remains the primary power model",
        "ci_width_scope": "plug-in t critical times pilot SD / sqrt(N), not expected future CI width",
        "pilot_caps": pilot_caps,
        "cells": cells,
        "required_n_by_cap": required,
    }
