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
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from stats.student_t import t_ppf


POWER_SCHEMA_VERSION = "skymizer-sequential-power-v1"


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
    """Production paired-t statistic and analytic SE for one item sample."""
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    values, weights = _arrays(values, weights)
    n = values.size
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n))
    return estimate, se


def paired_t_interval(values, weights, weighting: str, confidence_level: float):
    """The exact interval used by stats.inference for ``--ci-method t``."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    values, weights = _arrays(values, weights)
    estimate, se = estimate_and_se(values, weights, weighting)
    critical = t_ppf(0.5 + confidence_level / 2.0, values.size - 1)
    half_width = critical * se
    return {
        "estimate": estimate,
        "standard_error": se,
        "lower": estimate - half_width,
        "upper": estimate + half_width,
        "degrees_of_freedom": int(values.size - 1),
    }


def wilson_interval(hits: int, total: int, confidence_level: float = 0.95):
    """Wilson score interval for a Monte-Carlo rejection proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= hits <= total:
        raise ValueError("hits must lie in [0, total]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = hits / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (
        max(0.0, (centre - half) / denominator),
        min(1.0, (centre + half) / denominator),
    )


def _batch_estimate_and_se(values, weights, weighting: str):
    """Vectorized counterpart of :func:`estimate_and_se` for MC draws."""
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    n = values.shape[1]
    estimate = values.mean(axis=1)
    se = values.std(axis=1, ddof=1) / math.sqrt(n)
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
    weights_by_cap: Mapping[int, np.ndarray],
    effects_by_cap: Mapping[int, float],
    sample_sizes: Sequence[int],
    weighting: str,
    confidence_level: float,
    reps: int,
    seed: int,
    mc_confidence_level: float,
    chunk_size: int = 256,
):
    caps = tuple(sorted(residuals_by_cap))
    pilot_n = len(residuals_by_cap[caps[0]])
    if any(len(residuals_by_cap[cap]) != pilot_n for cap in caps):
        raise ValueError("every cap must contain the same aligned pilot items")
    rng = np.random.default_rng(seed)
    cells = []

    for sample_size in sample_sizes:
        critical = t_ppf(0.5 + confidence_level / 2.0, sample_size - 1)
        correct = {cap: 0 for cap in caps}
        wrong = {cap: 0 for cap in caps}
        null_reject = {cap: 0 for cap in caps}
        remaining = reps
        while remaining:
            count = min(chunk_size, remaining)
            indices = rng.integers(0, pilot_n, size=(count, sample_size))
            for cap in caps:
                values = residuals_by_cap[cap][indices]
                weights = weights_by_cap[cap][indices]
                null_estimate, se = _batch_estimate_and_se(values, weights, weighting)
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
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    values, weights = _arrays(values, weights)
    return float(values.std(ddof=1))


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
            draws[cap][rep] = _effective_sd(
                differences_by_cap[cap][indices],
                weights_by_cap[cap][indices],
                weighting,
            )
    return draws


def _normal_approx_power(effect, effective_sd, sample_size, confidence_level):
    """Fast nuisance-band approximation; exact power is empirical MC above."""
    if effective_sd == 0.0:
        return 1.0 if effect != 0.0 else 0.0
    critical = t_ppf(0.5 + confidence_level / 2.0, sample_size - 1)
    noncentrality = abs(effect) * math.sqrt(sample_size) / effective_sd
    normal = NormalDist()
    return float(
        normal.cdf(-critical - noncentrality)
        + 1.0
        - normal.cdf(critical - noncentrality)
    )


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
    """
    caps = tuple(sorted(int(cap) for cap in differences_by_cap))
    if not caps or any(cap < 1 for cap in caps):
        raise ValueError("token caps must be positive")
    if set(caps) != set(weights_by_cap):
        raise ValueError("differences_by_cap and weights_by_cap need identical caps")
    sizes = tuple(sorted({int(n) for n in sample_sizes}))
    if not sizes or any(n < 2 for n in sizes):
        raise ValueError("sample sizes must all be >= 2")
    if weighting != "item":
        raise ValueError("paired-test power requires weighting='item'; token weighting is descriptive only")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if not 0.5 < target_power < 1.0:
        raise ValueError("target_power must be in (0.5, 1)")
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if outer_reps < 0:
        raise ValueError("outer_reps must be >= 0")

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
    sd_draws = _bootstrap_effective_sd(
        differences, weights, weighting, outer_reps, seed + 104729
    )

    cells_by_key = {}
    if effects is not None:
        simulated = _simulate_t_surface(
            residuals, weights, effects, sizes, weighting, confidence_level,
            reps, seed, mc_confidence_level,
        )
        cells_by_key = {
            (cell["token_cap"], cell["sample_size"]): cell
            for cell in simulated
        }

    z_power = NormalDist().inv_cdf(target_power)
    cells = []
    for cap in caps:
        for sample_size in sizes:
            critical = t_ppf(0.5 + confidence_level / 2.0, sample_size - 1)
            scale = effective_sd[cap] / math.sqrt(sample_size)
            cell = cells_by_key.get((cap, sample_size), {
                "token_cap": int(cap),
                "sample_size": int(sample_size),
                "assumed_effect": None,
            })
            cell.update({
                "expected_ci_half_width": float(critical * scale),
                "mde_approx": float((critical + z_power) * scale),
                "expected_evaluated_tokens": float(
                    sample_size * float(weights[cap].mean())
                ),
            })
            if sd_draws is not None:
                half_widths = critical * sd_draws[cap] / math.sqrt(sample_size)
                mdes = (critical + z_power) * sd_draws[cap] / math.sqrt(sample_size)
                uncertainty = {
                    "method": "outer_item_bootstrap_plus_normal_power_approximation",
                    "ci_half_width_p10": float(np.quantile(half_widths, 0.10)),
                    "ci_half_width_p50": float(np.quantile(half_widths, 0.50)),
                    "ci_half_width_p90": float(np.quantile(half_widths, 0.90)),
                    "mde_p10": float(np.quantile(mdes, 0.10)),
                    "mde_p50": float(np.quantile(mdes, 0.50)),
                    "mde_p90": float(np.quantile(mdes, 0.90)),
                }
                if effects is not None and effects[cap] != 0.0:
                    approx_power = np.array([
                        _normal_approx_power(
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
            entry["effective_sd_outer_p10"] = float(np.quantile(sd_draws[cap], 0.10))
            entry["effective_sd_outer_p50"] = float(np.quantile(sd_draws[cap], 0.50))
            entry["effective_sd_outer_p90"] = float(np.quantile(sd_draws[cap], 0.90))
        pilot_caps.append(entry)

    required = []
    for cap in caps:
        cap_cells = [cell for cell in cells if cell["token_cap"] == cap]
        point = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("power") is not None
             and cell["power"] >= target_power),
            None,
        )
        conservative = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("power_mc_lower") is not None
             and cell["power_mc_lower"] >= target_power),
            None,
        )
        nuisance = next(
            (cell["sample_size"] for cell in cap_cells
             if cell.get("pilot_uncertainty", {}).get("power_p10") is not None
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
        "pilot_caps": pilot_caps,
        "cells": cells,
        "required_n_by_cap": required,
    }
