# stats/engine.py -- compare_items and its six stage helpers: the one entry point both
# comparators call to turn aligned per-item scores into the result dict.
import math
from typing import Any, Mapping, Sequence

import numpy as np

from stats.contracts import (
    AlignmentError,
    DEFAULT_CI_METHOD,
    DEFAULT_PRIMARY_METRIC,
    DEFAULT_PRIMARY_WEIGHTING,
    LOWER_IS_BETTER,
    MissingMetricError,
    NonFiniteMetricError,
    PER_ITEM_TAIL_LADDER,
    PER_ITEM_TAIL_METRICS,
    POOLED_TOKEN_METRICS,
    SCHEMA_VERSION,
)
from stats.inference import (
    _build_weighting_block,
    _exp_nll,
    _ppl_block,
    _ppl_ratio_block,
    _rms_dp_block,
    holm_adjust,
)
from stats.tokens import (
    DEFAULT_POSITION_BUCKETS,
    _per_item_tail_blocks,
    _pooled_distribution_block,
    _position_bucket_blocks,
)

def _metric_arrays(scores: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    """Pull a per-item metric column; raise if absent in any item."""
    out = []
    for i, s in enumerate(scores):
        if metric not in s:
            raise MissingMetricError(f"metric {metric!r} missing for item index {i}")
        out.append(float(s[metric]))
    return np.asarray(out, dtype=float)


def _optional_metric_arrays(scores: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray | None:
    if not scores or any(metric not in s for s in scores):
        return None
    return np.asarray([float(s[metric]) for s in scores], dtype=float)


def _weighted_metric_mean(values: np.ndarray, weights: np.ndarray, weighting: str) -> float:
    if weighting == "item":
        mean = float(values.mean())
    elif weighting == "token":
        mean = float((weights * values).sum() / weights.sum())
    else:
        raise ValueError(f"weighting must be item|token; got {weighting!r}")
    if not math.isfinite(mean):
        raise NonFiniteMetricError("non-finite metric mean; paired comparison aborted")
    return mean


def _candidate_metric_summary(
    scores_a: Sequence[Mapping[str, Any]],
    scores_b: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    *,
    metric: str,
    unit: str,
    model_a_label: str,
    model_b_label: str,
) -> dict[str, Any] | None:
    a_vals = _optional_metric_arrays(scores_a, metric)
    b_vals = _optional_metric_arrays(scores_b, metric)
    if a_vals is None or b_vals is None:
        return None
    return {
        "scope": "candidate_self",
        "unit": unit,
        "n_items_used": int(a_vals.size),
        "description": (
            "Candidate-only full-logits metric. No reference comparison, "
            "candidate-pair delta, confidence interval, or verdict is computed."
        ),
        "candidate_a": {
            "label": model_a_label,
            "item_weighted_mean": _weighted_metric_mean(a_vals, weights, "item"),
            "token_weighted_mean": _weighted_metric_mean(a_vals, weights, "token"),
        },
        "candidate_b": {
            "label": model_b_label,
            "item_weighted_mean": _weighted_metric_mean(b_vals, weights, "item"),
            "token_weighted_mean": _weighted_metric_mean(b_vals, weights, "token"),
        },
    }


def _validate_compare_inputs(scores_a, scores_b, weights, token_metrics_a,
                             token_metrics_b, item_keys, seed) -> None:
    if seed is None:
        raise ValueError("seed must be an int (item & token bootstraps share it)")
    if len(scores_a) != len(scores_b) or len(scores_a) != len(weights):
        raise AlignmentError("scores_a, scores_b, weights must have equal length")
    if len(scores_a) < 2:
        raise AlignmentError("paired comparison requires at least two items")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be a one-dimensional sequence of finite positive values")
    for role, scores in (("candidate-a", scores_a), ("candidate-b", scores_b)):
        for i, score in enumerate(scores):
            for metric, value in score.items():
                if not math.isfinite(float(value)):
                    raise NonFiniteMetricError(f"{role} item {i}: non-finite {metric!r}; paired comparison aborted")
    if item_keys is not None and len(item_keys) != len(scores_a):
        raise AlignmentError("item_keys must align 1:1 with scores_a")
    if (token_metrics_a is None) != (token_metrics_b is None):
        raise ValueError(
            "token_metrics_a and token_metrics_b must be given together")
    if token_metrics_a is not None:
        if set(token_metrics_a) != set(token_metrics_b):
            raise ValueError(
                f"token_metrics_a keys {sorted(token_metrics_a)} != "
                f"token_metrics_b keys {sorted(token_metrics_b)}")
        for name in token_metrics_a:
            cols_a, cols_b = token_metrics_a[name], token_metrics_b[name]
            if len(cols_a) != len(scores_a) or len(cols_b) != len(scores_b):
                raise AlignmentError(
                    f"token_metrics[{name!r}] must align 1:1 with "
                    "scores_a/scores_b")
            # Per-item length guard: every kept item's token arrays must both
            # hold exactly weights[i] (= that item's scored position count)
            # values. A backend bug that dropped or duplicated a segment (e.g.
            # in the GPU retry path) would otherwise silently skew the pooled
            # ladders instead of failing loudly.
            for i, (ta, tb) in enumerate(zip(cols_a, cols_b)):
                na, nb = np.asarray(ta).size, np.asarray(tb).size
                if not (na == nb == weights[i]):
                    raise AlignmentError(
                        f"item {i}: per-token {name!r} lengths (a={na}, "
                        f"b={nb}) != weights[{i}]={weights[i]}; the per-token "
                        "arrays are misaligned with the per-item scores")
                if not (np.all(np.isfinite(ta)) and np.all(np.isfinite(tb))):
                    raise NonFiniteMetricError(f"item {i}: non-finite per-token {name!r}; paired comparison aborted")


def _metric_blocks(scores_a, scores_b, w, metrics, resolved_primary,
                   equivalence_margin, confidence_level, bootstrap_iters,
                   seed, ci_method) -> dict[str, dict[str, Any]]:
    """Item inference and token-weighted descriptions in metric request order."""
    metric_results: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        margin = equivalence_margin if metric == resolved_primary else None
        a_vals = _metric_arrays(scores_a, metric)
        b_vals = _metric_arrays(scores_b, metric)
        score_direction = "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"
        item_block = _build_weighting_block(
            a_vals, b_vals, w, weighting="item", score_direction=score_direction,
            confidence_level=confidence_level, bootstrap_iters=bootstrap_iters,
            seed=seed, ci_method=ci_method, equivalence_margin=margin)
        token_block = _build_weighting_block(
            a_vals, b_vals, w, weighting="token", score_direction=score_direction,
            confidence_level=confidence_level, bootstrap_iters=bootstrap_iters,
            seed=seed, ci_method=ci_method)
        metric_results[metric] = {
            "score_direction": score_direction,
            "n_items_used": int(a_vals.size),
            "item_weighted": item_block,
            "token_weighted": token_block,
        }
    return metric_results


def _add_derived_blocks(metric_results: dict[str, dict[str, Any]]) -> None:
    """ppl / ppl_ratio (from nll) and rms_dp (from mse_dp): display-only
    transforms of an existing block's numbers, appended in place."""
    if "nll" in metric_results:
        nll_r = metric_results["nll"]
        metric_results["ppl"] = {
            "source_metric": "nll", "score_direction": "lower_is_better",
            "item_weighted": _ppl_block(nll_r["item_weighted"], "nll.item_weighted"),
            "token_weighted": _ppl_block(nll_r["token_weighted"], "nll.token_weighted"),
        }
        metric_results["ppl_ratio"] = {
            "source_metric": "nll", "score_direction": "lower_is_better",
            "item_weighted": _ppl_ratio_block(nll_r["item_weighted"], "nll.item_weighted"),
            "token_weighted": _ppl_ratio_block(nll_r["token_weighted"], "nll.token_weighted"),
        }
    if "mse_dp" in metric_results:
        mse_r = metric_results["mse_dp"]
        metric_results["rms_dp"] = {
            "source_metric": "mse_dp", "score_direction": "lower_is_better",
            "item_weighted": _rms_dp_block(mse_r["item_weighted"], "mse_dp.item_weighted"),
            "token_weighted": _rms_dp_block(mse_r["token_weighted"], "mse_dp.token_weighted"),
        }


def _side_metrics(scores_a, scores_b, w, model_a_label, model_b_label):
    """Candidate-only summaries (entropy, signed mean_dp) and the shared
    reference's own NLL/PPL. Returns (candidate_metrics, reference_metrics)."""
    candidate_metrics: dict[str, Any] = {}
    for cname, cunit in (("entropy", "nats"), ("mean_dp", "pp, signed")):
        summary = _candidate_metric_summary(
            scores_a, scores_b, w, metric=cname, unit=cunit,
            model_a_label=model_a_label, model_b_label=model_b_label)
        if summary is not None:
            candidate_metrics[cname] = summary

    # Report reference NLL from side A and record any difference from side B.
    # The caller checks the shared reference and target-token provenance.
    reference_metrics = None
    ref_nll_a = _optional_metric_arrays(scores_a, "nll_ref")
    ref_nll_b = _optional_metric_arrays(scores_b, "nll_ref")
    if ref_nll_a is not None and ref_nll_b is not None:
        drift = float(np.abs(ref_nll_a - ref_nll_b).max()) if ref_nll_a.size else 0.0
        item_mean = _weighted_metric_mean(ref_nll_a, w, "item")
        token_mean = _weighted_metric_mean(ref_nll_a, w, "token")
        reference_metrics = {
            "scope": "reference_self",
            "n_items_used": int(ref_nll_a.size),
            "nll": {"item_weighted_mean": item_mean,
                    "token_weighted_mean": token_mean},
            "ppl": {"item_weighted": _exp_nll(item_mean),
                    "token_weighted": _exp_nll(token_mean)},
            "max_abs_side_difference": drift,
            "description": (
                "The shared FP reference's teacher-forced NLL and perplexity "
                "from full logits at the same target tokens. These differ from "
                "PPL(base) reconstructed from llama-perplexity's saved clipped "
                "and quantized reference. Values use candidate-a's recorded "
                "reference column; max_abs_side_difference reports any "
                "difference from candidate-b's reference column."),
        }
    return candidate_metrics, reference_metrics


def _adjust_exploratory_cells(cells, confidence_level):
    """Retain every planned cell; unavailable tests count as p=1 internally."""
    adjusted = holm_adjust([c.get("p_value", 1.0) for c in cells])
    censored = any(c.get("p_value_metadata", {}).get("censored", False) for c in cells)
    for cell, p_adj in zip(cells, adjusted):
        cell["holm_family_size"] = len(cells)
        if "p_value" not in cell:
            continue
        cell["p_value_holm"] = p_adj
        if censored:
            cell["p_value_holm_is_upper_bound"] = True
        cell["holm_significant"] = bool(p_adj < 1.0 - confidence_level)


def _apply_multiplicity(metric_results, metrics, primary_metric, primary_weighting,
                        confidence_level, equivalence_margin) -> dict[str, Any]:
    """One item-weighted primary endpoint; Holm over the remaining item endpoints."""
    if primary_weighting != "item":
        raise ValueError("primary_weighting must be 'item'; token weighting is descriptive only")
    primary_key = f"{primary_weighting}_weighted"
    if primary_metric is None:
        # Auto: the standing default when it is being compared, else the
        # first requested metric -- a --metrics subset that omits kld must
        # still get a designated confirmatory endpoint, not an error.
        primary_metric = (DEFAULT_PRIMARY_METRIC
                          if DEFAULT_PRIMARY_METRIC in metric_results
                          else next((m for m in metrics if m in metric_results),
                                    None))
    if primary_metric not in metric_results:
        raise ValueError(
            f"primary_metric {primary_metric!r} is not among the compared "
            f"metrics {sorted(metric_results)}")
    family: list[tuple[str, str]] = []
    for name in metrics:
        mres = metric_results.get(name)
        if not mres:
            continue
        for weighting_key in ("item_weighted",):
            block = mres.get(weighting_key)
            if not isinstance(block, dict) or block.get("p_value") is None:
                continue
            if (name, weighting_key) == (primary_metric, primary_key):
                block["role"] = "primary"
                continue
            block["role"] = "exploratory"
            family.append((name, weighting_key))
    _adjust_exploratory_cells([metric_results[n][k] for n, k in family], confidence_level)
    alpha = 1.0 - confidence_level
    return {
        "primary_endpoint": {"metric": primary_metric,
                             "weighting": primary_weighting,
                             "equivalence_margin": equivalence_margin},
        "correction": "holm_bonferroni",
        "family_size": len(family),
        "alpha": alpha,
        "policy": (
            "One confirmatory endpoint whose interval spends the whole alpha; "
            "every other item-weighted endpoint is exploratory and carries a "
            "Holm-Bonferroni-adjusted p-value over the remaining "
            f"{len(family)} cells. Holm controls the family-wise error rate "
            "under arbitrary dependence: kld/reversed_kld/js_kld are functionals "
            "of the same distribution pair. Token-weighted summaries are descriptive only."),
    }


def _token_level_blocks(token_metrics_a, token_metrics_b, item_keys, position_buckets,
                        confidence_level, bootstrap_iters, seed, ci_method):
    """Pooled per-token ladders (descriptive, no inference), the
    answer-position strata (their OWN Holm family) and the per-item tails
    (their OWN Holm family too). Returns (pooled_distribution,
    position_strata, per_item_tails), all empty without token data."""
    pooled_distribution: dict[str, Any] = {}
    position_strata: dict[str, Any] = {}
    per_item_tails: dict[str, Any] = {}
    if token_metrics_a is not None:
        tail_cells: list[dict[str, Any]] = []
        for name in PER_ITEM_TAIL_METRICS:
            if name not in token_metrics_a:
                continue
            direction = ("lower_is_better" if name in LOWER_IS_BETTER
                         else "higher_is_better")
            block = _per_item_tail_blocks(
                token_metrics_a[name], token_metrics_b[name], item_keys,
                token_metrics_a.get("target"),
                metric=name, score_direction=direction,
                confidence_level=confidence_level,
                bootstrap_iters=bootstrap_iters, seed=seed, ci_method=ci_method)
            per_item_tails[name] = block
            tail_cells.extend(block["cells"])
        _adjust_exploratory_cells(tail_cells, confidence_level)
        bucket_cells: list[dict[str, Any]] = []
        for name in POOLED_TOKEN_METRICS:
            if name not in token_metrics_a:
                continue
            direction = ("lower_is_better" if name in LOWER_IS_BETTER
                         else "higher_is_better")
            # A KLD tail is far more actionable next to how bad AGREEMENT got
            # at exactly those tokens, so pair kld with ear where available.
            companion = "ear" if (name == "kld" and "ear" in token_metrics_a) else None
            pooled_distribution[name] = _pooled_distribution_block(
                token_metrics_a[name], token_metrics_b[name],
                metric=name, score_direction=direction, item_keys=item_keys,
                companion_a=token_metrics_a.get(companion) if companion else None,
                companion_b=token_metrics_b.get(companion) if companion else None,
                companion_metric=companion)
            if position_buckets:
                cells = _position_bucket_blocks(
                    token_metrics_a[name], token_metrics_b[name],
                    metric=name, score_direction=direction,
                    edges=position_buckets, confidence_level=confidence_level,
                    bootstrap_iters=bootstrap_iters, seed=seed,
                    ci_method=ci_method)
                position_strata[name] = cells
                bucket_cells.extend(cells)
        _adjust_exploratory_cells(bucket_cells, confidence_level)
    return pooled_distribution, position_strata, per_item_tails


def compare_items(
    scores_a: Sequence[Mapping[str, Any]],
    scores_b: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    metrics: Sequence[str],
    confidence_level: float,
    bootstrap_iters: int,
    seed: int,
    model_a_label: str,
    model_b_label: str,
    ci_method: str = DEFAULT_CI_METHOD,
    primary_metric: str | None = None,
    primary_weighting: str = DEFAULT_PRIMARY_WEIGHTING,
    equivalence_margin: float | None = None,
    position_buckets: Sequence[int] | None = DEFAULT_POSITION_BUCKETS,
    token_metrics_a: Mapping[str, Sequence[np.ndarray]] | None = None,
    token_metrics_b: Mapping[str, Sequence[np.ndarray]] | None = None,
    item_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Paired comparison of two aligned lists of per-item score dicts.
    scores_a[i] and scores_b[i] describe the SAME item i; weights[i] = T_i.

    `token_metrics_a`/`token_metrics_b` (optional, both or neither) map a
    per-token metric name (POOLED_TOKEN_METRICS: "kld", "ear", "dp") to the
    aligned per-item arrays of that metric's per-token values, optionally
    plus the TOKEN_ANNOTATION_COLUMNS ("target": the per-position target
    token ids, used only to label tail witnesses). When provided they
    produce result["per_item_tails"] (each item's own p99 / p99.9 / max of
    the PER_ITEM_TAIL_METRICS columns with witness positions, and one
    exploratory paired test per level) and
    result["pooled_token_distribution"]: for each metric, the full
    quantile ladder over ALL tokens of ALL items flattened together (the
    llama-perplexity --kl-divergence scale and convention), per candidate,
    with the (item, position) each level's value came from and the raw b - a
    difference. Those rows carry NO confidence interval, NO bootstrap and NO
    verdict -- see _pooled_ladder for why. `item_keys` (optional) labels the
    witnesses with the item's dump key instead of only its index.

    Callers without token-level data (e.g. the golden-fixture parity test)
    simply omit them and get the mean-metric blocks alone.

    Pipeline (each stage is a helper above, in this order): validate ->
    per-metric weighting blocks -> derived blocks -> side metrics ->
    multiplicity -> token-level blocks -> assemble."""
    if primary_weighting != "item":
        raise ValueError("primary_weighting must be 'item'; token weighting is descriptive only")
    _validate_compare_inputs(scores_a, scores_b, weights, token_metrics_a,
                             token_metrics_b, item_keys, seed)

    w = np.asarray(weights, dtype=float)
    # The equivalence margin is in the PRIMARY metric's own units (nats for
    # kld, pp^2 for mse_dp, ...), so it is applied to that metric only --
    # one number cannot be a meaningful margin for all of them at once. It
    # is resolved here for the item-weighted primary metric.
    resolved_primary = primary_metric
    if resolved_primary is None:
        resolved_primary = (DEFAULT_PRIMARY_METRIC
                            if DEFAULT_PRIMARY_METRIC in metrics
                            else (metrics[0] if metrics else None))
    metric_results = _metric_blocks(
        scores_a, scores_b, w, metrics, resolved_primary, equivalence_margin,
        confidence_level, bootstrap_iters, seed, ci_method)
    _add_derived_blocks(metric_results)
    candidate_metrics, reference_metrics = _side_metrics(
        scores_a, scores_b, w, model_a_label, model_b_label)

    # ---- multiplicity: one confirmatory endpoint, Holm over the rest ---- #
    multiplicity = _apply_multiplicity(
        metric_results, metrics, primary_metric, primary_weighting,
        confidence_level, equivalence_margin)

    pooled_distribution, position_strata, per_item_tails = _token_level_blocks(
        token_metrics_a, token_metrics_b, item_keys, position_buckets,
        confidence_level, bootstrap_iters, seed, ci_method)

    result = {
        "schema_version": SCHEMA_VERSION,
        "n_items": len(scores_a),
        "confidence_level": confidence_level,
        "ci_method": ci_method,
        "multiplicity": multiplicity,
        # The t interval draws no replicates; say 0 rather than echo an
        # argument nothing consumed.
        "bootstrap_iters": 0 if ci_method == "t" else bootstrap_iters,
        "seed": seed,
        "model_a_label": model_a_label,
        "model_b_label": model_b_label,
        "metrics": metric_results,
    }
    if candidate_metrics:
        result["candidate_metrics"] = candidate_metrics
    if reference_metrics is not None:
        result["reference_metrics"] = reference_metrics
    if pooled_distribution:
        result["pooled_token_distribution"] = pooled_distribution
    if per_item_tails:
        result["per_item_tails"] = {
            "unit": "per-item quantile / maximum of that item's per-token "
                    "column, each with the answer position (and target "
                    "token, when known) it occurred at; one paired test per "
                    "level over items",
            "role": "exploratory",
            "correction": "holm_bonferroni within the per-item tail family",
            "convention": ("linear interpolation between neighbouring order "
                           "statistics (numpy default / Hyndman-Fan type 7), "
                           "the pooled ladder's definition applied to one "
                           "item; witness = nearest order statistic, ties to "
                           "the earliest position; max is exact"),
            "levels": [name for name, _ in PER_ITEM_TAIL_LADDER],
            "metrics": per_item_tails,
        }
    if position_strata:
        result["position_strata"] = {
            "edges": [int(x) for x in position_buckets],
            "unit": "per-item mean over that item's positions in the bucket; "
                    "paired item bootstrap; items with no positions in a "
                    "bucket are dropped from it",
            "role": "exploratory",
            "correction": "holm_bonferroni within the bucket family",
            "metrics": position_strata,
        }
    return result


# --------------------------------------------------------------------------- #
