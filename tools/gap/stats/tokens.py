# stats/tokens.py -- Per-token machinery: answer-position strata and the pooled quantile
# ladders with witnesses (descriptive; no bootstrap). NUMERICAL_CONTRACT.md #7.
import math
from typing import Any, Sequence

import numpy as np

from stats.contracts import (
    InferenceUnavailableError,
    PER_ITEM_TAIL_LADDER,
    POOLED_LADDER,
    POOLED_TAIL_ROWS,
)
from stats.inference import _build_weighting_block

# Pooled-token distribution ladders (descriptive; NO bootstrap, NO CI)
# --------------------------------------------------------------------------- #
# Default answer-position strata. The repo's own measurements
# (knowledge/variance-estimation-nonstationarity-theory.md) put the mmproj
# F16-vs-Q8_0 signal in the FIRST ~32 answer tokens, decaying more than 10x
# after that, while the default eval cap is 1024 -- so an item-mean over all
# positions dilutes the vision effect roughly 30x. Without a position
# breakdown the only lever is a prefix cap, which throws away the rest of the
# data instead of stratifying it.
DEFAULT_POSITION_BUCKETS = (0, 32, 256)


def _bucket_ranges(edges: Sequence[int]) -> list[tuple[int, int, str]]:
    """[(lo, hi, label), ...] from ascending edges; the last bucket is open."""
    e = [int(x) for x in edges]
    if not e or e[0] != 0 or any(b <= a for a, b in zip(e, e[1:])):
        raise ValueError(
            f"position bucket edges must start at 0 and increase; got {edges}")
    out = []
    for i, lo in enumerate(e):
        hi = e[i + 1] if i + 1 < len(e) else None
        out.append((lo, hi, f"{lo}-{hi}" if hi is not None else f"{lo}+"))
    return out


def _optional_weighting_block(a, b, weights, **kwargs):
    """Keep valid descriptive values when the requested exploratory CI is unsupported."""
    try:
        return _build_weighting_block(a, b, weights, **kwargs)
    except InferenceUnavailableError as error:
        return {"baseline_mean": float(a.mean()), "candidate_mean": float(b.mean()),
                "delta_candidate_minus_baseline": float((b - a).mean()),
                "inference_status": "unavailable", "requested_ci_method": kwargs["ci_method"],
                "skipped": str(error)}


def _position_bucket_blocks(
    token_a: Sequence[np.ndarray],
    token_b: Sequence[np.ndarray],
    *,
    metric: str,
    score_direction: str,
    edges: Sequence[int],
    confidence_level: float,
    bootstrap_iters: int,
    seed: int,
    ci_method: str,
) -> list[dict[str, Any]]:
    """Per-item means of one per-token metric restricted to each answer-position
    stratum, tested with the SAME paired item bootstrap as everything else.

    The unit stays the ITEM: within a bucket an item contributes the mean of
    its own positions there, and items with no positions in the bucket are
    dropped from that bucket's test (their count is reported). That keeps the
    exchangeable unit intact -- a per-position bootstrap would understate SE
    exactly as it does for the overall means.

    Buckets are EXPLORATORY by construction: the strata are chosen from prior
    measurements on this data, so the caller Holm-adjusts across them and the
    renderer labels them.
    """
    blocks = []
    for lo, hi, label in _bucket_ranges(edges):
        a_vals, b_vals, weights = [], [], []
        for ta, tb in zip(token_a, token_b):
            sa = np.asarray(ta, dtype=np.float64)[lo:hi]
            sb = np.asarray(tb, dtype=np.float64)[lo:hi]
            if sa.size == 0 or sb.size == 0:
                continue
            a_vals.append(float(sa.mean()))
            b_vals.append(float(sb.mean()))
            weights.append(float(sa.size))
        n_items = len(a_vals)
        entry = {
            "metric": metric,
            "bucket": label,
            "position_lo": lo,
            "position_hi": hi,
            "n_items_used": n_items,
            "n_items_without_positions": len(token_a) - n_items,
            "role": "exploratory",
        }
        if n_items < 2:
            # A one-item bootstrap CI is zero-width and would read as
            # significant for every metric; say why the row is empty instead.
            entry["skipped"] = (
                f"only {n_items} item(s) reach positions {label}; paired "
                "inference needs >= 2")
            blocks.append(entry)
            continue
        entry.update(_optional_weighting_block(
            np.asarray(a_vals), np.asarray(b_vals), np.asarray(weights),
            weighting="item", score_direction=score_direction,
            confidence_level=confidence_level,
            bootstrap_iters=bootstrap_iters, seed=seed, ci_method=ci_method))
        # Re-assert after update(): the weighting block carries its own
        # `role` key (None until the multiplicity pass names the primary),
        # and it just overwrote the one set above.
        entry["role"] = "exploratory"
        blocks.append(entry)
    return blocks


def _pooled_ladder(token_values: Sequence[np.ndarray],
                   item_keys: Sequence[str] | None = None,
                   *, ladder=POOLED_LADDER) -> dict[str, Any]:
    """Quantile ladder over ALL scored tokens of ALL items, flattened into one
    multiset -- llama-perplexity --kl-divergence's convention and scale -- plus,
    for each level, the token the value came from.

    DESCRIPTIVE ONLY: no confidence interval, no bootstrap, no verdict. Two
    reasons, and they point the same way:

    * The maximum has no valid nonparametric bootstrap. A resample can only
      contain a subset of the observed items, so the replicate distribution of
      the pooled max is degenerate (a point mass at the observed value plus a
      one-sided tail) and its percentile interval does not approximate the
      sampling distribution. Measured coverage of a true zero delta under the
      exact null was 0.887 at n=30 and 0.860 at n=150 against a nominal 0.95 --
      roughly three times the intended Type-I rate, and getting WORSE as items
      are added. That is not fixable by collecting more data.
    * The far quantiles (p99, p99.9) are estimated from a handful of tokens in
      a handful of items. Their interval, even where it is technically valid,
      is too wide to support a verdict and inviting one invites tail-mining.

    So these rows are reported as what they are: descriptive statistics of the
    observed token distribution, on the same footing as the numbers
    llama-perplexity prints. The confirmatory inference lives on the mean
    metrics, which have a well-behaved item-level bootstrap.

    Witnesses: for each level, `witness` names the pooled token whose value the
    (generally interpolated) quantile is NEAREST to -- for `max`/`min` that is
    the extreme token exactly. `position` is the 0-based index among that
    item's scored ANSWER positions, after any compare-time cap. Ties resolve
    to the earliest (item, position) in input order, so the witness is
    deterministic.

    Returns {"n_tokens", "n_items", "levels": [ {...}, ... ]}.
    """
    arrays = [np.asarray(t, dtype=np.float64).reshape(-1) for t in token_values]
    n = len(arrays)
    if n == 0 or any(a.size == 0 for a in arrays):
        raise ValueError(
            "pooled distribution needs >= 1 item and >= 1 token per item "
            f"(got {n} items, min tokens "
            f"{min((a.size for a in arrays), default=0)})")
    if item_keys is not None and len(item_keys) != n:
        raise ValueError("item_keys must align 1:1 with token_values")
    lens = np.array([a.size for a in arrays], dtype=np.int64)
    pooled = np.concatenate(arrays)
    item_of = np.repeat(np.arange(n, dtype=np.int64), lens)
    pos_of = np.concatenate([np.arange(L, dtype=np.int64) for L in lens])
    # stable sort so equal values keep (item, position) input order, which is
    # what makes the witness deterministic across runs and across --jobs.
    order = np.argsort(pooled, kind="stable")
    sorted_vals = pooled[order]

    levels = [q for _, q in ladder]
    values = np.quantile(pooled, levels, method="linear")

    total = pooled.size

    def token_record(rank: int) -> dict[str, Any]:
        """The pooled token at a given rank of the sorted multiset."""
        rank = int(np.clip(rank, 0, total - 1))
        t = int(order[rank])
        item_index = int(item_of[t])
        rec = {"item_index": item_index, "position": int(pos_of[t]),
               "token_value": float(pooled[t]), "flattened_rank": rank}
        if item_keys is not None:
            rec["item_key"] = str(item_keys[item_index])
        return rec

    out_levels = []
    for j, (name, q) in enumerate(ladder):
        # Linear interpolation (numpy's default / llama-perplexity's lambda)
        # reads the value at fractional rank q*(N-1) BETWEEN two order
        # statistics. Record BOTH of them plus the weight, so the number is
        # reconstructible from the report instead of merely attributed to one
        # token -- for an interpolated level no single token holds the value.
        frank = float(q) * (total - 1)
        lo_rank = int(math.floor(frank))
        hi_rank = min(lo_rank + 1, total - 1)
        lower, upper = token_record(lo_rank), token_record(hi_rank)
        weight_upper = float(frank - lo_rank)
        # `witness` stays the single nearest order statistic: the one row a
        # narrow markdown table can carry.
        witness = upper if weight_upper >= 0.5 else lower
        out_levels.append({
            "name": name, "quantile": float(q), "value": float(values[j]),
            "fractional_rank": frank, "flattened_token_count": total,
            "interpolation_weight_upper": weight_upper,
            "lower": lower, "upper": upper,
            "witness": {k: v for k, v in witness.items()
                        if k != "flattened_rank"},
        })
    return {"n_tokens": int(total), "n_items": n, "levels": out_levels}


# Which ladder levels get a per-item "who is driving this tail" breakdown,
# and which side of the distribution counts as the tail. `dp` is signed and
# two-sided, so it has no single degradation tail and gets none.
_CRITICAL_SAMPLE_TAILS = {
    "kld": ("upper", ("p99", "p999")),
    "ear": ("lower", ("p01", "p001")),
}
_CRITICAL_SAMPLE_TABLE_ROWS = 5     # markdown cap; JSON keeps every sample


def _critical_samples(token_values, item_keys, *, threshold: float, side: str,
                      companion=None):
    """Per-item breakdown of the tokens at or beyond `threshold`.

    A pooled quantile says how bad the tail is; it does not say WHERE the tail
    came from. In practice a handful of items usually own it, and knowing
    which ones is the difference between "the p99 is 0.86 nats" and "one chart
    image contributes 60% of the tail tokens". Sorted by tail-token count,
    then by that item's own extreme.

    `companion` (optional) is the same items' per-token values of ANOTHER
    metric — e.g. EAR alongside a KLD tail — so the report can say how bad
    agreement got at exactly the tokens the KLD tail is made of.
    """
    worst = (lambda a: float(a.max())) if side == "upper" else (lambda a: float(a.min()))
    out = []
    for i, values in enumerate(token_values):
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        hit = np.flatnonzero(v >= threshold if side == "upper" else v <= threshold)
        if hit.size == 0:
            continue
        entry = {
            "item_index": i,
            "item_key": (str(item_keys[i]) if item_keys is not None else str(i)),
            "item_token_count": int(v.size),
            "tail_token_count": int(hit.size),
            "tail_token_fraction": float(hit.size / v.size),
            "item_extreme": worst(v),
            "positions": [int(x) for x in hit],
            "values": [float(x) for x in v[hit]],
        }
        if companion is not None:
            c = np.asarray(companion[i], dtype=np.float64).reshape(-1)
            if c.size == v.size:
                entry["companion_values"] = [float(x) for x in c[hit]]
                entry["companion_min"] = float(c[hit].min())
        out.append(entry)
    out.sort(key=lambda e: (-e["tail_token_count"], -abs(e["item_extreme"]),
                            e["item_key"]))
    return out


def _per_item_tail_blocks(
    token_a: Sequence[np.ndarray],
    token_b: Sequence[np.ndarray],
    item_keys: Sequence[str] | None,
    targets: Sequence[np.ndarray] | None,
    *,
    metric: str,
    score_direction: str,
    confidence_level: float,
    bootstrap_iters: int,
    seed: int,
    ci_method: str,
) -> dict[str, Any]:
    """Per-item tail statistics of one per-token metric -- each item's own
    PER_ITEM_TAIL_LADDER levels (p99 / p99.9 / max) with the answer position
    (and target token, when `targets` is given) each came from -- plus one
    paired test per level over items.

    Per item the ladder is _pooled_ladder applied to that item alone, so the
    quantile convention (linear interpolation between neighbouring order
    statistics) and the witness rule (nearest order statistic, ties to the
    earliest position) are exactly the pooled ladder's. `max` is exact. For
    the interpolated levels the two bracketing positions and the weight are
    recorded too, so the number is reconstructible from the report.

    A quantile of a SHORT item rests on its top two tokens: p99 needs
    > 100 positions and p99.9 > 1000 to interpolate anywhere else, so for a
    typical 64..1024-token answer p999 is the max in all but name. Each
    interpolated level therefore records `rests_on_top_two` per item and the
    cell reports how many items that was.

    The tests: each level is a per-item scalar, tested with the SAME paired
    item statistics as the mean metrics (unit: the item). They are
    EXPLORATORY by construction -- a per-item maximum is a single-token
    statistic, far heavier-tailed than a mean, so its interval reaches
    nominal coverage later than the mean's -- and the caller Holm-adjusts
    them as their own family so they never borrow the primary endpoint's
    alpha and never cost the main family any power."""
    n = len(token_a)
    if len(token_b) != n:
        raise ValueError("token_a and token_b must align 1:1")
    if item_keys is not None and len(item_keys) != n:
        raise ValueError("item_keys must align 1:1 with token_a")
    if targets is not None and len(targets) != n:
        raise ValueError("targets must align 1:1 with token_a")
    level_names = [name for name, _ in PER_ITEM_TAIL_LADDER]
    per_level = {"candidate_a": {L: [] for L in level_names},
                 "candidate_b": {L: [] for L in level_names}}
    weights: list[float] = []
    top_two = {L: 0 for L in level_names if L != "max"}
    items: list[dict[str, Any]] = []
    for i, (ta, tb) in enumerate(zip(token_a, token_b)):
        ta = np.asarray(ta, dtype=np.float64).reshape(-1)
        tb = np.asarray(tb, dtype=np.float64).reshape(-1)
        if ta.size != tb.size:
            raise ValueError(f"item {i}: per-token lengths differ (a={ta.size}, "
                             f"b={tb.size})")
        npos = int(ta.size)
        rec: dict[str, Any] = {"item_index": i, "n_positions": npos}
        if item_keys is not None:
            rec["item_key"] = str(item_keys[i])
        if npos == 0:
            rec["skipped"] = "no scored positions"
            items.append(rec)
            continue
        tgt = None if targets is None else np.asarray(targets[i]).reshape(-1)
        if tgt is not None and tgt.size != npos:
            raise ValueError(f"item {i}: targets length {tgt.size} != {npos}")
        for side, col in (("candidate_a", ta), ("candidate_b", tb)):
            ladder = _pooled_ladder([col], None, ladder=PER_ITEM_TAIL_LADDER)
            side_rec: dict[str, Any] = {}
            for lev in ladder["levels"]:
                pos = int(lev["witness"]["position"])
                entry: dict[str, Any] = {"value": float(lev["value"]),
                                         "position": pos}
                if tgt is not None:
                    entry["target_token"] = int(tgt[pos])
                if lev["name"] != "max":
                    entry["lower_position"] = int(lev["lower"]["position"])
                    entry["upper_position"] = int(lev["upper"]["position"])
                    entry["interpolation_weight_upper"] = float(
                        lev["interpolation_weight_upper"])
                    entry["rests_on_top_two"] = bool(
                        lev["fractional_rank"] > npos - 2)
                side_rec[lev["name"]] = entry
                per_level[side][lev["name"]].append(entry["value"])
            rec[side] = side_rec
        rec["delta_b_minus_a"] = {
            L: rec["candidate_b"][L]["value"] - rec["candidate_a"][L]["value"]
            for L in level_names}
        for L in top_two:
            # a property of npos alone, identical for both candidates
            top_two[L] += int(rec["candidate_a"][L]["rests_on_top_two"])
        weights.append(float(npos))
        items.append(rec)

    cells: list[dict[str, Any]] = []
    for name, q in PER_ITEM_TAIL_LADDER:
        a_vals = np.asarray(per_level["candidate_a"][name], dtype=np.float64)
        b_vals = np.asarray(per_level["candidate_b"][name], dtype=np.float64)
        entry = {
            "metric": metric,
            "level": name,
            "quantile": float(q),
            "n_items_used": int(a_vals.size),
            "n_items_without_positions": n - int(a_vals.size),
            "role": "exploratory",
        }
        if name in top_two:
            entry["n_items_resting_on_top_two"] = top_two[name]
        if a_vals.size < 2:
            entry["skipped"] = (
                f"only {a_vals.size} item(s) with scored positions; paired "
                "inference needs >= 2")
            cells.append(entry)
            continue
        entry.update(_optional_weighting_block(
            a_vals, b_vals, np.asarray(weights, dtype=np.float64),
            weighting="item", score_direction=score_direction,
            confidence_level=confidence_level,
            bootstrap_iters=bootstrap_iters, seed=seed, ci_method=ci_method))
        entry["role"] = "exploratory"      # re-assert after update()
        cells.append(entry)
    return {"metric": metric, "score_direction": score_direction,
            "levels": level_names, "n_items": n, "cells": cells,
            "items": items}


def _pooled_distribution_block(
    token_values_a: Sequence[np.ndarray],
    token_values_b: Sequence[np.ndarray],
    *,
    metric: str,
    score_direction: str,
    item_keys: Sequence[str] | None = None,
    companion_a: Sequence[np.ndarray] | None = None,
    companion_b: Sequence[np.ndarray] | None = None,
    companion_metric: str | None = None,
) -> dict[str, Any]:
    """Both candidates' ladders for one per-token metric, plus the raw
    b - a difference per level and, for the metrics with a clear degradation
    direction, the per-item breakdown of who owns each far tail. The
    difference carries NO interval and NO verdict -- see _pooled_ladder."""
    a = _pooled_ladder(token_values_a, item_keys)
    b = _pooled_ladder(token_values_b, item_keys)
    deltas = [
        {"name": la["name"], "quantile": la["quantile"],
         "delta_b_minus_a": lb["value"] - la["value"]}
        for la, lb in zip(a["levels"], b["levels"])
    ]
    block = {
        "metric": metric,
        "score_direction": score_direction,
        "statistic": "pooled_token_quantile",
        "scale": "all scored tokens of all items, flattened",
        "convention": ("linear interpolation between neighbouring order "
                       "statistics (numpy default / Hyndman-Fan type 7), the "
                       "same definition llama-perplexity --kl-divergence uses"),
        "inference": "none",
        "no_bootstrap_reason": (
            "descriptive only: the pooled max has no consistent nonparametric "
            "bootstrap (measured 0.86-0.89 coverage for a nominal 0.95, "
            "degrading as items are added) and the far quantiles rest on too "
            "few tokens to carry a verdict"),
        "degradation_tail": list(POOLED_TAIL_ROWS.get(metric, ())),
        "candidate_a": a,
        "candidate_b": b,
        "delta_b_minus_a": deltas,
    }
    spec = _CRITICAL_SAMPLE_TAILS.get(metric)
    if spec is not None:
        side, level_names = spec
        by_name = {"candidate_a": {L["name"]: L["value"] for L in a["levels"]},
                   "candidate_b": {L["name"]: L["value"] for L in b["levels"]}}
        critical: dict[str, Any] = {
            "side": side,
            "companion_metric": companion_metric,
            "description": (
                f"Items owning the {side} tail of `{metric}`: every token at "
                f"or beyond that candidate's OWN quantile. Each side is "
                "thresholded by its own value, so the two lists answer "
                "'which items drive my tail', not 'which items differ'."),
            "levels": {},
        }
        for lname in level_names:
            critical["levels"][lname] = {
                "candidate_a": _critical_samples(
                    token_values_a, item_keys, threshold=by_name["candidate_a"][lname],
                    side=side, companion=companion_a),
                "candidate_b": _critical_samples(
                    token_values_b, item_keys, threshold=by_name["candidate_b"][lname],
                    side=side, companion=companion_b),
                "threshold_a": by_name["candidate_a"][lname],
                "threshold_b": by_name["candidate_b"][lname],
            }
        block["critical_samples"] = critical
    return block
