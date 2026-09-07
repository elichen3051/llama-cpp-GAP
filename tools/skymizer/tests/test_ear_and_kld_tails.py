"""EAR (Expected Acceptance Rate, arXiv:2605.02404) and the pooled-token
distribution ladders.

The ladders are DESCRIPTIVE: quantiles over every scored token of every item
flattened into one multiset -- llama-perplexity --kl-divergence's scale and
convention -- reported per candidate with the token each value came from, and
with no confidence interval, no bootstrap and no verdict. These tests pin the
ladder definition, the witness lookup, and the absence of inference.

Hermetic: pure-numpy fixtures; no GPU, no models, no built binary.
Run from tools/skymizer:  python3 -m pytest tests/test_ear_and_kld_tails.py -v
"""

import json
import math

import numpy as np
import pytest

from stats.contracts import (
    AlignmentError,
    DEFAULT_METRICS,
    LOWER_IS_BETTER,
    POOLED_LADDER,
    POOLED_TAIL_ROWS,
    POOLED_TOKEN_METRICS,
)
from stats.engine import compare_items
from stats.render import format_comparison_table
from stats.tokens import _pooled_distribution_block, _pooled_ladder


def _llama_perplexity_percentile(values, fraction):
    """llama-perplexity's percentile() lambda, transcribed
    (tools/perplexity/perplexity.cpp): sort, then linearly interpolate at the
    fractional rank fraction*(n-1). Identical to numpy's default method, which
    is what makes our ladder directly comparable to its printed table."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if fraction <= 0:
        return float(v[0])
    if fraction >= 1:
        return float(v[-1])
    p = fraction * (v.size - 1)
    ip = int(p)
    p -= ip
    return float((1 - p) * v[ip] + p * v[min(ip + 1, v.size - 1)])


# --------------------------------------------------------------------------- #
# EAR policy
# --------------------------------------------------------------------------- #
def test_ear_direction_is_higher_is_better():
    assert "ear" in DEFAULT_METRICS
    assert "ear" not in LOWER_IS_BETTER
    assert set(POOLED_TOKEN_METRICS) == {"kld", "ear", "dp"}


# --------------------------------------------------------------------------- #
# pooled ladder
# --------------------------------------------------------------------------- #
def _cols(rng, sizes, scale=1.0):
    return [np.abs(rng.normal(scale=scale, size=n)).astype(np.float32)
            for n in sizes]


def test_pooled_ladder_matches_numpy_and_llama_perplexity_on_the_flat_pool():
    """The ladder is a quantile of ALL tokens of ALL items concatenated -- not
    a mean of per-item quantiles -- using the same linear interpolation
    llama-perplexity prints."""
    rng = np.random.default_rng(7)
    cols = _cols(rng, [11, 4, 23, 7])
    flat = np.concatenate([c.astype(np.float64) for c in cols])
    ladder = _pooled_ladder(cols)

    assert ladder["n_tokens"] == flat.size and ladder["n_items"] == 4
    assert [L["name"] for L in ladder["levels"]] == [n for n, _ in POOLED_LADDER]
    for L in ladder["levels"]:
        assert L["value"] == pytest.approx(np.quantile(flat, L["quantile"]),
                                           rel=0, abs=0)
        assert L["value"] == pytest.approx(
            _llama_perplexity_percentile(flat, L["quantile"]), rel=1e-12)


def test_pooled_ladder_extremes_are_exact_order_statistics():
    rng = np.random.default_rng(8)
    cols = _cols(rng, [5, 9, 3])
    flat = np.concatenate([c.astype(np.float64) for c in cols])
    by_name = {L["name"]: L for L in _pooled_ladder(cols)["levels"]}
    assert by_name["max"]["value"] == float(flat.max())
    assert by_name["min"]["value"] == float(flat.min())


def test_pooled_ladder_witness_names_the_item_and_position():
    """The whole point of the witness: which prediction position produced the
    extreme. Planted maximum and minimum must be found exactly."""
    cols = [np.array([0.1, 0.2, 0.3], np.float32),
            np.array([0.4, 9.0, 0.5, 0.6], np.float32),
            np.array([0.01, 0.7], np.float32)]
    keys = ["000_a", "001_b", "002_c"]
    by_name = {L["name"]: L for L in _pooled_ladder(cols, keys)["levels"]}

    wmax = by_name["max"]["witness"]
    assert (wmax["item_index"], wmax["position"], wmax["item_key"]) == (1, 1, "001_b")
    assert wmax["token_value"] == pytest.approx(9.0)
    wmin = by_name["min"]["witness"]
    assert (wmin["item_index"], wmin["position"], wmin["item_key"]) == (2, 0, "002_c")


def test_pooled_ladder_witness_is_the_nearest_order_statistic():
    """Interpolated levels sit between two pooled tokens; the witness must be
    whichever of the two the value is closer to, and its recorded
    token_value must be that token's actual value."""
    rng = np.random.default_rng(9)
    cols = _cols(rng, [13, 17, 5])
    flat = np.concatenate([c.astype(np.float64) for c in cols])
    srt = np.sort(flat)
    for L in _pooled_ladder(cols)["levels"]:
        v, tv = L["value"], L["witness"]["token_value"]
        assert tv in set(flat.tolist())
        # nearest sorted neighbour of v
        i = int(np.searchsorted(srt, v, side="left"))
        cand = [srt[min(i, srt.size - 1)], srt[max(i - 1, 0)]]
        assert abs(tv - v) == pytest.approx(min(abs(c - v) for c in cand))


def test_pooled_ladder_ties_resolve_by_stable_rank():
    """Every token has the same value, so the stable sort keeps (item,
    position) input order and each level's neighbours are the tokens at its
    own fractional rank — deterministic across runs and --jobs."""
    cols = [np.full(4, 0.5, np.float32), np.full(6, 0.5, np.float32)]
    flat = [(0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    for L in _pooled_ladder(cols, ["000_a", "001_b"])["levels"]:
        assert L["value"] == pytest.approx(0.5)
        for side in ("lower", "upper"):
            rec = L[side]
            assert (rec["item_index"], rec["position"]) == flat[rec["flattened_rank"]]
    by_name = {L["name"]: L for L in _pooled_ladder(cols, ["000_a", "001_b"])["levels"]}
    assert (by_name["min"]["lower"]["item_index"],
            by_name["min"]["lower"]["position"]) == (0, 0)
    assert (by_name["max"]["upper"]["item_index"],
            by_name["max"]["upper"]["position"]) == (1, 5)


def test_pooled_ladder_records_both_interpolation_neighbours():
    """An interpolated quantile lies BETWEEN two order statistics, so no
    single token holds its value. Recording both neighbours and the weight
    makes the printed number reconstructible from the report."""
    rng = np.random.default_rng(77)
    cols = _cols(rng, [17, 9, 23])
    flat = np.concatenate([c.astype(np.float64) for c in cols])
    total = flat.size
    for L in _pooled_ladder(cols)["levels"]:
        lo, hi, w = L["lower"], L["upper"], L["interpolation_weight_upper"]
        assert 0.0 <= w < 1.0
        assert L["fractional_rank"] == pytest.approx(L["quantile"] * (total - 1))
        assert L["flattened_token_count"] == total
        # the reported value really is the weighted blend of the two
        assert L["value"] == pytest.approx(
            (1 - w) * lo["token_value"] + w * hi["token_value"], rel=1e-12)
        # ... and the witness is whichever neighbour carries more weight
        assert L["witness"]["token_value"] == pytest.approx(
            hi["token_value"] if w >= 0.5 else lo["token_value"])


def test_pooled_ladder_rejects_empty_input():
    with pytest.raises(ValueError, match=">= 1 item"):
        _pooled_ladder([])
    with pytest.raises(ValueError, match=">= 1 token per item"):
        _pooled_ladder([np.ones(3, np.float32), np.empty(0, np.float32)])
    with pytest.raises(ValueError, match="item_keys"):
        _pooled_ladder([np.ones(3, np.float32)], ["a", "b"])


def test_pooled_distribution_block_carries_no_inference():
    rng = np.random.default_rng(10)
    a = _cols(rng, [8, 12])
    b = [c * 2.0 for c in a]
    block = _pooled_distribution_block(a, b, metric="kld",
                                          score_direction="lower_is_better")
    assert block["inference"] == "none"
    assert block["degradation_tail"] == ["p99", "p999", "max"]
    # no inferential key anywhere in the structure (the prose in
    # no_bootstrap_reason is the only place the word "bootstrap" may appear)
    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _keys(v)
    seen = set(_keys(block))
    assert seen.isdisjoint({"ci", "ci_delta", "confidence_level",
                            "bootstrap_iters", "bootstrap_std", "decision",
                            "verdict"})
    assert "bootstrap" not in json.dumps(
        {k: v for k, v in block.items() if k != "no_bootstrap_reason"})
    for la, lb, d in zip(block["candidate_a"]["levels"],
                         block["candidate_b"]["levels"],
                         block["delta_b_minus_a"]):
        assert d["delta_b_minus_a"] == pytest.approx(lb["value"] - la["value"])


def test_ear_degradation_tail_is_the_low_end():
    """EAR is higher-is-better, so the informative tail is p01/p001/min. A
    literal "max EAR" is ~1.0 for any candidate worth measuring."""
    assert POOLED_TAIL_ROWS["ear"] == ("p01", "p001", "min")
    assert POOLED_TAIL_ROWS["kld"] == ("p99", "p999", "max")
    rng = np.random.default_rng(11)
    a = [1.0 - np.abs(rng.normal(scale=0.01, size=n)).astype(np.float32)
         for n in (30, 40)]
    b = [c - 0.02 for c in a]
    block = _pooled_distribution_block(a, b, metric="ear",
                                          score_direction="higher_is_better")
    assert block["degradation_tail"] == ["p01", "p001", "min"]
    by_a = {L["name"]: L["value"] for L in block["candidate_a"]["levels"]}
    by_b = {L["name"]: L["value"] for L in block["candidate_b"]["levels"]}
    # the worse candidate is worse at the LOW end; the max is uninformative
    assert by_b["min"] < by_a["min"]
    assert by_a["max"] > 0.97 and by_b["max"] > 0.95


# --------------------------------------------------------------------------- #
# compare_items integration
# --------------------------------------------------------------------------- #
def _paired_fixture(n=6, seed=5):
    rng = np.random.default_rng(seed)
    keys = [f"{i:03d}_item{i}" for i in range(n)]
    scores_a, scores_b, weights = [], [], []
    tok_a = {"kld": [], "ear": []}
    tok_b = {"kld": [], "ear": []}
    for i in range(n):
        T = 12 + i
        ka = np.abs(rng.normal(1e-3, 4e-4, T)).astype(np.float32)
        kb = (ka * 1.3).astype(np.float32)
        ea = (1.0 - np.abs(rng.normal(0.01, 0.004, T))).astype(np.float32)
        eb = (ea - 0.01).astype(np.float32)
        tok_a["kld"].append(ka); tok_b["kld"].append(kb)
        tok_a["ear"].append(ea); tok_b["ear"].append(eb)
        scores_a.append({"kld": float(ka.mean()), "ear": float(ea.mean())})
        scores_b.append({"kld": float(kb.mean()), "ear": float(eb.mean())})
        weights.append(T)
    return keys, scores_a, scores_b, weights, tok_a, tok_b


def _compare(**kw):
    keys, sa, sb, w, ta, tb = _paired_fixture()
    kw.setdefault("token_metrics_a", ta)
    kw.setdefault("token_metrics_b", tb)
    kw.setdefault("item_keys", keys)
    return compare_items(sa, sb, w, metrics=["kld", "ear"],
                            confidence_level=0.95, bootstrap_iters=500,
                            seed=1234, model_a_label="A", model_b_label="B",
                            **kw)


def test_compare_items_emits_both_pooled_ladders():
    res = _compare()
    pooled = res["pooled_token_distribution"]
    assert set(pooled) == {"kld", "ear"}   # this fixture supplies only those
    assert pooled["kld"]["candidate_a"]["n_items"] == 6
    # witnesses carry the dump key, not just an index
    w = pooled["kld"]["candidate_a"]["levels"][0]["witness"]
    assert w["item_key"].startswith("00")
    # and the verdict family holds ONLY the mean metrics
    assert set(res["metrics"]) == {"kld", "ear"}


def test_compare_items_without_token_metrics_has_no_pooled_block():
    keys, sa, sb, w, _ta, _tb = _paired_fixture()
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=500, seed=1, model_a_label="A",
                           model_b_label="B")
    assert "pooled_token_distribution" not in res
    assert set(res["metrics"]) == {"kld"}


def test_compare_items_ladder_is_independent_of_the_metric_list():
    """The pooled ladders describe the retained per-token columns; they do not
    require the matching mean metric to be selected."""
    keys, sa, sb, w, ta, tb = _paired_fixture()
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=500, seed=1, model_a_label="A",
                           model_b_label="B", token_metrics_a=ta,
                           token_metrics_b=tb, item_keys=keys)
    assert set(res["pooled_token_distribution"]) == {"kld", "ear"}


def test_compare_items_kld_only_columns_give_only_the_kld_ladder():
    """A v1 VLMK dir has no per-token `ear`; the EAR ladder is then absent
    rather than fabricated."""
    keys, sa, sb, w, ta, tb = _paired_fixture()
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=500, seed=1, model_a_label="A",
                           model_b_label="B",
                           token_metrics_a={"kld": ta["kld"]},
                           token_metrics_b={"kld": tb["kld"]},
                           item_keys=keys)
    assert set(res["pooled_token_distribution"]) == {"kld"}


def test_compare_items_rejects_one_sided_or_misaligned_token_metrics():
    keys, sa, sb, w, ta, tb = _paired_fixture()
    common = dict(metrics=["kld"], confidence_level=0.95, bootstrap_iters=500,
                  seed=1, model_a_label="A", model_b_label="B")
    with pytest.raises(ValueError, match="must be given together"):
        compare_items(sa, sb, w, **common, token_metrics_a=ta)
    with pytest.raises(ValueError, match="keys"):
        compare_items(sa, sb, w, **common, token_metrics_a=ta,
                         token_metrics_b={"kld": tb["kld"]})
    with pytest.raises(AlignmentError, match="align 1:1"):
        compare_items(sa, sb, w, **common,
                         token_metrics_a={k: v[:-1] for k, v in ta.items()},
                         token_metrics_b=tb)
    short = {k: [v[0][:-1]] + v[1:] for k, v in ta.items()}
    with pytest.raises(AlignmentError, match="per-token"):
        compare_items(sa, sb, w, **common, token_metrics_a=short,
                         token_metrics_b=tb)
    with pytest.raises(AlignmentError, match="item_keys"):
        compare_items(sa, sb, w, **common, token_metrics_a=ta,
                         token_metrics_b=tb, item_keys=keys[:-1])


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_report_renders_both_ladders_with_witnesses_and_no_ci():
    res = _compare()
    md = format_comparison_table(res, reference_label="F16")
    assert "## Per-token distributions (pooled tokens — NO bootstrap, NO CI)" in md
    assert "### KLD" in md and "### EAR" in md
    assert "witness (a)" in md and "witness (b)" in md
    # the degradation tail is bolded at opposite ends for the two metrics
    kld_section = md.split("### KLD")[1].split("### EAR")[0]
    ear_section = md.split("### EAR")[1]
    def _bold_levels(section):
        return {row.split("|")[1].strip().strip("*")
                for row in section.splitlines()
                if row.startswith("| **")}
    assert _bold_levels(kld_section) == {"Maximum", "99.9%", "99.0%"}
    assert _bold_levels(ear_section) == {"Minimum", "0.1%", "1.0%"}
    # no verdict / CI leaked into the descriptive tables
    for section in (kld_section, ear_section):
        assert "BETTER" not in section and "sig. diff." not in section
    # and no tail metric rows survive in the verdict table
    verdict_table = md.split("## Per-token distributions")[0]
    assert "kld_p99" not in verdict_table and "kld_max" not in verdict_table


def test_report_without_pooled_block_renders_unchanged():
    keys, sa, sb, w, _ta, _tb = _paired_fixture()
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=500, seed=1, model_a_label="A",
                           model_b_label="B")
    md = format_comparison_table(res, reference_label="F16")
    assert "Per-token distributions" not in md


# --------------------------------------------------------------------------- #
# statistics llama-perplexity reports and this tool used not to
# --------------------------------------------------------------------------- #
def test_signed_delta_p_is_reported_where_mse_dp_cannot_show_it():
    """mse_dp and rms_dp are unsigned by construction, so a candidate that is
    systematically OVER-confident and one that is systematically UNDER-confident
    look identical. The signed per-candidate mean and the signed pooled ladder
    (median included) are what separate them.

    An over-confident candidate's dp column (p_cand - p_ref at the target, pp)
    is all-positive; an under-confident one's is the negation -- equal and
    opposite systematic biases with identical mse_dp."""
    rng = np.random.default_rng(21)
    npos = 40
    dp_over = np.abs(rng.normal(1.0, 0.3, npos)).astype(np.float32)
    dp_under = (-dp_over).astype(np.float32)
    kld_col = np.abs(rng.normal(1e-3, 3e-4, npos)).astype(np.float32)
    mse = float((dp_over.astype(np.float64) ** 2).mean())
    s_over = {"kld": float(kld_col.mean()), "mse_dp": mse,
              "mean_dp": float(dp_over.astype(np.float64).mean())}
    s_under = {"kld": float(kld_col.mean()), "mse_dp": mse,
               "mean_dp": float(dp_under.astype(np.float64).mean())}

    # the unsigned metric cannot tell them apart in SIGN...
    assert s_over["mse_dp"] == s_under["mse_dp"] > 0
    # ... the signed one does
    assert s_over["mean_dp"] > 0 > s_under["mean_dp"]

    res = compare_items([s_over] * 4, [s_under] * 4, [npos] * 4,
                           metrics=["kld", "mse_dp"], confidence_level=0.95,
                           bootstrap_iters=1000, seed=1,
                           model_a_label="OVER", model_b_label="UNDER",
                           token_metrics_a={"kld": [kld_col] * 4,
                                            "dp": [dp_over] * 4},
                           token_metrics_b={"kld": [kld_col] * 4,
                                            "dp": [dp_under] * 4})
    assert res["candidate_metrics"]["mean_dp"]["candidate_a"][
        "item_weighted_mean"] > 0
    assert res["candidate_metrics"]["mean_dp"]["candidate_b"][
        "item_weighted_mean"] < 0
    dp_ladder = res["pooled_token_distribution"]["dp"]
    assert dp_ladder["degradation_tail"] == ["max", "p999", "median",
                                             "p001", "min"]
    med = {L["name"]: L["value"] for L in dp_ladder["candidate_a"]["levels"]}
    assert med["median"] > 0


def test_reference_nll_and_ppl_are_reported():
    """"baseline" and "candidate" in this report are candidates A and B, so
    the REFERENCE's own likelihood had nowhere to appear -- llama-perplexity's
    PPL(base). It is identical on both sides by construction (one shared
    reference), and the report says so."""
    rng = np.random.default_rng(22)
    npos = 12
    kld_a = np.abs(rng.normal(1e-3, 3e-4, npos)).astype(np.float32)
    kld_b = (kld_a * 1.5).astype(np.float32)
    nll_ref = 1.2345
    sa = {"kld": float(kld_a.mean()), "nll_ref": nll_ref}
    sb = {"kld": float(kld_b.mean()), "nll_ref": nll_ref}
    ta = {"kld": kld_a}
    tb = {"kld": kld_b}
    assert sa["nll_ref"] == pytest.approx(sb["nll_ref"], rel=1e-12)

    res = compare_items([sa] * 5, [sb] * 5, [npos] * 5, metrics=["kld"],
                           confidence_level=0.95, bootstrap_iters=1000, seed=1,
                           model_a_label="A", model_b_label="B",
                           token_metrics_a={k: [v] * 5 for k, v in ta.items()},
                           token_metrics_b={k: [v] * 5 for k, v in tb.items()})
    ref_m = res["reference_metrics"]
    assert ref_m["nll"]["item_weighted_mean"] == pytest.approx(sa["nll_ref"])
    assert ref_m["ppl"]["item_weighted"] == pytest.approx(
        math.exp(sa["nll_ref"]))
    assert ref_m["max_abs_side_difference"] == pytest.approx(0.0, abs=1e-12)
    # nll_ref must NOT be a paired metric: its delta is identically zero
    assert "nll_ref" not in res["metrics"]
    md = format_comparison_table(res, reference_label="F16")
    assert "## Reference corpus likelihood (F16)" in md
    assert "PPL(reference)" in md and "max_abs_side_difference" in md
    # it is NOT a row of the paired table -- there is nothing paired about it
    assert "| nll_ref " not in md


def test_pooled_ladder_covers_the_full_upstream_level_set():
    """llama-perplexity prints Maximum / 99.9 / 99 / 95 / 90 / Median / 10 /
    5 / 1 / 0.1 / Minimum. The tool used to report only the upper tail."""
    assert [n for n, _ in POOLED_LADDER] == [
        "max", "p999", "p99", "p95", "p90", "median",
        "p10", "p05", "p01", "p001", "min"]
    assert any(n == "median" for n, _ in POOLED_LADDER)


# --------------------------------------------------------------------------- #
# tail composition: WHICH items own the tail
# --------------------------------------------------------------------------- #
def test_critical_samples_name_the_items_that_own_the_tail():
    """A pooled quantile says how bad the tail is; it does not say where it
    came from, and in practice a handful of items own it."""
    quiet = np.full(50, 1e-4, np.float32)
    loud = np.concatenate([np.full(45, 1e-4), np.full(5, 9.0)]).astype(np.float32)
    a = [quiet, quiet, loud]                       # item 2 owns the whole tail
    b = [c.copy() for c in a]
    keys = ["000_calm_a", "001_calm_b", "002_loud"]
    block = _pooled_distribution_block(a, b, metric="kld",
                                          score_direction="lower_is_better",
                                          item_keys=keys)
    crit = block["critical_samples"]
    assert crit["side"] == "upper"
    p99 = crit["levels"]["p99"]["candidate_a"]
    assert p99[0]["item_key"] == "002_loud"
    assert p99[0]["tail_token_count"] == len(p99[0]["positions"]) > 0
    assert p99[0]["positions"] == list(range(45, 45 + p99[0]["tail_token_count"]))
    assert p99[0]["item_extreme"] == pytest.approx(9.0)
    # the calm items hold nothing at or above the threshold
    assert [s["item_key"] for s in p99] == ["002_loud"]


def test_critical_samples_carry_the_companion_ear_at_those_tokens():
    """A KLD tail is far more actionable next to how bad agreement got at
    exactly those tokens."""
    kld = [np.concatenate([np.full(40, 1e-4), np.full(10, 5.0)]).astype(np.float32)]
    ear = [np.concatenate([np.full(40, 0.999), np.full(10, 0.31)]).astype(np.float32)]
    block = _pooled_distribution_block(
        kld, [k.copy() for k in kld], metric="kld",
        score_direction="lower_is_better", item_keys=["000_x"],
        companion_a=ear, companion_b=[e.copy() for e in ear],
        companion_metric="ear")
    smp = block["critical_samples"]["levels"]["p99"]["candidate_a"][0]
    assert smp["companion_min"] == pytest.approx(0.31, rel=1e-6)
    assert len(smp["companion_values"]) == smp["tail_token_count"]


def test_ear_critical_samples_use_the_LOW_tail():
    """EAR is higher-is-better, so its tail is the bottom of the ladder."""
    ear = [np.concatenate([np.full(45, 0.999), np.full(5, 0.20)]).astype(np.float32)]
    block = _pooled_distribution_block(ear, [e.copy() for e in ear],
                                          metric="ear",
                                          score_direction="higher_is_better",
                                          item_keys=["000_x"])
    crit = block["critical_samples"]
    assert crit["side"] == "lower"
    assert set(crit["levels"]) == {"p01", "p001"}
    smp = crit["levels"]["p01"]["candidate_a"][0]
    assert smp["item_extreme"] == pytest.approx(0.20, rel=1e-6)
    assert all(p >= 45 for p in smp["positions"])


def test_signed_dp_gets_no_critical_samples():
    """dp is signed and two-sided, so it has no single degradation tail."""
    dp = [np.linspace(-3.0, 3.0, 40).astype(np.float32)]
    block = _pooled_distribution_block(dp, [d.copy() for d in dp], metric="dp",
                                          score_direction="lower_is_better")
    assert "critical_samples" not in block


def test_report_renders_the_tail_composition_table():
    res = _compare()
    md = format_comparison_table(res, reference_label="F16")
    assert "**kld tail composition**" in md
    assert "tail tokens" in md and "of item" in md
