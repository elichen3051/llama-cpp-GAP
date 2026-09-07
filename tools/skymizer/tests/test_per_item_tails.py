"""Per-item KLD tails: each item's own p99 / p99.9 / max of its per-token KLD
with the position (and target token) it occurred at, plus one exploratory
paired test per level in its own Holm family (result["per_item_tails"]).

Pinned here: the per-item values against numpy on the raw column, the witness
positions, the top-two degeneracy flag, the target annotation, the family
separation from the primary verdicts, the CI-method plumbing, the renderer,
and the two producers' token dicts carrying the `target` column."""

import json
import re

import numpy as np
import pytest

from stats.contracts import AlignmentError
from stats.engine import compare_items
from stats.inference import holm_adjust
from stats.render import format_comparison_table


def _items(n=12, seed=5, lengths=None):
    rng = np.random.default_rng(seed)
    lengths = lengths or [int(x) for x in rng.integers(20, 300, size=n)]
    keys = [f"{i:03d}_item{i}" for i in range(len(lengths))]
    tok_a, tok_b, tgt, sa, sb = {"kld": []}, {"kld": []}, [], [], []
    for T in lengths:
        ka = np.abs(rng.normal(1e-3, 5e-4, T)).astype(np.float32)
        kb = (ka * 1.3 + np.abs(rng.normal(0, 3e-4, T))).astype(np.float32)
        # plant one catastrophic token per item at a known position
        pos = int(rng.integers(0, T))
        kb[pos] = np.float32(0.5 + rng.random())
        tok_a["kld"].append(ka)
        tok_b["kld"].append(kb)
        tgt.append(rng.integers(0, 50000, size=T).astype(np.int32))
        sa.append({"kld": float(ka.mean(dtype=np.float64)), "nll": 2.0})
        sb.append({"kld": float(kb.mean(dtype=np.float64)), "nll": 2.1})
    return keys, sa, sb, lengths, tok_a, tok_b, tgt


def _run(with_target=True, ci_method="t", **kw):
    keys, sa, sb, w, ta, tb, tgt = _items(**kw)
    if with_target:
        ta = dict(ta, target=tgt)
        tb = dict(tb, target=tgt)
    res = compare_items(sa, sb, w, metrics=["kld", "nll"], confidence_level=0.95,
                           bootstrap_iters=500, seed=3, model_a_label="A",
                           model_b_label="B", ci_method=ci_method,
                           token_metrics_a=ta, token_metrics_b=tb, item_keys=keys)
    return res, (keys, sa, sb, w, ta, tb, tgt)


def test_per_item_values_match_numpy_and_witness_the_planted_token():
    res, (keys, sa, sb, w, ta, tb, tgt) = _run()
    block = res["per_item_tails"]["metrics"]["kld"]
    assert res["per_item_tails"]["levels"] == ["p99", "p999", "max"]
    assert block["n_items"] == len(keys)
    for it in block["items"]:
        i = it["item_index"]
        assert it["item_key"] == keys[i] and it["n_positions"] == w[i]
        for side, col in (("candidate_a", ta["kld"][i]), ("candidate_b", tb["kld"][i])):
            col64 = col.astype(np.float64)
            rec = it[side]
            assert rec["max"]["value"] == float(col64.max())
            assert rec["max"]["position"] == int(np.argmax(col64))
            assert rec["max"]["target_token"] == int(tgt[i][int(np.argmax(col64))])
            for name, q in (("p99", 0.99), ("p999", 0.999)):
                assert rec[name]["value"] == pytest.approx(
                    float(np.quantile(col64, q)), rel=1e-12)
                # witness is a real position holding a value that brackets the quantile
                lo, hi = rec[name]["lower_position"], rec[name]["upper_position"]
                assert sorted([col64[lo], col64[hi]])[0] <= rec[name]["value"] <= \
                    sorted([col64[lo], col64[hi]])[1]
                assert rec[name]["position"] in (lo, hi)
                assert rec[name]["target_token"] == int(tgt[i][rec[name]["position"]])
                assert rec[name]["rests_on_top_two"] == (q * (w[i] - 1) > w[i] - 2)
        # the planted catastrophic token IS candidate b's max
        assert it["candidate_b"]["max"]["value"] > 0.5
        assert it["delta_b_minus_a"]["max"] == pytest.approx(
            it["candidate_b"]["max"]["value"] - it["candidate_a"]["max"]["value"])


def test_top_two_count_is_a_function_of_length_only():
    res, (keys, sa, sb, w, *_rest) = _run(lengths=[5, 50, 101, 150, 1000, 1001, 2048])
    cells = {c["level"]: c for c in res["per_item_tails"]["metrics"]["kld"]["cells"]}
    # p99 interpolates above the top two only when 0.99 (L-1) <= L-2  <=> L >= 101
    assert cells["p99"]["n_items_resting_on_top_two"] == 2       # L = 5, 50
    assert cells["p999"]["n_items_resting_on_top_two"] == 5      # L < 1001: 5, 50, 101, 150, 1000
    assert "n_items_resting_on_top_two" not in cells["max"]


def test_cells_are_their_own_exploratory_holm_family():
    res, _ = _run()
    main_family = res["multiplicity"]["family_size"]
    cells = res["per_item_tails"]["metrics"]["kld"]["cells"]
    assert [c["level"] for c in cells] == ["p99", "p999", "max"]
    for c in cells:
        assert c["role"] == "exploratory"
        assert c["ci_delta"]["ci_method"] == "t"
        assert "p_value_holm" in c and isinstance(c["holm_significant"], bool)
        assert c["p_value_holm"] >= c["p_value"]
        assert c["n_items_used"] == 12
    assert holm_adjust([c["p_value"] for c in cells]) == pytest.approx(
        [c["p_value_holm"] for c in cells])
    # the main family did not grow: 2 metrics x 2 weightings - 1 primary
    assert main_family == 3
    assert res["per_item_tails"]["role"] == "exploratory"
    assert "holm" in res["per_item_tails"]["correction"]


def test_ci_method_and_seed_plumb_through():
    res_t, _ = _run(ci_method="t")
    res_s, _ = _run(ci_method="studentized")
    for res, m in ((res_t, "t"), (res_s, "studentized")):
        for c in res["per_item_tails"]["metrics"]["kld"]["cells"]:
            assert c["ci_delta"]["ci_method"] in {m, "percentile"}   # (fallback allowed)
    # the planted b-side catastrophe makes b worse at the max: A closer expected
    cmax = next(c for c in res_t["per_item_tails"]["metrics"]["kld"]["cells"]
                if c["level"] == "max")
    assert cmax["decision"]["verdict"] == "A closer"


def test_without_target_column_there_is_no_token_annotation():
    res, _ = _run(with_target=False)
    it = res["per_item_tails"]["metrics"]["kld"]["items"][0]
    assert "target_token" not in it["candidate_a"]["max"]
    assert "position" in it["candidate_a"]["max"]


def test_absent_without_token_data_and_rejected_below_two_items():
    scores_a = [{"kld": 0.1}, {"kld": 0.2}]
    scores_b = [{"kld": 0.15}, {"kld": 0.25}]
    res = compare_items(scores_a, scores_b, [10, 10], metrics=["kld"],
                           confidence_level=0.95, bootstrap_iters=500, seed=1,
                           model_a_label="A", model_b_label="B")
    assert "per_item_tails" not in res
    with pytest.raises(AlignmentError, match="at least two"):
        compare_items(scores_a[:1], scores_b[:1], [4], metrics=["kld"],
                      confidence_level=0.95, bootstrap_iters=500, seed=1,
                      model_a_label="A", model_b_label="B", position_buckets=None,
                      token_metrics_a={"kld": [np.ones(4, np.float32)]},
                      token_metrics_b={"kld": [np.ones(4, np.float32) * 2]},
                      item_keys=["k"])


def test_markdown_section_and_worst_items_table():
    res, (keys, *_rest) = _run()
    md = format_comparison_table(res, reference_label="F16")
    assert "## Per-item KLD tails (exploratory — own Holm family)" in md
    assert re.search(r"\| kld\s+\| p99\.9\s+\|", md) and re.search(r"\| kld\s+\| max\s+\|", md)
    assert "Worst 10 of 12 items by per-item max `kld`" in md
    # witness cells carry value @position tok id
    worst = md.split("Worst 10 of 12 items")[1]
    assert " tok " in worst and "@" in worst
    # the section sits between the strata and the pooled ladders
    assert md.index("## Per-item KLD tails") < md.index("## Per-token distributions")
    # round-trips through JSON (no numpy scalars leaked)
    json.dumps(res)


def test_engine_validates_the_target_annotation_column():
    """The `target` annotation column rides beside the metric columns (the
    saved-metrics producer attaches it; see
    test_saved_metrics_paired_compare.py) and the engine validates its length
    like a metric column."""
    rng = np.random.default_rng(9)
    tgt = rng.integers(0, 21, size=7).astype(np.int32)
    kld_a = np.abs(rng.normal(1e-3, 3e-4, 7)).astype(np.float32)
    kld_b = (kld_a * 2).astype(np.float32)
    sa = {"kld": float(kld_a.mean())}
    sb = {"kld": float(kld_b.mean())}
    tok_a = {"kld": [kld_a, kld_a], "target": [tgt, tgt]}
    tok_b = {"kld": [kld_b, kld_b], "target": [tgt, tgt]}
    # a well-formed annotation column is accepted
    compare_items([sa, sa], [sb, sb], [7, 7], metrics=["kld"], confidence_level=0.95,
                     bootstrap_iters=500, seed=1, model_a_label="A",
                     model_b_label="B", position_buckets=None,
                     token_metrics_a=tok_a, token_metrics_b=tok_b,
                     item_keys=["k0", "k1"])
    # a short one is rejected exactly like a short metric column
    with pytest.raises(AlignmentError):
        compare_items([sa, sa], [sb, sb], [7, 7], metrics=["kld"], confidence_level=0.95,
                         bootstrap_iters=500, seed=1, model_a_label="A",
                         model_b_label="B", position_buckets=None,
                         token_metrics_a=dict(tok_a, target=[tgt[:3], tgt]),
                         token_metrics_b=tok_b, item_keys=["k0", "k1"])
