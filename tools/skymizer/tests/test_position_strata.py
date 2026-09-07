"""Answer-position stratification.

The repo's own measurements put the mmproj F16-vs-Q8_0 signal in the FIRST
~32 answer tokens, decaying more than 10x after, while the default eval cap
is 1024 — so a single item-mean over all positions dilutes a vision effect by
roughly 30x. There was no position-stratified report anywhere; the only lever
was a prefix cap, which throws the rest of the data away instead of
stratifying it.
"""

import numpy as np
import pytest

from stats.engine import compare_items
from stats.inference import holm_adjust
from stats.render import format_comparison_table
from stats.tokens import DEFAULT_POSITION_BUCKETS, _bucket_ranges


def _front_loaded(n_items=20, npos=400, front=32, effect=3e-4, seed=4):
    """Per-token fixtures whose A-vs-B difference lives ONLY in the first
    `front` positions — the shape the notes measured."""
    rng = np.random.default_rng(seed)
    scores_a, scores_b, weights, keys = [], [], [], []
    tok_a = {"kld": [], "ear": []}
    tok_b = {"kld": [], "ear": []}
    for i in range(n_items):
        ka = np.abs(rng.normal(1e-3, 3e-4, npos)).astype(np.float32)
        bump = np.zeros(npos, np.float32)
        bump[:front] = effect
        kb = (ka + bump).astype(np.float32)
        ea = (1.0 - np.abs(rng.normal(0.01, 0.003, npos))).astype(np.float32)
        eb = (ea - 2.0 * bump).astype(np.float32)
        tok_a["kld"].append(ka); tok_b["kld"].append(kb)
        tok_a["ear"].append(ea); tok_b["ear"].append(eb)
        scores_a.append({"kld": float(ka.mean()), "ear": float(ea.mean())})
        scores_b.append({"kld": float(kb.mean()), "ear": float(eb.mean())})
        weights.append(npos)
        keys.append(f"{i:03d}_item{i}")
    return keys, scores_a, scores_b, weights, tok_a, tok_b


def _compare(**kw):
    keys, sa, sb, w, ta, tb = _front_loaded(**{k: v for k, v in kw.items()
                                               if k in ("n_items", "npos",
                                                        "front", "effect")})
    rest = {k: v for k, v in kw.items()
            if k not in ("n_items", "npos", "front", "effect")}
    return compare_items(sa, sb, w, metrics=["kld", "ear"],
                            confidence_level=0.95, bootstrap_iters=1000,
                            seed=7, model_a_label="A", model_b_label="B",
                            token_metrics_a=ta, token_metrics_b=tb,
                            item_keys=keys, **rest)


# --------------------------------------------------------------------------- #
# bucket edges
# --------------------------------------------------------------------------- #
def test_bucket_ranges_last_bucket_is_open():
    assert _bucket_ranges([0, 32, 256]) == [
        (0, 32, "0-32"), (32, 256, "32-256"), (256, None, "256+")]
    assert _bucket_ranges([0]) == [(0, None, "0+")]


@pytest.mark.parametrize("bad", [[], [1, 32], [0, 32, 32], [0, 256, 32]])
def test_bucket_ranges_rejects_bad_edges(bad):
    with pytest.raises(ValueError, match="edges"):
        _bucket_ranges(bad)


def test_default_buckets_match_the_measured_signal_window():
    assert DEFAULT_POSITION_BUCKETS == (0, 32, 256)


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def test_strata_localize_a_front_loaded_effect_the_overall_mean_dilutes():
    """The whole point: averaged over 400 positions a 32-position effect is
    diluted ~12x, but the 0-32 stratum shows it at full size."""
    res = _compare()
    strata = res["position_strata"]["metrics"]["kld"]
    by_bucket = {c["bucket"]: c for c in strata}
    front = by_bucket["0-32"]["delta_candidate_minus_baseline"]
    overall = res["metrics"]["kld"]["item_weighted"]["delta_candidate_minus_baseline"]
    assert front == pytest.approx(3e-4, rel=1e-3)
    assert overall < front / 5                       # diluted by the tail
    for later in ("32-256", "256+"):
        assert by_bucket[later]["delta_candidate_minus_baseline"] == \
            pytest.approx(0.0, abs=1e-9)
        assert by_bucket[later]["decision"]["verdict"] == "inconclusive"
    assert by_bucket["0-32"]["decision"]["verdict"] in ("A closer", "B closer")


def test_strata_cover_every_retained_per_token_metric():
    res = _compare()
    assert set(res["position_strata"]["metrics"]) == {"kld", "ear"}
    # EAR is higher-is-better; its front stratum must read as a degradation
    front = [c for c in res["position_strata"]["metrics"]["ear"]
             if c["bucket"] == "0-32"][0]
    assert front["delta_candidate_minus_baseline"] < 0
    assert front["decision"]["verdict"] == "A closer"


def test_items_too_short_for_a_bucket_are_dropped_from_it_not_zero_filled():
    """An item that never reaches a bucket must not contribute a 0 to it —
    that would drag the stratum's mean towards zero and inflate n."""
    keys, sa, sb, w, ta, tb = _front_loaded(n_items=6, npos=400)
    # make two items short: they reach 0-32 only
    for i in (0, 1):
        for tok in (ta, tb):
            for m in tok:
                tok[m][i] = tok[m][i][:20]
        w[i] = 20
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=1000, seed=1, model_a_label="A",
                           model_b_label="B", token_metrics_a=ta,
                           token_metrics_b=tb, item_keys=keys)
    by_bucket = {c["bucket"]: c for c in res["position_strata"]["metrics"]["kld"]}
    assert by_bucket["0-32"]["n_items_used"] == 6
    assert by_bucket["32-256"]["n_items_used"] == 4
    assert by_bucket["32-256"]["n_items_without_positions"] == 2


def test_bucket_with_fewer_than_two_items_is_skipped_with_a_reason():
    """A one-item paired bootstrap CI is zero-width and would read as
    significant for every metric."""
    keys, sa, sb, w, ta, tb = _front_loaded(n_items=4, npos=40)
    res = compare_items(sa, sb, w, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=1000, seed=1, model_a_label="A",
                           model_b_label="B", token_metrics_a=ta,
                           token_metrics_b=tb, item_keys=keys)
    tail = [c for c in res["position_strata"]["metrics"]["kld"]
            if c["bucket"] == "256+"][0]
    assert "p_value" not in tail
    assert "only 0 item(s)" in tail["skipped"]


# --------------------------------------------------------------------------- #
# multiplicity + rendering
# --------------------------------------------------------------------------- #
def test_strata_are_exploratory_with_their_own_holm_family():
    """The strata were chosen from prior measurements on this data, so they
    must not borrow the confirmatory endpoint's alpha."""
    res = _compare()
    cells = [c for cells in res["position_strata"]["metrics"].values()
             for c in cells if "p_value" in c]
    assert cells and all(c["role"] == "exploratory" for c in cells)
    want = holm_adjust([c["p_value"] for c in cells])
    assert [c["p_value_holm"] for c in cells] == pytest.approx(want)
    # ... and the primary endpoint's own family is untouched by them
    assert res["multiplicity"]["family_size"] == 3   # 2 metrics x 2 weightings - primary


def test_position_buckets_can_be_collapsed_to_one():
    res = _compare(position_buckets=[0])
    for cells in res["position_strata"]["metrics"].values():
        assert [c["bucket"] for c in cells] == ["0+"]


def test_report_renders_the_strata_section_as_exploratory():
    md = format_comparison_table(_compare(), reference_label="F16")
    assert "## Answer-position strata (exploratory)" in md
    assert "first ~32 answer tokens" in md
    assert "| kld    | 0-32" in md
    assert "p (Holm)" in md
