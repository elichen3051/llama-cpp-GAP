"""Multiplicity policy: one confirmatory endpoint + Holm over the rest.

A default report emits every base metric item- AND token-weighted, i.e. 14
verdicts at nominal alpha = 0.05. Driven with exchangeable A/B data
(identical distributions, a shared per-item latent, n=50) the uncorrected
grid's family-wise false-positive rate measured 0.35. So exactly one cell is
confirmatory and spends the whole alpha; the rest are exploratory and carry
Holm-Bonferroni-adjusted p-values.
"""

import numpy as np
import pytest

from stats.contracts import DEFAULT_METRICS, DEFAULT_PRIMARY_WEIGHTING
from stats.engine import compare_items
from stats.inference import holm_adjust
from stats.render import format_comparison_table


def _fixture(n=24, seed=3, effect=0.0):
    rng = np.random.default_rng(seed)
    scores_a, scores_b, weights = [], [], []
    for i in range(n):
        latent = rng.normal(0.0, 0.3)
        base = {m: 0.5 + latent + rng.normal(0, 0.1)
                for m in DEFAULT_METRICS}
        cand = {m: base[m] + effect + rng.normal(0, 0.1)
                for m in DEFAULT_METRICS}
        scores_a.append(base)
        scores_b.append(cand)
        weights.append(10 + i)
    return scores_a, scores_b, weights


def _compare(**kw):
    a, b, w = _fixture(**{k: v for k, v in kw.items()
                          if k in ("n", "seed", "effect")})
    rest = {k: v for k, v in kw.items() if k not in ("n", "seed", "effect")}
    return compare_items(a, b, w, metrics=list(DEFAULT_METRICS),
                            confidence_level=0.95, bootstrap_iters=1000,
                            seed=1, model_a_label="A", model_b_label="B",
                            **rest)


# --------------------------------------------------------------------------- #
# holm_adjust
# --------------------------------------------------------------------------- #
def test_holm_adjust_matches_the_step_down_definition():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    # (m-j) * p_(j) with a running max: 5*.001, 4*.008, 3*.039, 2*.041, 1*.9
    assert holm_adjust(p) == pytest.approx(
        [0.005, 0.032, 0.117, 0.117, 0.9])


def test_holm_adjust_is_monotone_and_clipped():
    p = [0.5, 0.02, 0.9, 0.021]
    adj = holm_adjust(p)
    assert all(0.0 <= v <= 1.0 for v in adj)
    # order is preserved and never inverted by the running max
    assert sorted(range(4), key=lambda i: p[i]) == sorted(
        range(4), key=lambda i: (adj[i], p[i]))
    assert holm_adjust([0.4, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_holm_adjust_never_shrinks_a_p_value():
    rng = np.random.default_rng(1)
    p = list(rng.uniform(0, 1, 20))
    assert all(a >= b - 1e-15 for a, b in zip(holm_adjust(p), p))


def test_holm_adjust_empty():
    assert holm_adjust([]) == []


# --------------------------------------------------------------------------- #
# policy wiring
# --------------------------------------------------------------------------- #
def test_exactly_one_primary_and_only_item_endpoints_exploratory():
    res = _compare()
    mult = res["multiplicity"]
    assert mult["primary_endpoint"]["metric"] == "kld"
    assert mult["primary_endpoint"]["weighting"] == "item"
    assert mult["correction"] == "holm_bonferroni"

    roles = {}
    for name, mres in res["metrics"].items():
        if "source_metric" in mres:
            continue
        for key in ("item_weighted", "token_weighted"):
            block = mres.get(key)
            if isinstance(block, dict) and block.get("p_value") is not None:
                roles[(name, key)] = block["role"]
    assert list(roles.values()).count("primary") == 1
    assert roles[("kld", "item_weighted")] == "primary"
    assert mult["family_size"] == len(roles) - 1
    assert mult["family_size"] == len(DEFAULT_METRICS) - 1
    for mres in res["metrics"].values():
        token = mres["token_weighted"]
        assert token["role"] == "descriptive"
        assert not ({"ci", "ci_delta", "p_value", "p_value_holm", "holm_significant", "decision", "bootstrap_std"} & token.keys())
        assert "weighting_consensus" not in mres
    # the primary is NOT adjusted; every exploratory cell is
    assert "p_value_holm" not in res["metrics"]["kld"]["item_weighted"]
    for (name, key), role in roles.items():
        if role == "exploratory":
            block = res["metrics"][name][key]
            assert block["p_value_holm"] >= block["p_value"] - 1e-15
            assert isinstance(block["holm_significant"], bool)


def test_holm_values_equal_holm_adjust_of_the_family():
    res = _compare(effect=0.05)
    family = [(n, k) for n, m in res["metrics"].items()
              if "source_metric" not in m
              for k in ("item_weighted", "token_weighted")
              if isinstance(m.get(k), dict)
              and m[k].get("role") == "exploratory"]
    raw = [res["metrics"][n][k]["p_value"] for n, k in family]
    want = holm_adjust(raw)
    got = [res["metrics"][n][k]["p_value_holm"] for n, k in family]
    assert got == pytest.approx(want)


def test_primary_endpoint_is_selectable():
    """The primary metric is selectable; its weighting must remain item."""
    assert DEFAULT_PRIMARY_WEIGHTING == "item"
    res = _compare(primary_metric="ear", primary_weighting="item")
    assert res["multiplicity"]["primary_endpoint"]["metric"] == "ear"
    assert res["multiplicity"]["primary_endpoint"]["weighting"] == "item"
    assert res["metrics"]["ear"]["item_weighted"]["role"] == "primary"
    assert res["metrics"]["kld"]["token_weighted"]["role"] == "descriptive"
    with pytest.raises(ValueError, match="primary_metric"):
        _compare(primary_metric="nonesuch")
    with pytest.raises(ValueError, match="primary_weighting"):
        _compare(primary_weighting="sideways")
    with pytest.raises(ValueError, match="primary_weighting"):
        _compare(primary_weighting="token")


@pytest.mark.statistical
def test_holm_shrinks_the_family_wise_error_rate_on_exchangeable_data():
    """The point of the whole exercise. Under a shared-latent exchangeable
    null the uncorrected grid fires far more often than alpha; requiring the
    Holm-adjusted p keeps it at or below it."""
    raw_any = holm_any = 0
    # 400 simulations: the bound below is a hard 0.05 on a binomial estimate
    # (SD 0.011 here; 0.02 at the 120 this test used to run), and the default
    # t interval makes each simulated report ~instant. Measured at 400 sims:
    # uncorrected-any 0.415, Holm-any 0.048 (t); 0.425 / 0.035 (studentized).
    sims = 400
    for s in range(sims):
        res = _compare(n=40, seed=100 + s, effect=0.0)
        raw, holm = False, False
        for name, mres in res["metrics"].items():
            if "source_metric" in mres:
                continue
            for key in ("item_weighted", "token_weighted"):
                block = mres.get(key)
                if not isinstance(block, dict) or block.get("p_value") is None:
                    continue
                raw |= block["decision"]["statistically_distinguishable_from_null"]
                if block.get("role") == "exploratory":
                    holm |= block["holm_significant"]
        raw_any += raw
        holm_any += holm
    assert holm_any / sims <= 0.05 + 1e-9
    assert holm_any <= raw_any


def test_verdict_summary_honours_the_weighting_flag():
    """--weighting item|token filters the results table; the summary above it
    hard-coded both rows, so `--weighting item` advertised a token verdict the
    table below then omitted."""
    res = _compare(effect=0.08)
    seen = {}
    for w in ("both", "item", "token"):
        md = format_comparison_table(res, reference_label="F16",
                                        display_weighting=w)
        if "## Verdict summary" not in md:
            seen[w] = []
            continue
        section = md.split("## Verdict summary")[1].split("## Results")[0]
        seen[w] = sorted({row.split("|")[1].strip().split(" ")[-1]
                          for row in section.splitlines()
                          if row.startswith("| ")
                          and not row.startswith("| ---")} - {"endpoint"})
    assert seen["item"] == ["(item)"]
    assert seen["token"] == []
    assert seen["both"] == ["(item)"]


def test_token_rows_never_render_test_results_including_derived_metrics():
    res = _compare(effect=0.08)
    md = format_comparison_table(res, reference_label="F16", display_weighting="token",
                                 show_diagnostic_metrics=True)
    rows = [[cell.strip() for cell in row.split("|")[1:-1]]
            for row in md.splitlines() if row.startswith("| ")]
    token_rows = [row for row in rows if len(row) == 8 and row[1] == "token"]
    assert len(token_rows) >= len(DEFAULT_METRICS) + 3
    for row in token_rows:
        assert row[5:8] == ["-", "-", "descriptive only"]
    assert "## Verdict summary" not in md


def test_report_marks_the_primary_and_shows_the_holm_column():
    res = _compare(effect=0.08)
    md = format_comparison_table(res, reference_label="F16")
    assert "p (Holm)" in md
    assert "★ = the one confirmatory endpoint" in md
    star_rows = [r for r in md.splitlines() if r.startswith("| ★ ")]
    assert len(star_rows) == 1
    assert "kld" in star_rows[0] and "(primary)" in star_rows[0]
    assert "| item" in star_rows[0] and "| token" not in star_rows[0]


@pytest.mark.parametrize("bad_p", [float("nan"), float("inf"), -0.01, 1.01])
def test_holm_rejects_invalid_p_values(bad_p):
    with pytest.raises(ValueError, match="finite and in"):
        holm_adjust([0.01, bad_p])


def test_unavailable_cells_retain_family_and_censored_bounds():
    from stats.engine import _adjust_exploratory_cells
    from stats.render import _format_p_value
    cells = [{"p_value": .01, "p_value_metadata": {"censored": True}}, {"skipped": "unsupported"}, {"p_value": .025}]
    _adjust_exploratory_cells(cells, .95)
    assert [c["holm_family_size"] for c in cells] == [3, 3, 3]
    assert cells[0]["p_value_holm"] == pytest.approx(.03)
    assert "p_value_holm" not in cells[1]
    assert _format_p_value(cells[2], adjusted=True).startswith("<= ")
    boundary = [{"p_value": 1 - .95}]
    _adjust_exploratory_cells(boundary, .95)
    assert not boundary[0]["holm_significant"]


@pytest.mark.parametrize("method, expected", [
    ("holm", [.04, .09, .09, .2]),
    ("fdr_bh", [.04, .16/3, .16/3, .2]),
    ("fdr_by", [1/12, 1/9, 1/9, 5/12]),
])
def test_package_adjustments_against_hand_calculated_family(method, expected):
    from stats.multiplicity import adjust_pvalues
    p = [.01, .04, .03, .2]
    assert adjust_pvalues(p, method) == pytest.approx(expected)
    order = [3, 1, 0, 2]
    assert adjust_pvalues([p[i] for i in order], method) == pytest.approx([expected[i] for i in order])


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
def test_fdr_package_matches_independent_scipy_api(method):
    from scipy.stats import false_discovery_control
    from stats.multiplicity import adjust_pvalues
    p = [0, .001, .01, .03, .05, .05, .5, 1]
    assert adjust_pvalues(p, method) == pytest.approx(false_discovery_control(p, method=method[4:]), abs=1e-15)


def test_adjustment_rejects_matrix_and_unknown_method():
    from stats.multiplicity import adjust_pvalues
    with pytest.raises(ValueError, match="1-D"):
        adjust_pvalues([[.01, .02]])
    with pytest.raises(ValueError, match="method"):
        adjust_pvalues([.01], "auto")
