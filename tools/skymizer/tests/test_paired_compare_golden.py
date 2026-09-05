"""Legacy golden for the percentile bootstrap path.

The fixture and expected JSON files in tests/data/ were copied verbatim from
the producer repo (llm_quant_fidelity_comparison). That repo is no longer
kept in parity, so this test's role is now REPO-LOCAL: a regression pin on
the shared first-order (percentile) engine, checked to 1e-12 on estimate/CI.

The engine's production default is `t` (DEFAULT_CI_METHOD in
paired_compare.py; the percentile interval under-covers on the right-skewed
per-item deltas this tool produces). This test passes ci_method="percentile"
DELIBERATELY — the fixtures pin that path; do not regenerate them with
another method. Full-surface, bit-exact coverage of all four CI methods
lives in test_paired_compare_engine_golden.py.
"""

import json
from pathlib import Path

import pytest

from compare.engine import compare_items


DATA_DIR = Path(__file__).resolve().parent / "data"


def test_paired_compare_matches_producer_golden_fixture():
    fixture = json.loads((DATA_DIR / "paired_compare_golden_fixture.json").read_text())
    expected = json.loads((DATA_DIR / "paired_compare_golden_expected.json").read_text())
    scores_a = [record["metrics"] for record in fixture["baseline_records"]]
    scores_b = [record["metrics"] for record in fixture["candidate_records"]]
    weights = [record["lengths"]["num_eval_tokens"] for record in fixture["baseline_records"]]
    bootstrap = fixture["bootstrap"]
    labels = fixture["model_labels"]

    result = compare_items(
        scores_a,
        scores_b,
        weights,
        metrics=fixture["metrics"],
        confidence_level=bootstrap["confidence_level"],
        bootstrap_iters=bootstrap["bootstrap_iters"],
        seed=bootstrap["seed"],
        model_a_label=labels["baseline"],
        model_b_label=labels["candidate"],
        ci_method="percentile",
    )

    assert result["n_items"] == expected["n_items"]
    for metric, metric_expected in expected["metrics"].items():
        got = result["metrics"][metric]
        assert got["weighting_consensus"] == metric_expected["weighting_consensus"]
        for weighting in ("item_weighted", "token_weighted"):
            block = got[weighting]
            want = metric_expected[weighting]
            assert block["delta_candidate_minus_baseline"] == pytest.approx(
                want["estimate"], abs=1e-12
            )
            assert block["ci_delta"]["lower"] == pytest.approx(want["ci"]["lower"], abs=1e-12)
            assert block["ci_delta"]["upper"] == pytest.approx(want["ci"]["upper"], abs=1e-12)
