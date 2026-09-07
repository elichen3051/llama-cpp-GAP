"""Tests for the compare package — the paired statistics engine and its
markdown/JSON rendering (decision table, bootstrap streams, weighting blocks,
compare_items, execution metadata, report text).

Hermetic: small per-item score dicts; no GPU, no models.
Run from tools/skymizer:  python3 -m pytest tests/test_paired_compare.py -v
"""

import math

import numpy as np

import pytest

from stats.cli.common import resolve_item_end
from stats.contracts import (
    AlignmentError,
    DEFAULT_METRICS,
    LOWER_IS_BETTER,
    MissingMetricError,
    NonFiniteMetricError,
    POOLED_LADDER,
    POOLED_TOKEN_METRICS,
    SCHEMA_VERSION,
)
from stats.engine import compare_items
from stats.inference import (
    _build_weighting_block,
    _classify_consensus,
    _compute_decision,
    _paired_bootstrap_delta,
)
from stats.render import (
    _format_execution_section,
    _sorted_prefix_warning,
    build_execution_metadata,
    build_inputs_metadata,
    format_comparison_table,
)
import stats.render
import stats.cli.saved_metrics_paired_compare as smpc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _fake_scores(n, base, delta, metrics):
    """n items; side a = base, side b = base + delta, for each metric."""
    a = [{m: base[m] for m in metrics} for _ in range(n)]
    b = [{m: base[m] + delta[m] for m in metrics} for _ in range(n)]
    weights = [10.0] * n
    return a, b, weights


def _result_for_render(*, with_inputs: bool = True):
    metrics = ("nll", "kld", "reversed_kld", "js_kld", "same_top_rate", "mse_dp")
    base = {"nll": 1.0, "kld": 0.5, "reversed_kld": 0.6, "js_kld": 0.1,
            "same_top_rate": 0.9, "mse_dp": 4.0}
    delta = {m: 0.0 for m in metrics}
    delta["nll"] = 0.2
    a = [{m: base[m] for m in metrics} for _ in range(6)]
    b = [{m: base[m] + delta[m] for m in metrics} for _ in range(6)]
    result = compare_items(a, b, [10.0] * 6, metrics=metrics, confidence_level=0.95,
                              bootstrap_iters=500, seed=1234,
                              model_a_label="Q4_K_M", model_b_label="UNSLOTH")
    if with_inputs:
        result["inputs"] = {
            "reference":   {"label": "F16",    "model_path": "/m/qwen3vl-f16.gguf"},
            "candidate_a": {"label": "Q4_K_M", "model_path": "/m/qwen3vl-q4km.gguf"},
            "candidate_b": {"label": "UNSLOTH", "model_path": "/m/qwen3vl-unsloth.gguf"},
            "kind":    "vlm_logits",
            "logits_dtype": "fp16",
            "dataset": "lmms-lab/MMMU",
            "subset":  "default",
            "split":   "validation",
            "sort_by": "row",
        }
    return result


# --------------------------------------------------------------------------- #
# Task 1: scaffold
# --------------------------------------------------------------------------- #
def test_constants_and_imports():
    assert SCHEMA_VERSION == "vlm-paired-compare-v3"
    assert DEFAULT_METRICS == (
        "nll", "kld", "reversed_kld", "js_kld", "ear", "ear_20", "ear_10", "ear_5",
        "ear_20_normalized", "ear_10_normalized", "ear_5_normalized",
        "ear_64", "ear_64_normalized",
        "same_top_rate", "mse_dp",
    )
    assert POOLED_TOKEN_METRICS == ("kld", "ear", "dp")
    assert [n for n, _ in POOLED_LADDER] == [
        "max", "p999", "p99", "p95", "p90", "median",
        "p10", "p05", "p01", "p001", "min"]
    assert LOWER_IS_BETTER == {
        "nll", "kld", "reversed_kld", "js_kld", "mse_dp", "rms_dp", "ppl_ratio",
        "ppl",
    }


# --------------------------------------------------------------------------- #
# Task 2: per-item metric math
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Task 3: verdict truth table
# --------------------------------------------------------------------------- #
def test_decision_contains_zero_is_inconclusive_not_equivalent():
    """"no sig. diff." set beside "A closer"/"B closer" read as a third
    finding -- "these two are the same" -- which it never was. The verdict
    says "inconclusive" and carries the tightest margin the interval
    actually rules out."""
    d = _compute_decision(delta_estimate=0.01, ci_lower=-0.02, ci_upper=0.05,
                             null_value=0.0, score_direction="lower_is_better")
    assert d["verdict"] == "inconclusive"
    assert d["statistically_distinguishable_from_null"] is False
    assert d["equivalence_bound"] == pytest.approx(0.05)
    assert d["equivalence_margin"] is None
    assert d["equivalence_established"] is None
    assert "bound |Δ| ≤ 0.05" in d["reason"]


def test_decision_reports_equivalence_when_the_ci_clears_the_margin():
    """Interval inclusion IS a TOST: a CI inside (-m, m) establishes
    equivalence at margin m."""
    inside = _compute_decision(delta_estimate=0.01, ci_lower=-0.02,
                                  ci_upper=0.05, null_value=0.0,
                                  score_direction="lower_is_better",
                                  equivalence_margin=0.1)
    assert inside["verdict"] == "EQUIVALENT"
    assert inside["equivalence_established"] is True
    assert inside["equivalence_alpha"] == pytest.approx(0.025)
    assert "TOST" in inside["reason"]
    outside = _compute_decision(delta_estimate=0.01, ci_lower=-0.02,
                                   ci_upper=0.05, null_value=0.0,
                                   score_direction="lower_is_better",
                                   equivalence_margin=0.04)
    assert outside["verdict"] == "inconclusive"
    assert outside["equivalence_margin"] == 0.04
    assert outside["equivalence_established"] is False


def test_equivalence_margin_applies_only_to_the_primary_metric():
    """The margin is in the primary metric's own units (nats for kld, pp^2
    for mse_dp), so it must not leak onto the other metrics."""
    a = [{"kld": 0.10, "nll": 1.0 + 0.001 * i} for i in range(12)]
    b = [{"kld": 0.10 + 0.0001 * ((-1) ** i), "nll": 1.0 + 0.001 * i}
         for i in range(12)]
    res = compare_items(a, b, [10] * 12, metrics=["kld", "nll"],
                           confidence_level=0.95, bootstrap_iters=1000,
                           seed=1, model_a_label="A", model_b_label="B",
                           equivalence_margin=1.0)
    assert res["metrics"]["kld"]["item_weighted"]["decision"]["verdict"] == "EQUIVALENT"
    nll_decision = res["metrics"]["nll"]["item_weighted"]["decision"]
    assert nll_decision.get("equivalence_margin") is None


def test_decision_lower_is_better_b_smaller_is_b_better():
    d = _compute_decision(delta_estimate=-0.1, ci_lower=-0.2, ci_upper=-0.01,
                             null_value=0.0, score_direction="lower_is_better")
    assert d["verdict"] == "B closer"


def test_decision_higher_is_better_b_smaller_is_a_better():
    d = _compute_decision(delta_estimate=-0.1, ci_lower=-0.2, ci_upper=-0.01,
                             null_value=0.0, score_direction="higher_is_better")
    assert d["verdict"] == "A closer"


# --------------------------------------------------------------------------- #
# Task 4: paired bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_item_weighted_constant_diff():
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    b = a + 0.2
    w = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    r = _paired_bootstrap_delta(a, b, w, weighting="item",
                                   confidence_level=0.95, bootstrap_iters=2000, seed=7)
    assert r["estimate"] == pytest.approx(0.2, abs=1e-9)
    assert not r["ci"]["contains_zero"]


def test_bootstrap_token_weighted_matches_hand_calc():
    a = np.array([0.0, 0.0], dtype=float)
    b = np.array([0.10, 0.01], dtype=float)
    w = np.array([20.0, 400.0], dtype=float)
    r = _paired_bootstrap_delta(a, b, w, weighting="token",
                                   confidence_level=0.95, bootstrap_iters=10, seed=1)
    assert r["estimate"] == pytest.approx(6.0 / 420.0, rel=1e-9)


def test_bootstrap_shared_seed_same_indices():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    b = np.array([1.1, 2.2, 2.9, 4.3, 4.8], dtype=float)
    w = np.ones_like(a)
    ri = _paired_bootstrap_delta(a, b, w, weighting="item",
                                    confidence_level=0.9, bootstrap_iters=500, seed=99)
    rt = _paired_bootstrap_delta(a, b, w, weighting="token",
                                    confidence_level=0.9, bootstrap_iters=500, seed=99)
    assert ri["ci"]["lower"] == pytest.approx(rt["ci"]["lower"])
    assert ri["ci"]["upper"] == pytest.approx(rt["ci"]["upper"])


# --------------------------------------------------------------------------- #
# Task 5: weighting block + consensus
# --------------------------------------------------------------------------- #
def test_weighting_block_means_and_decision():
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    b = a + 0.2
    w = np.ones_like(a)
    blk = _build_weighting_block(a, b, w, weighting="item",
                                    score_direction="lower_is_better",
                                    confidence_level=0.95, bootstrap_iters=1000, seed=3)
    assert blk["baseline_mean"] == pytest.approx(1.0)
    assert blk["candidate_mean"] == pytest.approx(1.2)
    assert blk["delta_candidate_minus_baseline"] == pytest.approx(0.2)
    assert blk["decision"]["verdict"] == "A closer"   # b larger, lower is better


def test_classify_consensus():
    assert _classify_consensus({"verdict": "B closer"}, {"verdict": "B closer"}) == "agree"
    assert _classify_consensus({"verdict": "A closer"}, {"verdict": "B closer"}) == "direction_reversal"
    assert _classify_consensus({"verdict": "B closer"}, {"verdict": "no sig. diff."}) == "disagree"


# --------------------------------------------------------------------------- #
# Task 6: top-level compare + derived metrics
# --------------------------------------------------------------------------- #
def test_compare_items_structure_and_derived():
    metrics = ("nll", "kld", "reversed_kld", "js_kld", "same_top_rate", "mse_dp")
    base = {"nll": 1.0, "kld": 0.5, "reversed_kld": 0.6, "js_kld": 0.1,
            "same_top_rate": 0.9, "mse_dp": 4.0}
    delta = {"nll": 0.2, "kld": 0.0, "reversed_kld": 0.0, "js_kld": 0.0,
             "same_top_rate": 0.0, "mse_dp": 0.0}
    a, b, w = _fake_scores(6, base, delta, metrics)
    res = compare_items(a, b, w, metrics=metrics, confidence_level=0.95,
                           bootstrap_iters=500, seed=1234,
                           model_a_label="A", model_b_label="B")
    assert res["schema_version"] == SCHEMA_VERSION
    assert res["n_items"] == 6
    assert set(res["metrics"]) >= set(metrics) | {"ppl_ratio", "rms_dp"}
    nll_item = res["metrics"]["nll"]["item_weighted"]["delta_candidate_minus_baseline"]
    assert res["metrics"]["ppl_ratio"]["item_weighted"]["estimate"] == pytest.approx(
        math.exp(nll_item), rel=1e-9)
    assert res["metrics"]["nll"]["item_weighted"]["decision"]["verdict"] == "A closer"


def test_compare_items_derives_absolute_ppl_from_nll():
    """ppl is a display-only derived metric: PPL = exp(mean nll) per model, with
    its verdict LINKED to nll (a monotonic transform needs no own bootstrap)."""
    a = [{"nll": 1.0} for _ in range(6)]
    b = [{"nll": 1.2} for _ in range(6)]
    res = compare_items(a, b, [10.0] * 6, metrics=("nll",), confidence_level=0.95,
                           bootstrap_iters=200, seed=1234,
                           model_a_label="A", model_b_label="B")
    assert "ppl" in res["metrics"]
    assert res["metrics"]["ppl"]["source_metric"] == "nll"
    ppl_item = res["metrics"]["ppl"]["item_weighted"]
    assert ppl_item["baseline_ppl"] == pytest.approx(math.exp(1.0), rel=1e-9)
    assert ppl_item["candidate_ppl"] == pytest.approx(math.exp(1.2), rel=1e-9)
    assert ppl_item["delta_ppl_b_minus_a"] == pytest.approx(
        math.exp(1.2) - math.exp(1.0), rel=1e-9)
    assert ppl_item["decision"]["linked_to"] == "nll.item_weighted"


def test_compare_items_requested_metric_missing_raises():
    a = [{"nll": 1.0}] * 2
    b = [{"nll": 1.1}] * 2
    w = [1.0] * 2
    with pytest.raises(MissingMetricError):
        compare_items(a, b, w, metrics=("kld",), confidence_level=0.95,
                         bootstrap_iters=10, seed=1, model_a_label="A", model_b_label="B")


# --------------------------------------------------------------------------- #
# Task 7: dir loading + alignment guard
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Task 8: streaming per-item score builder
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Task 9: rendering
# --------------------------------------------------------------------------- #
def test_render_table_has_rows_and_verdicts():
    out = format_comparison_table(_result_for_render(),
                                     reference_label="F16", display_weighting="item")
    assert "Paired comparison: Q4_K_M (a) vs UNSLOTH (b)" in out
    # New ## Inputs section -- model paths + logits format + dataset visible.
    assert "## Inputs" in out
    assert "/m/qwen3vl-f16.gguf" in out
    assert "/m/qwen3vl-q4km.gguf" in out
    assert "/m/qwen3vl-unsloth.gguf" in out
    assert "Logits format: dense fp16, full vocab" in out
    assert "Dataset: lmms-lab/MMMU / default / validation (sort_by=row)" in out
    # Legacy single-line bullet is replaced by the Inputs table.
    assert "Reference (FP teacher): F16" not in out
    assert "nll" in out and "ppl_ratio" in out
    assert "do NOT compare per-model means independently" in out
    # rms_dp row hidden by default (its derived table label is absent)
    assert "√mean(mse_dp)" not in out


def test_render_shows_absolute_ppl_by_default():
    """Unlike rms_dp, absolute ppl is shown by default; a/b means are exp(nll)."""
    out = format_comparison_table(_result_for_render(),
                                     reference_label="F16", display_weighting="item")
    assert "ppl (a, b = exp(nll))" in out
    # _result_for_render uses a-side mean nll = 1.0 -> PPL_a = exp(1.0).
    assert f"{math.exp(1.0):.6f}" in out


def test_render_ppl_note_explains_exp_and_points_to_ratio():
    out = format_comparison_table(_result_for_render(),
                                     reference_label="F16", display_weighting="item")
    assert "ppl = exp(mean nll) per model" in out
    assert "tested change is ppl_ratio" in out


def test_render_inputs_missing_meta_path():
    """If a side's collect_meta.json was unreadable, the Inputs row falls back
    to a clear sentinel instead of crashing or printing 'None'."""
    r = _result_for_render()
    r["inputs"]["candidate_a"]["model_path"] = None
    out = format_comparison_table(r, reference_label="F16", display_weighting="item")
    assert "(collect_meta.json missing)" in out


def test_build_execution_metadata_records_cli_args_and_system(tmp_path, monkeypatch):
    monkeypatch.setattr(stats.render, "_system_metadata", lambda: {
        "platform": "Linux-test",
        "python": "3.12.0",
        "cpu_count": 12,
        "cpu_model": "Unit Test CPU",
        "ram_total_bytes": 32 * 1024 ** 3,
        "ram_available_bytes": 8 * 1024 ** 3,
    })
    args = smpc.parse_args([
        "--candidate-a", str(tmp_path / "a"),
        "--candidate-b", str(tmp_path / "b"),
        "--out", str(tmp_path / "report.md"),
        "--bootstrap-iters", "800",
    ])

    meta = build_execution_metadata(
        args,
        ["saved_metrics_paired_compare.py", "--candidate-a", str(tmp_path / "a")],
        device="cpu",
        jobs=1,
        auto_jobs_estimate={"jobs": 3, "mem_cap": 4},
    )

    assert meta["command"].startswith("saved_metrics_paired_compare.py --candidate-a ")
    assert meta["argv"] == ["saved_metrics_paired_compare.py",
                            "--candidate-a", str(tmp_path / "a")]
    assert meta["args"]["candidate_a"] == str(tmp_path / "a")
    assert meta["args"]["out"] == str(tmp_path / "report.md")
    assert meta["args"]["bootstrap_iters"] == 800
    assert meta["resolved"]["device"] == "cpu"
    assert meta["resolved"]["jobs"] == 1
    assert meta["resolved"]["auto_jobs_estimate"]["mem_cap"] == 4
    assert meta["system"]["cpu_model"] == "Unit Test CPU"
    assert meta["system"]["ram_available_bytes"] == 8 * 1024 ** 3


def test_execution_metadata_omits_host_volatile_fields_when_asked(tmp_path):
    """--omit-host-metadata drops exactly the two host-volatile blocks (the
    system dict and the auto-jobs estimate derived from MemAvailable) so two
    runs of one command are byte-identical -- the refactor plan's oracle."""
    args = smpc.parse_args([
        "--candidate-a", str(tmp_path / "a"),
        "--candidate-b", str(tmp_path / "b"),
        "--out", str(tmp_path / "report.md"),
    ])
    kw = dict(device="cpu", jobs=1, auto_jobs_estimate={"jobs": 3, "mem_cap": 4})
    full = build_execution_metadata(args, ["prog"], **kw)
    slim = build_execution_metadata(args, ["prog"], omit_host_metadata=True, **kw)
    assert "system" in full and "auto_jobs_estimate" in full["resolved"]
    assert "system" not in slim
    assert "auto_jobs_estimate" not in slim["resolved"]
    # the deterministic parts are untouched
    assert slim["command"] == full["command"]
    assert slim["resolved"]["device"] == "cpu" and slim["resolved"]["jobs"] == 1
    # and the renderer degrades cleanly: no CPU/RAM/Platform lines
    md = "\n".join(_format_execution_section(slim))
    assert "CPU:" not in md and "RAM:" not in md and "Platform:" not in md
    assert "Resolved backend: cpu; jobs=1" in md


def test_reproducible_report_env_sets_the_flag_default(tmp_path, monkeypatch):
    base = ["--candidate-a", str(tmp_path / "a"),
            "--candidate-b", str(tmp_path / "b"), "--out", str(tmp_path / "o.md")]
    monkeypatch.delenv("SKYMIZER_REPRODUCIBLE_REPORT", raising=False)
    assert smpc.parse_args(base).omit_host_metadata is False
    monkeypatch.setenv("SKYMIZER_REPRODUCIBLE_REPORT", "1")
    assert smpc.parse_args(base).omit_host_metadata is True
    monkeypatch.setenv("SKYMIZER_REPRODUCIBLE_REPORT", "0")
    assert smpc.parse_args(base).omit_host_metadata is False


def test_render_execution_metadata():
    r = _result_for_render()
    r["execution"] = {
        "command": "saved_metrics_paired_compare.py --candidate-a a --candidate-b b",
        "args": {"candidate_a": "a", "candidate_b": "b"},
        "resolved": {"device": "cpu", "jobs": 3},
        "system": {
            "cpu_count": 12,
            "cpu_model": "Unit Test CPU",
            "ram_total_bytes": 32 * 1024 ** 3,
            "ram_available_bytes": 8 * 1024 ** 3,
        },
    }

    out = format_comparison_table(r, reference_label="F16", display_weighting="item")

    assert "## Execution" in out
    assert "saved_metrics_paired_compare.py --candidate-a a --candidate-b b" in out
    assert "Resolved backend: cpu; jobs=3" in out
    assert "CPU: Unit Test CPU (12 logical)" in out
    assert "RAM: 32.00 GiB total, 8.00 GiB available" in out
    assert "| argument    | value |" in out
    assert "| candidate_a | a     |" in out


def test_render_falls_back_when_inputs_absent():
    """Older JSON / direct API users that don't populate result['inputs'] still
    get the legacy 'Reference (FP teacher)' bullet."""
    r = _result_for_render(with_inputs=False)
    out = format_comparison_table(r, reference_label="F16", display_weighting="item")
    assert "## Inputs" not in out
    assert "Reference (FP teacher): F16" in out


def test_render_alignment_block_is_prominent():
    out = format_comparison_table(
        _result_for_render(), reference_label="F16", display_weighting="item",
        drops={"reference": [], "candidate-a": [], "candidate-b": ["011_test_Math_3"]},
        n_matched=6)
    assert "## Alignment" in out
    assert "⚠" in out
    assert "candidate-b" in out and "011_test_Math_3" in out


def test_render_small_n_caveat():
    out = format_comparison_table(_result_for_render(),
                                     reference_label="F16", display_weighting="item")
    assert "small sample" in out  # n_items=6 < 30


# --------------------------------------------------------------------------- #
# Task 10: CLI end-to-end
# --------------------------------------------------------------------------- #


def _auto_job_dirs(tmp_path, keys, write_logits):
    ref = tmp_path / "ref"; a = tmp_path / "a"; b = tmp_path / "b"
    for d in (ref, a, b):
        (d / "logits").mkdir(parents=True)
    for key in keys:
        for d in (ref, a, b):
            write_logits(d / "logits" / f"{key}.bin")
    return ref, a, b


def test_resolve_item_end_uses_all_items_for_default_or_minus_one():
    assert resolve_item_end(None, 10) == (10, None)
    assert resolve_item_end(-1, 10) == (10, None)


def test_resolve_item_end_clamps_oversized_end_with_warning():
    end, warning = resolve_item_end(12, 10)

    assert end == 10
    assert warning == (
        "WARNING: --end 12 exceeds matched item count 10; "
        "falling back to 10 (all available items)."
    )


# --------------------------------------------------------------------------- #
# Code-review follow-ups (non-finite aborts, fail-closed n<2, --metrics guard,
# same_top_rate higher-is-better, reference==candidate warning)
# --------------------------------------------------------------------------- #


def test_compare_items_same_top_rate_higher_is_better():
    a = [{"same_top_rate": 0.80} for _ in range(5)]
    b = [{"same_top_rate": 0.90} for _ in range(5)]
    res = compare_items(a, b, [1.0] * 5, metrics=("same_top_rate",),
                           confidence_level=0.95, bootstrap_iters=500, seed=1,
                           model_a_label="A", model_b_label="B")
    d = res["metrics"]["same_top_rate"]["item_weighted"]["decision"]
    assert d["verdict"] == "B closer"   # b higher, higher-is-better


# --------------------------------------------------------------------------- #
# .npz dump dirs (--store-logits-type npz leaves logits/<key>.npz, no .bin)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("level,label", [(0.95, "95%"), (0.99, "99%"),
                                         (0.995, "99.5%"), (0.9, "90%")])
def test_report_never_contradicts_its_own_confidence_level(level, label):
    """--confidence-level is fully plumbed, but the verdict reason strings and
    the table header used to hard-code the literal "95% CI": at
    --confidence-level 0.99 the summary bullet said 99% while the table
    column and the JSON `reason` said 95%. Any archived report or JSON at a
    non-default level was self-contradicting."""
    scores_a = [{"kld": 0.10 + 0.01 * i} for i in range(8)]
    scores_b = [{"kld": 0.12 + 0.01 * i} for i in range(8)]
    weights = [10] * 8
    for ci_method, how in (("t", "paired Student-t (df = n - 1, no bootstrap)"),
                           ("studentized", "500 paired-bootstrap iters")):
        result = compare_items(scores_a, scores_b, weights, metrics=["kld"],
                                  confidence_level=level, bootstrap_iters=500,
                                  seed=1, model_a_label="A", model_b_label="B",
                                  ci_method=ci_method)
        for weighting in ("item_weighted", "token_weighted"):
            decision = result["metrics"]["kld"][weighting]["decision"]
            assert decision["reason"].startswith(f"{label} CI ")
            assert decision["confidence_level"] == level
        md = format_comparison_table(result, reference_label="F16")
        assert f"| {label} CI " in md or f"{label} CI" in md
        assert f"{label} CI via {how}" in md
        if level != 0.95:
            assert "95% CI" not in md


def test_report_flags_a_sorted_prefix_as_not_a_random_sample():
    """The collectors sort by num_images by default, so "the first N rows" is
    the FEWEST-image tail of the dataset. A report that names the dataset with
    no qualifier invites generalizing from a systematically atypical slice —
    and the effect being measured is a vision one."""
    scores_a = [{"kld": 0.10 + 0.01 * i} for i in range(6)]
    scores_b = [{"kld": 0.11 + 0.01 * i} for i in range(6)]
    result = compare_items(scores_a, scores_b, [10] * 6, metrics=["kld"],
                              confidence_level=0.95, bootstrap_iters=1000,
                              seed=1, model_a_label="A", model_b_label="B")
    result["inputs"] = build_inputs_metadata(
        {"model": "/m/ref.gguf", "dataset": "ds", "subset": "cfg",
         "split": "train", "sort_by": "num_images"},
        {"model": "/m/a.gguf"}, {"model": "/m/b.gguf"}, "F16", "A", "B")
    md = format_comparison_table(result, reference_label="F16")
    assert "sort_by=num_images" in md
    assert "PREFIX of that order" in md
    assert "smallest-`num_images` slice" in md

    # an unsorted collection has no such hazard and gets no note
    result["inputs"] = build_inputs_metadata(
        {"model": "/m/ref.gguf", "dataset": "ds", "sort_by": ""},
        {"model": "/m/a.gguf"}, {"model": "/m/b.gguf"}, "F16", "A", "B")
    assert "not a random sample" not in format_comparison_table(
        result, reference_label="F16")


def test_sorted_prefix_warning_names_a_window_when_start_end_are_used():
    inputs = {"sort_by": "num_images", "sort_desc": True}
    note = _sorted_prefix_warning(inputs, 20, 10, 30)
    assert "WINDOW of that order" in note and "--start 10" in note
    assert "largest-`num_images`" in note
    assert _sorted_prefix_warning({"sort_by": ""}, 20, 0, -1) is None


# ---------------------------------------------------------------------------
# source/docs hygiene: the top-K pipeline must stay gone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1])
def test_compare_items_rejects_too_few_units(count):
    with pytest.raises(AlignmentError, match="at least two"):
        compare_items([{"kld": 0.0}] * count, [{"kld": 0.1}] * count,
                      [1.0] * count, metrics=["kld"], confidence_level=0.95,
                      bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("metric", ["kld", "nll_ref"])
def test_compare_items_rejects_nonfinite_scores(bad, metric):
    a = [{"kld": 0.0, "nll_ref": 1.0}] * 2
    b = [{"kld": 0.1, "nll_ref": 1.0}, {"kld": 0.1, "nll_ref": 1.0}]
    b[0][metric] = bad
    with pytest.raises(NonFiniteMetricError, match="non-finite"):
        compare_items(a, b, [1.0, 1.0], metrics=["kld"], confidence_level=0.95,
                      bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_compare_items_rejects_invalid_weights(bad):
    with pytest.raises(ValueError, match="weights"):
        compare_items([{"kld": 0.0}] * 2, [{"kld": 0.1}] * 2,
                      [bad, 1.0], metrics=["kld"], confidence_level=0.95,
                      bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B")


def test_compare_items_rejects_nonfinite_token_arrays():
    with pytest.raises(NonFiniteMetricError, match="per-token"):
        compare_items([{"kld": 0.0}] * 2, [{"kld": 0.1}] * 2,
                      [2, 2], metrics=["kld"], confidence_level=0.95,
                      bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B",
                      token_metrics_a={"kld": [np.zeros(2), np.zeros(2)]},
                      token_metrics_b={"kld": [np.array([0.1, np.nan]), np.ones(2)]})


@pytest.mark.parametrize("metric", ["nll", "nll_ref"])
def test_compare_items_reports_ppl_range_error_in_log_units(metric):
    a = [{"nll": 1.0, "nll_ref": 2.0, metric: 1000.0}] * 3
    b = [{"nll": 1.1, "nll_ref": 2.0, metric: 1001.0}] * 3
    with pytest.raises(NonFiniteMetricError, match="PPL.*range.*log-scale NLL"):
        compare_items(a, b, [1.0] * 3, metrics=["nll"], confidence_level=0.95,
                      bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B")


def test_statistical_difference_can_also_establish_practical_equivalence():
    deltas = [0.001, 0.0011, 0.0009, 0.00105, 0.00095]
    result = compare_items([{"kld": 1.0}] * 5,
                           [{"kld": 1.0 + delta} for delta in deltas],
                           [512] * 5, metrics=["kld"], confidence_level=0.95,
                           bootstrap_iters=0, seed=1, model_a_label="A", model_b_label="B",
                           equivalence_margin=0.01)
    decision = result["metrics"]["kld"]["item_weighted"]["decision"]
    assert decision["verdict"] == "A closer"
    assert decision["statistically_distinguishable_from_null"] is True
    assert decision["equivalence_established"] is True
    assert decision["equivalence_bound"] == pytest.approx(0.00109816215807)
    assert decision["equivalence_margin"] == 0.01
    assert decision["equivalence_alpha"] == pytest.approx(0.025)
    assert decision["confidence_level"] == 0.95
    assert "equivalent at that margin" in decision["reason"]
