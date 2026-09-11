"""Prospective power for ordered token-prefix estimands."""

import json
from argparse import Namespace

import numpy as np
import pytest

import stats.cli.power_analysis as power_cli
from stats.engine import compare_items
from stats.power import _batch_item_estimate_and_se, build_design, paired_t_interval, wilson_interval
import lib.kld_metrics_io as kio
from fakes import make_records, write_vlmk


def test_token_weighting_is_rejected_before_power_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("token-weighted power must not simulate")

    monkeypatch.setattr("stats.power._simulate_t_surface", forbidden)
    values = np.array([-0.1, 0.2, 0.3])
    weights = np.array([1., 10., 100.])
    with pytest.raises(ValueError, match="descriptive only"):
        build_design({16: values}, {16: weights}, [25], weighting="token", sesoi=0.1)
    with pytest.raises(ValueError, match="descriptive only"):
        paired_t_interval(values, weights, "token", 0.95)
    with pytest.raises(SystemExit):
        power_cli.parse_args(["--candidate-a", "a", "--candidate-b", "b", "--token-caps", "16",
                              "--weighting", "token", "--out", "power.md"])


@pytest.mark.parametrize("total,hits", [(1, 0), (1, 1), (16, 0), (16, 16), (16, 8), (100, 3), (100, 97)])
def test_wilson_endpoints_solve_the_binomial_score_equation(total, hits):
    from statistics import NormalDist
    confidence = 2.0 * NormalDist().cdf(2.0) - 1.0
    lower, upper = wilson_interval(hits, total, confidence)
    proportion = hits / total
    assert 0 <= lower <= proportion <= upper <= 1
    for bound in (lower, upper):
        assert total * (proportion - bound) ** 2 == pytest.approx(4.0 * bound * (1.0 - bound), abs=1e-13)
    if hits == 0:
        assert upper == pytest.approx(4.0 / (total + 4.0), abs=1e-14)
    if hits == total:
        assert lower == pytest.approx(total / (total + 4.0), abs=1e-14)


@pytest.mark.parametrize("hits,total,confidence", [(0, 0, .95), (-1, 4, .95), (5, 4, .95),
                                                  (1.5, 4, .95), (1, 4.5, .95), (1, 4, 1.0), (1, 4, float("nan"))])
def test_wilson_rejects_invalid_counts_and_confidence(hits, total, confidence):
    with pytest.raises((ValueError, TypeError)):
        wilson_interval(hits, total, confidence)


@pytest.mark.parametrize("values", [[1e200, -1e200, 1e200], [1e-200, -1e-200, 1e-200], [1.1e-162, 2.2e-162, 3.3e-162], [1e-161, 2e-161, 4e-161]])
def test_planning_interval_cannot_turn_invalid_variance_into_significance(values):
    with pytest.raises(ValueError, match="standard error"):
        paired_t_interval(values, np.ones(3), "item", .95)
    with pytest.raises(ValueError, match="standard error"):
        _batch_item_estimate_and_se(np.array([values]))


@pytest.mark.parametrize("weighting", ["item"])
def test_paired_t_interval_matches_production_compare(weighting):
    differences = np.array([-0.4, -0.1, 0.2, 0.5, 0.9], dtype=float)
    weights = np.array([3, 5, 7, 11, 13], dtype=float)
    scores_a = [{"kld": 1.0} for _ in differences]
    scores_b = [{"kld": 1.0 + value} for value in differences]
    production = compare_items(
        scores_a, scores_b, weights,
        metrics=("kld",), confidence_level=0.95, bootstrap_iters=0,
        seed=7, model_a_label="A", model_b_label="B", ci_method="t",
        primary_metric="kld", primary_weighting=weighting,
        position_buckets=None,
    )
    block = production["metrics"]["kld"][f"{weighting}_weighted"]
    planned = paired_t_interval(differences, weights, weighting, 0.95)
    assert planned["estimate"] == pytest.approx(
        block["delta_candidate_minus_baseline"], abs=1e-15
    )
    assert planned["standard_error"] == pytest.approx(
        block["ci_delta"]["standard_error"], abs=1e-15
    )
    assert planned["lower"] == pytest.approx(block["ci_delta"]["lower"], abs=1e-15)
    assert planned["upper"] == pytest.approx(block["ci_delta"]["upper"], abs=1e-15)


def test_precision_only_never_substitutes_the_observed_effect():
    rng = np.random.default_rng(4)
    differences = {16: rng.normal(0.8, 0.5, 200)}
    weights = {16: np.full(200, 16.0)}
    design = build_design(
        differences, weights, [25, 100], sesoi=None, outer_reps=0
    )
    assert design["mode"] == "precision_only"
    assert design["sesoi"] is None
    assert all("power" not in cell for cell in design["cells"])
    assert design["cells"][1]["normal_model_mde"] < design["cells"][0]["normal_model_mde"]
    assert all(row["first_evaluated_n_point_estimate"] is None
               for row in design["required_n_by_cap"])


@pytest.mark.parametrize("weighting", ["item"])
def test_duplicating_within_item_tokens_does_not_create_power(weighting):
    """A longer cap with identical item summaries is correlated evidence,
    not more independent observations."""
    rng = np.random.default_rng(9)
    item_differences = rng.normal(0.0, 0.4, 200)
    differences = {1: item_differences, 128: item_differences.copy()}
    weights = {1: np.ones(200), 128: np.full(200, 128.0)}
    design = build_design(
        differences, weights, [40], weighting=weighting, sesoi=0.10,
        reps=800, outer_reps=0, seed=21,
    )
    by_cap = {cell["token_cap"]: cell for cell in design["cells"]}
    assert by_cap[1]["power"] == by_cap[128]["power"]
    assert by_cap[1]["null_rejection_rate"] == by_cap[128]["null_rejection_rate"]
    assert (
        by_cap[128]["expected_evaluated_tokens"]
        == 128 * by_cap[1]["expected_evaluated_tokens"]
    )


@pytest.mark.statistical
def test_power_can_be_nonmonotone_across_token_caps_and_null_is_calibrated():
    rng = np.random.default_rng(11)
    n_pilot = 800
    base = rng.normal(size=n_pilot)
    differences = {
        16: 0.20 * base,
        32: 1.00 * rng.normal(size=n_pilot),
        64: 0.10 * rng.normal(size=n_pilot),
    }
    weights = {cap: np.full(n_pilot, cap, dtype=float) for cap in differences}
    design = build_design(
        differences, weights, [40], sesoi=0.10, reps=4000,
        outer_reps=0, seed=19,
    )
    by_cap = {cell["token_cap"]: cell for cell in design["cells"]}
    assert by_cap[16]["power"] > by_cap[32]["power"]
    assert by_cap[64]["power"] > by_cap[16]["power"]
    for cell in by_cap.values():
        assert cell["null_rejection_mc_lower"] <= 0.05 <= \
            cell["null_rejection_mc_upper"]


def test_pilot_profile_is_one_coherent_constant_shift():
    residual = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    differences = {
        16: residual + 0.10,
        32: residual + 0.30,
        64: residual + 0.50,
    }
    weights = {
        16: np.full(5, 16.0),
        32: np.full(5, 32.0),
        64: np.full(5, 64.0),
    }
    design = build_design(
        differences, weights, [20], sesoi=-0.20,
        effect_profile="pilot", reference_cap=32, reps=20,
        outer_reps=0, seed=2,
    )
    cells = {cell["token_cap"]: cell for cell in design["cells"]}
    effects = {cap: cell["assumed_effect"] for cap, cell in cells.items()}
    assert effects == pytest.approx({16: -0.40, 32: -0.20, 64: 0.0})
    assert cells[64]["directional_power_defined"] is False
    assert cells[64]["power"] is None


def _metrics_pair(tmp_path, n_items=6, drift=False):
    dirs = (tmp_path / "a", tmp_path / "b")
    for directory in dirs:
        (directory / "metrics").mkdir(parents=True)
    for item in range(n_items):
        key = f"{item:03d}_item{item}"
        record_a = make_records(npos=6, seed=100 + item)
        record_b = record_a.copy()
        record_b["kld"] = record_a["kld"] + np.array(
            [0.02 * (item - 2), 0.02 * (item - 2), 0.04, 0.04, -0.01, -0.01],
            dtype=np.float32,
        )
        if drift and item == 0:
            record_b["nll_ref"][0] += np.float32(1e-3)
        for directory, record in zip(dirs, (record_a, record_b)):
            binary = directory / "metrics" / f"{key}.bin"
            write_vlmk(binary, record)
            kio.convert_kld_bin_to_npz(binary, binary.with_suffix(".npz"))
            binary.unlink()
    from fakes import completed_collection
    from test_saved_metrics_paired_compare import BASE_META
    import json
    for directory in dirs:
        completed_collection(directory, [f"{i:03d}_item{i}" for i in range(n_items)])
        (directory / "collect_meta.json").write_text(json.dumps(BASE_META))
    return dirs


def _panel_args(a_dir, b_dir):
    return Namespace(
        candidate_a=a_dir, candidate_b=b_dir, start=0, end=None,
        allow_interaction=False, allow_ref_drift=False, metric="kld",
    )


def test_cap_panel_uses_the_production_score_item_contract(tmp_path):
    a_dir, b_dir = _metrics_pair(tmp_path)
    panel = power_cli.load_cap_panel(_panel_args(a_dir, b_dir), [2, 6])
    for cap in (2, 6):
        manual = []
        manual_weights = []
        for key in panel["used"]:
            score_a, score_b, _ta, _tb, keep, *_ = power_cli.paired_io.score_item(
                key, a_dir, b_dir, cap
            )
            manual.append(score_b["kld"] - score_a["kld"])
            manual_weights.append(keep)
        np.testing.assert_allclose(panel["differences"][cap], manual, atol=0, rtol=0)
        np.testing.assert_array_equal(panel["weights"][cap], manual_weights)


def test_cap_panel_fails_on_the_same_reference_drift_guard(tmp_path):
    a_dir, b_dir = _metrics_pair(tmp_path, drift=True)
    with pytest.raises(SystemExit, match="reference drift"):
        power_cli.load_cap_panel(_panel_args(a_dir, b_dir), [2, 6])


@pytest.mark.statistical
def test_cli_writes_prospective_schema_and_report(tmp_path):
    a_dir, b_dir = _metrics_pair(tmp_path)
    report = tmp_path / "power.md"
    payload_path = tmp_path / "power.json"
    rc = power_cli.main([
        "--candidate-a", str(a_dir), "--candidate-b", str(b_dir),
        "--token-caps", "2", "6", "--sample-sizes", "4", "8",
        "--sesoi", "0.03", "--reps", "80", "--outer-reps", "10",
        "--seed", "3", "--out", str(report), "--output-json", str(payload_path),
    ])
    assert rc == 0
    payload = json.loads(payload_path.read_text())
    assert payload["schema_version"] == "company-sequential-power-v2"
    assert payload["estimand"]["sampling_unit"] == "item"
    assert payload["design"]["mode"] == "prospective_power"
    assert len(payload["design"]["cells"]) == 4
    assert all(
        "first_evaluated_n_pilot_power_p10" in row
        for row in payload["design"]["required_n_by_cap"]
    )
    markdown = report.read_text()
    assert "never resampled or counted as independent" in markdown
    assert "pilot power p10" in markdown
    assert "pilot's observed effect" not in markdown
    assert markdown.endswith("\n")
    assert "pointwise" in markdown and "no simultaneous guarantee" in markdown
    assert "plug-in" in markdown
    assert payload["estimand"]["score_direction"] == "lower_is_better"
    assert len(payload["inputs"]["selected_item_keys"]) == payload["alignment"]["n_used"]
    import hashlib
    keys = payload["inputs"]["selected_item_keys"]
    assert payload["inputs"]["selected_item_keys_sha256"] == hashlib.sha256(json.dumps(keys, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("options", [{"reps": float("nan")}, {"reps": 2.5}, {"outer_reps": float("inf")},
                                      {"outer_reps": True}, {"mc_confidence_level": float("nan")},
                                      {"mc_confidence_level": 2}, {"seed": -1}, {"effect_profile": "other"},
                                      {"effect_profile": "pilot"}, {"reference_cap": 1}])
def test_public_design_rejects_invalid_domains_before_simulation(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid arguments reached simulation")
    monkeypatch.setattr("stats.power._simulate_t_surface", forbidden)
    with pytest.raises(ValueError):
        build_design({1: [-1., 1.]}, {1: [1., 1.]}, [2], **options)


@pytest.mark.parametrize("sizes", [[2.9], [True], [float("nan")], []])
def test_public_design_does_not_truncate_or_accept_invalid_sample_sizes(sizes):
    with pytest.raises(ValueError):
        build_design({1: [-1., 1.]}, {1: [1., 1.]}, sizes, outer_reps=0)


@pytest.mark.parametrize("pilot", [np.zeros(5), np.full(3, .1), np.array([1., 1., np.nextafter(1., 2.)])])
def test_unidentified_pilot_variance_cannot_report_perfect_power_or_zero_mde(pilot):
    for effect in (None, .1):
        with pytest.raises(ValueError, match="unidentified"):
            build_design({1: pilot}, {1: np.ones(len(pilot))}, [2], sesoi=effect, outer_reps=0)


def test_small_metric_units_preserve_power_and_scale_precision():
    pilot = np.array([-2., -1., 0., 1., 2.])
    options = dict(outer_reps=0, reps=100, seed=1)
    regular = build_design({1: pilot}, {1: np.ones(5)}, [10], sesoi=.5, **options)["cells"][0]
    small = build_design({1: pilot * 1e-12}, {1: np.ones(5)}, [10], sesoi=.5e-12, **options)["cells"][0]
    assert small["power"] == regular["power"]
    assert small["normal_model_mde"] == pytest.approx(regular["normal_model_mde"] * 1e-12, rel=1e-12, abs=0)


def test_null_inflation_suppresses_all_sample_size_crossings():
    design = build_design({1: [-1., 1.]}, {1: [1., 1.]}, [2], sesoi=20., reps=4000, outer_reps=0, seed=1)
    cell = design["cells"][0]
    assert cell["null_rejection_rate"] == pytest.approx(.5, abs=.04)
    assert cell["power"] > .99
    assert cell["null_calibration_status"] == "detected_inflation"
    assert cell["crossing_eligible"] is False
    assert all(value is None for key, value in design["required_n_by_cap"][0].items() if key != "token_cap")


def test_degenerate_outer_draws_make_nuisance_band_unavailable():
    design = build_design({1: [0., 0., 0., 0., 10.]}, {1: np.ones(5)}, [20], sesoi=1., reps=100, outer_reps=200)
    band = design["cells"][0]["pilot_uncertainty"]
    assert band["degenerate_draws"] > 0
    assert band["status"] == "unavailable_degenerate_resamples"
    assert band["mde_p10"] is None
    assert "power_p90" not in band
    assert design["required_n_by_cap"][0]["first_evaluated_n_pilot_power_p10"] is None


@pytest.mark.parametrize("effect", [0., .01, 1., 10., -1.])
def test_directional_normal_model_matches_independent_gaussian_integral(effect):
    """At df=1, condition on |Z| in (Z0+nc)/|Z|; integrate elementary normal tails."""
    import math
    from scipy.integrate import quad
    from stats.power import _normal_model_power
    critical = 1.0 / math.tan(math.pi * .025)
    nc = abs(effect) * math.sqrt(2.)
    want = quad(lambda z: .5 * math.erfc((critical*z - nc) / math.sqrt(2.))
                * math.sqrt(2./math.pi) * math.exp(-z*z/2.), 0., math.inf, epsabs=1e-12)[0]
    assert _normal_model_power(effect, 1., 2, .95) == pytest.approx(want, rel=1e-10)


def test_normal_model_mde_reaches_target_in_independent_df1_integral():
    import math
    from scipy.integrate import quad
    from stats.power import _normal_model_mde_per_sd
    effect = _normal_model_mde_per_sd(2, .95, .8)
    critical = 1.0 / math.tan(math.pi * .025)
    nc = effect * math.sqrt(2.)
    got = quad(lambda z: .5 * math.erfc((critical*z-nc)/math.sqrt(2.))
               * math.sqrt(2./math.pi) * math.exp(-z*z/2.), 0., math.inf, epsabs=1e-12)[0]
    assert got == pytest.approx(.8, abs=1e-10)


def test_pilot_crossing_also_requires_mc_lower_bound(monkeypatch):
    import stats.power as power
    def cells(*args, **kwargs):
        return [{"token_cap": 1, "sample_size": 20, "assumed_effect": 10.,
                 "power": .7, "power_mc_lower": .6, "power_mc_upper": .8,
                 "wrong_sign_rate": 0., "directional_power_defined": True,
                 "null_rejection_rate": .05, "null_rejection_mc_lower": .02, "null_rejection_mc_upper": .09}]
    monkeypatch.setattr(power, "_simulate_t_surface", cells)
    design = build_design({1: np.linspace(-1.,1.,50)}, {1: np.ones(50)}, [20], sesoi=10., outer_reps=20)
    assert design["cells"][0]["pilot_uncertainty"]["power_p10"] > .8
    assert design["required_n_by_cap"][0]["first_evaluated_n_pilot_power_p10"] is None


@pytest.mark.parametrize("effect", [.01, 1e-15])
def test_outer_resamples_use_the_same_near_constant_guard_as_the_pilot(effect):
    pilot = np.r_[1. + (np.arange(50) % 3) * np.spacing(1.), 2.]
    design = build_design({1: pilot}, {1: np.ones(51)}, [100], sesoi=effect, reps=100, outer_reps=200)
    band = design["cells"][0]["pilot_uncertainty"]
    assert band["status"] == "unavailable_degenerate_resamples"
    assert band["degenerate_draws"] > 0
    assert "power_p90" not in band



def test_production_unrepresentable_confidence_rejected_before_noncentral_t(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unrepresentable confidence reached nct")
    monkeypatch.setattr("stats.power._normal_model_mde_per_sd", forbidden)
    with pytest.raises(ValueError, match="too close to 1"):
        build_design({1: [-1., 0., 1.]}, {1: np.ones(3)}, [2], confidence_level=np.nextafter(1., 0.), outer_reps=0)


def test_unresolved_near_null_mde_cannot_be_reported_as_zero():
    with pytest.raises(ValueError, match="MDE inversion"):
        build_design({1: [-1., 0., 1.]}, {1: np.ones(3)}, [25], confidence_level=1e-14,
                     target_power=.500000000000001, outer_reps=0)



def test_wilson_rejects_unrepresentable_confidence():
    with pytest.raises(ValueError, match="too close to 1"):
        wilson_interval(8, 10, np.nextafter(1., 0.))
    with pytest.raises(ValueError, match="too close to 1"):
        build_design({1: [-1., 0., 1.]}, {1: np.ones(3)}, [2], mc_confidence_level=np.nextafter(1., 0.))


def test_noncentral_t_convergence_warning_cannot_be_reported_as_power(monkeypatch):
    import warnings
    from stats.power import _normal_model_power
    def unconverged(*args, **kwargs):
        warnings.warn("series did not converge", RuntimeWarning)
        return .9
    monkeypatch.setattr("stats.power.nct.sf", unconverged)
    with pytest.raises(ValueError, match="failed numerical evaluation"):
        _normal_model_power(1., 1., 2, .95)


def test_outer_variance_underflow_disables_bands_without_aborting_valid_pilot():
    pilot = np.array([0., 1e-160, 2e-160, -1e-150, 1e-150])
    design = build_design({1: pilot}, {1: np.ones(5)}, [10], outer_reps=100, seed=7)
    band = design["cells"][0]["pilot_uncertainty"]
    assert band["status"] == "unavailable_degenerate_resamples"
    assert band["degenerate_draws"] > 0
    assert design["pilot_caps"][0]["effective_sd_outer_p10"] is None
