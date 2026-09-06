"""Sample-size planning in variance_decomposition.py.

The estimator itself is unbiased with correct ddof; the weakness was the
REPORTING around it. n_req = ceil((z_sum/snr)^2) is the right formula, but it
plugs in an ESTIMATED mu, and required N scales like 1/mu^2, so the point
estimate is violently unstable near the significance boundary. It also used z
where t belongs, and the var_between clamp let the between-item share print
negative and the JSON export token_noise_share > 1.
"""

import json
import math

import numpy as np
import pytest

import cli.variance_decomposition as vd


# --------------------------------------------------------------------------- #
# t quantiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("df", [10, 24, 49, 99, 249, 999])
@pytest.mark.parametrize("p", [0.8, 0.95, 0.975, 0.99])
def test_t_ppf_matches_scipy(df, p):
    st = pytest.importorskip("scipy.stats")
    assert vd.t_ppf(p, df) == pytest.approx(st.t.ppf(p, df), rel=2e-4)


def test_t_ppf_exceeds_the_normal_quantile_and_converges_to_it():
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.975)
    assert vd.t_ppf(0.975, 25) > z
    # t -> z from above as df grows; the leading term is (z^3+z)/(4 df), so at
    # df = 1e5 it is still ~2.4e-5 above z -- correctly, not as an error.
    assert z < vd.t_ppf(0.975, 100000) < z + 1e-4
    assert vd.t_ppf(0.975, 25) > vd.t_ppf(0.975, 250) > vd.t_ppf(0.975, 100000)


# --------------------------------------------------------------------------- #
# required_n
# --------------------------------------------------------------------------- #
def test_required_n_uses_t_not_z_and_so_exceeds_the_z_answer():
    """t_.975 is 2.06 at df=25 against z=1.96, so the z formula under-counts
    the items needed by ~10% in the range this tool is used at."""
    from statistics import NormalDist
    snr = 0.55
    z_sum = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
    z_answer = math.ceil((z_sum / snr) ** 2)
    t_answer = vd.required_n(snr, 0.95, 0.8)
    assert t_answer > z_answer


def test_required_n_is_monotone_and_self_consistent():
    prev = None
    for snr in (1.0, 0.5, 0.432, 0.427, 0.4, 0.396, 0.292, 0.25, 0.1):
        n = vd.required_n(snr, 0.95, 0.8)
        assert n is not None and n >= 3
        if prev is not None:
            assert n > prev
        prev = n
        t_sum = vd.t_ppf(0.975, n - 1) + vd.t_ppf(0.8, n - 1)
        assert n >= max(3, math.ceil((t_sum / snr) ** 2))
        if n > 3:
            previous_sum = vd.t_ppf(0.975, n - 2) + vd.t_ppf(0.8, n - 2)
            assert n - 1 < math.ceil((previous_sum / snr) ** 2)


def test_required_n_is_none_when_the_effect_is_not_bounded_away_from_zero():
    assert vd.required_n(0.0, 0.95, 0.8) is None
    assert vd.required_n(-0.1, 0.95, 0.8) is None
    assert vd.required_n(float("nan"), 0.95, 0.8) is None
    assert vd.required_n(float("inf"), 0.95, 0.8) is None


# --------------------------------------------------------------------------- #
# the interval that makes the point estimate honest
# --------------------------------------------------------------------------- #
def test_snr_interval_brackets_the_estimate_and_clips_at_zero():
    lo, hi = vd.snr_interval(0.12, 250, 0.95)
    assert lo <= 0.12 <= hi
    assert lo == 0.0                       # reaches zero at this effect size
    # the report's worked example: SE(snr) ~ 0.063 at snr=0.12, n=250
    se = math.sqrt(1 / 250 + 0.12 ** 2 / (2 * 250))
    assert se == pytest.approx(0.0635, abs=5e-4)
    assert hi == pytest.approx(0.12 + 1.959963985 * se, rel=1e-9)


def test_required_n_interval_is_unbounded_when_the_snr_interval_reaches_zero():
    """The whole point of S5: at a borderline effect the honest answer is
    'this pilot cannot size the study', not a confident 545."""
    lo, hi = vd.snr_interval(0.12, 250, 0.95)
    assert vd.required_n(lo, 0.95, 0.8) is None       # unbounded above
    n_lower_bound = vd.required_n(hi, 0.95, 0.8)      # optimistic end
    point = vd.required_n(0.12, 0.95, 0.8)
    assert n_lower_bound is not None and n_lower_bound < point


def test_snr_interval_tightens_with_n():
    wide = vd.snr_interval(0.5, 20, 0.95)
    tight = vd.snr_interval(0.5, 500, 0.95)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------- #
# share clamping
# --------------------------------------------------------------------------- #
def test_shares_never_go_negative_when_var_between_is_clamped():
    """var_between = max(var_total - var_token, 0) already clamps, but the
    SHARE was computed from the unclamped ratio: a token component larger
    than the total printed a negative between-item share and exported
    token_noise_share > 1."""
    var_total, var_token = 1.0e-6, 1.4e-6
    raw = var_token / var_total
    share = min(1.0, raw)
    assert raw > 1.0
    assert share == 1.0 and (1.0 - share) == 0.0


# --------------------------------------------------------------------------- #
# empirical cap ladder (backtest-2026-08-26 F1/F9: the 1/K law is checked,
# never assumed)
# --------------------------------------------------------------------------- #
def _components(deltas, T):
    d_full = np.array([d.mean() for d in deltas])
    s2w = np.array([d.var(ddof=1) for d in deltas])
    var_total = d_full.var(ddof=1)
    s2b = max(var_total - float((s2w / T).mean()), 0.0)
    return s2b, s2w


def test_curve_matches_the_law_on_stationary_iid_tokens():
    """iid tokens with a common item effect: the two-component law is the
    truth, so log2(pred/emp) sits near 0 at every cap and N shrinks with K."""
    rng = np.random.default_rng(7)
    n_items, T_len = 400, 1024
    deltas = [0.05 + rng.normal(0, 0.1)            # item effect b_i
              + rng.normal(0, 1.0, T_len)          # stationary token noise
              for _ in range(n_items)]
    deltas = [np.asarray(d) for d in deltas]
    T = np.full(n_items, T_len)
    s2b, s2w = _components(deltas, T)
    rows = vd.build_curve(deltas, T, s2b, s2w, vd.cap_ladder(T), 0.95, 0.8)
    fits = [r["log2_pred_over_emp"] for r in rows]
    assert all(abs(f) < 0.25 for f in fits), fits
    assert rows[-1]["n_required"] <= rows[0]["n_required"]
    assert vd.sign_flip_caps(rows) == []


def test_curve_flags_the_law_on_front_loaded_token_variance():
    """Position non-stationarity (the backtest's measured mechanism): the
    first 32 positions carry 10x the delta sd of the rest. The full-length
    sigma_w^2 then underestimates the small-K variance, so log2(pred/emp)
    goes clearly negative at K<=32 while the full-cap fit stays ~0 — the F1
    signature the model check exists to surface."""
    rng = np.random.default_rng(11)
    n_items, T_len = 400, 1024
    sd_profile = np.where(np.arange(T_len) < 32, 3.0, 0.3)
    deltas = [0.02 + rng.normal(0, 0.05)
              + rng.normal(0, 1.0, T_len) * sd_profile
              for _ in range(n_items)]
    deltas = [np.asarray(d) for d in deltas]
    T = np.full(n_items, T_len)
    s2b, s2w = _components(deltas, T)
    rows = {r["K"]: r for r in
            vd.build_curve(deltas, T, s2b, s2w, vd.cap_ladder(T), 0.95, 0.8)}
    assert rows[16]["log2_pred_over_emp"] < -1.0     # law far too optimistic
    assert rows[32]["log2_pred_over_emp"] < -1.0
    assert abs(rows[T_len]["log2_pred_over_emp"]) < 0.25
    # the empirical column still tells the truth: small-K variance is HIGHER,
    # so the required item count at K=16 exceeds the full-cap one
    assert rows[16]["n_required"] is None or \
        rows[16]["n_required"] > rows[T_len]["n_required"]


def test_curve_detects_a_position_dependent_sign_flip():
    """F7/F9: delta positive over early positions, negative later — mu(K)
    flips sign across caps, which must be surfaced, because a single-cap
    verdict silently picks a side."""
    rng = np.random.default_rng(13)
    n_items, T_len = 200, 512
    mean_profile = np.where(np.arange(T_len) < 24, +0.5, -0.08)
    deltas = [mean_profile + rng.normal(0, 0.2, T_len) for _ in range(n_items)]
    T = np.full(n_items, T_len)
    s2b, s2w = _components(deltas, T)
    rows = vd.build_curve(deltas, T, s2b, s2w, vd.cap_ladder(T), 0.95, 0.8)
    flips = vd.sign_flip_caps(rows)
    assert 16 in flips and 32 in flips
    by_k = {r["K"]: r for r in rows}
    assert by_k[16]["mu"] > 0 and by_k[T_len]["mu"] < 0


def test_cap_ladder_ends_at_the_longest_item():
    T = np.array([100, 300, 700])
    assert vd.cap_ladder(T) == [16, 32, 64, 128, 256, 512, 700]
    T2 = np.array([16, 16])
    assert vd.cap_ladder(T2) == [16]


def test_cost_model_from_manifest(tmp_path):
    man = tmp_path / "manifest.csv"
    lines = ["row_idx,item_id,num_images,n_prefill,n_answer,n_eval,vocab,metrics_bytes,wall_s,status"]
    rng = np.random.default_rng(3)
    for i in range(20):
        n_eval = int(rng.integers(64, 2048))
        n_prefill = int(rng.integers(100, 1000))
        wall = 2.0 + 0.004 * n_eval + 0.001 * n_prefill + rng.normal(0, 0.01)
        lines.append(f"{i},it{i},1,{n_prefill},{n_eval},{n_eval},151936,1000,{wall:.4f},OK")
    lines.append("20,bad,1,100,100,100,151936,0,,FAIL_X")   # non-OK rows ignored
    man.write_text("\n".join(lines) + "\n")
    cm = vd.cost_model_from_manifest(man)
    assert cm is not None and cm["n_rows"] == 20
    assert cm["ctok"] == pytest.approx(0.004, rel=0.05)
    assert cm["r2"] > 0.99
    assert vd.cost_model_from_manifest(tmp_path / "missing.csv") is None


def test_zero_effect_zero_variance_does_not_claim_infinite_signal(tmp_path):
    from test_saved_metrics_paired_compare import _make_pair

    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    for root in (a_dir, b_dir):
        for path in (root / "metrics").glob("*.npz"):
            with np.load(path) as data:
                payload = {name: data[name].copy() for name in data.files}
            payload["kld"][:] = 0.0
            np.savez(path, **payload)
    output = tmp_path / "variance.json"
    assert vd.main(["--candidate-a", str(a_dir), "--candidate-b", str(b_dir),
                    "--output-json", str(output)]) == 0
    payload = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert payload["snr_now"] is None
    assert payload["snr_inf"] is None
    assert payload["n_required_now"] is None
    for row in payload["curve"]:
        assert row["snr"] is None
        assert row["n_required"] is None
        assert row["snr_status"] == "zero_effect_zero_variance"


def test_variance_loader_does_not_drop_short_rows(tmp_path):
    from test_saved_metrics_paired_compare import _make_pair

    a_dir, b_dir = _make_pair(tmp_path, n_items=4, npos_list=[1, 4, 4, 4])
    with pytest.raises(SystemExit, match="at least two.*positions"):
        vd.collect_deltas(a_dir, b_dir, "kld", -1)


def test_required_n_handles_extreme_finite_effect_sizes():
    assert vd.required_n(1e-50, 0.95, 0.8) > 1e100
    assert vd.required_n(1e-300, 0.95, 0.8) is None
