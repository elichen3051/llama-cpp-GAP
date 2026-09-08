"""Independent normal-model and descriptive-proxy checks for variance analysis."""

import json
import math

import numpy as np
import pytest
from scipy import integrate, special, stats

import stats.cli.variance_decomposition as vd


# --------------------------------------------------------------------------- #
# t quantiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.8, 0.95, 0.975, 0.99])
def test_planning_t_quantile_matches_independent_df2_formula(p):
    expected = (2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p))
    assert vd.t_ppf(p, 2) == pytest.approx(expected, rel=1e-10)


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


def _normal_t_power_quadrature(effect, n, confidence):
    """Integrate over the independent chi-square denominator, without nct APIs."""
    df = n - 1
    critical = stats.t.isf((1-confidence)/2, df)
    noncentrality = effect * math.sqrt(n)
    upper = stats.chi2.isf(1e-12, df)

    def integrand(v):
        threshold = critical * math.sqrt(v/df)
        return (special.ndtr(noncentrality-threshold) + special.ndtr(-noncentrality-threshold)) * stats.chi2.pdf(v, df)

    return integrate.quad(integrand, 0, upper, epsabs=1e-10)[0]


def test_required_n_is_monotone_and_matches_independent_normal_mixture():
    prev = None
    for snr in (1.0, 0.5, 0.432, 0.427, 0.4, 0.396, 0.292, 0.25, 0.1):
        n = vd.required_n(snr, 0.95, 0.8)
        assert n is not None and n >= 2
        if prev is not None:
            assert n > prev
        prev = n
        assert _normal_t_power_quadrature(snr, n, .95) >= .8
        if n > 2:
            assert _normal_t_power_quadrature(snr, n-1, .95) < .8


def test_required_n_supports_low_power_large_effect_and_extreme_confidence():
    assert vd.required_n(.01, .95, .01) == 2
    assert vd.required_n(0., .95, .01) == 2
    assert vd.required_n(20., .95, .8) == 2
    assert _normal_t_power_quadrature(20., 2, .95) > .8
    confidence = np.nextafter(1., 0.)
    n = vd.required_n(.1, confidence, .8)
    assert n is not None
    assert _normal_t_power_quadrature(.1, n, confidence) >= .8
    assert _normal_t_power_quadrature(.1, n-1, confidence) < .8


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
    assert stats.nct.cdf(.12*math.sqrt(250), 249, hi*math.sqrt(250)) == pytest.approx(.025, rel=1e-7)


def test_snr_interval_does_not_exclude_zero_for_an_inconclusive_small_normal_sample():
    lo, hi = vd.snr_interval(2., 3, .95)
    observed_t = 2*math.sqrt(3)
    # df=2 has the closed-form two-sided tail 1-t/sqrt(t*t+2).
    assert 1-observed_t/math.sqrt(observed_t**2+2) > .05
    assert lo == 0.0
    # For df=2, the denominator's squared value has density exp(-v/2)/2.
    cdf, _ = integrate.quad(lambda v: special.ndtr(observed_t*math.sqrt(v/2)-hi*math.sqrt(3))*math.exp(-v/2)/2, 0, math.inf)
    assert cdf == pytest.approx(.025, rel=1e-9)


@pytest.mark.parametrize('snr,n', [(.5,250),(1.,50),(2.,100)])
def test_snr_interval_local_brackets_match_independent_normal_chi_square_mixture(snr, n):
    lower, upper = vd.snr_interval(snr, n, .95)
    assert math.isfinite(lower) and math.isfinite(upper)
    df = n-1
    observed_t = snr*math.sqrt(n)
    integration_limit = stats.chi2.isf(1e-13, df)
    for endpoint, expected_cdf in ((lower,.975),(upper,.025)):
        noncentrality = endpoint*math.sqrt(n)
        cdf, _ = integrate.quad(lambda v: special.ndtr(observed_t*math.sqrt(v/df)-noncentrality)*stats.chi2.pdf(v,df),
                                0, integration_limit, epsabs=1e-11)
        assert cdf == pytest.approx(expected_cdf, abs=2e-10)


@pytest.mark.parametrize('tail_name', ['cdf','sf'])
def test_snr_interval_rejects_finite_tail_values_with_convergence_warnings(monkeypatch, tail_name):
    import warnings
    original = getattr(vd.stats.nct,tail_name)

    def unconverged(*args, **kwargs):
        value = original(*args, **kwargs)
        warnings.warn('noncentral t did not converge', RuntimeWarning)
        return value

    monkeypatch.setattr(vd.stats.nct,tail_name,unconverged)
    assert all(math.isnan(value) for value in vd.snr_interval(.4,30,.95))


def test_required_n_rejects_finite_power_values_with_convergence_warnings(monkeypatch):
    import warnings

    def unconverged(*args, **kwargs):
        warnings.warn('noncentral t did not converge', RuntimeWarning)
        return .9

    monkeypatch.setattr(vd.stats.nct,'sf',unconverged)
    assert vd.required_n(.5,.95,.8) is None


def test_snr_interval_zero_statistic_has_a_normal_noncentrality_interval():
    from statistics import NormalDist
    lo, hi = vd.snr_interval(0., 3, .95)
    assert lo == 0.
    assert hi == pytest.approx(NormalDist().inv_cdf(.975)/math.sqrt(3), rel=1e-12)


def test_snr_interval_rejects_an_unverified_root(monkeypatch):
    monkeypatch.setattr(vd.optimize, 'brentq', lambda *a, **k: 0.)
    assert all(math.isnan(v) for v in vd.snr_interval(1., 30, .95))


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
def test_deterministic_position_profile_keeps_raw_proxy_failure_and_suppresses_advice(tmp_path, monkeypatch, capsys):
    offsets = np.linspace(.09, .11, 40)
    profile = np.tile([-1., 1.], 32)
    deltas = [profile+x for x in offsets]
    lengths = np.full(40, 64)
    monkeypatch.setattr(vd, 'collect_deltas', lambda *args: (deltas, lengths, lengths, -1., 0))
    monkeypatch.setattr(vd, 'cost_model_from_manifest', lambda *args: dict(cfix=2.,ctok=.01,r2=1.,n_rows=40))
    (tmp_path/'metrics').mkdir()
    output = tmp_path/'variance.json'
    assert vd.main(['--candidate-a',str(tmp_path),'--candidate-b',str(tmp_path),'--output-json',str(output)]) == 0
    result = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert result['var_between_raw'] < 0
    assert result['token_noise_share_raw'] > 400
    assert result['n_required_floor'] is None and result['k_star_sqrt_rule'] is None
    assert result['snr_inf'] is None
    assert all(row['sd']**2 == pytest.approx(np.var(offsets,ddof=1)) for row in result['curve'])
    report = capsys.readouterr().out
    assert 'token noise explains everything' not in report
    assert 'extending --num-eval-tokens helps' not in report
    assert 'common conditional mean' in report


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


@pytest.mark.parametrize('rows', [
    [(100,0,5),(200,0,15),(300,0,25)],
    [(100,10,5),(100,20,6),(100,30,7)],
    [(100,10,5),(200,20,6),(300,30,7)],
    [(100,10,5),(200,float('nan'),6),(300,30,7)],
    [(100,10,5),(200,20,-1),(300,30,7)],
])
def test_cost_model_rejects_negative_unidentified_or_nonfinite_fits(tmp_path, rows):
    path = tmp_path/'manifest.csv'
    path.write_text('n_eval,n_prefill,wall_s,status\n'+''.join(f'{ne},{npref},{wall},OK\n' for ne,npref,wall in rows))
    assert vd.cost_model_from_manifest(path) is None


@pytest.mark.parametrize('name,column', [('kld','kld'),('reversed_kld','reversed_kld'),('js_kld','js_kld'),('nll','nll_cand')])
def test_metric_differences_promote_stored_operands_before_subtraction(name, column):
    from fractions import Fraction
    a = np.array([1e-8,1.], dtype=np.float32)
    b = np.array([1.,2e-8], dtype=np.float32)
    want = sum(Fraction(float(y))-Fraction(float(x)) for x,y in zip(a,b))/2
    delta = vd.METRIC_COLUMNS[name]({column:a},{column:b})
    assert delta.dtype == np.float64
    assert delta.mean() > 0
    assert delta.mean() == pytest.approx(float(want), rel=1e-8)


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
    assert vd.required_n(1e-50, 0.95, 0.8) is None
    assert vd.required_n(1e-300, 0.95, 0.8) is None


def test_zero_variance_nonzero_effect_has_no_fabricated_normal_model_size():
    delta = [np.ones(16) for _ in range(3)]
    rows = vd.build_curve(delta,np.full(3,16),0.,np.zeros(3),[16],.95,.8)
    assert rows[0]['n_required'] is None
    assert rows[0]['snr_status'] == 'zero_variance_nonzero_effect'
