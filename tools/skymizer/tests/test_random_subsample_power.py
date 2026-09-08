"""The subsample sweep: reproducibility mode vs power mode.

The original tool drew subsets WITHOUT replacement from the same finite pool
whose full-sample verdict was the target, and called the result "power". Two
biases compound there: a size-N draw's mean carries variance
(sigma^2/N)(1 - N/N_pop) while the bootstrap CI inside each draw estimates
sigma^2/N with no finite-population correction (so N == N_pop reproduces the
verdict by construction), and the target verdict is an observed one, so a
significant population conditions the curve on an inflated effect. These
tests pin the separation of the two modes and the pointwise first-crossing
semantics.
"""

import json

import numpy as np
import pytest

from stats.contracts import DEFAULT_METRICS
import stats.cli.random_subsample_power as rsp


def _population(n=40, seed=5, effect=0.02):
    rng = np.random.default_rng(seed)
    scores_a, scores_b, weights, keys = [], [], [], []
    for i in range(n):
        latent = rng.normal(0.0, 0.05)
        a = {m: 0.5 + latent + rng.normal(0, 0.02) for m in DEFAULT_METRICS}
        b = {m: a[m] + effect + rng.normal(0, 0.02) for m in DEFAULT_METRICS}
        scores_a.append(a)
        scores_b.append(b)
        weights.append(20 + i)
        keys.append(f"{i:03d}_item{i}")
    return scores_a, scores_b, weights, keys


def _run(tmp_path, monkeypatch, *extra, n=40, effect=0.02):
    monkeypatch.setattr(rsp, "load_population",
                        lambda a, b: _population(n=n, effect=effect))
    out = tmp_path / "curve.md"
    js = tmp_path / "json-output" / "curve.json"
    rc = rsp.main(["--candidate-a", str(tmp_path), "--candidate-b", str(tmp_path),
                   "--sizes", "10", "20", "40", "--reps", "12",
                   "--bootstrap-iters", "800", "--seed", "3",
                   "--out", str(out), "--output-json", str(js), *extra])
    assert rc == 0
    return out.read_text(), json.loads(js.read_text())


# --------------------------------------------------------------------------- #
# Wilson lower bound
# --------------------------------------------------------------------------- #
def test_wilson_lower_is_below_the_point_estimate_and_bounded():
    for hits, n in ((0, 100), (50, 100), (80, 100), (100, 100), (1, 5)):
        lo = rsp.wilson_lower(hits, n)
        assert 0.0 <= lo <= hits / n + 1e-12
    assert rsp.wilson_lower(0, 0) == 0.0
    # even a perfect 100/100 does not certify 1.0
    assert rsp.wilson_lower(100, 100) < 1.0
    # and it tightens towards the point estimate as n grows
    assert rsp.wilson_lower(800, 1000) > rsp.wilson_lower(80, 100)


def test_wilson_lower_gates_the_first_crossing():
    """At --reps 100 the MC SE is ~5pp, so a point estimate of exactly 0.80
    must NOT certify a 0.80 target; the interval's lower end must."""
    assert 80 / 100 >= 0.80
    assert rsp.wilson_lower(80, 100) < 0.80
    assert rsp.wilson_lower(90, 100) >= 0.80


def test_first_crossing_is_labeled_pointwise_and_can_reverse():
    from scipy.stats import binom

    args = rsp.parse_args(['--candidate-a','A','--candidate-b','B','--mode','power','--out','unused.md'])
    key = ('kld','item_weighted')
    truth = {key:'A closer'}
    counts = {key:{10:{'A closer':90,'inconclusive':10},20:{'A closer':40,'inconclusive':60}}}
    effect = dict(metric='kld',weighting='item',estimate=.1,ci_lower=.01,ci_upper=.19,verdict='A closer')
    report, crossing = rsp._render_report(args,30,[10,20],truth,counts,effect)
    # Pointwise Wilson can cross on an unlucky binomial draw below the target.
    threshold = next(k for k in range(101) if rsp.wilson_lower(k,100) >= .8)
    assert binom.sf(threshold-1,100,.799) > .02
    assert crossing[key] == 10
    assert 'no simultaneous coverage' in report
    assert 'later evaluated sizes can fall below' in report
    assert 'lucky first crossing cannot' not in report


@pytest.mark.parametrize('option,value', [('--reps','0'),('--reps','-1'),('--seed','-1'),
    ('--power-target','nan'),('--power-target','0'),('--power-target','1'),
    ('--confidence-level','inf'),('--mc-confidence-level','nan'),
    ('--mc-confidence-level',str(np.nextafter(1.,0.)))])
def test_invalid_numeric_options_fail_before_loading(tmp_path, monkeypatch, option, value):
    def forbidden(*args):
        raise AssertionError('invalid options must not read collections')
    monkeypatch.setattr(rsp,'load_population',forbidden)
    with pytest.raises(SystemExit, match='must be'):
        rsp.main(['--candidate-a','A','--candidate-b','B','--out',str(tmp_path/'out.md'),option,value])


def test_nonfinite_mc_interval_cannot_silently_become_no_crossing(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("stats.power.binomtest", lambda *args: SimpleNamespace(proportion_ci=lambda **kwargs: (float("nan"), float("nan"))))
    with pytest.raises(ValueError, match="invalid Wilson"):
        rsp.wilson_lower(8,10)


def test_empty_eligible_size_grid_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rsp,'load_population',lambda *args:_population(n=3))
    with pytest.raises(SystemExit, match='no eligible sample sizes'):
        rsp.main(['--candidate-a','A','--candidate-b','B','--sizes','100','--out',str(tmp_path/'out.md')])


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
@pytest.mark.statistical
def test_reproducibility_mode_labels_itself_as_not_power(tmp_path, monkeypatch):
    md, payload = _run(tmp_path, monkeypatch, "--mode", "reproducibility")
    assert payload["mode"] == "reproducibility"
    assert payload["is_power_estimate"] is False
    assert "REPRODUCIBILITY curve" in md
    assert "not power" in md
    assert "This is not a required sample size" in md
    assert "without replacement" in md


@pytest.mark.statistical
def test_power_mode_resamples_with_replacement_and_may_exceed_the_pool(
        tmp_path, monkeypatch):
    """Drawing WITH replacement is what removes the finite-population bias --
    and it is also what lets the curve answer "what if I collected MORE
    rows", which the without-replacement sweep structurally cannot."""
    monkeypatch.setattr(rsp, "load_population", lambda a, b: _population(n=30))
    out, js = tmp_path / "p.md", tmp_path / "p.json"
    rc = rsp.main(["--candidate-a", str(tmp_path), "--candidate-b", str(tmp_path),
                   "--mode", "power", "--sizes", "20", "60", "--reps", "10",
                   "--bootstrap-iters", "800", "--seed", "3",
                   "--out", str(out), "--output-json", str(js)])
    assert rc == 0
    payload = json.loads(js.read_text())
    assert payload["is_power_estimate"] is True
    assert payload["power_scope"] == "conditional_on_observed_pilot_effect"
    assert payload["ci_method"] == "t"
    assert payload["bootstrap_iters_used"] == 0
    assert payload['verdict_scope'] == 'nominal_unadjusted_per_metric_ci_agreement'
    assert payload['mc_interval_method'] == 'wilson'
    assert payload['mc_confidence_level'] == .95
    assert payload['mc_interval_scope'] == 'pointwise_conditional_on_fixed_pilot'
    assert payload['power_target'] == .8
    assert payload["sizes"] == [20, 60]        # 60 > the 30-item pool
    assert all(key.endswith("/item_weighted") for key in payload["full_population_verdicts"])
    report = out.read_text()
    assert "WITH replacement" in report
    assert "paired Student-t" in report
    assert "paired bootstrap" not in report
    assert 'unadjusted per-metric CI verdict agreement' in report


def test_exploratory_nominal_target_does_not_pretend_to_apply_holm():
    import math
    residual = np.arange(10)-4.5
    shift = 2.3*residual.std(ddof=1)/math.sqrt(10)
    a = [{m:10. for m in DEFAULT_METRICS} for _ in residual]
    b = [{m:10.+(0 if m=='kld' else .1*shift)+.1*x for m in DEFAULT_METRICS} for x in residual]
    result = rsp.compare_items(a,b,[3]*10,metrics=DEFAULT_METRICS,confidence_level=.95,
                               bootstrap_iters=0,seed=1,model_a_label='A',model_b_label='B')
    block = result['metrics']['nll']['item_weighted']
    assert block['p_value'] < .05 < block['p_value_holm']
    assert rsp.verdicts_of(result)[('nll','item_weighted')] == 'A closer'


@pytest.mark.statistical
def test_power_mode_prints_the_pilot_effect_and_its_interval(tmp_path, monkeypatch):
    """Required N scales like 1/effect^2, so a power number computed from one
    pilot is only as good as that pilot's effect estimate. The report has to
    show the interval it is conditioning on."""
    md, payload = _run(tmp_path, monkeypatch, "--mode", "power")
    effect = payload["primary_effect"]
    assert effect["metric"] == "kld" and effect["weighting"] == "item"
    assert effect["ci_lower"] <= effect["estimate"] <= effect["ci_upper"]
    assert "conditions on the PILOT's effect estimate" in md
    assert "1/effect^2" in md


@pytest.mark.statistical
def test_reproducibility_at_full_population_is_saturated(tmp_path, monkeypatch):
    """The structural artefact the warning describes: at N == N_pop every draw
    IS the population, so the curve reads 100% whatever the effect."""
    md, _payload = _run(tmp_path, monkeypatch, "--mode", "reproducibility",
                        n=40, effect=0.0)
    rows = [r for r in md.splitlines() if r.startswith("| kld ")]
    assert rows, md
    assert rows[0].rstrip().endswith("100% |")


@pytest.mark.statistical
def test_null_verdict_rows_are_not_counted_as_detections(tmp_path, monkeypatch):
    md, _payload = _run(tmp_path, monkeypatch, "--mode", "power", effect=0.0)
    assert "never a detection rate" in md
    assert "inconclusive" in md


@pytest.mark.parametrize("loader", ["random", "variance"])
@pytest.mark.parametrize("failure", ["runtime", "execution", "missing", "rejected", "nonfinite", "reference", "budget"])
def test_legacy_loaders_reject_invalid_paired_collections(tmp_path, loader, failure):
    from test_saved_metrics_paired_compare import _make_pair
    import stats.cli.variance_decomposition as vd

    options = {"n_items": 4}
    if failure == "runtime":
        options["meta_b_overrides"] = {"tf_chunk": 32}
    elif failure == "missing":
        options["n_items_b"] = 3
    elif failure == "budget":
        options["skipped_a"] = (3,)
    a_dir, b_dir = _make_pair(tmp_path, **options)
    if failure == "execution":
        meta = json.loads((b_dir / "collect_meta.json").read_text())
        meta["execution_identity"]["unexpected_change"] = True
        (b_dir / "collect_meta.json").write_text(json.dumps(meta))
    elif failure == "rejected":
        (a_dir / "metrics" / "004_failed.rejected").write_text("failed")
    elif failure in ("nonfinite", "reference"):
        path = b_dir / "metrics" / "000_item0.npz"
        with np.load(path) as data:
            payload = {name: data[name].copy() for name in data.files}
        if failure == "nonfinite":
            payload["kld"][0] = np.nan
        else:
            payload["entropy_ref"][0] += 0.5
        np.savez(path, **payload)
    with pytest.raises(SystemExit):
        if loader == "random":
            rsp.load_population(a_dir, b_dir)
        else:
            vd.collect_deltas(a_dir, b_dir, "kld", -1)


@pytest.mark.parametrize("loader", ["random", "variance"])
def test_legacy_loaders_keep_every_valid_paired_item(tmp_path, loader):
    from test_saved_metrics_paired_compare import _make_pair
    import stats.cli.variance_decomposition as vd

    a_dir, b_dir = _make_pair(tmp_path, n_items=4)
    if loader == "random":
        a, b, weights, keys = rsp.load_population(a_dir, b_dir)
        assert len(a) == len(b) == len(weights) == len(keys) == 4
    else:
        deltas, weights, npos, _rho, dropped = vd.collect_deltas(a_dir, b_dir, "kld", -1)
        assert len(deltas) == len(weights) == len(npos) == 4
        assert dropped == 0
