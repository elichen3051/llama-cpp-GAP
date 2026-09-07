"""The subsample sweep: reproducibility mode vs power mode.

The original tool drew subsets WITHOUT replacement from the same finite pool
whose full-sample verdict was the target, and called the result "power". Two
biases compound there: a size-N draw's mean carries variance
(sigma^2/N)(1 - N/N_pop) while the bootstrap CI inside each draw estimates
sigma^2/N with no finite-population correction (so N == N_pop reproduces the
verdict by construction), and the target verdict is an observed one, so a
significant population conditions the curve on an inflated effect. These
tests pin the separation of the two modes and the conservative first
crossing.
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
    js = tmp_path / "curve.json"
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
    assert payload["sizes"] == [20, 60]        # 60 > the 30-item pool
    assert all(key.endswith("/item_weighted") for key in payload["full_population_verdicts"])
    report = out.read_text()
    assert "WITH replacement" in report
    assert "paired Student-t" in report
    assert "paired bootstrap" not in report


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
