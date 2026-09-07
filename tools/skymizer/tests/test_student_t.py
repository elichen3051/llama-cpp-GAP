"""stats/student_t.py -- the scipy-free Student-t distribution behind
--ci-method t. Pinned against closed forms that need no library (Cauchy,
df = 2) and, when scipy is installed, against scipy.stats.t across the df
and tail ranges a paired report can produce."""

import math

import pytest

from stats import student_t as S


# --------------------------------------------------------------------------- #
# closed forms (no scipy needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [0.0, 1e-8, 1e-3, 0.5, 1.0, 2.5, 30.0, 1e4, 1e9])
def test_two_sided_p_matches_the_cauchy_closed_form(t):
    """df = 1 is Cauchy: P(|T| >= t) = 1 - (2/pi) atan(t). Also the case
    where scipy's own 2*sf loses ~3e-9 at tiny t; the direct incomplete-beta
    form keeps 1 - p to full precision."""
    # (2/pi) atan(1/t) == 1 - (2/pi) atan(t) without the cancellation
    want = (2.0 / math.pi) * math.atan(1.0 / t) if t > 0.0 else 1.0
    assert S.t_two_sided_p(t, 1) == pytest.approx(want, rel=1e-13, abs=1e-300)
    assert S.t_two_sided_p(-t, 1) == S.t_two_sided_p(t, 1)


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 4.0, 100.0])
def test_two_sided_p_matches_the_df2_closed_form(t):
    """df = 2: P(|T| >= t) = 1 - t / sqrt(t^2 + 2), written in its
    rationalized form 2 / (r (r + t)), r = sqrt(t^2 + 2), so the oracle
    itself does not cancel at large t."""
    r = math.sqrt(t * t + 2.0)
    want = 2.0 / (r * (r + t))
    assert S.t_two_sided_p(t, 2) == pytest.approx(want, rel=1e-13, abs=1e-300)


@pytest.mark.parametrize("p", [0.6, 0.9, 0.975, 0.995, 1 - 1e-9])
def test_ppf_inverts_the_cauchy_closed_form(p):
    # cot(pi (1 - p)) == tan(pi (p - 1/2)); the former keeps the tail
    # precise (1 - p is what both sides actually resolve)
    want = 1.0 / math.tan(math.pi * (1.0 - p))
    assert S.t_ppf(p, 1) == pytest.approx(want, rel=1e-12)


def test_ppf_endpoints_symmetry_and_domain():
    assert S.t_ppf(0.5, 7) == 0.0
    for df in (1, 3, 24, 1000):
        assert S.t_ppf(0.975, df) == -S.t_ppf(0.025, df)      # exact, by construction
        assert S.t_ppf(0.975, df) > S.t_ppf(0.9, df) > 0.0
    # t exceeds the normal quantile and converges to it from above
    z = 1.959963984540054
    assert S.t_ppf(0.975, 25) > S.t_ppf(0.975, 250) > S.t_ppf(0.975, 1e6) > z
    assert S.t_ppf(0.975, 1e6) < z + 1e-5
    assert S.t_two_sided_p(0.0, 5) == 1.0
    assert S.t_two_sided_p(math.inf, 5) == 0.0
    assert math.isnan(S.t_two_sided_p(math.nan, 5))
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            S.t_ppf(bad, 5)
    with pytest.raises(ValueError):
        S.t_ppf(0.9, 0)
    with pytest.raises(ValueError):
        S.t_two_sided_p(1.0, 0)


def test_ppf_and_two_sided_p_are_mutual_inverses():
    for df in (1, 2, 5, 24, 99, 5000):
        for p in (0.51, 0.75, 0.975, 0.9995, 1 - 1e-7):
            t = S.t_ppf(p, df)
            assert S.t_two_sided_p(t, df) == pytest.approx(2.0 * (1.0 - p), rel=1e-11)
            assert S.t_cdf(t, df) == pytest.approx(p, rel=1e-12)


def test_incomplete_beta_reference_values():
    """I_x(a, b) against values that are exact by hand: I_x(1, 1) = x,
    I_x(1, b) = 1 - (1-x)^b, I_x(a, 1) = x^a, plus symmetry."""
    assert S.regularized_incomplete_beta(1.0, 1.0, 0.3) == pytest.approx(0.3, rel=1e-14)
    assert S.regularized_incomplete_beta(1.0, 4.0, 0.3) == pytest.approx(1.0 - 0.7 ** 4, rel=1e-14)
    assert S.regularized_incomplete_beta(3.0, 1.0, 0.3) == pytest.approx(0.3 ** 3, rel=1e-14)
    for a, b, x in ((0.5, 0.5, 0.2), (2.5, 7.0, 0.61), (30.0, 0.5, 0.97)):
        assert (S.regularized_incomplete_beta(a, b, x)
                + S.regularized_incomplete_beta(b, a, 1.0 - x)) == pytest.approx(1.0, rel=1e-13)
    assert S.regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert S.regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        S.regularized_incomplete_beta(0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        S.regularized_incomplete_beta(1.0, 1.0, 1.5)


# --------------------------------------------------------------------------- #
# scipy oracle (optional)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("df", [1, 2, 3, 5, 10, 24, 49, 99, 499, 1000])
def test_ppf_matches_scipy(df):
    st = pytest.importorskip("scipy.stats")
    for p in (0.5001, 0.6, 0.75, 0.9, 0.95, 0.975, 0.995, 0.9995, 1 - 1e-6, 1 - 1e-9):
        assert S.t_ppf(p, df) == pytest.approx(st.t.ppf(p, df), rel=1e-10), (df, p)


@pytest.mark.parametrize("df", [1, 2, 3, 5, 10, 24, 49, 99, 499, 1000])
def test_two_sided_p_matches_scipy(df):
    st = pytest.importorskip("scipy.stats")
    for t in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e4):
        want = 2.0 * float(st.t.sf(t, df))
        assert S.t_two_sided_p(t, df) == pytest.approx(want, rel=1e-10), (df, t)
