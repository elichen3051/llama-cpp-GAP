"""SciPy-backed t distribution checked against independent formulas and published values."""

import math

import pytest

from stats import student_t as S


@pytest.mark.parametrize("t", [0.0, 1e-3, 0.5, 1.0, 2.5, 30.0, 1e4, 1e9])
def test_two_sided_p_matches_the_cauchy_closed_form(t):
    """df=1 is Cauchy: https://reference.wolfram.com/language/ref/StudentTDistribution.html"""
    want = (2.0 / math.pi) * math.atan(1.0 / t) if t > 0.0 else 1.0
    assert S.t_two_sided_p(t, 1) == pytest.approx(want, rel=1e-13, abs=1e-300)
    assert S.t_two_sided_p(-t, 1) == S.t_two_sided_p(t, 1)


def test_tiny_cauchy_statistic_has_bounded_scipy_tail_error():
    # SciPy 1.18.1 loses about 3.1e-9 here; other tail checks remain strict.
    t = 1e-8
    want = 2.0 / math.pi * math.atan(1.0 / t)
    assert S.t_two_sided_p(t, 1) == pytest.approx(want, rel=0, abs=4e-9)


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 4.0, 100.0])
def test_two_sided_p_matches_the_df2_closed_form(t):
    """Elementary df=2 tail: https://mathworld.wolfram.com/Studentst-Distribution.html"""
    r = math.sqrt(t * t + 2.0)
    want = 2.0 / (r * (r + t))
    assert S.t_two_sided_p(t, 2) == pytest.approx(want, rel=1e-13, abs=1e-300)


@pytest.mark.parametrize("p", [1e-20, 0.025, 0.6, 0.9, 0.975, 0.995, 1 - 1e-9])
def test_ppf_inverts_the_cauchy_closed_form(p):
    want = -1.0 / math.tan(math.pi * p) if p < 0.5 else 1.0 / math.tan(math.pi * (1.0 - p))
    assert S.t_ppf(p, 1) == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("p", [1e-20, 0.025, 0.6, 0.975, 1 - 1e-9])
def test_ppf_matches_the_df2_closed_form(p):
    want = (2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p))
    assert S.t_ppf(p, 2) == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("df,want", [(1, 31.8205), (2, 6.9646), (3, 4.5407),
                                     (4, 3.7469), (5, 3.3649), (6, 3.1427)])
def test_ppf_matches_matlab_published_example(df, want):
    """Published tinv(0.99, 1:6), four decimal places: https://www.mathworks.com/help/stats/tinv.html"""
    assert S.t_ppf(0.99, df) == pytest.approx(want, rel=0, abs=5e-5)


def test_ppf_endpoints_symmetry_and_domain():
    assert S.t_ppf(0.5, 7) == 0.0
    for df in (1, 3, 24, 1000):
        assert S.t_ppf(0.975, df) == pytest.approx(-S.t_ppf(0.025, df), rel=1e-13)
        assert S.t_ppf(0.975, df) > S.t_ppf(0.9, df) > 0.0
    z = 1.959963984540054
    assert S.t_ppf(0.975, 25) > S.t_ppf(0.975, 250) > S.t_ppf(0.975, 1e6) > z
    assert S.t_ppf(0.975, 1e6) < z + 1e-5
    assert S.t_two_sided_p(0.0, 5) == 1.0
    assert S.t_two_sided_p(math.inf, 5) == 0.0
    assert math.isnan(S.t_two_sided_p(math.nan, 5))
    for bad in (0.0, 1.0, -0.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            S.t_ppf(bad, 5)
    for df in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError):
            S.t_ppf(0.9, df)
        with pytest.raises(ValueError):
            S.t_two_sided_p(1.0, df)
