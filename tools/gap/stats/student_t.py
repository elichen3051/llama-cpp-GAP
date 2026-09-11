"""Validated scalar access to SciPy's Student-t distribution."""

import math

from scipy.stats import t as student_t


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile for finite positive df and an interior probability.

    Uses scipy.stats.t.ppf(p, df): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    MATLAB tinv(p, df): https://www.mathworks.com/help/stats/tinv.html
    Julia quantile(TDist(df), p): https://juliastats.org/Distributions.jl/latest/univariate/
    """
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError(f"t distribution needs finite df > 0; got {df}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf needs 0 < p < 1; got {p}")
    if p == 0.5:
        return 0.0
    return float(student_t.ppf(p, float(df)))


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided tail; sf avoids subtracting a nearly unit CDF.

    Uses 2*t.sf(abs(t), df): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    MATLAB 2*tcdf(abs(t), df, 'upper'): https://www.mathworks.com/help/stats/tcdf.html
    Julia 2*ccdf(TDist(df), abs(t)): https://juliastats.org/Distributions.jl/latest/univariate/
    Independent closed-form checks: https://mathworld.wolfram.com/Studentst-Distribution.html
    """
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError(f"t distribution needs finite df > 0; got {df}")
    return float(2.0 * student_t.sf(abs(t), float(df)))
