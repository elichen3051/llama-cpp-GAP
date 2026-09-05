# compare/student_t.py -- Student's t distribution, exactly, with no scipy.
#
# The classical paired t interval (--ci-method t) needs two things the
# standard library does not have: the t quantile for the endpoints and the
# two-sided tail probability for the p-value. Both come from the regularized
# incomplete beta function I_x(a, b):
#
#     P(|T_nu| >= t) = I_x(nu/2, 1/2),   x = nu / (nu + t^2)
#
# which gives the two-sided p DIRECTLY, with no 1 - F(t) cancellation in the
# far tail. The quantile inverts the CDF by a bracketed Newton iteration to
# machine precision, so a report at --ci-method t is a deterministic function
# of the per-item deltas alone (no seed, no replicate count).
#
# I_x is evaluated by Lentz's modified continued fraction (Numerical Recipes
# `betacf`) after the usual symmetry switch, accurate to ~1e-15 for the
# arguments this tool produces (nu >= 1, 0 <= x <= 1). scipy, when present,
# is used by the tests as the oracle, never by the engine.
import math
from statistics import NormalDist

_EPS = 3e-16
_FPMIN = 1e-300
_MAX_ITER = 2000


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for I_x(a, b) (modified Lentz), valid when
    x < (a + 1) / (a + b + 2); the caller applies the symmetry otherwise."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float,
                                one_minus_x: float | None = None) -> float:
    """I_x(a, b) for a, b > 0 and 0 <= x <= 1.

    `one_minus_x` lets a caller that KNOWS 1 - x to full precision (the t
    tail passes x = nu/(nu+t^2) and 1 - x = t^2/(nu+t^2), both formed
    directly) avoid the rounding of `1.0 - x` when x is within a few ulp of
    1 -- that rounding alone cost ~1e-8 relative in the p-value of a tiny t."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"incomplete beta needs a, b > 0; got a={a}, b={b}")
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"incomplete beta needs 0 <= x <= 1; got x={x}")
    y = (1.0 - x) if one_minus_x is None else one_minus_x
    # Only the side formed directly decides the exact endpoints: x = 1e-16
    # with y rounded to 1.0 is a genuine (tiny) tail, not zero.
    if x == 0.0:
        return 0.0
    if y == 0.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(y))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, y) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T_df| >= |t|): the two-sided p-value of a t statistic, computed
    from the incomplete beta directly so the far tail keeps its precision
    (no 2 * (1 - cdf) cancellation). t = +-inf gives 0.0; t = 0 gives 1.0."""
    if df <= 0.0:
        raise ValueError(f"t distribution needs df > 0; got {df}")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    if t == 0.0:
        return 1.0
    denom = df + t * t
    return regularized_incomplete_beta(df / 2.0, 0.5, df / denom,
                                       one_minus_x=(t * t) / denom)


def t_cdf(t: float, df: float) -> float:
    """P(T_df <= t)."""
    p = t_two_sided_p(t, df)
    return 1.0 - 0.5 * p if t >= 0.0 else 0.5 * p


def t_pdf(t: float, df: float) -> float:
    return math.exp(math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
                    - 0.5 * math.log(df * math.pi)
                    - (df + 1.0) / 2.0 * math.log1p(t * t / df))


def t_ppf(p: float, df: float) -> float:
    """Quantile: the t with P(T_df <= t) = p, to machine precision.

    Solved in the TAIL domain -- find t > 0 with P(|T| >= t) = 2 (1 - p) --
    so a quantile deep in the tail (p = 1 - 1e-9) is as precise as one near
    the centre; solving cdf(t) = p there would cancel to ~1e-7. Starts from
    the Cornish-Fisher expansion around the normal quantile (Hill 1970) and
    polishes with Newton steps kept inside a bracket that is bisected
    whenever Newton would leave it. Symmetric by construction:
    t_ppf(p) == -t_ppf(1 - p) exactly, and t_ppf(0.5) == 0.0."""
    if df <= 0.0:
        raise ValueError(f"t distribution needs df > 0; got {df}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"t_ppf needs 0 < p < 1; got {p}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    target = 2.0 * (1.0 - p)              # two-sided tail mass beyond +-t
    z = NormalDist().inv_cdf(p)
    z2 = z * z
    g1 = (z2 + 1.0) * z / 4.0
    g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0
    g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0
    g4 = ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2
          - 945.0) * z / 92160.0
    t = z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4
    if not math.isfinite(t) or t <= 0.0:
        t = z
    # Bracket [lo, hi] with tail(lo) >= target >= tail(hi); tail(0) = 1.
    lo, hi = 0.0, max(t, 1.0)
    while t_two_sided_p(hi, df) > target:
        hi *= 2.0
        if hi > 1e300:
            return float("inf")
    t = min(max(t, lo), hi)
    for _ in range(200):
        f = t_two_sided_p(t, df) - target     # decreasing in t
        if f == 0.0:
            return t
        if f > 0.0:
            lo = t
        else:
            hi = t
        nxt = t + f / (2.0 * t_pdf(t, df))
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - t) <= 4.0 * _EPS * max(1.0, abs(t)):
            return nxt
        t = nxt
    return t
