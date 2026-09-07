# Statistical API references and verification

Paired inference is restricted to item-weighted endpoints. Token-weighted means, differences and derived PPL/RMS values are descriptive only. Sampling units, pairing, provenance, valid inputs, primary endpoints and multiplicity families remain project responsibilities. Each replacement's function docstring includes the API reference used by that implementation.

## Runtime APIs

| Calculation | Python implementation | Official reference |
|---|---|---|
| Student-t quantile | `stats.student_t.t_ppf` calls `scipy.stats.t.ppf(p, df)` | [SciPy Student-t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) |
| Two-sided Student-t tail | `stats.student_t.t_two_sided_p` calls `2 * scipy.stats.t.sf(abs(t), df)` | [SciPy Student-t survival function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) |
| Item paired t-test and CI | `stats.inference._student_t_delta` calls `scipy.stats.ttest_rel(d, zeros, alternative="two-sided", nan_policy="raise")`, then `confidence_interval(confidence_level)`, where `d = B - A` | [SciPy paired t-test and result API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) |
| Monte Carlo proportion interval | `stats.power.wilson_interval` calls `scipy.stats.binomtest(hits, total).proportion_ci(confidence_level=..., method="wilson")` | [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [proportion_ci](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html) |

The power planner's scalar interval reuses the production paired-test helper. Its batched simulation retains vectorized item means and standard errors, using SciPy t quantiles. The legacy variance planner also uses the shared SciPy quantile wrapper; its sample-size search remains a documented approximation, not exact noncentral-t power inversion. The legacy subsampling tool reuses the Wilson helper's lower endpoint.

`method="wilson"` is explicit: SciPy's default proportion interval is Clopper-Pearson, a different method. The reported Monte Carlo lower bound is the lower endpoint of a two-sided interval, not a one-sided interval at the same confidence level.

## MATLAB correspondence

The runtime calls SciPy; it does not copy MATLAB source. These official MATLAB pages specify the corresponding mathematical operations and supply independent published reference values:

| MATLAB operation | Python correspondence | Official MATLAB reference |
|---|---|---|
| `ttest(B, A, 'Alpha', 1-confidence_level)` | Paired differences `B-A`, with `df = n_items - 1`; SciPy `ttest_rel` returns the same kind of two-sided test and mean-difference CI | [ttest](https://www.mathworks.com/help/stats/ttest.html) |
| `tinv(p, df)` | SciPy `t.ppf(p, df)` | [tinv](https://www.mathworks.com/help/stats/tinv.html) |
| `2 * tcdf(abs(t), df, 'upper')` | SciPy `2 * t.sf(abs(t), df)` | [tcdf](https://www.mathworks.com/help/stats/tcdf.html) |

The `tinv` page publishes `tinv(0.99, 1:6)` to four decimal places. `tests/test_student_t.py` checks those values with an absolute tolerance of half the displayed final decimal unit. This is a check against published MATLAB results, not a claim that a MATLAB runtime was executed. No MATLAB-specific algorithm is ported in this change.

## Julia correspondence

[Distributions.jl](https://juliastats.org/Distributions.jl/latest/univariate/) provides `TDist(df)`, `quantile(TDist(df), p)` and `ccdf(TDist(df), t)`. These correspond to SciPy's Student-t distribution, quantile and survival function. [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/parametric/#t-test) provides the paired test as `OneSampleTTest(B, A)`, with p-values and confidence intervals. These are API correspondences checked against the official documentation; no Julia runtime was executed.

## Independent checks and retained guards

[Wolfram StudentTDistribution](https://reference.wolfram.com/language/ref/StudentTDistribution.html) identifies `df=1` with the standard Cauchy distribution. [Wolfram MathWorld's Student t distribution](https://mathworld.wolfram.com/Studentst-Distribution.html) supplies the distribution formulas. The tests evaluate their elementary special cases in Python, without using the production t implementation to generate expected values:

- For `df=1` and positive `t`, the two-sided probability is `2/pi * atan(1/t)` and the upper quantile is `cot(pi*(1-p))`.
- For `df=2`, the two-sided probability is `2 / (r*(r+t))`, where `r=sqrt(t*t+2)`. The quantile is `(2*p-1)/sqrt(2*p*(1-p))`.
- Paired differences `[0, 1, 2]` have mean 1, SE `1/sqrt(3)`, df 2 and two-sided p-value `1-sqrt(3/5)`. Tests verify both signs and several confidence levels through the production entry point.
- Wilson endpoints solve the binomial score equation `n*(observed_proportion-bound)^2 = z^2*bound*(1-bound)`. Tests use `z=2`, including zero successes, all successes and asymmetric counts.
- A seeded normal-null simulation checks the production paired test's rejection frequency. This checks the normal model; it does not establish coverage for skewed or dependent real data.

The finite-value, variance overflow/underflow and zero-spread guards remain active before calling SciPy. Nonconstant samples with subnormal `SE^2` also fail: SciPy's variance-then-division order can lose precision or produce a zero SE even when NumPy's std-then-division order returns a positive SE. Scalar inference and batched power simulations enforce the same threshold. A nonfinite SciPy t statistic fails explicitly. Identical differences retain the explicit project policy: a point CI, p=1 for zero differences or p=0 otherwise, with a recorded zero-spread note. This limiting convention is not evidence that a small constant sample establishes a population effect with certainty.

SciPy 1.18.1 has about `3.1e-9` absolute two-sided tail error at `df=1, t=1e-8`, where p is near 1. One narrowly scoped Cauchy test bounds this error by `4e-9`; the remaining tail checks stay strict. Quantile tests use `1e-10` relative tolerance against closed forms. Floating-point output can therefore differ from the former handwritten implementation, so full-engine goldens are updated only after these independent checks. No Wolfram runtime was executed.

The checks do not certify the entire distribution API domain. Production paired tests use integer `df = n_items - 1 >= 1`. At extreme statistics, SciPy can still lose representable tails: for `df=1, t=1e160`, the two-sided API returns 0 although the Cauchy formula is about `6.37e-161`. At extremely small fractional df its quantile can also be inaccurate. These package limits must not be interpreted as exact mathematical zero or exhaustive numerical validation.

The optional BCa, percentile and studentized bootstrap algorithms, their p-value construction, and Holm adjustment remain project code in this migration. Their numerical and statistical assumptions require separate review; SciPy-backed t tests do not validate them indirectly.
