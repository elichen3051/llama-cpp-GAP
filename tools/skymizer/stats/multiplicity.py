"""Package-backed corrections over a declared hypothesis family."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def adjust_pvalues(p_values, method="holm"):
    """Return Holm, BH or BY adjusted p-values in input order.

    Holm controls strong FWER under arbitrary dependence. BH controls FDR under independence or PRDS; BY permits arbitrary dependence. All require valid marginal p-values and the full declared family. This wrapper uses strict p < alpha elsewhere for closed-CI boundary consistency.
    API: https://www.statsmodels.org/v0.14.6/generated/statsmodels.stats.multitest.multipletests.html
    Holm (1979): https://www.jstor.org/stable/4615733
    Benjamini and Hochberg (1995): https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
    Benjamini and Yekutieli (2001): https://doi.org/10.1214/aos/1013699998
    SciPy BH/BY correspondence: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
    MATLAB BH correspondence requires mafdr(p, 'BHFDR', true); its default is a different method: https://www.mathworks.com/help/bioinfo/ref/mafdr.html
    Julia adjust(p, Holm()/BenjaminiHochberg()/BenjaminiYekutieli()): https://juliangehring.github.io/MultipleTesting.jl/stable/adjustment/
    """
    if method not in ("holm", "fdr_bh", "fdr_by"):
        raise ValueError("method must be holm, fdr_bh or fdr_by")
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be 1-D, finite and in [0, 1]")
    if not p.size:
        return []
    return multipletests(p, method=method, is_sorted=False, returnsorted=False)[1].tolist()
