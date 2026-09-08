#!/usr/bin/env python3
"""Describe paired item effects and variances across saved token prefixes.

The item is the sampling unit. Each cap recomputes the per-item mean delta
from stored positions, without assuming a 1/K variance law. Longer prefixes
can change both the effect and its variance.

mean_i(s_i^2/T_i) is an IID residual-variance proxy, not an observed variance
component. Its interpretation requires independent residuals with a common
conditional mean; token covariance and position profiles invalidate it. The
raw residual Var(d_i) - proxy is retained, including negative values. A
nonpositive residual suppresses the conditional infinite-length floor and
sqrt-rule cost extrapolation. Positive residuals do not validate the model.

SNR intervals invert the noncentral-t distribution under IID normal item
deltas. N_obs uses exact two-sided t power under that same normal model and
plugs in the observed standardized effect. These are conditional diagnostics;
use power_analysis.py with an externally chosen SESOI for prospective planning.

Example:
    python3 stats/cli/variance_decomposition.py --candidate-a A/ --candidate-b B/ \
        --metric kld --num-eval-tokens 512 --output-json variance.json
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stats import collection_io as paired_io                 # noqa: E402
from stats.student_t import t_ppf                           # noqa: E402

# metric name -> per-token delta column extractor (B minus A)
METRIC_COLUMNS = {
    name: (lambda ma, mb, column=column: np.asarray(mb[column], dtype=np.float64)
           - np.asarray(ma[column], dtype=np.float64))
    for name, column in (("kld", "kld"), ("reversed_kld", "reversed_kld"),
                         ("js_kld", "js_kld"), ("nll", "nll_cand"))
}

# Cap ladder for the empirical curve (backtest convention; per item the cap
# acts as min(K, T_i)). The analysis cap / full stored length is appended.
CAP_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Empirical var/effect-vs-cap curve + sigma_b^2 / sigma_w^2 "
                    "decomposition of paired per-token metric deltas "
                    "(decide: more eval tokens vs more items).")
    p.add_argument("--candidate-a", required=True, type=Path,
                   help="collect_kld.py dir of the ref-vs-A run")
    p.add_argument("--candidate-b", required=True, type=Path,
                   help="collect_kld.py dir of the ref-vs-B run (same reference)")
    p.add_argument("--metric", default="kld", choices=sorted(METRIC_COLUMNS),
                   help="per-token metric to decompose (default: kld)")
    p.add_argument("--num-eval-tokens", type=int, default=-1,
                   help="analysis cap: use only the first min(K, npos) "
                        "positions per item; -1 = all stored (default)")
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--power-target", type=float, default=0.80)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def collect_deltas(a_dir: Path, b_dir: Path, metric: str, cap: int):
    """Load every paired delta array and reject rows unsuitable for variance analysis."""
    try:
        with paired_io.comparison_locks((a_dir, b_dir)):
            paired_io.validate_collection_pair(a_dir, b_dir)
            matched, drops = paired_io.aligned_metric_items(a_dir, b_dir)
            if not matched:
                sys.exit("no matched items between the two dirs")
            delta_fn = METRIC_COLUMNS[metric]
            deltas, T, npos_all, rho1 = [], [], [], []
            for key in matched:
                ma, ha, mb, _hb = paired_io.load_item_pair(key, a_dir, b_dir)
                _sa, _sb, _ta, _tb, keep, finite, drift, _versions = paired_io.score_records(
                    key, ma, ha, mb, _hb, cap)
                if drift is not None:
                    sys.exit(f"reference drift on {key}: {drift['msg']}")
                if not finite:
                    sys.exit(f"{key}: non-finite metric score; variance analysis aborted")
                if keep < 2:
                    sys.exit(f"{key}: variance analysis requires at least two scored positions")
                dt = delta_fn(ma, mb)[:keep].astype(np.float64)
                if not np.all(np.isfinite(dt)):
                    sys.exit(f"{key}: non-finite paired differences; variance analysis aborted")
                deltas.append(dt)
                T.append(keep)
                npos_all.append(ha["npos"])
                if keep >= 3 and dt.std() > 0:
                    rho1.append(float(np.corrcoef(dt[:-1], dt[1:])[0, 1]))
            if len(deltas) < 3:
                sys.exit(f"need >= 3 usable items, got {len(deltas)}")
            return (deltas, np.array(T), np.array(npos_all),
                    float(np.mean(rho1)) if rho1 else float("nan"), 0)
    except ValueError as error:
        sys.exit(str(error))


def required_n(snr: float, confidence_level: float, power_target: float):
    """Smallest integer n >= 2 for two-sided t power under IID normal deltas.

    This plugs in the observed standardized effect; it is not prospective power.
    Return None for undefined effects or counts beyond exact float64 integers.
    Uses scipy.stats.t.isf and scipy.stats.nct.cdf/sf:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    """
    if not (0.0 < confidence_level < 1.0 and 0.0 < power_target < 1.0):
        raise ValueError("confidence_level and power_target must be in (0, 1)")
    if not np.isfinite(snr) or snr < 0.0:
        return None
    alpha = 1.0 - confidence_level
    if power_target <= alpha:
        return 2
    if snr == 0.0:
        return None

    def sufficient(n):
        df = float(n - 1)
        critical = stats.t.isf(alpha / 2.0, df)
        noncentrality = snr * math.sqrt(n)
        with warnings.catch_warnings(record=True) as numerical_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            right = float(stats.nct.sf(critical, df, noncentrality))
            wrong = float(stats.nct.cdf(-critical, df, noncentrality))
        if any(issubclass(w.category, RuntimeWarning) for w in numerical_warnings):
            return None
        if not math.isfinite(right) or not 0.0 <= right <= 1.0:
            return None
        # T < -critical requires a negative normal numerator. This bound
        # resolves decisions when SciPy loses the much smaller wrong-sign tail.
        wrong_bound = float(stats.norm.sf(noncentrality))
        if math.isfinite(wrong) and 0.0 <= wrong <= wrong_bound:
            return min(1.0, right + wrong) >= power_target
        if right >= power_target:
            return True
        if min(1.0, right + wrong_bound) < power_target:
            return False
        return None

    low, high = 1, 2
    while True:
        enough = sufficient(high)
        if enough is None:
            return None
        if enough:
            break
        if high >= 2**53:
            return None
        low = high
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        enough = sufficient(middle)
        if enough is None:
            return None
        if enough:
            high = middle
        else:
            low = middle
    return high


def snr_interval(snr: float, n: int, confidence_level: float):
    """Normal-model interval for |mu|/sigma from noncentral-t inversion.

    Invert the signed noncentrality, then map its interval through absolute value.
    Return undefined bounds if the roots or their tail residuals cannot be verified.
    Uses scipy.stats.nct.cdf/sf and scipy.optimize.brentq:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    undefined = (float("nan"), float("nan"))
    if not np.isfinite(snr) or snr < 0.0 or not isinstance(n, (int, np.integer)) or not 2 <= n <= 2**53:
        return undefined
    scale = math.sqrt(n)
    observed_t = snr * scale
    if not math.isfinite(observed_t):
        return undefined
    tail_probability = (1.0 - confidence_level) / 2.0
    roots = []
    for tail, increasing in ((stats.nct.sf, True), (stats.nct.cdf, False)):
        def probability(noncentrality):
            with warnings.catch_warnings(record=True) as numerical_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                value = float(tail(observed_t, n - 1, noncentrality))
            if any(issubclass(w.category, RuntimeWarning) for w in numerical_warnings):
                return float("nan")
            return value if math.isfinite(value) and 0.0 <= value <= 1.0 else float("nan")

        def residual(noncentrality):
            return probability(noncentrality) - tail_probability

        center = observed_t
        at_center = residual(center)
        if not math.isfinite(at_center):
            return undefined
        direction = -1.0 if (at_center > 0.0) == increasing else 1.0
        step = max(1.0, observed_t / math.sqrt(2.0 * (n - 1)))
        for _ in range(64):
            endpoint = center + direction * step
            if not math.isfinite(endpoint):
                return undefined
            at_endpoint = residual(endpoint)
            if not math.isfinite(at_endpoint):
                return undefined
            if at_center == 0.0 or at_endpoint == 0.0 or (at_center < 0.0) != (at_endpoint < 0.0):
                break
            step *= 2.0
        else:
            return undefined
        try:
            root = optimize.brentq(residual, min(center, endpoint), max(center, endpoint), xtol=1e-12,
                                   rtol=8*np.finfo(float).eps, maxiter=256)
        except (ValueError, RuntimeError):
            return undefined
        if not math.isclose(probability(root), tail_probability,
                            rel_tol=1e-7, abs_tol=np.finfo(float).tiny):
            return undefined
        roots.append(root / scale)
    lower, upper = roots
    if lower > upper:
        return undefined
    return (0.0 if lower <= 0.0 <= upper else min(abs(lower), abs(upper)),
            max(abs(lower), abs(upper)))


def cap_ladder(T: np.ndarray):
    """The caps to evaluate: ladder rungs strictly below the longest item,
    then the full (analysis-capped) length."""
    caps = [K for K in CAP_LADDER if K < int(T.max())]
    caps.append(int(T.max()))
    return caps


def build_curve(deltas, T, s2b, s2w, caps, confidence_level, power_target):
    """The empirical design table: one row per cap K.

    mu/sd/d/N_power come straight from the K-truncated data (no 1/K
    assumption). log2(pred/emp) checks the analytic law
    pred = sigma_b^2 + mean_i(s2w_i / min(K, T_i)) against the empirical
    variance — the 2026-08-26 backtest's F1 metric (negative = the law
    underestimates the real variance at this cap). noise_share is the
    cap-LOCAL decomposition mean_i(s_i^2(K)/K_i)/emp. Correlated residuals
    or changing positional means invalidate its interpretation as noise."""
    rows = []
    for K in caps:
        dK = np.array([d[: min(K, len(d))].mean() for d in deltas])
        kept = np.minimum(K, T)
        emp = float(dK.var(ddof=1))
        mu = float(dK.mean())
        s2_local = np.array([d[: min(K, len(d))].var(ddof=1) for d in deltas])
        noise_local = float(np.mean(s2_local / kept))
        pred = float(s2b + np.mean(s2w / kept))
        snr = abs(mu) / math.sqrt(emp) if emp > 0 else (float("inf") if mu != 0 else float("nan"))
        rows.append({
            "K": int(K),
            "mu": mu,
            "sd": float(math.sqrt(emp)),
            "snr": float(snr),
            "n_required": required_n(snr, confidence_level, power_target),
            "noise_share_local": (min(1.0, noise_local / emp)
                                  if emp > 0 else float("nan")),
            "noise_share_local_raw": noise_local / emp if emp > 0 else float("nan"),
            "iid_variance_proxy": noise_local,
            "variance_residual_raw": emp - noise_local,
            "log2_pred_over_emp": (float(math.log2(pred / emp))
                                   if emp > 0 and pred > 0 else float("nan")),
        })
        if emp == 0.0:
            rows[-1]["snr_status"] = "zero_effect_zero_variance" if mu == 0.0 else "zero_variance_nonzero_effect"
    return rows


def sign_flip_caps(rows):
    """Caps whose mu(K) sign disagrees with the full-cap mu. K = 16 alone is
    reported but noted as likely small-sample wobble (the backtest's two
    K=16-only flips did not replicate; its one real flip crossed at
    t ~ 20-40 and replicated across independent item sets)."""
    full_sign = np.sign(rows[-1]["mu"])
    if full_sign == 0:
        return []
    return [r["K"] for r in rows[:-1]
            if r["mu"] != 0 and np.sign(r["mu"]) != full_sign]


def cost_model_from_manifest(manifest_path: Path):
    """wall_s ~ a + b*n_eval (+ c*n_prefill) from OK manifest rows, or None.
    Returns dict(cfix, ctok, r2, n_rows). cfix folds the mean-prefill term.
    Uses numpy.linalg.lstsq: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """
    import csv
    if not manifest_path.exists():
        return None
    rows = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() != "OK":
                continue
            try:
                rows.append((float(r["n_eval"]), float(r.get("n_prefill", 0.0)),
                             float(r["wall_s"])))
            except (KeyError, TypeError, ValueError):
                return None
    if len(rows) < 3:
        return None
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)) or np.any(data[:, 0] <= 0) or np.any(data[:, 1:] < 0):
        return None
    if np.any(data[:, :2] != np.floor(data[:, :2])):
        return None
    varied_prefill = bool(np.any(data[:, 1] != data[0, 1]))
    M = np.column_stack((np.ones(len(rows)), data[:, 0]))
    if varied_prefill:
        M = np.column_stack((M, data[:, 1]))
    y = data[:, 2]
    try:
        coef, _, rank, _ = np.linalg.lstsq(M, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank != M.shape[1]:
        return None
    resid = M @ coef - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if not np.all(np.isfinite(resid)) or not math.isfinite(ss_tot) or ss_tot <= 0:
        return None
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
    if not math.isfinite(r2):
        return None
    cfix = float(coef[0] + (coef[2] * data[:, 1].mean() if varied_prefill else 0.0))
    ctok = float(coef[1])
    if not (np.isfinite(cfix) and np.isfinite(ctok)) or cfix < 0 or ctok <= 0:
        return None
    if varied_prefill and coef[2] < 0:
        return None
    return {"cfix": cfix, "ctok": ctok, "r2": r2, "n_rows": len(rows)}


def _planning_hint(mu):
    if mu == 0.0:
        return "No observed mean difference; this pilot does not size a nonzero effect."
    return ("Use the empirical cap curve as a pilot diagnostic. Allocation needs a "
            "prespecified effect and a justified dependence model; the IID variance "
            "proxy does not establish whether more tokens or more items help.")


def _print_cap_ladder(curve, worst, flips, power_target):
    print()
    print("empirical cap ladder (design table — nothing here assumes the 1/K law):")
    npow = f"Nobs_{int(power_target*100)}"
    print(f"  {'K':>5}  {'mu(K)':>12}  {'sd(K)':>10}  {'d(K)':>8}  "
          f"{npow + '(K)':>8}  {'IIDproxy%':>9}  {'log2(pred/emp)':>14}")
    for r in curve:
        nreq = "n/a" if r["n_required"] is None else str(r["n_required"])
        share = ("   n/a" if not np.isfinite(r["noise_share_local_raw"])
                 else f"{100*r['noise_share_local_raw']:8.1f}%")
        fit = ("   n/a" if not np.isfinite(r["log2_pred_over_emp"])
               else f"{r['log2_pred_over_emp']:+14.2f}")
        print(f"  {r['K']:>5}  {r['mu']:>+12.4e}  {r['sd']:>10.3e}  "
              f"{(r['mu']/r['sd'] if r['sd'] > 0 else math.copysign(r['snr'], r['mu'])):>+8.3f}  "
              f"{nreq:>8}  {share}  {fit}")
    if worst is not None and abs(worst["log2_pred_over_emp"]) > 0.3:
        fit = worst["log2_pred_over_emp"]
        direction = "underestimates" if fit < 0 else "overestimates"
        print(f"  model check: the IID proxy law {direction} empirical variance by "
              f"up to {2**abs(fit):.2f}x (worst at K={worst['K']}). Dependence or "
              "position-dependent means/variances can cause this mismatch; its sign "
              "does not identify the cause.")
    if flips:
        wobble = " (K=16 alone is often small-sample wobble)" if flips == [16] else ""
        print(f"  WARNING: mu(K) changes SIGN vs the full cap at K={flips}{wobble} — "
              "the cap is an estimator choice here, not a precision knob. "
              "Pre-register the verdict cap and inspect the position-resolved "
              "delta profile before trusting any single-cap verdict "
              "(backtest-2026-08-26 F7/F9).")
    else:
        print("  sign(mu(K)) stable across all caps.")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.num_eval_tokens != -1 and args.num_eval_tokens < 1:
        sys.exit(f"--num-eval-tokens must be -1 (all) or >= 1, got {args.num_eval_tokens}")
    if not (0.0 < args.confidence_level < 1.0):
        sys.exit(f"--confidence-level must be in (0, 1), got {args.confidence_level}")
    if not (0.0 < args.power_target < 1.0):
        sys.exit(f"--power-target must be in (0, 1), got {args.power_target}")
    for flag, dirpath in (("--candidate-a", args.candidate_a),
                          ("--candidate-b", args.candidate_b)):
        if not (dirpath / "metrics").is_dir():
            sys.exit(f"{flag}: {dirpath} has no metrics/ subdir — not a "
                     "collect_kld.py output dir?")

    deltas, T, npos_all, rho1, dropped = collect_deltas(
        args.candidate_a, args.candidate_b, args.metric, args.num_eval_tokens)

    n = len(deltas)
    d_full = np.array([d.mean() for d in deltas])
    s2w = np.array([d.var(ddof=1) for d in deltas])
    mu = float(d_full.mean())
    var_total = float(d_full.var(ddof=1))
    var_token = float((s2w / T).mean())
    var_between_raw = var_total - var_token
    var_between = max(var_between_raw, 0.0)
    clamped = bool(var_between_raw < 0.0)
    model_extrapolation_valid = bool(math.isfinite(var_between_raw) and var_between_raw > 0)

    caps = cap_ladder(T)
    curve = build_curve(deltas, T, var_between, s2w, caps,
                        args.confidence_level, args.power_target)
    flips = sign_flip_caps(curve)
    worst = max((r for r in curve if np.isfinite(r["log2_pred_over_emp"])),
                key=lambda r: abs(r["log2_pred_over_emp"]), default=None)

    # |mu|: the test is two-sided, so detectability is sign-agnostic (a
    # negative delta just means B beats A on this metric).
    snr_now = abs(mu) / np.sqrt(var_total) if var_total > 0 else (float("inf") if mu != 0 else float("nan"))
    snr_inf = abs(mu) / np.sqrt(var_between) if model_extrapolation_valid else float("nan")

    def n_req(snr):
        return required_n(snr, args.confidence_level, args.power_target)

    n80_now, n80_min = n_req(snr_now), n_req(snr_inf)
    # Transform the effect interval through the same conditional power model.
    snr_lo, snr_hi = snr_interval(snr_now, n, args.confidence_level)
    n80_now_lo, n80_now_hi = n_req(snr_hi), n_req(snr_lo)   # note the swap
    # var_between is clamped at 0, so the raw ratio can exceed 1; clamp the
    # SHARES too or the between-item share prints negative and the JSON
    # exports token_noise_share > 1.
    token_share_raw = var_token / var_total if var_total > 0 else float("nan")
    token_share = (min(1.0, token_share_raw) if np.isfinite(token_share_raw)
                   else token_share_raw)

    # Items at the collection cap may have unstored positions. The
    # collection cap comes from collect_meta.json (same field the verdict
    # pipeline guards on).
    coll_cap = None
    meta_path = args.candidate_a / "collect_meta.json"
    if meta_path.exists():
        try:
            coll_cap = json.loads(meta_path.read_text()).get("num_eval_tokens")
        except (json.JSONDecodeError, OSError):
            pass
    cap_saturated = (float((npos_all >= coll_cap).mean())
                     if coll_cap not in (None, -1) else 0.0)

    cost = cost_model_from_manifest(args.candidate_a / "manifest.csv")
    k_star = None
    if cost is not None and model_extrapolation_valid:
        squared = (cost["cfix"] / cost["ctok"]) * (float(s2w.mean()) / var_between)
        if math.isfinite(squared) and squared > 0:
            k_star = math.sqrt(squared)

    hint = _planning_hint(mu)

    cap_str = "all stored" if args.num_eval_tokens == -1 else str(args.num_eval_tokens)
    print(f"metric={args.metric}  analysis-cap={cap_str}  items={n} "
          f"(dropped {dropped})  T: mean={T.mean():.0f} min={T.min()} max={T.max()}")
    print(f"mu (mean paired delta)      = {mu:+.6e}"
          + ("  (negative: B beats A; SNR/N use |mu|)" if mu < 0 else ""))
    print(f"Var(d_i)                    = {var_total:.3e}")
    print(f"  IID proxy mean(s_i^2/T_i) = {var_token:.3e}  ({100*token_share_raw:.1f}% of empirical variance)")
    print(f"  raw variance residual    = {var_between_raw:.3e}"
          + ("  [negative: suppress model extrapolation]" if clamped else ""))
    print("  proxy assumptions: independent residuals with a common conditional mean; "
          "token dependence or position profiles invalidate the component interpretation.")
    print(f"SNR now / model T->inf     = {snr_now:.4f} / "
          + ("undefined" if np.isnan(snr_inf) else f"{snr_inf:.4f}"))
    print(f"  normal-model SNR {int(args.confidence_level*100)}% interval = "
          f"[{snr_lo:.4f}, {snr_hi:.4f}]"
          + ("   (the standardized effect may be zero)"
             if snr_lo <= 0.0 else ""))
    n80_hi_str = ("unbounded or unavailable" if n80_now_hi is None else str(n80_now_hi))
    print("WARNING: every N_obs value below conditions on this pilot's observed "
          "mu; it is a cap-profile diagnostic, not prospective power. Use "
          "power_analysis.py --sesoi ... to plan a collection.")
    print(f"N_obs{int(args.power_target*100)} now / model floor = "
          f"{n80_now} / {n80_min if n80_min is not None else 'unavailable'}"
          f"   (have {n} items)")
    print(f"  N_obs{int(args.power_target*100)} interval        = "
          f"[{n80_now_lo if n80_now_lo is not None else '?'}, {n80_hi_str}]"
          "   (normal-model uncertainty in the observed standardized effect; "
          "not a prospective design guarantee)")

    _print_cap_ladder(curve, worst, flips, args.power_target)

    if k_star is not None:
        print(f"K* (conditional IID model)  = {k_star:.0f}   "
              f"(cfix={cost['cfix']:.2f}s, ctok={cost['ctok']*1e3:.3f}ms/token, "
              f"cost-model r2={cost['r2']:.2f}; assumptions are not validated by this fit)")
    if coll_cap not in (None, -1):
        print(f"collection cap              = {coll_cap}  "
              f"({100*cap_saturated:.0f}% of items saturate it)")
    print(f"lag-1 token autocorrelation = {rho1:+.3f}  "
          "(descriptive; lag 1 alone does not measure all token dependence)")
    print(f"interpretation: {hint}")
    print("caveat: SNR(inf)/N_min extrapolate beyond the stored cap: they "
          "assume later positions carry the same mu and sigma_b^2 — the "
          "measured mu(K) above shows whether that is plausible for this pair.")
    print("caveat: the SNR interval and N_obs assume IID normal item deltas. "
          "They do not establish this model for the pilot or select an effect for a future study.")

    if args.output_json:
        payload = {
            "metric": args.metric,
            "num_eval_tokens": args.num_eval_tokens,
            "n_items": n, "n_dropped": dropped,
            "T_mean": float(T.mean()),
            "mu": float(mu),
            "var_total": float(var_total),
            "var_token_component": var_token,
            "var_between": float(var_between),
            "var_between_raw": float(var_between_raw),
            "var_between_clamped": clamped,
            "variance_proxy_assumptions": "independent residuals with a common conditional mean; no token autocovariance or position-mean variation",
            "model_extrapolation_status": ("conditional_unvalidated_iid_common_mean_model"
                                           if model_extrapolation_valid else "unavailable_nonpositive_variance_residual"),
            "token_noise_share": float(token_share),
            "token_noise_share_raw": float(token_share_raw),
            "between_item_share": float(1.0 - token_share),
            "snr_now": float(snr_now),
            "snr_ci_lower": float(snr_lo),
            "snr_ci_upper": float(snr_hi),
            "snr_ci_method": "noncentral_t_inversion_iid_normal_item_deltas",
            "snr_ci_status": ("available" if math.isfinite(snr_lo) and math.isfinite(snr_hi)
                              else "unavailable_numerical_or_undefined_effect"),
            "snr_inf": float(snr_inf) if np.isfinite(snr_inf) else None,
            "n_required_now": n80_now,
            "n_required_now_lower": n80_now_lo,
            "n_required_now_upper": n80_now_hi,
            "n_required_null_means": "unbounded, undefined effect, or no numerically resolved representable count",
            "n_required_uses": (
                "observed pilot effect; exact two-sided noncentral-t power under IID normal item deltas; "
                "integer search limited to 2**53; diagnostic only, not prospective power"
            ),
            "n_required_role": "conditional_observed_effect_diagnostic",
            "n_required_floor": n80_min,
            "curve": [{k: (None if isinstance(v, float) and not np.isfinite(v)
                           else v) for k, v in r.items()} for r in curve],
            "sign_flip_caps": flips,
            "model_fit_worst_log2_pred_over_emp":
                (worst["log2_pred_over_emp"] if worst is not None else None),
            "k_star_sqrt_rule": k_star,
            "cost_model": cost,
            "collection_cap": coll_cap,
            "cap_saturated_fraction": cap_saturated,
            "lag1_autocorr": rho1 if np.isfinite(rho1) else None,
            "confidence_level": args.confidence_level,
            "power_target": args.power_target,
            "hint": hint,
        }
        if var_total == 0.0:
            payload["snr_status"] = "zero_effect_zero_variance" if mu == 0.0 else "zero_variance_nonzero_effect"
        for block in (payload, cost or {}):
            for key, value in block.items():
                if isinstance(value, float) and not np.isfinite(value):
                    block[key] = None
        args.output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote json -> {args.output_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
