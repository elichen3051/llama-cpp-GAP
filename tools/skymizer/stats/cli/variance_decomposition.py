#!/usr/bin/env python3
"""Empirical var/effect-vs-cap curve + variance decomposition for paired KLD.

Answers "should I extend --num-eval-tokens, or collect more items?" BEFORE
spending GPU time. Runs entirely on already-collected .npz metric dumps
(CPU, seconds).

Model — and where it is allowed to speak
----------------------------------------
Per item i and answer position t, the paired per-token delta of a metric
between candidates B and A (both measured against one bit-identical
reference) is

    delta_{i,t} = metric_B(i,t) - metric_A(i,t)

The classical two-component model (knowledge/adding-error-bars-to-evals-
2411.00640.pdf, Sec. 3.1/5, with tokens playing the role of the paper's K
resamples) says the item score d_i(K) = mean of the first K deltas has

    Var(d_i(K)) ~= sigma_b^2 + sigma_w^2 / K .

The 2026-08-26 power-allocation backtest (697 pair x cap cells over 9
campaigns) showed this law holds ONLY at large K: at K <= 256 it
UNDERESTIMATES the real Var(d_i(K)) by up to ~2x. The measured cause is not
token autocorrelation (median lag-1 ~ +0.03, batch-means design effect
~1.2) but POSITION NON-STATIONARITY: the first tens of answer tokens carry
far more delta variance than the whole-sequence average, so a full-length
sigma_w^2 dilutes them. The mean delta mu(K) is not constant in K either
(44/75 pairs peak at an interior cap; 4/75 flip sign across caps).

This tool therefore treats the EMPIRICAL cap ladder as the design surface:
at each cap K it recomputes mu(K), Var(d_i(K)) and the required item count
directly from the truncated data — no 1/K assumption anywhere in those
columns. The analytic law appears only as (a) an explicit per-cap model
check, log2(pred/emp), so you can see how wrong it is on YOUR pair, and
(b) the T -> inf floor, which has no empirical alternative and is labeled
as an extrapolation.

Reported quantities
-------------------
* Full-cap decomposition: Var(d_i), sigma_w^2/T share, sigma_b^2 share
  (moment estimators; sigma_b^2 = Var(d_i) - mean_i(s_i^2/T_i), clamped 0).
* Cap ladder (16, 32, ... , full): mu(K), sd(K), d(K) = mu/sd, and an
  OBSERVED-EFFECT conditional N diagnostic from the two-sided t-quantile
  formula. It is not prospective power; use power_analysis.py with an external
  SESOI for collection planning. The cap-LOCAL token-noise
  share mean_i(s_i^2(K)/K_i)/Var (requires independent residuals with a
  common conditional mean), and the model check log2(pred/emp).
* Sign profile of mu(K): a sign flip across caps means the cap is an
  ESTIMATOR choice, not a precision knob — pre-register it, and inspect the
  position-resolved delta profile before trusting any single-cap verdict.
* sqrt-rule K* = sqrt((c_fix/c_tok) * sigma_w^2/sigma_b^2) when
  manifest.csv carries a usable cost model — an order-of-magnitude SEED
  (the backtest found K* clustered within ~2x inside one corpus x mode and
  the budget frontier flat within 4x of it), not a substitute for the
  ladder.
* Observed-effect N at the analysis cap with the SNR interval: mu is ESTIMATED and
  N ~ 1/mu^2, so the interval is the answer, not the point (an interval
  reaching infinity means this pilot cannot size the study).

Usage
-----
Basic (kld, all stored positions):

    python3 stats/cli/variance_decomposition.py \\
        --candidate-a outputs/vlm-kld-ref-vs-a \\
        --candidate-b outputs/vlm-kld-ref-vs-b

What-if at a shorter analysis cap (first min(K, npos) positions):

    python3 stats/cli/variance_decomposition.py \\
        --candidate-a A/ --candidate-b B/ --num-eval-tokens 512

Other metrics, machine-readable output, custom power target:

    python3 stats/cli/variance_decomposition.py \\
        --candidate-a A/ --candidate-b B/ \\
        --metric js_kld --power-target 0.9 --output-json decomp.json

Reading it:
  * do not plan from the observed-effect N column; use power_analysis.py with
    a signed, externally chosen SESOI. The column is retained only to diagnose
    how the pilot's observed signal changes across caps;
  * the token-noise share assumes independent residuals with a common conditional mean;
    autocorrelation and position profiles can invalidate that interpretation;
  * use the empirical cap curve to inspect changes, and check saturation before
    assuming a longer collection cap adds scored positions.

Pair with: power_analysis.py (prospective power / MDE surface),
saved_metrics_paired_compare.py (the production verdict).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stats import collection_io as paired_io                 # noqa: E402
from stats.student_t import t_ppf                           # noqa: E402

# metric name -> per-token delta column extractor (B minus A)
METRIC_COLUMNS = {
    "kld":          lambda ma, mb: mb["kld"] - ma["kld"],
    "reversed_kld": lambda ma, mb: mb["reversed_kld"] - ma["reversed_kld"],
    "js_kld":       lambda ma, mb: mb["js_kld"] - ma["js_kld"],
    "nll":          lambda ma, mb: mb["nll_cand"] - ma["nll_cand"],
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
    """Smallest n satisfying the two-sided t-quantile sample-size approximation.

    This is not exact noncentral-t power inversion. Return None for an
    undefined effect or a required count outside floating-point range.
    Quantiles use the shared scipy.stats.t.ppf wrapper: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    MATLAB quantile correspondence only: https://www.mathworks.com/help/stats/tinv.html
    """
    if not np.isfinite(snr) or snr <= 0.0:
        return None
    if not (0.0 < confidence_level < 1.0 and 0.0 < power_target < 1.0):
        raise ValueError("confidence_level and power_target must be in (0, 1)")
    alpha = 1.0 - confidence_level
    z_sum = (NormalDist().inv_cdf(1.0 - alpha / 2.0)
             + NormalDist().inv_cdf(power_target))
    try:
        initial = (z_sum / snr) ** 2
    except OverflowError:
        return None
    if not math.isfinite(initial):
        return None

    def sufficient(n):
        df = n - 1
        t_sum = t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power_target, df)
        return n >= (t_sum / snr) ** 2

    low, high = 2, max(3, int(math.ceil(initial)))
    while not sufficient(high):
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if sufficient(middle):
            high = middle
        else:
            low = middle
    return high


def snr_interval(snr: float, n: int, confidence_level: float):
    """(lo, hi) for the standardized effect |mu|/sd, clipped at 0.

    SNR is a one-sample standardized mean (Cohen's d for the paired delta),
    whose standard error is well approximated by sqrt(1/n + d^2/(2n)). This
    interval is the whole point of S5: required n scales like 1/snr^2, so a
    snr interval that reaches 0 means the required n is unbounded above, and
    printing a single number for it is a fiction."""
    if not np.isfinite(snr) or n < 2:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    se = math.sqrt(1.0 / n + snr * snr / (2.0 * n))
    return (max(0.0, snr - z * se), snr + z * se)


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
            "n_required": (3 if emp == 0.0 and mu != 0.0
                           else required_n(snr, confidence_level, power_target)),
            "noise_share_local": (min(1.0, noise_local / emp)
                                  if emp > 0 else float("nan")),
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
    Returns dict(cfix, ctok, r2, n_rows). cfix folds the mean-prefill term."""
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
    M = np.array([[1.0, ne, npre] for ne, npre, _ in rows])
    y = np.array([w for _, _, w in rows])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = M @ coef - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    cfix = float(coef[0] + coef[2] * M[:, 2].mean())
    ctok = float(coef[1])
    if not (np.isfinite(cfix) and np.isfinite(ctok)) or ctok <= 0:
        return None
    return {"cfix": cfix, "ctok": ctok, "r2": r2, "n_rows": len(rows)}


def _planning_hint(mu, token_share, coll_cap, cap_saturated, n80_min, n):
    floor_str = (f"N_min={n80_min}" if n80_min is not None
                 else "no floor (token noise explains all variance)")
    if mu == 0.0:
        hint = "no observed mean difference; this pilot does not size a nonzero effect"
    elif token_share > 0.5:
        if coll_cap not in (None, -1) and cap_saturated < 0.05:
            hint = ("token noise dominates, BUT answers are already fully "
                    f"stored ({100*cap_saturated:.0f}% of items hit the "
                    f"collection cap {coll_cap}) — extending --num-eval-tokens "
                    "adds no tokens; treat as item-bound: add items or switch "
                    "dataset")
        else:
            hint = ("token noise dominates -> extending --num-eval-tokens helps"
                    + (f"; but N_min={n80_min} exceeds the {n} items on hand — "
                       "more items are needed regardless"
                       if n80_min is not None and n80_min > n else
                       f"; floor {floor_str} is within the {n} items on hand"))
    elif token_share < 0.3:
        hint = ("item heterogeneity dominates -> more eval tokens are nearly "
                "useless; add items or switch to a higher-signal dataset")
    else:
        hint = ("mixed regime -> token extension gives moderate gains, "
                f"bounded by {floor_str}")
    return hint


def _print_cap_ladder(curve, worst, flips, power_target):
    print()
    print("empirical cap ladder (design table — nothing here assumes the 1/K law):")
    npow = f"Nobs_{int(power_target*100)}"
    print(f"  {'K':>5}  {'mu(K)':>12}  {'sd(K)':>10}  {'d(K)':>8}  "
          f"{npow + '(K)':>8}  {'noise%':>6}  {'log2(pred/emp)':>14}")
    for r in curve:
        nreq = "inf" if r["n_required"] is None else str(r["n_required"])
        share = ("   n/a" if not np.isfinite(r["noise_share_local"])
                 else f"{100*r['noise_share_local']:5.1f}%")
        fit = ("   n/a" if not np.isfinite(r["log2_pred_over_emp"])
               else f"{r['log2_pred_over_emp']:+14.2f}")
        print(f"  {r['K']:>5}  {r['mu']:>+12.4e}  {r['sd']:>10.3e}  "
              f"{(r['mu']/r['sd'] if r['sd'] > 0 else math.copysign(r['snr'], r['mu'])):>+8.3f}  "
              f"{nreq:>8}  {share}  {fit}")
    if worst is not None and abs(worst["log2_pred_over_emp"]) > 0.3:
        fit = worst["log2_pred_over_emp"]
        if fit < 0:
            direction = ("UNDERestimates the real variance (front-loaded token "
                         "variance: short caps are noisier than the law thinks; "
                         "the F1 pattern)")
        else:
            direction = ("OVERestimates the real variance (rear-loaded token "
                         "variance: short caps are quieter than the law thinks)")
        print(f"  model check: the sigma_b^2 + sigma_w^2/K law {direction} by up "
              f"to {2**abs(fit):.2f}x (worst at K={worst['K']}) — position "
              "non-stationarity (backtest-2026-08-26 F1). Plan from the "
              "empirical columns above, not from the law.")
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
    var_between = max(var_total - var_token, 0.0)
    clamped = bool(var_total - var_token < 0.0)

    caps = cap_ladder(T)
    curve = build_curve(deltas, T, var_between, s2w, caps,
                        args.confidence_level, args.power_target)
    flips = sign_flip_caps(curve)
    worst = max((r for r in curve if np.isfinite(r["log2_pred_over_emp"])),
                key=lambda r: abs(r["log2_pred_over_emp"]), default=None)

    # |mu|: the test is two-sided, so detectability is sign-agnostic (a
    # negative delta just means B beats A on this metric).
    snr_now = abs(mu) / np.sqrt(var_total) if var_total > 0 else (float("inf") if mu != 0 else float("nan"))
    snr_inf = abs(mu) / np.sqrt(var_between) if var_between > 0 else (float("inf") if mu != 0 else float("nan"))

    def n_req(snr):
        return required_n(snr, args.confidence_level, args.power_target)

    n80_now, n80_min = n_req(snr_now), n_req(snr_inf)
    # The interval that makes the point estimate honest: mu is ESTIMATED, and
    # required n scales like 1/snr^2, so the answer is violently unstable near
    # the significance boundary. A snr interval that reaches 0 means required
    # n is unbounded above, and the tool must say so instead of printing a
    # single confident number.
    snr_lo, snr_hi = snr_interval(snr_now, n, args.confidence_level)
    n80_now_lo, n80_now_hi = n_req(snr_hi), n_req(snr_lo)   # note the swap
    # var_between is clamped at 0, so the raw ratio can exceed 1; clamp the
    # SHARES too or the between-item share prints negative and the JSON
    # exports token_noise_share > 1.
    token_share_raw = var_token / var_total if var_total > 0 else float("nan")
    token_share = (min(1.0, token_share_raw) if np.isfinite(token_share_raw)
                   else token_share_raw)

    # Items whose stored positions hit the collection-time cap still have
    # unstored answer tokens; only those gain from a larger cap. The
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
    if cost is not None and var_between > 0:
        k_star = math.sqrt((cost["cfix"] / cost["ctok"])
                           * (float(s2w.mean()) / var_between))

    hint = _planning_hint(mu, token_share, coll_cap, cap_saturated, n80_min, n)

    cap_str = "all stored" if args.num_eval_tokens == -1 else str(args.num_eval_tokens)
    print(f"metric={args.metric}  analysis-cap={cap_str}  items={n} "
          f"(dropped {dropped})  T: mean={T.mean():.0f} min={T.min()} max={T.max()}")
    print(f"mu (mean paired delta)      = {mu:+.6e}"
          + ("  (negative: B beats A; SNR/N use |mu|)" if mu < 0 else ""))
    print(f"Var(d_i)                    = {var_total:.3e}")
    print(f"  token-noise  sigma_w^2/T  = {var_token:.3e}  ({100*token_share:.1f}%)")
    print(f"  between-item sigma_b^2    = {var_between:.3e}  "
          f"({100*(1-token_share):.1f}%)"
          + ("  [clamped to 0 — token noise explains everything]" if clamped else ""))
    print(f"SNR now / at T->inf         = {snr_now:.4f} / "
          + ("undefined" if np.isnan(snr_inf) else f"{snr_inf:.4f}"))
    print(f"  SNR {int(args.confidence_level*100)}% interval        = "
          f"[{snr_lo:.4f}, {snr_hi:.4f}]"
          + ("   <-- reaches 0: the required N below has NO upper bound"
             if snr_lo <= 0.0 else ""))
    n80_hi_str = ("infinity" if n80_now_hi is None else str(n80_now_hi))
    print("WARNING: every N_obs value below conditions on this pilot's observed "
          "mu; it is a cap-profile diagnostic, not prospective power. Use "
          "power_analysis.py --sesoi ... to plan a collection.")
    print(f"N_obs{int(args.power_target*100)} now / model floor = "
          f"{n80_now} / {n80_min if n80_min is not None else 'none (no floor)'}"
          f"   (have {n} items)")
    print(f"  N_obs{int(args.power_target*100)} interval        = "
          f"[{n80_now_lo if n80_now_lo is not None else '?'}, {n80_hi_str}]"
          "   (from the SNR interval; mu is ESTIMATED and N ~ 1/mu^2, so this "
          "interval is the answer, not the point estimate)")

    _print_cap_ladder(curve, worst, flips, args.power_target)

    if k_star is not None:
        print(f"K* (sqrt-rule seed)         = {k_star:.0f}   "
              f"(cfix={cost['cfix']:.2f}s, ctok={cost['ctok']*1e3:.3f}ms/token, "
              f"cost-model r2={cost['r2']:.2f}; order-of-magnitude seed only — "
              "choose the working cap from the ladder)")
    if coll_cap not in (None, -1):
        print(f"collection cap              = {coll_cap}  "
              f"({100*cap_saturated:.0f}% of items saturate it)")
    print(f"lag-1 token autocorrelation = {rho1:+.3f}  "
          "(diagnostic; measured campaigns sat near +0.03 — the dominant "
          "model deviation is position non-stationarity, not autocorrelation)")
    print(f"verdict: {hint}")
    print("caveat: SNR(inf)/N_min extrapolate beyond the stored cap: they "
          "assume later positions carry the same mu and sigma_b^2 — the "
          "measured mu(K) above shows whether that is plausible for this pair.")
    print("caveat: every N above plugs in an ESTIMATED mu. Because N ~ 1/mu^2 "
          "the point estimate is unstable near the significance boundary — "
          "plan from the interval's UPPER end, and treat an interval that "
          "reaches infinity as 'this pilot cannot size the study'.")

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
            "var_between_clamped": clamped,
            "token_noise_share": float(token_share),
            "token_noise_share_raw": float(token_share_raw),
            "between_item_share": float(1.0 - token_share),
            "snr_now": float(snr_now),
            "snr_ci_lower": float(snr_lo),
            "snr_ci_upper": float(snr_hi),
            "snr_inf": float(snr_inf) if np.isfinite(snr_inf) else None,
            "n_required_now": n80_now,
            "n_required_now_lower": n80_now_lo,
            "n_required_now_upper": n80_now_hi,   # null = unbounded
            "n_required_uses": (
                "observed pilot effect; two-sided paired t approximation "
                "solved as an integer bound on df; diagnostic only, not prospective power"
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
