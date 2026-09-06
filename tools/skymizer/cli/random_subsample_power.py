#!/usr/bin/env python3
# =============================================================================
# random_subsample_power.py
#
# Two DIFFERENT curves over subsample size N, selected with --mode:
#
#   --mode reproducibility  (default)
#       Draws N items WITHOUT replacement from the collected population and
#       reports how often the verdict matches the full-population verdict.
#       This is a stability/consistency diagnostic. It is NOT power, and it
#       is biased UPWARDS as an estimate of power for two compounding
#       reasons:
#         * finite population. A subset mean of a size-N draw without
#           replacement has variance (sigma^2/N)(1 - N/N_pop), while the
#           bootstrap CI inside each draw estimates sigma^2/N with no
#           finite-population correction. So the interval is too wide
#           relative to the actual spread of the draws, and at N == N_pop
#           every draw reproduces the verdict by construction: the curve
#           ends at 100% no matter what the effect is.
#         * conditioning. The "truth" it reproduces is the observed
#           population verdict, so whenever the population came out
#           significant the curve is conditioned on an effect estimate
#           selected for being large.
#       Measured on simulated data: in a borderline regime this curve reads
#       0.76 at N=200 / 1.00 at N=250 where the true power is 0.42 / 0.50;
#       under a PURE NULL, conditional on a spuriously significant
#       population, it reads 0.65 / 1.00 where the true detection rate is
#       0.05. Read it as "how stable is this verdict under resampling of
#       the rows I already have", nothing more.
#
#   --mode power
#       Draws N items WITH replacement, i.e. from the empirical population
#       distribution rather than from the finite pool. That removes the
#       finite-population bias (N may exceed N_pop) and estimates the
#       probability that a FRESH sample of N items would detect the effect.
#       It still conditions on the pilot's effect ESTIMATE -- unavoidable
#       with one pilot -- so the report also prints the population effect's
#       own confidence interval, and the required-N line is reported as a
#       range over that interval rather than as a point.
#
# In both modes the reported N is conservative: a size only counts as
# clearing the target when the LOWER end of the Wilson interval on the
# Monte-Carlo fraction clears it, so a lucky first crossing at --reps 100
# (MC SE ~ 5pp) cannot bias it down.
#
# Reuses the EXACT scoring + statistics path of the real report:
#   saved_metrics_paired_compare.score_item  -> per-item score dicts
#   paired_compare.compare_items             -> seeded paired bootstrap + verdict
#
# Usage:
#   python3 random_subsample_power.py \
#       --candidate-a outputs/kld-ins-q4km-mmf16 \
#       --candidate-b outputs/kld-ins-q4km-mmq80 \
#       [--mode reproducibility|power] \
#       --sizes 25 50 75 100 125 150 200 250 \
#       --reps 100 --bootstrap-iters 2000 --seed 7 \
#       --out outputs/power-curve.md [--output-json outputs/power-curve.json]
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare.contracts import DEFAULT_METRICS              # noqa: E402
from compare.engine import compare_items                   # noqa: E402
import cli.saved_metrics_paired_compare as smpc                # noqa: E402

WEIGHTINGS = ("item_weighted", "token_weighted")


# Verdicts that mean "the interval contains the null": the tool used to say
# "no sig. diff.", which read as an equivalence claim it never made.
# EQUIVALENT is a real finding but still not a detected DIFFERENCE, so both
# count as "nothing to reproduce" here.
_NULL_VERDICTS = ("inconclusive", "EQUIVALENT", "no sig. diff.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-a", required=True, type=Path)
    p.add_argument("--candidate-b", required=True, type=Path)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[25, 50, 75, 100, 125, 150, 200, 250])
    p.add_argument("--reps", type=int, default=100,
                   help="random subsets drawn per size")
    p.add_argument("--bootstrap-iters", type=int, default=2000,
                   help="bootstrap iterations per draw (report default is 5000; "
                        "2000 keeps the sweep fast with ~the same verdicts)")
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7,
                   help="master seed for subset draws and per-rep bootstrap seeds")
    p.add_argument("--mode", choices=("reproducibility", "power"),
                   default="reproducibility",
                   help="reproducibility (default): draw WITHOUT replacement "
                        "from the collected rows and count verdict agreement "
                        "with the full population - a stability diagnostic, "
                        "biased upwards as power. power: draw WITH "
                        "replacement from the empirical distribution, which "
                        "removes the finite-population bias and estimates "
                        "the detection rate of a FRESH sample of N items.")
    p.add_argument("--power-target", type=float, default=0.80,
                   help="target rate for the 'smallest N' line (default 0.80)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def load_population(a_dir: Path, b_dir: Path):
    """Load the complete paired population under the production guards."""
    try:
        with smpc.comparison_locks((a_dir, b_dir)):
            for role, root in (("candidate-a", a_dir), ("candidate-b", b_dir)):
                smpc.require_collection_success(root, role)
            a_meta = smpc.load_kld_collect_meta(a_dir)
            b_meta = smpc.load_kld_collect_meta(b_dir)
            smpc.require_execution_alignment(a_meta, b_meta)
            for warning in smpc.check_kld_meta_alignment(a_meta, b_meta):
                print(f"WARNING: {warning}", file=sys.stderr)
            smpc.require_common_budget_skips({"candidate-a": a_dir, "candidate-b": b_dir})
            matched, drops = smpc.find_metric_items(a_dir, b_dir)
            smpc.require_complete_item_alignment(drops, allow_interaction=False)
            if not matched:
                sys.exit("no matched items between the two dirs")
            scores_a, scores_b, weights, keys = [], [], [], []
            for key in matched:
                sa, sb, _tok_a, _tok_b, keep, finite, drift, _versions = smpc.score_item(
                    key, a_dir, b_dir, -1)
                if drift is not None:
                    sys.exit(f"reference drift on {key}: {drift['msg']}")
                if not finite:
                    sys.exit(f"{key}: non-finite metric score; subsampling aborted")
                scores_a.append(sa)
                scores_b.append(sb)
                weights.append(keep)
                keys.append(key)
            smpc.require_min_items(scores_a, matched)
            return scores_a, scores_b, weights, keys
    except ValueError as error:
        sys.exit(str(error))


def wilson_lower(hits: int, n: int, z: float = 1.959963985) -> float:
    """Lower end of the Wilson score interval for hits/n.

    The "smallest N" line is a FIRST CROSSING of a Monte-Carlo estimate, and
    at --reps 100 that estimate has SE ~ 5pp, so taking the first size whose
    point estimate clears the target systematically picks a lucky draw and
    reports an N that is too small. Requiring the interval's lower end to
    clear it instead makes the answer conservative."""
    if n <= 0:
        return 0.0
    phat = hits / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    half = z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return float(max(0.0, (centre - half) / denom))


def verdicts_of(result):
    """{(metric, weighting): verdict} for the base (non-derived) metrics."""
    out = {}
    for met, m in result["metrics"].items():
        if met not in DEFAULT_METRICS:
            continue   # ppl/ppl_ratio/rms_dp are linked to nll/mse_dp
        for wt in WEIGHTINGS:
            w = m.get(wt)
            if isinstance(w, dict) and "decision" in w:
                out[(met, wt)] = w["decision"]["verdict"]
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == "power":
        print(
            "WARNING: --mode power is a legacy conditional-replication "
            "diagnostic driven by the pilot's observed effect. Use "
            "power_analysis.py --sesoi ... for prospective planning.",
            file=sys.stderr,
        )
    scores_a, scores_b, weights, keys = load_population(
        args.candidate_a, args.candidate_b)
    pop = len(keys)
    # Without replacement, N cannot exceed the pool; WITH replacement it can,
    # which is precisely the point of --mode power (it answers "what if I
    # collected MORE rows", which the reproducibility curve structurally
    # cannot).
    if args.mode == "power":
        sizes = sorted({n for n in args.sizes if n >= 2})
    else:
        sizes = sorted({n for n in args.sizes if 2 <= n <= pop})
    skipped = sorted(set(args.sizes) - set(sizes))
    if skipped:
        limit = "N >= 2" if args.mode == "power" else f"2 <= N <= {pop}"
        print(f"WARNING: skipping sizes {skipped} (population is {pop}, "
              f"need {limit})", file=sys.stderr)

    # Restrict to the metrics every item actually carries: v1 metric dumps
    # predate the `ear` column, and this sweep's verdict counting only needs
    # the base per-item means (tail metrics need per-token arrays, which this
    # tool deliberately does not thread through).
    base_metrics = tuple(
        m for m in DEFAULT_METRICS
        if all(m in s for s in (*scores_a, *scores_b)))
    dropped_metrics = tuple(m for m in DEFAULT_METRICS if m not in base_metrics)
    if dropped_metrics:
        print(f"WARNING: metrics {list(dropped_metrics)} missing from at least "
              "one item (pre-ear VLMK v1 dumps?); excluded from the sweep",
              file=sys.stderr)

    # Full-population verdicts define the "true" direction each metric tests
    # against (use the report's own bootstrap settings for this single call).
    full = compare_items(
        scores_a, scores_b, weights, metrics=base_metrics,
        confidence_level=args.confidence_level,
        bootstrap_iters=max(args.bootstrap_iters, 5000), seed=args.seed,
        model_a_label="A", model_b_label="B")
    truth = verdicts_of(full)

    # The effect the power curve conditions on, with its own uncertainty:
    # a power number computed from one pilot is only as good as that pilot's
    # effect estimate, and the report has to say so out loud.
    primary = full["multiplicity"]["primary_endpoint"]
    primary_block = full["metrics"][primary["metric"]][
        f"{primary['weighting']}_weighted"]
    effect = {
        "metric": primary["metric"],
        "weighting": primary["weighting"],
        "estimate": primary_block["delta_candidate_minus_baseline"],
        "ci_lower": primary_block["ci_delta"]["lower"],
        "ci_upper": primary_block["ci_delta"]["upper"],
        "verdict": primary_block["decision"]["verdict"],
    }

    rng = np.random.default_rng(args.seed)
    replace = args.mode == "power"
    # counts[(met, wt)][n] = {verdict: hits}
    counts = {k: {n: {} for n in sizes} for k in truth}
    for n in sizes:
        for rep in range(args.reps):
            idx = rng.choice(pop, size=n, replace=replace)
            res = compare_items(
                [scores_a[i] for i in idx],
                [scores_b[i] for i in idx],
                [weights[i] for i in idx],
                metrics=base_metrics,
                confidence_level=args.confidence_level,
                bootstrap_iters=args.bootstrap_iters,
                seed=int(rng.integers(0, 2**31 - 1)),
                model_a_label="A", model_b_label="B")
            for k, v in verdicts_of(res).items():
                counts[k][n][v] = counts[k][n].get(v, 0) + 1
        print(f"size {n}: {args.reps} reps done", file=sys.stderr)

    # rate(k, n) = fraction of draws whose verdict matches the population's.
    # In reproducibility mode that is verdict STABILITY; in power mode, for a
    # metric whose population verdict is a detected difference, it is an
    # estimate of the detection rate of a fresh N-item sample.
    def hits(k, n):
        return counts[k][n].get(truth[k], 0)

    def rate(k, n):
        return hits(k, n) / args.reps

    is_power = args.mode == "power"
    title = ("Random-resample power curve" if is_power
             else "Random-subsample verdict REPRODUCIBILITY curve")
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- population: {pop} items "
                 f"({args.candidate_a} vs {args.candidate_b})")
    lines.append(f"- draws: {args.reps} random subsets per size "
                 f"({'WITH' if is_power else 'without'} replacement), "
                 "paired Student-t verdict per draw")
    lines.append(f"- test: paired Student-t, {args.confidence_level:.0%} CI "
                 "(--bootstrap-iters and per-draw seeds are ignored by this "
                 "CI method)")
    lines.append(f"- master seed: {args.seed}")
    lines.append("")
    if is_power:
        lines.append(
            "> [!IMPORTANT]\n"
            "> This legacy curve conditions on the PILOT's effect estimate. "
            "It is a conditional replication sensitivity, not prospective "
            "power for a predeclared effect. Use `power_analysis.py --sesoi` "
            "for collection planning. The primary endpoint "
            f"(`{effect['metric']}`, {effect['weighting']}-weighted) measured "
            f"**{effect['estimate']:+.6g}** with a "
            f"{args.confidence_level:.0%} CI of "
            f"[{effect['ci_lower']:+.6g}, {effect['ci_upper']:+.6g}] "
            f"({effect['verdict']}). Since required N scales like 1/effect^2, "
            "that interval maps to a WIDE range of required sample sizes - "
            "and if it contains 0, required N is unbounded above. Read the "
            "numbers below as the curve for the observed effect, not as the "
            "curve for the true one.")
    else:
        lines.append(
            "> [!WARNING]\n"
            "> This is a stability diagnostic, **not power**, and it is biased "
            "UPWARDS as an estimate of power. Draws are taken without "
            "replacement from the same finite pool whose verdict is the "
            "target, so (a) a size-N draw's mean has variance "
            "sigma^2/N * (1 - N/N_pop) while the bootstrap CI inside each "
            "draw estimates sigma^2/N with no finite-population correction, "
            f"and every draw at N = {pop} reproduces the verdict by "
            "construction; and (b) the target verdict is itself an observed "
            "one, so a significant population conditions the curve on an "
            "effect selected for being large. Measured on simulated data: "
            "0.76 at N=200 / 1.00 at N=250 where the true power was "
            "0.42 / 0.50, and 0.65 / 1.00 under a pure null where the true "
            "detection rate was 0.05. For an estimate of what a FRESH sample "
            "would detect, re-run with `--mode power`.")
    lines.append("")
    lines.append("Each cell: % of draws reproducing the full-population "
                 "verdict (shown per row). Rows whose population verdict is "
                 "`inconclusive` or `EQUIVALENT` are agreement with the NULL, "
                 "never a detection rate.")
    lines.append("")
    header = ("| metric | weighting | full-pop verdict | "
              + " | ".join(f"N={n}" for n in sizes) + " |")
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(sizes)))
    min_n = {}
    for k in sorted(truth):
        met, wt = k
        cells = []
        for n in sizes:
            cells.append(f"{100 * rate(k, n):.0f}%")
            # CONSERVATIVE first crossing: the Wilson lower bound, not the
            # point estimate, must clear the target.
            if (truth[k] not in _NULL_VERDICTS and k not in min_n
                    and wilson_lower(hits(k, n), args.reps) >= args.power_target):
                min_n[k] = n
        lines.append(f"| {met} | {wt.split('_')[0]} | {truth[k]} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    if is_power:
        lines.append(f"## Smallest N with estimated power >= "
                     f"{args.power_target:.0%}\n")
        lines.append("A size counts only when the LOWER end of the Wilson "
                     f"interval on {args.reps} draws clears the target, so a "
                     "lucky first crossing cannot report an N that is too "
                     "small. These are still estimates for the OBSERVED "
                     "effect - see the note above.\n")
    else:
        lines.append("## Smallest N reproducing the full-population verdict "
                     f">= {args.power_target:.0%}\n")
        lines.append("> [!CAUTION]\n"
                     "> **This is not a required sample size.** It is the "
                     "point at which this pool's own verdict becomes stable "
                     "under subsampling, which is biased small for the "
                     "reasons above. Use `power_analysis.py` with an external "
                     "SESOI to plan a collection.\n")
    for k in sorted(truth):
        if truth[k] in _NULL_VERDICTS:
            continue
        met, wt = k
        got = min_n.get(k)
        lines.append(f"- {met} ({wt.split('_')[0]}): "
                     + (f"N = {got}" if got is not None
                        else f"not reached by N = {sizes[-1]}"))
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote report -> {args.out}", file=sys.stderr)

    if args.output_json:
        payload = {
            "mode": args.mode,
            "is_power_estimate": is_power,
            "power_scope": (
                "conditional_on_observed_pilot_effect"
                if is_power else "finite_pilot_verdict_reproducibility"
            ),
            "ci_method": "t",
            "bootstrap_iters_used": 0,
            "population": pop,
            "primary_effect": effect,
            "smallest_n_by_wilson_lower_bound":
                {f"{m}/{w}": min_n.get((m, w)) for (m, w) in truth},
            "sizes": sizes,
            "reps": args.reps,
            "bootstrap_iters": args.bootstrap_iters,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "full_population_verdicts":
                {f"{m}/{w}": v for (m, w), v in truth.items()},
            "verdict_counts":
                {f"{m}/{w}": {str(n): c for n, c in per_n.items()}
                 for (m, w), per_n in counts.items()},
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n",
                                    encoding="utf-8")
        print(f"wrote json -> {args.output_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
