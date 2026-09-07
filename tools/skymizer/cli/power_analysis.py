#!/usr/bin/env python3
"""Prospective power / precision planning over ordered token-prefix caps.

The pilot inputs are the same two VLMK metric directories consumed by
saved_metrics_paired_compare.py.  Every requested cap is aggregated to one
paired scalar per item.  Future datasets resample whole items with replacement;
individual tokens are never resampled or treated as independent observations.

TODO(review, before merge):
- Gate required-N crossings on an explicit null-calibration acceptance rule.
- Treat zero/near-zero pilot variance as unidentified unless an external
  variance floor is supplied; do not report perfect power or zero MDE.
- Make outer-bootstrap power bands use correct-direction power, excluding the
  wrong-sign tail currently included by the two-sided normal approximation.
- Either use simultaneous MC bands when selecting across the N-by-cap grid or
  label the existing Wilson crossing as pointwise only.
- Rename the pilot-SD CI half-width as a plug-in approximation, or estimate its
  distribution from the future-sample simulations.
- Validate mc_confidence_level in the public engine and persist candidate
  direction plus the effective pilot-row selection/fingerprint in artifacts.
- Harden the wrapper contract: separate prospective and reproducibility size
  grids, derive default caps from the collection cap, and split metric support.
- Clean up companion diagnostics: deterministic position profiles are not
  token noise; handle zero-variance/strict-JSON and negative-cost-fit cases;
  remove remaining guidance toward legacy observed-effect power.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli.saved_metrics_paired_compare as smpc  # noqa: E402
from compare.cli_common import (
    resolve_item_end,
    write_report_and_json,
)
from compare.contracts import AlignmentError, DEFAULT_METRICS  # noqa: E402
from compare.power import build_design  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-a", required=True, type=Path)
    p.add_argument("--candidate-b", required=True, type=Path)
    p.add_argument(
        "--token-caps", type=int, nargs="+", required=True,
        help="compare-time --num-eval-tokens values to evaluate; each must be >= 1",
    )
    p.add_argument(
        "--sample-sizes", type=int, nargs="+",
        default=[25, 50, 75, 100, 150, 200, 300, 500],
        help="future item counts N to evaluate (default: 25 50 75 100 150 200 300 500)",
    )
    p.add_argument("--metric", choices=DEFAULT_METRICS, default="kld")
    p.add_argument("--weighting", choices=("item", "token"), default="item")
    p.add_argument(
        "--sesoi", type=float, default=None,
        help="signed smallest effect of interest in candidate-B minus candidate-A units; "
             "without it the report is precision/MDE only",
    )
    p.add_argument(
        "--effect-profile", choices=("flat", "pilot"), default="flat",
        help="flat: each cap separately has the same cap-level SESOI; pilot: shift the "
             "whole prefix profile so --reference-cap has the SESOI",
    )
    p.add_argument(
        "--reference-cap", type=int, default=None,
        help="required by --effect-profile pilot and must appear in --token-caps",
    )
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument(
        "--ci-method", choices=("t",), default="t",
        help="production test to replay; currently the default paired Student-t endpoint",
    )
    p.add_argument("--reps", type=int, default=2000,
                   help="future-dataset Monte-Carlo replicates per (K,N) cell")
    p.add_argument(
        "--outer-reps", type=int, default=200,
        help="whole-item pilot bootstrap replicates for nuisance-uncertainty bands; 0 disables",
    )
    p.add_argument("--mc-confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument(
        "--allow-interaction", "--allow-intersection", dest="allow_interaction",
        action="store_true",
        help="explicitly allow planning on the clean item intersection",
    )
    p.add_argument(
        "--allow-ref-drift", action="store_true",
        help="allow non-bit-identical reference columns and persist the downgrade warning",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def _validate_args(args):
    caps = sorted(set(args.token_caps))
    sizes = sorted(set(args.sample_sizes))
    if any(cap < 1 for cap in caps):
        sys.exit(f"--token-caps values must be >= 1, got {args.token_caps}")
    if any(size < 2 for size in sizes):
        sys.exit(f"--sample-sizes values must be >= 2, got {args.sample_sizes}")
    if args.start < 0:
        sys.exit(f"--start must be >= 0, got {args.start}")
    if args.end is not None and args.end < -1:
        sys.exit(f"--end must be -1 (all) or >= 0, got {args.end}")
    if not 0.0 < args.confidence_level < 1.0:
        sys.exit("--confidence-level must be in (0, 1)")
    if not 0.5 < args.target_power < 1.0:
        sys.exit("--target-power must be in (0.5, 1)")
    if not 0.0 < args.mc_confidence_level < 1.0:
        sys.exit("--mc-confidence-level must be in (0, 1)")
    if args.reps < 1:
        sys.exit("--reps must be >= 1")
    if args.outer_reps < 0:
        sys.exit("--outer-reps must be >= 0")
    if args.sesoi is not None and (not math.isfinite(args.sesoi) or args.sesoi == 0.0):
        sys.exit("--sesoi must be finite and non-zero")
    if args.effect_profile == "pilot":
        if args.sesoi is None:
            sys.exit("--effect-profile pilot needs --sesoi")
        if args.reference_cap is None:
            sys.exit("--effect-profile pilot needs --reference-cap")
        if args.reference_cap not in caps:
            sys.exit("--reference-cap must appear in --token-caps")
    elif args.reference_cap is not None:
        sys.exit("--reference-cap is only meaningful with --effect-profile pilot")
    if args.sesoi is not None and args.reps < 400:
        print(
            f"WARNING: --reps {args.reps} gives wide Monte-Carlo intervals near "
            "the target; use >= 2000 for a design decision",
            file=sys.stderr,
        )
    return caps, sizes


def _check_dirs(args):
    for flag, directory in (("--candidate-a", args.candidate_a),
                            ("--candidate-b", args.candidate_b)):
        if not directory.is_dir():
            sys.exit(f"{flag}: {directory} is not a directory")
        if not (directory / "metrics").is_dir():
            sys.exit(f"{flag}: {directory} has no metrics/ subdir")


def load_cap_panel(args, caps):
    try:
        with smpc.comparison_locks((args.candidate_a, args.candidate_b)):
            return _load_cap_panel_locked(args, caps)
    except ValueError as error:
        sys.exit(str(error))


def _load_cap_panel_locked(args, caps):
    """Validated aligned candidate-minus-baseline scores for every cap."""
    _check_dirs(args)
    try:
        meta_a, meta_b, warnings, common_budget_skips = smpc.validate_collection_pair(args.candidate_a, args.candidate_b)
    except (AlignmentError, ValueError) as error:
        sys.exit(str(error))
    matched, drops = smpc.aligned_metric_items(args.candidate_a, args.candidate_b, allow_interaction=args.allow_interaction)
    if not matched:
        sys.exit("no items present in both metrics dirs")
    end, end_warning = resolve_item_end(args.end, len(matched))
    if end_warning:
        print(end_warning, file=sys.stderr)
    matched = matched[args.start:end]

    collection_cap = None
    meta = meta_a or meta_b
    if meta is not None:
        raw_cap = meta.get("num_eval_tokens")
        if isinstance(raw_cap, int):
            collection_cap = raw_cap
    if collection_cap is not None and collection_cap != -1 and max(caps) > collection_cap:
        sys.exit(
            f"requested token cap {max(caps)} exceeds the collection cap "
            f"{collection_cap}; collect a longer pilot instead of extrapolating"
        )

    differences = {cap: [] for cap in caps}
    weights = {cap: [] for cap in caps}
    used = []
    drifts = []
    for key in matched:
        try:
            ma, ha, mb, hb = smpc.load_item_pair(key, args.candidate_a, args.candidate_b)
        except AlignmentError as error:
            sys.exit(str(error))
        item_scores = {}
        item_drift = None
        for cap in caps:
            try:
                score_a, score_b, _tok_a, _tok_b, keep, finite, drift, _versions = \
                    smpc.score_records(key, ma, ha, mb, hb, cap)
            except AlignmentError as exc:
                sys.exit(str(exc))
            if item_drift is None and drift is not None:
                item_drift = drift
            if not finite:
                sys.exit(
                    f"{key}: non-finite metric score at token cap {cap}; "
                    "power planning aborted"
                )
            if args.metric not in score_a or args.metric not in score_b:
                sys.exit(
                    f"{key}: metric {args.metric!r} is unavailable at token cap {cap} "
                    "(legacy VLMK version?)"
                )
            item_scores[cap] = (
                float(score_b[args.metric] - score_a[args.metric]),
                int(keep),
            )
        if item_drift is not None:
            drifts.append(item_drift)
            if not args.allow_ref_drift:
                continue
        for cap, (difference, keep) in item_scores.items():
            differences[cap].append(difference)
            weights[cap].append(keep)
        used.append(key)

    if drifts:
        for drift in drifts:
            level = "WARNING" if args.allow_ref_drift else "ERROR"
            print(f"{level}: {drift['msg']}", file=sys.stderr)
        if not args.allow_ref_drift:
            sys.exit(
                f"{len(drifts)} item(s) show reference drift; prospective "
                "paired power analysis aborted"
            )
        warnings.append(
            "reference drift was explicitly allowed; the design is only approximately paired"
        )

    if len(used) < 2:
        sys.exit(
            f"power planning needs >= 2 usable items; got {len(used)} "
            f"from {len(matched)} selected matches"
        )
    arrays_d = {cap: np.asarray(differences[cap], dtype=float) for cap in caps}
    arrays_w = {cap: np.asarray(weights[cap], dtype=float) for cap in caps}
    for cap in caps:
        if np.all(arrays_w[cap] < cap):
            warning = (
                f"no pilot item reaches token cap {cap}; it is a saturated no-op "
                "on this pilot, not evidence about unobserved later positions"
            )
            warnings.append(warning)
            print(f"WARNING: {warning}", file=sys.stderr)
    return {
        "differences": arrays_d,
        "weights": arrays_w,
        "used": used,
        "drops": drops,
        "warnings": warnings,
        "common_budget_skips": common_budget_skips,
        "meta_a": meta_a,
        "meta_b": meta_b,
        "collection_cap": collection_cap,
        "drifts": drifts,
    }


def render_markdown(result):
    design = result["design"]
    lines = ["# Sequential-prefix power analysis", ""]
    lines.extend([
        "> [!IMPORTANT]",
        "> `--num-eval-tokens` defines an ordered prefix estimand. Tokens were ",
        "> never resampled or counted as independent observations; every Monte-Carlo ",
        "> draw resampled complete paired items with replacement.",
        "",
    ])
    if design["mode"] == "precision_only":
        lines.extend([
            "> [!NOTE]",
            "> No SESOI was supplied. This report gives precision and MDE only; it ",
            "> intentionally does not substitute the pilot's observed effect and call ",
            "> that prospective power.",
            "",
        ])
    elif design["effect_profile"] == "flat":
        lines.extend([
            "> [!NOTE]",
            "> `flat` asks a separate question at each cap: power if that cap-level ",
            "> estimand equals the same SESOI. The cells are comparable design ",
            "> scenarios, not one joint token-level data-generating process.",
            "",
        ])
    else:
        lines.extend([
            "> [!NOTE]",
            "> `pilot` applies one constant shift to the whole prefix profile so the ",
            "> reference cap equals the SESOI. It is coherent across caps but retains ",
            "> the pilot's effect-vs-position shape as a modeling assumption.",
            "",
        ])

    lines.extend([
        "## Design",
        "",
        f"- metric / weighting: `{result['estimand']['metric']}` / "
        f"`{design['weighting']}`",
        f"- pilot items: {design['pilot_n_items']}",
        f"- test: paired Student-t, {design['confidence_level']:.1%} CI",
        f"- target power: {design['target_power']:.1%}",
        f"- SESOI: `{design['sesoi']}`" if design["sesoi"] is not None else "- SESOI: not supplied",
        f"- future-dataset reps: {design['reps']}; outer pilot reps: {design['outer_reps']}",
        f"- seed: {design['seed']}",
        "",
        "## Pilot cap profile",
        "",
        "| K | observed pilot effect | effective SD [outer p10, p90] | mean evaluated tokens/item | reaching K |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in design["pilot_caps"]:
        if "effective_sd_outer_p10" in row:
            effective_sd = (
                f"{row['effective_sd']:.6g} "
                f"[{row['effective_sd_outer_p10']:.6g}, "
                f"{row['effective_sd_outer_p90']:.6g}]"
            )
        else:
            effective_sd = f"{row['effective_sd']:.6g}"
        lines.append(
            f"| {row['token_cap']} | {row['pilot_effect']:+.6g} | "
            f"{effective_sd} | {row['mean_tokens_per_item']:.1f} | "
            f"{row['fraction_reaching_cap']:.1%} |"
        )

    lines.extend(["", "## Design surface", ""])
    if design["mode"] == "prospective_power":
        lines.extend([
            "Power counts only rejection in the assumed effect's direction. `null rej.` ",
            "is a separately null-centered calibration simulation; its Monte-Carlo ",
            "interval should cover the nominal alpha before trusting the power cell.",
            "`pilot power p10–p90` is an outer whole-item pilot-bootstrap band ",
            "using the labeled normal approximation, not another MC interval.",
            "",
            "| K | N | assumed effect | power [MC interval] | pilot power p10–p90 (approx.) | "
            "wrong sign | null rej. [MC interval] | CI half-width | MDE approx | expected tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cell in design["cells"]:
            if cell["directional_power_defined"]:
                power = (
                    f"{cell['power']:.1%} "
                    f"[{cell['power_mc_lower']:.1%}, {cell['power_mc_upper']:.1%}]"
                )
                wrong = f"{cell['wrong_sign_rate']:.1%}"
            else:
                power = "undefined (effect = 0)"
                wrong = "—"
            uncertainty = cell.get("pilot_uncertainty", {})
            if uncertainty.get("power_p10") is not None:
                nuisance_power = (
                    f"{uncertainty['power_p10']:.1%}–"
                    f"{uncertainty['power_p90']:.1%}"
                )
            else:
                nuisance_power = "—"
            null_rejection = (
                f"{cell['null_rejection_rate']:.1%} "
                f"[{cell['null_rejection_mc_lower']:.1%}, "
                f"{cell['null_rejection_mc_upper']:.1%}]"
            )
            lines.append(
                f"| {cell['token_cap']} | {cell['sample_size']} | "
                f"{cell['assumed_effect']:+.6g} | {power} | {nuisance_power} | "
                f"{wrong} | "
                f"{null_rejection} | "
                f"{cell['expected_ci_half_width']:.6g} | {cell['mde_approx']:.6g} | "
                f"{cell['expected_evaluated_tokens']:.0f} |"
            )
    else:
        lines.extend([
            "| K | N | expected CI half-width | MDE approx | expected tokens |",
            "|---:|---:|---:|---:|---:|",
        ])
        for cell in design["cells"]:
            lines.append(
                f"| {cell['token_cap']} | {cell['sample_size']} | "
                f"{cell['expected_ci_half_width']:.6g} | {cell['mde_approx']:.6g} | "
                f"{cell['expected_evaluated_tokens']:.0f} |"
            )

    if design["mode"] == "prospective_power":
        lines.extend([
            "",
            f"## First evaluated N reaching {design['target_power']:.0%}",
            "",
            "The MC-lower column requires the lower end of the future-simulation ",
            "Wilson interval to clear the target. The pilot-p10 column additionally ",
            "requires the lower decile across outer whole-item pilot resamples to ",
            "clear it. `—` means the supplied N grid did not establish a crossing.",
            "",
            "| K | point estimate | MC-lower-bound criterion | pilot-power-p10 criterion (approx.) |",
            "|---:|---:|---:|---:|",
        ])
        for row in design["required_n_by_cap"]:
            lines.append(
                f"| {row['token_cap']} | "
                f"{row['first_evaluated_n_point_estimate'] or '—'} | "
                f"{row['first_evaluated_n_mc_lower_bound'] or '—'} | "
                f"{row['first_evaluated_n_pilot_power_p10'] or '—'} |"
            )

    lines.extend([
        "",
        "## Assumptions and limits",
        "",
        "- Pilot and future items are exchangeable draws from the same population.",
        "- Items are the independent sampling units. If rows share a passage/image, "
        "this version needs a higher-level cluster-aware final test before use.",
        "- Caps must be chosen on an independent pilot or pre-registered. Scanning "
        "the confirmatory sample for the best cap invalidates nominal error rates.",
        "- No power is extrapolated beyond the stored collection cap.",
        "- Outer-bootstrap bands quantify pilot nuisance uncertainty only; they do "
        "not cover corpus/model drift or uncertainty in the SESOI.",
        "",
    ])
    if result["alignment"]["warnings"]:
        lines.extend(["## Persisted warnings", ""])
        lines.extend(f"- {warning}" for warning in result["alignment"]["warnings"])
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    caps, sizes = _validate_args(args)
    panel = load_cap_panel(args, caps)
    try:
        design = build_design(
            panel["differences"], panel["weights"], sizes,
            weighting=args.weighting,
            confidence_level=args.confidence_level,
            target_power=args.target_power,
            sesoi=args.sesoi,
            effect_profile=args.effect_profile,
            reference_cap=args.reference_cap,
            reps=args.reps,
            outer_reps=args.outer_reps,
            seed=args.seed,
            mc_confidence_level=args.mc_confidence_level,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    meta = panel["meta_a"] or panel["meta_b"] or {}
    result = {
        "schema_version": design["schema_version"],
        "estimand": {
            "metric": args.metric,
            "weighting": args.weighting,
            "paired_difference": "candidate_b_minus_candidate_a",
            "item_score": "mean over first min(token_cap, stored item length) positions",
            "sampling_unit": "item",
        },
        "inputs": {
            "candidate_a": str(args.candidate_a),
            "candidate_b": str(args.candidate_b),
            "dataset": meta.get("dataset"),
            "subset": meta.get("subset"),
            "split": meta.get("split"),
            "collection_cap": panel["collection_cap"],
        },
        "alignment": {
            "n_used": len(panel["used"]),
            "drops": panel["drops"],
            "allow_interaction": bool(args.allow_interaction),
            "common_skipped_over_budget": panel["common_budget_skips"],
            "n_ref_drift": len(panel["drifts"]),
            "ref_drift_allowed": bool(args.allow_ref_drift),
            "warnings": panel["warnings"],
        },
        "design": design,
    }
    markdown = render_markdown(result)
    write_report_and_json(args, markdown, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
