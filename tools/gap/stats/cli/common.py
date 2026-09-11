"""Shared arguments and report output for statistical CLIs."""
import json
import math
import sys
from pathlib import Path

from stats.contracts import (
    BOOTSTRAP_ITERS_ADVISORY,
    CI_METHODS,
    DEFAULT_CI_METHOD,
    DEFAULT_METRICS,
    DEFAULT_PRIMARY_WEIGHTING,
    min_bootstrap_iters,
)
from stats.tokens import DEFAULT_POSITION_BUCKETS
from stats.render import _reproducible_report_env


def metadata_argv(argv, script_name: str) -> list[str]:
    """The command line recorded in the report: the real one, or the given
    argv behind the script's name when main() is called programmatically."""
    return [str(x) for x in (sys.argv if argv is None else [script_name, *argv])]


def resolve_item_end(end_arg: int | None, n_items: int) -> tuple[int, str | None]:
    if end_arg is None or end_arg == -1:
        return n_items, None
    if end_arg > n_items:
        return (
            n_items,
            f"WARNING: --end {end_arg} exceeds matched item count {n_items}; "
            f"falling back to {n_items} (all available items).",
        )
    return end_arg, None


def add_shared_paired_args(p, *, num_eval_tokens_help: str, end_help: str) -> None:
    """Add report arguments with caller-specific selection descriptions."""
    p.add_argument("--out", required=True, type=Path, help="Markdown report path")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--metrics", nargs="*", default=None,
                   help="Base metrics (default: " + " ".join(DEFAULT_METRICS) + ")")
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--position-buckets", type=int, nargs="+",
                   default=list(DEFAULT_POSITION_BUCKETS),
                   help="ascending answer-position edges (first must be 0) for "
                        "the exploratory position-strata table; the last "
                        "bucket is open-ended. Default 0 32 256. "
                        "Pass a single 0 to report one "
                        "bucket, i.e. effectively disable the breakdown.")
    p.add_argument("--equivalence-margin", type=float, default=None,
                   help="TOST margin for the PRIMARY metric, in that metric's "
                        "own units (nats for kld, pp^2 for mse_dp, ...). With "
                        "it, a CI strictly inside +-margin establishes "
                        "equivalence independently of the directional verdict; "
                        "EQUIVALENT labels an otherwise inconclusive cell. "
                        "One-sided alpha=(1-confidence-level)/2. Without it, "
                        "every non-significant cell still reports the tightest "
                        "margin its own interval rules out.")
    p.add_argument("--primary-metric", default=None,
                   choices=list(DEFAULT_METRICS),
                   help="the ONE confirmatory endpoint (default: kld). Its "
                        "interval spends the whole alpha; every other "
                        "item-weighted metric is reported as exploratory "
                        "with a Holm-adjusted p-value.")
    p.add_argument("--primary-weighting", default=DEFAULT_PRIMARY_WEIGHTING,
                   choices=["item"],
                   help="paired inference is item-weighted only; token weighting is descriptive")
    p.add_argument("--ci-method", choices=CI_METHODS, default=DEFAULT_CI_METHOD,
                   help="interval construction. 't' (default) is the classical "
                        "paired Student-t interval on the per-item deltas "
                        "(analytic SE, df = n-1; no bootstrap, so no seed and "
                        "no replicate count enter the result). 'studentized' "
                        "is the bootstrap-t: each replicate is divided by its "
                        "own analytic SE. 'bca' is bias-corrected and "
                        "accelerated. 'percentile' is the first-order "
                        "interval. Nonconstant two-item samples, unsupported "
                        "tails and unusable studentized pivots fail explicitly. "
                        "docs/compare.md explains assumptions and limitations.")
    p.add_argument("--bootstrap-iters", type=int, default=5000,
                   help="replicates for the bootstrap methods (default 5000); "
                        "ignored by --ci-method t. Bootstrap CIs require ten draws "
                        "strictly outside each endpoint; p-values beyond that "
                        "supported range are reported as conservative upper bounds.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--weighting", choices=["item", "token", "both"], default="both")
    p.add_argument("--num-eval-tokens", type=int, default=-1, help=num_eval_tokens_help)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help=end_help)
    p.add_argument(
        "--allow-interaction", "--allow-intersection",
        dest="allow_interaction", action="store_true",
        help="Explicitly allow paired inference on the clean item intersection "
             "when one input dir is missing artifact keys. The default is to "
             "fail before inference. This never permits non-finite/failed/partial "
             "artifacts, reference drift, or mismatched SKIP_OVER_BUDGET sets.")



def add_shared_output_args(p, *, omit_host_metadata_help: str) -> None:
    p.add_argument("--show-diagnostic-metrics", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--omit-host-metadata", action="store_true",
                   default=_reproducible_report_env(),
                   help=omit_host_metadata_help)


def validate_shared_paired_args(args) -> None:
    """The validation preamble both main()s run first; every message
    verbatim from the tools."""
    if args.num_eval_tokens != -1 and args.num_eval_tokens < 1:
        sys.exit(f"--num-eval-tokens must be -1 (all) or >= 1, got {args.num_eval_tokens}")
    if args.start < 0:
        sys.exit(f"--start must be >= 0, got {args.start} (negative Python "
                 "slice semantics would silently shrink the sample)")
    if args.end is not None and args.end < -1:
        sys.exit(f"--end must be -1 (all) or >= 0, got {args.end}")
    if not (0.0 < args.confidence_level < 1.0):
        sys.exit(f"--confidence-level must be in (0, 1), got {args.confidence_level}")
    if args.equivalence_margin is not None and (not math.isfinite(args.equivalence_margin) or args.equivalence_margin <= 0.0):
        sys.exit("--equivalence-margin must be finite and > 0 (it is a two-sided TOST "
                 f"margin), got {args.equivalence_margin}")
    if args.ci_method == "t":
        return                       # no replicates are drawn; nothing to floor
    floor = min_bootstrap_iters(args.confidence_level, args.ci_method)
    if args.bootstrap_iters < floor:
        sys.exit(
            f"--bootstrap-iters must be >= {floor} at --confidence-level "
            f"{args.confidence_level:g} with --ci-method {args.ci_method}, "
            f"got {args.bootstrap_iters}. This nominal floor is a prefilter; "
            "actual CI endpoints must also have ten strict-outside draws per tail.")
    if args.bootstrap_iters < BOOTSTRAP_ITERS_ADVISORY:
        print(f"WARNING: --bootstrap-iters {args.bootstrap_iters} is below the "
              f"customary {BOOTSTRAP_ITERS_ADVISORY}; the CI endpoints carry "
              "visible Monte-Carlo noise and a borderline verdict may flip "
              "between seeds.", file=sys.stderr)


def resolve_metrics(args) -> tuple[str, ...]:
    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    invalid = [m for m in metrics if m not in DEFAULT_METRICS]
    if invalid:
        sys.exit(f"unknown --metrics {invalid}; choose from {list(DEFAULT_METRICS)} "
                 "(ppl, ppl_ratio and rms_dp are derived and cannot be selected "
                 "directly; the pooled per-token kld/ear distribution ladders "
                 "are a descriptive block, always emitted when the per-token "
                 "columns are available)")
    return metrics



def write_report_and_json(args, table: str, result) -> None:
    """Serialize strict JSON before writing either report, and create output directories."""
    if args.output_json and args.out.resolve() == args.output_json.resolve():
        raise ValueError("report and JSON paths must differ")
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n" if args.output_json else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print(f"wrote report -> {args.out}", file=sys.stderr)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
        print(f"wrote json -> {args.output_json}", file=sys.stderr)
