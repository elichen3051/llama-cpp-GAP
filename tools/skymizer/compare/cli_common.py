# compare/cli_common.py -- the argparse / validation / output scaffolding the
# two comparators share (paired_compare over dense logit dirs and
# saved_metrics_paired_compare over VLMK metric dirs). Their parse_args and
# main() bodies were copy-pasted and had already drifted once (the stale
# --ci-method help that commit 0f13f361 had to sweep); every shared piece now
# has one copy here, and the three help strings that legitimately differ are
# parameters.
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from compare.contracts import (
    BOOTSTRAP_ITERS_ADVISORY,
    CI_METHODS,
    DEFAULT_CI_METHOD,
    DEFAULT_METRICS,
    DEFAULT_PRIMARY_WEIGHTING,
    min_bootstrap_iters,
)
from compare.tokens import DEFAULT_POSITION_BUCKETS
from compare.render import _reproducible_report_env
from lib.collect_common import SKIP_OVER_BUDGET, manifest_row_statuses


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
    """--out through --end: identical in both tools (same defaults, same
    help) apart from the two help strings passed in."""
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
                        "bucket is open-ended. Default 0 32 256, because the "
                        "measured mmproj signal on this data sits in the first "
                        "~32 answer tokens. Pass a single 0 to report one "
                        "bucket, i.e. effectively disable the breakdown.")
    p.add_argument("--equivalence-margin", type=float, default=None,
                   help="TOST margin for the PRIMARY metric, in that metric's "
                        "own units (nats for kld, pp^2 for mse_dp, ...). With "
                        "it, a primary cell whose CI lies entirely inside "
                        "+-margin is reported EQUIVALENT rather than merely "
                        "inconclusive (interval-inclusion TOST). Without it, "
                        "every non-significant cell still reports the tightest "
                        "margin its own interval rules out.")
    p.add_argument("--primary-metric", default=None,
                   choices=list(DEFAULT_METRICS),
                   help="the ONE confirmatory endpoint (default: kld). Its "
                        "interval spends the whole alpha; every other "
                        "metric x weighting cell is reported as exploratory "
                        "with a Holm-adjusted p-value.")
    p.add_argument("--primary-weighting", default=DEFAULT_PRIMARY_WEIGHTING,
                   choices=["item", "token"],
                   help="weighting of the confirmatory endpoint (default: "
                        "item, the exchangeable unit the bootstrap resamples "
                        "-- its CI generalizes to unseen items from the same "
                        "population; 'token' is llama-perplexity's corpus "
                        "aggregation, kept for cross-tool comparability)")
    p.add_argument("--ci-method", choices=CI_METHODS, default=DEFAULT_CI_METHOD,
                   help="interval construction. 't' (default) is the classical "
                        "paired Student-t interval on the per-item deltas "
                        "(analytic SE, df = n-1; no bootstrap, so no seed and "
                        "no replicate count enter the result). 'studentized' "
                        "is the bootstrap-t: each replicate is divided by its "
                        "own analytic SE. 'bca' is bias-corrected and "
                        "accelerated. 'percentile' is the first-order "
                        "interval, kept for cross-repo parity checks. "
                        "docs/compare.md tabulates their measured coverage.")
    p.add_argument("--bootstrap-iters", type=int, default=5000,
                   help="replicates for the bootstrap methods (default 5000); "
                        "ignored by --ci-method t")
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
    if args.equivalence_margin is not None and args.equivalence_margin <= 0.0:
        sys.exit("--equivalence-margin must be > 0 (it is a two-sided TOST "
                 f"margin), got {args.equivalence_margin}")
    if args.ci_method == "t":
        return                       # no replicates are drawn; nothing to floor
    floor = min_bootstrap_iters(args.confidence_level, args.ci_method)
    if args.bootstrap_iters < floor:
        sys.exit(
            f"--bootstrap-iters must be >= {floor} at --confidence-level "
            f"{args.confidence_level:g} with --ci-method {args.ci_method}, "
            f"got {args.bootstrap_iters}. Fewer "
            "replicates cannot resolve the requested percentile: the CI "
            "endpoints collapse towards the min/max of the replicate sample "
            "(at 1 iteration lower == upper, so EVERY metric reads as "
            "significant).")
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



def require_common_budget_skips(roots: Mapping[str, Path]) -> list[int]:
    """Require every available manifest to declare the same budget-skip set.

    A common SKIP_OVER_BUDGET set is an intentional corpus filter and is safe
    for pairing. A different set changes the sample per side and always aborts;
    --allow-interaction cannot override it. If one manifest is absent while
    another declares skips, the equality cannot be verified, so fail closed.
    """
    skip_sets: dict[str, set[int] | None] = {}
    for role, root in roots.items():
        manifest = Path(root) / "manifest.csv"
        if not manifest.exists():
            skip_sets[role] = None
            continue
        try:
            statuses = manifest_row_statuses(manifest)
        except ValueError as e:
            sys.exit(str(e))
        skip_sets[role] = {
            idx for idx, status in statuses.items()
            if status == SKIP_OVER_BUDGET
        }

    declared = {role: rows for role, rows in skip_sets.items() if rows is not None}
    missing_manifests = [role for role, rows in skip_sets.items() if rows is None]
    any_declared_skip = any(rows for rows in declared.values())
    if any_declared_skip and missing_manifests:
        print(
            "ERROR: SKIP_OVER_BUDGET alignment cannot be verified because "
            f"manifest.csv is missing for {', '.join(missing_manifests)}",
            file=sys.stderr)
        sys.exit("paired inference aborted: common budget-skip set is unverified")

    distinct = {tuple(sorted(rows)) for rows in declared.values()}
    if len(distinct) > 1:
        print("ERROR: SKIP_OVER_BUDGET sets differ across paired inputs:",
              file=sys.stderr)
        for role, rows in skip_sets.items():
            shown = "manifest missing" if rows is None else str(sorted(rows))
            print(f"  {role}: {shown}", file=sys.stderr)
        sys.exit("paired inference aborted: SKIP_OVER_BUDGET set mismatch")

    common = list(next(iter(distinct))) if distinct else []
    if common:
        sample = ", ".join(str(idx) for idx in common[:10])
        more = "" if len(common) <= 10 else f", … (+{len(common) - 10})"
        print(
            f"INFO: {len(common)} row(s) excluded identically from all paired "
            f"inputs as {SKIP_OVER_BUDGET}: {sample}{more}",
            file=sys.stderr)
    return common


def require_complete_item_alignment(
    drops: Mapping[str, Sequence[str]], *, allow_interaction: bool
) -> None:
    """Abort on any one-sided artifact-key gap unless explicitly overridden."""
    missing = [(role, list(keys)) for role, keys in drops.items() if keys]
    if not missing:
        return
    level = "WARNING" if allow_interaction else "ERROR"
    print(f"{level}: paired input data has one-sided missing item(s):",
          file=sys.stderr)
    for role, keys in missing:
        sample = ", ".join(keys[:10])
        more = "" if len(keys) <= 10 else f", … (+{len(keys) - 10})"
        print(f"  {role} missing {len(keys)}: {sample}{more}", file=sys.stderr)
    if allow_interaction:
        print("WARNING: --allow-interaction enabled; paired inference will use "
              "only the clean item intersection", file=sys.stderr)
        return
    sys.exit(
        "paired inference aborted: input item sets differ; re-collect the "
        "missing data or pass --allow-interaction to explicitly use the intersection")


def require_min_items(scores_a, matched) -> None:
    """Fail closed: a single item makes the bootstrap CI zero-width, which
    would report spurious "significant" verdicts. Paired inference needs
    >= 2 items."""
    if len(scores_a) < 2:
        sys.exit(
            f"paired inference needs >= 2 usable items; only {len(scores_a)} usable "
            f"(matched {len(matched)} after --start/--end). "
            "Widen the item range or check the inputs.")


def write_report_and_json(args, table: str, result) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print(f"wrote report -> {args.out}", file=sys.stderr)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote json -> {args.output_json}", file=sys.stderr)
