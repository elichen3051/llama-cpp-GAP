#!/usr/bin/env python3
"""Compare paired saved metrics using a validated item or corpus-group protocol."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.kld_metrics_io import KLD_METRIC_KEYS, VLMK_VERSION, kld_metric_keys
from stats.collection_io import (
    _meta_kind, _side_scores, aligned_metric_items, comparison_locks,
    load_item_pair, require_min_items, score_records, validate_collection_pair,
)
from stats.contracts import AlignmentError
from stats.engine import compare_items
from stats.render import build_execution_metadata, format_comparison_table
from stats.cli.common import (
    add_shared_output_args, add_shared_paired_args, metadata_argv,
    resolve_item_end, resolve_metrics, validate_shared_paired_args, write_report_and_json,
)


def _infer_cand_label(meta, override, fallback):
    if override:
        return override
    if meta and meta.get("cand_model"):
        return Path(meta["cand_model"]).name
    return fallback


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Paired comparison of two candidates from completed VLM or LLM "
                    "metric collections sharing one reference.")
    p.add_argument("--candidate-a", required=True, type=Path,
                   help="completed metric collection for the ref-vs-A run")
    p.add_argument("--candidate-b", required=True, type=Path,
                   help="completed metric collection for the ref-vs-B run (same reference)")
    p.add_argument("--pilot-candidate-a", type=Path,
                   help="optional completed pilot100 A collection; candidate-a then supplies tail400")
    p.add_argument("--pilot-candidate-b", type=Path,
                   help="matching pilot100 B collection; requires an explicit common token prefix")
    add_shared_paired_args(
        p,
        num_eval_tokens_help=(
            "Compare-time cap: per item, use only the first "
            "min(N, npos) stored positions. -1 = all (default)."),
        end_help=(
            "Exclusive item end after matching; default or -1 = all "
            "matched items. Values past the matched count warn and "
            "use all available items."))
    p.add_argument("--unit", choices=("item", "window", "article", "block"), default="item",
                   help="Paired sampling unit; corpus item mode means one original PPL window.")
    p.add_argument("--block-windows", type=int, default=8,
                   help="Original consecutive corpus windows per block (default: 8).")
    p.add_argument("--allow-ref-drift", action="store_true",
                   help="Allow differing stored reference columns with a recorded warning. "
                        "The comparison is approximately paired; collection completion, "
                        "runtime and execution-identity checks still apply.")
    add_shared_output_args(
        p,
        omit_host_metadata_help=(
            "drop host-volatile fields (RAM/CPU/platform) from "
            "the report and JSON so two runs of the same command "
            "are byte-identical; also the default when "
            "SKYMIZER_REPRODUCIBLE_REPORT=1"))
    return p.parse_args(argv)


def _resolve_available_metrics(metrics, requested_metrics, versions_seen, warnings):
    """Keep shared columns, but fail if an explicitly requested metric is missing."""
    version_desc = {role: "v" + "/v".join(str(v) for v in sorted(vs))
                    for role, vs in versions_seen.items()}
    if versions_seen["candidate-a"] != versions_seen["candidate-b"] or \
            any(len(vs) > 1 for vs in versions_seen.values()):
        warnings.append(
            "mixed VLMK versions: candidate-a dumps are "
            f"{version_desc['candidate-a']}, candidate-b dumps are "
            f"{version_desc['candidate-b']}; only the columns every dump "
            "carries are compared")
        print(f"WARNING: {warnings[-1]}", file=sys.stderr)
    missing_by_metric: dict[str, list[str]] = {}
    for role, vs in versions_seen.items():
        for v in sorted(vs):
            for m in metrics:
                if m in KLD_METRIC_KEYS and m not in kld_metric_keys(v):
                    missing_by_metric.setdefault(m, []).append(f"{role} (VLMK v{v})")
    for m, where in missing_by_metric.items():
        if requested_metrics and m in requested_metrics:
            sys.exit(
                f"--metrics {m} requested, but the {m!r} column is absent from "
                f"{', '.join(where)}; re-collect that dir with the current "
                "llama-{vlm,llm}-kld (writes VLMK v" + str(VLMK_VERSION) + ").")
        metrics = tuple(x for x in metrics if x != m)
        warnings.append(
            f"the {m!r} metric was dropped from this report: its column is "
            f"absent from {', '.join(where)} — re-collect that dir with the "
            "current scorer to get it")
        print(f"WARNING: {warnings[-1]}", file=sys.stderr)

    return metrics


def _format_report(result, args, *, reference_label, metrics_source, drops, n_used, warnings, drift_summary):
    table = format_comparison_table(
        result, reference_label=reference_label, display_weighting=args.weighting,
        show_diagnostic_metrics=args.show_diagnostic_metrics,
        drops=drops, n_matched=n_used,
        num_eval_tokens=args.num_eval_tokens)
    collector = "collect_llm_kld.py" if metrics_source == "llm_kld_metrics" else "collect_kld.py"
    table += (f"note: computed from on-the-fly VLMK metric dumps ({collector}); "
              "no logits were stored. KLD-family metrics are full-vocab by "
              "construction.\n")
    for w in warnings:
        table += f"warning: {w}\n"
    if drift_summary is not None and args.allow_ref_drift:
        table += (f"note: --allow-ref-drift accepted {drift_summary['n_items']} "
                  f"item(s) whose reference columns differ between the dirs "
                  f"({drift_summary['n_used']} used in the comparison; worst "
                  f"max |Δ nll_ref| = {drift_summary['max_abs_dnll_ref']:.3e}, "
                  f"{drift_summary['argmax_ref_flips']} argmax_ref flip(s)) — "
                  "the comparison is only approximately paired.\n")
    return table


def main(argv=None) -> int:
    argv_for_metadata = metadata_argv(argv, "saved_metrics_paired_compare.py")
    args = parse_args(argv)
    validate_shared_paired_args(args)
    if args.block_windows < 1:
        sys.exit("--block-windows must be >= 1")
    if bool(args.pilot_candidate_a) != bool(args.pilot_candidate_b):
        sys.exit("both --pilot-candidate-a and --pilot-candidate-b are required together")
    roots = [args.candidate_a, args.candidate_b]
    if args.pilot_candidate_a:
        roots += [args.pilot_candidate_a, args.pilot_candidate_b]
        if len({p.resolve() for p in roots}) != 4:
            sys.exit("pilot/tail requires four distinct collection directories")
        for output in (args.out, args.output_json):
            if output and any(Path(output).resolve().is_relative_to(root.resolve()) for root in roots):
                sys.exit("pilot/tail report outputs must be outside all source collection directories")
    try:
        with comparison_locks(roots):
            return _main_locked(args, argv_for_metadata)
    except ValueError as error:
        sys.exit(str(error))


def _main_locked(args, argv_for_metadata):

    for flag, d in (("--candidate-a", args.candidate_a),
                    ("--candidate-b", args.candidate_b)):
        if not d.is_dir():
            sys.exit(f"{flag}: {d} is not a directory")
        if not (d / "metrics").is_dir():
            sys.exit(f"{flag}: {d} has no metrics/ subdir — not a "
                     "collect_kld.py output dir?")

    metrics = resolve_metrics(args)
    parts = None
    if args.pilot_candidate_a:
        from stats.collection_parts import compose_pilot_tail
        records, parts, warnings = compose_pilot_tail(args)
        a_meta, b_meta = parts[1]["candidate_a_meta"], parts[1]["candidate_b_meta"]
        matched = [record[0] for record in records]
        drops = {"candidate-a": [], "candidate-b": []}
        common_budget_skips = [f"{part['part']}:{idx}" for part in parts
                               for idx in part["common_skipped_over_budget"]]
    else:
        try:
            a_meta, b_meta, warnings, common_budget_skips = validate_collection_pair(args.candidate_a, args.candidate_b)
        except (AlignmentError, ValueError) as error:
            sys.exit(str(error))

        matched, drops = aligned_metric_items(args.candidate_a, args.candidate_b, allow_interaction=args.allow_interaction)
        if not matched:
            def _describe(d):
                n_npz = len(list((Path(d) / "metrics").glob("*.npz")))
                n_bin = len(list((Path(d) / "metrics").glob("*.bin")))
                extra = (f" (+{n_bin} unconverted .bin — an interrupted/incomplete "
                         "collection; re-collect those rows into a fresh --out)") if n_bin else ""
                return f"{d}: {n_npz} .npz{extra}"
            sys.exit("no items present in both metrics dirs:\n"
                     f"  {_describe(args.candidate_a)}\n"
                     f"  {_describe(args.candidate_b)}")
        end, end_warning = resolve_item_end(args.end, len(matched))
        if end_warning:
            print(end_warning, file=sys.stderr)
        matched = matched[args.start:end]
        records = [(key, key, args.candidate_a, args.candidate_b, None) for key in matched]

    corpus_protocol, corpus_windows, groups = None, None, None
    if a_meta.get("perplexity_window"):
        from lib.text_corpus import CorpusGroups, load_corpus_map
        corpus_protocol, corpus_windows = load_corpus_map(args.candidate_a, a_meta)
        if load_corpus_map(args.candidate_b, b_meta) != (corpus_protocol, corpus_windows):
            raise ValueError("candidate corpus window maps differ")
        if args.num_eval_tokens != -1:
            raise ValueError("corpus comparisons require all stored scoring targets")
        if args.unit in ("article", "block"):
            if set(matched) != set(corpus_windows):
                raise ValueError("article/block aggregation requires the complete corpus window set; do not slice or omit windows")
            groups = CorpusGroups(args.unit, args.block_windows)
    elif args.unit != "item":
        raise ValueError("window/article/block units require a frozen corpus collection")

    scores_a, scores_b, weights, used = [], [], [], []
    # Per-token columns for the pooled distribution ladders. Keyed lazily off
    # the first kept item so a dir of v1 dumps (no `ear` column) simply has no
    # EAR ladder instead of raising.
    token_a: dict[str, list[np.ndarray]] = {}
    token_b: dict[str, list[np.ndarray]] = {}
    drifts: list[dict] = []
    versions_seen = {"candidate-a": set(), "candidate-b": set()}
    def append_unit(key, sa, sb, tok_a, tok_b, keep):
        nonlocal token_a, token_b
        scores_a.append(sa)
        scores_b.append(sb)
        cols = sorted(set(tok_a) & set(tok_b))
        if not token_a:
            token_a = {c: [] for c in cols}
            token_b = {c: [] for c in cols}
        elif sorted(token_a) != cols:
            sys.exit(f"{key}: per-token columns {cols} differ from the earlier "
                     f"items' {sorted(token_a)}; the two dirs mix VLMK versions "
                     "(re-collect so every dump carries the same columns)")
        for c in cols:
            token_a[c].append(tok_a[c])
            token_b[c].append(tok_b[c])
        weights.append(keep)
        used.append(key)

    for key, local_key, a_root, b_root, expected_header in records:
        try:
            ma, ha, mb, hb = load_item_pair(local_key, a_root, b_root)
            if expected_header is not None and any(ha[field] != value or hb[field] != value
                                                   for field, value in expected_header.items()):
                raise AlignmentError(f"{key}: stored header differs from native vocabulary/length and scoring cap")
            sa, sb, tok_a, tok_b, keep, finite, drift, versions = score_records(
                key, ma, ha, mb, hb, args.num_eval_tokens)
        except AlignmentError as e:
            sys.exit(str(e))
        versions_seen["candidate-a"].add(versions[0])
        versions_seen["candidate-b"].add(versions[1])
        if drift is not None:
            drifts.append(drift)
            if not args.allow_ref_drift:
                continue   # collect every drifted item before failing below
        if not finite:
            bad = []
            for role, scores in (("candidate-a", sa), ("candidate-b", sb)):
                names = sorted(name for name, value in scores.items()
                               if not np.isfinite(value))
                if names:
                    bad.append(f"{role}={','.join(names)}")
            sys.exit(
                f"{key}: non-finite metric score(s) ({'; '.join(bad)}); "
                "paired inference aborted")
        if corpus_protocol is not None:
            from lib.text_corpus import check_corpus_metrics
            window = check_corpus_metrics(key, ma, ha, corpus_protocol, corpus_windows)
            check_corpus_metrics(key, mb, hb, corpus_protocol, corpus_windows)
            if groups is not None:
                groups.add(window, ma, mb)
                continue
        append_unit(key, sa, sb, tok_a, tok_b, keep)

    if drifts:
        for d in drifts:
            print(f"{'WARNING' if args.allow_ref_drift else 'ERROR'}: {d['msg']}",
                  file=sys.stderr)
        if not args.allow_ref_drift:
            sys.exit(
                f"{len(drifts)} item(s) show reference drift between the two "
                "dirs: the runs do not share a bit-identical reference (different "
                "build / GPU / --tf-chunk / -ub?). The paired verdict premise is "
                "broken. Re-collect one side, or pass --allow-ref-drift to "
                "compare anyway (approximately paired).")
    # Drift keys name the source windows, before any article/block grouping.
    used_set = set(matched)
    # Persisted drift summary: the artifact must be able to distinguish 1-ulp
    # batching noise from a completely different reference model — stderr is
    # gone by the time anyone reads an archived report.
    drift_summary = None
    if drifts:
        drift_summary = {
            "n_items": len(drifts),
            "n_used": sum(1 for d in drifts if d["key"] in used_set),
            "max_abs_dnll_ref": max(d["max_dnll_ref"] for d in drifts),
            "argmax_ref_flips": sum(d["argmax_flips"] for d in drifts),
        }

    if groups is not None:
        for key, (ma, mb) in groups.records():
            keep = len(ma["target"])
            sa, tok_a = _side_scores(ma, keep)
            sb, tok_b = _side_scores(mb, keep)
            append_unit(key, sa, sb, tok_a, tok_b, keep)
    require_min_items(scores_a, matched)

    metrics = _resolve_available_metrics(metrics, args.metrics, versions_seen, warnings)

    a_label = _infer_cand_label(a_meta, args.label_a, "candidate-a")
    b_label = _infer_cand_label(b_meta, args.label_b, "candidate-b")
    # When both metas exist the cross-dir guard verified they agree; when one
    # is missing this is just "whichever survives" and ## Inputs reflects only
    # that side's claims (the persisted missing-meta warning says so).
    ref_meta = a_meta or b_meta
    metrics_source = _meta_kind(ref_meta) if ref_meta else "vlm_kld_metrics"
    ref_label = (Path(ref_meta["ref_model"]).name
                 if ref_meta and ref_meta.get("ref_model") else "reference")

    result = compare_items(
        scores_a, scores_b, weights, metrics=metrics,
        confidence_level=args.confidence_level, bootstrap_iters=args.bootstrap_iters,
        seed=args.seed, model_a_label=a_label, model_b_label=b_label,
        ci_method=args.ci_method,
        primary_metric=args.primary_metric,
        primary_weighting=args.primary_weighting,
        equivalence_margin=args.equivalence_margin,
        position_buckets=None if groups is not None else args.position_buckets,
        token_metrics_a=token_a, token_metrics_b=token_b, item_keys=used)
    if corpus_protocol is not None:
        result["sampling"] = {
            "unit": args.unit if groups is not None else "window",
            "n_units": len(used), "n_windows": len(matched),
            "block_windows": args.block_windows if args.unit == "block" else None,
            "protocol": corpus_protocol,
            "scope": "Conditional comparison of the fixed corpus; windows and adjacent groups may remain dependent. CI and test calculations treat groups as independent sampling units; residual dependence can invalidate coverage and p-values.",
        }
        result["multiplicity"]["policy"] = "One primary endpoint for a conditional fixed-corpus comparison; inferential validity depends on independent sampling units. Remaining endpoints are exploratory."
        if "per_item_tails" in result:
            result["per_item_tails"]["unit"] = "per-group quantile/maximum; witness position indexes the group's concatenated scored targets"
    result["reference_label"] = ref_label
    result["metrics_source"] = metrics_source
    result["vlmk_versions"] = {role.replace("-", "_"): sorted(vs)
                               for role, vs in versions_seen.items()}
    result["inputs"] = {
        "reference":   {"label": ref_label,
                        "model_path": ref_meta.get("ref_model") if ref_meta else None},
        "candidate_a": {"label": a_label,
                        "model_path": a_meta.get("cand_model") if a_meta else None},
        "candidate_b": {"label": b_label,
                        "model_path": b_meta.get("cand_model") if b_meta else None},
        "kind":    metrics_source,
        "dataset": ref_meta.get("dataset") if ref_meta else None,
        "subset":  ref_meta.get("subset") if ref_meta else None,
        "split":   ref_meta.get("split") if ref_meta else None,
        "sort_by": ref_meta.get("sort_by") if ref_meta else None,
    }
    if parts is not None:
        result["inputs"].update(dataset="pilot100+tail400", subset=None, sort_by=None,
                                parts=parts, comparison_token_prefix=args.num_eval_tokens)
        warnings.append("pilot100 + tail400: inference assumes independent items and is conditional on both native eligible cohorts; generation caps and repetition exclusions may differ; source ID disjointness alone does not establish image independence")
    result["execution"] = build_execution_metadata(
        args, argv_for_metadata, device="cpu", jobs=1,
        omit_host_metadata=args.omit_host_metadata)
    # The metrics were computed by the C++ dual-model scorer during collection;
    # only the lightweight paired statistics below run on CPU. Keep the two
    # stages separate so an archived report does not mislabel GPU-collected
    # metrics as a CPU evaluation. gpu_name is hardware provenance recorded by
    # the collectors, not a claim reconstructed from the comparison host.
    result["execution"]["metrics_collection"] = {
        "mode": "on-the-fly",
        "scorer_identity": a_meta["execution_identity"],
        "gpu_by_candidate": {
            "candidate-a": (a_meta or {}).get("gpu_name") or "unknown",
            "candidate-b": (b_meta or {}).get("gpu_name") or "unknown",
        },
    }
    result["alignment"] = {
        "n_matched": len(matched),
        "n_statistical_units": len(used),
        "collection_complete": True,
        "execution_identity_verified": True,
        "drops": drops,
        "warnings": warnings,
        "allow_interaction": bool(args.allow_interaction),
        "n_common_skipped_over_budget": len(common_budget_skips),
        "common_skipped_over_budget": common_budget_skips,
        "n_ref_drift": len(drifts),
        "ref_drift_allowed": bool(args.allow_ref_drift),
    }
    if drift_summary is not None:
        result["alignment"]["ref_drift"] = drift_summary

    table = _format_report(
        result, args, reference_label=ref_label, metrics_source=metrics_source,
        drops=drops, n_used=len(used), warnings=warnings, drift_summary=drift_summary)
    write_report_and_json(args, table, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
