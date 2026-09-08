#!/usr/bin/env python3
"""Compare a declared quantization campaign or select a training-only benchmark."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stats.campaign import analyze_campaign, load_campaign, select_benchmark, validate_selection_plan
from stats.cli.common import write_report_and_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "select"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output_json is None:
        args.output_json = args.out.with_name(args.out.name + ".json")
    if args.out.resolve() == args.output_json.resolve():
        parser.error("report and JSON paths must differ")
    if args.manifest.resolve() in (args.out.resolve(), args.output_json.resolve()):
        parser.error("output paths must not overwrite the input manifest")
    try:
        plan = json.loads(args.manifest.read_text())
        if args.action == "select":
            validate_selection_plan(plan, plan.get("selection_rule", {}))
        loaded = load_campaign(plan, args.manifest.parent)
        if args.action == "select":
            result = select_benchmark(plan, loaded, plan.get("selection_rule", {}))
            report = f"# Training-only benchmark selection\n\nSelected {len(result['selected_item_ids'])} of {result['pool_size']} items.\n\nThis greedy training result requires evaluation on the declared heldout families. It is not a global optimum or a validated detection guarantee.\n"
        else:
            result = analyze_campaign(plan, loaded)
            lines = ["# Quantization campaign", "", f"{result['family_size']} paired hypotheses; correction: {result['correction']} ({result['error_control']}).", "", "Prespecification and cluster identities are user declared. Cluster inference is approximate; fixed-N significance is not a power guarantee.", "", "| Cell | A | B | Items | Clusters | B-A | Adjusted p | Reject equal means |", "|---|---|---|---:|---:|---:|---:|---|"]
            for row in result["pairs"]:
                lines.append(f"| {row['cell_id']} | {row['quant_a']} | {row['quant_b']} | {row['n_items']} | {row['n_clusters']} | {row['estimate']:.6g} | {row['p_adjusted']:.6g} | {row['reject_equal_mean']} |")
            lines.extend(["", "| Cell | Descriptive minimum KLD | Possible best set | Unique best by simultaneous intervals |", "|---|---|---|---|"])
            for cell in result["cells"]:
                possible = ", ".join(cell["possible_best_by_simultaneous_intervals"])
                lines.append(f"| {cell['cell_id']} | {cell['descriptive_minimum']} | {possible} | {cell['unique_best_by_simultaneous_intervals'] or 'unresolved'} |")
            lines.extend(["", "Possible-best sets use separate Bonferroni simultaneous intervals. See JSON for intervals, descriptive rankings, provenance and the complete plan."])
            report = "\n".join(lines) + "\n"
        write_report_and_json(args, report, result)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"campaign: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
