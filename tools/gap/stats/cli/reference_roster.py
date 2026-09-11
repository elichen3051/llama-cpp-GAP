#!/usr/bin/env python3
"""Export verified image clusters and optional completed-collection roster fields."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli.prep_vlm_score_from_hf import load_dataset_sorted
from stats.reference_roster import bind_reference_roster, reference_identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="local native Dataset.save_to_disk directory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--expected-item-ids", type=Path, help="JSON list required for a partial collection")
    parser.add_argument("--out", required=True, type=Path, help="new JSON output path")
    args = parser.parse_args(argv)
    if not (args.reference / "state.json").is_file() and not (args.reference / "dataset_dict.json").is_file():
        parser.error("--reference must be a local saved dataset; Hub loading is not supported here")
    if args.expected_item_ids and not args.collection:
        parser.error("--expected-item-ids requires --collection")
    try:
        dataset = load_dataset_sorted(str(args.reference), None, args.split, None)
        if args.collection:
            expected = json.loads(args.expected_item_ids.read_text()) if args.expected_item_ids else None
            result = bind_reference_roster(dataset, args.collection, expected, args.source_field)
        else:
            result = reference_identities(dataset, args.source_field)
        result["reference"] = {"local_path": str(args.reference.resolve()), "split": args.split}
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(encoded)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"reference roster: {error}\n")
    print(f"Reference pool: {result['n_items']} items, {result['n_clusters']} image clusters; {result['binding_status']}; {args.out}")
    if args.collection:
        print(f"Bound collection: {result['n_bound_items']} items, {result['n_bound_clusters']} image clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
