#!/usr/bin/env python3
"""Make text-continuation smoke data from frozen VLM reference answer tokens."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="user-company/qwen3.5-4b-500-ref")
    parser.add_argument("--subset", default="mmstar-subsample-500-ins-2048")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("paired smoke needs at least two rows")
    source = load_dataset(args.dataset, args.subset, split="train")
    rows = []
    for row in source.select(range(args.rows)):
        start = row["n_prefill_tokens"]
        tokens = row["input_ids"][start:start + 96]
        assert len(tokens) >= 80, (row["id"], "need 16 prompt + 64 answer tokens")
        assert row["labels"][start:start + len(tokens)] == tokens
        rows.append({
            "id": row["id"], "source": args.dataset, "category": row["category"],
            "question": "Continue the reference answer text from its first 16 tokens.",
            "input_ids": tokens, "input_tokens_len": len(tokens),
            "n_prefill_tokens": 16, "generated_tokens_len": len(tokens) - 16,
            "labels": [-100] * 16 + tokens[16:],
            "generated_texts": row["generated_texts"], "seed": row["seed"],
            "generation_model_name_or_path": row["generation_model_name_or_path"],
        })
    args.out.mkdir(parents=True, exist_ok=False)
    Dataset.from_list(rows).to_parquet(args.out / "train.parquet")
    provenance = {
        "dataset": args.dataset, "subset": args.subset,
        "source_fingerprint": source._fingerprint,
        "rows": [row["id"] for row in rows], "answer_prefix_tokens": 16,
        "description": "Text-only smoke: original answer token IDs, no image prompt or re-tokenization.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
