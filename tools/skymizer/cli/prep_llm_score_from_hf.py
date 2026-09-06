#!/usr/bin/env python3
# =============================================================================
# prep_llm_score_from_hf.py
#
# Translate ONE row of the text-only LLM ground-truth HF dataset into the
# intermediate files that llama-llm-kld consumes:
#
#     <out>/tokens.bin              int32[L] of full input_ids
#     <out>/meta.json               n_prefill, n_answer, item_id, provenance
#
# The dataset's input_ids are the canonical token stream. This prepper never
# imports a tokenizer, never decodes the prompt for scoring, never reapplies a
# chat template, and never adds/removes BOS/EOS. n_prefill_tokens is the
# prompt/answer split and labels[n_prefill:] must match input_ids[n_prefill:].
#
# Usage:
#   python3 tools/skymizer/prep_llm_score_from_hf.py \
#       --out tmp/llm-ds-0 \
#       (--row N | --item-id ID) \
#       [--dataset skymizer/llm-ground-truth-general-fix-double-BOS] \
#       [--subset Qwen3-4B-Instruct-2507-vllm] \
#       [--split train] \
#       [--sort-by ""] [--sort-desc]
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_DATASET = "skymizer/llm-ground-truth-general-fix-double-BOS"
DEFAULT_SUBSET = "Qwen3-4B-Instruct-2507-vllm"


class PrepError(ValueError):
    """A row cannot be turned into scorer inputs because its token/label
    contract is inconsistent. Raised instead of sys.exit so collectors can mark
    one row failed and continue sweeping."""


def load_dataset_sorted(dataset, subset, split, sort_by, sort_desc=False):
    """Load and optionally sort the HF dataset. Lazy `datasets` import so this
    module stays importable for hermetic tests with only numpy installed."""
    from datasets import load_dataset
    local = Path(dataset)
    if (local / "state.json").is_file() or (local / "dataset_dict.json").is_file():
        from datasets import DatasetDict, load_from_disk
        if subset:
            raise ValueError("--subset does not apply to a local dataset")
        ds = load_from_disk(str(local))
        if isinstance(ds, DatasetDict):
            ds = ds[split]
    else:
        ds = (load_dataset(dataset, subset, split=split) if subset
              else load_dataset(dataset, split=split))
    if sort_by:
        if sort_desc:
            ds = ds.sort([sort_by], reverse=True)
        else:
            ds = ds.sort([sort_by])
    return ds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--subset", default=DEFAULT_SUBSET)
    p.add_argument("--split", default="train")
    p.add_argument("--sort-by", default="")
    p.add_argument("--sort-desc", action="store_true",
                   help="Sort --sort-by descending, matching the collectors' "
                        "--sort-desc. Required to reproduce 'row N' of a "
                        "collection that was gathered with --sort-desc; "
                        "--item-id selection is order-independent.")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--row", type=int)
    sel.add_argument("--item-id", type=str)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def pick_row(ds, args):
    if args.row is not None:
        return ds[args.row]
    want = str(args.item_id)
    matches = [i for i, x in enumerate(ds["id"]) if str(x) == want]
    if not matches:
        sys.exit(f"item_id {args.item_id!r} not found")
    return ds[matches[0]]


def sanity_check_row(row):
    L = int(row["input_tokens_len"])
    n_pre = int(row["n_prefill_tokens"])
    n_ans = int(row["generated_tokens_len"])
    input_ids = row["input_ids"]
    labels = row["labels"]

    if len(input_ids) != L:
        raise PrepError(f"len(input_ids)={len(input_ids)} != input_tokens_len={L}")
    if not (0 < n_pre < L):
        raise PrepError(f"n_prefill_tokens={n_pre} out of range for L={L}")
    if n_pre + n_ans != L:
        raise PrepError(f"n_prefill({n_pre}) + n_answer({n_ans}) != L({L})")
    if len(labels) != L:
        raise PrepError(f"len(labels)={len(labels)} != input_tokens_len={L}")
    if any(int(x) != -100 for x in labels[:n_pre]):
        raise PrepError("labels[:n_prefill] must all be -100")
    if [int(x) for x in labels[n_pre:]] != [int(x) for x in input_ids[n_pre:]]:
        raise PrepError("labels[n_prefill:] must equal input_ids[n_prefill:]")


def build_meta(row) -> dict:
    return {
        "item_id": str(row["id"]),
        "source": row["source"],
        "category": row["category"],
        "n_prefill": int(row["n_prefill_tokens"]),
        "n_answer": int(row["generated_tokens_len"]),
        "input_tokens_len": int(row["input_tokens_len"]),
        "generated_texts_preview": row["generated_texts"][:200],
        "dataset_seed": int(row["seed"]),
        "question_preview": row["question"][:200],
    }


def prep_row(row, out_dir: Path) -> dict:
    """Write tokens.bin and meta.json for one LLM dataset row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity_check_row(row)
    if row.get("num_images", 0):
        raise PrepError("LLM scoring requires text-only rows; use the VLM collector for images")
    if row.get("generation_schema_version"):
        from lib.reference_dataset import validate_reference_row
        try:
            validate_reference_row(row)
        except ValueError as error:
            raise PrepError(str(error)) from error
    np.asarray(row["input_ids"], dtype=np.int32).tofile(out_dir / "tokens.bin")

    meta = build_meta(row)
    from lib.reference_dataset import reference_provenance
    meta.update(reference_provenance(row))
    if row.get("corpus_protocol"):
        from lib.text_corpus import validate_corpus_row
        protocol, window = validate_corpus_row(row)
        meta.update(reference_vocabulary=protocol["vocabulary"], corpus_protocol=protocol, corpus_window=window)
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.dataset} :: {args.subset} :: {args.split}", file=sys.stderr)
    ds = load_dataset_sorted(args.dataset, args.subset, args.split, args.sort_by,
                             args.sort_desc)
    row = pick_row(ds, args)
    print(f"selected item_id={str(row['id'])!r}  "
          f"L={row['input_tokens_len']}  n_prefill={row['n_prefill_tokens']}",
          file=sys.stderr)

    try:
        meta = prep_row(row, args.out)
    except PrepError as e:
        sys.exit(str(e))

    print(f"wrote tokens.bin, meta.json for item_id={meta['item_id']}", file=sys.stderr)
    print(f"prep complete -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
