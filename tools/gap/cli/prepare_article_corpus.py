#!/usr/bin/env python3
"""Freeze a text corpus as one native-tokenized sequence per article (every token after position 0 is scored)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.article_corpus import FIXED_PROTOCOL, article_row, article_spans, articles_digest, bos_split, row_id, token_digest
from lib.model_files import sha256_file
from lib.reference_dataset import canonical_json


def run_json(command, log, data=None):
    """Run a native tool on the CPU, its stderr into `log`, and parse its stdout as JSON."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with Path(log).open("wb") as error:
        result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=error, env=env)
    if result.returncode:
        raise ValueError(f"native command failed ({result.returncode}); see {log}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"native command printed no JSON; see {log}") from error


def tokenize(command, data, logs, index):
    with_bos = run_json(command, logs / f"tokenize-{index:04d}.log", data)
    without_bos = run_json([*command, "--no-bos"], logs / f"tokenize-{index:04d}-no-bos.log", data)
    if not all(isinstance(tokens, list) and all(type(t) is int for t in tokens) for tokens in (with_bos, without_bos)):
        raise ValueError(f"article {index}: tokenizer did not print a list of token ids")
    return with_bos, bos_split(with_bos, without_bos)


def prepare(args):
    raw = args.corpus.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"corpus is not valid UTF-8 at byte {error.start}") from error
    if b"\x00" in raw:
        raise ValueError("corpus contains NUL bytes")
    index_bytes = args.article_index.read_bytes()
    try:
        index = json.loads(index_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"article index is not JSON: {error}") from error
    spans = article_spans(raw, index)

    logs = args.out / "logs"
    logs.mkdir(parents=True)
    vocabulary = run_json([str(args.llama_llm_kld), "--vocab-identity", str(args.ref_model)], logs / "vocabulary.log")
    command = [str(args.llama_tokenize), "-m", str(args.ref_model), "--stdin", "--ids", "--no-escape", "--no-parse-special"]
    sequences, bos_ids = [], set()
    for i, span in enumerate(spans):
        tokens, bos = tokenize(command, raw[span["byte_start"]:span["byte_end_exclusive"]], logs, i)
        if len(tokens) < 2:
            raise ValueError(f"article {i} has {len(tokens)} token(s); at least two are needed to score one")
        sequences.append(tokens)
        bos_ids.add(bos)
        print(f"article {i + 1}/{len(spans)}: {len(tokens)} tokens", flush=True)
    if len(bos_ids) != 1:
        raise ValueError("the tokenizer added a BOS to some articles but not to others")

    records = [{"index": i, **span, "n_tokens": len(tokens), "n_targets": len(tokens) - 1,
                "tokens_sha256": token_digest(tokens), "targets_sha256": token_digest(tokens[1:])}
               for i, (span, tokens) in enumerate(zip(spans, sequences))]
    protocol = {
        **FIXED_PROTOCOL, "corpus_name": args.corpus_name,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(), "corpus_bytes": len(raw),
        "article_index_sha256": hashlib.sha256(index_bytes).hexdigest(), "articles_sha256": articles_digest(records),
        "bos_id": bos_ids.pop(), "vocabulary": vocabulary, "tokenizer_binary_sha256": sha256_file(args.llama_tokenize),
    }
    for record in records:
        record["id"] = row_id(protocol, record["index"])
    rows = [article_row(protocol, record, tokens) for record, tokens in zip(records, sequences)]
    receipt = {"protocol": protocol, "articles": records,
               "total_tokens": sum(r["n_tokens"] for r in records), "total_targets": sum(r["n_targets"] for r in records)}

    from datasets import Dataset
    Dataset.from_list(rows).save_to_disk(str(args.out / "dataset"))
    # Written last: its presence marks a complete preparation.
    partial = args.out / "corpus_articles.json.partial"
    partial.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    partial.replace(args.out / "corpus_articles.json")
    print(f"prepared {len(rows)} articles, {receipt['total_targets']} scored tokens: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--corpus", required=True, type=Path, help="corpus text; each article is tokenized from its exact bytes")
    parser.add_argument("--corpus-name", required=True, help="protocol label, e.g. wikitext-2-test-article or pg-full-rss-article")
    parser.add_argument("--article-index", required=True, type=Path,
                        help="JSON with an 'articles' or 'items' list of byte_start/byte_end_exclusive spans partitioning the corpus")
    parser.add_argument("--ref-model", required=True, type=Path, help="reference GGUF whose tokenizer and vocabulary define the tokens")
    parser.add_argument("--llama-tokenize", required=True, type=Path)
    parser.add_argument("--llama-llm-kld", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new directory: dataset/, logs/, corpus_articles.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.corpus_name):
        parser.error("--corpus-name must use lowercase letters, digits, and hyphens")
    if os.path.lexists(args.out):
        parser.exit(1, f"output already exists, refusing to overwrite: {args.out}\n")
    try:
        prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, " ".join(str(error).split()) + "\n")


if __name__ == "__main__":
    main()
