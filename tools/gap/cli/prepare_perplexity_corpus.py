#!/usr/bin/env python3
"""Freeze native GGUF tokenization into classic llama-perplexity windows."""

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.reference_dataset import canonical_json, sha256_file
from lib.text_corpus import SCHEMA, corpus_row, digest_json, stable_prefix_boundary, token_digest, validate_protocol


def article_spans(raw, name, index_path=None):
    if index_path:
        index = json.loads(Path(index_path).read_text())
        expected = index.get("corpus_sha256")
        if expected and expected != hashlib.sha256(raw).hexdigest():
            raise ValueError("article index corpus SHA256 mismatch")
        spans = index["articles"] if "articles" in index else index["items"]
        spans = [{"id": str(entry.get("id", entry.get("index", i))),
                  "title": entry.get("title", entry.get("rss_title", "")),
                  "byte_start": entry["byte_start"], "byte_end": entry["byte_end_exclusive"]}
                 for i, entry in enumerate(spans)]
    elif name == "wikitext-2-test":
        headers = list(re.finditer(rb"(?m)^ = ([^=\r\n]+) = *\r?$", raw))
        if not headers or raw[:headers[0].start()].strip():
            raise ValueError("WikiText needs top-level headers and a whitespace-only prelude")
        spans = [{"id": str(i), "title": match[1].decode("utf-8"),
                  "byte_start": match.start() if i else 0,
                  "byte_end": headers[i + 1].start() if i + 1 < len(headers) else len(raw)}
                 for i, match in enumerate(headers)]
    else:
        raise ValueError("--article-index is required outside WikiText-2")
    cursor = 0
    for span in spans:
        if span["byte_start"] != cursor or not cursor < span["byte_end"] <= len(raw):
            raise ValueError("article spans must partition the exact corpus bytes without gaps")
        span["sha256"] = hashlib.sha256(raw[cursor:span["byte_end"]]).hexdigest()
        cursor = span["byte_end"]
    if cursor != len(raw):
        raise ValueError("article index does not cover the whole corpus")
    return spans


def run_json(command, log, data=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with Path(log).open("wb") as error:
        result = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=error, env=env)
    if result.returncode:
        raise ValueError(f"native command failed ({result.returncode}); see {log}")
    return json.loads(result.stdout)


def prepare(args):
    if args.out.exists():
        raise ValueError(f"output already exists: {args.out}")
    raw = args.corpus.read_bytes()
    raw.decode("utf-8")
    if b"\x00" in raw:
        raise ValueError("corpus contains NUL bytes")
    spans = article_spans(raw, args.corpus_name, args.article_index)
    effective = raw[:-1] if raw.endswith(b"\n") else raw
    args.out.mkdir(parents=True)
    (args.out / "corpus.txt").write_bytes(raw)
    (args.out / "effective.txt").write_bytes(effective)
    command = [str(args.llama_tokenize.resolve()), "-m", str(args.ref_model.resolve()),
               "--stdin", "--ids", "--no-escape", "--no-parse-special"]
    stream = run_json(command, args.out / "tokenize.log", effective)
    no_bos = run_json(command + ["--no-bos"], args.out / "tokenize-no-bos.log", effective)
    bos = None
    if stream != no_bos:
        if len(stream) != len(no_bos) + 1 or stream[1:] != no_bos:
            raise ValueError("native tokenizer differs by more than one leading BOS")
        bos = stream[0]
    vocabulary = run_json([str(args.llama_llm_kld.resolve()), "--vocab-identity", str(args.ref_model.resolve())], args.out / "vocabulary.log")
    boundaries = [0]
    for i, span in enumerate(spans[1:], 1):
        prefix = run_json(command, args.out / f"boundary-{i:04d}.log", effective[:span["byte_start"]])
        boundary = stable_prefix_boundary(stream, prefix)
        if boundary <= boundaries[-1]:
            raise ValueError("article token boundaries are not strictly increasing")
        span["prefix_tokens"] = len(prefix)
        span["unstable_prefix_tokens"] = len(prefix) - boundary
        boundaries.append(boundary)
        print(f"article boundary {i}/{len(spans) - 1}: token {boundary}", flush=True)
    for i, span in enumerate(spans):
        span["token_start"] = boundaries[i]
        span["token_end"] = boundaries[i + 1] if i + 1 < len(spans) else len(stream)
    protocol = {
        "schema": SCHEMA, "corpus_name": args.corpus_name,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "effective_text_sha256": hashlib.sha256(effective).hexdigest(),
        "strip_single_final_newline": True, "escape": False, "parse_special": False,
        "stream_tokens": len(stream), "stream_sha256": token_digest(stream),
        "vocabulary": vocabulary, "bos_id": bos, "bos_policy": "replace-window-position-zero",
        "window_size": args.window_size, "stride": args.window_size,
        "n_prefill": args.window_size // 2 + 1, "targets_per_window": args.window_size // 2 - 1,
        "tail_policy": "drop-incomplete-window", "articles_sha256": digest_json(spans),
        "article_boundary_rule": "longest-stable-prefix; first boundary-affected token belongs to next article",
        "tokenizer_binary_sha256": sha256_file(args.llama_tokenize),
    }
    validate_protocol(protocol)
    rows = []
    for index in range(len(stream) // args.window_size):
        offset = index * args.window_size
        tokens = stream[offset:offset + args.window_size]
        if bos is not None:
            tokens[0] = bos
        articles = [bisect.bisect_right(boundaries, position) - 1
                    for position in range(offset + protocol["n_prefill"], offset + args.window_size)]
        rows.append(corpus_row(protocol, tokens, index, articles))
    from datasets import Dataset
    Dataset.from_list(rows).save_to_disk(str(args.out / "dataset"))
    receipt = {"protocol": protocol, "articles": spans,
               "windows": [json.loads(row["corpus_window"]) for row in rows],
               "dropped_tail_tokens": len(stream) % args.window_size,
               "tokenizer_command": command, "reference_model": str(args.ref_model.resolve())}
    (args.out / "manifest.json").write_text(canonical_json(receipt) + "\n")
    (args.out / "stream.json").write_text(canonical_json(stream) + "\n")
    print(f"prepared {len(rows)} windows, {len(spans)} articles: {args.out / 'dataset'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--article-index", type=Path, help="JSON articles with exact byte_start/byte_end_exclusive spans")
    parser.add_argument("--ref-model", required=True, type=Path)
    parser.add_argument("--llama-tokenize", required=True, type=Path)
    parser.add_argument("--llama-llm-kld", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.window_size < 4 or args.window_size % 2:
        parser.error("--window-size must be even and at least 4")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.corpus_name):
        parser.error("--corpus-name must use lowercase letters, digits, and hyphens")
    try:
        prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
