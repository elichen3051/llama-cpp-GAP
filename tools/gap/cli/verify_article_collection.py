#!/usr/bin/env python3
"""Verify one per-article llm-kld collection against its corpus_articles.json and the planned runtime.

Writes a check,passed,detail CSV (manifest, metrics_files, lengths, finite, targets, runtime, reference_parity, zero_kld);
every check is evaluated even after another fails. Exit 0 when all pass, 1 otherwise, 2 for usage errors (no CSV).
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.article_corpus import load_articles, token_digest
from lib.kld_metrics_io import load_kld_metrics

CHECKS = ("manifest", "metrics_files", "lengths", "finite", "targets", "runtime", "reference_parity", "zero_kld")
RUNTIME_FLAGS = ("n_ctx", "n_batch", "n_ubatch", "tf_chunk", "n_threads", "metric_threads", "n_gpu_layers")
FIXED_RUNTIME = {"kind": "llm_kld_metrics", "perplexity_window": False, "num_eval_tokens": -1,
                 "flash_attn": "enabled", "swa_full": False}
REFERENCE_COLUMNS = ("nll_ref", "argmax_ref", "entropy_ref")
SELF_COLUMNS = (("nll_cand", "nll_ref"), ("entropy_cand", "entropy_ref"), ("argmax_cand", "argmax_ref"))
NOT_REQUESTED = (True, "not requested")
GPU_WITH_UUID = re.compile(r"(.*), GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}, (.*)")


def same(a, b):
    """Equal value and type (True is not 1); arrays compare bit for bit (-0.0 is not 0.0)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return type(a) is type(b) and a == b


def summary(problems):
    if not problems:
        return True, "ok"
    shown = "; ".join(problems[:5])
    return False, shown + (f"; and {len(problems) - 5} more" if len(problems) > 5 else "")


def one_line(error):
    return " ".join(str(error).split()) or type(error).__name__


def read_record(path):
    """(metrics, header, None) of one npz record, or (None, None, reason) when it cannot be read as a complete record."""
    try:
        metrics, header = load_kld_metrics(path)
        return metrics, header, None
    except Exception as error:   # truncated, empty, not a zip, missing or misshaped arrays: never stop the verifier
        return None, None, f"{type(error).__name__}: {one_line(error)}"


def read_manifest(collection):
    with (collection / "manifest.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_manifest(articles, rows):
    problems = []
    if len(rows) != len(articles):
        problems.append(f"{len(rows)} manifest rows for {len(articles)} articles")
    for position, (article, row) in enumerate(zip(articles, rows)):
        if row.get("row_idx") != str(position) or row.get("item_id") != article["id"]:
            problems.append(f"row {position}: row_idx={row.get('row_idx')!r} item_id={row.get('item_id')!r}, expected {position} {article['id']}")
        if row.get("status") != "OK":
            problems.append(f"row {position}: status {row.get('status')!r}")
    return summary(problems)


def expected_files(articles):
    return {f"{article['index']:03d}_{article['id']}.npz": article for article in articles}


def check_metrics_files(collection, articles):
    expected = set(expected_files(articles))
    directory = collection / "metrics"
    present = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    problems = [f"missing {name}" for name in sorted(expected - present)] + [f"unexpected {name}" for name in sorted(present - expected)]
    return summary(problems)


def check_lengths(articles, rows, records, vocab_size):
    problems = []
    by_id = {row.get("item_id"): row for row in rows}
    for article in articles:
        n = article["n_targets"]
        row = by_id.get(article["id"])
        if row is None:
            problems.append(f"{article['id']}: no manifest row")
        elif (row.get("n_prefill"), row.get("n_answer"), row.get("n_eval")) != ("1", str(n), str(n)):
            problems.append(f"{article['id']}: manifest n_prefill/n_answer/n_eval {row.get('n_prefill')}/{row.get('n_answer')}/{row.get('n_eval')}, expected 1/{n}/{n}")
        if article["id"] not in records:
            continue
        path, metrics, header, error = records[article["id"]]
        if metrics is None:
            problems.append(f"{path.name}: unreadable ({error})")
            continue
        if (header["n_prefill"], header["npos"], header["vocab"]) != (1, n, vocab_size):
            problems.append(f"{path.name}: header n_prefill/npos/vocab {header['n_prefill']}/{header['npos']}/{header['vocab']}, expected 1/{n}/{vocab_size}")
        short = [name for name, values in metrics.items() if len(values) != n]
        if short:
            problems.append(f"{path.name}: arrays {short} do not have {n} values")
    return summary(problems)


def check_finite(records):
    problems = []
    for path, metrics, _, _ in records.values():
        if metrics is None:
            problems.append(f"{path.name}: unreadable")
            continue
        bad = [name for name, values in metrics.items() if values.dtype.kind == "f" and not np.isfinite(values).all()]
        if bad:
            problems.append(f"{path.name}: non-finite {bad}")
    return summary(problems)


def check_targets(articles, records):
    problems = []
    for article in articles:
        if article["id"] not in records:
            continue
        path, metrics, _, _ = records[article["id"]]
        if metrics is None:
            problems.append(f"{path.name}: unreadable")
        elif token_digest(metrics["target"]) != article["targets_sha256"]:
            problems.append(f"{path.name}: target tokens differ from targets_sha256")
    return summary(problems)


def runtime_problems(meta, expected):
    return [f"{key}={meta.get(key)!r}, expected {value!r}" for key, value in expected.items() if not same(meta.get(key), value)]


def comparable_identity(identity):
    """The execution identity as reference parity compares it (user decision 2026-10-02): every key except `environment`, GPU
    entries without their UUID ("<model>, GPU-<uuid>, <driver>" -> "<model>, <driver>") as a sorted list."""
    if not isinstance(identity, dict):
        return identity
    comparable = {key: value for key, value in identity.items() if key != "environment"}
    gpu = comparable.get("gpu")
    if isinstance(gpu, list) and all(isinstance(entry, str) for entry in gpu):
        comparable["gpu"] = sorted(f"{match[1]}, {match[2]}" if (match := GPU_WITH_UUID.fullmatch(entry)) else entry for entry in gpu)
    return comparable


def check_parity(articles, records, meta, anchor_dir):
    problems = []
    try:
        anchor_meta = json.loads((anchor_dir / "collect_meta.json").read_text())
    except (OSError, ValueError) as error:
        return False, f"parity anchor collect_meta.json unreadable: {one_line(error)}"
    for key in ("ref_model_fingerprint", *FIXED_RUNTIME, *RUNTIME_FLAGS):
        if key not in meta or key not in anchor_meta or not same(meta[key], anchor_meta[key]):
            problems.append(f"{key} differs from the parity anchor")
    if ("execution_identity" not in meta or "execution_identity" not in anchor_meta
            or not same(comparable_identity(meta["execution_identity"]), comparable_identity(anchor_meta["execution_identity"]))):
        problems.append("execution_identity (binaries, libraries, GPU model and driver) differs from the parity anchor")
    for name, article in expected_files(articles).items():
        if article["id"] not in records:
            continue
        _, metrics, _, _ = records[article["id"]]
        if metrics is None:
            problems.append(f"{name}: unreadable")
            continue
        anchor_path = anchor_dir / "metrics" / name
        if not anchor_path.is_file():
            problems.append(f"{name}: missing in the parity anchor")
            continue
        anchor, _, error = read_record(anchor_path)
        if anchor is None:
            problems.append(f"{name}: unreadable in the parity anchor ({error})")
            continue
        differ = [column for column in REFERENCE_COLUMNS if not same(metrics[column], anchor[column])]
        if differ:
            problems.append(f"{name}: reference columns {differ} differ bitwise from the parity anchor")
    return summary(problems)


def check_zero_kld(records):
    """Reference scored against itself: kld and reversed_kld exactly 0, candidate columns bitwise equal to the reference
    columns (js_kld is not checked: its log1p/LOG2 form can leave a 1-ulp residue)."""
    problems = []
    for path, metrics, _, _ in records.values():
        if metrics is None:
            problems.append(f"{path.name}: unreadable")
            continue
        nonzero = [column for column in ("kld", "reversed_kld") if not (metrics[column] == 0).all()]
        differ = [cand for cand, ref in SELF_COLUMNS if not same(metrics[cand], metrics[ref])]
        if nonzero:
            problems.append(f"{path.name}: {nonzero} not exactly 0")
        if differ:
            problems.append(f"{path.name}: {differ} differ bitwise from the reference columns")
    return summary(problems)


def evaluate(name, function, *args):
    try:
        return function(*args)
    except Exception as error:   # every check is always evaluated and reported, never a traceback
        return False, f"{name} could not be evaluated: {one_line(error)}"


def verify(args):
    receipt = load_articles(args.articles)
    articles = receipt["articles"]
    vocab_size = receipt["protocol"]["vocabulary"]["size"]
    collection = args.collection
    try:
        rows = read_manifest(collection)
    except (OSError, ValueError, csv.Error):   # unreadable or unparsable: `manifest` and `lengths` fail, the CSV is still written
        rows = []
    records = {}
    for name, article in expected_files(articles).items():
        path = collection / "metrics" / name
        if path.is_file():
            records[article["id"]] = (path, *read_record(path))
    try:
        meta = json.loads((collection / "collect_meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    expected_runtime = {**FIXED_RUNTIME, **{key: getattr(args, key) for key in RUNTIME_FLAGS}}
    results = {
        "manifest": evaluate("manifest", lambda: check_manifest(articles, read_manifest(collection))),
        "metrics_files": evaluate("metrics_files", check_metrics_files, collection, articles),
        "lengths": evaluate("lengths", check_lengths, articles, rows, records, vocab_size),
        "finite": evaluate("finite", check_finite, records),
        "targets": evaluate("targets", check_targets, articles, records),
        "runtime": evaluate("runtime", lambda: summary(runtime_problems(meta, expected_runtime)) if meta else (False, "collect_meta.json unreadable")),
        "reference_parity": evaluate("reference_parity", check_parity, articles, records, meta, args.parity_with) if args.parity_with else NOT_REQUESTED,
        "zero_kld": evaluate("zero_kld", check_zero_kld, records) if args.expect_zero_kld else NOT_REQUESTED,
    }
    with args.checks_out.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["check", "passed", "detail"])
        for check in CHECKS:
            passed, detail = results[check]
            writer.writerow([check, "true" if passed else "false", detail])
    failed = [check for check in CHECKS if not results[check][0]]
    print(f"{collection}: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""),
          file=sys.stderr)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--articles", required=True, type=Path, help="corpus_articles.json of the prepared corpus")
    parser.add_argument("--collection", required=True, type=Path, help="the candidate's llm-kld/ directory")
    for key in RUNTIME_FLAGS:
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=int, help=f"expected collect_meta.json {key}")
    parser.add_argument("--parity-with", type=Path, help="another candidate's llm-kld/ of the same model and corpus: reference columns must be bit-identical")
    parser.add_argument("--expect-zero-kld", action="store_true", help="reference scored against itself: every KLD 0, nll_cand == nll_ref")
    parser.add_argument("--checks-out", required=True, type=Path, help="new CSV: check,passed,detail")
    args = parser.parse_args()
    if os.path.lexists(args.checks_out):
        parser.error(f"--checks-out already exists, refusing to overwrite: {args.checks_out}")
    try:
        status = verify(args)
    except (ValueError, OSError) as error:
        parser.error(" ".join(str(error).split()))
    sys.exit(status)


if __name__ == "__main__":
    main()
