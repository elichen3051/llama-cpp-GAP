#!/usr/bin/env python3
"""Verify saved classic PPL windows against full-window llm-kld metrics."""

import argparse
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.collection_state import comparison_locks
from lib.kld_metrics_io import load_kld_metrics
from lib.reference_dataset import sha256_file
from lib.text_corpus import check_corpus_metrics, load_corpus_map


def read_ppl_header(path):
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(20)
        if len(header) != 20 or header[:8] != b"_logits_":
            raise ValueError("unsupported saved PPL logits header")
        size, vocabulary, chunks = struct.unpack("<III", header[8:])
        if size < 4 or size % 2 or not vocabulary or not chunks:
            raise ValueError("invalid saved PPL dimensions")
        width = 2 * ((vocabulary + 1) // 2) + 4
        offset = 20 + chunks * size * 4
        shape = (chunks, size // 2 - 1, width)
        expected_bytes = offset + math.prod(shape) * 2
        if path.stat().st_size != expected_bytes:
            raise ValueError("saved PPL logits file is incomplete or has trailing bytes")
        tokens = np.fromfile(handle, dtype="<i4", count=chunks * size)
        if tokens.size != chunks * size:
            raise ValueError("saved PPL token stream was truncated while reading")
    return {"window_size": size, "vocabulary": vocabulary, "chunks": chunks,
            "tokens": tokens.reshape(chunks, size),
            "rows": np.memmap(path, dtype="<u2", mode="r", offset=offset, shape=shape)}


def printed_number(log, pattern):
    matches = re.findall(pattern, log)
    if len(matches) != 1:
        raise ValueError(f"expected one completed PPL result matching {pattern!r}")
    value = float(matches[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("PPL result must be finite and positive")
    return value


def verify(args):
    with comparison_locks((args.llm_collection,)):
        return _verify_locked(args)


def _verify_locked(args):
    meta = json.loads((args.llm_collection / "collect_meta.json").read_text())
    from stats.collection_io import require_collection_success
    require_collection_success(args.llm_collection, "llm-kld")
    protocol, windows = load_corpus_map(args.llm_collection, meta)
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    if prepared["protocol"] != protocol:
        raise ValueError("prepared corpus and llm-kld collection protocols differ")
    stream = np.asarray(json.loads((args.prepared / "stream.json").read_text()), dtype=np.int32)
    from lib.text_corpus import token_digest
    if token_digest(stream) != protocol["stream_sha256"]:
        raise ValueError("prepared corpus token stream changed")
    saved = read_ppl_header(args.ppl_logits)
    size = protocol["window_size"]
    chunks = saved["chunks"]
    if (saved["window_size"] != size or saved["vocabulary"] != protocol["vocabulary"]["size"]
            or not np.array_equal(saved["tokens"].reshape(-1), stream[:chunks * size])):
        raise ValueError("PPL corpus tokens, vocabulary or window size differ from llm-kld")
    paths = sorted((args.llm_collection / "metrics").glob("*.npz"))
    by_key = {path.stem: path for path in paths}
    expected_keys = list(windows)[:chunks]
    if set(by_key) != set(expected_keys):
        raise ValueError("PPL and llm-kld must contain the same leading complete windows")
    ref_nll, cand_nll, kld, quantized_nll, clipped = [], [], [], [], []
    for index, key in enumerate(expected_keys):
        metrics, header = load_kld_metrics(by_key[key])
        check_corpus_metrics(key, metrics, header, protocol, windows)
        rows = saved["rows"][index]
        params = np.ascontiguousarray(rows[:, :4]).view("<f4").reshape(-1, 2)
        if not np.isfinite(params).all() or (params[:, 0] < 0).any():
            raise ValueError("invalid saved PPL quantization parameters")
        codes = rows[np.arange(len(rows)), metrics["target"] + 4]
        reconstructed = -(params[:, 0] * codes + params[:, 1])
        ref_nll.extend(metrics["nll_ref"].astype(float))
        cand_nll.extend(metrics["nll_cand"].astype(float))
        kld.extend(metrics["kld"].astype(float))
        quantized_nll.extend(reconstructed.astype(float))
        clipped.extend((codes == 0).tolist())
    logs = [path.read_text(errors="replace") for path in (args.ppl_reference_log, args.ppl_candidate_log)]
    shape = rf"n_ctx={size}, batch_size={size}, n_seq=1\b"
    if any(re.search(shape, log) is None for log in logs):
        raise ValueError("PPL logs do not confirm the required single-sequence runtime shape")
    threads = meta["n_threads"]
    metric_threads = meta["metric_threads"]
    if (threads < 1 or metric_threads != threads
            or any(re.search(rf"n_threads = {threads} \(n_threads_batch = {threads}\)", log) is None
                   or re.search(rf"metric_threads = {metric_threads}\b", log) is None for log in logs)):
        raise ValueError("PPL and llm-kld inference/batch/metric threads must agree explicitly")
    ref_ppl = printed_number(logs[0], r"Final estimate: PPL = ([0-9.eE+-]+)")
    cand_ppl = printed_number(logs[1], r"Mean PPL\(Q\)\s*:\s*([0-9.eE+-]+)")
    base_ppl = printed_number(logs[1], r"Mean PPL\(base\)\s*:\s*([0-9.eE+-]+)")
    kld_match = re.findall(r"Mean\s+KLD:\s*([0-9.eE+-]+)", logs[1])
    if len(kld_match) != 1 or not math.isfinite(float(kld_match[0])):
        raise ValueError("missing finite completed PPL KL result")
    ref_mean, cand_mean = float(np.mean(ref_nll)), float(np.mean(cand_nll))
    errors = {"reference": abs(ref_mean - math.log(ref_ppl)), "candidate": abs(cand_mean - math.log(cand_ppl))}
    tolerances = {"reference": args.nll_tolerance + 0.00005 / ref_ppl,
                  "candidate": args.nll_tolerance + 0.0000005 / cand_ppl}
    if any(errors[key] > tolerances[key] for key in errors):
        raise ValueError(f"uncompressed NLL differs across tools: errors={errors}, tolerances={tolerances}")
    quantized_mean = float(np.mean(quantized_nll))
    if abs(quantized_mean - math.log(base_ppl)) > args.nll_tolerance + 0.0000005 / base_ppl:
        raise ValueError("PPL KL base likelihood differs from its saved uint16 log probabilities")
    return {
        "status": "passed", "windows": chunks, "targets": len(ref_nll), "protocol": protocol,
        "llm_kld": {"mean_nll_ref": ref_mean, "mean_nll_cand": cand_mean, "mean_kld": float(np.mean(kld))},
        "ppl": {"reference_ppl": ref_ppl, "candidate_ppl": cand_ppl, "quantized_base_ppl": base_ppl, "mean_kld": float(kld_match[0])},
        "uncompressed_nll_errors": errors, "nll_tolerances": tolerances,
        "quantized_reference": {"mean_nll": quantized_mean, "mean_error": quantized_mean - ref_mean,
                                "max_abs_target_error": float(np.max(np.abs(np.asarray(quantized_nll) - ref_nll))),
                                "targets_at_quantization_floor": sum(clipped)},
        "saved_logits_sha256": sha256_file(args.ppl_logits),
        "interpretation": "Exact corpus tokens/windows/targets and matched uncompressed NLL; PPL KLD uses quantized, clipped reference log probabilities and is reported separately from full-vocabulary llm-kld.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "llm-collection", "ppl-logits", "ppl-reference-log", "ppl-candidate-log", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--nll-tolerance", type=float, default=1e-5,
                        help="Uncompressed mean NLL tolerance in nats, plus printed-PPL rounding")
    args = parser.parse_args()
    if not math.isfinite(args.nll_tolerance) or args.nll_tolerance <= 0:
        parser.error("--nll-tolerance must be finite and positive")
    try:
        result = verify(args)
        with args.out.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
