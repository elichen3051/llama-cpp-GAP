#!/usr/bin/env python3
"""Collect two candidates and an independent repeat, then verify paired reports."""

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

COMPANY = Path(__file__).resolve().parents[1]
REPO = COMPANY.parents[1]
sys.path.insert(0, str(COMPANY))
from lib.kld_metrics_io import VLMK_VERSION


def run(cmd, log):
    print(shlex.join(map(str, cmd)), flush=True)
    with log.open("x") as stream:
        subprocess.run(list(map(str, cmd)), check=True, stdout=stream, stderr=subprocess.STDOUT)


def read_collection(directory, rows, cap):
    with (directory / "manifest.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == rows, (directory, manifest)
    assert all(row["status"] == "OK" for row in manifest), manifest
    files = sorted((directory / "metrics").glob("*.npz"))
    assert len(files) == rows, (directory, files)
    result = {}
    for path in files:
        with np.load(path) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        assert int(data["version"]) == VLMK_VERSION
        assert 0 < int(data["npos"]) <= cap
        assert int(data["n_past_actual"]) > 0
        for key, column in data.items():
            if column.dtype.kind == "f":
                assert np.isfinite(column).all(), (path, key)
        assert (data["kld"] >= -1e-6).all(), path
        assert ((data["ear"] >= 0) & (data["ear"] <= 1)).all(), path
        result[path.name] = data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=REPO / "build/bin")
    parser.add_argument("--lane", choices=("vlm", "llm"), default="vlm")
    parser.add_argument("--ref-model", required=True)
    parser.add_argument("--cand-a-model", required=True)
    parser.add_argument("--cand-b-model", required=True)
    parser.add_argument("--mmproj", help="Shared projector for VLM quantization smoke")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", default="")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--num-eval-tokens", type=int, default=64)
    parser.add_argument("--image-min-tokens", type=int)
    parser.add_argument("--image-max-tokens", type=int)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--tf-chunk", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing output is refused")
    parser.add_argument("--verify-only", action="store_true", help="Validate existing collections and reports without rescoring")
    args = parser.parse_args()
    if args.rows < 2 or args.num_eval_tokens < 1:
        parser.error("paired smoke needs >= 2 rows and a positive token cap")
    if args.lane == "vlm" and not args.mmproj:
        parser.error("--mmproj is required for VLM smoke")
    args.out = args.out.resolve()
    if args.verify_only:
        assert args.out.is_dir(), args.out
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    binary = args.bin_dir.resolve() / f"llama-{args.lane}-kld"
    if not args.verify_only:
        run([binary, "--self-test"], args.out / "self-test.log")
    collector = "collect_kld.py" if args.lane == "vlm" else "collect_llm_kld.py"
    common = [sys.executable, COMPANY / "cli" / collector,
              "--ref-model", args.ref_model, "--dataset", args.dataset,
              "--subset", args.subset, "--dataset-limit", args.rows,
              "--num-eval-tokens", args.num_eval_tokens, "--tf-chunk", args.tf_chunk,
              "--n-ctx", args.n_ctx, "--n-batch", args.n_batch, "--n-ubatch", args.n_ubatch,
              "--n-threads", 8, "--metric-threads", 8, "--flash-attn", "--keep-prep",
              f"--llama-{args.lane}-kld", binary]
    if args.lane == "vlm":
        common += ["--ref-mmproj", args.mmproj, "--cand-mmproj", args.mmproj]
        for flag, value in (("--image-min-tokens", args.image_min_tokens), ("--image-max-tokens", args.image_max_tokens)):
            if value is not None:
                common += [flag, value]
    collections = {}
    for name, model in (("a", args.cand_a_model), ("b", args.cand_b_model), ("a-repeat", args.cand_a_model)):
        if not args.verify_only:
            run([*common, "--cand-model", model, "--out", args.out / name], args.out / f"{name}.log")
        collections[name] = read_collection(args.out / name, args.rows, args.num_eval_tokens)
    a, repeat = collections["a"], collections["a-repeat"]
    assert a.keys() == repeat.keys()
    for name in a:
        assert a[name].keys() == repeat[name].keys()
        for key in a[name]:
            assert np.array_equal(a[name][key], repeat[name][key]), (name, key, "repeat drift")
    for name, other in (("paired", "b"), ("self-paired", "a-repeat")):
        report = args.out / f"{name}.json"
        if not args.verify_only:
            run([sys.executable, COMPANY / "stats/cli/saved_metrics_paired_compare.py",
                 "--candidate-a", args.out / "a", "--candidate-b", args.out / other,
                 "--out", args.out / f"{name}.md", "--output-json", report,
                 "--omit-host-metadata"], args.out / f"{name}.log")
        result = json.loads(report.read_text())
        assert result["n_items"] == args.rows
        assert result["alignment"]["n_ref_drift"] == 0
        assert not any(result["alignment"]["drops"].values())
        if name == "self-paired":
            for metric_name, metric in result["metrics"].items():
                for weighting in ("item_weighted", "token_weighted"):
                    block = metric[weighting]
                    if weighting == "token_weighted":
                        assert block["role"] == "descriptive"
                        assert not ({"ci", "ci_delta", "p_value", "p_value_holm", "decision", "bootstrap_std"} & block.keys())
                    if metric_name == "ppl_ratio":
                        assert block["estimate"] == 1
                        if weighting == "item_weighted":
                            assert block["ci"]["lower"] == block["ci"]["upper"] == 1
                    elif metric_name == "ppl":
                        assert block["delta_ppl_b_minus_a"] == 0
                    elif metric_name == "rms_dp":
                        assert block["delta_rms_b_minus_a"] == 0
                    else:
                        assert block["delta_candidate_minus_baseline"] == 0
                        if weighting == "item_weighted":
                            assert block["ci_delta"]["lower"] == block["ci_delta"]["upper"] == 0
    summary = {"lane": args.lane, "rows": args.rows, "repeat_bit_exact": True,
               "paired_reference_drift": 0, "self_paired_zero_deltas": True, "collections": {}}
    for name, items in collections.items():
        summary["collections"][name] = {
            "positions": sum(int(item["npos"]) for item in items.values()),
            "mean_kld": float(np.mean([np.mean(item["kld"], dtype=np.float64) for item in items.values()])),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
