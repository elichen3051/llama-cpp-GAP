#!/usr/bin/env python3
"""Integration test for llama-vlm-kld --manifest (persistent-model) mode.

Validates the three correctness properties that the hermetic unit tests in
test_collect_kld.py cannot cover (they need a real GPU + models):

  1. BIT-IDENTITY     — for each row, the VLMK metrics dump produced in
                        default --manifest mode (n_seq_max=1) must be
                        byte-for-byte identical to the dump produced by a
                        standalone single-mode invocation. Manifest mode is a
                        model-reuse optimization; at n_seq_max=1 it must not
                        change results.

  2. KV ISOLATION     — running rows together in one process must not let one
     (order independence)  sample's state bleed into the next. We verify this by
                        scoring the same rows in two different manifest orders
                        and confirming every row matches its single-mode
                        reference in BOTH orders. If llama_memory_clear() did not
                        fully reset state, a row preceded by a different row
                        would differ.

  3. PARTIAL FAILURE  — a manifest with one bad entry (missing input file) must
                        still produce the good rows' outputs and exit non-zero.

Also reports a wall-clock comparison (N single-mode invocations vs. one
manifest-mode invocation) to confirm the model-reload saving. The scorer supports one sequence; this check preserves that runtime.

Usage:
  python3 verify_and_validation_scripts/check_manifest_mode.py \
      [--kld build/bin/llama-vlm-kld] \
      --ref-model  .../Qwen3VL-4B-Instruct-F16.gguf \
      --ref-mmproj .../mmproj-Qwen3VL-4B-Instruct-F16.gguf \
      --cand-model .../Qwen3VL-4B-Instruct-Q4_K_M.gguf \
      --cand-mmproj .../mmproj-Qwen3VL-4B-Instruct-F16.gguf \
      [--prep-root /tmp/bench_prep] \
      [--rows row0 row100 row300] \
      [--num-eval-tokens 64] \
      [--work /tmp/manifest_test]

Requires the per-row prep dirs already to exist under <prep-root>/<row>/ —
each must contain the images declared by meta.json, formatted_chat.txt and tokens.bin (as
produced by cli/prep_vlm_score_from_hf.py with --out <prep-root>/<row>).

Exit code 0 iff all checks pass.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli.collect_kld import write_kld_manifest  # noqa: E402


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_args(args):
    return ["--ref-model", args.ref_model, "--ref-mmproj", args.ref_mmproj,
            "--cand-model", args.cand_model, "--cand-mmproj", args.cand_mmproj]


def run_single(args, sample, out_path, *, n_eval):
    cmd = [
        args.kld, *_pair_args(args),
        "--image", sample["images"][0],
        "--formatted-chat", sample["formatted_chat"],
        "--tokens-in", sample["tokens_in"],
        "--n-prefill", str(sample["n_prefill"]),
        "--output-metrics", str(out_path),
        "--num-eval-tokens", str(n_eval),
    ]
    for img in sample["images"][1:]:
        cmd.extend(["--image", img])
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return time.time() - t0, r.returncode, r.stderr.decode(errors="replace")


def run_manifest(args, manifest_path, *, n_eval):
    cmd = [
        args.kld, *_pair_args(args),
        "--manifest", str(manifest_path),
        "--num-eval-tokens", str(n_eval),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return time.time() - t0, r.returncode, r.stderr.decode(errors="replace")


def write_manifest(path: Path, samples):
    """The manifest the scorer reads, written by the SAME writer the
    collector uses (collect_kld.write_kld_manifest): this script exists
    to prove manifest mode is byte-identical to single-row mode, so it must
    not hand the scorer a manifest encoded differently from production."""
    write_kld_manifest(path, [
        {k: s[k] for k in ("images", "formatted_chat", "tokens_in",
                           "n_prefill", "output_metrics")}
        for s in samples
    ])


def main():
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser()
    p.add_argument("--kld", default=str(repo_root / "build/bin/llama-vlm-kld"))
    p.add_argument("--ref-model", required=True, help="reference GGUF LLM path")
    p.add_argument("--ref-mmproj", required=True, help="reference GGUF mmproj path")
    p.add_argument("--cand-model", required=True, help="candidate GGUF LLM path")
    p.add_argument("--cand-mmproj", required=True, help="candidate GGUF mmproj path")
    p.add_argument("--prep-root", default="/tmp/bench_prep")
    p.add_argument("--rows", nargs="+", default=["row0", "row100", "row300"])
    p.add_argument("--num-eval-tokens", type=int, default=64)
    p.add_argument("--work", default="/tmp/manifest_test")
    args = p.parse_args()

    work = Path(args.work)
    (work / "single").mkdir(parents=True, exist_ok=True)
    (work / "fwd").mkdir(parents=True, exist_ok=True)
    (work / "rev").mkdir(parents=True, exist_ok=True)

    # Build sample descriptors from prep dirs.
    samples = []
    for name in args.rows:
        d = Path(args.prep_root) / name
        meta = json.loads((d / "meta.json").read_text())
        imgs = [str(d / name) for name in meta["image_files"]]
        if not imgs or not all(Path(path).is_file() for path in imgs):
            raise ValueError(f"missing prepared images for {name}")
        samples.append({
            "name": name,
            "images": imgs,
            "formatted_chat": str(d / "formatted_chat.txt"),
            "tokens_in": str(d / "tokens.bin"),
            "n_prefill": meta["n_prefill"],
        })

    failures = []

    # ---- single-mode references ----
    print("== single-mode references ==")
    single_md5 = {}
    single_wall = 0.0
    for s in samples:
        out = work / "single" / f"{s['name']}.bin"
        dt, rc, err = run_single(args, s, out, n_eval=args.num_eval_tokens)
        if rc != 0:
            failures.append(f"single-mode {s['name']} exited {rc}: {err[-200:]}")
            continue
        single_md5[s["name"]] = md5(out)
        single_wall += dt
        print(f"  {s['name']:8s} md5={single_md5[s['name']]} wall={dt:.2f}s")

    # ---- manifest forward order ----
    print("== manifest mode (forward order) ==")
    fwd_samples = [dict(s, output_metrics=str(work / "fwd" / f"{s['name']}.bin")) for s in samples]
    man_fwd = work / "manifest_fwd.jsonl"
    write_manifest(man_fwd, fwd_samples)
    fwd_wall, rc, err = run_manifest(args, man_fwd, n_eval=args.num_eval_tokens)
    if rc != 0:
        failures.append(f"manifest fwd exited {rc}: {err[-300:]}")
    for s in fwd_samples:
        out = Path(s["output_metrics"])
        if not out.exists():
            failures.append(f"manifest fwd: missing output for {s['name']}")
            continue
        m = md5(out)
        ok = single_md5.get(s["name"]) == m
        print(f"  {s['name']:8s} md5={m} {'OK' if ok else 'MISMATCH vs single'}")
        if not ok:
            failures.append(f"BIT-IDENTITY fail {s['name']}: single={single_md5.get(s['name'])} fwd={m}")

    # ---- manifest reverse order (KV isolation / order independence) ----
    print("== manifest mode (reverse order) — KV isolation ==")
    rev_samples = [dict(s, output_metrics=str(work / "rev" / f"{s['name']}.bin")) for s in reversed(samples)]
    man_rev = work / "manifest_rev.jsonl"
    write_manifest(man_rev, rev_samples)
    _, rc, err = run_manifest(args, man_rev, n_eval=args.num_eval_tokens)
    if rc != 0:
        failures.append(f"manifest rev exited {rc}: {err[-300:]}")
    for s in rev_samples:
        out = Path(s["output_metrics"])
        if not out.exists():
            failures.append(f"manifest rev: missing output for {s['name']}")
            continue
        m = md5(out)
        ok = single_md5.get(s["name"]) == m
        print(f"  {s['name']:8s} md5={m} {'OK' if ok else 'MISMATCH (KV leak?)'}")
        if not ok:
            failures.append(f"KV-ISOLATION fail {s['name']}: single={single_md5.get(s['name'])} rev={m}")

    # ---- partial failure ----
    print("== partial failure (one bad entry) ==")
    bad = dict(samples[0], output_metrics=str(work / "good_after_bad.bin"))
    broken = {
        "images": ["/tmp/does_not_exist_img.png"],
        "formatted_chat": "/tmp/does_not_exist.txt",
        "tokens_in": "/tmp/does_not_exist.bin",
        "n_prefill": 10,
        "output_metrics": str(work / "should_not_exist.bin"),
    }
    man_partial = work / "manifest_partial.jsonl"
    with open(man_partial, "w") as f:
        f.write(json.dumps(broken) + "\n")
        f.write(json.dumps({k: bad[k] for k in
                ("images", "formatted_chat", "tokens_in", "n_prefill", "output_metrics")}) + "\n")
    _, rc, err = run_manifest(args, man_partial, n_eval=args.num_eval_tokens)
    good_out = Path(bad["output_metrics"])
    if rc == 0:
        failures.append("partial failure: expected non-zero exit, got 0")
    else:
        print(f"  exit={rc} (non-zero as expected)")
    if not good_out.exists():
        failures.append("partial failure: good row after bad entry was not produced")
    elif md5(good_out) != single_md5.get(samples[0]["name"]):
        failures.append("partial failure: good row output differs from single-mode reference")
    else:
        print("  good row after bad entry produced + matches single-mode ✓")

    # ---- wall-clock summary ----
    print("== wall-clock ==")
    print(f"  sum of {len(samples)} single-mode runs : {single_wall:.2f}s")
    print(f"  one manifest-mode run (fwd)         : {fwd_wall:.2f}s")
    if fwd_wall > 0:
        print(f"  saved                               : {single_wall - fwd_wall:.2f}s "
              f"({single_wall / fwd_wall:.2f}x)")

    print()
    if failures:
        print("FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
