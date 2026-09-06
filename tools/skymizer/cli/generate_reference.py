#!/usr/bin/env python3
"""Generate text/image reference datasets directly with llama.cpp, without HTTP.

Example (all llama.cpp flags follow --):
  python cli/generate_reference.py --dataset /path/to/prepared --out /tmp/ref \
    --num-samples 4 --no-enable-thinking -- -m model.gguf --mmproj mmproj.gguf \
    -ngl 99 -c 8192 -n 64 --seed 1234 --temp 1.0 --top-p 0.95 --top-k 64

The output contains dataset/ (save_to_disk), metadata.json (effective settings
and file hashes), and native/ (incremental C++ results). No tokenizer or
processor is loaded from Hugging Face; datasets is used only for storage.
"""

import argparse
import hashlib
import shutil
import signal
import secrets
import json
import subprocess
import sys
from pathlib import Path

SKYMIZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYMIZER))
from lib.reference_dataset import build_row, canonical_json, model_files, raw_images, reference_features, sha256_file
from lib.reference_contract import FS_SAFE_ID_RE
from lib.collect_meta_provenance import execution_identity
from lib.reference_run import atomic_json, run_native


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--subset")
    p.add_argument("--split", default="train")
    p.add_argument("--revision", help="Hub dataset revision (prefer a commit hash)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-samples", type=int)
    p.add_argument("--question-column", default="question")
    p.add_argument("--images-column", default="images")
    p.add_argument("--id-column")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--system-prompt")
    p.add_argument("--thinking-column", help="per-row boolean thinking mode; takes precedence over --enable-thinking")
    p.add_argument("--llama-reference", type=Path, default=SKYMIZER.parents[1] / "build/bin/llama-reference")
    p.add_argument("--native-timeout", type=float, default=None, help="optional total subprocess timeout in seconds")
    p.add_argument("--row-retries", type=int, default=1, help="retries after a native row crash; data errors are terminal")
    p.add_argument("--startup-retries", type=int, default=1)
    p.add_argument("--row-timeout", type=float, default=1800, help="restart when any active row exceeds this many seconds, or startup makes no journal progress")
    p.add_argument("llama_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.row_timeout <= 0:
        p.error("--row-timeout must be positive")
    if args.row_retries < 0 or args.startup_retries < 0:
        p.error("retry counts must be nonnegative")
    if args.num_samples is not None and args.num_samples <= 0:
        p.error("--num-samples must be positive")
    if args.native_timeout is not None and args.native_timeout <= 0:
        p.error("--native-timeout must be positive")
    if args.llama_args[:1] == ["--"]:
        args.llama_args = args.llama_args[1:]
    if not args.llama_args:
        p.error("supply native model/sampling options after --")
    if any(flag.split("=", 1)[0] in {"--out-dir", "--requests", "--describe"} for flag in args.llama_args):
        p.error("--out-dir, --requests and --describe are managed by the dataset driver")
    for index, flag in enumerate(args.llama_args):
        if flag in ("--seed=-1", "--seed=4294967295"):
            args.llama_args[index] = "--seed=" + str(secrets.randbelow(4294967295))
    seed_index = None
    for index, flag in enumerate(args.llama_args):
        if flag in ("--seed", "-s"):
            if index + 1 >= len(args.llama_args):
                p.error("missing seed value")
            seed_index = index + 1
    if seed_index is None and not any(flag.startswith("--seed=") for flag in args.llama_args):
        args.llama_args += ["--seed", str(secrets.randbelow(4294967295))]
    elif seed_index is not None and args.llama_args[seed_index] in ("-1", "4294967295"):
        args.llama_args[seed_index] = str(secrets.randbelow(4294967295))
    return args


def load_source(args):
    from datasets import DatasetDict, Image, Sequence, load_dataset, load_from_disk
    path = Path(args.dataset)
    if (path / "state.json").is_file() or (path / "dataset_dict.json").is_file():
        if args.subset or args.revision:
            raise ValueError("--subset and --revision do not apply to a local save_to_disk dataset")
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            ds = ds[args.split]
    else:
        ds = load_dataset(args.dataset, args.subset, split=args.split, revision=args.revision)
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    if not len(ds):
        raise ValueError("source dataset is empty")
    if args.thinking_column and args.thinking_column not in ds.column_names:
        raise ValueError(f"missing thinking column {args.thinking_column!r}")
    if args.question_column not in ds.column_names:
        raise ValueError(f"missing question column {args.question_column!r}")
    if args.images_column in ds.column_names:
        ds = ds.cast_column(args.images_column, Sequence(Image(decode=False)))
    return ds


def generate(args):
    from datasets import Dataset
    binary = args.llama_reference.resolve()
    if not binary.is_file():
        raise ValueError(f"build llama-reference first: {binary}")
    if args.out.exists():
        raise ValueError(f"output already exists: {args.out}")
    ds = load_source(args)
    id_column = args.id_column or next((c for c in ("item_id", "id") if c in ds.column_names), None)
    if args.id_column and args.id_column not in ds.column_names:
        raise ValueError(f"missing id column {args.id_column!r}")
    out = args.out.resolve()
    out.mkdir(parents=True)
    source_info = {"path": args.dataset, "subset": args.subset, "split": args.split,
                   "revision": args.revision, "fingerprint": ds._fingerprint, "num_rows": len(ds)}
    state = {"status": "preparing", "dataset_source": source_info, "requested": len(ds)}
    atomic_json(out / "run_state.json", state)
    try:
        requests, source_ids, failures = [], [], {}
        with (out / "requests.jsonl").open("w") as stream:
            for index, source in enumerate(ds):
                row_id = str(source[id_column]) if id_column else f"row-{index:08d}"
                if FS_SAFE_ID_RE.fullmatch(row_id) is None or row_id in (".", "..") or row_id in source_ids:
                    raise ValueError(f"row id must be unique and filesystem-safe: {row_id!r}")
                source_ids.append(row_id)
                try:
                    question = source[args.question_column]
                    if not isinstance(question, str):
                        raise ValueError("question must be a string")
                    pictures = raw_images(source, args.images_column)
                    if not question and not pictures:
                        raise ValueError("provide question text or at least one image")
                    paths = []
                    for number, raw in enumerate(pictures):
                        image_path = out / "inputs" / str(index) / f"image-{number}.bin"
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(raw)
                        paths.append(str(image_path))
                    request = {"id": row_id, "question": question, "images": paths}
                    if args.enable_thinking is not None:
                        request["enable_thinking"] = args.enable_thinking
                    if args.thinking_column:
                        thinking = source[args.thinking_column]
                        if not isinstance(thinking, bool):
                            raise ValueError("thinking column must be boolean")
                        request["enable_thinking"] = thinking
                    if args.system_prompt is not None:
                        request["system_prompt"] = args.system_prompt
                    requests.append(request)
                    stream.write(canonical_json(request) + "\n")
                except (ValueError, OSError, TypeError) as error:
                    failures[row_id] = {"id": row_id, "status": "preparation_failed", "error": str(error), "attempts": 0}
        snapshot = out / "scripts"
        snapshot.mkdir()
        for relative in ("cli/generate_reference.py", "cli/generate_model_reference.py", "lib/reference_run.py",
                         "lib/reference_dataset.py", "lib/reference_contract.py", "lib/collect_meta_provenance.py",
                         "scripts/reference_model_profiles.json", "reference.cpp", "skymizer-repetition.h"):
            path = SKYMIZER / relative
            if path.is_file():
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        metadata, results, execution = {"dataset_source": source_info}, {}, None
        state.update(status="running", requested_ids=source_ids, native_args=args.llama_args,
                     row_retries=args.row_retries, startup_retries=args.startup_retries, row_timeout=args.row_timeout)
        if requests:
            execution, _ = execution_identity(binary)
            state["execution_identity"] = execution
            atomic_json(out / "run_state.json", state)
            atomic_json(out / "run_start.json", state)
            results, native_failures, native_metadata = run_native(binary, args.llama_args, requests, out,
                timeout=args.native_timeout, row_retries=args.row_retries, startup_retries=args.startup_retries, row_timeout=args.row_timeout)
            failures.update(native_failures)
            if execution_identity(binary)[0] != execution:
                raise ValueError("reference executable/backend environment changed during generation")
            if native_metadata is not None:
                metadata = native_metadata
            if results:
                metadata["binary_sha256"] = execution["binary_sha256"]
                metadata["execution_identity"] = execution
                metadata["model_files"] = []
                for path in model_files(metadata["model_path"]):
                    print(f"Hashing {path}", flush=True)
                    metadata["model_files"].append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
                if metadata["decoding"]["method"] == "mtp":
                    mtp = metadata["decoding"]["mtp"]
                    if mtp["head_source"] == "sidecar":
                        mtp["head_files"] = [{"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
                                             for path in model_files(mtp["head_path"])]
                if metadata["mmproj_path"]:
                    metadata["mmproj_sha256"] = sha256_file(metadata["mmproj_path"])
        else:
            atomic_json(out / "run_start.json", state)
        metadata["dataset_source"] = source_info
        metadata["requested_enable_thinking"] = args.enable_thinking
        metadata["thinking_column"] = args.thinking_column
        metadata["driver_recovery"] = {"row_retries": args.row_retries, "startup_retries": args.startup_retries,
                                       "row_timeout": args.row_timeout, "native_timeout": args.native_timeout}
        requests_by_id = {r["id"]: r for r in requests}
        eligible, excluded = [], []
        for row_id, source in zip(source_ids, ds):
            if row_id not in results:
                continue
            result = results[row_id]
            request = requests_by_id[row_id]
            try:
                pictures = [Path(path).read_bytes() for path in request["images"]]
                source_meta = {k: v for k, v in source.items() if k != args.images_column}
                row = build_row(source_meta, request, result, metadata, pictures)
            except (ValueError, KeyError, TypeError, OSError) as error:
                failures[row_id] = {"id": row_id, "status": "validation_failed", "error": str(error)}
                continue
            if result.get("repetition") is not None:
                excluded.append({"id": row_id, "reason": "repetition", "evidence": result["repetition"],
                                 "generated_tokens": row["generated_tokens_len"], "generated_text": result["content"]})
            else:
                eligible.append(row)
        eligible_ids = [r["id"] for r in eligible]
        excluded_ids = [r["id"] for r in excluded]
        failed_ids = [i for i in source_ids if i in failures]
        if set(eligible_ids + excluded_ids + failed_ids) != set(source_ids) or len(eligible_ids + excluded_ids + failed_ids) != len(source_ids):
            raise ValueError("eligible/excluded/failed IDs do not partition the source")
        cohort = {"requested": len(ds), "native_generated": len(results), "generated": len(eligible) + len(excluded), "eligible": len(eligible), "excluded": len(excluded),
                  "failed": len(failed_ids), "requested_ids": source_ids, "eligible_ids": eligible_ids,
                  "excluded_ids": excluded_ids, "failed_ids": failed_ids}
        for key in ("requested_ids", "eligible_ids", "excluded_ids", "failed_ids"):
            cohort[key + "_sha256"] = hashlib.sha256(canonical_json(cohort[key]).encode()).hexdigest()
        metadata["cohort"] = cohort
        atomic_json(out / "metadata.json", metadata)
        for row in eligible:
            row["generation_metadata"] = canonical_json(metadata)
        if eligible:
            Dataset.from_list(eligible, features=reference_features()).save_to_disk(str(out / "dataset"))
        (out / "excluded.jsonl").write_text("".join(canonical_json(r) + "\n" for r in excluded))
        (out / "failures.jsonl").write_text("".join(canonical_json(failures[i]) + "\n" for i in failed_ids))
        state.update(status="complete_with_failures" if failed_ids else "complete", cohort=cohort)
        atomic_json(out / "run_state.json", state)
        atomic_json(out / "complete.json", {"rows": len(eligible), "status": state["status"], "cohort": cohort,
                    "schema_version": metadata.get("schema_version", "skymizer-reference-v2")})
        print(f"Processed {len(ds)} rows: {len(eligible)} eligible, {len(excluded)} repetition exclusions, {len(failed_ids)} failures", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", error=str(error))
        atomic_json(out / "run_state.json", state)
        raise


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    args = parse_args()
    try:
        generate(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
