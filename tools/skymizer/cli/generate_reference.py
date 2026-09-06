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
import itertools
import json
import subprocess
import sys
from pathlib import Path

SKYMIZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYMIZER))
from lib.reference_dataset import build_row, canonical_json, model_files, raw_images, reference_features, sha256_file
from lib.reference_contract import FS_SAFE_ID_RE


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--subset")
    p.add_argument("--split", default="train")
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
    p.add_argument("llama_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
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
    return args


def load_source(args):
    from datasets import DatasetDict, Image, Sequence, load_dataset, load_from_disk
    path = Path(args.dataset)
    if (path / "state.json").is_file() or (path / "dataset_dict.json").is_file():
        if args.subset:
            raise ValueError("--subset does not apply to a local save_to_disk dataset")
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            ds = ds[args.split]
    else:
        ds = load_dataset(args.dataset, args.subset, split=args.split)
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
    from datasets.arrow_writer import ArrowWriter
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
    requests_path = out / "requests.jsonl"
    ids = set()
    with requests_path.open("w") as stream:
        for index, source in enumerate(ds):
            row_id = str(source[id_column]) if id_column else f"row-{index:08d}"
            if FS_SAFE_ID_RE.fullmatch(row_id) is None or row_id in (".", "..") or row_id in ids:
                raise ValueError(f"row id must be unique and filesystem-safe: {row_id!r}")
            ids.add(row_id)
            question = source[args.question_column]
            if not isinstance(question, str) or not question:
                raise ValueError(f"row {row_id}: question must be a nonempty string")
            pictures = raw_images(source, args.images_column)
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
                    raise ValueError(f"row {row_id}: thinking column must be boolean")
                request["enable_thinking"] = thinking
            if args.system_prompt is not None:
                request["system_prompt"] = args.system_prompt
            stream.write(canonical_json(request) + "\n")
    binary_hash = sha256_file(binary)
    command = [str(binary), "--requests", str(requests_path), "--out-dir", str(out / "native"), *args.llama_args]
    (out / "command.json").write_text(canonical_json(command) + "\n")
    print(f"Generating {len(ds)} rows; log: {out / 'native.log'}", flush=True)
    with (out / "native.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.native_timeout)
    metadata = json.loads((out / "native/metadata.json").read_text())
    metadata["binary_sha256"] = binary_hash
    metadata["model_files"] = []
    for path in model_files(metadata["model_path"]):
        print(f"Hashing {path}", flush=True)
        metadata["model_files"].append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    if metadata["mmproj_path"]:
        metadata["mmproj_sha256"] = sha256_file(metadata["mmproj_path"])
    metadata["dataset_source"] = {"path": args.dataset, "subset": args.subset, "split": args.split,
                                  "fingerprint": ds._fingerprint, "num_rows": len(ds)}
    metadata["requested_enable_thinking"] = args.enable_thinking
    metadata["thinking_column"] = args.thinking_column
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    arrow = out / "reference.arrow"
    writer = ArrowWriter(path=str(arrow), features=reference_features())
    try:
        with requests_path.open() as requests, (out / "native/generations.jsonl").open() as results:
            count = 0
            for source, request_line, result_line in itertools.zip_longest(ds, requests, results):
                if source is None or request_line is None or result_line is None:
                    raise ValueError("native result count differs from request count")
                request, result = json.loads(request_line), json.loads(result_line)
                pictures = [Path(path).read_bytes() for path in request["images"]]
                source_meta = {k: v for k, v in source.items() if k != args.images_column}
                row = build_row(source_meta, request, result, metadata, pictures)
                writer.write(row)
                count += 1
        writer.finalize()
    finally:
        writer.close()
    Dataset.from_file(str(arrow)).save_to_disk(str(out / "dataset"))
    (out / "complete.json").write_text(canonical_json({"rows": count, "schema_version": metadata["schema_version"]}) + "\n")
    print(f"Saved {count} validated references: {out / 'dataset'}", flush=True)


def main():
    args = parse_args()
    try:
        generate(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
