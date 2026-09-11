#!/usr/bin/env python3
"""Audit embedded images in prepared-dataset Parquet configs, using CPU only."""

import argparse
import datetime
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import traceback
from pathlib import Path, PurePosixPath


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def matches(config, patterns):
    return any(fnmatch.fnmatchcase(config, pattern) for pattern in patterns)


def hub_manifest(args):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(
        args.dataset, revision=args.revision, files_metadata=True
    )
    entries = []
    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        path = PurePosixPath(sibling.rfilename)
        if (
            len(path.parts) != 2
            or not path.name.startswith(args.split + "-")
            or path.suffix != ".parquet"
        ):
            continue
        if not matches(path.parts[0], args.config):
            continue
        local = hf_hub_download(
            args.dataset, sibling.rfilename, repo_type="dataset", revision=info.sha
        )
        entries.append(
            {
                "config": path.parts[0],
                "split": args.split,
                "remote_path": sibling.rfilename,
                "local_cache_path": local,
                "size": sibling.size,
                "official_lfs_sha256": sibling.lfs.sha256 if sibling.lfs else None,
            }
        )
    return {"repo": args.dataset, "revision": info.sha, "selected_files": entries}


def validate_manifest(metadata, patterns, split, require_expected=True):
    import pyarrow.parquet as pq

    if not metadata.get("repo") or not re.fullmatch(
        r"[0-9a-fA-F]{40}", str(metadata.get("revision", ""))
    ):
        raise ValueError(
            "Input manifest needs repo and a pinned 40-hex commit revision"
        )
    entries = [
        dict(e)
        for e in metadata["selected_files"]
        if e["split"] == split and matches(e["config"], patterns)
    ]
    if not entries:
        raise ValueError(
            "No matching Parquet files; expected <config>/<split>-*.parquet"
        )
    names = set()
    for entry in entries:
        key = (entry["config"], entry["split"], entry["remote_path"])
        if key in names:
            raise ValueError("Repeated manifest file: " + str(key))
        names.add(key)
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError(
                "Manifest file needs a nonnegative integer size: " + str(key)
            )
        expected = {}
        for field in ("sha256", "official_lfs_sha256"):
            value = entry.get(field)
            if value is not None:
                if not isinstance(value, str) or not re.fullmatch(
                    r"[0-9a-fA-F]{64}", value
                ):
                    raise ValueError("Invalid expected SHA256: " + str(key))
                expected[field] = value.lower()
        if require_expected and not expected:
            raise ValueError(
                "Offline manifest file needs an expected SHA256: " + str(key)
            )
        path = Path(entry["local_cache_path"]).expanduser().resolve(strict=True)
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError("File changed while hashing: " + str(path))
        if after.st_size != entry["size"]:
            raise ValueError("Size mismatch: " + str(path))
        for value in expected.values():
            if value != digest:
                raise ValueError("SHA256 mismatch: " + str(path))
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        if count <= 0:
            raise ValueError("Empty Parquet file: " + str(path))
        if entry.get("rows") is not None and entry["rows"] != count:
            raise ValueError("Row count mismatch: " + str(path))
        entry.update(
            {
                "local_cache_path": str(path),
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
                "rows": count,
                "schema": str(parquet.schema_arrow),
                "integrity_basis": "verified_expected_sha256"
                if expected
                else "computed_from_pinned_download_without_official_lfs_digest",
            }
        )
        print(
            json.dumps(
                {
                    "event": "verified",
                    "config": entry["config"],
                    "file": entry["remote_path"],
                    "rows": count,
                }
            ),
            flush=True,
        )
    return {
        "repo": metadata["repo"],
        "revision": metadata["revision"],
        "selected_files": sorted(
            entries, key=lambda e: (e["config"], e["remote_path"])
        ),
    }


def markdown_summary(output):
    summary = json.loads((output / "summary.json").read_text())
    lines = [
        "# Dataset image clusters",
        "",
        f"Dataset: `{summary['dataset']}`. Revision: `{summary['revision']}`.",
        "",
        "Exact means identical dimensions and RGB pixels after EXIF correction. Alpha is excluded from this comparison; RGBA hashes are retained for review. pHash matches are candidates only and are never merged into exact clusters.",
        "",
        "| Config | Rows | Exact row clusters with reuse | Rows involved | Different-text image groups | Nonexact pHash groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config, value in summary["configs"].items():
        exact = value["shared_image_connected_components"]
        lines.append(
            f"| {config} | {value['rows']} | {exact['duplicate_groups']} | {exact['rows_in_duplicate_groups']} | {value['exact_image']['distinct_textual_question_clusters']} | {value['phash_nonexact_candidates']['groups']} |"
        )
    lines += [
        "",
        "Multi-image rows are joined when they share any exact image. Complete ordered and unordered image-list groups are also saved. Different text includes changed options and is not a claim of a different semantic question. Empty text does not establish question identity.",
        "",
        "The 100/500 overlap is reported separately. When 100 rows are the prefix of 500, that overlap is expected. Check `ordered_prefix_equal` in `overlap_100_500.json`.",
        "",
        "Shared base images with changed annotations, crops, or rendered question text can evade exact hashing. Review `phash_candidates.jsonl`, source IDs, and original images before defining additional clusters. In particular, MMMU-Pro Vision can share the base images of Standard even when rendered-image hashes differ. This scanner does not infer that source relationship automatically.",
        "",
        "`row_cluster_mapping.csv` has one row per input UID. Keep `metadata.json` and its revision with the mapping when joining metrics. Pixel cluster counts are structural counts, not measured effective sample sizes. The ICC scenario CSV contains hypothetical values under homogeneous variance and common within-cluster correlation, not measured statistical power. For model comparisons, cluster the paired per-question differences; see [Miller (2024), sections 2.2 and 4.2](https://arxiv.org/pdf/2411.00640).",
        "",
        "This run does not modify the dataset. See `status.json`, `analysis_provenance.json`, `files.json`, and the copied scripts for provenance.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="user-company/vlm-prepared-dataset")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision; resolved to a commit before downloading",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Config glob; repeat to select several (default: *-subsample-100 and *-subsample-500)",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out", type=Path, required=True, help="New or empty output directory"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent config decoders; use 1 to reduce CPU and I/O pressure",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Read local files from a prior metadata.json; no Hub access. The manifest supplies dataset and revision.",
    )
    args = parser.parse_args(argv)
    args.config = args.config or ["*-subsample-100", "*-subsample-500"]
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.out.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--out must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ],
        "output": str(output),
        "stage": "acquire",
    }
    save_json(output / "status.json", status)
    try:
        scripts = output / "scripts"
        scripts.mkdir()
        for name in ("scan.py", "analysis.py", "requirements.txt", "README.md"):
            shutil.copyfile(Path(__file__).with_name(name), scripts / name)
        if args.input_manifest:
            metadata = json.loads(args.input_manifest.expanduser().read_text())
        else:
            metadata = hub_manifest(args)
        metadata = validate_manifest(
            metadata,
            args.config,
            args.split,
            require_expected=bool(args.input_manifest),
        )
        save_json(output / "metadata.json", metadata)
        status.update(
            {
                "stage": "decode_and_group",
                "dataset": metadata["repo"],
                "revision": metadata["revision"],
            }
        )
        save_json(output / "status.json", status)
        import analysis

        analysis.run(output, workers=args.workers)
        for entry in metadata["selected_files"]:
            current = Path(entry["local_cache_path"]).stat()
            if (current.st_size, current.st_mtime_ns) != (
                entry["size"],
                entry["mtime_ns"],
            ):
                raise ValueError(
                    "Input changed during analysis: " + entry["local_cache_path"]
                )
        markdown_summary(output)
        files = [
            {
                "path": str(p.relative_to(output)),
                "sha256": file_sha256(p),
                "size": p.stat().st_size,
            }
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "status.json"
        ]
        save_json(output / "files.json", files)
        status.update(
            {
                "status": "complete",
                "stage": "complete",
                "rows": sum(e["rows"] for e in metadata["selected_files"]),
            }
        )
        result = 0
    except KeyboardInterrupt:
        status.update({"status": "interrupted"})
        result = 130
    except Exception as error:
        status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (output / "error.log").write_text(traceback.format_exc())
        print(status["error"], file=sys.stderr)
        result = 1
    status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(output / "status.json", status)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
