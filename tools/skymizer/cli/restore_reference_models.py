#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, record):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == record["size"]
            and sha256(path) == record["sha256"])


def main():
    parser = argparse.ArgumentParser(description="Restore pinned BF16 model shards and projectors; never replace existing files")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    parser.add_argument("--models", nargs="+", help="default: all six production models in the profiles")
    parser.add_argument("--download", action="store_true", help="download after validating the full plan; otherwise print the plan only")
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text())
    models = args.models or list(profiles["models"])
    if len(set(models)) != len(models) or set(models) - set(profiles["models"]):
        raise ValueError("unknown or repeated model selection")
    root = args.models_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets, seen = [], set()
    for model in models:
        profile = profiles["models"][model]
        identity = profile["identity"]
        prefix = identity["manifest"].rsplit("/", 1)[0]
        if prefix != "s3://research-kld-benchmark/reference_model/" + model:
            raise ValueError("unexpected pinned S3 prefix: " + prefix)
        names = {record["name"] for record in identity["files"]}
        if Path(profile["model"]).name not in names or Path(profile["mmproj"]).name not in names:
            raise ValueError("pinned files do not include the profile model and projector")
        for record in identity["files"]:
            name = record["name"]
            if (Path(name).name != name or name in (".", "..")
                    or record["role"] not in ("llm", "mmproj")
                    or type(record["size"]) is not int or record["size"] <= 0
                    or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
                raise ValueError("invalid pinned model record")
            anchor = Path(profile["model"] if record["role"] == "llm" else profile["mmproj"])
            if anchor.is_absolute() or ".." in anchor.parts:
                raise ValueError("model path must stay under --models-dir")
            if record["role"] == "mmproj" and anchor.name != name:
                raise ValueError("projector name differs from the profile")
            destination = root / anchor.parent / name
            if destination.is_symlink() or not destination.resolve().is_relative_to(root) or destination in seen:
                raise ValueError("unsafe or duplicate destination: " + str(destination))
            seen.add(destination)
            present = destination.exists()
            if present and not matches(destination, record):
                raise ValueError("existing file differs; retained without overwrite: " + str(destination))
            targets.append((model, prefix + "/" + name, destination, record, present))
    pending_bytes = sum(record["size"] for _, _, _, record, present in targets if not present)
    print(json.dumps({"models": models, "pending_bytes": pending_bytes, "download": args.download}), flush=True)
    if args.download and shutil.disk_usage(root).free < pending_bytes:
        raise ValueError("insufficient free space under --models-dir")
    for model, source, destination, record, present in targets:
        action = "verified_existing" if present else "planned"
        if args.download and not present:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".reference-download-", dir=destination.parent) as temporary:
                partial = Path(temporary) / "download.gguf"
                subprocess.run(["aws", "s3", "cp", source, str(partial), "--only-show-errors"], check=True)
                if not matches(partial, record):
                    raise ValueError("download failed full-file size/SHA256 verification: " + source)
                try:
                    # A hard link publishes the verified bytes without replacing a racing writer.
                    os.link(partial, destination)
                except FileExistsError:
                    if not matches(destination, record):
                        raise ValueError("destination appeared with different bytes; retained: " + str(destination))
                action = "verified_restored"
        print(json.dumps({"model": model, "source": source, "destination": str(destination),
                          "size": record["size"], "sha256": record["sha256"], "status": action}), flush=True)


if __name__ == "__main__":
    main()
