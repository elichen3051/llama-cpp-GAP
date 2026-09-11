"""Resolve and hash local GGUF model files."""

import hashlib
import re
from pathlib import Path


def model_files(path):
    # Keep snapshot symlink names: llama.cpp expands shards beside the supplied entrypoint.
    path = Path(path).absolute()
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    if match:
        index, total = int(match[2]), int(match[3])
        if index != 1 or total < 1:
            raise ValueError(f"GGUF model must start at shard 00001 with a positive shard count: {path}")
        files = [path.with_name(f"{match[1]}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    else:
        files = [path]
    for file in files:
        if not file.is_file():
            raise ValueError(f"missing GGUF model file: {file}")
    return files


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
