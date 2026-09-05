#!/usr/bin/env python3
"""Import-safe helpers for additive collect_meta.json provenance."""

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
UNKNOWN = "unknown"


def _first_stdout_line(cmd) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except Exception:
        return UNKNOWN
    lines = result.stdout.splitlines()
    if not lines:
        return UNKNOWN
    value = lines[0].strip()
    return value or UNKNOWN


def _llama_cpp_build_commit() -> str:
    return _first_stdout_line(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def _gpu_name() -> str:
    return _first_stdout_line(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )


def build_collect_provenance() -> dict:
    """Return run-level reproducibility metadata for collect_meta.json.

    GPU name uses nvidia-smi instead of parsing scorer stderr: collect_meta.json
    is written before the scorer starts, and the collectors stream scorer stderr
    directly to log files while retaining only per-row DONE timing markers. The
    nvidia-smi query keeps metadata write-once and falls back to "unknown".
    """
    return {
        "llama_cpp_build_commit": _llama_cpp_build_commit(),
        "gpu_name": _gpu_name(),
    }
