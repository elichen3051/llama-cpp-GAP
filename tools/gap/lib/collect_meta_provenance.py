"""Record the executed scorer and its loaded-code dependencies."""

import json
import os
import re
import subprocess
from pathlib import Path

from lib.reference_dataset import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
UNKNOWN = "unknown"
EXECUTION_SCHEME = "company-execution-sha256-v2"


def _stdout_lines(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()] or [UNKNOWN]
    except (OSError, subprocess.SubprocessError):
        return [UNKNOWN]


def _first_stdout_line(cmd):
    return _stdout_lines(cmd)[0]


def _execution_environment():
    prefixes = ("GGML_", "LLAMA_ATTN_", "LLAMA_GRAPH_", "LLAMA_KV_CACHE_", "LLAMA_DSV4_", "LLAMA_BATCH_", "CUDA_", "CUBLAS_", "CUDNN_", "NCCL_", "OMP_", "KMP_",
                "MKL_", "OPENBLAS_", "BLIS_", "VECLIB_", "ROCBLAS_", "HIP_", "HSA_", "SYCL_", "ONEAPI_")
    names = {"LLAMA_TRACE", "NVIDIA_TF32_OVERRIDE", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
             "GOMP_CPU_AFFINITY", "LD_PRELOAD", "LD_LIBRARY_PATH", "GLIBC_TUNABLES"}
    return {key: value for key, value in sorted(os.environ.items())
            if key.startswith(prefixes) or key in names}


def _llama_cpp_build_commit():
    return _first_stdout_line(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def _gpu_name():
    return _first_stdout_line(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])


def execution_identity(scorer):
    binary = Path(scorer).resolve()
    if not binary.is_file():
        raise ValueError(f"scorer binary does not exist: {binary}")
    result = subprocess.run([str(binary), "--build-info"], check=True, capture_output=True,
                            text=True, timeout=30)
    info = json.loads(result.stdout)
    if "reference-vocabulary-v1" not in info.get("contracts", []):
        raise ValueError("scorer lacks the reference vocabulary contract; rebuild it")
    libraries = set()
    for name in info.get("loaded_libraries", []):
        path = Path(name)
        if path.is_file():
            libraries.add(path.resolve())
        elif name.startswith("/"):
            raise ValueError(f"loaded scorer dependency is missing: {name}")
    if not libraries:
        raise ValueError("scorer did not report its loaded libraries; cannot establish build identity")
    records = sorted(({"name": p.name, "sha256": sha256_file(p)} for p in libraries),
                     key=lambda r: (r["name"], r["sha256"]))
    identity = {
        "scheme": EXECUTION_SCHEME,
        "binary_sha256": sha256_file(binary),
        "libraries": records,
        "gpu": _stdout_lines(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]),
        "environment": _execution_environment(),
    }
    return identity, info


def build_collect_provenance(scorer=None):
    provenance = {"source_checkout_commit": _llama_cpp_build_commit(), "gpu_name": _gpu_name(),
                  "llama_cpp_build_commit": UNKNOWN, "execution_identity": None}
    if scorer is not None:
        identity, info = execution_identity(scorer)
        provenance.update(execution_identity=identity, llama_cpp_build_commit=info["build"],
                          scorer_binary=str(Path(scorer).resolve()))
    return provenance


def require_execution_alignment(a_meta, b_meta):
    identities = []
    for role, meta in (("candidate-a", a_meta), ("candidate-b", b_meta)):
        identity = (meta or {}).get("execution_identity")
        if (not isinstance(identity, dict) or identity.get("scheme") != EXECUTION_SCHEME
                or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("binary_sha256", "")))
                or not isinstance(identity.get("libraries"), list)):
            raise ValueError(f"{role}: executed scorer identity missing; re-collect legacy data")
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError("executed scorer/build/backend identity differs; re-collect both candidates with the same execution environment")


def require_cross_part_execution_alignment(pilot_meta, tail_meta, policy="strict"):
    """Allow an explicit physical GPU UUID difference only between completed parts."""
    if policy == "strict":
        return require_execution_alignment(pilot_meta, tail_meta)
    if policy != "same-gpu-model-v1":
        raise ValueError(f"unknown cross-part execution policy: {policy}")
    normalized = []
    for meta in (pilot_meta, tail_meta):
        require_execution_alignment(meta, meta)
        identity = meta["execution_identity"]
        libraries, environment = identity["libraries"], identity.get("environment")
        if (not libraries or any(not isinstance(record, dict) or not record.get("name")
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) for record in libraries)
                or not isinstance(environment, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items())):
            raise ValueError("same-gpu-model-v1 requires complete library hashes and execution environment")
        gpu = identity.get("gpu")
        if not isinstance(gpu, list) or len(gpu) != 1 or not isinstance(gpu[0], str):
            raise ValueError("same-gpu-model-v1 requires exactly one recorded GPU")
        fields = gpu[0].split(", ")
        if (len(fields) != 3 or not fields[0] or fields[0].lower() in ("unknown", "n/a")
                or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", fields[1])
                or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", fields[2])):
            raise ValueError("same-gpu-model-v1 requires a recorded GPU model, UUID and driver version")
        normalized.append({"execution_identity": {**identity, "gpu": [fields[0] + ", " + fields[2]]}})
    require_execution_alignment(*normalized)
