# =============================================================================
# kld_metrics_io.py
#
# Shared helpers for parsing VLMK .bin metric dumps (produced by collect_kld.py
# via llama-vlm-kld) and their lossless .npz counterparts.
#
# A VLMK dump holds PER-ANSWER-TOKEN fidelity metrics computed on the fly from
# a (reference, candidate) model pair — no logits are stored. One position is
# one packed record (see KLD_RECORD_DT; 44 bytes at the current version 2,
# 40 bytes for legacy v1 dumps without `ear`); a 1024-position item is ~44 KiB
# vs ~600 MiB of fp32 dense logits at vocab ~152k (Qwen3-VL).
#
# Version history:
#     v1  40-byte records (kld .. entropy_cand, target/argmax_ref/argmax_cand)
#     v2  44-byte records: + float32 `ear` (Expected Acceptance Rate,
#         sum min(p_ref, p_cand) = 1 - TV distance; arXiv:2605.02404) between
#         entropy_cand and target
# Readers accept both versions (per-version dtype dispatch); the collectors
# only WRITE the current version (their scorer binary is version-gated).
#
# Exposed surface:
#     VLMK_MAGIC / VLMK_VERSION       format identity (VLMK_VERSION = writer-
#                                     current; VLMK_SUPPORTED_VERSIONS readable,
#                                     derived from the per-version dtype table)
#     assert_kld_current_version(h,p) collectors' writer-current gate
#     KLD_HEADER_DT / KLD_RECORD_DT   on-disk layouts (keep in sync with the
#                                     kld_record struct in vlm-kld.cpp);
#                                     KLD_RECORD_DT is the CURRENT version's
#     KLD_METRIC_KEYS                 current record field names, on-disk order
#     kld_record_dt(version)          record dtype for a specific version
#     kld_metric_keys(version)        record field names for a specific version
#     read_kld_header(path)           header fields without reading the payload
#     assert_kld_npz_payload(path)    validate an .npz's columns sans decompress
#     load_kld_metrics(path)          -> (metrics dict of 1-d arrays, header)
#     assert_kld_file_complete(path)  validate a .bin's size against its header
#     convert_kld_bin_to_npz(b, n)    lossless VLMK .bin -> .npz conversion
#
# The .npz schema stores the header fields as 0-d uint32 arrays
# (KLD_NPZ_HEADER_KEYS) and each record column as its own 1-d array, so
# postprocessing needs nothing but numpy (or torch.from_numpy):
#
#     with np.load("metrics/000_test_X.npz") as z:
#         kld = z["kld"]              # (npos,) float32
#         same_top = z["argmax_ref"] == z["argmax_cand"]
# =============================================================================

import os
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


def read_npz_member_headers(path) -> dict[str, tuple[tuple, np.dtype]]:
    """{member_name: (shape, dtype)} for every array in an .npz, WITHOUT
    decompressing any payload.

    np.load(...)[name] inflates the whole member; this instead parses each
    member's npy header (magic + version + header dict, a few hundred bytes)
    through the zip stream. Cost contract: sub-ms per file, safe to run per
    row before any compare. Raises ValueError on a member that is not a
    parseable .npy (zip corruption, foreign files, unsupported npy version).
    """
    headers: dict[str, tuple[tuple, np.dtype]] = {}
    with zipfile.ZipFile(Path(path)) as zf:
        for member in zf.namelist():
            name = member[:-4] if member.endswith(".npy") else member
            with zf.open(member) as fp:
                try:
                    version = npy_format.read_magic(fp)
                    if version == (1, 0):
                        shape, _fortran, dtype = npy_format.read_array_header_1_0(fp)
                    elif version == (2, 0):
                        shape, _fortran, dtype = npy_format.read_array_header_2_0(fp)
                    else:
                        raise ValueError(f"unsupported npy version {version}")
                except Exception as e:
                    raise ValueError(
                        f"{path}: member {member!r} is not a readable .npy "
                        f"header: {e}") from e
            headers[name] = (tuple(shape), dtype)
    return headers


VLMK_MAGIC = 0x564C4D4B   # "VLMK"
VLMK_VERSION = 2          # writer-current version (44-byte records with ear)

# Sixth word: llama.cpp's OWN position count after prefill. n_prefill (the
# fifth) is the HF ground-truth sequential length echoed from the manifest,
# which for a VLM differs from it by design (M-RoPE counts one position per
# merged patch group) and does NOT move when the vision-token budget changes.
# n_past_actual does, so it is the on-disk witness of the real conditioning.
# 0 = not recorded (every dump written before the field existed).
KLD_HEADER_DT = np.dtype([
    ("magic", "<u4"), ("version", "<u4"), ("vocab", "<u4"),
    ("npos", "<u4"), ("n_prefill", "<u4"), ("n_past_actual", "<u4"),
])

# One record per scored answer position; field order/types mirror the packed
# kld_record struct in vlm-kld.cpp. v1 = 40 bytes; v2 adds float32 `ear`
# (Expected Acceptance Rate, sum min(p_ref, p_cand) = 1 - TV) -> 44 bytes.
KLD_RECORD_DT_V1 = np.dtype([
    ("kld", "<f4"), ("reversed_kld", "<f4"), ("js_kld", "<f4"),
    ("nll_ref", "<f4"), ("nll_cand", "<f4"),
    ("entropy_ref", "<f4"), ("entropy_cand", "<f4"),
    ("target", "<i4"), ("argmax_ref", "<i4"), ("argmax_cand", "<i4"),
])
KLD_RECORD_DT_V2 = np.dtype([
    ("kld", "<f4"), ("reversed_kld", "<f4"), ("js_kld", "<f4"),
    ("nll_ref", "<f4"), ("nll_cand", "<f4"),
    ("entropy_ref", "<f4"), ("entropy_cand", "<f4"), ("ear", "<f4"),
    ("target", "<i4"), ("argmax_ref", "<i4"), ("argmax_cand", "<i4"),
])
assert KLD_RECORD_DT_V1.itemsize == 40, "v1 record layout drifted"
assert KLD_RECORD_DT_V2.itemsize == 44, "v2 record layout drifted from vlm-kld.cpp"

# The ONE table the version policy derives from: readable versions are its
# keys, the writer-current layout is its VLMK_VERSION entry. A v3 bump edits
# this dict + VLMK_VERSION (+ the C++ constant); nothing else is hand-listed.
_KLD_RECORD_DT_BY_VERSION = {1: KLD_RECORD_DT_V1, 2: KLD_RECORD_DT_V2}
VLMK_SUPPORTED_VERSIONS = tuple(sorted(_KLD_RECORD_DT_BY_VERSION))
assert VLMK_VERSION in _KLD_RECORD_DT_BY_VERSION, "VLMK_VERSION has no record layout"


def kld_record_dt(version: int) -> np.dtype:
    """Record dtype for one VLMK format version; raises on unknown versions."""
    _check_supported_version(version, None)
    return _KLD_RECORD_DT_BY_VERSION[version]


def kld_metric_keys(version: int) -> tuple[str, ...]:
    """Record field names (on-disk order) for one VLMK format version."""
    return kld_record_dt(version).names


def _check_supported_version(version: int, path) -> None:
    """The single 'unsupported VLMK version' raise shared by every reader."""
    if version not in _KLD_RECORD_DT_BY_VERSION:
        where = f"{path}: " if path is not None else ""
        raise ValueError(
            f"{where}unsupported VLMK version {version} "
            f"(this reader understands {VLMK_SUPPORTED_VERSIONS})")


def assert_kld_current_version(header: dict, path) -> dict:
    """Writer-current gate for the collectors: a dump this process just
    produced (or is about to keep) must carry VLMK_VERSION — an older version
    means the scorer binary on disk predates this Python checkout."""
    if header["version"] != VLMK_VERSION:
        raise ValueError(
            f"{path}: VLMK version {header['version']} != current "
            f"{VLMK_VERSION} — the scorer binary is outdated; rebuild it "
            "(cmake --build build --target llama-vlm-kld llama-llm-kld)")
    return header


# Writer-current aliases: code that only ever handles freshly-collected dumps
# (the collectors' validation, tests synthesizing current-version fixtures)
# uses these; readers of possibly-old dumps go through kld_record_dt(version).
KLD_RECORD_DT = kld_record_dt(VLMK_VERSION)
KLD_METRIC_KEYS = KLD_RECORD_DT.names

# Header fields preserved in a .npz dump as 0-d uint32 arrays (magic excluded —
# the .npz container itself plays that role).
KLD_NPZ_HEADER_KEYS = ("version", "vocab", "npos", "n_prefill")
# Optional in an .npz: dumps converted before the field existed have only the
# four required members, and read back as 0 = not recorded.
KLD_NPZ_OPTIONAL_HEADER_KEYS = ("n_past_actual",)


def _decode_kld_header(raw: bytes, path: Path) -> dict[str, int]:
    if len(raw) != KLD_HEADER_DT.itemsize:
        raise ValueError(
            f"{path}: header short: got {len(raw)} expected {KLD_HEADER_DT.itemsize}")
    h = np.frombuffer(raw, KLD_HEADER_DT)[0]
    if int(h["magic"]) != VLMK_MAGIC:
        raise ValueError(f"{path}: bad VLMK magic {int(h['magic']):#x}")
    _check_supported_version(int(h["version"]), path)
    return {
        "version": int(h["version"]),
        "vocab": int(h["vocab"]),
        "npos": int(h["npos"]),
        "n_prefill": int(h["n_prefill"]),
        "n_past_actual": int(h["n_past_actual"]),
    }


def read_kld_header(path) -> dict[str, int]:
    """Read only the header fields; the payload is never loaded
    (.bin: fixed-size prefix read; .npz: 0-d zip members only)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            h = {k: int(z[k]) for k in KLD_NPZ_HEADER_KEYS}
            for k in KLD_NPZ_OPTIONAL_HEADER_KEYS:
                h[k] = int(z[k]) if k in z else 0
        _check_supported_version(h["version"], path)
        return h
    with open(path, "rb") as f:
        return _decode_kld_header(f.read(KLD_HEADER_DT.itemsize), path)


def assert_kld_file_complete(path) -> dict[str, int]:
    """Validate a VLMK .bin's on-disk size against its own header npos, so a
    file truncated by a crashed/killed scorer fails loudly instead of being
    trusted. Returns the header on success. (The scorer commits via
    tmp + fsync + atomic rename, so this is defense in depth.)"""
    path = Path(path)
    if path.suffix == ".npz":
        # The raw-bytes size formula below is meaningless for a zip container;
        # validate an .npz by reading its header members instead.
        raise ValueError(f"{path}: expected a VLMK .bin (use read_kld_header for .npz)")
    h = read_kld_header(path)
    record_dt = kld_record_dt(h["version"])
    expected = KLD_HEADER_DT.itemsize + h["npos"] * record_dt.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: size {actual} != expected {expected} for npos={h['npos']} "
            f"at version {h['version']}; truncated or corrupt dump")
    return h


def assert_kld_npz_payload(path, header: dict | None = None) -> dict:
    """Validate a VLMK .npz's metric columns against its own header WITHOUT
    decompressing them (zip-member npy headers only, see
    read_npz_member_headers; sub-ms per file). Every
    KLD_METRIC_KEYS column must be present with shape (npos,) and its
    KLD_RECORD_DT field dtype. This is the single source of the
    expected-member contract the loader (load_kld_metrics) relies on -- a
    header-only .npz would otherwise crash the loader downstream.
    Raises ValueError on any mismatch; returns the header on success."""
    path = Path(path)
    header = header if header is not None else read_kld_header(path)
    members = read_npz_member_headers(path)
    npos = header["npos"]
    record_dt = kld_record_dt(header["version"])
    problems = []
    for key in record_dt.names:
        if key not in members:
            problems.append(f"{key!r} missing")
            continue
        shape, dtype = members[key]
        if shape != (npos,):
            problems.append(f"{key!r} shape {shape} != ({npos},)")
        elif dtype != record_dt[key]:
            problems.append(f"{key!r} dtype {dtype} != {record_dt[key]}")
    if problems:
        raise ValueError(
            f"{path}: metric columns do not match the dump's own header "
            f"(npos={npos}, version={header['version']}): " + "; ".join(problems))
    return header


def load_kld_metrics(path):
    """
    Load a VLMK dump (.bin or its lossless .npz counterpart; dispatch on
    suffix). Returns (metrics, header):

        metrics: dict keyed by the dump version's record field names (see
                 kld_metric_keys; v1 dumps have no "ear" column); each value
                 is a contiguous (npos,) array — float32 for the metric
                 columns, int32 for target/argmax_ref/argmax_cand.
        header:  {"version", "vocab", "npos", "n_prefill", "n_past_actual"}
                 ints (n_past_actual is 0 for dumps predating the field).
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            header = {k: int(z[k]) for k in KLD_NPZ_HEADER_KEYS}
            for k in KLD_NPZ_OPTIONAL_HEADER_KEYS:
                header[k] = int(z[k]) if k in z else 0
            _check_supported_version(header["version"], path)
            record_dt = kld_record_dt(header["version"])
            missing = [k for k in record_dt.names if k not in z]
            if missing:
                raise ValueError(f"{path}: metric column(s) missing: {missing}")
            metrics = {k: z[k] for k in record_dt.names}
        # Mirror the .bin branch's payload-vs-header validation with the SAME
        # schema table assert_kld_npz_payload uses (KLD_RECORD_DT): the .bin
        # branch gets dtypes for free from the structured read, so the .npz
        # branch must check them explicitly or a float64 metric column / an
        # int64 index column with the right shape loads silently and poisons
        # every downstream mean at a different precision than the contract.
        npos = header["npos"]
        problems = []
        for k in record_dt.names:
            m = metrics[k]
            if m.shape != (npos,):
                problems.append(f"{k!r} shape {m.shape} != ({npos},)")
            elif m.dtype != record_dt[k]:
                problems.append(f"{k!r} dtype {m.dtype} != {record_dt[k]}")
        if problems:
            raise ValueError(
                f"{path}: metric columns do not match the dump's own header "
                f"(npos={npos}, version={header['version']}): "
                + "; ".join(problems))
        return metrics, header
    # Exact-size gate first (assert_kld_file_complete: short AND trailing
    # bytes fail closed) so loader strictness never depends on which caller
    # remembered to pre-validate.
    header = assert_kld_file_complete(path)
    record_dt = kld_record_dt(header["version"])
    with open(path, "rb") as f:
        f.seek(KLD_HEADER_DT.itemsize)
        records = np.fromfile(f, dtype=record_dt, count=header["npos"])
    if records.size != header["npos"]:
        raise ValueError(
            f"{path}: payload short: got {records.size} records, "
            f"header says {header['npos']}")
    # np.ascontiguousarray: a structured-array field view is strided (stride
    # 40/44); copying gives downstream numpy/torch dense 1-d arrays.
    metrics = {k: np.ascontiguousarray(records[k]) for k in record_dt.names}
    return metrics, header


def item_means(m, keep=None):
    """Per-item means of one VLMK dump's metric columns, over the first
    `keep` positions (None = all) -- the ONE VLMK per-item aggregation
    (saved_metrics_paired_compare's per-item scores). Every column is
    upcast to float64 before averaging. `ear` is present only when the
    dump carries it (VLMK
    v2). Returns (scores, dp): the flat score dict and the signed per-token
    delta-p in percentage points, which callers derive tails / per-token
    columns from."""
    sl = slice(None, keep)
    nll_ref = m["nll_ref"][sl].astype(np.float64)
    nll_cand = m["nll_cand"][sl].astype(np.float64)
    dp = (np.exp(-nll_cand) - np.exp(-nll_ref)) * 100.0
    scores = {
        "nll": float(nll_cand.mean()),
        "kld": float(m["kld"][sl].astype(np.float64).mean()),
        "reversed_kld": float(m["reversed_kld"][sl].astype(np.float64).mean()),
        "js_kld": float(m["js_kld"][sl].astype(np.float64).mean()),
        "same_top_rate": float((m["argmax_ref"][sl] == m["argmax_cand"][sl]).mean()),
        "mse_dp": float((dp * dp).mean()),
        "mean_dp": float(dp.mean()),
        "nll_ref": float(nll_ref.mean()),
        "entropy": float(m["entropy_cand"][sl].astype(np.float64).mean()),
    }
    if "ear" in m:
        scores["ear"] = float(m["ear"][sl].astype(np.float64).mean())
    return scores, dp


def convert_kld_bin_to_npz(bin_path, npz_path) -> None:
    """Losslessly convert a VLMK .bin dump to the .npz schema (one 1-d array
    per record column + 0-d uint32 header fields; uncompressed np.savez).

    Writes <npz_path>.tmp, fsyncs, then atomically renames — neither a process
    crash nor a power loss can leave a complete-looking .npz whose data was
    never persisted (the caller deletes the .bin right after, so a
    silently-empty .npz would be unrecoverable). Deleting the .bin is the
    caller's responsibility."""
    bin_path, npz_path = Path(bin_path), Path(npz_path)
    h = assert_kld_file_complete(bin_path)
    metrics, _ = load_kld_metrics(bin_path)

    header = {k: np.uint32(h[k]) for k in KLD_NPZ_HEADER_KEYS}
    for k in KLD_NPZ_OPTIONAL_HEADER_KEYS:
        header[k] = np.uint32(h.get(k, 0))
    # np.savez appends ".npz" to a str/Path that lacks it, which would mangle
    # the ".tmp" name — hand it an open file object instead.
    tmp_path = npz_path.with_name(npz_path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, **header, **metrics)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, npz_path)
    finally:
        tmp_path.unlink(missing_ok=True)
