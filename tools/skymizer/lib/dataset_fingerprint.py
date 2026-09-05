#!/usr/bin/env python3
# =============================================================================
# dataset_fingerprint.py
#
# Whole-dataset content fingerprint recorded in every collector's
# collect_meta.json (provenance), verified by the downstream tools.
#
# Why: collect_meta.json records the dataset by NAME (dataset/subset/split/
# sort_by strings), which cannot see a dataset regenerated under the same
# name (same IDs, same lengths, different tokens or images). This module
# hashes the CONTENT the collector actually conditions on -- row id,
# input_ids, the n_prefill_tokens prompt/answer boundary, the generation
# model identity (when the column exists; it selects the tokenizer that
# formats VLM chat), and the encoded image bytes -- over the
# post-load_dataset_sorted view, in row order (row_idx keys the on-disk
# artifacts, so order is part of the identity). The value is scheme-tagged
# ("ds-v2:...") like the model fingerprints, so a future field-set change
# coexists instead of colliding. The collectors record the hashes when a
# --out is first created AND re-derive + verify them for every later shard
# (check_dataset_content_hash / check_model_fingerprints);
# saved_metrics_paired_compare additionally compares the recorded strings
# between dirs at use time.
#
# Cost, measured on real cached datasets (once per collector invocation):
#   llm-ground-truth (199 rows, 83k tokens)          0.02 s
#   coco ppl-1500    (1500 rows, 247 MB images)      0.42 s
#   mmmu-pro-500     (348 rows, 333 MB images)       1.19 s
# The HF fast path never decodes images (Image(decode=False) yields the
# encoded bytes) and only selects the hashed columns.
#
# Raw bytes are hashed deliberately: some datasets carry prep-side
# image_fingerprint columns, but trusting them would chain this guard's
# integrity to the prep pipeline's. Bytes are self-evident.
#
# Import-safe: module import needs numpy only; `datasets` is imported lazily
# inside the HF fast path (mirrors the collectors' hermetic-test constraint).
# =============================================================================

import hashlib
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np


def _update(h, tag: bytes, payload: bytes):
    """Length-prefixed update: concatenating variable-length fields without
    framing would let different (id, tokens) splits collide."""
    h.update(tag)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)


def _image_bytes(value):
    """Yield the encoded bytes of one image cell: a decode=False HF cell is
    {"bytes": ..., "path": ...}; a List(Image) cell is a list of those. The
    generic fallback also accepts raw bytes."""
    items = value if isinstance(value, list) else [value]
    for item in items:
        if item is None:
            continue
        if isinstance(item, (bytes, bytearray)):
            yield bytes(item)
        elif isinstance(item, Mapping):
            b = item.get("bytes")
            if b is None and item.get("path"):
                b = Path(item["path"]).read_bytes()
            if b is not None:
                yield b


def _resolve_id_column(column_names) -> str:
    # VLM ground-truth datasets use item_id; LLM ones use id.
    for name in ("item_id", "id"):
        if name in column_names:
            return name
    raise ValueError(
        f"dataset has neither 'item_id' nor 'id' column: {list(column_names)}")


# Scalar columns hashed beyond the id: n_prefill_tokens moves the
# teacher-forcing prompt/answer boundary (identical input_ids under a shifted
# boundary is DIFFERENT conditioning), and generation_model_name_or_path
# selects the tokenizer the VLM prep uses to reconstruct formatted chat.
# Hashed when present, with the same normalization in both paths so the HF
# fast path and the generic fallback stay hash-equal.
_SCALAR_FIELDS = (("n_prefill_tokens", b"pre"),
                  ("generation_model_name_or_path", b"gen"))

DATASET_HASH_SCHEME = "ds-v2"


def _hash_hf(ds) -> str:
    """Fast path for a datasets.Dataset: select only the hashed columns and
    read image cells as encoded bytes, never decoding to PIL."""
    from datasets import Image as HFImage  # lazy: heavy dep, HF path only

    id_col = _resolve_id_column(ds.column_names)
    img_cols, keep = [], [id_col]
    if "input_ids" in ds.column_names:
        keep.append("input_ids")
    scalar_cols = [(name, tag) for name, tag in _SCALAR_FIELDS
                   if name in ds.column_names]
    keep.extend(name for name, _ in scalar_cols)
    for name, feat in ds.features.items():
        if type(feat).__name__ == "Image":
            img_cols.append((name, HFImage(decode=False)))
        elif hasattr(feat, "feature") and type(feat.feature).__name__ == "Image":
            # List(Image) / Sequence(Image): rebuild the same outer feature
            # around a decode=False Image (mmmu-pro uses List(Image)).
            img_cols.append((name, type(feat)(HFImage(decode=False))))
    view = ds.select_columns(keep + [c for c, _ in img_cols])
    for name, no_decode in img_cols:
        view = view.cast_column(name, no_decode)

    h = hashlib.sha256()
    for batch in view.iter(batch_size=64):
        for j in range(len(batch[id_col])):
            _update(h, b"id", str(batch[id_col][j]).encode())
            if "input_ids" in keep:
                _update(h, b"tok",
                        np.asarray(batch["input_ids"][j], dtype=np.int32).tobytes())
            for name, tag in scalar_cols:
                _update(h, tag, str(batch[name][j]).encode())
            for name, _ in img_cols:
                for b in _image_bytes(batch[name][j]):
                    _update(h, b"img", b)
    return h.hexdigest()


def _hash_generic(ds) -> str:
    """Fallback for any len+getitem mapping-row dataset (tests, fakes)."""
    h = hashlib.sha256()
    for i in range(len(ds)):
        row = ds[i]
        rid = row.get("item_id", row.get("id"))
        if rid is None:
            raise ValueError(f"row {i} has neither 'item_id' nor 'id'")
        _update(h, b"id", str(rid).encode())
        if "input_ids" in row:
            _update(h, b"tok",
                    np.asarray(row["input_ids"], dtype=np.int32).tobytes())
        for name, tag in _SCALAR_FIELDS:
            if name in row:
                _update(h, tag, str(row[name]).encode())
        for key in ("image", "images"):
            if key in row:
                for b in _image_bytes(row[key]):
                    _update(h, b"img", b)
    return h.hexdigest()


def dataset_content_hash(ds) -> str:
    """Scheme-tagged sha256 over (row id, input_ids, n_prefill_tokens,
    generation model, encoded image bytes) in dataset order. ds-v2 added the
    boundary and generation-model fields; no unprefixed v1 value ever shipped,
    so there is no legacy decode path."""
    if hasattr(ds, "select_columns") and hasattr(ds, "features"):
        return f"{DATASET_HASH_SCHEME}:{_hash_hf(ds)}"
    return f"{DATASET_HASH_SCHEME}:{_hash_generic(ds)}"


def check_content_fingerprint(stored: Mapping, current: Mapping, field: str,
                              *, subject: str, explanation: str) -> str | None:
    """Three-tier content-identity check for extending an --out with a later
    shard, deliberately NOT an IDENTITY_FIELDS entry: a plain comparison
    would hard-fail every legacy dir whose stored meta predates the field.
    Semantics: both present and different -> refuse the dir (sys.exit, like
    an identity mismatch); stored missing -> return a warning string for the
    caller to print (identity unverified, the shard proceeds); current
    missing (library callers without the input in hand) -> no-op. The stored
    meta is never rewritten -- absence keeps meaning "predates"."""
    cur = current.get(field)
    if cur is None:
        return None
    old = stored.get(field)
    if old is None:
        return (f"WARNING: {subject} identity unverified "
                f"(collect_meta.json predates {field})")
    if old != cur:
        sys.exit(
            f"collect_meta.json mismatch: {field}\n"
            f"  stored:  {old}\n"
            f"  current: {cur}\n"
            f"{explanation}\n"
            "Use a different --out (nothing was deleted).")
    return None


def check_dataset_content_hash(stored: Mapping, current: Mapping) -> str | None:
    """Dataset-content tier of the shard-identity check: pins the HF dataset
    (dataset/subset/split) by CONTENT. The hash covers the whole
    post-load_dataset_sorted view, so which --start/--end window a shard
    collects does not affect it -- only a regenerated/edited dataset does."""
    return check_content_fingerprint(
        stored, current, "dataset_content_hash",
        subject="dataset content",
        explanation=(
            "The dataset's CONTENT changed under the same name (regenerated "
            "rows, drifted revision, or edited local files); a later shard "
            "would mix trajectories from two dataset versions in one dir."))


def check_model_fingerprints(stored: Mapping, current: Mapping,
                             fields) -> list[str]:
    """Model-file tiers of the shard-identity check: one three-tier check per
    fingerprint field (model_fingerprint, ref_model_fingerprint, ...).
    Returns the legacy-dir warning strings; mismatches sys.exit inside
    check_content_fingerprint."""
    warnings = []
    for field in fields:
        role = field.removesuffix("_fingerprint")
        warning = check_content_fingerprint(
            stored, current, field,
            subject=f"{role} file content",
            explanation=(
                f"The {role} file's CONTENT changed under the same path "
                "(requantized or replaced in place); a later shard would mix "
                "rows scored by two different models in one output dir."))
        if warning:
            warnings.append(warning)
    return warnings


_SAMPLE_BLOCK = 4 << 20   # 4 MiB per sampled window
_SAMPLE_STRIDES = 17      # 16 interior offsets at int(size * i / 17)
FILE_FINGERPRINT_SCHEME = "gguf-sampled-v1"


def file_content_fingerprint(path) -> str:
    """Strided sampled content fingerprint of a model file, scheme-tagged.

    sha256 over the file size plus 4 MiB windows at deterministic offsets
    (0, int(size*i/17) for i in 1..16, and size-4MiB) -- a fixed ~72 MiB
    read regardless of file size: 0.037 s on an 8 GB GGUF where a full
    sha256 costs 10 s and scales linearly (~75 s at 60 GB). Coverage by
    component: the size catches any length change (truncation, different
    quant type); the first window spans the GGUF header + metadata KV +
    tensor index, so structural changes surface there; the strided windows
    sample the tensor data, which realistic modifications (requantization
    with a different imatrix, a different upstream checkpoint) alter
    throughout the file. A byte flip BETWEEN windows is invisible by
    design -- this guards against in-place replacement accidents, not
    adversaries. Byte-based, no GGUF parsing: any file size works (small
    files' windows overlap into full coverage). The scheme prefix lets a
    future full-hash variant coexist in the same meta field."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    _update(h, b"size", str(size).encode())
    offsets = {0, max(0, size - _SAMPLE_BLOCK)}
    offsets.update(int(size * i / _SAMPLE_STRIDES)
                   for i in range(1, _SAMPLE_STRIDES))
    with open(path, "rb") as f:
        for off in sorted(offsets):
            f.seek(off)
            _update(h, b"blk", f.read(_SAMPLE_BLOCK))
    return f"{FILE_FINGERPRINT_SCHEME}:{h.hexdigest()}"


def maybe_file_fingerprint(path) -> str | None:
    """file_content_fingerprint when `path` is an existing file, else None.

    build_collect_meta computes fingerprints itself (so every caller gets
    the guard, not just main()) but its unit tests pass fake model paths --
    the None branch omits the field instead of failing meta construction.
    Real collector runs validate the paths before main() reaches here."""
    p = Path(path)
    return file_content_fingerprint(p) if p.is_file() else None


def logged_file_fingerprint(label: str, path) -> str:
    """file_content_fingerprint + the collectors' startup stderr line
    (scheme + 16 hex digits + timing), mirroring the dataset-hash log."""
    t0 = time.time()
    fp = file_content_fingerprint(path)
    print(f"  {label} = {fp[:len(FILE_FINGERPRINT_SCHEME) + 17]}… "
          f"({time.time() - t0:.3f}s)", file=sys.stderr)
    return fp
