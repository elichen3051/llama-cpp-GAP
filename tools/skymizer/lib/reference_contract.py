"""Pure-python ground-truth dataset contract validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

TOKEN_COLUMNS = (
    "input_ids",
    "input_tokens_len",
    "generated_tokens_len",
    "n_prefill_tokens",
    "labels",
)
REQUIRED_COLUMNS = (
    "id",
    "item_id",
    "source",
    "category",
    "question",
    "generated_texts",
    "seed",
    *TOKEN_COLUMNS,
)
GT_V12_COLUMNS = (*REQUIRED_COLUMNS, "finish_reason")
FS_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

# Rows produced by native llama.cpp (including legacy server schema v1.4) carry the engine's
# own prompt echo; these columns must agree with the token columns.
LLAMACPP_ENGINE = "llama.cpp"
LLAMACPP_REQUIRED_COLUMNS = (
    "llamacpp_prompt_string",
    "llamacpp_media_marker",
    "llamacpp_prompt_layout",
    "per_image_vision_token_counts",
    "llamacpp_n_past_prefill",
    "llamacpp_tokens_evaluated",
)


class GTContractError(ValueError):
    """Raised when a GT dataset violates the producer/consumer row contract."""


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("expected a sequence, not a string, bytes, or mapping")
    if isinstance(value, list):
        return value
    return list(value)


def _has_present_columns(row: Mapping[str, Any], columns: Iterable[str]) -> bool:
    return all(column in row and row[column] is not None for column in columns)


def validate_gt_row(row: Mapping[str, Any]) -> list[str]:
    """Return contract violations for one ground-truth row.

    Violation strings start with stable rule IDs:

    - R1-R6 are token-shape invariants.
    - R7 is required metadata / id aliasing.
    - R8 is filesystem-safe id syntax.
    """
    violations: list[str] = []

    for column in REQUIRED_COLUMNS:
        if column not in row:
            violations.append(f"R7 missing required column {column!r}")
        elif row[column] is None:
            violations.append(f"R7 required column {column!r} is None")

    if _has_present_columns(row, ("id", "item_id")) and row["id"] != row["item_id"]:
        violations.append(f"R7 id/item_id mismatch: id={row['id']!r}, item_id={row['item_id']!r}")

    if "id" in row and row["id"] is not None:
        row_id = str(row["id"])
        if FS_SAFE_ID_RE.fullmatch(row_id) is None:
            violations.append(f"R8 id is not filesystem-safe: {row_id!r}")

    if not _has_present_columns(row, TOKEN_COLUMNS):
        return violations

    try:
        input_ids = _as_list(row["input_ids"])
    except TypeError as exc:
        rule = "R11" if row.get("generation_engine") == LLAMACPP_ENGINE else "R1"
        violations.append(f"{rule} input_ids must be a sequence: {exc}")
        return violations
    try:
        labels = _as_list(row["labels"])
        input_tokens_len = int(row["input_tokens_len"])
        generated_tokens_len = int(row["generated_tokens_len"])
        n_prefill_tokens = int(row["n_prefill_tokens"])
    except (TypeError, ValueError, OverflowError) as exc:
        violations.append(f"R1 token columns must be sequence/int compatible: {exc}")
        return violations

    if len(input_ids) != input_tokens_len:
        violations.append(
            f"R1 len(input_ids) != input_tokens_len: " f"{len(input_ids)} != {input_tokens_len}"
        )

    if not (0 < n_prefill_tokens < input_tokens_len):
        violations.append(
            f"R2 n_prefill_tokens={n_prefill_tokens} out of range for "
            f"input_tokens_len={input_tokens_len} (expected 0 < n_prefill < len)"
        )

    if n_prefill_tokens + generated_tokens_len != input_tokens_len:
        violations.append(
            f"R3 n_prefill_tokens + generated_tokens_len != input_tokens_len: "
            f"{n_prefill_tokens} + {generated_tokens_len} != {input_tokens_len}"
        )

    if len(labels) != input_tokens_len:
        violations.append(
            f"R4 len(labels) != input_tokens_len: {len(labels)} != {input_tokens_len}"
        )

    if labels[:n_prefill_tokens] != [-100] * n_prefill_tokens:
        violations.append("R5 labels[:n_prefill_tokens] must all equal -100")

    if labels[n_prefill_tokens:] != input_ids[n_prefill_tokens:]:
        violations.append("R6 labels[n_prefill_tokens:] must equal input_ids[n_prefill_tokens:]")

    violations.extend(
        _llamacpp_row_violations(
            row, n_prefill_tokens=n_prefill_tokens, generated_tokens_len=generated_tokens_len
        )
    )
    return violations


def _int_or_none(value: Any) -> int | None:
    """``int(value)`` or ``None`` when the value is missing or not an integer."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _int_list_or_none(values: Any) -> list[int] | None:
    """All entries as ints, or ``None`` if ``values`` is not a list of integers."""
    if values is None or isinstance(values, (str, bytes, Mapping)):
        return None
    try:
        out = [_int_or_none(v) for v in values]
    except TypeError:
        return None
    return None if any(v is None for v in out) else out


def _llamacpp_row_violations(
    row: Mapping[str, Any], *, n_prefill_tokens: int, generated_tokens_len: int
) -> list[str]:
    """R10-R13: llama.cpp prompt-echo columns must match the token columns.

    - R10: the evaluated prompt length equals ``n_prefill_tokens``.
    - R11: the echoed prompt layout flattens to ``n_prefill_tokens`` tokens
      and its position count equals ``llamacpp_n_past_prefill``.
    - R12: per-image vision token counts sum to ``sum_vision_tokens`` and
      the prompt string carries one media marker per image.
    - R13: sampled-token log-probs, when recorded, cover every generated token.
    """
    if row.get("generation_engine") != LLAMACPP_ENGINE:
        return []
    violations: list[str] = []
    missing = [c for c in LLAMACPP_REQUIRED_COLUMNS if c not in row or row[c] is None]
    if missing:
        violations.append(f"R10 llama.cpp row is missing columns {missing}")
        return violations

    tokens_evaluated = _int_or_none(row["llamacpp_tokens_evaluated"])
    if tokens_evaluated is None:
        violations.append(
            f"R10 llamacpp_tokens_evaluated={row['llamacpp_tokens_evaluated']!r} is not an integer"
        )
    elif tokens_evaluated != n_prefill_tokens:
        violations.append(
            f"R10 llamacpp_tokens_evaluated={tokens_evaluated} != "
            f"n_prefill_tokens={n_prefill_tokens}"
        )

    counts = _int_list_or_none(row["per_image_vision_token_counts"])
    if counts is None:
        violations.append("R12 per_image_vision_token_counts is not a list of integers")
        return violations
    n_past_prefill = _int_or_none(row["llamacpp_n_past_prefill"])
    if n_past_prefill is None:
        violations.append(
            f"R11 llamacpp_n_past_prefill={row['llamacpp_n_past_prefill']!r} is not an integer"
        )
    try:
        input_ids = _as_list(row["input_ids"]) if row.get("input_ids") is not None else []
    except TypeError:
        violations.append("R11 input_ids must be a sequence")
        input_ids = []
    try:
        layout = json.loads(row["llamacpp_prompt_layout"])
        layout_n_tokens = int(layout["n_tokens"])
        layout_n_pos = int(layout["n_pos"])
        chunks = list(layout["chunks"])
    except (TypeError, ValueError, OverflowError, KeyError) as exc:
        violations.append(f"R11 llamacpp_prompt_layout is not a valid layout JSON: {exc}")
    else:
        if layout_n_tokens != n_prefill_tokens:
            violations.append(
                f"R11 prompt_layout n_tokens={layout_n_tokens} != n_prefill_tokens={n_prefill_tokens}"
            )
        if n_past_prefill is not None and layout_n_pos != n_past_prefill:
            violations.append(
                f"R11 prompt_layout n_pos={layout_n_pos} != llamacpp_n_past_prefill={n_past_prefill}"
            )
        violations.extend(_layout_chunks_violations(chunks, input_ids[:n_prefill_tokens], counts))

    sum_vision_tokens = row.get("sum_vision_tokens")
    if sum_vision_tokens is not None and sum(counts) != _int_or_none(sum_vision_tokens):
        violations.append(
            f"R12 sum(per_image_vision_token_counts)={sum(counts)} != "
            f"sum_vision_tokens={sum_vision_tokens!r}"
        )
    marker = str(row["llamacpp_media_marker"])
    n_markers = str(row["llamacpp_prompt_string"]).count(marker) if marker else -1
    if n_markers != len(counts):
        violations.append(
            f"R12 llamacpp_prompt_string has {n_markers} media marker(s) for "
            f"{len(counts)} image chunk(s)"
        )

    sha_list = row.get("image_bytes_sha256")
    if sha_list is not None:
        try:
            sha_list = _as_list(sha_list)
        except TypeError:
            violations.append("R12 image_bytes_sha256 must be a sequence")
        else:
            if len(sha_list) != len(counts):
                violations.append(
                    f"R12 len(image_bytes_sha256)={len(sha_list)} != {len(counts)} image chunk(s)"
                )
    num_images = row.get("num_images")
    if num_images is not None and _int_or_none(num_images) != len(counts):
        violations.append(f"R12 num_images={num_images!r} != {len(counts)} image chunk(s)")

    logprobs = row.get("generation_token_logprobs")
    if logprobs is not None:
        try:
            values = _as_list(logprobs)
        except TypeError:
            violations.append("R13 generation_token_logprobs must be a sequence")
            return violations
        if len(values) != generated_tokens_len:
            violations.append(
                f"R13 len(generation_token_logprobs)={len(values)} != "
                f"generated_tokens_len={generated_tokens_len}"
            )
        non_finite = 0
        for value in values:
            try:
                finite = value is not None and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            non_finite += not finite
        if non_finite:
            violations.append(
                f"R13 generation_token_logprobs has {non_finite} non-finite or non-numeric value(s)"
            )
    return violations


def _layout_chunks_violations(
    chunks: list[Any], prompt_ids: list[Any], per_image_counts: list[int]
) -> list[str]:
    """R11 detail: the layout's chunks must reproduce ``input_ids[:n_prefill]``.

    Text chunks carry their token ids verbatim; each image chunk must cover a
    run of identical placeholder ids of exactly ``n_tokens`` and match the
    ordered ``per_image_vision_token_counts`` entry.
    """
    pos = 0
    image_index = 0
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            return [f"R11 layout chunk {chunk_index} is not an object"]
        ctype = chunk.get("type")
        n_tokens = _int_or_none(chunk.get("n_tokens"))
        if n_tokens is None:
            return [f"R11 layout chunk {chunk_index} lacks a valid n_tokens"]
        if "start" in chunk and _int_or_none(chunk["start"]) != pos:
            return [
                f"R11 layout chunk {chunk_index} start={chunk['start']!r} != running offset {pos}"
            ]
        if ctype == "text":
            tokens = _int_list_or_none(chunk.get("tokens", []))
            if tokens is None:
                return [f"R11 layout text chunk {chunk_index} tokens is not a list of integers"]
            if len(tokens) != n_tokens:
                return [
                    f"R11 layout text chunk {chunk_index} carries {len(tokens)} ids "
                    f"for n_tokens={n_tokens}"
                ]
            prefix = _int_list_or_none(prompt_ids[pos : pos + n_tokens])
            if prefix is None:
                return [f"R11 input_ids prefix at index {pos} is not a list of integers"]
            if prefix != tokens:
                return [
                    f"R11 layout text chunk {chunk_index} differs from input_ids at prefix index {pos}"
                ]
        elif ctype == "image":
            span = _int_list_or_none(prompt_ids[pos : pos + n_tokens]) if n_tokens > 0 else []
            if span is None:
                return [f"R11 input_ids prefix at index {pos} is not a list of integers"]
            if n_tokens <= 0 or len(span) != n_tokens or len(set(span)) != 1:
                return [
                    f"R11 layout image chunk {chunk_index} does not cover a uniform placeholder "
                    f"run of {n_tokens} ids at prefix index {pos}"
                ]
            if image_index >= len(per_image_counts) or per_image_counts[image_index] != n_tokens:
                return [
                    f"R11 layout image chunk {chunk_index} n_tokens={n_tokens} does not match "
                    f"per_image_vision_token_counts[{image_index}]"
                ]
            image_index += 1
        else:
            return [f"R11 layout chunk {chunk_index} has unsupported type {ctype!r}"]
        pos += n_tokens
    violations: list[str] = []
    if pos != len(prompt_ids):
        violations.append(
            f"R11 layout chunks cover {pos} tokens, input_ids prefix has {len(prompt_ids)}"
        )
    if image_index != len(per_image_counts):
        violations.append(
            f"R11 layout has {image_index} image chunk(s) but per_image_vision_token_counts has "
            f"{len(per_image_counts)}"
        )
    return violations


def collect_gt_dataset_violations(dataset: Iterable[Mapping[str, Any]]) -> list[str]:
    """Collect row-indexed contract violations for a dataset-like iterable."""
    violations: list[str] = []
    seen_ids: dict[str, int] = {}
    for row_index, row in enumerate(dataset):
        for violation in validate_gt_row(row):
            violations.append(f"row {row_index}: {violation}")

        if "id" not in row or row["id"] is None:
            continue
        row_id = str(row["id"])
        if row_id in seen_ids:
            violations.append(
                f"row {row_index}: R9 duplicate id {row_id!r}; "
                f"first seen at row {seen_ids[row_id]}"
            )
        else:
            seen_ids[row_id] = row_index
    return violations


def validate_gt_dataset(dataset: Iterable[Mapping[str, Any]]) -> None:
    """Raise when any row or split-level GT contract invariant is violated."""
    violations = collect_gt_dataset_violations(dataset)
    if violations:
        raise GTContractError(
            "GT dataset contract violations:\n" + "\n".join(f"- {v}" for v in violations)
        )
