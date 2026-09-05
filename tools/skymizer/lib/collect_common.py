#!/usr/bin/env python3
"""Small import-safe helpers shared by both KLD collectors (llm/vlm).
VisionBudgetReporter lives in collect_vision.py."""

import csv
import datetime
import errno
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


from lib.dataset_fingerprint import (
    check_dataset_content_hash,
    check_model_fingerprints,
)
from lib.kld_metrics_io import VLMK_VERSION
from lib.collect_vision import VisionBudgetReporter  # noqa: F401  (re-exported)


SKIP_OVER_BUDGET = "SKIP_OVER_BUDGET"


def check_manifest_header(manifest_path: Path, columns) -> list[str]:
    """The manifest's ACTUAL header (its first line), required to equal the
    current collector schema `columns` EXACTLY before rows are appended
    (disjoint --start/--end shards append to one manifest). A missing or
    0-byte manifest is fresh and returns []. Older-layout headers are not
    extended; a torn last line (no terminator) is refused rather than
    repaired. Raises ValueError — callers turn it into a loud exit BEFORE
    any expensive preflight."""
    columns = list(columns)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return []
    with open(manifest_path, "rb") as f:
        head = f.read(64 * 1024)
        f.seek(-1, 2)
        torn = f.read(1) not in (b"\n", b"\r")
    # utf-8-sig drops a BOM (a manifest re-saved as "CSV UTF-8" would otherwise
    # refuse with two visually identical headers); splitlines handles CR-only
    # and CRLF files, which csv.reader would reject outright with csv.Error.
    lines = head.decode("utf-8-sig", "replace").splitlines()
    try:
        fieldnames = next(csv.reader(lines[:1]), [])
    except csv.Error as e:
        raise ValueError(f"{manifest_path}: unreadable header line ({e}); "
                         "use a fresh --out") from None
    if fieldnames != columns:
        raise ValueError(
            f"{manifest_path}: existing header {fieldnames} != current manifest "
            f"schema {columns}; refusing to append — use a fresh --out")
    if torn:
        raise ValueError(
            f"{manifest_path}: last line is not newline-terminated (a run was "
            "killed mid-write, or an editor dropped the final newline); "
            "refusing to append. Append a newline to that line if it is a "
            "complete record, otherwise remove it — or use a fresh --out")
    return fieldnames


def manifest_row_writer(mf, manifest_path: Path, columns):
    """Row writer for the manifest: a missing/0-byte file gets `columns` as
    its header; an existing file must already carry exactly that header
    (check_manifest_header). Returns write_row(row): `row` is a dict of
    column -> value (unlisted columns are written blank — FAIL/SKIP rows name
    only what they have) or a full positional list ordered as `columns`."""
    columns = list(columns)
    if not check_manifest_header(manifest_path, columns):
        csv.writer(mf).writerow(columns)
        mf.flush()
    writer = csv.DictWriter(mf, fieldnames=columns, restval="")
    column_set = set(columns)

    def write_row(row):
        if isinstance(row, dict):
            unknown = [k for k in row if k not in column_set]
            if unknown:
                raise ValueError(f"manifest row has unknown column(s) {unknown}")
            writer.writerow(row)
            return
        if len(row) != len(columns):
            raise ValueError(
                f"manifest row has {len(row)} fields, expected "
                f"{len(columns)} ({columns})")
        writer.writerow(dict(zip(columns, row)))
    return write_row


# ---------------------------------------------------------------------------
# Output-root policy: never skip, never delete, refuse on collision.
#
# A collector run writes every requested row exactly once. Re-running the same
# window, or any window that overlaps rows that already have output, is an
# error detected BEFORE any work; disjoint --start/--end windows may be
# collected sequentially into one --out. Nothing is ever deleted or
# overwritten by a collector.
# ---------------------------------------------------------------------------

LOCK_FILE_NAME = ".collect.lock"


def acquire_out_lock(out_dir: Path):
    """Exclusive advisory lock on the output root for the life of the
    process (the returned file object must be kept referenced). Sequential
    shards are supported; two collectors sharing one root concurrently are
    not (they share manifest.csv, logs/ and _manifest.jsonl), so the second
    one is refused with RuntimeError."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import fcntl        # POSIX-only: imported here so the read-only consumers
                        # of this module stay importable on non-POSIX platforms
    lock_path = out_dir / LOCK_FILE_NAME
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        fh.close()
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            raise RuntimeError(
                f"{out_dir}: another collector holds {LOCK_FILE_NAME}; refusing "
                "to run two collectors in one output root concurrently") from None
        # Anything else (ENOLCK/EOPNOTSUPP on a mount without advisory locks,
        # EROFS, ...) is NOT contention: say what actually failed instead of
        # sending the operator hunting a process that does not exist.
        raise RuntimeError(
            f"{out_dir}: cannot lock {LOCK_FILE_NAME}: {e.strerror} "
            f"(errno {e.errno}); if this filesystem does not support advisory "
            "locks, collect to one that does") from e
    return fh


def manifest_row_statuses(manifest_path: Path) -> dict[int, str]:
    """row_idx -> LAST recorded status for every row that has ANY record in
    the manifest (OK, FAIL_*, SKIP_OVER_BUDGET, ...). Strict: a row whose
    row_idx is not an integer means the manifest is damaged -> ValueError
    (the historical readers skip such lines; the collision scan must not)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {}
    statuses: dict[int, str] = {}
    # utf-8-sig, matching append_manifest_row's header probe: a manifest
    # re-saved as "CSV UTF-8" carries a BOM, which plain utf-8 leaves glued to
    # the first fieldname ("\ufeffrow_idx") so EVERY row_idx reads as missing
    # and this scan raises "damaged manifest — use a fresh --out" about a
    # perfectly good file.
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("row_idx") or "").strip()
            try:
                idx = int(raw)
            except ValueError:
                raise ValueError(
                    f"{manifest_path}:{reader.line_num}: row_idx {raw!r} is not an "
                    "integer; damaged manifest — use a fresh --out") from None
            statuses[idx] = (row.get("status") or "").strip() or "<no status>"
    return statuses


ARTIFACT_SUBDIRS = ("metrics", "_prep")


def row_stem_index(name: str) -> int | None:
    """The row index encoded in an artifact name f"{idx:03d}_{item_id}", or
    None when the entry is not a row artifact at all (README/notes files,
    editor swapfiles, .DS_Store, NFS silly-renames). The ONE parser both
    output scans use, so they can never disagree about what counts as a
    row's output."""
    head, sep, _rest = name.partition("_")
    # isdecimal, NOT isdigit: '²'.isdigit() is True but int('²') raises, and a
    # foreign filename must be skipped, never abort the scan. Any padding is
    # accepted ("0001_x" is row 1) so a non-canonically named leftover cannot
    # slip past the scan and get silently overwritten by the scorer.
    if not sep or not head.isdecimal():
        return None
    return int(head)


def scan_output_collisions(out_dir: Path, row_idxs, manifest_path: Path) -> list[str]:
    """Everything under `out_dir` that already belongs to one of `row_idxs`:
    any entry in the artifact subdirs whose name starts with the row's stem
    prefix `{idx:03d}_` (dumps, sidecars, .tmp/.rejected leftovers, prep
    dirs — whatever item_id they carry, so item-id drift cannot hide a
    collision), plus manifest records — with ONE exception: a row whose
    LAST manifest status is FAIL_<...> (exactly the collectors' own failure
    statuses) and which left no artifact at all (a
    transient prep failure: a truncated cache image, an OOM) is retryable,
    not a collision, so the row can be re-collected in place. Its retry
    appends a second record; the last-wins readers resolve it. Any FAIL row
    that did leave files (e.g. a rejected KLD dump kept as evidence) still
    collides — the operator must look at the evidence first. Returns
    human-readable conflict lines (empty = clean). Read-only."""
    out_dir = Path(out_dir)
    wanted = set(int(i) for i in row_idxs)
    if not wanted:
        return []
    conflicts: list[str] = []
    artifact_idxs: set[int] = set()
    for sub in ARTIFACT_SUBDIRS:
        d = out_dir / sub
        if not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            name = entry.name
            idx = row_stem_index(name)
            if idx is not None and idx in wanted:
                artifact_idxs.add(idx)
                conflicts.append(f"row {idx}: {sub}/{name}")
    statuses = manifest_row_statuses(manifest_path)
    for idx in sorted(set(statuses) & wanted):
        status = statuses[idx]
        # exactly the collectors' FAIL_<ExcType>/FAIL_EXIT_<rc> statuses are
        # retryable; foreign strings (FAILED, FAILURE, bare FAIL) fail closed
        if status.startswith("FAIL_") and idx not in artifact_idxs:
            continue                     # transient failure, nothing on disk: retryable
        conflicts.append(f"row {idx}: {Path(manifest_path).name} status={status}")
    return sorted(conflicts, key=lambda c: (int(c.split()[1].rstrip(":")), c))


def root_has_prior_output(out_dir: Path, manifest_path: Path) -> str | None:
    """A description of the first prior output found ANYWHERE under the root
    (any manifest record, any entry in an artifact subdir), or None for a
    genuinely fresh root. A root that already holds rows but has no
    collect_meta.json cannot have an identity assigned to it after the fact
    — main() refuses instead of writing a fresh meta over unknown rows."""
    out_dir = Path(out_dir)
    statuses = manifest_row_statuses(manifest_path)
    if statuses:
        idx = min(statuses)
        return f"{Path(manifest_path).name} holds {len(statuses)} row record(s) (e.g. row {idx})"
    for sub in ARTIFACT_SUBDIRS:
        d = out_dir / sub
        if not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            # only ROW artifacts count: a stray notes file or editor swapfile
            # must not brick a root that holds no rows (same parser as
            # scan_output_collisions).
            if row_stem_index(entry.name) is not None:
                return f"{sub}/{entry.name}"
    return None


def collision_error(conflicts: list[str], start: int, end: int, limit: int = 20) -> str:
    shown = conflicts[:limit]
    more = "" if len(conflicts) <= limit else f"\n  ... (+{len(conflicts) - limit} more)"
    return (
        f"--out collision for requested rows [{start}, {end}): "
        f"{len(conflicts)} existing output(s)\n  " + "\n  ".join(shown) + more +
        "\nRefusing before prep/scoring. Nothing was deleted or overwritten.\n"
        "Use a fresh --out, or a --start/--end window fully disjoint from the "
        "rows already collected there.")


def ensure_collect_meta(out_dir: Path, current: dict, identity_fields,
                        fingerprint_fields=()) -> bool:
    """The collectors' shared output-root identity step.

    First shard into an --out: write collect_meta.json (identity + provenance
    + content fingerprints) and return True. Later shard: require the stored
    `identity_fields` to equal the current invocation's exactly
    (check_collect_identity), then verify the CONTENT tiers — the whole-
    dataset hash (dataset/subset/split content; the shard's --start/--end
    window does not affect it) and each `fingerprint_fields` model-file
    fingerprint — via the three-tier checks (mismatch refuses; a legacy meta
    that predates a field warns and proceeds). Returns False; the file is
    never rewritten, so shard 1's record stays authoritative. Any refusal is
    a SystemExit naming what differs."""
    meta_path = Path(out_dir) / "collect_meta.json"
    if not meta_path.exists():
        record = dict(current)
        record["created"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        return True
    try:
        stored = json.loads(meta_path.read_text())
        if not isinstance(stored, dict):
            raise ValueError(f"expected a JSON object, got {type(stored).__name__}")
    except (OSError, ValueError) as e:
        sys.exit(f"{meta_path}: unreadable ({e}); cannot extend this --out — "
                 "use a fresh one")
    mismatched = check_collect_identity(stored, current, identity_fields)
    if mismatched:
        sys.exit("\n".join([
            f"{meta_path}: this --out was collected with a different identity; "
            "refusing to add a shard to it:",
            *mismatched,
            "Use a different --out (nothing was deleted).",
        ]))
    warnings = [check_dataset_content_hash(stored, current)]
    warnings += check_model_fingerprints(stored, current, fingerprint_fields)
    for warning in warnings:
        if warning:
            print(warning, file=sys.stderr)
    return False


def check_collect_identity(stored: dict, current: dict, fields) -> list[str]:
    """Cheap, exact identity check for extending an existing --out with a
    disjoint shard: every field in `fields` must be present in the stored
    collect_meta.json and equal the current value (no legacy defaults — a
    dir whose meta lacks a field cannot be extended). Fingerprints are not
    this function's job — ensure_collect_meta re-derives and verifies them
    via the three-tier guards. Returns mismatch lines (empty = OK)."""
    lines: list[str] = []
    for f in fields:
        if f not in stored:
            lines.append(f"  {f}: missing from the existing collect_meta.json")
        elif stored[f] != current.get(f):
            lines.append(f"  {f}\n    stored:  {stored[f]}\n    current: {current.get(f)}")
    return lines


def scorer_vlmk_version(scorer_path) -> int:
    """Ask the scorer binary which VLMK version it writes (`--vlmk-version`
    prints the integer and exits 0). Raises RuntimeError for a binary that
    does not understand the flag (pre-v2 build) or prints garbage."""
    try:
        proc = subprocess.run(
            [str(scorer_path), "--vlmk-version"],
            capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"{scorer_path}: cannot run --vlmk-version: {e}") from e
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out.isdigit():
        raise RuntimeError(
            f"{scorer_path}: does not report a VLMK version (exit "
            f"{proc.returncode}, stdout={out!r}); it predates the "
            "--vlmk-version flag, i.e. it is an outdated build")
    return int(out)


def preflight_scorer_vlmk_version(scorer_path) -> int:
    """Run-wide gate, called BEFORE the dataset load and before any row is
    touched: the scorer on disk must write exactly VLMK_VERSION. Without it
    a stale binary would only be detected per row in postprocess, AFTER the
    whole --manifest sweep ran (every output then rejected, kept on disk as
    FAIL evidence, and the whole sweep wasted).
    Returns the version on success; raises SystemExit with the rebuild hint
    otherwise."""
    try:
        version = scorer_vlmk_version(scorer_path)
    except RuntimeError as e:
        sys.exit(f"{e}\nRebuild it: cmake --build build --target "
                 "llama-vlm-kld llama-llm-kld")
    if version != VLMK_VERSION:
        sys.exit(
            f"{scorer_path} writes VLMK v{version} but this checkout expects "
            f"v{VLMK_VERSION}; rebuild it: cmake --build build --target "
            "llama-vlm-kld llama-llm-kld")
    return version


def effective_eval_tokens(n_answer: int, num_eval_tokens: int) -> int:
    """Answer positions the scorer will evaluate for one row."""
    n_answer = int(n_answer)
    return n_answer if num_eval_tokens == -1 else min(int(num_eval_tokens), n_answer)


def max_total_tokens_skip_info(
    row: Mapping[str, Any],
    num_eval_tokens: int,
    max_total_tokens: int | None,
) -> dict[str, int | str] | None:
    """Return skip details when a GT row exceeds --max-total-tokens.

    The row need is computed only from dataset columns, matching the HF logits
    runtime: n_prefill_tokens + effective_eval_tokens(generated_tokens_len, K).
    """
    if max_total_tokens is None:
        return None
    max_total_tokens = int(max_total_tokens)
    if max_total_tokens < 1:
        raise ValueError("--max-total-tokens must be a positive integer when set")
    n_prefill = int(row["n_prefill_tokens"])
    n_answer = int(row["generated_tokens_len"])
    n_eval = effective_eval_tokens(n_answer, int(num_eval_tokens))
    needed = n_prefill + n_eval
    if needed <= max_total_tokens:
        return None
    return {
        "n_prefill": n_prefill,
        "n_answer": n_answer,
        "n_eval": n_eval,
        "needed": needed,
        "max_total_tokens": max_total_tokens,
        "error": f"evaluated length {needed} exceeds --max-total-tokens={max_total_tokens}",
    }


def compute_n_ctx_requirement(pending, num_eval_tokens: int, n_seq_max: int = 1):
    """Return the row that determines the required scorer context, or None.

    For the wave scorer, n_seq_max sequences share one KV budget, so the
    conservative preflight requirement is max(row_context) * n_seq_max.
    """
    n_seq_max = int(n_seq_max)
    worst = None
    for row in pending:
        n_prefill = int(row["n_prefill"])
        n_answer = int(row["n_answer"])
        n_eval = effective_eval_tokens(n_answer, num_eval_tokens)
        row_need = n_prefill + n_eval
        need = row_need * n_seq_max
        current = {
            "idx": row.get("idx"),
            "item_id": str(row.get("item_id")),
            "n_prefill": n_prefill,
            "n_answer": n_answer,
            "n_eval": n_eval,
            "row_need": row_need,
            "n_seq_max": n_seq_max,
            "need": need,
        }
        if worst is None or current["need"] > worst["need"]:
            worst = current
    return worst


def format_n_ctx_preflight_error(requirement: dict, n_ctx: int) -> str:
    lines = [
        f"n_ctx preflight failed: current --n-ctx {n_ctx} < required {requirement['need']}",
        f"offending row: row_idx={requirement['idx']} item_id={requirement['item_id']}",
        "token need: "
        f"n_prefill={requirement['n_prefill']} + n_eval={requirement['n_eval']} "
        f"(n_answer={requirement['n_answer']}) = {requirement['row_need']}",
    ]
    if requirement["n_seq_max"] != 1:
        lines.append(
            f"wave scorer: row need {requirement['row_need']} * "
            f"n_seq_max={requirement['n_seq_max']} = {requirement['need']}"
        )
    lines.append(f"suggested --n-ctx {requirement['need']}")
    return "\n".join(lines)


def preflight_n_ctx(pending, n_ctx: int, num_eval_tokens: int, n_seq_max: int = 1):
    requirement = compute_n_ctx_requirement(pending, num_eval_tokens, n_seq_max)
    if requirement is not None and requirement["need"] > int(n_ctx):
        sys.exit(format_n_ctx_preflight_error(requirement, int(n_ctx)))
