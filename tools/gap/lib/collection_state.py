"""Durable declared work and terminal statuses for metric collection."""

import csv
import hashlib
import errno
import json
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

SCHEME = "company-collection-attempt-v1"

# Published archives ship collections without their .attempts/ records (collector scheduling state). With
# LEGACY_ATTEMPT_RECORDS=1 a collection that has NO .attempts/ directory is accepted on the strength of its
# data alone: every manifest row must carry a terminal status and metrics/*.npz must be exactly the OK rows.
# A collection that does have .attempts/ is still checked strictly.
LEGACY_ATTEMPTS_ENV = "LEGACY_ATTEMPT_RECORDS"
_legacy_attempts_notice_shown = False


def legacy_attempt_records_accepted():
    return os.environ.get(LEGACY_ATTEMPTS_ENV) == "1"


def require_manifest_matches_metrics(root):
    """Data-only completeness check: manifest statuses are terminal and metrics/*.npz are exactly the OK rows."""
    root = Path(root)
    try:
        with (root / "manifest.csv").open(newline="") as stream:
            manifest = {int(r["row_idx"]): r for r in csv.DictReader(stream)}
        if not manifest:
            raise ValueError("manifest has no rows")
        for idx, row in manifest.items():
            status = row["status"]
            if status not in ("OK", "SKIP_OVER_BUDGET") and not status.startswith("FAIL_"):
                raise ValueError(f"unknown terminal status {status!r} for row {idx}")
        expected_npz = {f"{idx:03d}_{row['item_id']}.npz" for idx, row in manifest.items() if row["status"] == "OK"}
        actual_npz = {p.name for p in (root / "metrics").glob("*.npz")}
        if actual_npz != expected_npz:
            raise ValueError("metric artifacts differ from completed OK rows")
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{root}: incomplete collection: {error}") from error


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fsync_directory(path.parent)


class CollectionAttempt:
    def __init__(self, root, start, end):
        self.root = Path(root)
        fsync_directory(self.root.parent)
        parent = self.root / ".attempts"
        parent.mkdir(exist_ok=True)
        fsync_directory(self.root)
        self.path = parent / f"{time.time_ns()}-{uuid.uuid4().hex}"
        self.path.mkdir()
        fsync_directory(parent)
        self.rows = None
        self.statuses = {}
        atomic_json(self.path / "request.json", {
            "scheme": SCHEME, "start": start, "end": end, "rows": None,
            "pid": os.getpid(), "created_ns": time.time_ns(),
        })
        self.state("running")

    def state(self, state, error=None):
        atomic_json(self.path / "state.json", {
            "scheme": SCHEME, "state": state, "updated_ns": time.time_ns(),
            "error": error,
        })

    def declare(self, rows):
        self.rows = [{"row_idx": int(i), "item_id": str(item)} for i, item in rows]
        request = json.loads((self.path / "request.json").read_text())
        request["rows"] = self.rows
        atomic_json(self.path / "request.json", request)

    def record(self, row):
        value = {key: row[key] for key in ("row_idx", "item_id", "status")}
        value["row_idx"] = int(value["row_idx"])
        with (self.path / "statuses.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(self.path)
        self.statuses[value["row_idx"]] = value

    def reference(self, idx, item_id, metadata, row_settings):
        directory = self.path / "generators"
        directory.mkdir(exist_ok=True)
        encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        destination = directory / f"{digest}.json"
        if not destination.exists():
            atomic_json(destination, metadata)
        with (self.path / "references.jsonl").open("a") as stream:
            stream.write(json.dumps({"row_idx": idx, "item_id": item_id,
                                     "generator_sha256": digest, "row_settings": row_settings},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(self.path)

    def finish(self):
        if self.rows is None or set(self.statuses) != {r["row_idx"] for r in self.rows}:
            raise ValueError("collection attempt has rows without terminal status")
        fsync_directory(self.root)
        self.state("failed" if any(r["status"].startswith("FAIL_") for r in self.statuses.values()) else "completed")

    def stop(self, error):
        if self.rows is None and isinstance(error, (Exception, SystemExit)):
            self.state("aborted", f"{type(error).__name__}: {error}")
        elif isinstance(error, SystemExit) and len(self.statuses) == len(self.rows):
            self.finish()
        else:
            self.state("interrupted", f"{type(error).__name__}: {error}")


def require_completed_attempts(root):
    global _legacy_attempts_notice_shown
    root = Path(root)
    parent = root / ".attempts"
    if not parent.is_dir() or not any(parent.iterdir()):
        if legacy_attempt_records_accepted():
            if not _legacy_attempts_notice_shown:
                import sys
                print(f"[collection_state] {LEGACY_ATTEMPTS_ENV}=1: accepting collections without .attempts/ records "
                      "on the strength of manifest.csv and metrics/*.npz", file=sys.stderr)
                _legacy_attempts_notice_shown = True
            require_manifest_matches_metrics(root)
            return
        raise ValueError(f"{root}: collection attempt records missing; re-collect legacy data")
    requested, terminal = {}, {}
    for path in sorted(parent.iterdir()):
        try:
            request = json.loads((path / "request.json").read_text())
            state = json.loads((path / "state.json").read_text())
            if request.get("scheme") != SCHEME or state.get("scheme") != SCHEME:
                raise ValueError("unknown attempt scheme")
            if state["state"] == "aborted" and request["rows"] is None:
                continue
            if state["state"] not in ("completed", "failed"):
                raise ValueError(f"attempt is {state['state']}")
            rows = request["rows"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("missing declared rows")
            expected = {int(row["row_idx"]): str(row["item_id"]) for row in rows}
            if len(expected) != len(rows):
                raise ValueError("duplicate declared row")
            statuses = {}
            for line in (path / "statuses.jsonl").read_text().splitlines():
                row = json.loads(line)
                idx = int(row["row_idx"])
                if idx in statuses or expected.get(idx) != row["item_id"]:
                    raise ValueError("terminal row does not match declared work")
                status = row["status"]
                if status not in ("OK", "SKIP_OVER_BUDGET") and not status.startswith("FAIL_"):
                    raise ValueError("unknown terminal status")
                statuses[idx] = row
            if set(statuses) != set(expected):
                raise ValueError("declared rows have no terminal status")
            for idx, item in expected.items():
                if idx in requested and requested[idx] != item:
                    raise ValueError("row identity changed between attempts")
            requested.update(expected)
            terminal.update(statuses)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: incomplete collection: {error}") from error
    if not requested:
        raise ValueError(f"{root}: no completed declared work")
    try:
        with (root / "manifest.csv").open(newline="") as stream:
            manifest = {int(r["row_idx"]): r for r in csv.DictReader(stream)}
        if set(manifest) != set(requested):
            raise ValueError("manifest row set differs from declared work")
        for idx, item in requested.items():
            if manifest[idx]["item_id"] != item or manifest[idx]["status"] != terminal[idx]["status"]:
                raise ValueError(f"manifest disagrees with attempt status for row {idx}")
        expected_npz = {f"{idx:03d}_{item}.npz" for idx, item in requested.items() if terminal[idx]["status"] == "OK"}
        actual_npz = {p.name for p in (root / "metrics").glob("*.npz")}
        if actual_npz != expected_npz:
            raise ValueError("metric artifacts differ from completed OK rows")
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{root}: incomplete collection: {error}") from error


@contextmanager
def comparison_locks(roots):
    import fcntl
    with ExitStack() as stack:
        for root in sorted({Path(p).resolve() for p in roots}):
            if not root.is_dir():
                raise ValueError(f"{root}: not a directory")
            try:
                stream = stack.enter_context((root / ".collect.lock").open("a+"))
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EAGAIN, errno.EACCES):
                    raise ValueError(f"{root}: collector is still active") from error
                raise ValueError(f"{root}: cannot lock collection: {error}") from error
        yield


def refuse_unfinished_attempts(root):
    for path in sorted((Path(root) / ".attempts").glob("*")):
        try:
            state = json.loads((path / "state.json").read_text())["state"]
            if state not in ("completed", "failed", "aborted"):
                raise ValueError(f"previous attempt is {state}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: unfinished collection; use a fresh output directory: {error}") from error
