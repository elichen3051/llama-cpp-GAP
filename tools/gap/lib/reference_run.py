"""Supervise native reference attempts without losing completed row outcomes."""

import json
import os
import time
import subprocess
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_records(path, allow_partial=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for index, raw in enumerate(lines):
        try:
            records.append(json.loads(raw))
        except (ValueError, UnicodeError):
            if allow_partial and index == len(lines) - 1 and not raw.endswith(b"\n"):
                break
            raise ValueError(f"corrupt native journal: {path}, line {index + 1}")
    return records


def run_native(binary, native_args, requests, out, timeout=None, row_retries=1, startup_retries=1, row_timeout=1800):
    out = Path(out)
    expected = {r["id"]: r for r in requests}
    if len(expected) != len(requests):
        raise ValueError("duplicate request IDs")
    outcomes, failures, tries = {}, {}, {}
    pending = list(requests)
    attempts, startup_failures = [], 0
    metadata = None
    while pending:
        number = len(attempts) + 1
        directory = out / "attempts" / f"{number:04d}"
        directory.mkdir(parents=True)
        native = directory / "native"
        request_path = directory / "requests.jsonl"
        request_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pending))
        command = [str(binary), "--requests", str(request_path), "--out-dir", str(native),
                   "--continue-on-error", *native_args]
        atomic_json(directory / "command.json", command)
        print(f"Native attempt {number}: {len(pending)} pending; log: {directory / 'native.log'}", flush=True)
        timed_out = False
        with (directory / "native.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                started = last_progress = time.monotonic()
                event_size = event_count = 0
                active_since = {}
                while True:
                    try:
                        code = proc.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        journal = native / "events.jsonl"
                        size = journal.stat().st_size if journal.exists() else 0
                        now = time.monotonic()
                        if size != event_size:
                            observed = read_records(journal, allow_partial=True)
                            for event in observed[event_count:]:
                                if event.get("event") == "row_started":
                                    active_since[event["id"]] = now
                                elif event.get("event") in ("row_succeeded", "row_failed"):
                                    active_since.pop(event["id"], None)
                            last_progress, event_size, event_count = now, size, len(observed)
                        oldest = min(active_since.values(), default=last_progress)
                        if ((timeout is not None and now - started > timeout)
                                or (row_timeout is not None and now - oldest > row_timeout)):
                            timed_out = True
                            proc.kill()
                            code = proc.wait()
                            break
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        record = {"attempt": number, "exit_code": code, "timed_out": timed_out,
                  "requested_ids": [r["id"] for r in pending]}
        attempts.append(record)
        atomic_json(directory / "exit.json", record)
        allowed = set(record["requested_ids"])
        results = read_records(native / "generations.jsonl", allow_partial=code != 0)
        events = read_records(native / "events.jsonl", allow_partial=code != 0)
        try:
            current = json.loads((native / "metadata.json").read_text())
        except (OSError, ValueError):
            if results or events:
                raise ValueError("native row work has missing/corrupt metadata")
            current = None
        if current is not None:
            identity = {k: v for k, v in current.items() if k != "command"}
            if metadata is not None and identity != {k: v for k, v in metadata.items() if k != "command"}:
                raise ValueError("effective native settings changed between attempts")
            metadata = current
        slots = (current or {}).get("total_slots", 1)
        if type(slots) is not int or slots <= 0:
            raise ValueError("invalid native slot count")
        active = set()
        seen = set()
        for result in results:
            row_id = result.get("id")
            if row_id not in allowed or row_id in seen or row_id in outcomes:
                raise ValueError(f"unexpected/duplicate native result ID: {row_id!r}")
            outcomes[row_id] = result
            seen.add(row_id)
        starts, errors, terminals = set(), {}, set()
        for event in events:
            row_id = event.get("id")
            if row_id not in allowed:
                raise ValueError(f"unexpected native journal ID: {row_id!r}")
            if event["event"] == "row_started":
                if row_id in starts:
                    raise ValueError(f"duplicate native row start: {row_id}")
                starts.add(row_id)
                active.add(row_id)
                if len(active) > slots:
                    raise ValueError("native journal exceeds declared parallel slots")
                tries[row_id] = tries.get(row_id, 0) + 1
            elif event["event"] in ("row_failed", "row_succeeded"):
                if row_id not in starts or row_id in terminals:
                    raise ValueError(f"native terminal event without one unique start: {row_id}")
                terminals.add(row_id)
                active.remove(row_id)
                if event["event"] == "row_failed":
                    if row_id in seen:
                        raise ValueError(f"native row has both a result and failure: {row_id}")
                    errors[row_id] = event
                elif row_id not in seen:
                    raise ValueError(f"native success has no durable result: {row_id}")
            else:
                raise ValueError(f"unknown native journal event: {event['event']}")
        if not seen <= starts:
            raise ValueError("native result without a row start")
        inflight = starts - seen - set(errors)
        if code == 0 and starts != terminals:
            raise ValueError("native process succeeded with unfinished journal rows")
        for row_id in inflight:
            errors[row_id] = {"id": row_id, "retryable": True,
                              "error": "native timeout" if timed_out else f"native process exit {code}"}
        if code == 0 and any(r["id"] not in seen and r["id"] not in errors for r in pending):
            raise ValueError("native process succeeded without a terminal outcome for every request")
        for row_id, error in errors.items():
            if row_id in outcomes:
                continue
            if not error.get("retryable", False) or tries.get(row_id, 0) > row_retries:
                failures[row_id] = {"id": row_id, "status": "failed", "error": error["error"],
                                    "attempts": tries.get(row_id, 0), "last_native_attempt": number}
        remaining = [r for r in pending if r["id"] not in outcomes and r["id"] not in failures]
        if code != 0 and not starts and not results and remaining:
            startup_failures += 1
            if startup_failures > startup_retries:
                for request in remaining:
                    row_id = request["id"]
                    failures[row_id] = {"id": row_id, "status": "unattempted_startup_failure",
                                        "error": f"native startup failed {startup_failures} times; see attempt logs",
                                        "attempts": 0, "last_native_attempt": number}
                remaining = []
        # Process untouched rows before retrying interrupted rows.
        pending = [r for r in remaining if r["id"] not in errors] + [r for r in remaining if r["id"] in errors]
        atomic_json(out / "progress.json", {"status": "running" if pending else "processed",
                    "requested": len(requests), "generated_ids": list(outcomes), "failures": list(failures.values()),
                    "pending_ids": [r["id"] for r in pending], "native_attempts": attempts})
    if set(outcomes) | set(failures) != set(expected) or set(outcomes) & set(failures):
        raise ValueError("native outcomes do not partition requested IDs")
    merged = out / "native"
    merged.mkdir()
    if metadata is not None:
        atomic_json(merged / "metadata.json", metadata)
    (merged / "generations.jsonl").write_text("".join(json.dumps(outcomes[r["id"]], ensure_ascii=False) + "\n" for r in requests if r["id"] in outcomes))
    atomic_json(out / "native-attempts.json", attempts)
    return outcomes, failures, metadata
