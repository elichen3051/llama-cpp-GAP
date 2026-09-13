"""LEGACY_ATTEMPT_RECORDS=1 accepts collections shipped without .attempts/; the data-only checks stay strict."""
import csv
import json

import pytest

from lib import collection_state
from lib.collection_state import require_completed_attempts


def make_collection(tmp_path, statuses, npz_for=None):
    rows = [{"row_idx": i, "item_id": f"item{i}", "n_prefill": 3, "n_answer": 2, "n_eval": 2, "vocab": 10,
             "metrics_bytes": 1, "wall_s": 0.1, "status": s} for i, s in enumerate(statuses)]
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (tmp_path / "metrics").mkdir()
    for i, s in enumerate(statuses):
        if (npz_for is None and s == "OK") or (npz_for is not None and i in npz_for):
            (tmp_path / "metrics" / f"{i:03d}_item{i}.npz").write_bytes(b"")
    return tmp_path


@pytest.fixture(autouse=True)
def no_flag(monkeypatch):
    monkeypatch.delenv(collection_state.LEGACY_ATTEMPTS_ENV, raising=False)
    collection_state._legacy_attempts_notice_shown = False


def test_missing_attempts_rejected_without_flag(tmp_path):
    make_collection(tmp_path, ["OK", "OK"])
    with pytest.raises(ValueError, match="attempt records missing"):
        require_completed_attempts(tmp_path)


def test_missing_attempts_accepted_with_flag(tmp_path, monkeypatch, capsys):
    make_collection(tmp_path, ["OK", "SKIP_OVER_BUDGET", "OK"])
    monkeypatch.setenv(collection_state.LEGACY_ATTEMPTS_ENV, "1")
    require_completed_attempts(tmp_path)
    assert collection_state.LEGACY_ATTEMPTS_ENV in capsys.readouterr().err


@pytest.mark.parametrize("corrupt", ["missing_npz", "extra_npz", "bad_status", "flag_value"])
def test_flag_keeps_data_checks(tmp_path, monkeypatch, corrupt):
    if corrupt == "missing_npz":
        make_collection(tmp_path, ["OK", "OK"], npz_for={0})
    elif corrupt == "extra_npz":
        make_collection(tmp_path, ["OK", "SKIP_OVER_BUDGET"], npz_for={0, 1})
    elif corrupt == "bad_status":
        make_collection(tmp_path, ["OK", "RUNNING"])
    else:
        make_collection(tmp_path, ["OK"])
    monkeypatch.setenv(collection_state.LEGACY_ATTEMPTS_ENV, "true" if corrupt == "flag_value" else "1")
    with pytest.raises(ValueError):
        require_completed_attempts(tmp_path)


def test_present_attempts_still_checked_strictly(tmp_path, monkeypatch):
    make_collection(tmp_path, ["OK"])
    att = tmp_path / ".attempts" / "1-abc"; att.mkdir(parents=True)
    (att / "request.json").write_text(json.dumps({"scheme": collection_state.SCHEME, "start": 0, "end": None,
                                                  "rows": [{"row_idx": 0, "item_id": "item0"}]}))
    (att / "state.json").write_text(json.dumps({"scheme": collection_state.SCHEME, "state": "running", "error": None}))
    (att / "statuses.jsonl").write_text("")
    monkeypatch.setenv(collection_state.LEGACY_ATTEMPTS_ENV, "1")
    with pytest.raises(ValueError, match="attempt is running"):
        require_completed_attempts(tmp_path)
