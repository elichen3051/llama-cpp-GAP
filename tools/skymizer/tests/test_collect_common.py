"""Tests for collect_common's manifest journal readers. Hermetic: stdlib only."""

import pytest

from lib.collect_common import (
    VisionBudgetReporter,
    acquire_out_lock,
    root_has_prior_output,
    manifest_row_statuses,
    check_collect_identity,
    check_manifest_header,
    collision_error,
    manifest_row_writer,
    scan_output_collisions,
)


def _write_manifest(tmp_path, lines):
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_duplicate_record_collapses_to_one_row(tmp_path):
    # The rescore-after-OK journal shape from finding 15: one dataset row,
    # two historical OK lines. Consumers must see ONE row.
    path = _write_manifest(tmp_path, [
        "row_idx,item_id,status",
        "0,item-0,OK",
        "0,item-0,OK",
    ])
    assert manifest_row_statuses(path) == {0: "OK"}


def test_error_then_ok_last_record_wins(tmp_path):
    path = _write_manifest(tmp_path, [
        "row_idx,item_id,status",
        "3,item-3,FAIL_RuntimeError",
        "3,item-3,OK",
    ])
    assert manifest_row_statuses(path)[3] == "OK"


def test_ok_then_fail_supersedes_the_stale_ok(tmp_path):
    # Behavior fix over the raw-journal read: the stale OK must not be
    # consumed once a later record superseded it.
    path = _write_manifest(tmp_path, [
        "row_idx,item_id,status",
        "5,item-5,OK",
        "5,item-5,FAIL_ValueError",
    ])
    assert manifest_row_statuses(path)[5] == "FAIL_ValueError"


def test_rows_keep_first_seen_order(tmp_path):
    # A rescored row stays at its first appearance, so healthy manifests keep
    # their journal (= collection) order.
    path = _write_manifest(tmp_path, [
        "row_idx,item_id,status",
        "2,item-2,OK",
        "0,item-0,OK",
        "1,item-1,OK",
        "2,item-2,OK",
    ])
    assert list(manifest_row_statuses(path)) == [2, 0, 1]


def test_missing_manifest_is_no_records(tmp_path):
    # The collision scanner treats a missing manifest as "no records".
    missing = tmp_path / "manifest.csv"
    assert manifest_row_statuses(missing) == {}


@pytest.mark.parametrize("extra_column", [False, True])
def test_manifest_reader_tolerates_retired_mid_row_column(tmp_path, extra_column):
    """A retired column left by an older writer sits MID-ROW (not at the
    end) — the reader must align by name regardless."""
    header = "row_idx,item_id,dtype,retired_col,status" if extra_column \
        else "row_idx,item_id,dtype,status"
    row = "0,item-0,fp32,-1,OK" if extra_column else "0,item-0,fp32,OK"
    path = _write_manifest(tmp_path, [header, row])
    assert manifest_row_statuses(path) == {0: "OK"}


# ---------------------------------------------------------------------------
# manifest_row_writer / check_manifest_header: exact current schema only
# ---------------------------------------------------------------------------

def _append_with_writer(path, columns, row):
    with open(path, "a", newline="") as mf:
        write_row = manifest_row_writer(mf, path, columns)
        write_row(row)


def test_manifest_row_writer_fresh_file_writes_header_and_row(tmp_path):
    path = tmp_path / "manifest.csv"
    cols = ("row_idx", "item_id", "status")
    _append_with_writer(path, cols, [0, "a", "OK"])
    assert path.read_text().splitlines() == ["row_idx,item_id,status", "0,a,OK"]
    assert manifest_row_statuses(path) == {0: "OK"}


def test_manifest_row_writer_treats_empty_file_as_fresh(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"")
    _append_with_writer(path, ("row_idx", "item_id", "status"), [0, "a", "OK"])
    assert path.read_text().splitlines() == ["row_idx,item_id,status", "0,a,OK"]


def test_manifest_row_writer_appends_under_exact_current_header(tmp_path):
    """Sequential disjoint shards append to ONE manifest: second writer finds
    the exact current header and appends without a second header line."""
    path = tmp_path / "manifest.csv"
    cols = ("row_idx", "item_id", "status")
    _append_with_writer(path, cols, [0, "a", "OK"])
    _append_with_writer(path, cols, [7, "b", "OK"])
    assert path.read_text().splitlines() == ["row_idx,item_id,status", "0,a,OK", "7,b,OK"]
    assert sorted(manifest_row_statuses(path)) == [0, 7]


@pytest.mark.parametrize("header", [
    "row_idx,item_id,retired_col,status",    # older layout with an extra column
    "row_idx,item_id",                        # missing a current column
    "item_id,row_idx,status",                 # same columns, different order
])
def test_manifest_header_must_equal_current_schema(tmp_path, header):
    """No older-layout tolerance: any header that is not exactly the current
    schema refuses the append, raising ValueError BEFORE any write and
    leaving the file byte-identical."""
    path = tmp_path / "manifest.csv"
    path.write_text(header + "\n0,a,OK\n")
    before = path.read_bytes()
    cols = ("row_idx", "item_id", "status")
    with pytest.raises(ValueError, match="!= current manifest schema"):
        check_manifest_header(path, cols)
    with pytest.raises(ValueError, match="!= current manifest schema"):
        _append_with_writer(path, cols, [1, "b", "OK"])
    assert path.read_bytes() == before
    assert check_manifest_header(tmp_path / "missing.csv", cols) == []


def test_manifest_torn_last_line_refuses_without_mutation(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("row_idx,item_id,status\n0,a,OK\n1")   # torn tail "1"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not newline-terminated"):
        _append_with_writer(path, ("row_idx", "item_id", "status"), [2, "b", "OK"])
    assert path.read_bytes() == before


def test_manifest_row_writer_dict_rows_pad_unlisted_columns(tmp_path):
    """FAIL/SKIP rows name only the columns they have; the writer blanks the
    rest (no hand-counted padding at call sites) and rejects unknown keys."""
    path = tmp_path / "manifest.csv"
    cols = ("row_idx", "item_id", "n_eval", "wall_s", "status")
    _append_with_writer(path, cols, {"row_idx": 3, "item_id": "c",
                                     "wall_s": "0.5", "status": "FAIL_X"})
    assert path.read_text().splitlines()[1] == "3,c,,0.5,FAIL_X"
    with pytest.raises(ValueError, match="unknown column"):
        _append_with_writer(path, cols, {"row_idx": 4, "bogus": 1})


def test_manifest_row_writer_rejects_wrong_arity(tmp_path):
    path = tmp_path / "manifest.csv"
    with pytest.raises(ValueError, match="expected 3"):
        _append_with_writer(path, ("row_idx", "item_id", "status"), [1, "b"])


# ---------------------------------------------------------------------------
# output-root policy helpers: collision scan, identity, lock
# ---------------------------------------------------------------------------

def test_manifest_row_statuses_every_status_counts_and_malformed_raises(tmp_path):
    path = _write_manifest(tmp_path, [
        "row_idx,item_id,status",
        "0,a,OK", "1,b,FAIL_X", "2,c,SKIP_OVER_BUDGET", "0,a,FAIL_Y",
    ])
    assert set(manifest_row_statuses(path)) == {0, 1, 2}
    assert manifest_row_statuses(path) == {0: "FAIL_Y", 1: "FAIL_X", 2: "SKIP_OVER_BUDGET"}
    (tmp_path / "bad").mkdir()
    bad = _write_manifest(tmp_path / "bad", ["row_idx,item_id,status", "0,a,OK", "x,a,OK"])
    with pytest.raises(ValueError, match=r"manifest.csv:3: row_idx 'x' is not an integer"):
        manifest_row_statuses(bad)


def _seed_out(tmp_path):
    out = tmp_path / "out"
    for sub in ("metrics", "_prep"):
        (out / sub).mkdir(parents=True)
    return out


def test_scan_output_collisions_enumerates_every_prior_output_kind(tmp_path):
    """Dumps, sidecars, .tmp leftovers, prep dirs and manifest records of any
    status all count; the row prefix is matched by index, whatever item_id
    the artifact carries (item-id drift cannot hide a collision)."""
    out = _seed_out(tmp_path)
    (out / "metrics" / "003_itemA.bin").write_bytes(b"x")
    (out / "metrics" / "004_other.npz.tmp").write_bytes(b"x")
    (out / "metrics" / "005_itemC.npz").write_bytes(b"x")
    (out / "metrics" / "006_itemD.bin.rejected").write_bytes(b"x")
    (out / "_prep" / "007_itemE").mkdir()
    (out / "metrics" / "1000_big.bin").write_bytes(b"x")
    (out / "metrics" / "\u00b2_unicode.bin").write_bytes(b"x")   # isdigit but not int()-able
    (out / "metrics" / "notes.txt").write_bytes(b"x")          # not a row artifact
    manifest = _write_manifest(out, ["row_idx,item_id,status", "8,h,FAIL_ValueError",
                                     "9,i,SKIP_OVER_BUDGET", "10,j,OK"])
    hits = scan_output_collisions(out, range(0, 1001), manifest)
    rows = sorted(int(h.split()[1].rstrip(":")) for h in hits)
    # row 8 (FAIL_* with NO artifact) is retryable, not a collision
    assert rows == [3, 4, 5, 6, 7, 9, 10, 1000]
    assert not any(h.startswith("row 8:") for h in hits)
    # manifest collisions carry the row's (last) status
    assert "row 9: manifest.csv status=SKIP_OVER_BUDGET" in hits
    assert "row 10: manifest.csv status=OK" in hits
    assert any("004_other.npz.tmp" in h for h in hits)
    assert any("_prep/007_itemE" in h for h in hits)
    # a disjoint window is clean; prefix "100_" must not match "1000_big"
    assert scan_output_collisions(out, range(11, 200), manifest) == []
    assert scan_output_collisions(out, range(100, 101), manifest) == []
    assert scan_output_collisions(out, [], manifest) == []


def test_collision_error_lists_conflicts_and_promises_no_deletion():
    msg = collision_error([f"row {i}: metrics/{i:03d}_x.bin" for i in range(25)], 0, 30)
    assert "rows [0, 30)" in msg and "25 existing output(s)" in msg
    assert "(+5 more)" in msg
    assert "Nothing was deleted or overwritten" in msg
    assert "fresh --out" in msg and "disjoint" in msg


def test_check_collect_identity_exact_no_legacy_defaults():
    fields = ("model", "tf_chunk", "media_wrapper")
    stored = {"model": "/m/a.gguf", "tf_chunk": -1, "media_wrapper": "stripped",
              "created": "x", "n_ctx": 1}
    current = {"model": "/m/a.gguf", "tf_chunk": -1, "media_wrapper": "stripped",
               "n_ctx": 99}                     # non-identity fields ignored
    assert check_collect_identity(stored, current, fields) == []
    lines = check_collect_identity(dict(stored, tf_chunk=1), current, fields)
    assert len(lines) == 1 and "tf_chunk" in lines[0] and "stored:  1" in lines[0]
    # a dir whose meta lacks a field cannot be extended (no default applied)
    legacy = {k: v for k, v in stored.items() if k != "media_wrapper"}
    lines = check_collect_identity(legacy, current, fields)
    assert lines == ["  media_wrapper: missing from the existing collect_meta.json"]


def test_acquire_out_lock_is_exclusive_per_root(tmp_path):
    out = tmp_path / "out"
    first = acquire_out_lock(out)
    with pytest.raises(RuntimeError, match="another collector holds"):
        acquire_out_lock(out)
    first.close()                       # releasing lets the next collector in
    second = acquire_out_lock(out)
    second.close()
    assert (out / ".collect.lock").exists()


def test_acquire_out_lock_reports_non_contention_errors_faithfully(tmp_path, monkeypatch):
    """A filesystem without advisory locks fails with ENOLCK/EOPNOTSUPP, not
    EWOULDBLOCK: the message must name the real errno instead of blaming a
    concurrent collector that does not exist."""
    import errno as errno_mod
    import fcntl
    out = tmp_path / "out"

    def no_locks(fh, flags):
        raise OSError(errno_mod.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(RuntimeError, match="cannot lock .*No locks available"):
        acquire_out_lock(out)

    def contended(fh, flags):
        raise OSError(errno_mod.EWOULDBLOCK, "Resource temporarily unavailable")
    monkeypatch.setattr(fcntl, "flock", contended)
    with pytest.raises(RuntimeError, match="another collector holds"):
        acquire_out_lock(out)


def test_root_has_prior_output_ignores_non_row_files(tmp_path):
    """A stray notes file / editor swapfile / NFS silly-rename must not brick
    a genuinely fresh root — the same row-stem filter the collision scan uses
    applies here (they can never disagree)."""
    out = tmp_path / "out"
    (out / "metrics").mkdir(parents=True)
    manifest = out / "manifest.csv"
    for name in ("notes.txt", ".DS_Store", ".nfs00000001", "README"):
        (out / "metrics" / name).write_bytes(b"x")
    assert scan_output_collisions(out, range(0, 10), manifest) == []
    assert root_has_prior_output(out, manifest) is None
    (out / "metrics" / "003_item.bin").write_bytes(b"x")     # a real row artifact
    assert root_has_prior_output(out, manifest) == "metrics/003_item.bin"


def test_root_has_prior_output_detects_rows_without_meta(tmp_path):
    """A root holding any row record or artifact is not fresh — main() must
    refuse to stamp a new collect_meta.json over rows of unknown provenance."""
    out = tmp_path / "out"
    out.mkdir()
    manifest = out / "manifest.csv"
    assert root_has_prior_output(out, manifest) is None
    (out / "metrics").mkdir()
    assert root_has_prior_output(out, manifest) is None          # empty subdir is fine
    (out / "metrics" / "004_x.npz").write_bytes(b"x")
    assert root_has_prior_output(out, manifest) == "metrics/004_x.npz"
    (out / "metrics" / "004_x.npz").unlink()
    _write_manifest(out, ["row_idx,item_id,status", "7,q,OK"])
    assert "row 7" in root_has_prior_output(out, manifest)


def test_row_stem_index_is_decimal_only_and_padding_agnostic():
    """isdecimal, not isdigit ('²'.isdigit() is True but int() raises), and any
    padding maps to its row so a non-canonically named leftover cannot slip
    past the scan and get overwritten by the scorer."""
    from lib.collect_common import row_stem_index
    assert row_stem_index("003_item.bin") == 3
    assert row_stem_index("0001_item.bin") == 1        # padding-agnostic
    assert row_stem_index("1000_item.bin") == 1000
    for name in ("\u00b2_item.bin", "notes.txt", "_leading.bin", "abc_1.bin", ".DS_Store"):
        assert row_stem_index(name) is None, name


def test_scan_output_collisions_catches_non_canonical_padding(tmp_path):
    out = tmp_path / "out"
    (out / "metrics").mkdir(parents=True)
    (out / "metrics" / "0001_item.bin").write_bytes(b"x")    # leftover for row 1
    manifest = out / "manifest.csv"
    hits = scan_output_collisions(out, range(0, 3), manifest)
    assert hits == ["row 1: metrics/0001_item.bin"]


@pytest.mark.parametrize("raw,expect_ok", [
    (b"\xef\xbb\xbfrow_idx,item_id,status\n0,a,OK\n", True),      # UTF-8 BOM
    (b"row_idx,item_id,status\r0,a,OK\r", True),                    # CR-only
    (b"row_idx,item_id,status\r\n0,a,OK\r\n", True),               # CRLF
])
def test_check_manifest_header_tolerates_bom_and_line_endings(tmp_path, raw, expect_ok):
    """A BOM must not produce a refusal whose two headers look identical, and a
    CR-only file must not escape as csv.Error (which main() does not catch)."""
    path = tmp_path / "manifest.csv"
    path.write_bytes(raw)
    cols = ("row_idx", "item_id", "status")
    assert check_manifest_header(path, cols) == list(cols)


def test_vision_budget_reporter_flags_the_dataset_mismatch_once(capsys):
    """prep derives the mtmd-equivalent vision bounds from the HF processor
    config the ground-truth answers were generated with, writes them into
    every row's meta.json — and nothing read them. The collectors forward the
    global --image-*-tokens flags instead, whose -1 default lets mtmd take its
    bounds from the mmproj GGUF. So the ground truth came from the HF config,
    llama.cpp's budget came from the GGUF, and nothing compared them."""
    r = VisionBudgetReporter(-1, -1)
    meta = {"image_token_limits": {"image_min_tokens": 4,
                                   "image_max_tokens": 1024}}
    r.check(meta)
    r.check(meta)                      # same limits again: still ONE warning
    r.check(dict(meta))                # a fresh dict, same values
    assert len(r.mismatches) == 1
    msg = r.mismatches[0]
    assert "min=4 max=1024" in msg
    assert "--image-min-tokens -1" in msg
    assert "mtmd's own default" in msg
    assert msg in capsys.readouterr().err


def test_vision_budget_reporter_is_silent_when_the_budgets_agree(capsys):
    r = VisionBudgetReporter(4, 1024)
    r.check({"image_token_limits": {"image_min_tokens": 4,
                                    "image_max_tokens": 1024}})
    assert r.mismatches == [] and capsys.readouterr().err == ""


def test_vision_budget_reporter_ignores_rows_without_derived_limits(capsys):
    """A dataset whose rows carry no image_processor_config (LLM prep, or a
    pre-derivation dataset) must not produce noise."""
    r = VisionBudgetReporter(-1, -1)
    r.check({})
    r.check({"image_token_limits": None})
    r.check(None)
    assert r.mismatches == [] and capsys.readouterr().err == ""


def test_vision_budget_reporter_warns_once_per_distinct_limit_pair(capsys):
    """A mixed dataset (rows generated under different processor configs) is
    itself worth seeing, so each distinct pair reports once."""
    r = VisionBudgetReporter(-1, -1)
    r.check({"image_token_limits": {"image_min_tokens": 4, "image_max_tokens": 1024}})
    r.check({"image_token_limits": {"image_min_tokens": 4, "image_max_tokens": 2048}})
    r.check({"image_token_limits": {"image_min_tokens": 4, "image_max_tokens": 1024}})
    assert len(r.mismatches) == 2


def test_manifest_readers_tolerate_a_bom(tmp_path):
    """check_manifest_header already accepts a BOM (a manifest re-saved as
    "CSV UTF-8"); the DictReader-based status reader must agree with it.
    Under plain utf-8 the BOM stays glued to the first fieldname, so row_idx
    reads as missing: manifest_row_statuses raised "damaged manifest — use a
    fresh --out" on a perfectly good file."""
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"\xef\xbb\xbfrow_idx,item_id,status\n"
                     b"0,item-0,OK\n1,item-1,FAIL_PREP\n")
    assert manifest_row_statuses(path) == {0: "OK", 1: "FAIL_PREP"}
    assert set(manifest_row_statuses(path)) == {0, 1}


def test_scan_output_collisions_fail_rows_retry_only_without_artifacts(tmp_path):
    """The minimal relaxation: a FAIL_* row with nothing on disk (transient
    prep failure) may be re-collected; a FAIL row that left ANY file — a
    rejected KLD dump kept as evidence included — still collides, and OK /
    SKIP_OVER_BUDGET records always collide."""
    out = _seed_out(tmp_path)
    manifest = _write_manifest(out, [
        "row_idx,item_id,status",
        "1,a,FAIL_ValueError",           # transient: no artifact -> retryable
        "2,b,FAIL_ValueError",           # left evidence -> collision
        "3,c,OK", "4,d,SKIP_OVER_BUDGET",
        "5,e,FAIL_OSError", "5,e,OK",    # resume-era journal: last status OK
    ])
    (out / "metrics" / "002_b.bin.rejected").write_bytes(b"evidence")
    hits = scan_output_collisions(out, range(0, 10), manifest)
    assert not any(h.startswith("row 1:") for h in hits)
    # exactly FAIL_<...> is retryable; foreign status strings fail closed
    (tmp_path / "f").mkdir()
    foreign = _write_manifest(tmp_path / "f", ["row_idx,item_id,status",
        "1,a,FAILED", "2,b,FAIL", "3,c,FAILURE", "4,d,fail_x", "5,e, FAIL_X ", "6,f,"])
    hits2 = scan_output_collisions(tmp_path / "f", range(0, 10), foreign)
    rows2 = sorted(int(h.split()[1].rstrip(":")) for h in hits2)
    assert rows2 == [1, 2, 3, 4, 6]        # only row 5 (stripped "FAIL_X") retries
    assert "row 2: metrics/002_b.bin.rejected" in hits
    assert "row 2: manifest.csv status=FAIL_ValueError" in hits
    assert "row 3: manifest.csv status=OK" in hits
    assert "row 4: manifest.csv status=SKIP_OVER_BUDGET" in hits
    assert "row 5: manifest.csv status=OK" in hits     # last-wins, not retryable
