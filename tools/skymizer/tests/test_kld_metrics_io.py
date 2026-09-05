"""Tests for kld_metrics_io (VLMK .bin parsing, completeness validation,
lossless .npz conversion).

Hermetic: synthesizes tiny VLMK/VLMS fixtures; no GPU, no models, no built
binary. Run from tools/skymizer:  python3 -m pytest tests/test_kld_metrics_io.py -v
"""



import numpy as np
import pytest

import lib.kld_metrics_io as kio


from fakes import make_records, write_vlmk  # noqa: E402,F401  (fixture writers live in fakes.py)


@pytest.fixture
def vlmk_bin(tmp_path):
    p = tmp_path / "row.bin"
    write_vlmk(p, make_records(npos=5), vocab=11, n_prefill=7)
    return p


# ---------------------------------------------------------------------------
# record layout contract
# ---------------------------------------------------------------------------

def test_record_layout_is_56_bytes_in_cpp_field_order():
    """KLD_RECORD_DT must mirror the packed kld_record struct in
    skymizer-vlmk-kernel.h (current = v3, 56 bytes with `ear` and the EAR_K
    family); the v1/v2 layouts stay pinned for the legacy readers."""
    assert kio.VLMK_VERSION == 3
    assert kio.KLD_RECORD_DT.itemsize == 56
    assert kio.KLD_METRIC_KEYS == (
        "kld", "reversed_kld", "js_kld", "nll_ref", "nll_cand",
        "entropy_ref", "entropy_cand", "ear", "ear_20", "ear_10", "ear_5",
        "target", "argmax_ref", "argmax_cand")
    assert kio.kld_record_dt(2).itemsize == 44
    assert kio.kld_metric_keys(2) == (
        "kld", "reversed_kld", "js_kld", "nll_ref", "nll_cand",
        "entropy_ref", "entropy_cand", "ear",
        "target", "argmax_ref", "argmax_cand")
    assert kio.kld_record_dt(1).itemsize == 40
    assert kio.kld_metric_keys(1) == (
        "kld", "reversed_kld", "js_kld", "nll_ref", "nll_cand",
        "entropy_ref", "entropy_cand", "target", "argmax_ref", "argmax_cand")
    assert kio.VERSIONED_METRIC_KEYS == ("ear", "ear_20", "ear_10", "ear_5")
    with pytest.raises(ValueError, match="version"):
        kio.kld_record_dt(4)


# ---------------------------------------------------------------------------
# header + load
# ---------------------------------------------------------------------------

def test_read_kld_header(vlmk_bin):
    assert kio.read_kld_header(vlmk_bin) == {
        "version": kio.VLMK_VERSION, "vocab": 11, "npos": 5, "n_prefill": 7,
        "n_past_actual": 0}


def test_header_records_the_actual_post_prefill_position(tmp_path):
    """n_prefill is the HF ground-truth sequential length echoed from the
    manifest; for a VLM it does NOT move when the vision-token budget does.
    n_past_actual is llama.cpp's own count, which does — the on-disk witness
    of the real conditioning. 0 = a dump written before the field existed."""
    p = tmp_path / "recorded.bin"
    write_vlmk(p, make_records(), n_prefill=7, n_past_actual=513)
    h = kio.read_kld_header(p)
    assert h["n_prefill"] == 7 and h["n_past_actual"] == 513
    assert kio.load_kld_metrics(p)[1]["n_past_actual"] == 513

    npz = tmp_path / "recorded.npz"
    kio.convert_kld_bin_to_npz(p, npz)
    assert kio.read_kld_header(npz)["n_past_actual"] == 513
    assert kio.load_kld_metrics(npz)[1]["n_past_actual"] == 513


def test_npz_without_the_optional_header_member_reads_as_unrecorded(tmp_path):
    """Dumps converted before the field existed carry only the four required
    header members; they must load, with n_past_actual reading 0."""
    src = tmp_path / "src.bin"
    rec = write_vlmk(src, make_records(), n_past_actual=99)
    metrics, _h = kio.load_kld_metrics(src)
    old = tmp_path / "old.npz"
    with open(old, "wb") as f:
        np.savez(f, version=np.uint32(kio.VLMK_VERSION), vocab=np.uint32(11),
                 npos=np.uint32(rec.size), n_prefill=np.uint32(7), **metrics)
    assert kio.read_kld_header(old)["n_past_actual"] == 0
    assert kio.load_kld_metrics(old)[1]["n_past_actual"] == 0


def test_read_kld_header_rejects_bad_magic(tmp_path):
    p = tmp_path / "bad.bin"
    write_vlmk(p, make_records(), magic=0xDEADBEEF)
    with pytest.raises(ValueError, match="magic"):
        kio.read_kld_header(p)


def test_read_kld_header_rejects_unknown_version(tmp_path):
    p = tmp_path / "v9.bin"
    write_vlmk(p, make_records(), version=9)
    with pytest.raises(ValueError, match="version"):
        kio.read_kld_header(p)


def test_v1_bin_roundtrip_without_ear(tmp_path):
    """Legacy v1 dumps (40-byte records, no ear) must stay readable: header,
    size validation, load, and lossless npz conversion — with no `ear` key
    anywhere."""
    p = tmp_path / "v1.bin"
    rec = write_vlmk(p, make_records(version=1), version=1)
    assert rec.dtype.itemsize == 40
    assert kio.read_kld_header(p)["version"] == 1
    assert kio.assert_kld_file_complete(p)["npos"] == 5
    metrics, header = kio.load_kld_metrics(p)
    assert set(metrics) == set(kio.kld_metric_keys(1))
    assert "ear" not in metrics
    npz = tmp_path / "v1.npz"
    kio.convert_kld_bin_to_npz(p, npz)
    assert kio.assert_kld_npz_payload(npz)["version"] == 1
    m2, h2 = kio.load_kld_metrics(npz)
    assert h2 == header
    assert "ear" not in m2
    for k in kio.kld_metric_keys(1):
        np.testing.assert_array_equal(m2[k], metrics[k])


def test_v1_size_validation_uses_v1_record_size(tmp_path):
    """A v1 file padded out to the v2 record size must be rejected — the size
    check has to use the HEADER version's record size, not the writer-current
    one."""
    p = tmp_path / "v1.bin"
    write_vlmk(p, make_records(npos=5, version=1), version=1)
    bad = tmp_path / "v1pad.bin"
    bad.write_bytes(p.read_bytes() + b"\x00" * (5 * 4))
    with pytest.raises(ValueError, match="size"):
        kio.assert_kld_file_complete(bad)


def test_npz_rejects_unknown_version(tmp_path):
    """Version gating must hold for the .npz form too, not just .bin."""
    p = tmp_path / "v9.bin"
    write_vlmk(p, make_records())
    npz = tmp_path / "v9.npz"
    kio.convert_kld_bin_to_npz(p, npz)
    # Rewrite the version member to an unsupported value.
    with np.load(npz) as z:
        payload = {k: z[k] for k in z.files}
    payload["version"] = np.uint32(9)
    with open(npz, "wb") as f:
        np.savez(f, **payload)
    with pytest.raises(ValueError, match="version"):
        kio.read_kld_header(npz)
    with pytest.raises(ValueError, match="version"):
        kio.load_kld_metrics(npz)


def test_npz_rejects_column_length_mismatch(tmp_path):
    """A corrupt-but-loadable .npz (column shorter than header npos) must fail
    with a clear message, mirroring the .bin short-payload check."""
    p = tmp_path / "row.bin"
    write_vlmk(p, make_records(npos=5))
    npz = tmp_path / "row.npz"
    kio.convert_kld_bin_to_npz(p, npz)
    with np.load(npz) as z:
        payload = {k: z[k] for k in z.files}
    payload["kld"] = payload["kld"][:3]
    with open(npz, "wb") as f:
        np.savez(f, **payload)
    with pytest.raises(ValueError, match="npos"):
        kio.load_kld_metrics(npz)


def test_load_kld_metrics_returns_contiguous_columns(vlmk_bin):
    metrics, header = kio.load_kld_metrics(vlmk_bin)
    assert header["npos"] == 5
    assert set(metrics) == set(kio.KLD_METRIC_KEYS)
    # derived from the record dtype so a new column (ear) cannot go unchecked
    float_keys = [k for k in kio.KLD_METRIC_KEYS if kio.KLD_RECORD_DT[k].kind == "f"]
    int_keys = [k for k in kio.KLD_METRIC_KEYS if kio.KLD_RECORD_DT[k].kind == "i"]
    assert "ear" in float_keys and len(float_keys) + len(int_keys) == len(kio.KLD_METRIC_KEYS)
    for k in float_keys:
        assert metrics[k].dtype == np.float32, k
        assert metrics[k].flags["C_CONTIGUOUS"], k
        assert metrics[k].shape == (5,), k
    for k in int_keys:
        assert metrics[k].dtype == np.int32, k
        assert metrics[k].flags["C_CONTIGUOUS"], k


def test_load_kld_metrics_values_roundtrip(tmp_path):
    p = tmp_path / "row.bin"
    rec = write_vlmk(p, make_records(npos=4, seed=3))
    metrics, _ = kio.load_kld_metrics(p)
    for k in kio.KLD_METRIC_KEYS:
        np.testing.assert_array_equal(metrics[k], rec[k])


# ---------------------------------------------------------------------------
# completeness validation
# ---------------------------------------------------------------------------

def test_assert_complete_accepts_exact_size(vlmk_bin):
    h = kio.assert_kld_file_complete(vlmk_bin)
    assert h["npos"] == 5


@pytest.mark.parametrize("delta", [-1, -40, +4])
def test_assert_complete_rejects_wrong_size(tmp_path, vlmk_bin, delta):
    data = vlmk_bin.read_bytes()
    bad = tmp_path / "bad.bin"
    bad.write_bytes(data[:delta] if delta < 0 else data + b"\x00" * delta)
    with pytest.raises(ValueError, match="size"):
        kio.assert_kld_file_complete(bad)


def test_load_rejects_short_payload(tmp_path, vlmk_bin):
    bad = tmp_path / "short.bin"
    bad.write_bytes(vlmk_bin.read_bytes()[:-40])
    with pytest.raises(ValueError, match="short"):
        kio.load_kld_metrics(bad)


def test_assert_complete_rejects_npz_input(tmp_path, vlmk_bin):
    """The raw-size formula is meaningless for a zip container; passing a
    .npz must be a clear error, not a bogus 'truncated' verdict."""
    npz = tmp_path / "row.npz"
    kio.convert_kld_bin_to_npz(vlmk_bin, npz)
    with pytest.raises(ValueError, match="expected a VLMK .bin"):
        kio.assert_kld_file_complete(npz)


# ---------------------------------------------------------------------------
# .npz conversion
# ---------------------------------------------------------------------------

def test_convert_roundtrip_matches_bin(tmp_path, vlmk_bin):
    npz_path = tmp_path / "row.npz"
    kio.convert_kld_bin_to_npz(vlmk_bin, npz_path)

    m_bin, h_bin = kio.load_kld_metrics(vlmk_bin)
    m_npz, h_npz = kio.load_kld_metrics(npz_path)
    assert h_bin == h_npz
    for k in kio.KLD_METRIC_KEYS:
        np.testing.assert_array_equal(m_bin[k], m_npz[k])
        assert m_bin[k].dtype == m_npz[k].dtype


def test_npz_loads_standalone_without_kld_metrics_io(tmp_path, vlmk_bin):
    """The .npz must be consumable with bare np.load (the whole point)."""
    npz_path = tmp_path / "row.npz"
    kio.convert_kld_bin_to_npz(vlmk_bin, npz_path)
    with np.load(npz_path) as z:
        assert int(z["vocab"]) == 11
        assert z["kld"].shape == (int(z["npos"]),)
        assert z["argmax_ref"].dtype == np.int32


def test_convert_keeps_bin_and_leaves_no_tmp(tmp_path, vlmk_bin):
    npz_path = tmp_path / "out.npz"
    kio.convert_kld_bin_to_npz(vlmk_bin, npz_path)
    assert npz_path.exists()
    assert vlmk_bin.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_convert_rejects_truncated_bin(tmp_path, vlmk_bin):
    bad = tmp_path / "short.bin"
    bad.write_bytes(vlmk_bin.read_bytes()[:-40])
    with pytest.raises(ValueError):
        kio.convert_kld_bin_to_npz(bad, tmp_path / "out.npz")
    assert not (tmp_path / "out.npz").exists()


def test_convert_fsyncs_payload_before_rename(tmp_path, vlmk_bin, monkeypatch):
    """Durability: the rename must never be persisted ahead of the payload
    (same contract as logits_io.convert_bin_to_npz)."""
    import os as real_os
    calls = []
    real_fsync, real_replace = real_os.fsync, real_os.replace
    monkeypatch.setattr(kio.os, "fsync",
                        lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(kio.os, "replace",
                        lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    kio.convert_kld_bin_to_npz(vlmk_bin, tmp_path / "out.npz")
    assert calls == ["fsync", "replace"]


# ---------------------------------------------------------------------------
# load_kld_metrics(.npz) validates dtypes, not just shapes (H2): the .bin
# branch gets dtypes for free from the structured read; the .npz branch must
# check them against the same KLD_RECORD_DT schema or a float64 metric column
# with the right shape loads silently at the wrong precision
# ---------------------------------------------------------------------------

def _npz_from_bin(tmp_path, mutate=None):
    """A converted-good .npz, optionally with one member rewritten."""
    p = tmp_path / "m.bin"
    write_vlmk_local = write_vlmk
    write_vlmk_local(p, make_records(npos=4))
    npz = tmp_path / "m.npz"
    kio.convert_kld_bin_to_npz(p, npz)
    if mutate is None:
        return npz
    members = dict(np.load(npz))
    mutate(members)
    np.savez(npz, **members)
    return npz


def test_npz_loader_rejects_wrong_metric_dtype(tmp_path):
    npz = _npz_from_bin(tmp_path, lambda m: m.update(
        kld=m["kld"].astype(np.float64)))
    with pytest.raises(ValueError, match=r"'kld' dtype float64"):
        kio.load_kld_metrics(npz)


def test_npz_loader_rejects_wrong_index_dtype(tmp_path):
    npz = _npz_from_bin(tmp_path, lambda m: m.update(
        target=m["target"].astype(np.int64)))
    with pytest.raises(ValueError, match=r"'target' dtype int64"):
        kio.load_kld_metrics(npz)


def test_npz_loader_rejects_missing_column_cleanly(tmp_path):
    def drop_kld(m):
        del m["kld"]
    npz = _npz_from_bin(tmp_path, drop_kld)
    with pytest.raises(ValueError, match=r"missing: \['kld'\]"):
        kio.load_kld_metrics(npz)


def test_npz_loader_still_accepts_a_good_dump(tmp_path):
    npz = _npz_from_bin(tmp_path)
    metrics, header = kio.load_kld_metrics(npz)
    assert header["npos"] == 4
    assert metrics["kld"].dtype == np.float32
    assert metrics["target"].dtype == np.int32


def test_load_kld_metrics_bin_rejects_trailing_bytes(tmp_path):
    """H3: the loader itself carries the exact-size gate; a .bin with bytes
    appended after the last record is not what its writer committed."""
    p = tmp_path / "trail.bin"
    write_vlmk(p, make_records(npos=3))
    with open(p, "ab") as f:
        f.write(b"\xff" * 7)
    with pytest.raises(ValueError, match="truncated or corrupt"):
        kio.load_kld_metrics(p)


# ---------------------------------------------------------------------------
# item_means: the ONE VLMK per-item aggregation (R8) -- the consumer must
# reproduce it exactly
# ---------------------------------------------------------------------------

def test_item_means_is_what_the_consumer_reports(tmp_path):
    import cli.saved_metrics_paired_compare as smpc
    rec = make_records(npos=9)
    m = {k: np.ascontiguousarray(rec[k]) for k in rec.dtype.names}
    for keep in (9, 4):
        scores, dp = kio.item_means(m, keep)
        side, tok = smpc._side_scores(m, keep)
        assert side == scores                      # exact float equality
        np.testing.assert_array_equal(tok["dp"], dp.astype(np.float32))
    # v1 dumps have no ear column: the aggregation does not invent one
    m1 = {k: v for k, v in m.items() if k != "ear"}
    assert "ear" not in kio.item_means(m1)[0]
    # v2 dumps have ear but not the EAR_K family
    m2 = {k: v for k, v in m.items() if k not in ("ear_20", "ear_10", "ear_5")}
    s2 = kio.item_means(m2)[0]
    assert "ear" in s2 and not ({"ear_20", "ear_10", "ear_5"} & set(s2))
    full = kio.item_means(m)[0]
    for key in ("ear_20", "ear_10", "ear_5"):
        assert full[key] == float(m[key].astype(np.float64).mean())
