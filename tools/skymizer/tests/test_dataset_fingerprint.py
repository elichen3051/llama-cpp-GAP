"""dataset_fingerprint: content hash + the ensure_collect_meta guard step.

The hash must be deterministic and sensitive to exactly the fields the
collectors condition on -- row order, row id, input_ids, encoded image
bytes -- and the generic fallback must accept the FakeDataset-style objects
the repro harness injects (len + int-getitem only).
"""


import pytest

import lib.dataset_fingerprint as df
from lib.dataset_fingerprint import (
    check_dataset_content_hash,
    check_model_fingerprints,
    dataset_content_hash,
    file_content_fingerprint,
    maybe_file_fingerprint,
)


class _FakeDataset:
    """Mirror of the repro harness FakeDataset: len + int __getitem__."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return self.rows[key]


def _row(rid, tokens, **extra):
    return {"id": rid, "input_ids": list(tokens), **extra}


def test_hash_is_deterministic_and_order_sensitive():
    a = _FakeDataset([_row("x", [1, 2]), _row("y", [3, 4])])
    b = _FakeDataset([_row("x", [1, 2]), _row("y", [3, 4])])
    swapped = _FakeDataset([_row("y", [3, 4]), _row("x", [1, 2])])
    assert dataset_content_hash(a) == dataset_content_hash(b)
    # row_idx keys the on-disk artifacts, so order is part of the identity
    assert dataset_content_hash(a) != dataset_content_hash(swapped)


def test_hash_sensitive_to_id_and_tokens():
    base = dataset_content_hash(_FakeDataset([_row("x", [1, 2])]))
    other_id = dataset_content_hash(_FakeDataset([_row("z", [1, 2])]))
    other_tok = dataset_content_hash(_FakeDataset([_row("x", [1, 9])]))
    assert base != other_id
    assert base != other_tok


def test_hash_covers_same_length_token_drift():
    """The finding-10 scenario: same id, same length, different answer tokens."""
    before = _FakeDataset([_row("same-id", [10, 11, 3, 4])])
    after = _FakeDataset([_row("same-id", [10, 11, 8, 9])])
    assert dataset_content_hash(before) != dataset_content_hash(after)


def test_hash_covers_image_bytes_and_item_id_naming():
    """VLM rows: item_id naming, image cells as dict{bytes}/list-of-dict."""
    img_a = {"item_id": "v", "input_ids": [1], "images": [{"bytes": b"AA", "path": None}]}
    img_b = {"item_id": "v", "input_ids": [1], "images": [{"bytes": b"AB", "path": None}]}
    assert (dataset_content_hash(_FakeDataset([img_a]))
            != dataset_content_hash(_FakeDataset([img_b])))


def test_hash_rejects_rows_without_id():
    with pytest.raises(ValueError, match="neither 'item_id' nor 'id'"):
        dataset_content_hash(_FakeDataset([{"input_ids": [1]}]))


def test_hf_fast_path_matches_generic_fallback():
    """The HF and generic paths hash identical logical content identically,
    so a fake-injected dataset (repros) and a real one agree -- including the
    ds-v2 boundary and generation-model fields."""
    datasets = pytest.importorskip("datasets")
    rows = [_row("x", [1, 2], n_prefill_tokens=1,
                 generation_model_name_or_path="org/gen"),
            _row("y", [3, 4], n_prefill_tokens=2,
                 generation_model_name_or_path="org/gen")]
    hf = datasets.Dataset.from_dict({
        "id": [r["id"] for r in rows],
        "input_ids": [r["input_ids"] for r in rows],
        "n_prefill_tokens": [r["n_prefill_tokens"] for r in rows],
        "generation_model_name_or_path":
            [r["generation_model_name_or_path"] for r in rows],
    })
    assert dataset_content_hash(hf) == dataset_content_hash(_FakeDataset(rows))


def test_hash_is_scheme_tagged():
    # ds-v2 added the boundary/generation-model fields; the prefix lets any
    # future field-set change coexist instead of silently colliding.
    assert dataset_content_hash(_FakeDataset([_row("x", [1, 2])])).startswith("ds-v2:")


def test_hash_sensitive_to_prompt_boundary():
    """codex round-2 finding 1: identical ids/tokens under a shifted
    n_prefill_tokens boundary is DIFFERENT teacher-forcing conditioning."""
    two = _FakeDataset([_row("x", [1, 2, 3, 4], n_prefill_tokens=2)])
    three = _FakeDataset([_row("x", [1, 2, 3, 4], n_prefill_tokens=3)])
    assert dataset_content_hash(two) != dataset_content_hash(three)


def test_hash_sensitive_to_generation_model():
    """generation_model_name_or_path selects the VLM prep tokenizer."""
    m1 = _FakeDataset([_row("x", [1], generation_model_name_or_path="org/m1")])
    m2 = _FakeDataset([_row("x", [1], generation_model_name_or_path="org/m2")])
    assert dataset_content_hash(m1) != dataset_content_hash(m2)


def test_hf_fast_path_hashes_image_columns_without_decoding():
    datasets = pytest.importorskip("datasets")
    from datasets import Features, Image, Value

    def build(png_bytes):
        return datasets.Dataset.from_dict(
            {"item_id": ["a"], "input_ids": [[1]],
             "image": [{"bytes": png_bytes, "path": None}]},
            features=Features({"item_id": Value("string"),
                               "input_ids": [Value("int32")],
                               "image": Image()}),
        )

    # Not valid PNG payloads, but decode=False never parses them -- which is
    # exactly the property under test.
    assert (dataset_content_hash(build(b"not-a-real-png-1"))
            != dataset_content_hash(build(b"not-a-real-png-2")))


# --------------------------------------------------------------------------- #
# check_dataset_content_hash guard semantics
# --------------------------------------------------------------------------- #
def test_file_fingerprint_small_file_fully_covered(tmp_path):
    """Below one 4 MiB window every offset collapses into whole-file
    coverage: ANY byte change (not just size) must alter the fingerprint."""
    f = tmp_path / "m.gguf"
    data = bytearray(b"model revision A" * 64)
    f.write_bytes(data)
    base = file_content_fingerprint(f)

    longer = tmp_path / "longer.gguf"
    longer.write_bytes(bytes(data) + b"!")
    assert file_content_fingerprint(longer) != base   # size change

    data[len(data) // 2] ^= 0xFF
    f.write_bytes(data)                               # same size, one byte
    assert file_content_fingerprint(f) != base


def test_file_fingerprint_window_coverage_and_blindspot(monkeypatch, tmp_path):
    """Shrink the window to 16 bytes so a 4 KiB file has real gaps, then pin
    both sides of the design: flips inside the first/middle/last sampled
    windows change the fingerprint; a flip BETWEEN windows does not. The
    blind spot is deliberate -- the scheme trades full coverage for a fixed
    ~72 MiB read (0.037 s on an 8 GB GGUF vs 10 s full sha256) and defends
    against in-place replacement accidents, which realistic modifications
    (requantization) spread across the whole tensor region anyway."""
    monkeypatch.setattr(df, "_SAMPLE_BLOCK", 16)
    size = 4096
    # offsets: 0, int(4096*i/17) i=1..16 (240, 481, ..., 3855), 4080
    def fp_with_flip(pos=None):
        data = bytearray(range(256)) * (size // 256)
        if pos is not None:
            data[pos] ^= 0xFF
        f = tmp_path / "m.gguf"
        f.write_bytes(data)
        return file_content_fingerprint(f)

    base = fp_with_flip()
    assert fp_with_flip(5) != base       # first window [0, 16)
    assert fp_with_flip(245) != base     # a middle window [240, 256)
    assert fp_with_flip(4090) != base    # last window [4080, 4096)
    assert fp_with_flip(100) == base     # gap between windows: invisible


def test_maybe_file_fingerprint_none_for_missing_path(tmp_path):
    assert maybe_file_fingerprint(tmp_path / "nope.gguf") is None
    f = tmp_path / "yes.gguf"
    f.write_bytes(b"w")
    assert maybe_file_fingerprint(f).startswith("gguf-sampled-v1:")


# --------------------------------------------------------------------------- #
# check_model_fingerprints guard semantics (shared three-tier checker)
# --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------
# three-tier shard-identity guards (restored with the later-shard verification)
# ---------------------------------------------------------------------------

def test_guard_noop_when_current_has_no_hash():
    assert check_dataset_content_hash({"dataset_content_hash": "ds-v2:x"}, {}) is None


def test_guard_warns_on_legacy_dir_without_hash():
    warning = check_dataset_content_hash({}, {"dataset_content_hash": "ds-v2:x"})
    assert "unverified" in warning and "dataset_content_hash" in warning


def test_guard_passes_on_equal_hash():
    meta = {"dataset_content_hash": "ds-v2:x"}
    assert check_dataset_content_hash(meta, dict(meta)) is None


def test_guard_refuses_on_hash_mismatch():
    with pytest.raises(SystemExit, match="dataset_content_hash"):
        check_dataset_content_hash({"dataset_content_hash": "ds-v2:x"},
                                   {"dataset_content_hash": "ds-v2:y"})


def test_model_guard_warns_per_legacy_field_and_passes_on_equal():
    fields = ("model_fingerprint", "mmproj_fingerprint")
    cur = {f: "gguf-sampled-v1:x" for f in fields}
    warnings = check_model_fingerprints({}, cur, fields)
    assert len(warnings) == 2 and all("unverified" in w for w in warnings)
    assert check_model_fingerprints(dict(cur), cur, fields) == []


def test_model_guard_refuses_on_mismatch():
    with pytest.raises(SystemExit, match="model_fingerprint"):
        check_model_fingerprints({"model_fingerprint": "gguf-sampled-v1:x"},
                                 {"model_fingerprint": "gguf-sampled-v1:y"},
                                 ("model_fingerprint",))
