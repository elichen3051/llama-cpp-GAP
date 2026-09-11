"""Shared hermetic fakes for the company test suite (no transformers/PIL).

FakeTok mimics the two tokenizer surfaces the scripts use: decode() for
prep's decode-and-collapse, and the convert_tokens_to_ids /
convert_ids_to_tokens pair for image-pad token resolution.
"""


class FakeTok:
    def __init__(self, text_by_id=None):
        self.text_by_id = text_by_id or {}
        self.id_by_text = {v: k for k, v in self.text_by_id.items()}

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(self.text_by_id.get(i, str(i)) for i in ids)

    def convert_tokens_to_ids(self, token):
        # Mirrors HF fast-tokenizer behavior for a missing token: returns the
        # unk id (0 here), never None and never raising.
        return self.id_by_text.get(token, 0)

    def convert_ids_to_tokens(self, idx):
        return self.text_by_id.get(idx, "<unk>")


class FakeImage:
    def save(self, path):
        path.write_bytes(b"PNG")


# --------------------------------------------------------------------------- #
# Canonical binary-fixture writer for VLMK metric dumps.
# --------------------------------------------------------------------------- #
import numpy as np

import lib.kld_metrics_io as kio


def write_vlmk(path, records, vocab=11, n_prefill=7, version=kio.VLMK_VERSION,
               magic=kio.VLMK_MAGIC, n_past_actual=0):
    """Write a VLMK file: 24B header + packed records (dtype follows the
    version, so v1 fixtures get genuine 40-byte records)."""
    record_dt = (kio.kld_record_dt(version)
                 if version in kio.VLMK_SUPPORTED_VERSIONS else kio.KLD_RECORD_DT)
    records = np.asarray(records, dtype=record_dt)
    header = np.array([magic, version, vocab, records.size, n_prefill,
                       n_past_actual], dtype="<u4")
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(records.tobytes())
    return records


def make_records(npos=5, vocab=11, seed=0, version=kio.VLMK_VERSION):
    rng = np.random.default_rng(seed)
    record_dt = kio.kld_record_dt(version)
    rec = np.zeros(npos, dtype=record_dt)
    # Draw the v2-era columns first, in their v2 order, so fixtures for the
    # pre-existing columns are bit-identical to what earlier versions of this
    # helper produced; columns added by later versions draw afterwards.
    legacy = [k for k in kio.KLD_RECORD_DT_V2.names if k in record_dt.names]
    names = legacy + [k for k in record_dt.names if k not in legacy]
    for k in names:
        if record_dt[k].kind == "f":
            rec[k] = rng.uniform(0, 5, size=npos).astype(np.float32)
        else:
            rec[k] = rng.integers(0, vocab, size=npos).astype(np.int32)
    return rec


EXECUTION_IDENTITY = {
    "scheme": "company-execution-sha256-v2", "binary_sha256": "a" * 64,
    "libraries": [], "gpu": ["CPU fixture"], "environment": {},
}


def completed_collection(root, keys, skipped=()):
    """Publish declared fixture rows independently of metric contents."""
    import csv
    from lib.collection_state import CollectionAttempt
    rows = [(int(k.split("_", 1)[0]), k.split("_", 1)[1]) for k in keys]
    attempt = CollectionAttempt(root, min(i for i, _ in rows), max(i for i, _ in rows) + 1)
    attempt.declare(rows)
    with (root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_idx", "item_id", "status"])
        writer.writeheader()
        for idx, item in rows:
            record = {"row_idx": idx, "item_id": item, "status": "SKIP_OVER_BUDGET" if idx in skipped else "OK"}
            writer.writerow(record)
            attempt.record(record)
    attempt.finish()
