"""Tests for prep_llm_score_from_hf.py.

Hermetic: no datasets/transformers import; rows are plain dicts and output is
tokens.bin + meta.json only.
"""

import json

import numpy as np
import pytest

import cli.prep_llm_score_from_hf as prep


def _row(**overrides):
    row = {
        "id": 17,
        "question": "What is 2+2?",
        "source": "unit",
        "category": "math",
        "input_ids": [10, 11, 12, 20, 21, 22],
        "input_tokens_len": 6,
        "generated_texts": "The answer is 4.",
        "generated_tokens_len": 3,
        "n_prefill_tokens": 3,
        "seed": 123,
        "labels": [-100, -100, -100, 20, 21, 22],
    }
    row.update(overrides)
    return row


def test_prep_row_writes_tokens_and_meta(tmp_path):
    meta = prep.prep_row(_row(), tmp_path)

    assert np.fromfile(tmp_path / "tokens.bin", dtype=np.int32).tolist() == [
        10, 11, 12, 20, 21, 22,
    ]
    assert meta == json.loads((tmp_path / "meta.json").read_text())
    assert meta == {
        "item_id": "17",
        "source": "unit",
        "category": "math",
        "n_prefill": 3,
        "n_answer": 3,
        "input_tokens_len": 6,
        "generated_texts_preview": "The answer is 4.",
        "dataset_seed": 123,
        "question_preview": "What is 2+2?",
    }
    assert not (tmp_path / "formatted_chat.txt").exists()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"input_ids": [1, 2]}, "len\\(input_ids\\)"),
        ({"n_prefill_tokens": 0}, "n_prefill_tokens"),
        ({"n_prefill_tokens": 6}, "n_prefill_tokens"),
        ({"generated_tokens_len": 2}, "n_prefill"),
        ({"labels": [-100, -100]}, "len\\(labels\\)"),
        ({"labels": [-100, 11, -100, 20, 21, 22]}, "labels\\[:n_prefill\\]"),
        ({"labels": [-100, -100, -100, 20, 99, 22]}, "labels\\[n_prefill:\\]"),
    ],
)
def test_sanity_check_row_rejects_contract_violations(overrides, match):
    with pytest.raises(prep.PrepError, match=match):
        prep.sanity_check_row(_row(**overrides))


def test_load_dataset_sorted_allows_falsy_sort_by(monkeypatch):
    calls = []

    class FakeDataset:
        def sort(self, keys):
            calls.append(("sort", keys))
            return self

    def fake_load_dataset(dataset, subset=None, split=None):
        calls.append(("load", dataset, subset, split))
        return FakeDataset()

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        type("D", (), {"load_dataset": fake_load_dataset}),
    )

    prep.load_dataset_sorted("ds", "cfg", "train", "")

    assert calls == [("load", "ds", "cfg", "train")]


def test_load_dataset_sorted_can_sort_descending(monkeypatch):
    calls = []

    class FakeDataset:
        def sort(self, keys, reverse=False):
            calls.append(("sort", keys, reverse))
            return self

    def fake_load_dataset(dataset, subset=None, split=None):
        calls.append(("load", dataset, subset, split))
        return FakeDataset()

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        type("D", (), {"load_dataset": fake_load_dataset}),
    )

    prep.load_dataset_sorted("ds", "cfg", "train", "generated_tokens_len",
                             sort_desc=True)

    assert calls == [
        ("load", "ds", "cfg", "train"),
        ("sort", ["generated_tokens_len"], True),
    ]


@pytest.mark.parametrize("flag,expected", [(["--sort-desc"], True), ([], False)])
def test_main_threads_sort_desc_into_dataset_load(tmp_path, monkeypatch,
                                                  flag, expected):
    """CLI regression: without --sort-desc threaded through, 'row N' of a
    collection gathered with --sort-desc resolves to a DIFFERENT item (the
    ascending N-th), so the tool debugs the wrong trajectory."""
    seen = {}

    def fake_load(dataset, subset, split, sort_by, sort_desc=False):
        seen.update(sort_by=sort_by, sort_desc=sort_desc)

        class DS:
            def __getitem__(self, idx):
                seen["row"] = idx
                return _row()

        return DS()

    monkeypatch.setattr(prep, "load_dataset_sorted", fake_load)
    monkeypatch.setattr("sys.argv", [
        "prep_llm_score_from_hf.py", "--row", "1",
        "--out", str(tmp_path / "o"),
        "--sort-by", "generated_tokens_len", *flag,
    ])

    prep.main()

    assert seen == {"sort_by": "generated_tokens_len",
                    "sort_desc": expected, "row": 1}
    assert (tmp_path / "o" / "tokens.bin").exists()
    assert (tmp_path / "o" / "meta.json").exists()
