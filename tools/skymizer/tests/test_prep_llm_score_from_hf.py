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


def _corpus_protocol():
    from lib.text_corpus import SCHEMA
    return {
        "schema": SCHEMA, "corpus_name": "example", "window_size": 8, "stride": 8,
        "n_prefill": 5, "targets_per_window": 3, "stream_tokens": 19,
        "corpus_sha256": "1" * 64, "effective_text_sha256": "2" * 64,
        "stream_sha256": "3" * 64, "articles_sha256": "4" * 64,
        "parse_special": False, "escape": False, "strip_single_final_newline": True,
        "bos_policy": "replace-window-position-zero", "bos_id": 1,
        "tail_policy": "drop-incomplete-window",
        "vocabulary": {"scheme": "llama-vocabulary-sha256-v1", "size": 100,
                       "type": 2, "mapping": "5" * 64, "attributes": "6" * 64},
    }


def test_corpus_window_preserves_targets_and_article_crossing(tmp_path):
    from lib.text_corpus import corpus_row, validate_corpus_dataset
    protocol = _corpus_protocol()
    rows = [corpus_row(protocol, [1] + list(range(10 + i * 8, 17 + i * 8)), i, [0, 1, 1]) for i in range(2)]
    meta = prep.prep_row(rows[0], tmp_path)
    assert np.fromfile(tmp_path / "tokens.bin", dtype=np.int32).tolist() == rows[0]["input_ids"]
    assert meta["n_prefill"] == 5 and meta["n_answer"] == 3
    assert meta["reference_vocabulary"] == protocol["vocabulary"]
    assert meta["corpus_window"]["target_article_ids"] == [0, 1, 1]
    assert len(validate_corpus_dataset(rows, 8)["windows"]) == 2
    for changed in (rows[:1], rows[::-1], rows + rows[:1]):
        with pytest.raises(ValueError):
            validate_corpus_dataset(changed, 8)
    rows[0]["input_ids"][-1] += 1
    rows[0]["labels"][-1] += 1
    with pytest.raises(ValueError, match="tokens or scoring targets changed"):
        prep.prep_row(rows[0], tmp_path)


def test_native_article_cut_assigns_boundary_affected_token_to_next_article():
    from lib.text_corpus import stable_prefix_boundary
    assert stable_prefix_boundary([1, 2, 3, 4], [1, 2]) == 2
    assert stable_prefix_boundary([1, 2, 3, 4], [1, 9]) == 1
    assert stable_prefix_boundary([1, 2, 3, 4], [1, 2, 3, 4]) == 4


def test_wikitext_article_index_preserves_bytes_and_ignores_section_headers():
    from cli.prepare_perplexity_corpus import article_spans
    raw = b" \n = First = \nbody\n == Section == \ntext\n = Second = \nend\n"
    spans = article_spans(raw, "wikitext-2-test")
    assert [span["title"] for span in spans] == ["First", "Second"]
    assert spans[0]["byte_start"] == 0
    assert spans[0]["byte_end"] == spans[1]["byte_start"] == raw.index(b" = Second")
    assert spans[1]["byte_end"] == len(raw)
