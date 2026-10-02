"""Executable contract for exact-span, per-article corpus preparation."""

import json
import runpy
import subprocess
import sys

import numpy as np
import pytest

from lib.reference_dataset import canonical_json
from article_fakes import (
    ARTICLES, GAP, SCHEMA, VOCABULARY, article_module, digest_json, digest_tokens,
    make_prepare_inputs, offline_env, prepare_args, prepared_articles, require_cli, run_cli, sha256,
)


def test_schema_and_lazy_dataset_import(tmp_path):
    assert article_module().SCHEMA == SCHEMA
    result = subprocess.run([sys.executable, "-c", "import sys; sys.modules['datasets'] = None; import lib.article_corpus"],
                            cwd=GAP, env=offline_env(tmp_path), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key", ["articles", "items"])
def test_article_spans_partition_and_metadata(key):
    module = article_module()
    raw = b"aa\nbb\ncc\n"
    entries = [
        {"id": "first", "title": "First", "byte_start": 0, "byte_end_exclusive": 3},
        {"index": 7, "rss_title": "Second", "byte_start": 3, "byte_end_exclusive": 6},
        {"byte_start": 6, "byte_end_exclusive": 9},
    ]
    expected = [
        {"source_id": source, "title": title, "byte_start": i * 3, "byte_end_exclusive": (i + 1) * 3,
         "sha256": sha256(raw[i * 3:(i + 1) * 3])}
        for i, (source, title) in enumerate([("first", "First"), ("7", "Second"), ("2", "")])
    ]
    for index in ({key: entries}, {key: entries, "corpus_sha256": sha256(raw)}):
        assert module.article_spans(raw, index) == expected


@pytest.mark.parametrize("bounds", [
    [(1, 3), (3, 6)], [(0, 2), (3, 6)], [(0, 4), (3, 6)], [(0, 3), (3, 5)],
    [(0, 3), (3, 7)], [(0, 0), (0, 6)], [(0, 3), (4, 3)], [(3, 6), (0, 3)],
    [(-1, 3), (3, 6)], [(0, 2.5), (2.5, 6)], [],
], ids=["leading-gap", "gap", "overlap", "tail-gap", "past-end", "empty", "reversed", "out-of-order", "negative", "noninteger", "no-spans"])
def test_article_spans_reject_invalid_partition(bounds):
    module = article_module()
    index = {"articles": [{"byte_start": start, "byte_end_exclusive": end} for start, end in bounds]}
    with pytest.raises(ValueError):
        module.article_spans(b"aa\nbb\n", index)


@pytest.mark.parametrize("index", [
    {}, {"articles": "not a list"}, {"articles": [{}]},
    {"articles": [{"byte_start": 0, "byte_end_exclusive": 2}], "corpus_sha256": "0" * 64},
])
def test_article_spans_reject_bad_index_or_digest(index):
    module = article_module()
    with pytest.raises(ValueError):
        module.article_spans(b"a\n", index)


@pytest.mark.parametrize("key", ["articles", "items"])
@pytest.mark.parametrize("character", ["\u00e9", "\u20ac", "\U0001f642"], ids=["two-byte", "three-byte", "four-byte"])
def test_article_spans_reject_split_utf8_character(key, character):
    raw = ("a" + character + "b\n").encode("utf-8")
    assert raw.decode("utf-8") == "a" + character + "b\n"
    for split in range(2, 1 + len(character.encode("utf-8"))):
        index = {key: [{"byte_start": 0, "byte_end_exclusive": split},
                       {"byte_start": split, "byte_end_exclusive": len(raw)}], "corpus_sha256": sha256(raw)}
        with pytest.raises(ValueError):
            article_module().article_spans(raw, index)
    boundary = 1 + len(character.encode("utf-8"))
    valid = {key: [{"byte_start": 0, "byte_end_exclusive": boundary},
                   {"byte_start": boundary, "byte_end_exclusive": len(raw)}]}
    assert len(article_module().article_spans(raw, valid)) == 2


@pytest.mark.parametrize("with_bos,without_bos,expected", [
    ([2, 17, 18], [17, 18], 2), ([17, 18], [17, 18], None), ([2], [], 2), ([], [], None),
])
def test_bos_split_accepts_only_optional_leading_bos(with_bos, without_bos, expected):
    assert article_module().bos_split(with_bos, without_bos) == expected


@pytest.mark.parametrize("with_bos,without_bos", [
    ([2, 17, 18, 3], [17, 18]), ([17, 18, 3], [17, 18]),
    ([2, 3, 17, 18], [17, 18]), ([2, 17, 19], [17, 18]), ([17], [17, 18]),
])
def test_bos_split_rejects_eos_or_other_token_changes(with_bos, without_bos):
    module = article_module()
    with pytest.raises(ValueError):
        module.bos_split(with_bos, without_bos)


@pytest.mark.parametrize("tokens", [[], [2, 17, 255, 65536], [-2147483648, -1, 0, 2147483647]])
def test_token_digest_is_little_endian_int32(tokens):
    module = article_module()
    from lib.text_corpus import token_digest
    expected = digest_tokens(tokens)
    assert module.token_digest(tokens) == expected == token_digest(tokens)
    assert module.token_digest(np.asarray(tokens, dtype=">i4")) == expected


@pytest.mark.parametrize("mode", ["bos", "no-bos"])
def test_prepare_exact_article_bytes_rows_and_native_calls(tmp_path, mode):
    out, receipt, rows, inputs = prepared_articles(tmp_path, mode)
    module = article_module()
    from cli.prep_llm_score_from_hf import sanity_check_row
    assert len(rows) == len(ARTICLES)
    protocol = receipt["protocol"]
    assert protocol["bos_id"] == (2 if mode == "bos" else None)
    for i, (raw, row, record) in enumerate(zip(ARTICLES, rows, receipt["articles"])):
        tokens = ([2] if mode == "bos" else []) + [byte + 16 for byte in raw]
        assert row == {
            "id": f"{protocol['corpus_name']}-{digest_json(protocol)[:16]}-{i:04d}",
            "source": protocol["corpus_name"], "category": "corpus-article", "question": "", "generated_texts": "", "seed": 0,
            "input_ids": tokens, "input_tokens_len": len(tokens), "n_prefill_tokens": 1,
            "generated_tokens_len": len(tokens) - 1, "labels": [-100] + tokens[1:],
            "article_protocol": canonical_json(protocol), "corpus_article": canonical_json(record),
        }
        assert "corpus_protocol" not in row
        sanity_check_row(row)
        assert module.validate_article_row(row) == (protocol, record)
        assert module.article_row(protocol, record, tokens) == row
    calls = [json.loads(line) for line in (inputs["corpus"].parent / "native-calls.jsonl").read_text().splitlines()]
    assert all(call["cuda"] == "" for call in calls)
    token_calls = [call for call in calls if call["tool"] == "tokenize"]
    assert len(token_calls) == 2 * len(ARTICLES)
    for raw in ARTICLES:
        pair = [call for call in token_calls if call["hex"] == raw.hex()]
        base = ["-m", str(inputs["ref_model"]), "--stdin", "--ids", "--no-escape", "--no-parse-special"]
        assert sorted(sorted(call["args"]) for call in pair) == sorted([sorted(base), sorted([*base, "--no-bos"])])
        assert all(call["args"][call["args"].index("-m") + 1] == str(inputs["ref_model"]) for call in pair)
    vocab_calls = [call for call in calls if call["tool"] == "vocab"]
    assert vocab_calls
    assert all(call["args"] == ["--vocab-identity", str(inputs["ref_model"])] for call in vocab_calls)
    logs = "\n".join(path.read_text() for path in (out / "logs").rglob("*") if path.is_file())
    assert logs.count("tokenizer diagnostic") == 2 * len(ARTICLES)
    assert logs.count("vocabulary diagnostic") == len(vocab_calls)


@pytest.mark.parametrize("mode", ["bos", "no-bos"])
def test_prepare_receipt_counts_hashes_and_determinism(tmp_path, mode):
    out, receipt, rows, inputs = prepared_articles(tmp_path, mode)
    raw = inputs["corpus"].read_bytes()
    protocol = receipt["protocol"]
    articles_without_ids = [{key: value for key, value in record.items() if key != "id"} for record in receipt["articles"]]
    assert set(receipt) == {"protocol", "articles", "total_tokens", "total_targets"}
    assert protocol == {
        "schema": SCHEMA, "corpus_name": "wikitext-2-test-article", "corpus_sha256": sha256(raw), "corpus_bytes": len(raw),
        "article_index_sha256": sha256(inputs["article_index"].read_bytes()), "articles_sha256": digest_json(articles_without_ids),
        "article_bytes": "exact-span", "escape": False, "parse_special": False, "bos_id": 2 if mode == "bos" else None,
        "bos_policy": "vocabulary-add-bos", "n_prefill": 1, "scoring": "every-token-after-position-0",
        "vocabulary": VOCABULARY, "tokenizer_binary_sha256": sha256(inputs["llama_tokenize"].read_bytes()),
    }
    cursor = 0
    for i, (text, row, record) in enumerate(zip(ARTICLES, rows, receipt["articles"])):
        tokens = row["input_ids"]
        assert row["id"] == f"{protocol['corpus_name']}-{digest_json(protocol)[:16]}-{i:04d}"
        assert record == {
            "index": i, "id": row["id"], "source_id": f"source-{i}", "title": f"Title {i}",
            "byte_start": cursor, "byte_end_exclusive": cursor + len(text), "sha256": sha256(text),
            "n_tokens": len(tokens), "n_targets": len(tokens) - 1,
            "tokens_sha256": digest_tokens(tokens), "targets_sha256": digest_tokens(tokens[1:]),
        }
        cursor += len(text)
    assert receipt["total_tokens"] == sum(len(row["input_ids"]) for row in rows)
    assert receipt["total_targets"] == receipt["total_tokens"] - len(rows)
    payload = (out / "corpus_articles.json").read_bytes()
    assert payload == (canonical_json(receipt) + "\n").encode("utf-8")
    assert all(text.decode() not in payload.decode() for text in ARTICLES)
    assert article_module().load_articles(out / "corpus_articles.json") == receipt
    other = tmp_path / "prepared-again"
    result = run_cli("prepare_article_corpus", prepare_args(inputs, other), tmp_path)
    assert result.returncode == 0, result.stderr
    from datasets import load_from_disk
    assert (other / "corpus_articles.json").read_bytes() == payload
    assert list(load_from_disk(str(other / "dataset"))) == rows


@pytest.mark.parametrize("field,value", [
    ("id", "wrong"), ("source", "wrong"), ("category", "wrong"), ("question", "not empty"),
    ("generated_texts", "not empty"), ("seed", 1), ("input_tokens_len", 999),
    ("n_prefill_tokens", 2), ("generated_tokens_len", 999), ("labels", [-100]),
    ("input_ids", [2, 17]),
])
def test_validate_article_row_rejects_changed_columns(tmp_path, field, value):
    _, _, rows, _ = prepared_articles(tmp_path)
    row = {**rows[0], field: value}
    with pytest.raises(ValueError):
        article_module().validate_article_row(row)


@pytest.mark.parametrize("field,change", [
    ("article_protocol", "schema"), ("article_protocol", "corpus_sha256"),
    ("corpus_article", "id"), ("corpus_article", "tokens_sha256"), ("corpus_article", "targets_sha256"),
])
def test_validate_article_row_rejects_changed_provenance(tmp_path, field, change):
    _, _, rows, _ = prepared_articles(tmp_path)
    row = dict(rows[0])
    value = json.loads(row[field])
    value[change] = "0" * 64 if change.endswith("sha256") else "wrong"
    row[field] = canonical_json(value)
    with pytest.raises(ValueError):
        article_module().validate_article_row(row)


@pytest.mark.parametrize("tokens", [[], [2]])
def test_article_row_rejects_no_scorable_token(tmp_path, tokens):
    _, receipt, _, _ = prepared_articles(tmp_path)
    with pytest.raises(ValueError):
        article_module().article_row(receipt["protocol"], receipt["articles"][0], tokens)


@pytest.mark.parametrize("change", ["total_tokens", "total_targets", "n_targets", "articles_sha256", "id", "order"])
def test_load_articles_rejects_invalid_receipt(tmp_path, change):
    out, receipt, _, _ = prepared_articles(tmp_path)
    if change in ("total_tokens", "total_targets"):
        receipt[change] += 1
    elif change == "n_targets":
        receipt["articles"][0][change] += 1
    elif change == "articles_sha256":
        receipt["protocol"][change] = "0" * 64
    elif change == "id":
        receipt["articles"][0][change] = "wrong"
    else:
        receipt["articles"].reverse()
    path = out / "bad-articles.json"
    path.write_text(canonical_json(receipt) + "\n")
    with pytest.raises(ValueError):
        article_module().load_articles(path)


@pytest.mark.parametrize("mode,articles", [
    ("eos", ARTICLES), ("mixed", ARTICLES), ("no-bos", (b"x",)),
    ("bos", (b"bad\x00text\n",)), ("bos", (b"bad\xff\n",)),
    ("fail-tokenize", ARTICLES), ("fail-vocab", ARTICLES),
], ids=["appended-eos", "inconsistent-bos", "one-token", "nul", "invalid-utf8", "tokenizer-exit", "vocabulary-exit"])
def test_prepare_errors_leave_no_completion_receipt(tmp_path, mode, articles):
    require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs", mode, articles)
    out = tmp_path / "prepared"
    result = run_cli("prepare_article_corpus", prepare_args(inputs, out), tmp_path)
    assert result.returncode == 1, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1, result.stderr
    assert "Traceback" not in result.stderr
    assert not (out / "corpus_articles.json").exists()


def test_prepare_refuses_existing_output_without_writes(tmp_path):
    require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs")
    out = tmp_path / "prepared"
    out.mkdir()
    (out / "evidence").write_bytes(b"preserve")
    result = run_cli("prepare_article_corpus", prepare_args(inputs, out), tmp_path)
    assert result.returncode == 1, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {"evidence": b"preserve"}
    assert not (inputs["corpus"].parent / "native-calls.jsonl").exists()


@pytest.mark.parametrize("change", ["gap", "corpus-sha256"])
def test_prepare_refuses_invalid_article_index(tmp_path, change):
    require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs")
    index = json.loads(inputs["article_index"].read_text())
    if change == "gap":
        index["articles"][1]["byte_start"] += 1
    else:
        index["corpus_sha256"] = "0" * 64
    inputs["article_index"].write_text(json.dumps(index))
    out = tmp_path / "prepared"
    result = run_cli("prepare_article_corpus", prepare_args(inputs, out), tmp_path)
    assert result.returncode == 1, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert not (out / "corpus_articles.json").exists()


def test_completion_receipt_is_not_written_before_dataset_save(tmp_path, monkeypatch, capsys):
    script = require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs")
    out = tmp_path / "prepared"
    from datasets import Dataset

    def fail_save(self, *args, **kwargs):
        assert not (out / "corpus_articles.json").exists()
        raise OSError("fixture dataset save failure")

    monkeypatch.setattr(Dataset, "save_to_disk", fail_save)
    for key, value in offline_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(script), *prepare_args(inputs, out)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 1
    assert "fixture dataset save failure" in capsys.readouterr().err
    assert not (out / "corpus_articles.json").exists()


@pytest.mark.parametrize("case", ["missing-index", "unknown-option", "uppercase-name", "path-name"])
def test_prepare_usage_errors_exit_two(tmp_path, case):
    require_cli("prepare_article_corpus")
    inputs = make_prepare_inputs(tmp_path / "inputs")
    if case == "missing-index":
        inputs.pop("article_index")
    name = {"uppercase-name": "Corpus", "path-name": "../corpus"}.get(case, "pg-full-rss-article")
    out = tmp_path / "prepared"
    args = prepare_args(inputs, out, name)
    if case == "unknown-option":
        args.append("--unknown-option")
    result = run_cli("prepare_article_corpus", args, tmp_path)
    assert result.returncode == 2, result.stderr
    assert not out.exists()
