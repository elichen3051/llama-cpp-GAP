"""LEGACY_CORPUS_DIGEST=1 accepts corpus maps whose stored digests predate a label rename; structure stays strict."""
import hashlib
import json

import pytest

from lib import text_corpus
from lib.text_corpus import SCHEMA, digest_json, load_corpus_map


def _hex(label):
    return hashlib.sha256(label.encode()).hexdigest()


def make_protocol(schema=SCHEMA, windows=4):
    return {
        "schema": schema, "corpus_name": "unit-test",
        "corpus_sha256": _hex("corpus"), "effective_text_sha256": _hex("effective"),
        "strip_single_final_newline": True, "escape": False, "parse_special": False,
        "stream_tokens": windows * 512 + 7, "stream_sha256": _hex("stream"),
        "vocabulary": {"scheme": "llama-vocabulary-sha256-v1", "size": 100, "type": 2,
                       "mapping": _hex("mapping"), "attributes": _hex("attributes")},
        "bos_id": None, "bos_policy": "replace-window-position-zero",
        "window_size": 512, "stride": 512, "n_prefill": 257, "targets_per_window": 255,
        "tail_policy": "drop-incomplete-window", "articles_sha256": _hex("articles"),
        "article_boundary_rule": "longest-stable-prefix; first boundary-affected token belongs to next article",
        "tokenizer_binary_sha256": _hex("tokenizer"),
    }


def make_windows(hashed_protocol, count=4):
    digest = digest_json(hashed_protocol)
    return [{"id": f"unit-test-{digest[:16]}-{i:06d}", "index": i, "offset": i * 512,
             "protocol_sha256": digest, "tokens_sha256": _hex(f"tokens{i}"),
             "targets_sha256": _hex(f"targets{i}"), "target_article_ids": [0] * 255}
            for i in range(count)]


def write_collection(tmp_path, protocol, windows, meta_digest_source):
    corpus = {"protocol": protocol, "windows": windows}
    (tmp_path / "corpus_windows.json").write_text(json.dumps(corpus))
    meta = {"perplexity_window": True, "corpus_protocol": protocol,
            "corpus_windows_sha256": digest_json(meta_digest_source)}
    return meta


@pytest.fixture(autouse=True)
def no_flag(monkeypatch):
    monkeypatch.delenv(text_corpus.LEGACY_DIGEST_ENV, raising=False)
    text_corpus._legacy_notice_shown = False


def test_current_collection_loads_without_flag(tmp_path):
    protocol = make_protocol()
    windows = make_windows(protocol)
    meta = write_collection(tmp_path, protocol, windows, {"protocol": protocol, "windows": windows})
    loaded, result = load_corpus_map(tmp_path, meta)
    assert loaded == protocol and len(result) == 4


def renamed_collection(tmp_path):
    """Digests computed over the original label, protocol renamed afterwards (the published collections)."""
    original = make_protocol(schema="original-perplexity-corpus-v1")
    renamed = make_protocol()
    windows = make_windows(original)
    meta = write_collection(tmp_path, renamed, windows, {"protocol": original, "windows": windows})
    return renamed, windows, meta


def test_renamed_collection_rejected_without_flag(tmp_path):
    _, _, meta = renamed_collection(tmp_path)
    with pytest.raises(ValueError, match="does not match collect_meta"):
        load_corpus_map(tmp_path, meta)


def test_renamed_collection_accepted_with_flag(tmp_path, monkeypatch, capsys):
    renamed, windows, meta = renamed_collection(tmp_path)
    monkeypatch.setenv(text_corpus.LEGACY_DIGEST_ENV, "1")
    loaded, result = load_corpus_map(tmp_path, meta)
    assert loaded == renamed
    assert list(result) == [f"{i:03d}_{w['id']}" for i, w in enumerate(windows)]
    assert text_corpus.LEGACY_DIGEST_ENV in capsys.readouterr().err


@pytest.mark.parametrize("corrupt", ["order", "id", "digest_format", "mixed_digest"])
def test_flag_keeps_structural_checks(tmp_path, monkeypatch, corrupt):
    renamed, windows, meta = renamed_collection(tmp_path)
    monkeypatch.setenv(text_corpus.LEGACY_DIGEST_ENV, "1")
    if corrupt == "order":
        windows[1], windows[2] = windows[2], windows[1]
    elif corrupt == "id":
        windows[3]["id"] = "unit-test-0000000000000000-000003"
    elif corrupt == "digest_format":
        for w in windows:
            w["protocol_sha256"] = "not-a-digest"
    elif corrupt == "mixed_digest":
        windows[2]["protocol_sha256"] = _hex("another protocol")
    (tmp_path / "corpus_windows.json").write_text(json.dumps({"protocol": renamed, "windows": windows}))
    with pytest.raises(ValueError):
        load_corpus_map(tmp_path, meta)


def test_flag_requires_well_formed_meta_digest(tmp_path, monkeypatch):
    renamed, windows, meta = renamed_collection(tmp_path)
    meta["corpus_windows_sha256"] = "corrupt"
    monkeypatch.setenv(text_corpus.LEGACY_DIGEST_ENV, "1")
    with pytest.raises(ValueError, match="does not match collect_meta"):
        load_corpus_map(tmp_path, meta)
