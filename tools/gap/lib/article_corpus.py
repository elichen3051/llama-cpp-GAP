"""Per-article text corpora: one sequence per article, every token after position 0 scored.

The classic text bridge (lib/text_corpus.py) cuts one concatenated token stream into 512-token windows and scores
their second half. This protocol tokenizes every article on its own, with the reference vocabulary's native BOS rule
(as llama-perplexity: BOS only when the vocabulary adds one, nothing else prepended), and scores tokens 1 .. L-1 of
each article through the generic collect_llm_kld.py path (n_prefill = 1).
"""

import json
import re

from lib.reference_dataset import canonical_json
from lib.text_corpus import digest_json, token_digest

SCHEMA = "company-article-corpus-v1"
CATEGORY = "corpus-article"
FIXED_PROTOCOL = {
    "schema": SCHEMA, "article_bytes": "exact-span", "escape": False, "parse_special": False,
    "bos_policy": "vocabulary-add-bos", "n_prefill": 1, "scoring": "every-token-after-position-0",
}
PROTOCOL_KEYS = {*FIXED_PROTOCOL, "corpus_name", "corpus_sha256", "corpus_bytes", "article_index_sha256",
                 "articles_sha256", "bos_id", "vocabulary", "tokenizer_binary_sha256"}
ARTICLE_KEYS = {"index", "id", "source_id", "title", "byte_start", "byte_end_exclusive", "sha256",
                "n_tokens", "n_targets", "tokens_sha256", "targets_sha256"}
RECEIPT_KEYS = {"protocol", "articles", "total_tokens", "total_targets"}
VOCABULARY_KEYS = {"scheme", "size", "type", "mapping", "attributes"}
HEX = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

__all__ = ["SCHEMA", "article_spans", "token_digest", "bos_split", "article_row", "validate_article_row", "load_articles",
           "row_id", "articles_digest", "validate_protocol"]


def _int(value):
    return type(value) is int


def _hex(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


def article_spans(raw, index):
    """Exact byte spans of the articles of an index ({"articles" | "items": [...]}) that partition `raw`."""
    import hashlib

    if not isinstance(index, dict):
        raise ValueError("article index must be a JSON object")
    entries = index["articles"] if "articles" in index else index.get("items")
    if not isinstance(entries, list) or not entries:
        raise ValueError("article index needs a non-empty 'articles' or 'items' list")
    expected = index.get("corpus_sha256")
    if expected is not None and expected != hashlib.sha256(raw).hexdigest():
        raise ValueError("article index corpus SHA256 does not match the corpus")
    spans, cursor = [], 0
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"article {position}: entry must be a JSON object")
        start, end = entry.get("byte_start"), entry.get("byte_end_exclusive")
        if not (_int(start) and _int(end)) or start != cursor or not start < end <= len(raw):
            raise ValueError(f"article {position}: spans must partition the corpus bytes in order, without gaps or empty articles")
        try:
            raw[start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"article {position}: its bytes are not valid UTF-8 on their own (a span splits a character)") from error
        source = entry["id"] if "id" in entry else entry.get("index", position)
        spans.append({"source_id": str(source), "title": str(entry.get("title", entry.get("rss_title", ""))),
                      "byte_start": start, "byte_end_exclusive": end,
                      "sha256": hashlib.sha256(raw[start:end]).hexdigest()})
        cursor = end
    if cursor != len(raw):
        raise ValueError("article spans do not cover the whole corpus")
    return spans


def bos_split(with_bos, without_bos):
    """The BOS id when the native tokenizer added exactly one leading token, None when it added nothing."""
    with_bos, without_bos = list(with_bos), list(without_bos)
    if with_bos == without_bos:
        return None
    if len(with_bos) == len(without_bos) + 1 and with_bos[1:] == without_bos:
        return with_bos[0]
    raise ValueError("native tokenizer output differs by more than one leading BOS (an appended EOS is not allowed)")


def articles_digest(articles):
    """SHA256 over the article records without their ids (the ids embed the protocol digest, which covers this one)."""
    return digest_json([{key: value for key, value in record.items() if key != "id"} for record in articles])


def row_id(protocol, index):
    return f"{protocol['corpus_name']}-{digest_json(protocol)[:16]}-{index:04d}"


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS:
        raise ValueError("article protocol has missing or unexpected fields")
    if any(protocol[key] != value or type(protocol[key]) is not type(value) for key, value in FIXED_PROTOCOL.items()):
        raise ValueError("article protocol fixed fields changed")
    if not isinstance(protocol["corpus_name"], str) or not NAME.fullmatch(protocol["corpus_name"]):
        raise ValueError("invalid article protocol corpus_name")
    if not all(_hex(protocol[key]) for key in ("corpus_sha256", "article_index_sha256", "articles_sha256", "tokenizer_binary_sha256")):
        raise ValueError("invalid article protocol digest")
    if not _int(protocol["corpus_bytes"]) or protocol["corpus_bytes"] < 1:
        raise ValueError("invalid article protocol corpus_bytes")
    vocabulary = protocol["vocabulary"]
    if (not isinstance(vocabulary, dict) or set(vocabulary) != VOCABULARY_KEYS
            or vocabulary["scheme"] != "llama-vocabulary-sha256-v1" or not _int(vocabulary["size"]) or vocabulary["size"] < 1
            or not _int(vocabulary["type"]) or not _hex(vocabulary["mapping"]) or not _hex(vocabulary["attributes"])):
        raise ValueError("invalid article protocol vocabulary identity")
    bos = protocol["bos_id"]
    if bos is not None and (not _int(bos) or not 0 <= bos < vocabulary["size"]):
        raise ValueError("invalid article protocol bos_id")
    return protocol


def _validate_record(protocol, record):
    if not isinstance(record, dict) or set(record) != ARTICLE_KEYS:
        raise ValueError("article record has missing or unexpected fields")
    if not all(_int(record[key]) for key in ("index", "byte_start", "byte_end_exclusive", "n_tokens", "n_targets")):
        raise ValueError("article record integer fields must be integers")
    if record["index"] < 0 or record["id"] != row_id(protocol, record["index"]):
        raise ValueError(f"article {record['index']}: id does not match the protocol")
    if record["n_tokens"] < 2 or record["n_targets"] != record["n_tokens"] - 1:
        raise ValueError(f"article {record['index']}: needs n_targets = n_tokens - 1 >= 1")
    if not all(_hex(record[key]) for key in ("sha256", "tokens_sha256", "targets_sha256")):
        raise ValueError(f"article {record['index']}: invalid digest")
    if not isinstance(record["source_id"], str) or not isinstance(record["title"], str):
        raise ValueError(f"article {record['index']}: source_id and title must be strings")
    return record


def article_row(protocol, article_record, tokens):
    """The dataset row of one article (collect_llm_kld.py generic path: n_prefill = 1, every later token scored)."""
    tokens = [int(token) for token in tokens]
    row = {
        "id": article_record["id"], "source": protocol["corpus_name"], "category": CATEGORY,
        "question": "", "generated_texts": "", "seed": 0,
        "input_ids": tokens, "input_tokens_len": len(tokens), "n_prefill_tokens": 1,
        "generated_tokens_len": len(tokens) - 1, "labels": [-100] + tokens[1:],
        "article_protocol": canonical_json(protocol), "corpus_article": canonical_json(article_record),
    }
    validate_article_row(row)
    return row


def validate_article_row(row):
    try:
        protocol = validate_protocol(json.loads(row["article_protocol"]))
        record = _validate_record(protocol, json.loads(row["corpus_article"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"article row provenance is malformed: {error}") from error
    ids = row.get("input_ids")
    if not isinstance(ids, list) or len(ids) < 2 or not all(_int(token) for token in ids):
        raise ValueError("article row needs at least two integer tokens")
    if any(not 0 <= token < protocol["vocabulary"]["size"] for token in ids):
        raise ValueError("article row token outside the vocabulary")
    if protocol["bos_id"] is not None and ids[0] != protocol["bos_id"]:
        raise ValueError("article row does not start with the vocabulary BOS")
    expected = {
        "id": record["id"], "source": protocol["corpus_name"], "category": CATEGORY, "question": "", "generated_texts": "",
        "seed": 0, "input_tokens_len": len(ids), "n_prefill_tokens": 1, "generated_tokens_len": len(ids) - 1,
        "labels": [-100] + ids[1:],
    }
    if any(row.get(key) != value or type(row.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("article row columns do not match its tokens and provenance")
    if (record["n_tokens"] != len(ids) or record["tokens_sha256"] != token_digest(ids)
            or record["targets_sha256"] != token_digest(ids[1:])):
        raise ValueError("article row tokens do not match the article record")
    return protocol, record


def load_articles(path):
    """Read and validate a corpus_articles.json receipt."""
    from pathlib import Path

    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not JSON: {error}") from error
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise ValueError(f"{path}: receipt has missing or unexpected fields")
    protocol = validate_protocol(receipt["protocol"])
    articles = receipt["articles"]
    if not isinstance(articles, list) or not articles:
        raise ValueError(f"{path}: no articles")
    cursor = 0
    for position, record in enumerate(articles):
        _validate_record(protocol, record)
        if record["index"] != position or record["byte_start"] != cursor or record["byte_end_exclusive"] <= cursor:
            raise ValueError(f"{path}: article {position} is out of order or does not continue the previous span")
        cursor = record["byte_end_exclusive"]
    if cursor != protocol["corpus_bytes"]:
        raise ValueError(f"{path}: article spans do not cover the corpus")
    if articles_digest(articles) != protocol["articles_sha256"]:
        raise ValueError(f"{path}: articles_sha256 does not match the article records")
    if (receipt["total_tokens"] != sum(record["n_tokens"] for record in articles)
            or receipt["total_targets"] != sum(record["n_targets"] for record in articles)):
        raise ValueError(f"{path}: totals do not match the article records")
    return receipt
