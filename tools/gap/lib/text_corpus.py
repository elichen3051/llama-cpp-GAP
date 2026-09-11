"""Frozen classic-perplexity windows and their corpus provenance."""

import hashlib
import json
import os
import re
import sys

import numpy as np

from lib.reference_dataset import canonical_json

SCHEMA = "company-perplexity-corpus-v1"


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


LEGACY_DIGEST_ENV = "LEGACY_CORPUS_DIGEST"
_legacy_notice_shown = False


def legacy_digests_accepted():
    """LEGACY_CORPUS_DIGEST=1 accepts stored protocol_sha256 / corpus_windows_sha256 values that no longer
    recompute from the protocol dict because identifier labels were renamed after collection (the
    published text-bridge collections). Structural checks stay strict: window order and offsets, ids
    carrying the stored digest prefix, token and target digests, metric shapes."""
    return os.environ.get(LEGACY_DIGEST_ENV) == "1"


def digest_matches(stored, value):
    """True when stored == digest_json(value), or when it is a well-formed digest accepted under the legacy flag."""
    global _legacy_notice_shown
    if stored == digest_json(value):
        return True
    if not legacy_digests_accepted() or not isinstance(stored, str) or not re.fullmatch(r"[0-9a-f]{64}", stored):
        return False
    if not _legacy_notice_shown:
        print(f"[text_corpus] {LEGACY_DIGEST_ENV}=1: accepting a stored corpus digest that does not recompute "
              "from the current protocol labels", file=sys.stderr)
        _legacy_notice_shown = True
    return True


def token_digest(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def validate_protocol(protocol):
    size = protocol.get("window_size")
    if (protocol.get("schema") != SCHEMA or type(size) is not int
            or size < 4 or size % 2 or protocol.get("stride") != size
            or protocol.get("n_prefill") != size // 2 + 1
            or protocol.get("targets_per_window") != size // 2 - 1
            or protocol.get("parse_special") is not False
            or protocol.get("escape") is not False
            or protocol.get("strip_single_final_newline") is not True
            or protocol.get("bos_policy") != "replace-window-position-zero"
            or protocol.get("tail_policy") != "drop-incomplete-window"):
        raise ValueError("invalid classic perplexity corpus protocol")
    for field in ("corpus_sha256", "effective_text_sha256", "stream_sha256", "articles_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(protocol.get(field, ""))):
            raise ValueError(f"invalid corpus protocol {field}")
    vocabulary = protocol.get("vocabulary", {})
    if (vocabulary.get("scheme") != "llama-vocabulary-sha256-v1"
            or type(vocabulary.get("size")) is not int or vocabulary["size"] < 1
            or type(vocabulary.get("type")) is not int
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(vocabulary.get(key, "")))
                   for key in ("mapping", "attributes"))):
        raise ValueError("invalid corpus vocabulary identity")
    bos = protocol.get("bos_id")
    if bos is not None and (type(bos) is not int or not 0 <= bos < vocabulary["size"]):
        raise ValueError("invalid corpus BOS token")
    count = protocol.get("stream_tokens")
    if type(count) is not int or count < 2 * size:
        raise ValueError("classic perplexity requires at least two full windows")
    return protocol


def validate_corpus_row(row):
    protocol = validate_protocol(json.loads(row["corpus_protocol"]))
    window = json.loads(row["corpus_window"])
    size = protocol["window_size"]
    index = window.get("index")
    if (type(index) is not int or not 0 <= index < protocol["stream_tokens"] // size
            or window.get("offset") != index * size
            or not digest_matches(window.get("protocol_sha256"), protocol)
            or window.get("id") != str(row["id"])):
        raise ValueError("corpus window identity mismatch")
    ids = row["input_ids"]
    prefill = protocol["n_prefill"]
    if (len(ids) != size or row["input_tokens_len"] != size
            or row["n_prefill_tokens"] != prefill
            or row["generated_tokens_len"] != size - prefill
            or row["labels"] != [-100] * prefill + ids[prefill:]
            or any(type(token) is not int or not 0 <= token < protocol["vocabulary"]["size"] for token in ids)
            or window.get("tokens_sha256") != token_digest(ids)
            or window.get("targets_sha256") != token_digest(ids[prefill:])):
        raise ValueError("corpus window tokens or scoring targets changed")
    if protocol["bos_id"] is not None and ids[0] != protocol["bos_id"]:
        raise ValueError("corpus window BOS replacement is missing")
    articles = window.get("target_article_ids")
    if (not isinstance(articles, list) or len(articles) != size - prefill
            or any(type(index) is not int or index < 0 for index in articles)):
        raise ValueError("each corpus scoring target requires an article index")
    return protocol, window


def corpus_row(protocol, tokens, index, target_article_ids):
    size = protocol["window_size"]
    prefill = protocol["n_prefill"]
    key = f"{protocol['corpus_name']}-{digest_json(protocol)[:16]}-{index:06d}"
    window = {
        "id": key, "index": index, "offset": index * size,
        "protocol_sha256": digest_json(protocol),
        "tokens_sha256": token_digest(tokens),
        "targets_sha256": token_digest(tokens[prefill:]),
        "target_article_ids": target_article_ids,
    }
    row = {
        "id": key, "source": protocol["corpus_name"], "category": "corpus-window",
        "question": "", "generated_texts": "", "seed": 0,
        "input_ids": tokens, "input_tokens_len": size,
        "n_prefill_tokens": prefill, "generated_tokens_len": size - prefill,
        "labels": [-100] * prefill + tokens[prefill:],
        "corpus_protocol": canonical_json(protocol), "corpus_window": canonical_json(window),
    }
    validate_corpus_row(row)
    return row


def validate_corpus_dataset(ds, size):
    windows = []
    protocol = None
    for index, row in enumerate(ds):
        current, window = validate_corpus_row(row)
        if protocol is None:
            protocol = current
        if current != protocol or window["index"] != index:
            raise ValueError("corpus dataset must contain all windows in their original order")
        windows.append(window)
    if (protocol is None or protocol["window_size"] != size
            or len(windows) != protocol["stream_tokens"] // size):
        raise ValueError("corpus dataset is incomplete or has the wrong window size")
    return {"protocol": protocol, "windows": windows}


def stable_prefix_boundary(full, prefix):
    """Assign the first token affected by an article cut to the next article."""
    n = min(len(full), len(prefix))
    different = np.flatnonzero(np.asarray(full[:n]) != np.asarray(prefix[:n]))
    return int(different[0]) if len(different) else n


def load_corpus_map(directory, meta):
    from pathlib import Path
    if not meta.get("perplexity_window") or not meta.get("corpus_protocol"):
        raise ValueError("corpus comparison requires an explicit full-window collection")
    path = Path(directory) / "corpus_windows.json"
    corpus = json.loads(path.read_text())
    protocol = validate_protocol(corpus["protocol"])
    if (protocol != meta["corpus_protocol"]
            or not digest_matches(meta.get("corpus_windows_sha256"), corpus)):
        raise ValueError(f"{path}: corpus identity does not match collect_meta.json")
    windows = corpus["windows"]
    if len(windows) != protocol["stream_tokens"] // protocol["window_size"]:
        raise ValueError("corpus window map is incomplete")
    result = {}
    for index, window in enumerate(windows):
        if (window["index"] != index or window["offset"] != index * protocol["window_size"]
                or window["protocol_sha256"] != windows[0]["protocol_sha256"]
                or window["id"] != f"{protocol['corpus_name']}-{window['protocol_sha256'][:16]}-{index:06d}"
                or not digest_matches(window["protocol_sha256"], protocol)):
            raise ValueError("corpus window map is out of order or uses a different protocol")
        result[f"{index:03d}_{window['id']}"] = window
    return protocol, result


def check_corpus_metrics(key, metrics, header, protocol, windows):
    window = windows.get(key)
    if window is None:
        raise ValueError(f"{key}: metric window is not in the frozen corpus map")
    if (header["npos"] != protocol["targets_per_window"]
            or header["n_prefill"] != protocol["n_prefill"]
            or header["n_past_actual"] != protocol["window_size"]
            or header["vocab"] != protocol["vocabulary"]["size"]
            or token_digest(metrics["target"]) != window["targets_sha256"]):
        raise ValueError(f"{key}: metric targets or runtime shape differ from the frozen corpus")
    if any(not np.isfinite(values).all() for values in metrics.values()):
        raise ValueError(f"{key}: non-finite corpus metric values")
    return window


class CorpusGroups:
    """Reduce validated window records into whole articles or fixed blocks."""

    def __init__(self, unit, block_windows):
        self.unit = unit
        self.block_windows = block_windows
        self.groups = {}
        self.n_windows = 0

    def add(self, window, a, b):
        self.n_windows += 1
        if self.unit == "article":
            assignments = np.asarray(window["target_article_ids"])
        else:
            assignments = np.full(len(a["target"]), window["index"] // self.block_windows)
        for group in np.unique(assignments):
            mask = assignments == group
            pair = self.groups.setdefault(int(group), ({}, {}))
            for output, records in zip(pair, (a, b)):
                for name, values in records.items():
                    output.setdefault(name, []).append(values[mask])

    def records(self):
        for group, pair in sorted(self.groups.items()):
            yield f"{self.unit}-{group:06d}", tuple({name: np.concatenate(parts) for name, parts in side.items()} for side in pair)
