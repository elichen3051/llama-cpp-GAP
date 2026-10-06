"""Derived teacher-forcing corpora: one model's reference answer replayed under another model's native prompt.

Each derived row keeps the tested (target) model's own prompt, images and prefix token IDs from its native
reference row, and replaces the answer:

  cross-family  source reasoning/answer text rewritten in the target's thinking format and re-tokenized
                with the target tokenizer (lib.thinking_trajectory, llama-trajectory)
  same-family   source answer token IDs copied unchanged; source and target vocabularies must be identical

`corpus_protocol` records how the row was made; validate_row() checks a row against it. Rows from the
earlier campaign scripts (LEGACY_SCHEMAS) get the structural checks only.
"""

import hashlib
import json
from pathlib import Path

from lib.reference_dataset import canonical_json as canonical
from lib.thinking_trajectory import text_sha, validate_trajectory

SCHEMA_CROSS_FAMILY = "gap-cross-family-thinking-v1"
SCHEMA_SAME_FAMILY = "gap-same-family-native-tokens-v1"
LEGACY_SCHEMAS = ("cross-family-teacher-forcing-v1", "cross-family-thinking-v1", "cross-family-instruct-new-corpora-v2",
                  "cross-family-thinking-new-corpora-v2", "same-family-native-tokens-v1")
SCHEMAS = (SCHEMA_CROSS_FAMILY, SCHEMA_SAME_FAMILY) + LEGACY_SCHEMAS
SAME_FAMILY_SCHEMAS = (SCHEMA_SAME_FAMILY, "same-family-native-tokens-v1")

# Columns copied from the target's native reference row.
TARGET_COLUMNS = ("id", "item_id", "source", "question", "images", "num_images", "llamacpp_prompt_string",
                  "llamacpp_prompt_layout", "llamacpp_n_past_prefill", "llamacpp_add_special", "image_bytes_sha256")


def is_derived_row(row):
    return "corpus_protocol" in row and row.get("generation_engine") in SCHEMAS


def _check_layout(row, ids, n, vocabulary):
    layout = json.loads(row["llamacpp_prompt_layout"])
    if layout["n_tokens"] != n or layout["n_pos"] != row["llamacpp_n_past_prefill"]:
        raise ValueError("target prefix layout size differs")
    offset = positions = 0
    for chunk in layout["chunks"]:
        count = chunk["n_tokens"]
        if chunk["start"] != offset:
            raise ValueError("non-contiguous prefix layout")
        if chunk["type"] == "text":
            expected = chunk["tokens"]
            if len(expected) != count or any(t < 0 or t >= vocabulary["size"] for t in expected):
                raise ValueError("invalid target prompt tokens")
        elif chunk["type"] == "image":
            expected = [-1] * count
        else:
            raise ValueError("unsupported prompt chunk")
        if ids[offset:offset + count] != expected:
            raise ValueError("prefix token IDs disagree with the native layout")
        offset += count
        positions += chunk["n_pos"]
    if offset != n or positions != layout["n_pos"]:
        raise ValueError("layout totals differ")
    return layout


def validate_row(row):
    """Check a derived row against its corpus_protocol; returns the protocol."""
    p = json.loads(row["corpus_protocol"])
    if p.get("schema") not in SCHEMAS or row["generation_engine"] != p["schema"]:
        raise ValueError("unknown derived corpus schema")
    ids = list(row["input_ids"])
    n = row["n_prefill_tokens"]
    if not 0 < n < len(ids) or row["input_tokens_len"] != len(ids) or row["generated_tokens_len"] != len(ids) - n:
        raise ValueError("invalid prompt/answer boundary")
    if list(row["labels"]) != [-100] * n + ids[n:]:
        raise ValueError("labels differ from the answer token IDs")
    vocabulary = json.loads(row["target_vocabulary"])
    if vocabulary != p["target_vocabulary"] or any(t < 0 or t >= vocabulary["size"] for t in ids[n:]):
        raise ValueError("invalid target vocabulary or answer IDs")
    if text_sha(row["generated_texts"]) != p["answer_sha256"]:
        raise ValueError("answer text changed")
    if text_sha(row["llamacpp_prompt_string"]) != p["prompt_sha256"]:
        raise ValueError("target chat prompt changed")
    if text_sha(canonical(ids[:n])) != p["prefix_ids_sha256"]:
        raise ValueError("target prefix token IDs changed")
    if text_sha(canonical(ids[n:])) != p["answer_ids_sha256"]:
        raise ValueError("answer token IDs changed")
    layout = _check_layout(row, ids, n, vocabulary)
    if text_sha(canonical(layout)) != p["layout_sha256"]:
        raise ValueError("target image/text layout changed")
    if row["llamacpp_prompt_string"].count("<__media__>") != row["num_images"]:
        raise ValueError("wrong image marker count")
    if list(row["image_bytes_sha256"]) != p["image_bytes_sha256"]:
        raise ValueError("image identity changed")
    if p["schema"] in SAME_FAMILY_SCHEMAS:
        if p["source_vocabulary"] != vocabulary or not p["tokens_transferred_without_retokenization"]:
            raise ValueError("native token transfer requires identical source and target vocabulary")
        if text_sha(canonical(ids[n:])) != p["source_ids_sha256"]:
            raise ValueError("source reasoning or answer tokens changed")
    elif p["schema"] == SCHEMA_CROSS_FAMILY:
        if p["target_prompt"] != row["llamacpp_prompt_string"]:
            raise ValueError("trajectory target prompt changed")
        validate_trajectory(p, row["generated_texts"], ids[n:])
    return p


def derived_row(peer, answer_ids, text, protocol, vocabulary):
    """A derived row: the target's native prompt and prefix, then the derived answer tokens."""
    n = peer["n_prefill_tokens"]
    prefix = list(peer["input_ids"][:n])
    ids = prefix + list(answer_ids)
    protocol = {**protocol, "answer_sha256": text_sha(text), "answer_ids_sha256": text_sha(canonical(list(answer_ids))),
                "prefix_ids_sha256": text_sha(canonical(prefix)), "prompt_sha256": text_sha(peer["llamacpp_prompt_string"]),
                "layout_sha256": text_sha(canonical(json.loads(peer["llamacpp_prompt_layout"]))),
                "target_vocabulary": vocabulary, "image_bytes_sha256": list(peer["image_bytes_sha256"])}
    row = {key: peer[key] for key in TARGET_COLUMNS}
    row.update(input_ids=ids, input_tokens_len=len(ids), n_prefill_tokens=n, generated_tokens_len=len(answer_ids),
               labels=[-100] * n + list(answer_ids), generated_texts=text, generation_engine=protocol["schema"],
               target_vocabulary=canonical(vocabulary), corpus_protocol=canonical(protocol))
    validate_row(row)
    return row


def derived_features():
    from datasets import Features, Image, Sequence, Value
    features = {k: Value("string") for k in ("id", "item_id", "source", "question", "llamacpp_prompt_string",
                                              "llamacpp_prompt_layout", "generated_texts", "generation_engine",
                                              "target_vocabulary", "corpus_protocol")}
    features.update({k: Value("int64") for k in ("num_images", "llamacpp_n_past_prefill", "input_tokens_len",
                                                 "n_prefill_tokens", "generated_tokens_len")})
    features.update({k: Sequence(Value("int64")) for k in ("input_ids", "labels")})
    features.update(images=Sequence(Image(decode=False)), image_bytes_sha256=Sequence(Value("string")),
                    llamacpp_add_special=Value("bool"))
    return Features(features)


def save_dataset(rows, directory):
    from datasets import Dataset, load_from_disk
    dataset = Dataset.from_list(rows, features=derived_features())
    dataset.save_to_disk(str(directory))
    saved = load_from_disk(str(directory))
    if not saved.data.table.equals(dataset.data.table):
        raise ValueError("saved dataset differs")
    for row in saved:
        validate_row(row)
    return saved


def load_reference_rows(spec, subset, cache_dir=None):
    """Native reference rows in stored order, images as encoded bytes.

    spec: a save_to_disk directory, a Parquet file, or hf://<repo>@<revision> (reads <subset>/train-00000-of-00001.parquet)."""
    from datasets import DatasetDict, load_dataset, load_from_disk
    from lib.reference_dataset import reference_features
    if spec.startswith("hf://"):
        from lib.reference_freeze import _download_pinned
        repo, _, revision = spec[len("hf://"):].partition("@")
        if not repo or not revision:
            raise ValueError("hf:// input needs <repo>@<revision>")
        spec = str(_download_pinned(repo, revision, f"{subset}/train-00000-of-00001.parquet", cache_dir))
    path = Path(spec)
    if path.is_dir():
        dataset = load_from_disk(str(path))
        return dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    if path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train", features=reference_features(),
                            cache_dir=cache_dir)
    raise ValueError(f"unsupported reference input: {spec}")


def write_prep(row, out_dir, raw_images):
    """Scorer inputs for one derived row (tokens.bin, images, formatted_chat.txt, meta.json); returns the meta dict."""
    import numpy as np
    from cli.prep_vlm_score_from_hf import image_extension
    protocol = validate_row(row)
    if [hashlib.sha256(data).hexdigest() for data in raw_images] != protocol["image_bytes_sha256"]:
        raise ValueError("raw image hashes differ")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, data in enumerate(raw_images):
        name = f"img_{i}.{image_extension(data)}"
        (out_dir / name).write_bytes(data)
        names.append(name)
    (out_dir / "formatted_chat.txt").write_text(row["llamacpp_prompt_string"], encoding="utf-8")
    np.asarray(row["input_ids"], dtype="<i4").tofile(out_dir / "tokens.bin")
    meta = {"reference_vocabulary": protocol["target_vocabulary"], "num_images": row["num_images"],
            "n_prefill": row["n_prefill_tokens"], "n_answer": row["generated_tokens_len"], "image_files": names,
            "n_past_expected": row["llamacpp_n_past_prefill"], "add_special": row["llamacpp_add_special"]}
    (out_dir / "meta.json").write_text(json.dumps({**meta, "derived_corpus_protocol": protocol}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
    return meta
