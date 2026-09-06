"""Dataset storage for the GGUF-native reference producer; no HF tokenizer."""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from lib.reference_contract import GTContractError, validate_gt_row
from lib.model_files import model_files

SCHEMA_VERSION = "skymizer-reference-v2"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_images(row, column="images"):
    """Read original encoded bytes from a decode=False dataset column."""
    result = []
    for image in row.get(column) or []:
        if isinstance(image, bytes):
            data = image
        elif isinstance(image, dict):
            data = image.get("bytes")
            if data is None and image.get("path"):
                data = Path(image["path"]).read_bytes()
        else:
            raise ValueError("images must be encoded bytes; load the column with Image(decode=False)")
        if not data:
            raise ValueError("image has no encoded bytes")
        result.append(bytes(data))
    return result


def build_row(source, request, result, metadata, images):
    if result["id"] != request["id"]:
        raise ValueError("native result id does not match request")
    ids = result["input_ids"]
    n_pre = result["n_prefill_tokens"]
    image_chunks = [c for c in result["prompt_layout"]["chunks"] if c["type"] == "image"]
    # Legacy per_image_* columns describe native chunks, which may be tiles of one image.
    counts = [c["n_tokens"] for c in image_chunks]
    processor = {
        "engine": "llama.cpp",
        "image_min_tokens": metadata["image_min_tokens"],
        "image_max_tokens": metadata["image_max_tokens"],
        "image_token_budget_source": metadata["image_token_budget_source"],
        "llama_cpp_build": metadata["build_info"],
        "mmproj_sha256": metadata.get("mmproj_sha256"),
        "media_marker": metadata["media_marker"],
    }
    processor_json = canonical_json(processor)
    row = {
        "id": request["id"], "item_id": request["id"],
        "source": str(source.get("source", "")), "category": str(source.get("category", "")),
        "question": request["question"], "generated_texts": result["content"],
        "seed": result["sampling"]["seed"],
        "input_ids": ids, "input_tokens_len": len(ids), "n_prefill_tokens": n_pre,
        "generated_tokens_len": len(ids) - n_pre, "labels": [-100] * n_pre + ids[n_pre:],
        "finish_reason": result["finish_reason"], "truncated_by_cap": result["finish_reason"] == "length",
        "images": [{"bytes": raw, "path": None} for raw in images], "num_images": len(images),
        "sum_vision_tokens": sum(counts), "max_vision_tokens": max(counts, default=0),
        "generation_engine": "llama.cpp", "generation_backend": "llama-reference",
        "generation_schema_version": SCHEMA_VERSION,
        "generation_model_name_or_path": metadata["model_path"],
        "generation_llama_cpp_build": metadata["build_info"],
        "generation_metadata": canonical_json(metadata),
        "generation_request": canonical_json({k: v for k, v in request.items() if k != "images"}),
        "generation_sampling_params": canonical_json(result["sampling"]),
        "generation_decoding_stats": canonical_json(result.get("decoding_stats", {})),
        "generation_chat_template_kwargs": canonical_json(result["chat_template_kwargs"]),
        "generation_enable_thinking": result["enable_thinking"],
        "generation_token_logprobs": result["token_logprobs"],
        "image_processor_config": processor_json,
        "image_processor_config_hash": hashlib.sha256(processor_json.encode()).hexdigest(),
        "image_bytes_sha256": [hashlib.sha256(raw).hexdigest() for raw in images],
        "llamacpp_prompt_string": result["prompt"], "llamacpp_media_marker": metadata["media_marker"],
        "llamacpp_prompt_layout": canonical_json(result["prompt_layout"]),
        "llamacpp_n_past_prefill": result["n_past_prefill"], "llamacpp_tokens_evaluated": n_pre,
        "per_image_vision_token_counts": counts, "per_image_n_pos": [c["n_pos"] for c in image_chunks],
        "per_image_grid": [[c["grid_x"], c["grid_y"]] for c in image_chunks],
        "llamacpp_add_special": result["add_special"],
        "llamacpp_stripped_leading_bos": result["stripped_leading_bos"],
        "llamacpp_stop_type": result["stop_type"], "llamacpp_stopping_word": result["stopping_word"],
        "llamacpp_content": result["content"],
        "source_metadata": canonical_json({k: v for k, v in source.items() if k not in ("images", "image")}),
    }
    validate_reference_row(row)
    return row


def reference_provenance(row):
    if not row.get("generation_schema_version"):
        return {}
    if row["generation_schema_version"] != SCHEMA_VERSION:
        raise ValueError("native reference schema predates vocabulary identity; regenerate the reference")
    metadata = json.loads(row["generation_metadata"])
    vocabulary = metadata.get("vocabulary", {})
    import re
    if (vocabulary.get("scheme") != "llama-vocabulary-sha256-v1"
            or vocabulary.get("size") != metadata.get("vocab_size")
            or not isinstance(vocabulary.get("type"), int)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(vocabulary.get(k, "")))
                   for k in ("mapping", "attributes"))):
        raise ValueError("native reference has no valid target vocabulary identity")
    decoding = metadata.get("decoding", {})
    if (decoding.get("method") not in ("autoregressive", "mtp")
            or decoding.get("logprob_source") != "target_raw_logits"
            or decoding.get("token_source") != "target_accepted"):
        raise ValueError("reference must store accepted target tokens and raw target logprobs")
    if decoding["method"] == "mtp":
        mtp = decoding.get("mtp", {})
        files = metadata.get("model_files") if mtp.get("head_source") == "embedded" else mtp.get("head_files")
        if (mtp.get("head_source") not in ("embedded", "sidecar") or not files
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))) for f in files)
                or not isinstance(mtp.get("settings"), dict)):
            raise ValueError("MTP reference requires complete head provenance and generation settings")
    return {"reference_vocabulary": vocabulary, "reference_generation": metadata}


def validate_reference_row(row):
    violations = validate_gt_row(row)
    if row.get("generation_schema_version") != SCHEMA_VERSION:
        violations.append("unsupported native reference schema")
    logprobs = row.get("generation_token_logprobs")
    if (not isinstance(logprobs, (list, tuple))
            or len(logprobs) != row.get("generated_tokens_len")
            or any(not isinstance(value, Real) or isinstance(value, bool)
                   or not math.isfinite(value) or value > 0 for value in logprobs)):
        violations.append("native generation_token_logprobs must contain one finite non-positive number per generated token")
    if violations:
        raise GTContractError("native reference contract: " + "; ".join(violations))
    reference_provenance(row)
    layout = json.loads(row["llamacpp_prompt_layout"])
    if sum(c["n_pos"] for c in layout["chunks"]) != layout["n_pos"]:
        violations.append("layout position spans do not sum to n_past_prefill")
    image_chunks = [c for c in layout["chunks"] if c["type"] == "image"]
    if row["per_image_n_pos"] != [c["n_pos"] for c in image_chunks]:
        violations.append("per-image positions differ from layout")
    if row["per_image_grid"] != [[c["grid_x"], c["grid_y"]] for c in image_chunks]:
        violations.append("per-image grids differ from layout")
    for chunk in layout["chunks"]:
        if chunk["n_pos"] < 0:
            violations.append("negative layout position span")
        if chunk["type"] == "image":
            start = chunk["start"]
            if row["input_ids"][start:start + chunk["n_tokens"]] != [-1] * chunk["n_tokens"]:
                violations.append("native image spans must use LLAMA_TOKEN_NULL (-1)")
    meta = json.loads(row["generation_metadata"])
    vocab_size = meta["vocab_size"]
    text_ids = [t for c in layout["chunks"] if c["type"] == "text" for t in c["tokens"]]
    if any(t < 0 or t >= vocab_size for t in text_ids):
        violations.append("prompt text token outside GGUF vocabulary")
    if any(t < 0 or t >= vocab_size for t in row["input_ids"][row["n_prefill_tokens"]:]):
        violations.append("generated token outside GGUF vocabulary")
    images = raw_images(row)
    if [hashlib.sha256(data).hexdigest() for data in images] != row["image_bytes_sha256"]:
        violations.append("image bytes do not match recorded hashes")
    if violations:
        raise GTContractError("native reference contract: " + "; ".join(violations))


def reference_features():
    from datasets import Features, Image, Sequence, Value
    text = "id item_id source category question generated_texts finish_reason generation_engine generation_backend generation_schema_version generation_model_name_or_path generation_llama_cpp_build generation_metadata generation_request generation_sampling_params generation_decoding_stats generation_chat_template_kwargs image_processor_config image_processor_config_hash llamacpp_prompt_string llamacpp_media_marker llamacpp_prompt_layout llamacpp_stop_type llamacpp_stopping_word llamacpp_content source_metadata"
    ints = "seed input_tokens_len n_prefill_tokens generated_tokens_len num_images sum_vision_tokens max_vision_tokens llamacpp_n_past_prefill llamacpp_tokens_evaluated"
    flags = "truncated_by_cap generation_enable_thinking llamacpp_add_special llamacpp_stripped_leading_bos"
    features = {key: Value("string") for key in text.split()}
    features.update({key: Value("int64") for key in ints.split()})
    features.update({key: Value("bool") for key in flags.split()})
    for key in "input_ids labels per_image_vision_token_counts per_image_n_pos".split():
        features[key] = Sequence(Value("int64"))
    features["generation_token_logprobs"] = Sequence(Value("float64"))
    features["image_bytes_sha256"] = Sequence(Value("string"))
    features["per_image_grid"] = Sequence(Sequence(Value("int64")))
    features["images"] = Sequence(Image(decode=False))
    return Features(features)
