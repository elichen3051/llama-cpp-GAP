"""Freeze one immutable materialized tail400 config for local KLD collection."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from cli.publish_collect400 import check_manifest
from cli.upload_reference import DETECTOR, verify_model
from lib.reference_cohort import tail_cohort
from lib.reference_dataset import reference_features, sha256_file, validate_reference_row
from lib.reference_run import atomic_json
from lib.reference_study import reference_template

SCHEMA = "company-reference-freeze-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _selection(profiles_path, model, repo, revision, subset):
    _require(re.fullmatch(r"[0-9a-f]{40}", revision), "reference revision must be an exact Hub commit")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", model), "invalid reference model")
    _require(repo == f"user-company/{model}-collect-400", "reference repo does not match the model and tail400 cohort")
    profiles_path = Path(profiles_path)
    profiles = json.loads(profiles_path.read_text())
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)-tail-400-(ins|think)", subset)
    _require(match and match[1] in profiles["sources"] and profiles["cohort_size"] == 500,
             "tail400 requires a supported source and the parent500 profile")
    mode = "thinking" if match[2] == "think" else "instruct"
    reference_template(profiles["models"][model], mode)
    identity = {"schema": SCHEMA, "repo": repo, "revision": revision, "subset": subset, "split": "train",
                "model": model, "profiles_sha256": sha256_file(profiles_path), "dataset": "dataset"}
    return profiles, mode, identity


def _download_pinned(repo, revision, name, cache_dir, expected_sha256=None):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, cache_dir=cache_dir))
    if expected_sha256 is not None:
        _require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "invalid pinned Parquet SHA256")
        _require(sha256_file(path) == expected_sha256, "pinned Parquet SHA256 mismatch")
    else:
        info = get_hf_file_metadata(hf_hub_url(repo, name, repo_type="dataset", revision=revision))
        _require(info.commit_hash == revision, "Hub file does not resolve to the requested commit")
        raw = path.read_bytes()
        digest = (hashlib.sha256(raw).hexdigest() if len(info.etag or "") == 64 else
                  hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest())
        _require(digest == info.etag, f"pinned audit content mismatch: {name}")
    return path


def _profile_metadata(metadata, profiles, model, mode):
    profile = profiles["models"][model]
    verify_model(metadata, profile)
    template = reference_template(profile, mode)
    defaults = {"chat_template_source": "gguf", "jinja": True, **template, "image_min_tokens": -1,
                "image_max_tokens": -1, "image_token_budget_source": "mtmd_init_params"}
    _require(all(metadata.get(k) == v for k, v in defaults.items()), "reference template or image policy differs")
    _require(metadata["sampling"] == profile["effective_sampling"][mode]
             and metadata["sampling"]["seed"] == profiles["seed"], "reference sampling differs")
    _require(metadata["requested_enable_thinking"] is (mode == "thinking") and not metadata.get("thinking_column"),
             "reference semantic mode differs")
    _require(metadata["n_predict"] == profiles["generation_caps"][mode], "reference generation cap differs")
    _require(metadata["repetition_detector"] == DETECTOR, "reference repetition policy differs")
    runtime = profile["runtime"]["pro6000"][mode]
    _require(all(metadata[key] == runtime[setting] for key, setting in
                 (("n_ctx_per_seq", "ctx"), ("n_batch", "batch"), ("n_ubatch", "ubatch"))),
             "reference generation runtime differs")
    _require(metadata["n_threads"] == metadata["n_threads_batch"] == runtime["threads"] == 8,
             "reference generation threads differ")
    _require(metadata["total_slots"] == runtime.get("parallel", 1)
             and metadata["n_ctx"] == metadata["n_ctx_per_seq"] * metadata["total_slots"],
             "reference generation context/slots differ")
    return template


def _validate_audits(out, profiles, model, repo, subset, mode):
    plan = json.loads((out / "composition.json").read_text())
    _require(plan.get("schema") == "company-collect400-materialized-v1", "reference must use the materialized tail400 schema")
    _require(plan["destination"] == repo and plan["parent_repo"] == f"user-company/{model}-collect-500"
             and plan["pilot_repo"] == f"user-company/{model}-pilot"
             and all(re.fullmatch(r"[0-9a-f]{40}", plan[k]) for k in ("parent_revision", "pilot_revision")),
             "invalid immutable composition identity")
    entries = [entry for entry in plan["configs"] if entry["config"] == subset]
    _require(len(entries) == 1, "composition must contain the selected config exactly once")
    entry = entries[0]
    _require(entry["parent_config"] == subset.replace("-tail-400-", "-subsample-500-")
             and entry["pilot_config"] == subset.replace("-tail-400-", "-subsample-100-")
             and entry["path"] == f"{subset}/train-00000-of-00001.parquet"
             and entry["source_path"] == f"{entry['parent_config']}/train-00000-of-00001.parquet",
             "materialized config/Parquet mapping differs")
    metadata = {}
    for label in ("parent", "pilot"):
        base = out / "audit" / subset / label
        manifest = json.loads((base / "manifest.json").read_text())
        check_manifest(manifest, plan[label + "_repo"], entry[label + "_config"])
        meta = json.loads((base / "metadata.json").read_text())
        _require(manifest == entry[label + "_audit"]
                 and sha256_file(base / "metadata.json") == manifest["metadata_sha256"] == manifest["audit_sha256"]["metadata.json"]
                 and meta["cohort"] == manifest["cohort"] and meta["dataset_source"] == manifest["source"]
                 and meta["n_predict"] == entry[label + "_generation_cap"], "native audit/metadata identity differs")
        verify_model(meta, profiles["models"][model])
        source = meta["dataset_source"]
        _require(source["path"] == profiles["dataset"]["repo"] and source["revision"] == profiles["dataset"]["revision"],
                 "prepared source identity differs from profile")
        metadata[label] = meta
    parent = metadata["parent"]
    tail = tail_cohort(metadata["pilot"]["cohort"], parent["cohort"])
    _require(tail == entry["cohort"] and tail["eligible"] > 0, "materialized tail differs from the native requested-ID cohort")
    _require(entry["source"] == parent["dataset_source"]
             and entry["source_parquet_sha256"] == entry["parent_audit"]["parquet_sha256"],
             "parent source/Parquet provenance differs")
    _require(re.fullmatch(r"[0-9a-f]{64}", entry["parquet_sha256"]), "invalid materialized Parquet SHA256")
    template = _profile_metadata(parent, profiles, model, mode)
    return plan, entry, parent, template


def _validate_rows(dataset, entry, metadata, template, mode):
    _require(dataset.features == reference_features(), "frozen dataset features differ from the native reference schema")
    _require(list(dataset["id"]) == entry["cohort"]["eligible_ids"], "dataset IDs/order differ from the eligible tail")
    kwargs = {**template["chat_template_kwargs"], "enable_thinking": "true" if mode == "thinking" else "false"}
    for row in dataset:
        _require(all(isinstance(image, dict) and image.get("bytes") for image in row["images"]),
                 "frozen reference images must contain their original encoded bytes")
        validate_reference_row(row)
        request = json.loads(row["generation_request"])
        _require(0 < row["generated_tokens_len"] <= metadata["n_predict"], "row generated length exceeds the declared cap or is empty")
        _require(request.get("enable_thinking") is (mode == "thinking"),
                 "row request thinking mode differs")
        _require(json.loads(row["generation_metadata"]) == metadata, "row generation metadata differs from parent500")
        _require(row["generation_enable_thinking"] is (mode == "thinking")
                 and json.loads(row["generation_sampling_params"]) == metadata["sampling"]
                 and json.loads(row["generation_chat_template_kwargs"]) == kwargs
                 and request["id"] == row["id"]
                 and request.get("system_prompt", metadata["system_prompt"]) == template["system_prompt"],
                 "row request/mode/sampling/template differs")


def _file_hashes(out):
    _require(out.is_dir() and not out.is_symlink(), "freeze must be a local directory without symlinks")
    hashes = {}
    for path in sorted(out.rglob("*")):
        _require(not path.is_symlink(), "freeze must not contain symlinks")
        if path.is_file() and path != out / "freeze-receipt.json":
            hashes[path.relative_to(out).as_posix()] = sha256_file(path)
    return hashes


def validate_frozen_reference(out, profiles_path, model, repo, revision, subset):
    """Verify a complete local freeze without accessing the Hub or the deleted parent."""
    from datasets import load_from_disk
    out = Path(out)
    profiles, mode, identity = _selection(profiles_path, model, repo, revision, subset)
    receipt_path = out / "freeze-receipt.json"
    _require(receipt_path.is_file(), "incomplete reference freeze; keep it and choose a fresh output path")
    receipt = json.loads(receipt_path.read_text())
    _require(all(receipt.get(k) == v for k, v in identity.items()) and receipt.get("all_rows_validated") is True,
             "reference freeze selection/profile/revision differs")
    hashes = _file_hashes(out)
    _require(hashes == receipt["files_sha256"], "frozen reference file SHA256 inventory differs")
    plan, entry, metadata, template = _validate_audits(out, profiles, model, repo, subset, mode)
    _require(receipt["parquet_sha256"] == entry["parquet_sha256"]
             and receipt["composition_sha256"] == hashes["composition.json"]
             and receipt["cohort"] == entry["cohort"] and receipt["rows"] == entry["cohort"]["eligible"]
             and all(receipt[k] == plan[k] for k in ("parent_repo", "parent_revision", "pilot_repo", "pilot_revision")),
             "reference receipt provenance differs")
    dataset = load_from_disk(str(out / "dataset"))
    _validate_rows(dataset, entry, metadata, template, mode)
    return receipt


def freeze_reference(profiles_path, model, repo, revision, subset, out, cache_dir=None):
    """Download and validate one pinned config, or fully verify a prior local freeze."""
    from datasets import load_dataset, load_from_disk
    out = Path(out)
    profiles, mode, identity = _selection(profiles_path, model, repo, revision, subset)
    if out.exists() or out.is_symlink():
        return validate_frozen_reference(out, profiles_path, model, repo, revision, subset)
    cache_dir = Path(cache_dir) if cache_dir is not None else out.parent / ".reference-cache"
    out.mkdir(parents=True)
    names = ["composition.json"] + [f"audit/{subset}/{label}/{name}.json"
             for label in ("parent", "pilot") for name in ("manifest", "metadata")]
    for name in names:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_download_pinned(repo, revision, name, cache_dir), target)
    plan, entry, metadata, template = _validate_audits(out, profiles, model, repo, subset, mode)
    parquet = _download_pinned(repo, revision, entry["path"], cache_dir, entry["parquet_sha256"])
    # A fresh Arrow cache prevents reuse of a previously altered conversion.
    with tempfile.TemporaryDirectory(prefix=".arrow-", dir=out) as arrow_cache:
        dataset = load_dataset("parquet", data_files={"train": str(parquet)}, split="train",
                               features=reference_features(), cache_dir=arrow_cache)
        _validate_rows(dataset, entry, metadata, template, mode)
        dataset.save_to_disk(str(out / "dataset"))
        saved = load_from_disk(str(out / "dataset"))
        _require(dataset.data.table.equals(saved.data.table), "frozen dataset roundtrip changed native row values")
        del saved, dataset
    receipt = {**identity, "rows": entry["cohort"]["eligible"], "cohort": entry["cohort"],
               "parquet_sha256": entry["parquet_sha256"], "composition_sha256": sha256_file(out / "composition.json"),
               "parent_repo": plan["parent_repo"], "parent_revision": plan["parent_revision"],
               "pilot_repo": plan["pilot_repo"], "pilot_revision": plan["pilot_revision"],
               "all_rows_validated": True, "files_sha256": _file_hashes(out)}
    atomic_json(out / "freeze-receipt.json", receipt)
    return receipt
