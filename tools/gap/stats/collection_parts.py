"""Read-only composition of native pilot100 and disjoint tail400 KLD pairs."""

import csv
import json
import math
import re

from lib.collect_common import dump_stem, SKIP_OVER_BUDGET
from lib.collect_meta_provenance import require_cross_part_execution_alignment
from lib.reference_cohort import canonical_digest, tail_cohort, validate_cohort
from lib.reference_dataset import reference_provenance
from stats.collection_io import (
    META_MUST_MATCH, _meta_guard_value, aligned_metric_items, validate_collection_pair,
)


def _observed_reference(root, keys, size):
    """Bind every completed metric to SHA-checked native generator sidecars.

    Completion is checked by validate_collection_pair before this function.
    A retry may repeat a provenance record, but cannot change its identity.
    Budget-skipped rows have no reference sidecar and stay in cohort coverage.
    """
    with (root / "manifest.csv").open(newline="") as stream:
        rows = {int(r["row_idx"]): r for r in csv.DictReader(stream)}
    if len({r["item_id"] for r in rows.values()}) != len(rows):
        raise ValueError("duplicate source IDs in completed collection")
    expected_keys = {dump_stem(i, r["item_id"]) for i, r in rows.items() if r["status"] == "OK"}
    if expected_keys != set(keys) or any(r["status"] not in ("OK", SKIP_OVER_BUDGET) for r in rows.values()):
        raise ValueError("native collection metrics do not match successful manifest rows")
    observed, generators = {}, {}
    for sidecar in sorted((root / ".attempts").glob("*/references.jsonl")):
        statuses = {r["row_idx"]: r["status"] for r in
                    map(json.loads, (sidecar.parent / "statuses.jsonl").read_text().splitlines())}
        for line in sidecar.read_text().splitlines():
            record = json.loads(line)
            idx, digest = record["row_idx"], record["generator_sha256"]
            if statuses.get(idx) != "OK":
                continue
            if idx not in rows or record["item_id"] != rows[idx]["item_id"]:
                raise ValueError("native reference sidecar differs from completed row identity")
            path = sidecar.parent / "generators" / (str(digest) + ".json")
            if len(str(digest)) != 64 or any(c not in "0123456789abcdef" for c in str(digest)):
                raise ValueError("invalid generator digest")
            metadata = json.loads(path.read_text())
            if canonical_digest(metadata) != digest:
                raise ValueError("native generator sidecar SHA256 mismatch")
            if idx in observed and observed[idx] != record:
                raise ValueError("native reference identity changed across retry records")
            observed[idx] = record
            generators[digest] = metadata
    if any(i not in observed for i, r in rows.items() if r["status"] == "OK") or len(generators) != 1:
        raise ValueError("every successful native row requires one consistent generator provenance")
    metadata = next(iter(generators.values()))
    if metadata.get("schema_version") != "company-reference-v2":
        raise ValueError("composition requires native reference-v2 provenance")
    reference_provenance({"generation_schema_version": metadata["schema_version"],
                          "generation_metadata": json.dumps(metadata)})
    validate_cohort(metadata["cohort"], size)
    return metadata, rows, observed


def _same_candidate(pilot, tail, cap, execution_policy="strict"):
    """Retain scorer/runtime identity while allowing distinct reference views."""
    require_cross_part_execution_alignment(pilot, tail, execution_policy)
    view_fields = {"dataset", "subset", "sort_by", "sort_desc", "num_eval_tokens"}
    if execution_policy == "same-gpu-model-v1":
        for meta in (pilot, tail):
            if type(meta.get("metric_threads")) is not int or meta["metric_threads"] < 1:
                raise ValueError("same-gpu-model-v1 requires positive metric_threads in both parts")
        view_fields.add("metric_threads")
    for field in (set(META_MUST_MATCH) - view_fields) | {"allow_vocab_attr_mismatch", "allow_prefix_drift"}:
        if field not in pilot or field not in tail:
            raise ValueError(f"pilot/tail scoring configuration missing: {field}")
        if _meta_guard_value(pilot, field) != _meta_guard_value(tail, field):
            raise ValueError(f"pilot/tail scoring configuration differs: {field}")
    for field in ("ref_model_fingerprint", "ref_mmproj_fingerprint",
                  "cand_model_fingerprint", "cand_mmproj_fingerprint"):
        if not pilot.get(field) or pilot[field] != tail.get(field):
            raise ValueError(f"pilot/tail candidate or reference identity differs/missing: {field}")
    for meta in (pilot, tail):
        stored = meta.get("num_eval_tokens")
        if type(stored) is not int or (stored != -1 and stored < cap):
            raise ValueError("comparison prefix exceeds stored KLD scoring capacity")
        if meta.get("kind") != "vlm_kld_metrics" or not meta.get("dataset_content_hash"):
            raise ValueError("composition requires native VLM item dataset fingerprints")


def _stored_length(record, metadata, collect_meta):
    """Check effective row generation settings and expected stored scoring length."""
    settings = record["row_settings"]
    request = json.loads(settings["generation_request"])
    expected = metadata["requested_enable_thinking"]
    if expected is None:
        expected = metadata["default_enable_thinking"]
    if (not isinstance(expected, bool) or type(settings["generation_enable_thinking"]) is not bool
            or ("enable_thinking" in request and type(request["enable_thinking"]) is not bool)
            or settings["generation_enable_thinking"] != expected
            or request.get("enable_thinking", expected) != expected
            or request.get("id") != record["item_id"]
            or request.get("system_prompt", metadata["system_prompt"]) != metadata["system_prompt"]):
        raise ValueError("native row mode, request identity or system prompt differs")
    if json.loads(settings["generation_sampling_params"]) != metadata["sampling"]:
        raise ValueError("native row sampling overrides the cohort protocol")
    kwargs = dict(metadata["chat_template_kwargs"])
    if request.get("enable_thinking") is not None:
        kwargs["enable_thinking"] = json.dumps(request["enable_thinking"])
    if json.loads(settings["generation_chat_template_kwargs"]) != kwargs:
        raise ValueError("native row chat template kwargs override the cohort protocol")
    logprobs = settings["generation_token_logprobs"]
    if (not isinstance(logprobs, list) or not logprobs or len(logprobs) > metadata["n_predict"]
            or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in logprobs)):
        raise ValueError("native row requires complete finite token logprobs")
    cap = collect_meta["num_eval_tokens"]
    return len(logprobs) if cap == -1 else min(cap, len(logprobs))


def compose_pilot_tail(args):
    """Concatenate validated item descriptors; never combine part p-values or means.

    One subsequent compare_items call weights each retained item equally.
    Native repetition exclusions and common budget skips condition this cohort;
    neither missing KLD artifacts nor a silent intersection is an exclusion rule.
    """
    if (args.unit != "item" or args.start != 0 or args.end not in (None, -1)
            or args.allow_interaction or args.allow_ref_drift or args.num_eval_tokens < 1):
        raise ValueError("pilot/tail requires complete item cohorts, strict alignment and an explicit positive --num-eval-tokens prefix")
    parts, descriptors, warnings, native = [], [], [], []
    for name, a, b, size in (("pilot", args.pilot_candidate_a, args.pilot_candidate_b, 100),
                             ("tail", args.candidate_a, args.candidate_b, 500)):
        am, bm, ws, skips = validate_collection_pair(a, b)
        keys, drops = aligned_metric_items(a, b)
        ga, ra, oa = _observed_reference(a, keys, size)
        gb, rb, ob = _observed_reference(b, keys, size)
        if ga != gb or oa != ob or {(i,r["item_id"],r["status"]) for i,r in ra.items()} != {(i,r["item_id"],r["status"]) for i,r in rb.items()}:
            raise ValueError("paired native row provenance differs")
        native.append((ga, ra))
        warnings.extend(ws)
        headers = {dump_stem(idx, record["item_id"]): {"npos": _stored_length(record, ga, am), "vocab": ga["vocab_size"]}
                   for idx, record in oa.items()}
        descriptors.extend((f"{name}:{key}", key, a, b, headers[key]) for key in keys)
        parts.append({"part": name, "candidate_a": str(a), "candidate_b": str(b),
                      "candidate_a_meta": am, "candidate_b_meta": bm,
                      "source": ga["dataset_source"], "generator_sha256": canonical_digest(ga),
                      "generation_cap": ga["n_predict"], "n_used": len(keys),
                      "common_skipped_over_budget": skips})
    policy = args.cross_part_execution_policy
    for side in ("candidate_a_meta", "candidate_b_meta"):
        _same_candidate(parts[0][side], parts[1][side], args.num_eval_tokens, policy)
    if policy != "strict":
        warnings.append("cross-part execution policy same-gpu-model-v1: only the physical GPU UUID may differ between pilot and tail; model, driver, binary, libraries, environment and decoder settings must match; positive token metric worker counts may differ; this does not establish numerical or bitwise equivalence across GPUs, and A/B alignment remains strict within each part")
    gp, rp = native[0]
    gt, rt = native[1]
    source_p, source_t = gp["dataset_source"], gt["dataset_source"]
    if (not re.fullmatch(r"[0-9a-f]{40}", str(source_p.get("revision", "")))
            or not source_p.get("path") or source_p.get("split") != "train"
            or source_p.get("num_rows") != 100 or source_t.get("num_rows") != 500
            or any(source_p.get(k) != source_t.get(k) for k in ("path", "revision", "split"))
            or not source_p.get("subset", "").endswith("-subsample-100")
            or source_t.get("subset") != source_p["subset"][:-3] + "500"):
        raise ValueError("pilot and tail must share a pinned prepared source and dataset family")
    # Paths, scheduling and cohort membership are provenance, not generation settings.
    ignored = {"cohort", "dataset_source", "n_predict", "command", "execution_identity", "driver_recovery"}
    required = {"model_files", "mmproj_sha256", "sampling", "vocabulary", "decoding",
                "chat_template", "requested_enable_thinking", "image_min_tokens",
                "image_max_tokens", "binary_sha256", "repetition_detector",
                "chat_template_source", "chat_template_kwargs", "system_prompt", "default_enable_thinking",
                "supports_enable_thinking", "thinking_column", "jinja", "add_bos", "bos_token_id",
                "eos_token_id", "eot_token_id", "media_marker", "image_placeholder_id",
                "image_token_budget_source", "n_ctx", "n_ctx_per_seq", "n_batch", "n_ubatch",
                "n_threads", "n_threads_batch", "total_slots", "execution_identity"}
    if not required <= gp.keys() or not required <= gt.keys():
        raise ValueError("native generator model/protocol identity is incomplete")
    if (not isinstance(gp["execution_identity"].get("libraries"), list)
            or not gp["execution_identity"]["libraries"]
            or gp["execution_identity"]["libraries"] != gt["execution_identity"].get("libraries")):
        raise ValueError("native generator libraries differ")
    if {k:v for k,v in gp.items() if k not in ignored} != {k:v for k,v in gt.items() if k not in ignored}:
        raise ValueError("pilot/tail generation model or protocol differs")
    tail = tail_cohort(gp["cohort"], gt["cohort"])
    for part, (gen, rows), cohort in zip(parts, native, (gp["cohort"], tail)):
        if {r["item_id"] for r in rows.values()} != set(cohort["eligible_ids"]):
            raise ValueError("completed metrics plus budget skips must cover the exact native eligible cohort")
        if type(gen.get("n_predict")) is not int or gen["n_predict"] < args.num_eval_tokens:
            raise ValueError("comparison prefix exceeds native generation cap")
        part["cohort"] = cohort
    if {r["item_id"] for r in rp.values()} & {r["item_id"] for r in rt.values()}:
        raise ValueError("pilot and tail overlap in source item IDs")
    return descriptors, parts, warnings
