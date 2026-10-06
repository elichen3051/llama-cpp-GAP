#!/usr/bin/env python3
"""Freeze a derived teacher-forcing corpus: a source model's reference answers under a tested model's prompts.

  cross-family  thinking answers of another model family, rewritten in the tested model's thinking format
                and tokenized with its tokenizer (needs the tested model's GGUF and llama-trajectory)
  same-family   answer token IDs copied unchanged (source and tested vocabularies must be identical)

Both read two native reference datasets of the same subset (same items, questions and images): --source
gives the answers, --target gives the tested model's own prompt, image layout and prefix token IDs. A
source item with no target row can take its prompt from the target's instruct dataset (--target-instruct):
the thinking prompt is rendered with the tested model's template and checked against every target row
that exists. The result (OUT/dataset) is scored with collect_kld.py like any reference dataset.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import derived_corpus as dc
from lib import thinking_trajectory as tt
from lib.collect_meta_provenance import _llama_cpp_build_commit
from lib.dataset_fingerprint import dataset_content_hash
from lib.model_files import sha256_file
from lib.reference_dataset import canonical_json as canonical, raw_images, validate_reference_row

REPO_ROOT = Path(__file__).resolve().parents[3]
GAP_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("cli/prepare_derived_corpus.py", "lib/derived_corpus.py", "lib/thinking_trajectory.py",
              "lib/thinking_source_exceptions.json")


def run_native(command, log):
    """Run llama-trajectory on the CPU with its stderr in `log`."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with Path(log).open("wb") as error:
        result = subprocess.run([str(c) for c in command], stdout=error, stderr=error, env=env)
    if result.returncode:
        raise ValueError(f"llama-trajectory failed ({result.returncode}); see {log}")


def write_jsonl(path, items):
    Path(path).write_text("".join(canonical(item) + "\n" for item in items), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def tool_provenance(binary=None):
    code = {name: sha256_file(GAP_ROOT / name) for name in CODE_FILES}
    return {"source_checkout_commit": _llama_cpp_build_commit(), "code_sha256": code,
            "llama_trajectory_sha256": sha256_file(binary) if binary else None}


def dataset_identity(spec, dataset):
    return {"input": spec, "dataset_content_hash": dataset_content_hash(dataset), "rows": len(dataset)}


def thinking_rows(row):
    return row["generation_enable_thinking"] is True and json.loads(row["generation_request"])["enable_thinking"] is True


def recover_prompts(args, out, instruct, targets, missing):
    """Thinking prompts of `missing` items, rebuilt from the target instruct rows with the tested model's template.

    The rebuilt prompt replaces the text chunks of the instruct layout (image chunks stay). It is accepted only
    if the same rebuild reproduces the native thinking prompt, prefix IDs and layout of every target row."""
    work = out / "prompt-recovery"
    work.mkdir()
    requests = []
    for row in instruct:
        validate_reference_row(row)
        meta = json.loads(row["generation_metadata"])
        for thinking in (False, True):
            request = {"id": row["id"] + ("::think" if thinking else "::ins"), "question": row["question"],
                       "num_images": row["num_images"], "add_special": row["llamacpp_add_special"],
                       "chat_template_kwargs": meta["chat_template_kwargs"], "enable_thinking": thinking}
            generation = json.loads(row["generation_request"])
            if "system_prompt" in generation or meta.get("system_prompt"):
                request["system_prompt"] = generation.get("system_prompt", meta.get("system_prompt"))
            requests.append(request)
    write_jsonl(work / "requests.jsonl", requests)
    run_native([args.llama_trajectory, "render-prompt", args.target_gguf, work / "requests.jsonl", work / "rendered.jsonl"],
               work / "render-prompt.log")
    rendered = {r["id"]: r for r in read_jsonl(work / "rendered.jsonl")}
    recovered, matches = {}, 0
    for row in instruct:
        old, new = rendered[row["id"] + "::ins"], rendered[row["id"] + "::think"]
        if old["prompt"] != row["llamacpp_prompt_string"]:
            raise ValueError("native instruct prompt rendering differs: " + row["id"])
        if old["template"] != json.loads(row["generation_metadata"])["chat_template"]:
            raise ValueError("GGUF template differs from the native metadata")
        layout = json.loads(row["llamacpp_prompt_layout"])
        text_chunks = [c for c in layout["chunks"] if c["type"] == "text"]
        if not len(text_chunks) == len(old["text_parts_tokens"]) == len(new["text_parts_tokens"]):
            raise ValueError("text/image partition differs: " + row["id"])
        for chunk, before, after in zip(text_chunks, old["text_parts_tokens"], new["text_parts_tokens"]):
            tokens = chunk["tokens"]
            at = [i for i in range(len(tokens) - len(before) + 1) if tokens[i:i + len(before)] == before]
            if len(at) != 1:
                raise ValueError("rendered text is not a unique native text chunk: " + row["id"])
            chunk["tokens"] = tokens[:at[0]] + after + tokens[at[0] + len(before):]
            chunk["n_tokens"] += len(after) - len(before)
            chunk["n_pos"] += len(after) - len(before)
        prefix, positions = [], 0
        for chunk in layout["chunks"]:
            chunk["start"] = len(prefix)
            prefix += chunk["tokens"] if chunk["type"] == "text" else [-1] * chunk["n_tokens"]
            positions += chunk["n_pos"]
        layout["n_tokens"], layout["n_pos"] = len(prefix), positions
        entry = {"id": row["id"], "llamacpp_prompt_string": new["prompt"], "llamacpp_prompt_layout": json.dumps(layout),
                 "n_prefill_tokens": len(prefix), "llamacpp_n_past_prefill": positions, "prefix_ids": prefix,
                 "template_sha256": tt.text_sha(new["template"]),
                 "method": "tested-model GGUF thinking template; native instruct image chunks kept; "
                           "checked against every native thinking row"}
        if row["id"] in targets:
            peer = targets[row["id"]]
            if raw_images(row) != raw_images(peer) or row["question"] != peer["question"]:
                raise ValueError("calibration inputs differ: " + row["id"])
            if (new["prompt"] != peer["llamacpp_prompt_string"] or prefix != list(peer["input_ids"][:peer["n_prefill_tokens"]])
                    or layout != json.loads(peer["llamacpp_prompt_layout"])):
                raise ValueError("rebuilt thinking prefix differs from the native thinking row: " + row["id"])
            matches += 1
        elif row["id"] in missing:
            recovered[row["id"]] = entry
    if not matches or set(recovered) != set(missing):
        raise ValueError(f"prompt recovery incomplete: {matches} calibration rows, missing {sorted(set(missing) - set(recovered))}")
    record = {"calibration_matches": matches, "rows": recovered}
    (work / "recovered.json").write_text(json.dumps(record, indent=1) + "\n")
    peers = {}
    by_id = {row["id"]: row for row in instruct}
    for item_id, entry in recovered.items():
        original = by_id[item_id]
        peer = dict(original)
        for key in ("llamacpp_prompt_string", "llamacpp_prompt_layout", "n_prefill_tokens", "llamacpp_n_past_prefill"):
            peer[key] = entry[key]
        peer["input_ids"] = entry["prefix_ids"] + list(original["input_ids"][original["n_prefill_tokens"]:])
        peers[item_id] = (peer, {**entry, "calibration_matches": matches})
    return peers


def pair_rows(args, out, source, target):
    """(source row, target peer, prompt recovery or None) for every source row, in source order."""
    targets = {row["id"]: row for row in target}
    missing = [row["id"] for row in source if row["id"] not in targets]
    recovered = {}
    if missing:
        if not args.target_instruct:
            raise ValueError(f"{len(missing)} source items have no target row; pass --target-instruct: {missing[:5]}")
        if not args.target_gguf:
            raise ValueError("prompt recovery needs --target-gguf")
        instruct = dc.load_reference_rows(args.target_instruct, args.subset.removesuffix("-think") + "-ins", args.cache_dir)
        recovered = recover_prompts(args, out, instruct, targets, missing)
    pairs = []
    for original in source:
        validate_reference_row(original)
        if original["id"] in targets:
            peer, recovery = targets[original["id"]], None
            validate_reference_row(peer)
        else:
            peer, recovery = recovered[original["id"]]
        if original["question"] != peer["question"] or raw_images(original) != raw_images(peer):
            raise ValueError("source and target question/images differ: " + original["id"])
        pairs.append((original, peer, recovery))
    return pairs


def save_outputs(out, rows, receipt):
    dc.save_dataset(rows, out / "dataset")
    (out / "freeze-receipt.json").write_text(json.dumps({**receipt, "status": "complete", "rows": len(rows),
                                                         "updated": time.time()}, indent=1) + "\n")
    print(json.dumps({"out": str(out), "rows": len(rows)}), flush=True)


def cross_family(args, out, source, target):
    if not args.subset.endswith("-think"):
        raise ValueError("cross-family conversion covers thinking subsets only")
    tt.source_rule(args.source_model)
    tt.target_family(args.target_model)
    model_sha = sha256_file(args.target_gguf)
    pairs = pair_rows(args, out, source, target)
    items, template_requests, parts_list = [], [], []
    for original, peer, recovery in pairs:
        if not thinking_rows(original) or (recovery is None and not thinking_rows(peer)):
            raise ValueError("non-thinking native row: " + original["id"])
        metadata = json.loads(peer["generation_metadata"])
        if metadata["image_min_tokens"] != -1 or metadata["image_max_tokens"] != -1:
            raise ValueError("target prompt used different image bounds: " + original["id"])
        if metadata["model_files"][0]["sha256"] != model_sha:
            raise ValueError("target prompt was not made with --target-gguf: " + original["id"])
        parts = tt.split_trajectory(original, args.source_model, args.subset)
        segments = tt.render_trajectory(parts, args.target_model, peer["llamacpp_prompt_string"])
        request = {"id": original["id"], "question": peer["question"], "num_images": peer["num_images"],
                   "chat_template_kwargs": metadata["chat_template_kwargs"], "native_prompt": peer["llamacpp_prompt_string"],
                   "reasoning": parts["reasoning"] if parts["has_answer_boundary"] else tt.INCOMPLETE_SENTINEL,
                   "answer": parts["answer"] if parts["has_answer_boundary"] else ""}
        generation = json.loads(peer["generation_request"])
        if "system_prompt" in generation or metadata.get("system_prompt"):
            request["system_prompt"] = generation.get("system_prompt", metadata.get("system_prompt"))
        template_requests.append(request)
        items.append({"id": original["id"], "segments": segments,
                      "allow_literal_special_text": tt.literal_tokenization_allowed(parts, args.target_model)})
        parts_list.append(parts)
    write_jsonl(out / "trajectories.jsonl", items)
    write_jsonl(out / "template-requests.jsonl", template_requests)
    run_native([args.llama_trajectory, "render-trajectory", args.target_gguf, out / "template-requests.jsonl",
                out / "template-rendered.jsonl"], out / "render-trajectory.log")
    rendered = read_jsonl(out / "template-rendered.jsonl")
    run_native([args.llama_trajectory, "tokenize", args.target_gguf, out / "trajectories.jsonl", out / "tokenized.jsonl"],
               out / "tokenize.log")
    encoded = read_jsonl(out / "tokenized.jsonl")
    if not len(items) == len(rendered) == len(encoded):
        raise ValueError("llama-trajectory did not cover every source row")
    tool = tool_provenance(args.llama_trajectory)
    source_id, target_id = dataset_identity(args.source, source), dataset_identity(args.target, target)
    rows = []
    for item, oracle, enc, parts, (original, peer, recovery) in zip(items, rendered, encoded, parts_list, pairs):
        metadata = json.loads(peer["generation_metadata"])
        if oracle["id"] != item["id"] or oracle["template"] != metadata["chat_template"]:
            raise ValueError("rendered template differs from the native model template: " + item["id"])
        if oracle["rendered"] != peer["llamacpp_prompt_string"] + oracle["suffix"]:
            raise ValueError("the tested template changed the native prefix: " + item["id"])
        text = "".join(s["text"] for s in item["segments"])
        if text != tt.native_continuation(oracle, parts, args.target_model):
            raise ValueError("converted trajectory differs from the tested-model template: " + item["id"])
        if enc["id"] != item["id"] or enc["roundtrip_exact"] is not True or enc["vocab_size"] != metadata["vocabulary"]["size"]:
            raise ValueError("tokenization identity mismatch: " + item["id"])
        if [{k: s[k] for k in ("text", "parse_special")} for s in enc["segments"]] != item["segments"]:
            raise ValueError("tokenizer changed structural/content segment boundaries: " + item["id"])
        if enc["canonical_tokenization_exact"] != (not item["allow_literal_special_text"]):
            raise ValueError("unexpected canonical tokenization result: " + item["id"])
        terminal = 248046 if tt.target_family(args.target_model) == "qwen" else 106
        if (enc["tokens"][-1] == terminal) != (parts["source_stop_type"] == "eos"):
            raise ValueError("target terminal token disagrees with the source stop condition: " + item["id"])
        protocol = {"schema": dc.SCHEMA_CROSS_FAMILY, "mode": "thinking", "target": args.target_model,
                    "teacher": args.source_model, "subset": args.subset, "source_dataset": source_id,
                    "target_dataset": target_id, "source_text_sha256": tt.text_sha(original["generated_texts"]),
                    "source_parts": parts, "reasoning_sha256": tt.text_sha(parts["reasoning"]),
                    "final_answer_sha256": tt.text_sha(parts["answer"]),
                    "native_template_oracle": {"template_sha256": tt.text_sha(oracle["template"]),
                                               "rendered_sha256": tt.text_sha(oracle["rendered"]),
                                               "continuation_sha256": tt.text_sha(text), "native_prefix_exact": True,
                                               "policy": "native assistant template through EOG for eos; no EOG for length; "
                                                         "unclosed reasoning stays open"},
                    "canonical_tokenization_exact": enc["canonical_tokenization_exact"],
                    "trajectory_segments": item["segments"], "encoded_trajectory_segments": enc["segments"],
                    "trajectory_sha256": tt.text_sha(text), "target_prompt": peer["llamacpp_prompt_string"],
                    "source_generation_metadata_sha256": tt.text_sha(original["generation_metadata"]),
                    "target_generation_metadata_sha256": tt.text_sha(peer["generation_metadata"]),
                    "target_model_sha256": model_sha, "add_special_answer": False,
                    "parse_special_answer": "structural-markers-only",
                    "prefix_method": "recovered" if recovery else "native tested-model thinking prompt",
                    "prefix_recovery": recovery, "tool": tool}
        rows.append(dc.derived_row(peer, enc["tokens"], text, protocol, metadata["vocabulary"]))
    save_outputs(out, rows, {"schema": dc.SCHEMA_CROSS_FAMILY, "target": args.target_model, "teacher": args.source_model,
                             "subset": args.subset, "source_dataset": source_id, "target_dataset": target_id,
                             "incomplete_reasoning_rows": sum(not p["has_answer_boundary"] for p in parts_list),
                             "direct_answer_rows": sum(not p["source_has_reasoning"] for p in parts_list),
                             "literal_tokenization_rows": sum(i["allow_literal_special_text"] for i in items),
                             "recovered_target_prompt_rows": sum(bool(p[2]) for p in pairs),
                             "answer_tokens": sum(r["generated_tokens_len"] for r in rows), "tool": tool})


def same_family(args, out, source, target):
    mode = "thinking" if args.subset.endswith("-think") else "instruct" if args.subset.endswith("-ins") else None
    if mode is None:
        raise ValueError("subset must end in -think or -ins")
    pairs = pair_rows(args, out, source, target)
    tool = tool_provenance(args.llama_trajectory if args.target_instruct else None)
    source_id, target_id = dataset_identity(args.source, source), dataset_identity(args.target, target)
    rows, different_prefix = [], 0
    for original, peer, recovery in pairs:
        if original["generation_enable_thinking"] != (mode == "thinking") or (
                recovery is None and peer["generation_enable_thinking"] != (mode == "thinking")):
            raise ValueError(f"row is not a {mode} row: " + original["id"])
        source_meta, target_meta = json.loads(original["generation_metadata"]), json.loads(peer["generation_metadata"])
        if source_meta["vocabulary"] != target_meta["vocabulary"]:
            raise ValueError("source and target vocabularies differ; use cross-family")
        answer = list(original["input_ids"][original["n_prefill_tokens"]:])
        n = peer["n_prefill_tokens"]
        different_prefix += list(peer["input_ids"][:n]) != list(original["input_ids"][:original["n_prefill_tokens"]])
        protocol = {"schema": dc.SCHEMA_SAME_FAMILY, "mode": mode, "target": args.target_model,
                    "teacher": args.source_model, "subset": args.subset, "source_dataset": source_id,
                    "target_dataset": target_id, "source_vocabulary": source_meta["vocabulary"],
                    "tokens_transferred_without_retokenization": True,
                    "source_ids_sha256": tt.text_sha(canonical(answer)),
                    "source_generation_metadata_sha256": tt.text_sha(original["generation_metadata"]),
                    "target_generation_metadata_sha256": tt.text_sha(peer["generation_metadata"]),
                    "prefix_method": "recovered" if recovery else "native tested-model prompt and image layout",
                    "prefix_recovery": recovery, "tool": tool}
        rows.append(dc.derived_row(peer, answer, original["generated_texts"], protocol, target_meta["vocabulary"]))
    save_outputs(out, rows, {"schema": dc.SCHEMA_SAME_FAMILY, "mode": mode, "target": args.target_model,
                             "teacher": args.source_model, "subset": args.subset, "source_dataset": source_id,
                             "target_dataset": target_id, "different_native_prefix_rows": different_prefix,
                             "recovered_target_prompt_rows": sum(bool(p[2]) for p in pairs),
                             "answer_tokens": sum(r["generated_tokens_len"] for r in rows), "tool": tool})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("cross-family", "same-family"))
    p.add_argument("--source", required=True, help="source reference rows: save_to_disk dir, Parquet, or hf://repo@revision")
    p.add_argument("--source-model", required=True, help="source model name, e.g. gemma-4-31b-it")
    p.add_argument("--target", required=True, help="tested model's native reference rows of the same subset")
    p.add_argument("--target-model", required=True, help="tested model name, e.g. qwen3.5-4b")
    p.add_argument("--subset", required=True, help="e.g. mmstar-subsample-100-think")
    p.add_argument("--target-gguf", help="tested model GGUF whose template and tokenizer made --target (cross-family)")
    p.add_argument("--target-instruct", help="tested model's instruct rows; rebuilds prompts of source items with no target row")
    p.add_argument("--llama-trajectory", default=str(REPO_ROOT / "build/bin/llama-trajectory"))
    p.add_argument("--cache-dir", help="Hugging Face cache for hf:// inputs")
    p.add_argument("--out", required=True, type=Path, help="new output directory")
    args = p.parse_args(argv)
    if args.kind == "cross-family" and not args.target_gguf:
        p.error("cross-family needs --target-gguf")
    return args


def main(argv=None):
    args = parse_args(argv)
    from datasets import disable_caching
    disable_caching()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    source = dc.load_reference_rows(args.source, args.subset, args.cache_dir)
    target = dc.load_reference_rows(args.target, args.subset, args.cache_dir)
    args.out.mkdir(parents=True)
    (cross_family if args.kind == "cross-family" else same_family)(args, args.out, source, target)


if __name__ == "__main__":
    main()
