#!/usr/bin/env python3
"""Publish one completed native reference cohort with immutable subset naming."""

import argparse
import fnmatch
import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

SKYMIZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYMIZER))
from lib.reference_dataset import validate_reference_row, sha256_file
from lib.reference_run import atomic_json, read_records
from lib.reference_study import reference_template, validate_reference_cohort


AUDIT_FILES = ("metadata.json", "complete.json", "run_start.json", "excluded.jsonl", "failures.jsonl", "native-attempts.json")
DETECTOR = {"enabled": True, "version": "exact-token-repeat-v1",
            "source_commit": "d4163fc1c328fe39310465680647b465cf96c4af",
            "source_sha256": "8c0ede4aa476d7ceea39e57ac4bd3e462569d45683c5672746fee1d65940bda9",
            "min_repeated_tokens": 96, "min_repeats": 3, "online_window": 2048, "check_interval": 32,
            "input": "generated_target_accepted_token_ids", "final_scan": "exact_consecutive_blocks_anywhere"}


def verify_model(metadata, expected):
    identity = expected["identity"]
    def records(files):
        return sorted((Path(x.get("path", x.get("name", ""))).name, x["size"], x["sha256"]) for x in files)
    main = [x for x in identity["files"] if x["role"] == "llm"]
    projector, = [x for x in identity["files"] if x["role"] == "mmproj"]
    if (Path(metadata["model_path"]).name != Path(expected["model"]).name
            or records(metadata["model_files"]) != records(main)
            or Path(metadata["mmproj_path"]).name != projector["name"]
            or metadata["mmproj_sha256"] != projector["sha256"]):
        raise ValueError("model/projector identities differ from the approved BF16 artifacts")
    decoding = metadata["decoding"]
    if decoding["method"] == "mtp":
        mtp = decoding["mtp"]
        if mtp["head_source"] != expected["mtp"]:
            raise ValueError("MTP head source differs from the approved model")
        if mtp["head_source"] == "sidecar" and records(mtp["head_files"]) != records(identity["head_files"]):
            raise ValueError("MTP head identity differs from the approved artifact")
    elif decoding["method"] != "autoregressive":
        raise ValueError("unsupported reference decoding method")


def prepare(run, model, mode, profiles):
    from datasets import load_from_disk
    template = reference_template(profiles["models"][model], mode)
    completion = json.loads((run / "complete.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    state = json.loads((run / "run_state.json").read_text())
    if state["status"] not in ("complete", "complete_with_failures") or completion["status"] != state["status"]:
        raise ValueError("reference run has not completed processing")
    source = metadata["dataset_source"]
    match = re.fullmatch(r"(.+)-subsample-(100|500)", source.get("subset") or "")
    if not match or match[1] not in profiles["sources"]:
        raise ValueError("source must be a supported full subsample-100 or subsample-500 cohort")
    size = int(match[2])
    validate_reference_cohort(profiles, size)
    if source["path"] != profiles["dataset"]["repo"] or source["revision"] != profiles["dataset"]["revision"] or source["split"] != "train":
        raise ValueError("source repository/revision differs from campaign profiles")
    cohort = metadata["cohort"]
    if cohort != completion["cohort"] or cohort["requested"] != size or source["num_rows"] != size:
        raise ValueError("short smoke or unreconciled cohort cannot be published as a full subset")
    if not cohort["eligible"]:
        raise ValueError("no eligible references to upload; retain the failure/exclusion manifests")
    requested = cohort["requested_ids"]
    partition = cohort["eligible_ids"] + cohort["excluded_ids"] + cohort["failed_ids"]
    if len(requested) != size or len(set(requested)) != size or len(partition) != size or set(partition) != set(requested):
        raise ValueError("cohort IDs do not form an exact partition")
    for key in ("requested_ids", "eligible_ids", "excluded_ids", "failed_ids"):
        payload = json.dumps(cohort[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(payload.encode()).hexdigest() != cohort[key + "_sha256"]:
            raise ValueError(f"cohort digest mismatch: {key}")
    for key in ("requested", "eligible", "excluded", "failed"):
        if type(cohort[key]) is not int or cohort[key] != len(cohort[key + "_ids"]):
            raise ValueError(f"cohort count differs from IDs: {key}")
    if (cohort["generated"] != cohort["eligible"] + cohort["excluded"]
            or type(cohort["native_generated"]) is not int
            or not cohort["generated"] <= cohort["native_generated"] <= size
            or completion["rows"] != cohort["eligible"]
            or state["cohort"] != cohort
            or completion["status"] != ("complete_with_failures" if cohort["failed"] else "complete")):
        raise ValueError("completion counts/status do not reconcile")
    verify_model(metadata, profiles["models"][model])
    defaults = {"chat_template_source": "gguf", "jinja": True, **template, "image_min_tokens": -1,
                "image_max_tokens": -1, "image_token_budget_source": "mtmd_init_params"}
    if any(metadata.get(key) != value for key, value in defaults.items()):
        raise ValueError("template or image budget differs from the native production defaults")
    sampling = profiles["models"][model]["effective_sampling"][mode]
    if metadata["sampling"] != sampling or sampling["seed"] != profiles["seed"]:
        raise ValueError("effective sampling differs from the frozen model-card/native profile")
    if metadata["requested_enable_thinking"] is not (mode == "thinking") or metadata.get("thinking_column"):
        raise ValueError("requested mode differs from the generated cohort")
    if metadata["n_predict"] != profiles["generation_caps"][mode]:
        raise ValueError("generation cap differs from the production profile")
    detector = metadata.get("repetition_detector", {})
    if detector != DETECTOR:
        raise ValueError("production upload requires the repetition filter")
    ds = load_from_disk(str(run / "dataset"))
    if list(ds["id"]) != cohort["eligible_ids"] or len(ds) != completion["rows"]:
        raise ValueError("saved dataset rows differ from the eligible cohort")
    for row in ds:
        validate_reference_row(row)
        if row["generation_enable_thinking"] is not (mode == "thinking"):
            raise ValueError("mixed reasoning modes in one subset")
        kwargs = {**template["chat_template_kwargs"], "enable_thinking": "true" if mode == "thinking" else "false"}
        if (json.loads(row["generation_chat_template_kwargs"]) != kwargs
                or json.loads(row["generation_request"]).get("system_prompt", metadata["system_prompt"]) != template["system_prompt"]):
            raise ValueError("row template differs from the native production defaults")
        if json.loads(row["generation_sampling_params"]) != sampling:
            raise ValueError("row sampling differs from the frozen production profile")
        if json.loads(row["generation_metadata"]) != metadata:
            raise ValueError("row metadata differs from run metadata")
    for filename, key in (("excluded.jsonl", "excluded_ids"), ("failures.jsonl", "failed_ids")):
        if [r["id"] for r in read_records(run / filename)] != cohort[key]:
            raise ValueError(f"{filename} differs from the cohort")
    failures = read_records(run / "failures.jsonl")
    if cohort["native_generated"] != cohort["generated"] + sum(r["status"] == "validation_failed" for r in failures):
        raise ValueError("native result count differs from validated results and validation failures")
    repo = f"elichen-skymizer/{model}-" + ("pilot" if size == 100 else "collect-500")
    subset = source["subset"] + ("-think" if mode == "thinking" else "-ins")
    export = run / "upload"
    export.mkdir(exist_ok=True)
    parquet = export / "train-00000-of-00001.parquet"
    ds.to_parquet(str(parquet))
    manifest = {"repo": repo, "subset": subset, "split": "train", "rows": len(ds), "cohort": cohort,
                "parquet_sha256": sha256_file(parquet), "metadata_sha256": sha256_file(run / "metadata.json"),
                "source": source, "audit_sha256": {name: sha256_file(run / name) for name in AUDIT_FILES}}
    atomic_json(export / "manifest.json", manifest)
    return manifest, parquet


def card_parts(readme):
    import yaml
    if readme.startswith("---\n"):
        _, front, body = readme.split("---\n", 2)
        card = yaml.safe_load(front) or {}
    else:
        card, body = {}, readme
    if not isinstance(card, dict) or not isinstance(card.get("configs", []), list):
        raise ValueError("malformed dataset card configs")
    return card, body


def config_entry(subset, parquet):
    return {"config_name": subset, "data_files": [{"split": "train", "path": f"{subset}/{parquet.name}"}]}


def require_disjoint_configs(configs, new_paths):
    from fsspec.utils import glob_translate
    for config in configs:
        files = config.get("data_files")
        if not files:
            raise ValueError("existing config uses inferred data files; make its mapping explicit first")
        if isinstance(files, str):
            files = [files]
        if not isinstance(files, list):
            raise ValueError("unsupported existing config data_files mapping")
        patterns = []
        for item in files:
            value = item.get("path") if isinstance(item, dict) else item
            if isinstance(value, str):
                patterns.append(value)
            elif isinstance(value, list) and all(isinstance(p, str) for p in value):
                patterns.extend(value)
            else:
                raise ValueError("unsupported existing config data_files pattern")
        for pattern in patterns:
            pattern = pattern.removeprefix("./")
            if config.get("data_dir"):
                pattern = str(config["data_dir"]).rstrip("/") + "/" + pattern
            if pattern.startswith("/") or "://" in pattern or ".." in pattern.split("/"):
                raise ValueError("unsupported existing config path outside the repository")
            pattern = posixpath.normpath(pattern)
            if any(fnmatch.fnmatchcase(path, pattern) or re.fullmatch(glob_translate(pattern), path)
                   or path.startswith(pattern.rstrip("/") + "/") for path in new_paths):
                raise ValueError("existing config would include the new reference files")


def publish(run, manifest, parquet, private=False):
    import yaml
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    api = HfApi()
    repo, subset = manifest["repo"], manifest["subset"]
    audit = f"audit/{subset}"
    expected_files = {f"{subset}/{parquet.name}": manifest["parquet_sha256"],
                      **{f"{audit}/{name}": digest for name, digest in manifest["audit_sha256"].items()}}
    manifest_path = f"{audit}/manifest.json"
    expected_paths = set(expected_files) | {manifest_path}
    entry = config_entry(subset, parquet)
    def download(path, revision):
        return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision))
    def occupied(revision):
        return {p for p in api.list_repo_files(repo, repo_type="dataset", revision=revision)
                if p == subset or p.startswith(subset + "/") or p == audit or p.startswith(audit + "/")}
    def verify(revision):
        if occupied(revision) != expected_paths:
            raise ValueError("remote subset/audit paths do not match the publication manifest")
        if json.loads(download(manifest_path, revision).read_text()) != manifest:
            raise ValueError("remote audit manifest differs")
        card, _ = card_parts(download("README.md", revision).read_text())
        if [c for c in card.get("configs", []) if c.get("config_name") == subset] != [entry]:
            raise ValueError("remote README config mapping differs")
        for path, digest in expected_files.items():
            if sha256_file(download(path, revision)) != digest:
                raise ValueError(f"remote checksum mismatch: {path}")
    # Check the staged bytes again before writing to the Hub.
    local = {f"{subset}/{parquet.name}": parquet,
             **{f"{audit}/{name}": run / name for name in manifest["audit_sha256"]}}
    for path, digest in expected_files.items():
        if sha256_file(local[path]) != digest:
            raise ValueError(f"local publication artifact changed: {path}")
    if json.loads((run / "upload/manifest.json").read_text()) != manifest:
        raise ValueError("local staged manifest changed")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)
    receipt = None
    for retry in range(6):
        head = api.repo_info(repo, repo_type="dataset").sha
        paths = occupied(head)
        if manifest_path in paths:
            verify(head)
            receipt = {"repo": repo, "subset": subset, "commit": head, "already_present": True}
            break
        if paths:
            raise ValueError(f"subset/audit namespace is already occupied: {repo}/{subset}")
        try:
            readme = download("README.md", head).read_text()
        except EntryNotFoundError:
            readme = "# Native llama.cpp reference datasets\n\nGenerated target trajectories with GGUF templates/tokenization. Each config's audit directory records generation settings, source IDs, exclusions and failures. The source size counts requested items; train contains eligible references only.\n"
        card, body = card_parts(readme)
        configs = card.setdefault("configs", [])
        if not configs and set(api.list_repo_files(repo, repo_type="dataset", revision=head)) - {"README.md", ".gitattributes"}:
            raise ValueError("existing repository may use implicit default data; add explicit configs before publication")
        if any(c["config_name"] == subset for c in configs):
            raise ValueError(f"existing config lacks a matching audit manifest: {subset}")
        require_disjoint_configs(configs, expected_paths)
        configs.append(entry)
        updated = "---\n" + yaml.safe_dump(card, sort_keys=False) + "---\n" + body
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local[path])) for path in sorted(local)]
        operations.extend([CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=updated.encode()),
                           CommitOperationAdd(path_in_repo=manifest_path, path_or_fileobj=str(run / "upload/manifest.json"))])
        try:
            commit = api.create_commit(repo, repo_type="dataset", operations=operations,
                parent_commit=head, commit_message=f"Add reference config {subset}")
            verify(commit.oid)
            receipt = {"repo": repo, "subset": subset, "commit": commit.oid, "already_present": False}
            break
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code not in (409, 412) or retry == 5:
                raise
    if receipt is None:
        raise ValueError("could not commit config after concurrent updates")
    receipt.update(rows=manifest["rows"], parquet_sha256=manifest["parquet_sha256"],
                   audit_sha256=manifest["audit_sha256"], status="verified")
    atomic_json(run / "upload/receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["instruct", "thinking"], required=True)
    p.add_argument("--profiles", type=Path, default=SKYMIZER / "scripts/reference_model_profiles.json")
    p.add_argument("--private", action="store_true", help="only affects a newly created dataset repo")
    p.add_argument("--dry-run", action="store_true", help="validate and export locally without Hub writes")
    args = p.parse_args()
    try:
        profiles = json.loads(args.profiles.read_text())
        manifest, parquet = prepare(args.run, args.model, args.mode, profiles)
        print(json.dumps(manifest if args.dry_run else publish(args.run, manifest, parquet, args.private), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
