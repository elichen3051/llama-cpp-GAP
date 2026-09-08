#!/usr/bin/env python3
"""Stage or publish a private, pinned tail400 view of one native collect500 repo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.reference_cohort import canonical_digest, tail_cohort


def read_json(repo, revision, path, token):
    """Read a small audit file at an immutable Hub revision.

    API: https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_download
    """
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                            token=token, cache_dir="/tmp/skymizer-collect400-cache")
    raw = Path(local).read_bytes()
    return json.loads(raw), raw


def parquet_ids(repo, revision, path, token):
    """Project IDs through HTTP range reads without downloading image columns.

    APIs: https://huggingface.co/docs/huggingface_hub/package_reference/hf_file_system
    https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{repo}@{revision}/{path}", "rb") as stream:
        return pq.ParquetFile(stream).read(columns=["item_id"])["item_id"].to_pylist()


def stage(parent_repo, directory, token, revision=None):
    """Prepare reviewed tail configs using native requested-ID membership.

    Parquet filters are standard builder parameters, applied by load_dataset.
    Physical Parquet bytes retain the full parent cohort. Viewer is disabled
    because its Parquet shortcut bypasses these logical filters.
    APIs: https://huggingface.co/docs/datasets/repository_structure#builder-parameters
    https://huggingface.co/docs/hub/datasets-viewer-configure#disable-the-viewer
    """
    from huggingface_hub import HfApi
    import yaml
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("stage directory must be empty")
    if not re.fullmatch(r"elichen-skymizer/[A-Za-z0-9_.-]+-collect-500", parent_repo):
        raise ValueError("parent must be an elichen-skymizer native collect-500 repo")
    api = HfApi(token=token)
    parent = api.dataset_info(parent_repo, revision=revision)
    pilot_repo = parent_repo.removesuffix("-collect-500") + "-pilot"
    pilot = api.dataset_info(pilot_repo)
    destination = parent_repo.removesuffix("-collect-500") + "-collect-400"
    configs, entries = [], []
    for config in parent.card_data.to_dict()["configs"]:
        name = config["config_name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+-subsample-500-(ins|think)", name):
            raise ValueError("unexpected parent config name")
        files = config["data_files"]
        if len(files) != 1 or files[0]["split"] != "train" or not isinstance(files[0]["path"], str):
            raise ValueError("expected one native train Parquet shard per config")
        source_path = files[0]["path"]
        if ".." in Path(source_path).parts or source_path.startswith("/") or not source_path.endswith(".parquet"):
            raise ValueError("invalid source Parquet path")
        pilot_name = name.replace("-subsample-500-", "-subsample-100-")
        audits = []
        for info, cfg in ((parent, name), (pilot, pilot_name)):
            manifest, raw = read_json(info.id, info.sha, f"audit/{cfg}/manifest.json", token)
            check_manifest(manifest, info.id, cfg)
            metadata, meta_raw = read_json(info.id, info.sha, f"audit/{cfg}/metadata.json", token)
            digest = hashlib.sha256(meta_raw).hexdigest()
            if (digest != manifest["metadata_sha256"] or digest != manifest["audit_sha256"]["metadata.json"]
                    or metadata["cohort"] != manifest["cohort"] or metadata["dataset_source"] != manifest["source"]):
                raise ValueError("parent audit/metadata identity mismatch")
            audits.append((manifest, metadata, raw, meta_raw))
        pm, pg, _, _ = audits[0]
        im, ig, _, _ = audits[1]
        source = pg["dataset_source"]
        if (source["path"] != "elichen-skymizer/vlm-prepared-dataset"
                or any(source[k] != ig["dataset_source"][k] for k in ("path", "revision", "split"))
                or ig["dataset_source"]["subset"] != source["subset"].removesuffix("500") + "100"):
            raise ValueError("pilot/parent prepared source identity mismatch")
        tail = tail_cohort(im["cohort"], pm["cohort"])
        remote = api.get_paths_info(parent_repo, [source_path], repo_type="dataset", revision=parent.sha)[0]
        if not remote.lfs or remote.lfs.sha256 != pm["parquet_sha256"]:
            raise ValueError("parent Parquet does not match audited SHA256")
        if parquet_ids(parent_repo, parent.sha, source_path, token) != pm["cohort"]["eligible_ids"]:
            raise ValueError("actual parent Parquet IDs differ from eligible cohort")
        target_name = name.replace("-subsample-500-", "-tail-400-")
        target_path = f"{target_name}/parent500.parquet"
        configs.append({"config_name": target_name,
                        "data_files": [{"split": "train", "path": target_path}],
                        "filters": [["item_id", "in", tail["requested_ids"]]]})
        entry = {"config": target_name, "path": target_path, "source_path": source_path,
                 "parent_config": name, "pilot_config": pilot_name, "parquet_sha256": pm["parquet_sha256"],
                 "source": source, "cohort": tail, "parent_generation_cap": pg["n_predict"],
                 "pilot_generation_cap": ig["n_predict"], "parent_audit": pm, "pilot_audit": im}
        entries.append(entry)
        for label, (_, _, raw, meta_raw) in zip(("parent", "pilot"), audits):
            out = directory / "audit" / target_name / label
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_bytes(raw)
            (out / "metadata.json").write_bytes(meta_raw)
        print(f"staged {target_name}: {tail['eligible']}/400 eligible", flush=True)
    plan = {"schema": "skymizer-collect400-view-v1", "destination": destination,
            "parent_repo": parent_repo, "parent_revision": parent.sha,
            "pilot_repo": pilot_repo, "pilot_revision": pilot.sha, "configs": entries}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "composition.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    frontmatter = yaml.safe_dump({"viewer": False, "configs": configs}, sort_keys=False, allow_unicode=True)
    body = (f"# Native tail400 reference views\n\nParent: `{parent_repo}@{parent.sha}`. "
            f"Pilot: `{pilot_repo}@{pilot.sha}`.\n\n"
            "Load with `datasets.load_dataset(repo_id, config_name, split='train')`. "
            "Each config filters requested source positions101..500 by stable item ID and preserves generation exclusions. "
            "It contains at most400 eligible rows. The physical Parquet is a byte-identical copy of the full parent500 file; "
            "direct Parquet readers bypass the filter and MUST NOT be used for tail-only collection. "
            "The Viewer is disabled because its Parquet shortcut does not honor these filters.\n\n"
            "Native row metadata and trajectories are unchanged. Parent and pilot generation caps can differ; "
            "use an explicit common scored token prefix when combining their saved KLD. "
            "See composition.json and audit/ for source revisions, exact cohorts and original audit evidence.\n")
    (directory / "README.md").write_text("---\n" + frontmatter + "---\n\n" + body)
    return plan


def publish(directory, token):
    """Copy immutable source bytes into a NEW private repo, then verify logical IDs.

    APIs: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.CommitOperationCopy
    https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset
    """
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy
    from huggingface_hub.errors import RepositoryNotFoundError
    from datasets import load_dataset
    plan = json.loads((directory / "composition.json").read_text())
    allowed = validate_stage(directory, plan)
    destination = plan["destination"]
    api = HfApi(token=token)
    try:
        api.dataset_info(destination)
    except RepositoryNotFoundError:
        pass
    else:
        raise ValueError("destination exists; refusing to overwrite an existing dataset")
    # Recheck the staged evidence against the immutable upstream objects.
    for entry in plan["configs"]:
        source_file = api.get_paths_info(plan["parent_repo"], [entry["source_path"]],
                                         repo_type="dataset", revision=plan["parent_revision"])[0]
        if not source_file.lfs or source_file.lfs.sha256 != entry["parquet_sha256"]:
            raise ValueError("immutable source Parquet SHA256 differs before publication")
        for label, repo, revision, config in (
                ("parent", plan["parent_repo"], plan["parent_revision"], entry["parent_config"]),
                ("pilot", plan["pilot_repo"], plan["pilot_revision"], entry["pilot_config"])):
            _, raw = read_json(repo, revision, f"audit/{config}/manifest.json", token)
            if raw != (directory / "audit" / entry["config"] / label / "manifest.json").read_bytes():
                raise ValueError("staged audit differs from immutable upstream manifest")
    operations = [CommitOperationCopy(src_repo_id=plan["parent_repo"], src_repo_type="dataset",
                    src_revision=plan["parent_revision"], src_path_in_repo=e["source_path"], path_in_repo=e["path"])
                  for e in plan["configs"]]
    for path in sorted(allowed):
        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(directory / path)))
    api.create_repo(destination, repo_type="dataset", private=True, exist_ok=False)
    initial = api.dataset_info(destination)
    if not initial.private:
        raise ValueError("new destination unexpectedly public; no data uploaded")
    commit = api.create_commit(destination, repo_type="dataset", operations=operations,
                               parent_commit=initial.sha, commit_message="Add pinned native tail400 reference views")
    receipt_path = directory.parent / (directory.name + "-published.json")
    receipt_path.write_text(json.dumps({"repo": destination, "revision": commit.oid,
                                        "status": "uploaded_verification_pending"}, indent=2) + "\n")
    info = api.dataset_info(destination, revision=commit.oid)
    if not info.private:
        raise ValueError("new destination unexpectedly public")
    verified = []
    for entry in plan["configs"]:
        file = api.get_paths_info(destination, [entry["path"]], repo_type="dataset", revision=commit.oid)[0]
        if not file.lfs or file.lfs.sha256 != entry["parquet_sha256"]:
            raise ValueError("copied Parquet SHA256 mismatch")
        ds = load_dataset(destination, entry["config"], split="train", revision=commit.oid,
                          token=token, streaming=True, columns=["item_id"])
        ids = [row["item_id"] for row in ds]
        if ids != entry["cohort"]["eligible_ids"]:
            raise ValueError("published logical IDs differ from the exact eligible tail")
        verified.append({"config": entry["config"], "rows": len(ids), "ids_sha256": canonical_digest(ids)})
    receipt = {"repo": destination, "revision": commit.oid, "private": info.private,
               "status": "verified", "verified": verified}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def validate_stage(directory, plan):
    """Require the exact reviewed upload files and logical tail filters before writes."""
    import yaml
    parent = plan["parent_repo"]
    if (plan.get("schema") != "skymizer-collect400-view-v1"
            or not re.fullmatch(r"elichen-skymizer/[A-Za-z0-9_.-]+-collect-500", parent)
            or plan["destination"] != parent.removesuffix("-collect-500") + "-collect-400"
            or plan["pilot_repo"] != parent.removesuffix("-collect-500") + "-pilot"
            or any(not re.fullmatch(r"[0-9a-f]{40}", plan[k]) for k in ("parent_revision", "pilot_revision"))
            or not plan["configs"]):
        raise ValueError("invalid immutable collect400 plan identity")
    allowed, configs, names, paths = {"README.md", "composition.json"}, [], set(), set()
    for entry in plan["configs"]:
        name, source_name = entry["config"], entry["parent_config"]
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+-subsample-500-(ins|think)", source_name)
                or name != source_name.replace("-subsample-500-", "-tail-400-")
                or entry["pilot_config"] != source_name.replace("-subsample-500-", "-subsample-100-")
                or entry["path"] != f"{name}/parent500.parquet" or name in names
                or entry["source_path"] != f"{source_name}/train-00000-of-00001.parquet"
                or entry["source_path"] in paths):
            raise ValueError("invalid or repeated config/Parquet path")
        names.add(name); paths.add(entry["source_path"])
        manifests = {}
        for label in ("parent", "pilot"):
            base = f"audit/{name}/{label}"
            allowed.update((base + "/manifest.json", base + "/metadata.json"))
            manifest = json.loads((directory / base / "manifest.json").read_text())
            check_manifest(manifest, plan[label + "_repo"], entry[label + "_config"])
            raw = (directory / base / "metadata.json").read_bytes()
            metadata = json.loads(raw)
            if (manifest != entry[label + "_audit"]
                    or hashlib.sha256(raw).hexdigest() != manifest["metadata_sha256"]
                    or manifest["metadata_sha256"] != manifest["audit_sha256"]["metadata.json"]
                    or metadata["cohort"] != manifest["cohort"] or metadata["dataset_source"] != manifest["source"]
                    or entry[label + "_generation_cap"] != metadata["n_predict"]):
                raise ValueError("staged native audit identity differs")
            manifests[label] = manifest
        tail = tail_cohort(manifests["pilot"]["cohort"], manifests["parent"]["cohort"])
        if (entry["source"] != manifests["parent"]["source"]
                or any(manifests["parent"]["source"][k] != manifests["pilot"]["source"][k]
                       for k in ("path", "revision", "split"))):
            raise ValueError("staged pilot/parent source identity differs")
        if tail != entry["cohort"] or entry["parquet_sha256"] != manifests["parent"]["parquet_sha256"]:
            raise ValueError("staged tail or Parquet identity differs")
        configs.append({"config_name": name, "data_files": [{"split": "train", "path": entry["path"]}],
                        "filters": [["item_id", "in", tail["requested_ids"]]]})
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if files != allowed or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("stage must contain exactly the reviewed files and no symlinks")
    card = (directory / "README.md").read_text().split("---", 2)
    if len(card) != 3 or card[0] or yaml.safe_load(card[1]) != {"viewer": False, "configs": configs}:
        raise ValueError("staged README must disable Viewer and contain the exact tail filters")
    return allowed


def check_manifest(manifest, repo, config):
    """Bind the published native audit to its actual repository/config and source."""
    size = 500 if "-subsample-500-" in config else 100
    source = manifest["source"]
    if (manifest.get("repo") != repo or manifest.get("subset") != config
            or manifest.get("split") != "train" or manifest.get("rows") != manifest["cohort"]["eligible"]
            or source.get("subset") != config.rsplit("-", 1)[0] or source.get("num_rows") != size
            or source.get("path") != "elichen-skymizer/vlm-prepared-dataset" or source.get("split") != "train"
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))):
        raise ValueError("native audit repository/config/source identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-repo")
    parser.add_argument("--revision", help="immutable parent500 commit to stage")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="publish an already reviewed stage directory")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN must be explicitly available in the environment")
    if args.publish:
        print(json.dumps(publish(args.stage, token), indent=2))
    elif args.parent_repo:
        stage(args.parent_repo, args.stage, token, args.revision)
    else:
        parser.error("staging requires --parent-repo")


if __name__ == "__main__":
    main()
