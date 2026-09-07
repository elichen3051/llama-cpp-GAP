#!/usr/bin/env python3
"""Prepare, independently review, then publish paired 500/100 replacements."""

import argparse
import concurrent.futures
import datetime
import json
import re
import shutil
import subprocess
import traceback
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from replacement import Fingerprints, exact_row_digest, repair_units, row_digest
from scan import file_sha256, save_json, validate_manifest

STANDARD = "mmmu-pro-standard-10"
VISION = "mmmu-pro-vision"


def safe_config(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("Unsafe config name: " + repr(value))
    return value


def read_config(entries):
    rows = []
    schema = None
    for entry in sorted(entries, key=lambda e: e["remote_path"]):
        table = pq.read_table(entry["local_cache_path"])
        if schema is None:
            schema = table.schema
        elif not schema.equals(table.schema, check_metadata=True):
            raise ValueError("Inconsistent shard schemas")
        rows.extend(table.to_pylist())
    return rows, schema


def validate_pair(standard, vision):
    if len(standard) != len(vision):
        raise ValueError("MMMU view lengths differ")
    for a, b in zip(standard, vision):
        for field in ("item_id", "origin_id", "category"):
            if a[field] != b[field]:
                raise ValueError("MMMU view mismatch: " + field)


def metadata_strata(path, metadata):
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if data["dataset"] != metadata["repo"] or data["revision"] != metadata["revision"]:
        raise ValueError("Strata belong to another dataset revision")
    result = {}
    for row in data["rows"]:
        key = (row["family"], row["item_id"])
        if key in result:
            raise ValueError("Duplicate strata key")
        result[key] = row
    return result


def prepare(args):
    output = args.out.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "preparing",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    save_json(output / "status.json", status)
    try:
        inputs = json.loads(args.inputs.read_text())
        pools = {
            safe_config(k): safe_config(v) for k, v in inputs["source_pools"].items()
        }
        names = {e["config"] for e in inputs["selected_files"]}
        families = {
            name.removesuffix("-subsample-500")
            for name in names
            if name.endswith("-subsample-500")
        }
        if set(pools) != families:
            raise ValueError("Source pool map must cover every selected 500 config")
        if (STANDARD in families) != (VISION in families):
            raise ValueError("MMMU Standard and Vision must be processed together")
        allowed = set(pools.values()) | {
            f"{family}-subsample-{n}" for family in families for n in (100, 500)
        }
        if not allowed <= names:
            raise ValueError("Missing source pools or paired 100 configs")
        selected = {
            **inputs,
            "selected_files": [
                e for e in inputs["selected_files"] if e["config"] in allowed
            ],
        }
        metadata = validate_manifest(selected, sorted(allowed), "train")
        entries = defaultdict(list)
        for entry in metadata["selected_files"]:
            if PurePosixPath(entry["remote_path"]).parts[0] != entry["config"]:
                raise ValueError("Manifest path does not match config")
            entries[entry["config"]].append(entry)
        strata = metadata_strata(args.strata, metadata)
        upload = output / "upload"
        upload.mkdir()
        scripts = output / "scripts"
        scripts.mkdir()
        for name in (
            "replace_subsamples.py",
            "replacement.py",
            "analysis.py",
            "scan.py",
            "requirements.txt",
        ):
            shutil.copyfile(Path(__file__).with_name(name), scripts / name)
        shutil.copyfile(args.inputs, output / "inputs.json")
        if args.strata:
            shutil.copyfile(args.strata, output / "strata.json")
        fingerprints = Fingerprints()

        def process(family):
            views = [family, VISION] if family == STANDARD else [family]
            originals, pool_rows, schemas, previous_100 = {}, {}, {}, {}
            source_by_id = {}
            for view in views:
                original, schema = read_config(entries[view + "-subsample-500"])
                prefix, prefix_schema = read_config(entries[view + "-subsample-100"])
                source, source_schema = read_config(entries[pools[view]])
                if len(original) != 500 or len(prefix) != 100:
                    raise ValueError("Expected exactly 500/100 rows: " + view)
                if not schema.equals(
                    prefix_schema, check_metadata=True
                ) or not schema.equals(source_schema, check_metadata=True):
                    raise ValueError("Source/500/100 schemas differ: " + view)
                if [exact_row_digest(r) for r in prefix] != [
                    exact_row_digest(r) for r in original[:100]
                ]:
                    raise ValueError("Original 100 is not an exact prefix: " + view)
                by_id = {r["item_id"]: r for r in source}
                if len(by_id) != len(source):
                    raise ValueError("Source item IDs are not unique: " + view)
                for row in original:
                    if row["item_id"] not in by_id or row_digest(row) != row_digest(
                        by_id[row["item_id"]]
                    ):
                        raise ValueError(
                            "Original item does not match source pool: "
                            + view
                            + "/"
                            + str(row["item_id"])
                        )
                originals[view], pool_rows[view], schemas[view], previous_100[view] = (
                    original,
                    source,
                    schema,
                    prefix,
                )
                source_by_id[view] = by_id
            if family == STANDARD:
                validate_pair(originals[STANDARD], originals[VISION])
                if set(source_by_id[STANDARD]) != set(source_by_id[VISION]):
                    raise ValueError("MMMU source pools have different item IDs")
                aligned = [
                    source_by_id[VISION][r["item_id"]] for r in pool_rows[STANDARD]
                ]
                validate_pair(pool_rows[STANDARD], aligned)
            original_units = [
                {view: originals[view][i] for view in views} for i in range(500)
            ]
            candidates = [
                {view: source_by_id[view][row["item_id"]] for view in views}
                for row in pool_rows[family]
            ]

            def describe(unit):
                row = unit[family]
                extra = strata.get((family, row["item_id"]))
                if extra is not None and (
                    extra["origin_id"] != row["origin_id"]
                    or extra["question"] != row["question"]
                ):
                    raise ValueError("Strata row does not match prepared question")
                return {
                    "category": row["category"] or "",
                    "task": extra["task"] if extra else "",
                    "dataset_name": extra["dataset_name"] if extra else "",
                }

            final, changes, diagnostics = repair_units(
                original_units,
                candidates,
                family,
                args.seed,
                args.phash_distance,
                fingerprints,
                describe,
            )
            final_by_view = {view: [unit[view] for unit in final] for view in views}
            if family == STANDARD:
                validate_pair(final_by_view[STANDARD], final_by_view[VISION])
                validate_pair(
                    final_by_view[STANDARD][:100], final_by_view[VISION][:100]
                )
            files, deletes, records, summaries = [], [], [], []
            for view in views:
                final_rows = final_by_view[view]
                changed_slots = {change["slot"] for change in changes}
                for i, row in enumerate(final_rows):
                    if row_digest(row) != row_digest(
                        source_by_id[view][row["item_id"]]
                    ):
                        raise ValueError("Final row is not a source-pool row")
                    if i not in changed_slots and exact_row_digest(
                        row
                    ) != exact_row_digest(originals[view][i]):
                        raise ValueError("Retained original row changed")
                view_changes = []
                for change in changes:
                    slot = change["slot"]
                    before, after = originals[view][slot], final_rows[slot]
                    value = {
                        **change,
                        "family": view,
                        "source_pool": pools[view],
                        "old_origin_id": before["origin_id"],
                        "new_origin_id": after["origin_id"],
                        "old_row_sha256": exact_row_digest(before),
                        "new_row_sha256": exact_row_digest(after),
                        "source_pool_row_sha256": row_digest(after),
                        "in_pilot_100": slot < 100,
                    }
                    records.append(value)
                    view_changes.append(value)
                for n in (500, 100):
                    config = f"{view}-subsample-{n}"
                    selected_rows = final_rows[:n]
                    before = originals[view] if n == 500 else previous_100[view]
                    unchanged = [exact_row_digest(r) for r in before] == [
                        exact_row_digest(r) for r in selected_rows
                    ]
                    if unchanged:
                        files.extend(
                            {**entry, "changed": False} for entry in entries[config]
                        )
                        continue
                    relative = f"{config}/train-00000-of-00001.parquet"
                    path = upload / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    table = pa.Table.from_pylist(selected_rows, schema=schemas[view])
                    pq.write_table(
                        table, path, compression="snappy", row_group_size=100
                    )
                    reread = pq.read_table(path)
                    if not table.equals(reread, check_metadata=True):
                        raise ValueError("Parquet round-trip changed rows or schema")
                    files.append(
                        {
                            "config": config,
                            "split": "train",
                            "remote_path": relative,
                            "local_cache_path": str(path),
                            "rows": n,
                            "size": path.stat().st_size,
                            "sha256": file_sha256(path),
                            "changed": True,
                            "dataset_bytes": table.nbytes,
                        }
                    )
                    deletes.extend(
                        entry["remote_path"]
                        for entry in entries[config]
                        if entry["remote_path"] != relative
                    )
                summaries.append(
                    {
                        "family": view,
                        "source_pool": pools[view],
                        "pool_rows": len(pool_rows[view]),
                        "replaced_500": len(view_changes),
                        "replaced_100": sum(c["in_pilot_100"] for c in view_changes),
                        "retained_original_500": 500 - len(view_changes),
                        "category_before": dict(
                            Counter(
                                r["category"] or "(missing)" for r in originals[view]
                            )
                        ),
                        "category_after": dict(
                            Counter(r["category"] or "(missing)" for r in final_rows)
                        ),
                        "stratum_tiers": diagnostics["replacement_stratum_tiers"],
                    }
                )
            save_json(
                output / (family + "-selection.json"),
                {"changes": records, "diagnostics": diagnostics},
            )
            print(
                json.dumps(
                    {
                        "event": "prepared",
                        "family": family,
                        "paired_views": views,
                        "replacements": len(changes),
                    }
                ),
                flush=True,
            )
            return files, deletes, records, summaries

        work = sorted(family for family in families if family != VISION)
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            for result in executor.map(process, work):
                results.append(result)
        final_files = [entry for result in results for entry in result[0]]
        changes = [entry for result in results for entry in result[2]]
        summaries = [entry for result in results for entry in result[3]]
        recipe = {
            "schema": "prepared-subsample-image-replacement-v1",
            "dataset": metadata["repo"],
            "parent_revision": metadata["revision"],
            "source_pools": pools,
            "seed": args.seed,
            "phash_max_distance": args.phash_distance,
            "scope": "within each subset; MMMU Standard/Vision retain aligned item IDs in both 100 and 500",
            "selection": "Keep earliest original nonconflicting rows at their original slots, reserve all survivors, then replace conflicting slots with unused source-pool items. Prefer same category/task/dataset and image count; stable SHA256 ranking breaks ties.",
            "perceptual_policy": "pHash distance exclusions are conservative candidates, not assertions that every excluded pair is an identical photograph.",
            "summaries": summaries,
            "changes": changes,
            "script_sha256": {
                p.name: file_sha256(p) for p in sorted(scripts.iterdir())
            },
        }
        provenance_path = (
            "sampling/image-cluster-replacements-" + metadata["revision"][:12] + ".json"
        )
        (upload / "sampling").mkdir()
        save_json(upload / provenance_path, recipe)
        from huggingface_hub import DatasetCard, hf_hub_download

        card_path = hf_hub_download(
            metadata["repo"],
            "README.md",
            repo_type="dataset",
            revision=metadata["revision"],
        )
        card = DatasetCard(Path(card_path).read_text())
        by_config = defaultdict(list)
        for entry in final_files:
            by_config[entry["config"]].append(entry)
        for entry in card.data["dataset_info"]:
            config = entry["config_name"]
            if config not in by_config or not any(
                e["changed"] for e in by_config[config]
            ):
                continue
            current = by_config[config]
            entry["download_size"] = sum(e["size"] for e in current)
            entry["dataset_size"] = sum(e["dataset_bytes"] for e in current)
            for split in entry["splits"]:
                if split["name"] == "train":
                    split["num_examples"] = sum(e["rows"] for e in current)
                    split["num_bytes"] = entry["dataset_size"]
        card.text += (
            "\n\n## Image diversity update\n\nThe 500-row subsets keep original nonconflicting rows in place and replace excluded slots from their corresponding source pools. Each 100-row subset is the exact first 100 rows of its updated 500-row subset. MMMU-Pro Standard and Vision preserve the same item IDs and order in both sizes. Exact RGB matches and conservative 64-bit pHash distances up to "
            + str(args.phash_distance)
            + " are excluded within each subset. This changes the sampling design; it does not establish measured statistical independence. Source pools and unrelated configs are unchanged. See the [replacement recipe and row ledger]("
            + provenance_path
            + ") for the seed, source revision, fallback strata, and replaced item IDs.\n"
        )
        card.save(upload / "README.md")
        plan = {
            "dataset": metadata["repo"],
            "parent_revision": metadata["revision"],
            "files": [
                {
                    "path": str(p.relative_to(upload)),
                    "size": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in sorted(upload.rglob("*"))
                if p.is_file()
            ],
            "deletions": sorted({p for result in results for p in result[1]}),
        }
        save_json(output / "publication-plan.json", plan)
        save_json(
            output / "draft-manifest.json",
            {
                "dataset": metadata["repo"],
                "parent_revision": metadata["revision"],
                "snapshot_kind": "unpublished_local_draft",
                "selected_files": final_files,
                "source_pools": pools,
                "summaries": summaries,
                "phash_max_distance": args.phash_distance,
                "publication_plan_sha256": file_sha256(
                    output / "publication-plan.json"
                ),
            },
        )
        save_json(output / "replacement-ledger.json", changes)
        status.update(
            {
                "status": "ready_for_independent_review",
                "dataset": metadata["repo"],
                "parent_revision": metadata["revision"],
                "changed_files": len(plan["files"]),
                "replaced_rows_500": sum(s["replaced_500"] for s in summaries),
            }
        )
        save_json(output / "status.json", status)
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        save_json(output / "status.json", status)
        (output / "error.log").write_text(traceback.format_exc())
        raise


def checked_plan(output, review_path):
    plan_path = output / "publication-plan.json"
    plan = json.loads(plan_path.read_text())
    review = json.loads(review_path.read_text())
    if review.get("status") != "approved" or review.get(
        "publication_plan_sha256"
    ) != file_sha256(plan_path):
        raise ValueError(
            "An independent approval for this exact publication plan is required"
        )
    draft = json.loads((output / "draft-manifest.json").read_text())
    if (
        draft["dataset"] != plan["dataset"]
        or draft["parent_revision"] != plan["parent_revision"]
    ):
        raise ValueError("Draft and publication plan disagree")
    allowed = {safe_config(entry["config"]) for entry in draft["selected_files"]}
    if not allowed or any(
        not re.fullmatch(r".+-subsample-(100|500)", name) for name in allowed
    ):
        raise ValueError("Only 100/500 target configs may be published")
    seen = set()
    for entry in plan["files"]:
        relative = PurePosixPath(entry["path"])
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != entry["path"]
        ):
            raise ValueError("Unsafe publication path")
        if entry["path"] in seen:
            raise ValueError("Repeated publication path")
        seen.add(entry["path"])
        if (
            relative.parts[0] not in allowed
            and entry["path"] != "README.md"
            and relative.parts[0] != "sampling"
        ):
            raise ValueError("Publication touches an unrelated config")
        path = output / "upload" / relative
        if not path.resolve().is_relative_to((output / "upload").resolve()):
            raise ValueError("Publication file escapes upload directory")
        if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("Reviewed publication content changed: " + entry["path"])
    for name in plan["deletions"]:
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or relative.parts[0] not in allowed
            or ".." in relative.parts
            or str(relative) != name
        ):
            raise ValueError("Deletion outside target configs")
        if name in seen:
            raise ValueError("Repeated or conflicting deletion path")
        seen.add(name)
    return plan


def verify_publication(output, plan, revision):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    before = api.dataset_info(
        plan["dataset"], revision=plan["parent_revision"], files_metadata=True
    )
    after = api.dataset_info(plan["dataset"], revision=revision, files_metadata=True)
    old = {s.rfilename: s for s in before.siblings}
    new = {s.rfilename: s for s in after.siblings}
    changed = {e["path"]: e for e in plan["files"]}
    deleted = set(plan["deletions"])
    if set(new) != (set(old) - deleted) | set(changed):
        raise ValueError("Published repository paths differ from the reviewed plan")
    for name, sibling in new.items():
        if name in changed:
            expected = changed[name]
            if sibling.lfs:
                digest = sibling.lfs.sha256
            else:
                path = hf_hub_download(
                    plan["dataset"], name, repo_type="dataset", revision=revision
                )
                digest = file_sha256(path)
            if sibling.size != expected["size"] or digest != expected["sha256"]:
                raise ValueError("Published content differs: " + name)
        else:
            original = old[name]

            def identity(value):
                return (
                    value.size,
                    value.blob_id,
                    value.lfs.sha256 if value.lfs else None,
                )

            if identity(sibling) != identity(original):
                raise ValueError("An unplanned file changed: " + name)
    receipt_path = output / "publication-receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt.update(
        {
            "status": "published_verified",
            "dataset": plan["dataset"],
            "parent_revision": plan["parent_revision"],
            "revision": after.sha,
            "commit_url": f"https://huggingface.co/datasets/{plan['dataset']}/commit/{after.sha}",
            "publication_plan_sha256": file_sha256(output / "publication-plan.json"),
            "verified_changed_files": len(changed),
            "verified_unchanged_files": len(set(new) - set(changed)),
        }
    )
    save_json(receipt_path, receipt)
    save_json(output / "status.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def publish(args):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    output = args.out.expanduser().resolve()
    plan = checked_plan(output, args.review)
    receipt_path = output / "publication-receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("publication_plan_sha256") != file_sha256(
            output / "publication-plan.json"
        ):
            raise ValueError("Existing receipt belongs to another plan")
        if receipt.get("revision"):
            return verify_publication(output, plan, receipt["revision"])
        raise ValueError(
            "A previous publication may have reached the Hub; inspect its commit and use verify --revision"
        )
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True
    ).strip()
    api = HfApi()
    before = api.dataset_info(plan["dataset"]).sha
    if before != plan["parent_revision"]:
        raise ValueError(
            "Hub HEAD changed; rebuild and review against the new revision"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=e["path"], path_or_fileobj=str(output / "upload" / e["path"])
        )
        for e in plan["files"]
    ]
    operations += [
        CommitOperationDelete(path_in_repo=path) for path in plan["deletions"]
    ]
    receipt = {
        "status": "publishing_outcome_not_yet_known",
        "dataset": plan["dataset"],
        "parent_revision": before,
        "publication_plan_sha256": file_sha256(output / "publication-plan.json"),
        "code_commit": code_commit,
    }
    save_json(receipt_path, receipt)
    save_json(output / "status.json", receipt)
    result = api.create_commit(
        repo_id=plan["dataset"],
        repo_type="dataset",
        operations=operations,
        commit_message="Replace shared-image questions and refresh aligned 100-row prefixes",
        revision="main",
        parent_commit=plan["parent_revision"],
    )
    receipt.update(
        {
            "status": "published_pending_verification",
            "revision": result.oid,
            "commit_url": result.commit_url,
        }
    )
    save_json(receipt_path, receipt)
    save_json(output / "status.json", receipt)
    return verify_publication(output, plan, result.oid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--strata", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=1234)
    prep.add_argument("--phash-distance", type=int, default=4)
    prep.add_argument("--workers", type=int, default=2)
    pub = sub.add_parser("publish")
    pub.add_argument("--out", type=Path, required=True)
    pub.add_argument("--review", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--out", type=Path, required=True)
    verify.add_argument("--review", type=Path, required=True)
    verify.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.workers < 1 or not 0 <= args.phash_distance <= 64:
            parser.error("workers must be positive and pHash distance must be 0..64")
        return prepare(args)
    if args.command == "verify":
        output = args.out.expanduser().resolve()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            parser.error("verify requires a pinned 40-hex revision")
        return verify_publication(
            output, checked_plan(output, args.review), args.revision
        )
    return publish(args)


if __name__ == "__main__":
    raise SystemExit(main())
