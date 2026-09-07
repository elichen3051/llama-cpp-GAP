"""Read-only pixel fingerprints and auditable image/question cluster statistics."""

from __future__ import annotations

import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import math
import platform
import struct
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy
import pyarrow
import pyarrow.parquet as pq
import scipy
from PIL import Image, ImageOps
from PIL import __version__ as pillow_version

OUT = Path(__file__).resolve().parent


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_question(value):
    return " ".join(unicodedata.normalize("NFC", value or "").split())


def fingerprint(data):
    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        source_mode = decoded.mode
        source_size = list(decoded.size)
        legacy = hashlib.sha1(decoded.tobytes()).hexdigest()
        exif_orientation = decoded.getexif().get(274)
        n_frames = getattr(decoded, "n_frames", 1)
        rgb = ImageOps.exif_transpose(decoded).convert("RGB")
        payload = rgb.tobytes()
        rgba = ImageOps.exif_transpose(decoded).convert("RGBA")
        rgba_sha256 = hashlib.sha256(
            b"RGBA\x00" + struct.pack(">QQ", rgba.width, rgba.height) + rgba.tobytes()
        ).hexdigest()
        h = hashlib.sha256(
            b"RGB\x00" + struct.pack(">QQ", rgb.width, rgb.height) + payload
        ).hexdigest()
        return {
            "rgba_sha256": rgba_sha256,
            "exact_sha256": h,
            "width": rgb.width,
            "height": rgb.height,
            "mode": "RGB",
            "encoded_sha256": hashlib.sha256(data).hexdigest(),
            "encoded_bytes": len(data),
            "source_mode": source_mode,
            "source_size": source_size,
            "exif_orientation": exif_orientation,
            "n_frames": n_frames,
            "legacy_decoded_native_mode_sha1": legacy,
            "rgb_pixel_bytes_only_sha1": hashlib.sha1(payload).hexdigest(),
            "phash": str(imagehash.phash(rgb, hash_size=8, highfreq_factor=4)),
        }


CACHE = {}
LOCK = threading.Lock()


def read_config(entries):
    config = entries[0]["config"]
    split = entries[0]["split"]
    rows = []
    for entry in sorted(entries, key=lambda x: x["remote_path"]):
        if (entry["config"], entry["split"]) != (config, split):
            raise ValueError("Cannot combine different configs or splits")
        pf = pq.ParquetFile(entry["local_cache_path"])
        required = {
            "source",
            "item_id",
            "origin_id",
            "question",
            "images",
            "num_images",
            "category",
            "ref_answer",
        }
        missing = required - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f"{config}: missing prepared columns: {sorted(missing)}")
        for batch in pf.iter_batches(batch_size=8, use_threads=False):
            for raw in batch.to_pylist():
                index = len(rows)
                images = []
                for position, img in enumerate(raw["images"]):
                    data = img.get("bytes")
                    if data is None:
                        raise ValueError(
                            f"Nonembedded image at {config}/{index}/{position}"
                        )
                    encoded = hashlib.sha256(data).hexdigest()
                    with LOCK:
                        fp = CACHE.get(encoded)
                    if fp is None:
                        fp = fingerprint(data)
                        with LOCK:
                            if len(CACHE) >= 16384:
                                CACHE.clear()
                            CACHE[encoded] = fp
                    images.append(
                        {**fp, "image_index": position, "stored_path": img.get("path")}
                    )
                row = {k: v for k, v in raw.items() if k != "images"}
                row.update(
                    {
                        "uid": f"{config}/{split}/{index}",
                        "config": config,
                        "split": split,
                        "row_index": index,
                        "family": config.rsplit("-subsample-", 1)[0],
                        "question_normalized": normalize_question(raw["question"]),
                        "question_identity_observable": bool(
                            normalize_question(raw["question"])
                        ),
                        "images": images,
                        "observed_num_images": len(images),
                    }
                )
                if row["num_images"] != len(images):
                    raise ValueError("num_images mismatch " + row["uid"])
                hashes = [im["exact_sha256"] for im in images]
                row["ordered_image_list_sha256"] = (
                    stable_hash(hashes)
                    if hashes
                    else stable_hash(["no-image", row["uid"]])
                )
                row["unordered_image_multiset_sha256"] = (
                    stable_hash(sorted(hashes))
                    if hashes
                    else row["ordered_image_list_sha256"]
                )
                row["same_source_item_key"] = stable_hash(
                    [row["source"], row["item_id"], row["origin_id"]]
                )
                row["content_identity"] = stable_hash(
                    [
                        row["source"],
                        row["item_id"],
                        row["origin_id"],
                        row["question"],
                        stable_hash(hashes),
                        row["ref_answer"],
                    ]
                )
                rows.append(row)
    if len(rows) != sum(e["rows"] for e in entries):
        raise ValueError("Row count changed in " + config)
    print(
        json.dumps(
            {
                "event": "fingerprinted",
                "config": config,
                "rows": len(rows),
                "images": sum(len(r["images"]) for r in rows),
            }
        ),
        flush=True,
    )
    return rows


def grouped(rows, kind):
    result = defaultdict(list)
    for row in rows:
        if kind == "exact_image":
            keys = set(im["exact_sha256"] for im in row["images"])
        elif kind == "phash":
            keys = set(im["phash"] for im in row["images"])
        elif kind == "ordered_image_list":
            keys = [row["ordered_image_list_sha256"]]
        elif kind == "image_list_and_question":
            keys = (
                [
                    stable_hash(
                        [row["ordered_image_list_sha256"], row["question_normalized"]]
                    )
                ]
                if row["question_normalized"] and row["images"]
                else [row["uid"]]
            )
        elif kind == "unordered_image_multiset":
            keys = [row["unordered_image_multiset_sha256"]]
        else:
            raise ValueError(kind)
        for key in keys:
            result[key].append(row)
    return dict(result)


def connected_groups(rows):
    parent = {r["uid"]: r["uid"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for r in rows:
        for key in set(im["exact_sha256"] for im in r["images"]):
            if key in owner:
                parent[find(r["uid"])] = find(owner[key])
            else:
                owner[key] = r["uid"]
    groups = defaultdict(list)
    for r in rows:
        groups[find(r["uid"])].append(r)
    return {
        stable_hash(sorted(x["uid"] for x in members)): members
        for members in groups.values()
    }


def group_record(key, members, kind):
    qs = {r["question_normalized"] for r in members if r["question_normalized"]}
    uids = sorted(r["uid"] for r in members)
    return {
        "group_id": key,
        "kind": kind,
        "row_count": len(members),
        "row_uids": uids,
        "distinct_nonempty_normalized_questions": len(qs),
        "different_textual_questions": len(qs) > 1,
        "blank_question_rows": sum(not r["question_normalized"] for r in members),
        "distinct_source_item_ids": len({(r["source"], r["item_id"]) for r in members}),
        "distinct_ordered_image_lists": len(
            {r["ordered_image_list_sha256"] for r in members}
        ),
        "configs": dict(Counter(r["config"] for r in members)),
        "families": dict(Counter(r["family"] for r in members)),
        "categories": dict(Counter(r.get("category") or "(missing)" for r in members)),
    }


def group_stats(groups):
    sizes = [len(v) for v in groups.values()]
    dups = [v for v in groups.values() if len(v) > 1]
    multi = [
        v
        for v in dups
        if len({r["question_normalized"] for r in v if r["question_normalized"]}) > 1
    ]
    return {
        "groups": len(groups),
        "duplicate_groups": len(dups),
        "rows_in_duplicate_groups": len({r["uid"] for v in dups for r in v}),
        "distinct_textual_question_clusters": len(multi),
        "rows_in_distinct_question_clusters": len({r["uid"] for v in multi for r in v}),
        "duplicate_groups_with_unknown_question_identity": sum(
            any(not r["question_normalized"] for r in v) for v in dups
        ),
        "max_cluster_rows": max(sizes, default=0),
        "cluster_size_distribution": dict(sorted(Counter(sizes).items())),
    }


def scenario(sizes, rho):
    n = sum(sizes)
    quadratic = sum(m * m for m in sizes)
    de = 1 + rho * (quadratic / n - 1) if n else 1
    return {
        "assumed_icc": rho,
        "rows": n,
        "clusters": len(sizes),
        "sum_cluster_sizes_squared": quadratic,
        "design_effect": de,
        "hypothetical_n_eff": n / de,
        "se_multiplier": math.sqrt(de),
    }


def summarize(rows):
    result = {
        "rows": len(rows),
        "image_occurrences": sum(len(r["images"]) for r in rows),
        "no_image_rows": sum(not r["images"] for r in rows),
        "multi_image_rows": sum(len(r["images"]) > 1 for r in rows),
        "blank_question_rows": sum(not r["question_normalized"] for r in rows),
        "within_row_repeated_image_rows": sum(
            len({im["exact_sha256"] for im in r["images"]}) < len(r["images"])
            for r in rows
        ),
        "image_count_distribution": dict(Counter(len(r["images"]) for r in rows)),
        "category_counts": dict(
            Counter(r.get("category") or "(missing)" for r in rows)
        ),
    }
    for kind in (
        "exact_image",
        "ordered_image_list",
        "unordered_image_multiset",
        "image_list_and_question",
    ):
        result[kind] = group_stats(grouped(rows, kind))
    conn = connected_groups(rows)
    result["shared_image_connected_components"] = group_stats(conn)
    result["hypothetical_icc_scenarios"] = [
        scenario([len(v) for v in conn.values()], rho) for rho in (0, 0.1, 0.3, 0.5, 1)
    ]
    candidates = {
        k: v
        for k, v in grouped(rows, "phash").items()
        if len(v) > 1
        and len(
            {im["exact_sha256"] for r in v for im in r["images"] if im["phash"] == k}
        )
        > 1
    }
    result["phash_nonexact_candidates"] = group_stats(candidates)
    return result


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(name, records):
    with (OUT / name).open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(name, records):
    if not records:
        (OUT / name).write_text("")
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in r.items()
                }
            )


def overlap(a, b):
    ia = {im["exact_sha256"] for r in a for im in r["images"]}
    ib = {im["exact_sha256"] for r in b for im in r["images"]}
    oa = {r["ordered_image_list_sha256"] for r in a}
    ob = {r["ordered_image_list_sha256"] for r in b}
    ca = {r["content_identity"] for r in a}
    cb = {r["content_identity"] for r in b}
    sa = {r["same_source_item_key"] for r in a}
    sb = {r["same_source_item_key"] for r in b}
    nested = (
        a[0]["family"] == b[0]["family"]
        and a[0]["config"].endswith("-100")
        and b[0]["config"].endswith("-500")
    )
    return {
        "ordered_prefix_equal": (
            [r["content_identity"] for r in a]
            == [r["content_identity"] for r in b[: len(a)]]
        )
        if nested
        else None,
        "tail_rows_sharing_any_image_with_prefix": sum(
            any(im["exact_sha256"] in ia for im in r["images"]) for r in b[len(a) :]
        )
        if nested
        else None,
        "config_a": a[0]["config"],
        "config_b": b[0]["config"],
        "rows_a": len(a),
        "rows_b": len(b),
        "shared_exact_images": len(ia & ib),
        "unique_images_a": len(ia),
        "unique_images_b": len(ib),
        "shared_ordered_image_lists": len(oa & ob),
        "shared_same_source_item_keys": len(sa & sb),
        "shared_identical_content_rows": len(ca & cb),
        "all_a_content_in_b": ca <= cb,
        "rows_a_sharing_any_image_with_b": sum(
            any(im["exact_sha256"] in ib for im in r["images"]) for r in a
        ),
        "rows_b_sharing_any_image_with_a": sum(
            any(im["exact_sha256"] in ia for im in r["images"]) for r in b
        ),
    }


def run(output, workers=2):
    global OUT
    OUT = Path(output)
    CACHE.clear()
    metadata = json.loads((OUT / "metadata.json").read_text())
    entries = defaultdict(list)
    for entry in metadata["selected_files"]:
        entries[entry["config"]].append(entry)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [
            r
            for batch in pool.map(read_config, [entries[k] for k in sorted(entries)])
            for r in batch
        ]
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row UIDs")
    write_jsonl("rows.jsonl", rows)
    byconfig = defaultdict(list)
    for r in rows:
        byconfig[r["config"]].append(r)
    configs = {c: summarize(rs) for c, rs in sorted(byconfig.items())}
    unique = {r["content_identity"]: r for r in rows}
    dedup = list(unique.values())
    by500 = [r for r in rows if r["config"].endswith("-500")]
    by100 = [r for r in rows if r["config"].endswith("-100")]
    summary = {
        "dataset": metadata["repo"],
        "revision": metadata["revision"],
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "configs": configs,
        "global_all_configs_including_nested_rows": summarize(rows),
        "global_content_deduplicated": summarize(dedup),
        "global_100_only": summarize(by100),
        "global_500_only": summarize(by500),
    }
    write_json("summary.json", summary)
    duplicates = []
    maps = []
    concentration = []
    scenarios = []
    scopes = {
        **byconfig,
        "ALL_CONFIGS": rows,
        "ALL_100": by100,
        "ALL_500": by500,
        "ALL_CONTENT_DEDUPLICATED": dedup,
    }
    for scope, rs in scopes.items():
        components = connected_groups(rs)
        component_by_uid = {r["uid"]: key for key, v in components.items() for r in v}
        for kind in (
            "exact_image",
            "ordered_image_list",
            "unordered_image_multiset",
            "image_list_and_question",
        ):
            for key, members in grouped(rs, kind).items():
                if len(members) > 1:
                    duplicates.append(
                        {"scope": scope, **group_record(key, members, kind)}
                    )
        for key, members in components.items():
            if len(members) > 1:
                duplicates.append(
                    {
                        "scope": scope,
                        **group_record(
                            key, members, "shared_image_connected_component"
                        ),
                    }
                )
        if scope in byconfig:
            exact = grouped(rs, "exact_image")
            in_dup = {r["uid"] for v in exact.values() if len(v) > 1 for r in v}
            in_multi = {
                r["uid"]
                for v in exact.values()
                if len(
                    {r["question_normalized"] for r in v if r["question_normalized"]}
                )
                > 1
                for r in v
            }
            for r in rs:
                maps.append(
                    {
                        "config": scope,
                        "split": r["split"],
                        "row_index": r["row_index"],
                        "uid": r["uid"],
                        "source": r["source"],
                        "item_id": r["item_id"],
                        "origin_id": r["origin_id"],
                        "category": r["category"],
                        "num_images": r["num_images"],
                        "exact_image_sha256s": [
                            im["exact_sha256"] for im in r["images"]
                        ],
                        "ordered_image_list_cluster_id": r["ordered_image_list_sha256"],
                        "unordered_image_multiset_cluster_id": r[
                            "unordered_image_multiset_sha256"
                        ],
                        "within_config_shared_image_component_id": component_by_uid[
                            r["uid"]
                        ],
                    }
                )
            for category in sorted({r["category"] or "(missing)" for r in rs}):
                cr = [r for r in rs if (r["category"] or "(missing)") == category]
                concentration.append(
                    {
                        "config": scope,
                        "category": category,
                        "rows": len(cr),
                        "rows_in_any_image_duplicate_groups": sum(
                            r["uid"] in in_dup for r in cr
                        ),
                        "rows_in_distinct_text_question_image_clusters": sum(
                            r["uid"] in in_multi for r in cr
                        ),
                        "fraction_of_config_rows": len(cr) / len(rs),
                    }
                )
            for sc in configs[scope]["hypothetical_icc_scenarios"]:
                scenarios.append({"config": scope, **sc})
    write_jsonl("duplicate_groups.jsonl", duplicates)
    global_components = {
        r["content_identity"]: key
        for key, v in connected_groups(dedup).items()
        for r in v
    }
    rowbyuid = {r["uid"]: r for r in rows}
    for m in maps:
        m["global_content_deduplicated_shared_image_component_id"] = global_components[
            rowbyuid[m["uid"]]["content_identity"]
        ]
    write_csv("row_cluster_mapping.csv", maps)
    write_csv("category_concentration.csv", concentration)
    write_csv("hypothetical_icc_scenarios.csv", scenarios)
    candidates = []
    for scope, rs in scopes.items():
        for key, members in grouped(rs, "phash").items():
            imgs = [im for r in members for im in r["images"] if im["phash"] == key]
            exact = sorted({im["exact_sha256"] for im in imgs})
            if len(members) > 1 and len(exact) > 1:
                candidates.append(
                    {
                        "scope": scope,
                        **group_record(
                            key, members, "phash_candidate_not_confirmed_duplicate"
                        ),
                        "exact_images": exact,
                        "dimensions": sorted(
                            {(im["width"], im["height"]) for im in imgs}
                        ),
                    }
                )
    write_jsonl("phash_candidates.jsonl", candidates)
    pairs = []
    names = sorted(byconfig)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append(overlap(byconfig[a], byconfig[b]))
    write_csv("all_config_pairwise_overlap.csv", pairs)
    nested = [
        p
        for p in pairs
        if p["config_a"].rsplit("-subsample-", 1)[0]
        == p["config_b"].rsplit("-subsample-", 1)[0]
    ]
    write_json("overlap_100_500.json", nested)
    write_csv("overlap_100_500.csv", nested)
    cross = [
        g
        for g in duplicates
        if g["scope"] == "ALL_CONTENT_DEDUPLICATED"
        and g["kind"] == "exact_image"
        and len(g["families"]) > 1
    ]
    write_json("cross_source_exact_image_groups.json", cross)
    flattened = []
    for c, s in configs.items():
        rec = {
            "config": c,
            "rows": s["rows"],
            "image_occurrences": s["image_occurrences"],
            "multi_image_rows": s["multi_image_rows"],
            "blank_question_rows": s["blank_question_rows"],
        }
        for kind in (
            "exact_image",
            "ordered_image_list",
            "unordered_image_multiset",
            "shared_image_connected_components",
            "phash_nonexact_candidates",
        ):
            for k, v in s[kind].items():
                rec[kind + "_" + k] = v
        flattened.append(rec)
    write_csv("per_config_summary.csv", flattened)
    provenance = {
        "python": sys.version,
        "platform": platform.platform(),
        "Pillow": pillow_version,
        "imagehash": imagehash.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "cpu_workers": workers,
        "decode": "Pillow first frame; EXIF transpose; RGB conversion; no resize for exact hash",
        "exact_fingerprint": 'SHA256(b"RGB\\0" + uint64_be(width) + uint64_be(height) + RGB pixel bytes)',
        "phash": "ImageHash 4.3.2 phash(hash_size=8,highfreq_factor=4), identical values only; not proof of identity",
        "original_sha1_note": "legacy_decoded_native_mode_sha1 is SHA1(decoded.tobytes()) without dimensions/mode; rgb_pixel_bytes_only_sha1 records RGB-only alternative; neither is primary identity",
        "exif_orientations": dict(
            Counter(str(im["exif_orientation"]) for r in rows for im in r["images"])
        ),
        "multi_frame_images": sum(
            im["n_frames"] != 1 for r in rows for im in r["images"]
        ),
        "source_modes": dict(
            Counter(im["source_mode"] for r in rows for im in r["images"])
        ),
        "paper": {
            "url": "https://arxiv.org/pdf/2411.00640",
            "title": "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations",
            "author": "Evan Miller",
            "related_sections": ["2.2", "4.2"],
        },
        "errors": [],
    }
    write_json("analysis_provenance.json", provenance)
    print(
        json.dumps(
            {
                "event": "complete",
                "rows": len(rows),
                "content_deduplicated_rows": len(dedup),
                "cross_source_exact_image_groups": len(cross),
                "nonexact_phash_global_candidates": sum(
                    c["scope"] == "ALL_CONTENT_DEDUPLICATED" for c in candidates
                ),
            }
        ),
        flush=True,
    )
