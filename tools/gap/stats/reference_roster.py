"""Verify native reference identities and bind them to completed metric rows."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from lib.collect_common import dump_stem
from lib.collection_state import comparison_locks, require_completed_attempts
from lib.dataset_fingerprint import dataset_content_hash
from lib.reference_dataset import validate_reference_row
from stats.contracts import content_hash


def _encoded_view(dataset):
    """Disable image decoding without changing stored bytes or row order.

    API: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.cast_column
    """
    if not hasattr(dataset, "features"):
        return dataset
    from datasets import Image
    view = dataset.with_format(None)
    feature = view.features.get("images")
    if not hasattr(feature, "feature") or not isinstance(feature.feature, Image):
        raise ValueError("native reference requires a list of Image features")
    return view.cast_column("images", type(feature)(Image(decode=False)))


def _reference_mode(row):
    """Bind the reported mode to request or default fields covered by ds-v3."""
    request = json.loads(row["generation_request"])
    metadata = json.loads(row["generation_metadata"])
    expected = request.get("enable_thinking")
    if expected is None:
        expected = metadata.get("requested_enable_thinking")
    if expected is None:
        expected = metadata.get("default_enable_thinking")
    mode = row.get("generation_enable_thinking")
    if not isinstance(expected, bool) or not isinstance(mode, bool) or expected != mode:
        raise ValueError("reference mode must match a boolean request or default covered by the dataset hash")
    return mode


def reference_identities(dataset, source_field="source"):
    """Verify native rows and group transitive shared-image links with SciPy.

    The sparse bipartite graph contains item and encoded-image SHA256 vertices. Image-free items stay isolated. Component IDs use sorted stable item IDs, so input order does not affect them. Source labels define quota strata, not independence groups. Recompression, cropping and broader source dependence are not detected.
    API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html
    """
    dataset = _encoded_view(dataset)
    if not len(dataset):
        raise ValueError("reference dataset must contain items")
    identities, seen, modes, images = [], set(), set(), {}
    edges_i, edges_j = [], []
    image_occurrences = 0
    for i, row in enumerate(dataset):
        validate_reference_row(row)
        item_id, source = row["item_id"], row.get(source_field)
        if not isinstance(item_id, str) or item_id in seen:
            raise ValueError("reference item IDs must be distinct strings")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{item_id}: {source_field} must contain a nonempty source stratum")
        modes.add(_reference_mode(row))
        seen.add(item_id)
        hashes = sorted(set(row["image_bytes_sha256"]))
        image_occurrences += len(row["image_bytes_sha256"])
        identities.append({"item_id": item_id, "source_id": source, "image_hashes": hashes})
        for digest in hashes:
            vertex = images.setdefault(digest, len(dataset) + len(images))
            edges_i.append(i)
            edges_j.append(vertex)
    if len(modes) != 1:
        raise ValueError("a reference roster must contain exactly one generation mode")
    size = len(dataset) + len(images)
    graph = csr_array((np.ones(len(edges_i), dtype=np.int8), (edges_i, edges_j)), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    members = {}
    for identity, label in zip(identities, labels[:len(dataset)]):
        members.setdefault(int(label), []).append(identity["item_id"])
    cluster_ids = {label: "image-component:" + content_hash(sorted(ids)) for label, ids in members.items()}
    for identity, label in zip(identities, labels[:len(dataset)]):
        identity["cluster_id"] = cluster_ids[int(label)]
    cluster_sizes = {cluster_ids[label]: len(ids) for label, ids in members.items()}
    return {
        "schema_version": "company-reference-identities-v1", "binding_status": "unbound",
        "mode": "thinking" if next(iter(modes)) else "instruct", "mode_status": "verified_against_hashed_request_or_default", "source_field": source_field,
        "reference_content_hash": dataset_content_hash(dataset),
        "identity_sha256": content_hash(sorted(identities, key=lambda row: row["item_id"])),
        "cluster_method": "encoded_image_sha256_connected_components",
        "image_hash_status": "verified_against_encoded_bytes", "other_dependence_status": "not_inferred",
        "n_items": len(dataset), "n_image_occurrences": image_occurrences, "n_unique_images": len(images),
        "n_clusters": len(members), "cluster_sizes": cluster_sizes,
        "source_counts": dict(Counter(row["source_id"] for row in identities)), "items": identities,
    }


def bind_reference_roster(dataset, collection_dir, expected_item_ids=None, source_field="source"):
    """Bind verified reference identities to completed artifact keys under a reader lock.

    Reproduce the collector's recorded sort and hash its full input view. Partial collections require an explicit expected-ID cohort; no missing or skipped rows are dropped. Metric contents and candidate alignment remain the campaign loader's responsibility.
    API: https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.sort
    """
    root = Path(collection_dir).resolve()
    with comparison_locks([root]):
        require_completed_attempts(root)
        meta = json.loads((root / "collect_meta.json").read_text())
        if meta.get("kind") not in ("vlm_kld_metrics", "llm_kld_metrics") or meta.get("perplexity_window"):
            raise ValueError("roster binding requires item-level metric collections")
        sort_by, sort_desc = meta.get("sort_by"), meta.get("sort_desc", False)
        if (sort_by is not None and not isinstance(sort_by, str)) or not isinstance(sort_desc, bool):
            raise ValueError("invalid recorded collection sort")
        view = _encoded_view(dataset)
        if sort_by:
            view = view.sort([sort_by], reverse=sort_desc)
        identities = reference_identities(view, source_field)
        if meta.get("dataset_content_hash") != identities["reference_content_hash"]:
            raise ValueError("reference content/order differs from the full collected dataset hash")
        all_ids = [row["item_id"] for row in identities["items"]]
        expected = all_ids if expected_item_ids is None else expected_item_ids
        if not isinstance(expected, list) or not expected or any(not isinstance(item, str) for item in expected) or len(set(expected)) != len(expected) or not set(expected) <= set(all_ids):
            raise ValueError("expected item IDs must be distinct IDs from the full reference dataset")
        with (root / "manifest.csv").open(newline="") as stream:
            final_rows = {}
            for row in csv.DictReader(stream):
                idx = int(row["row_idx"])
                if idx < 0 or idx >= len(all_ids) or row["item_id"] != all_ids[idx]:
                    raise ValueError("manifest row index or item ID differs from the sorted reference")
                previous = final_rows.get(idx)
                if previous and not previous["status"].startswith("FAIL_"):
                    raise ValueError("manifest contains a repeated non-retryable row")
                final_rows[idx] = row
        manifest = list(final_rows.values())
        if len(manifest) != len(expected) or {row["item_id"] for row in manifest} != set(expected):
            raise ValueError("completed collection must exactly match the expected item cohort")
        roster = []
        for idx, row in sorted(final_rows.items()):
            if row["status"] != "OK":
                raise ValueError("manifest must contain a successful final status for every expected item")
            roster.append({**identities["items"][idx], "item_key": dump_stem(idx, row["item_id"])})
        return {
            **identities, "binding_status": "completed_collection_verified", "collection_dir": str(root),
            "collect_meta_sha256": content_hash(meta), "expected_item_ids": list(expected),
            "n_bound_items": len(roster), "n_bound_clusters": len({row["cluster_id"] for row in roster}),
            "cell_fields": {"dataset_content_hash": identities["reference_content_hash"], "mode": identities["mode"], "roster": roster},
        }
