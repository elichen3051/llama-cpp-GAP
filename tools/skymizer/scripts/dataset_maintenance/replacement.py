"""Stable slot replacement with conservative, per-view image exclusion."""

import hashlib
import threading
from collections import Counter

from analysis import fingerprint, stable_hash


def row_digest(row, include_seed=False, include_image_paths=False):
    content = {
        k: v
        for k, v in row.items()
        if k != "images" and (include_seed or k != "sample_seed")
    }
    images = []
    for image in row["images"]:
        if image.get("bytes") is None:
            raise ValueError("Replacement inputs require embedded image bytes")
        value = {"sha256": hashlib.sha256(image["bytes"]).hexdigest()}
        if include_image_paths:
            value.update({k: v for k, v in image.items() if k != "bytes"})
        images.append(value)
    content["images"] = images
    return stable_hash(content)


def exact_row_digest(row):
    return row_digest(row, include_seed=True, include_image_paths=True)


def item_id(unit):
    return next(iter(unit.values()))["item_id"]


class Fingerprints:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def row(self, row):
        result = []
        if not row["images"] or row["num_images"] != len(row["images"]):
            raise ValueError("Image count mismatch or empty image list")
        for image in row["images"]:
            data = image["bytes"]
            encoded = hashlib.sha256(data).hexdigest()
            with self.lock:
                value = self.cache.get(encoded)
            if value is None:
                value = fingerprint(data)
                if value["n_frames"] != 1:
                    raise ValueError("Multi-frame images need a separate review")
                with self.lock:
                    self.cache[encoded] = value
            result.append(value)
        return result

    def unit(self, unit):
        return {view: self.row(row) for view, row in unit.items()}


class ImageIndex:
    def __init__(self, distance):
        if not 0 <= distance <= 64:
            raise ValueError("pHash distance must be between 0 and 64")
        self.distance = distance
        self.exact = {}
        self.perceptual = {}

    def conflict(self, signatures):
        for view, images in signatures.items():
            for position, image in enumerate(images):
                owner = self.exact.get((view, image["exact_sha256"]))
                if owner is not None:
                    return {
                        "kind": "exact_rgb",
                        "view": view,
                        "image_index": position,
                        "against_slot": owner,
                    }
                value = int(image["phash"], 16)
                for previous, slot in self.perceptual.get(view, {}).items():
                    distance = (value ^ previous).bit_count()
                    if distance <= self.distance:
                        return {
                            "kind": "conservative_phash_exclusion",
                            "distance": distance,
                            "view": view,
                            "image_index": position,
                            "against_slot": slot,
                        }
        return None

    def add(self, signatures, slot):
        for view, images in signatures.items():
            for image in images:
                self.exact.setdefault((view, image["exact_sha256"]), slot)
                self.perceptual.setdefault(view, {}).setdefault(
                    int(image["phash"], 16), slot
                )


def stratum_tier(candidate, target):
    if candidate["category"] != target["category"]:
        return 3
    if candidate["task"] != target["task"]:
        return 2
    if candidate["dataset_name"] != target["dataset_name"]:
        return 1
    return 0


def repair_units(original, pool, family, seed, distance, fingerprints, describe):
    if len({item_id(unit) for unit in original}) != len(original):
        raise ValueError("Original subset has repeated item IDs")
    if len({item_id(unit) for unit in pool}) != len(pool):
        raise ValueError("Source pool has repeated item IDs")
    index = ImageIndex(distance)
    output = list(original)
    holes = []
    for slot, unit in enumerate(original):
        signatures = fingerprints.unit(unit)
        conflict = index.conflict(signatures)
        if conflict:
            holes.append((slot, conflict))
        else:
            index.add(signatures, slot)
    # Reserve all original survivors before filling holes.
    original_ids = {item_id(unit) for unit in original}
    candidates = [unit for unit in pool if item_id(unit) not in original_ids]
    candidates.sort(key=lambda unit: stable_hash([seed, family, item_id(unit)]))
    used = set()
    rejected = {}
    changes = []
    for slot, conflict in holes:
        previous = original[slot]
        target = describe(previous)
        ranked = sorted(
            candidates,
            key=lambda unit: (
                stratum_tier(describe(unit), target),
                abs(
                    next(iter(unit.values()))["num_images"]
                    - next(iter(previous.values()))["num_images"]
                ),
            ),
        )
        chosen = None
        for candidate in ranked:
            key = item_id(candidate)
            if key in used or key in rejected:
                continue
            signatures = fingerprints.unit(candidate)
            obstruction = index.conflict(signatures)
            if obstruction:
                rejected[key] = obstruction
                continue
            chosen = {
                view: {**row, "sample_seed": seed} for view, row in candidate.items()
            }
            output[slot] = chosen
            used.add(key)
            index.add(signatures, slot)
            changes.append(
                {
                    "slot": slot,
                    "old_item_id": item_id(previous),
                    "new_item_id": key,
                    "reason": conflict,
                    "old_stratum": target,
                    "new_stratum": describe(candidate),
                    "stratum_tier": stratum_tier(describe(candidate), target),
                }
            )
            break
        if chosen is None:
            raise ValueError(
                f"{family}: source pool cannot fill slot {slot} without an image conflict"
            )
    check = ImageIndex(distance)
    for slot, unit in enumerate(output):
        signatures = fingerprints.unit(unit)
        conflict = check.conflict(signatures)
        if conflict:
            raise ValueError(
                f"{family}: final image conflict at slot {slot}: {conflict}"
            )
        check.add(signatures, slot)
    modified = {change["slot"] for change in changes}
    for slot, (before, after) in enumerate(zip(original, output)):
        if slot not in modified:
            for view in before:
                if exact_row_digest(before[view]) != exact_row_digest(after[view]):
                    raise ValueError("An original survivor changed")
    return (
        output,
        changes,
        {
            "rejected_candidates": rejected,
            "replaced_slots": len(changes),
            "replacement_stratum_tiers": dict(
                Counter(change["stratum_tier"] for change in changes)
            ),
        },
    )
