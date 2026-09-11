"""Exact native generation cohorts shared by tail publication and analysis."""

import hashlib

from lib.reference_dataset import canonical_json


def canonical_digest(value):
    """Use the native generator sidecar's canonical JSON SHA256 convention."""
    encoded = canonical_json(value)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_cohort(cohort, size):
    """Verify ordered ID lists, counts, hashes and an exact disjoint partition."""
    for name in ("requested", "eligible", "excluded", "failed"):
        ids = cohort.get(name + "_ids")
        if (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids) or type(cohort.get(name)) is not int
                or cohort[name] != len(ids)
                or cohort.get(name + "_ids_sha256") != canonical_digest(ids)):
            raise ValueError(f"invalid native cohort {name} IDs/count/hash")
    requested = cohort["requested_ids"]
    partition = sum((cohort[k + "_ids"] for k in ("eligible", "excluded", "failed")), [])
    if (len(requested) != size or len(partition) != size or set(partition) != set(requested)):
        raise ValueError("native cohort does not partition the requested source IDs")
    for name in ("eligible", "excluded", "failed"):
        ids = cohort[name + "_ids"]
        members = set(ids)
        if ids != [x for x in requested if x in members]:
            raise ValueError("native cohort IDs must preserve requested order")
    if cohort["failed"]:
        raise ValueError("native cohort has failed generation rows; resolve them before composition")
    if (type(cohort.get("generated")) is not int or type(cohort.get("native_generated")) is not int
            or cohort["generated"] != cohort["eligible"] + cohort["excluded"]
            or not cohort["generated"] <= cohort["native_generated"] <= size):
        raise ValueError("native generation counts do not reconcile")
    return cohort


def tail_cohort(pilot, parent):
    """Select positions101..500 by requested IDs, preserving native exclusions."""
    validate_cohort(pilot, 100)
    validate_cohort(parent, 500)
    if pilot["requested_ids"] != parent["requested_ids"][:100]:
        raise ValueError("pilot IDs are not the exact requested first100 of parent500")
    requested = parent["requested_ids"][100:]
    requested_set = set(requested)
    tail = {}
    for name in ("requested", "eligible", "excluded", "failed"):
        ids = requested if name == "requested" else [x for x in parent[name + "_ids"] if x in requested_set]
        tail.update({name: len(ids), name + "_ids": ids,
                     name + "_ids_sha256": canonical_digest(ids)})
    tail.update(generated=400, native_generated=400)
    return tail
