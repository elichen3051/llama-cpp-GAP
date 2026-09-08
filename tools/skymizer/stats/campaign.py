"""Declared comparison families and training-only benchmark selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from statsmodels.regression.linear_model import OLS

from stats import collection_io
from stats.multiplicity import adjust_pvalues


def content_hash(value):
    """Hash a JSON value independently of dictionary key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _names(values, label, minimum=1):
    if not isinstance(values, list) or len(values) < minimum or any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain at least {minimum} distinct nonempty strings")
    return values


def validate_plan(plan):
    """Validate the declared universe before loading any candidate outcomes."""
    if plan.get("schema_version") != "skymizer-campaign-plan-v1":
        raise ValueError("schema_version must be skymizer-campaign-plan-v1")
    _names([plan.get("family_id")], "family_id")
    if plan.get("metric") != "kld" or plan.get("weighting") != "item":
        raise ValueError("campaign inference requires metric=kld and weighting=item")
    if plan.get("sampling_plan") != "fixed_n" or plan.get("prespecification_status") != "user_declared":
        raise ValueError("declare fixed_n and user_declared prespecification; optional stopping is unsupported")
    alpha = plan.get("alpha", .05)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    method = plan.get("correction", "holm")
    if method not in ("holm", "fdr_bh", "fdr_by"):
        raise ValueError("correction must be holm, fdr_bh or fdr_by")
    if method == "fdr_bh" and plan.get("dependence_assumption") != "independent_or_prds":
        raise ValueError("BH requires a declared independent_or_prds assumption; shared pairs alone do not establish it")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a nonempty list")
    _names([c.get("cell_id") for c in cells], "cell IDs")
    for c in cells:
        for field in ("dataset_id", "model_family", "model_id", "dataset_content_hash"):
            _names([c.get(field)], field)
        if c.get("mode") not in ("instruct", "thinking"):
            raise ValueError("mode must be instruct or thinking")
        roster = c.get("roster", [])
        _names([r.get("item_key") for r in roster], "item keys", 2)
        item_ids = _names([r.get("item_id") for r in roster], "stable item IDs", 2)
        analysis_ids = _names(c.get("analysis_item_ids", item_ids), "analysis item IDs", 2)
        if not set(analysis_ids) <= set(item_ids):
            raise ValueError("analysis item IDs must be in the declared full collection roster")
        if not isinstance(c.get("fixed_analysis_n"), int) or isinstance(c.get("fixed_analysis_n"), bool) or c.get("fixed_analysis_n") != len(analysis_ids):
            raise ValueError("fixed_analysis_n must equal the declared analysis item count")
        image_clusters = {}
        for row in roster:
            prefix, separator, collected_id = row["item_key"].partition("_")
            if not separator or not prefix.isdecimal() or collected_id != row["item_id"]:
                raise ValueError("stable item_id must equal the collected ID encoded in item_key")
            _names([row.get("cluster_id")], "cluster_id")
            _names([row.get("source_id")], "source_id")
            hashes = row.get("image_hashes", [])
            if hashes:
                _names(hashes, "image hashes")
            for image_hash in hashes:
                if image_hash in image_clusters and image_clusters[image_hash] != row["cluster_id"]:
                    raise ValueError("items sharing an image must share a cluster, including multi-image links")
                image_clusters[image_hash] = row["cluster_id"]
        variants = c.get("variants", [])
        _names([v.get("quant_id") for v in variants], "quant IDs", 2)
        _names([v.get("collection_dir") for v in variants], "collection dirs", 2)
    return sum(math.comb(len(c["variants"]), 2) for c in cells)


def load_campaign(plan, base_dir=Path(".")):
    """Reuse completed-collection, target, reference and execution guards under reader locks."""
    validate_plan(plan)
    roots = [(Path(base_dir) / v["collection_dir"]).resolve() for c in plan["cells"] for v in c["variants"]]
    if len(set(roots)) != len(roots):
        raise ValueError("collection paths must be distinct across the declared campaign")
    loaded = {}
    with collection_io.comparison_locks(roots):
        for cell in plan["cells"]:
            variants = cell["variants"]
            dirs = [(Path(base_dir) / v["collection_dir"]).resolve() for v in variants]
            keys = [r["item_key"] for r in cell["roster"]]
            matrix = np.empty((len(keys), len(dirs)), dtype=float)
            provenance = []
            for j, root in enumerate(dirs):
                validation_anchor = dirs[1] if j == 0 else dirs[0]
                a_meta, b_meta, warnings, skips = collection_io.validate_collection_pair(validation_anchor, root)
                fatal_warnings = [w for w in warnings if ": candidate equals the reference (" not in w]
                if fatal_warnings or skips or b_meta.get("dataset_content_hash") != cell["dataset_content_hash"]:
                    raise ValueError(f"{cell['cell_id']}: incomplete or mismatched collection provenance")
                if not a_meta.get("ref_model_fingerprint") or not b_meta.get("ref_model_fingerprint"):
                    raise ValueError("campaign collections require reference fingerprints")
                candidate_fingerprint = b_meta.get("cand_model_fingerprint")
                if not candidate_fingerprint or (b_meta.get("kind") == "vlm_kld_metrics" and not b_meta.get("cand_mmproj_fingerprint")):
                    raise ValueError("campaign collections require candidate model/projector fingerprints")
                if b_meta.get("perplexity_window"):
                    raise ValueError("campaign corpus windows require a separate declared dependence protocol")
                matched, _ = collection_io.aligned_metric_items(dirs[0], root)
                if set(matched) != set(keys):
                    raise ValueError(f"{cell['cell_id']}: collection keys must equal the declared roster; no intersection or slicing")
                provenance.append({"quant_id": variants[j]["quant_id"], "collection_dir": str(root), "collect_meta_sha256": content_hash(b_meta), "reference_fingerprint_sha256": content_hash(b_meta["ref_model_fingerprint"]), "candidate_fingerprint_sha256": content_hash([candidate_fingerprint, b_meta.get("cand_mmproj_fingerprint")])})
                for i, key in enumerate(keys):
                    sa, sb, _, _, _, finite, drift, _ = collection_io.score_item(key, dirs[0], root, -1)
                    if not finite or drift is not None:
                        raise ValueError(f"{cell['cell_id']}/{key}: nonfinite metrics or reference drift")
                    if j and matrix[i, 0] != sa["kld"]:
                        raise ValueError("anchor metric changed while loading the campaign")
                    matrix[i, j] = sb["kld"]
            loaded[cell["cell_id"]] = {"scores": matrix, "provenance": provenance, "scores_sha256": content_hash(matrix.tolist())}
    return loaded


def clustered_mean_test(differences, cluster_ids, alpha, family_size):
    """Equal-item mean with CRV1 image-cluster covariance and G-1 t reference.

    This is an asymptotic cluster-robust approximation, not an exact small-G test. Intercept-only OLS preserves item weights for unequal cluster sizes. Both covariance and inference degree corrections are explicit.
    APIs: https://www.statsmodels.org/v0.14.6/generated/statsmodels.regression.linear_model.OLS.fit.html
    https://www.statsmodels.org/v0.14.6/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
    Method: Cameron and Miller (2015), https://doi.org/10.3368/jhr.50.2.317
    Bonferroni simultaneous intervals use the union bound; Holm (1979): https://www.jstor.org/stable/4615733
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(family_size, int) or isinstance(family_size, bool) or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    if np.iscomplexobj(differences):
        raise ValueError("paired differences must be real")
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or len(cluster_ids) != len(d) or not np.all(np.isfinite(d)):
        raise ValueError("finite differences and aligned cluster IDs are required")
    _, groups = np.unique(cluster_ids, return_inverse=True)
    g = len(np.unique(groups))
    if g < 2:
        raise ValueError("at least two independent image clusters are required")
    scale = float(np.max(np.abs(d)))
    if scale == 0 or np.ptp(d / scale) <= 10 * np.finfo(float).eps:
        raise ValueError("unresolved difference variance; campaign inference unavailable")
    if not 0 < alpha / family_size < 1 or 1 - alpha / (2 * family_size) == 1:
        raise ValueError("simultaneous confidence level exceeds float resolution")
    result = OLS(d / scale, np.ones((len(d), 1))).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True, "df_correction": True}, use_t=True)
    estimate, se, p = float(result.params[0] * scale), float(result.bse[0] * scale), float(result.pvalues[0])
    pointwise = np.asarray(result.conf_int(alpha=alpha)[0]) * scale
    simultaneous = np.asarray(result.conf_int(alpha=alpha / family_size)[0]) * scale
    if not all(math.isfinite(v) for v in (estimate, se, p, *pointwise, *simultaneous)) or se < max(math.sqrt(np.finfo(float).tiny), 10 * np.finfo(float).eps * scale):
        raise ValueError("unresolved cluster variance or nonfinite cluster inference")
    if (p < alpha) != (pointwise[0] > 0 or pointwise[1] < 0):
        raise ValueError("cluster p-value and closed interval disagree at float resolution")
    if result.df_resid_inference != g - 1:
        raise ValueError("unexpected cluster inference degrees of freedom")
    return {"estimate": estimate, "standard_error": se, "p_value": p, "n_items": len(d), "n_clusters": g,
            "degrees_of_freedom": g - 1, "ci_pointwise": pointwise.tolist(), "ci_bonferroni": simultaneous.tolist(),
            "method": "equal_item_mean_crv1_cluster_t", "approximation": "independent clusters; asymptotic in cluster count"}


def analyze_campaign(plan, loaded):
    """Compare every declared pair and adjust over the entire declared campaign."""
    family_size = validate_plan(plan)
    validate_selection_receipt(plan, loaded)
    alpha, method = plan.get("alpha", .05), plan.get("correction", "holm")
    if set(loaded) != {c["cell_id"] for c in plan["cells"]}:
        raise ValueError("loaded cells must exactly match the plan")
    pairs, summaries = [], []
    for cell in plan["cells"]:
        cid = cell["cell_id"]
        if np.iscomplexobj(loaded[cid]["scores"]):
            raise ValueError("campaign scores must be real")
        scores = np.asarray(loaded[cid]["scores"], dtype=float)
        labels = [v["quant_id"] for v in cell["variants"]]
        if scores.shape != (len(cell["roster"]), len(labels)) or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError(f"{cid}: expected finite nonnegative KLD matrix matching roster and variants")
        analysis_ids = cell.get("analysis_item_ids", [r["item_id"] for r in cell["roster"]])
        index = {r["item_id"]: i for i, r in enumerate(cell["roster"])}
        selected = [index[key] for key in analysis_ids]
        scores = scores[selected]
        clusters = [cell["roster"][i]["cluster_id"] for i in selected]
        local = []
        for a, b in itertools.combinations(range(len(labels)), 2):
            test = clustered_mean_test(scores[:, b] - scores[:, a], clusters, alpha, family_size)
            row = {"cell_id": cid, "quant_a": labels[a], "quant_b": labels[b], **test}
            pairs.append(row)
            local.append(row)
        dominated = set()
        for row in local:
            lo, hi = row["ci_bonferroni"]
            if lo > 0:
                dominated.add(row["quant_b"])
            elif hi < 0:
                dominated.add(row["quant_a"])
        best_set = [v for v in labels if v not in dominated]
        column_scale = np.max(scores, axis=0)
        means = np.mean(scores / np.where(column_scale > 0, column_scale, 1), axis=0) * column_scale
        if not np.all(np.isfinite(means)):
            raise ValueError("nonfinite campaign means")
        summaries.append({"cell_id": cid, "dataset_id": cell["dataset_id"], "mode": cell["mode"], "model_family": cell["model_family"],
                          "mean_kld": dict(zip(labels, means.tolist())),
                          "descriptive_minimum": labels[int(np.argmin(means))],
                          "possible_best_by_simultaneous_intervals": best_set,
                          "unique_best_by_simultaneous_intervals": best_set[0] if len(best_set) == 1 else None,
                          "input_provenance": loaded[cid].get("provenance", [])})
    adjusted = adjust_pvalues([p["p_value"] for p in pairs], method)
    for row, p in zip(pairs, adjusted):
        row["p_adjusted"] = p
        row["reject_equal_mean"] = p < alpha
    for summary in summaries:
        local = [p for p in pairs if p["cell_id"] == summary["cell_id"]]
        summary["all_pairs_rejected_under_selected_correction"] = all(p["reject_equal_mean"] for p in local)
        summary["observed_least_resolved_pair"] = {key: max(local, key=lambda p: p["p_adjusted"])[key] for key in ("quant_a", "quant_b", "p_adjusted")}
    return {"schema_version": "skymizer-campaign-result-v1", "plan_sha256": content_hash(plan), "plan": plan,
            "family_size": family_size, "alpha": alpha, "correction": method,
            "error_control": "FWER" if method == "holm" else "FDR",
            "prespecification_status": "user_declared_not_independently_verified",
            "cluster_map_status": "user_declared_image_or_shared_source_connected_components",
            "study_labels_status": "dataset, family, model and mode labels are user declared; stored collection compatibility and receipt provenance are checked",
            "interval_policy": "pointwise CI and Bonferroni simultaneous CI across all planned pairs; cluster approximation applies",
            "limitations": ["valid marginal p-values and independent clusters are required", "small cluster counts can be poorly calibrated", "fixed-n findings are not power or a minimum sample-size guarantee", "inconclusive does not establish equivalence", "selected-pool and unknown-family generalization need independent validation"],
            "cells": summaries, "pairs": pairs}


def _require_selection_provenance(plan, loaded):
    """Require complete observed collection identities for selection/validation receipts."""
    for cell in plan["cells"]:
        records = loaded[cell["cell_id"]].get("provenance", [])
        if [p.get("quant_id") for p in records] != [v["quant_id"] for v in cell["variants"]]:
            raise ValueError("selection requires complete observed collection provenance for every quant")
        for record in records:
            for key in ("collection_dir", "collect_meta_sha256", "reference_fingerprint_sha256"):
                _names([record.get(key)], "selection provenance " + key)


def validate_selection_receipt(plan, loaded):
    """Check declared split identities, not historical access to held-out outcomes.

    Selection bias requires independent evaluation: Cawley and Talbot (2010), https://www.jmlr.org/papers/v11/cawley10a.html
    """
    receipt = plan.get("selection_receipt")
    if receipt is None:
        return
    if receipt.get("schema_version") != "skymizer-selection-v1":
        raise ValueError("unknown selection receipt schema")
    _require_selection_provenance(plan, loaded)
    training_records = receipt.get("training_provenance", [])
    if not training_records:
        raise ValueError("selection receipt requires observed training provenance")
    for record in training_records:
        for key in ("collection_dir", "collect_meta_sha256", "reference_fingerprint_sha256"):
            _names([record.get(key)], "training provenance " + key)
    checked = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if content_hash(checked) != receipt.get("receipt_sha256"):
        raise ValueError("selection receipt hash mismatch")
    train = set(_names(receipt.get("training_families"), "training families"))
    heldout = set(_names(receipt.get("heldout_families"), "heldout families"))
    selected_ids = _names(receipt.get("selected_item_ids"), "selected item IDs", 3)
    if receipt.get("selected_item_ids_sha256") != content_hash(selected_ids) or receipt.get("rule", {}).get("size") != len(selected_ids):
        raise ValueError("selection IDs, count or hash disagree")
    if receipt.get("rule", {}).get("algorithm") != "greedy_backward_maximin_snr_v1":
        raise ValueError("unknown selection algorithm")
    if {c["model_family"] for c in plan["cells"]} != heldout:
        raise ValueError("validation must include every declared heldout family")
    if train & heldout:
        raise ValueError("training and heldout model families overlap")
    training_paths = {p["collection_dir"] for p in receipt.get("training_provenance", [])}
    validation_paths = {p["collection_dir"] for cell in loaded.values() for p in cell.get("provenance", [])}
    if training_paths & validation_paths:
        raise ValueError("training and heldout collections reuse the same resolved path")
    for field in ("collect_meta_sha256", "reference_fingerprint_sha256"):
        training_ids = {p[field] for p in receipt.get("training_provenance", []) if field in p}
        validation_ids = {p[field] for cell in loaded.values() for p in cell.get("provenance", []) if field in p}
        if training_ids & validation_ids:
            raise ValueError("training and heldout collections share model provenance")
    for cell in plan["cells"]:
        if cell["mode"] != receipt["mode"]:
            raise ValueError("validation mode differs from the selection mode")
        if cell["model_family"] not in heldout or cell["model_family"] in train:
            raise ValueError("selected-pool validation requires a declared heldout model family")
        ids = cell.get("analysis_item_ids", [r["item_id"] for r in cell["roster"]])
        if ids != receipt["selected_item_ids"]:
            raise ValueError("validation must use the frozen selected IDs in their declared order")
        identities = sorted([{k: row.get(k, []) for k in ("item_id", "source_id", "image_hashes")} for row in cell["roster"]], key=lambda r: r["item_id"])
        full_pool = len(identities) == receipt.get("pool_size") and content_hash(identities) == receipt["pool_identity_sha256"]
        selected_pool = len(identities) == len(selected_ids) and content_hash(identities) == receipt.get("selected_identity_sha256")
        if not (full_pool or selected_pool):
            raise ValueError("validation identities differ from both the frozen pool and selected benchmark")


def validate_selection_plan(plan, rule):
    """Reject heldout inputs and invalid selection rules before reading score files."""
    validate_plan(plan)
    if plan.get("selection_receipt") is not None:
        raise ValueError("selection input must be an untouched training pool")
    if rule.get("algorithm") != "greedy_backward_maximin_snr_v1":
        raise ValueError("algorithm must be greedy_backward_maximin_snr_v1")
    train = _names(rule.get("training_families"), "training families")
    heldout = _names(rule.get("heldout_families"), "heldout families")
    if set(train) & set(heldout) or {c["model_family"] for c in plan["cells"]} != set(train):
        raise ValueError("selector must receive exactly the training families, disjoint from heldout families")
    if len({c["mode"] for c in plan["cells"]}) != 1:
        raise ValueError("select instruct and thinking benchmarks separately")
    target = rule.get("size")
    if not isinstance(target, int) or isinstance(target, bool) or not 3 <= target < len(plan["cells"][0]["roster"]):
        raise ValueError("size must be an integer from 3 to pool size minus one")
    return train, heldout


def select_benchmark(plan, loaded, rule):
    """Greedy backward search maximizes the minimum subset SNR over training tasks.

    SNR is abs(mean(delta))/sample_SD(delta), never mean(abs(delta)). The objective is descriptive and optimistic after selection. This project-specific search has no claim of global optimality; a package p-value adjustment cannot validate the selected benchmark.
    NumPy reduction API: https://numpy.org/doc/stable/reference/generated/numpy.std.html
    Selection/evaluation separation: Cawley and Talbot (2010), https://www.jmlr.org/papers/v11/cawley10a.html
    """
    train, heldout = validate_selection_plan(plan, rule)
    if set(loaded) != {c["cell_id"] for c in plan["cells"]}:
        raise ValueError("loaded cells must exactly match the training plan")
    _require_selection_provenance(plan, loaded)
    if len({c["mode"] for c in plan["cells"]}) != 1:
        raise ValueError("select instruct and thinking benchmarks separately")
    first = plan["cells"][0]["roster"]
    identities = sorted([{k: r.get(k, []) for k in ("item_id", "source_id", "image_hashes")} for r in first], key=lambda r: r["item_id"])
    ids = [r["item_id"] for r in identities]
    target = rule.get("size")
    if not isinstance(target, int) or isinstance(target, bool) or not 3 <= target < len(ids):
        raise ValueError("size must be an integer from 3 to pool size minus one")
    tasks = []
    for cell in plan["cells"]:
        roster = cell["roster"]
        other = sorted([{k: r.get(k, []) for k in ("item_id", "source_id", "image_hashes")} for r in roster], key=lambda r: r["item_id"])
        if other != identities or cell.get("analysis_item_ids") is not None:
            raise ValueError("training cells must use the same complete source/image pool identities")
        if len({r["cluster_id"] for r in roster}) != len(roster):
            raise ValueError("this selector requires one item per independent image/source cluster")
        if np.iscomplexobj(loaded[cell["cell_id"]]["scores"]):
            raise ValueError("training scores must be real")
        scores = np.asarray(loaded[cell["cell_id"]]["scores"], dtype=float)
        if scores.shape != (len(roster), len(cell["variants"])) or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("training scores must be finite nonnegative KLD and match the roster")
        order = {r["item_id"]: i for i, r in enumerate(roster)}
        scores = scores[[order[key] for key in ids]]
        tasks.extend(scores[:, b] - scores[:, a] for a, b in itertools.combinations(range(scores.shape[1]), 2))
    values = np.column_stack(tasks)
    scale = np.max(np.abs(values), axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("training task has no resolved effect variation")
    values = values / scale
    sources = [r["source_id"] for r in identities]
    quotas = rule.get("source_min_counts", {})
    if not isinstance(quotas, dict) or any(k not in sources or not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > sources.count(k) for k, v in quotas.items()) or sum(quotas.values()) > target:
        raise ValueError("source_min_counts are infeasible")

    def objective(selected):
        subset = values[selected]
        magnitude = np.max(np.abs(subset), axis=0)
        if np.any(magnitude == 0):
            return -math.inf
        subset = subset / magnitude
        sd = subset.std(axis=0, ddof=1)
        if np.any(sd <= 10 * np.finfo(float).eps):
            return -math.inf
        return float(np.min(np.abs(subset.mean(axis=0)) / sd))

    selected = list(range(len(ids)))
    evaluations = 0
    while len(selected) > target:
        counts = {source: sum(sources[i] == source for i in selected) for source in set(sources)}
        best, remove = -math.inf, None
        for i in selected:
            if counts[sources[i]] <= quotas.get(sources[i], 0):
                continue
            score = objective([j for j in selected if j != i])
            evaluations += 1
            if score > best:
                best, remove = score, i
        if remove is None:
            raise ValueError("no feasible subset with resolved task variances")
        selected.remove(remove)
    selected_ids = [ids[i] for i in selected]
    receipt = {"schema_version": "skymizer-selection-v1", "rule": rule, "training_plan_sha256": content_hash(plan),
               "pool_identity_sha256": content_hash(identities), "pool_size": len(ids), "mode": plan["cells"][0]["mode"],
               "training_scores_sha256": {key: content_hash(np.asarray(value["scores"], dtype=float).tolist()) for key, value in loaded.items()},
               "training_provenance": [p for c in loaded.values() for p in c.get("provenance", [])],
               "training_families": train, "heldout_families": heldout, "selected_item_ids": selected_ids,
               "selected_identity_sha256": content_hash([r for r in identities if r["item_id"] in set(selected_ids)]),
               "selected_item_ids_sha256": content_hash(selected_ids), "objective": "minimum abs(mean(delta))/SD(delta) across all training pairs",
               "training_objective": objective(selected), "candidate_subsets_evaluated": evaluations,
               "status": "training_only_greedy_result_not_global_optimum",
               "validation_scope": "heldout-family transfer conditional on this source/image pool; fresh clusters are needed for broader item generalization"}
    receipt["receipt_sha256"] = content_hash(receipt)
    return receipt
