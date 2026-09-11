"""Load and validate paired metric collections before inference."""

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from lib.kld_metrics_io import VERSIONED_METRIC_KEYS, item_means, load_kld_metrics
from lib.collection_state import comparison_locks, require_completed_attempts
from lib.collect_meta_provenance import require_execution_alignment
from lib.collect_common import SKIP_OVER_BUDGET, manifest_row_statuses
from stats.contracts import AlignmentError, DEFAULT_METRICS, POOLED_TOKEN_METRICS


def stem_sort_key(stem: str):
    """Sort key restoring numeric dataset order for row-stem collections.

    Stems are f"{row_idx:03d}_{item_id}" with THREE-digit zero padding, so a
    plain string sort misorders four-digit rows ("1000_x" < "998_y") and
    --start/--end windows stop meaning numeric dataset ranges. Numeric-prefixed
    stems sort first by integer row index; stems without a decimal prefix sort
    after them in plain string order -- a deterministic total order either way."""
    prefix = stem.split("_", 1)[0]
    if prefix.isdecimal():
        return (0, int(prefix), stem)
    return (1, 0, stem)

# collect_meta.json fields that must agree across the two metrics dirs: the
# shared-reference identity plus everything that perturbs the logits both
# runs' metrics were computed from (decode batching, vision token bounds,
# prompt conditioning) or the item set (dataset/sort/cap). swa_full has NO
# legacy default on purpose: dirs written before the key existed were
# collected with either setting (full SWA until 564edf630, window-sized
# after it), so a legacy dir (None) only pairs with another legacy dir --
# which NUMERICAL_CONTRACT already requires to be the same build anyway.
META_MUST_MATCH_COMMON = ("kind", "ref_model", "dataset", "subset", "split",
                          "sort_by", "sort_desc", "num_eval_tokens",
                          "max_total_tokens", "tf_chunk", "n_batch", "n_ubatch",
                          "swa_full", "n_ctx", "n_gpu_layers", "n_threads",
                          "metric_threads", "flash_attn")
META_MUST_MATCH_BY_KIND = {
    "vlm_kld_metrics": ("ref_mmproj", "image_min_tokens", "image_max_tokens",
                        "media_wrapper"),
    "llm_kld_metrics": ("perplexity_window", "corpus_protocol", "corpus_windows_sha256"),
}
META_MUST_MATCH = META_MUST_MATCH_COMMON + META_MUST_MATCH_BY_KIND["vlm_kld_metrics"]

# The reference-side record columns that must be bit-identical between the two
# dirs when the shared-reference premise holds (plus `target`, which is checked
# separately because its mismatch means item misalignment, never mere drift).
REF_CONSISTENCY_KEYS = ("nll_ref", "entropy_ref", "argmax_ref")


def load_kld_collect_meta(d: Path) -> dict | None:
    p = Path(d) / "collect_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Path-tag the error: a bare JSONDecodeError says nothing about WHICH
        # of the two dirs is corrupt.
        raise ValueError(f"{p}: unreadable collect_meta.json: {e}") from e


def _meta_kind(meta: dict) -> str:
    return meta.get("kind", "vlm_kld_metrics")


def _meta_guard_value(meta: dict, field: str):
    """Comparison value for one META_MUST_MATCH field, normalizing legacy
    dirs: kind defaults to the VLM collector's, and sort_desc (recorded by
    the LLM collectors since F2.8) defaults to False — every dir written
    before the flag existed (all VLM dirs) was collected ascending. Every
    other field, swa_full included, compares as recorded (missing = None)."""
    if field == "kind":
        return _meta_kind(meta)
    if field == "sort_desc":
        return bool(meta.get("sort_desc", False))
    return meta.get(field)


def check_kld_meta_alignment(a_meta, b_meta) -> list[str]:
    """Hard-fail when the two metrics dirs disagree on any META_MUST_MATCH
    field; return human-readable warnings (missing meta, wrong kind,
    missing content fingerprints, same-model candidates, candidate ==
    reference) for the caller to print. dataset_content_hash and the
    ref-side fingerprints are guarded three-tier (present-and-different
    hard-fails; missing only warns): the shared-reference premise rides on
    them -- the stored nll_ref/entropy_ref/argmax_ref columns are not a
    unique fingerprint of the reference distribution, and name-equal
    ref_model/dataset fields cannot see content replaced in place."""
    warnings: list[str] = []
    metas = {"candidate-a": a_meta, "candidate-b": b_meta}
    for role, m in metas.items():
        if m is None:
            warnings.append(
                f"{role}: collect_meta.json missing — identity unverified, and "
                "ALL cross-dir config checks (reference identity, dataset, "
                "tf_chunk/batching, ...) and same-model detection are SKIPPED; "
                "only the per-item target/npos/reference-consistency guards "
                "still apply")
        else:
            kind = _meta_kind(m)
            if kind not in META_MUST_MATCH_BY_KIND:
                warnings.append(f"{role}: collect_meta kind is {kind!r}, "
                                "expected 'vlm_kld_metrics' or 'llm_kld_metrics'")
    if a_meta is not None and b_meta is not None:
        kind_a = _meta_kind(a_meta)
        kind_b = _meta_kind(b_meta)
        if kind_a != kind_b:
            raise AlignmentError(
                f"collect_meta mismatch on 'kind': "
                f"candidate-a={kind_a!r} candidate-b={kind_b!r}. "
                "The two dirs must share the same metric artifact kind.")
        fields = META_MUST_MATCH_COMMON + META_MUST_MATCH_BY_KIND.get(kind_a, ())
        for field in fields:
            a_val = _meta_guard_value(a_meta, field)
            b_val = _meta_guard_value(b_meta, field)
            if a_val != b_val:
                raise AlignmentError(
                    f"collect_meta mismatch on {field!r}: "
                    f"candidate-a={a_val!r} candidate-b={b_val!r}. "
                    "The two dirs must share the same reference run configuration.")
        # Build/GPU provenance: warn, never fail. The "preprocessing cancels
        # in the paired comparison" premise rests on both sides running the
        # SAME clip.cpp resize and decode kernels; nothing else on disk
        # witnesses that, and it was recorded and then ignored.
        for field, what in (("llama_cpp_build_commit", "llama.cpp build"),
                            ("gpu_name", "GPU")):
            vals = {role: m.get(field) for role, m in metas.items()
                    if m is not None and m.get(field) not in (None, "unknown")}
            if len(set(vals.values())) > 1:
                warnings.append(
                    f"{what} differs across dirs: {vals}. Image preprocessing "
                    "and decode kernels are only guaranteed to cancel in the "
                    "paired difference when both sides ran the same code.")

        def cand_id(m):
            if kind_a == "vlm_kld_metrics":
                return (m.get("cand_model"), m.get("cand_mmproj"))
            return (m.get("cand_model"),)
        def ref_id(m):
            if kind_a == "vlm_kld_metrics":
                return (m.get("ref_model"), m.get("ref_mmproj"))
            return (m.get("ref_model"),)
        same_detail = ("identical cand_model and cand_mmproj"
                       if kind_a == "vlm_kld_metrics" else "identical cand_model")
        ref_detail = ("identical model and mmproj; only the quantization should differ"
                      if kind_a == "vlm_kld_metrics" else
                      "identical model; only the quantization should differ")
        if cand_id(a_meta) == cand_id(b_meta):
            warnings.append("candidate-a and candidate-b are the SAME model "
                            f"({same_detail})")
        for role, m in metas.items():
            if cand_id(m) == ref_id(m):
                warnings.append(f"{role}: candidate equals the reference "
                                f"({ref_detail})")
        # Three-tier content fingerprints (NOT META_MUST_MATCH: a plain
        # comparison would hard-fail every legacy dir whose meta predates the
        # fields). ref_mmproj_fingerprint only exists on the VLM lane.
        fingerprint_fields = ("dataset_content_hash", "ref_model_fingerprint")
        if kind_a == "vlm_kld_metrics":
            fingerprint_fields += ("ref_mmproj_fingerprint",)
        for field in fingerprint_fields:
            vals = {}
            for role, m in metas.items():
                v = m.get(field)
                if v is None:
                    warnings.append(
                        f"{role}: collect_meta.json has no {field}; "
                        "content identity unverified")
                else:
                    vals[role] = v
            if len(set(vals.values())) > 1:
                raise AlignmentError(
                    f"collect_meta mismatch on {field!r}: "
                    f"candidate-a={vals.get('candidate-a')!r} "
                    f"candidate-b={vals.get('candidate-b')!r}. "
                    "The two dirs' metrics were computed against different "
                    "content under the same name; their paired difference "
                    "would mix that drift into the quant effect.")
    return warnings


def find_metric_items(a_dir: Path, b_dir: Path):
    """Return (matched_keys_numeric_order, drops). drops[role] = keys the OTHER
    dir has but `role` is missing, so the report points at the dir that
    dropped. Ordering is numeric by row-index prefix (stem_sort_key) --
    string sort would misorder stems past row 999 and break --start/--end."""
    def stems(d):
        return {p.stem for p in (Path(d) / "metrics").glob("*.npz")}
    a, b = stems(a_dir), stems(b_dir)
    return sorted(a & b, key=stem_sort_key), {
        "candidate-a": sorted(b - a, key=stem_sort_key),
        "candidate-b": sorted(a - b, key=stem_sort_key),
    }

def require_collection_success(root: Path, role: str) -> None:
    """Reject incomplete metric collections before selecting paired items.

    New binaries fail before writing a non-finite VLMK file; older collectors
    quarantine a postprocess failure as `*.rejected`. Unconverted `*.bin` and
    atomic-writer `*.tmp` leftovers also mean a row never completed. Checking
    all of them prevents numerical/incomplete rows from disappearing as an
    ordinary missing-key drop.
    """
    root = Path(root)
    manifest = root / "manifest.csv"
    failed = []
    if manifest.exists():
        failed = sorted(
            (idx, status)
            for idx, status in manifest_row_statuses(manifest).items()
            if status.startswith("FAIL_")
        )
    rejected = sorted((root / "metrics").glob("*.rejected"))
    unconverted = sorted((root / "metrics").glob("*.bin"))
    temporary = sorted((root / "metrics").glob("*.tmp"))
    if not failed and not rejected and not unconverted and not temporary:
        require_completed_attempts(root)
        return
    details = []
    if failed:
        sample = ", ".join(f"row {idx}={status}" for idx, status in failed[:5])
        details.append(f"{len(failed)} failed manifest row(s): {sample}")
    if rejected:
        sample = ", ".join(p.name for p in rejected[:5])
        details.append(f"{len(rejected)} rejected metric artifact(s): {sample}")
    if unconverted:
        sample = ", ".join(p.name for p in unconverted[:5])
        details.append(f"{len(unconverted)} unconverted metric artifact(s): {sample}")
    if temporary:
        sample = ", ".join(p.name for p in temporary[:5])
        details.append(f"{len(temporary)} incomplete temporary artifact(s): {sample}")
    raise ValueError(f"{role}: {'; '.join(details)}; paired inference aborted")



def validate_collection_pair(a_dir, b_dir):
    """Check completed collection identities while the caller holds both reader locks."""
    for role, root in (("candidate-a", a_dir), ("candidate-b", b_dir)):
        require_collection_success(root, role)
    a_meta = load_kld_collect_meta(a_dir)
    b_meta = load_kld_collect_meta(b_dir)
    require_execution_alignment(a_meta, b_meta)
    warnings = check_kld_meta_alignment(a_meta, b_meta)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    budget_skips = require_common_budget_skips({"candidate-a": a_dir, "candidate-b": b_dir})
    return a_meta, b_meta, warnings, budget_skips


def aligned_metric_items(a_dir, b_dir, *, allow_interaction=False):
    matched, drops = find_metric_items(a_dir, b_dir)
    require_complete_item_alignment(drops, allow_interaction=allow_interaction)
    return matched, drops


def _side_scores(m, keep: int):
    """Aggregate one candidate's scores and retain aligned per-token metrics."""
    if keep == 0:
        # No scored positions: NaN means -> a hard non-finite failure downstream,
        # mirroring paired_compare's handling of npos == 0 items. Same key
        # set as the keep > 0 branch (ear only when the dump has the column).
        nan_keys = [k for k in DEFAULT_METRICS if k not in VERSIONED_METRIC_KEYS or k in m]
        nan_keys += ["entropy", "mean_dp", "nll_ref"]
        empty = {k: np.empty(0, np.float32)
                 for k in POOLED_TOKEN_METRICS
                 if k in ("kld", "dp") or k in m}
        empty["target"] = np.empty(0, np.int32)
        return ({k: float("nan") for k in nan_keys}, empty)
    scores, dp = item_means(m, keep)
    token_metrics = {"kld": np.ascontiguousarray(m["kld"][:keep], dtype=np.float32)}
    if "ear" in m:
        token_metrics["ear"] = np.ascontiguousarray(m["ear"][:keep], dtype=np.float32)
    token_metrics["dp"] = np.ascontiguousarray(dp, dtype=np.float32)
    # annotation column (not a metric): the target token id per position,
    # so the per-item tail witnesses can name the token
    token_metrics["target"] = np.ascontiguousarray(m["target"][:keep], dtype=np.int32)
    return scores, token_metrics


def load_item_pair(key: str, a_dir: Path, b_dir: Path):
    """Read each candidate once and attach paths to read errors."""
    def _load(d, role):
        # np.load's own errors (BadZipFile, EOFError, KeyError, ...) carry no
        # path; tag them with the key + dir so the operator knows which of
        # potentially hundreds of files to re-collect.
        path = Path(d) / "metrics" / f"{key}.npz"
        try:
            return load_kld_metrics(path)
        except Exception as e:
            raise AlignmentError(
                f"{key}: {role}: failed to load {path}: "
                f"{type(e).__name__}: {e}") from e

    return (*_load(a_dir, "candidate-a"), *_load(b_dir, "candidate-b"))



def score_item(key: str, a_dir: Path, b_dir: Path, num_eval_tokens: int):
    return score_records(key, *load_item_pair(key, a_dir, b_dir), num_eval_tokens)


def score_records(key, ma, ha, mb, hb, num_eval_tokens):
    """Validate the pair and derive scores from already loaded records."""
    for field in ("vocab", "npos", "n_prefill"):
        if ha[field] != hb[field]:
            raise AlignmentError(
                f"{key}: {field} mismatch (candidate-a={ha[field]}, "
                f"candidate-b={hb[field]})")
    # n_prefill above is the HF ground-truth sequential length echoed from the
    # manifest, so it cannot see a vision-token change: the same prompt at a
    # different --image-max-tokens gives the same echo. n_past_actual is
    # llama.cpp's own post-prefill position count, which does move, so it is
    # the content-level backstop for the collect_meta image-bounds guard.
    # Both zero = dumps written before the field existed; checked only when
    # both sides recorded one.
    pa, pb = ha.get("n_past_actual", 0), hb.get("n_past_actual", 0)
    if pa > 0 and pb > 0 and pa != pb:
        raise AlignmentError(
            f"{key}: post-prefill position count mismatch (candidate-a={pa}, "
            f"candidate-b={pb}). The same prompt became a different number of "
            "embeddings in the two runs — different vision-token budgets "
            "(--image-min-tokens / --image-max-tokens, or differing mmproj "
            "vision metadata) — so every scored position is conditioned on a "
            "different prefix.")
    if not np.array_equal(ma["target"], mb["target"]):
        raise AlignmentError(
            f"{key}: target tokens differ between the two dirs — these are "
            "not the same items (different dataset rows or prep)")

    ref_drift = None
    drifted = [k for k in REF_CONSISTENCY_KEYS
               if not np.array_equal(ma[k], mb[k])]
    if drifted:
        d_nll = float(np.abs(ma["nll_ref"].astype(np.float64)
                             - mb["nll_ref"].astype(np.float64)).max())
        n_argmax = int((ma["argmax_ref"] != mb["argmax_ref"]).sum())
        ref_drift = {
            "key": key,
            "max_dnll_ref": d_nll,
            "argmax_flips": n_argmax,
            "msg": (f"{key}: reference columns differ between the two dirs "
                    f"({', '.join(drifted)}; max |Δ nll_ref|={d_nll:.3e}, "
                    f"{n_argmax} argmax_ref flip(s))"),
        }

    npos = ha["npos"]
    keep = npos if num_eval_tokens == -1 else min(num_eval_tokens, npos)
    sa, tok_a = _side_scores(ma, keep)
    sb, tok_b = _side_scores(mb, keep)
    finite = all(np.isfinite(v) for v in (*sa.values(), *sb.values()))
    return sa, sb, tok_a, tok_b, keep, finite, ref_drift, (ha["version"], hb["version"])


def require_common_budget_skips(roots: Mapping[str, Path]) -> list[int]:
    """Require every available manifest to declare the same budget-skip set.

    A common SKIP_OVER_BUDGET set is an intentional corpus filter and is safe
    for pairing. A different set changes the sample per side and always aborts;
    --allow-interaction cannot override it. If one manifest is absent while
    another declares skips, the equality cannot be verified, so fail closed.
    """
    skip_sets: dict[str, set[int] | None] = {}
    for role, root in roots.items():
        manifest = Path(root) / "manifest.csv"
        if not manifest.exists():
            skip_sets[role] = None
            continue
        try:
            statuses = manifest_row_statuses(manifest)
        except ValueError as e:
            sys.exit(str(e))
        skip_sets[role] = {
            idx for idx, status in statuses.items()
            if status == SKIP_OVER_BUDGET
        }

    declared = {role: rows for role, rows in skip_sets.items() if rows is not None}
    missing_manifests = [role for role, rows in skip_sets.items() if rows is None]
    any_declared_skip = any(rows for rows in declared.values())
    if any_declared_skip and missing_manifests:
        print(
            "ERROR: SKIP_OVER_BUDGET alignment cannot be verified because "
            f"manifest.csv is missing for {', '.join(missing_manifests)}",
            file=sys.stderr)
        sys.exit("paired inference aborted: common budget-skip set is unverified")

    distinct = {tuple(sorted(rows)) for rows in declared.values()}
    if len(distinct) > 1:
        print("ERROR: SKIP_OVER_BUDGET sets differ across paired inputs:",
              file=sys.stderr)
        for role, rows in skip_sets.items():
            shown = "manifest missing" if rows is None else str(sorted(rows))
            print(f"  {role}: {shown}", file=sys.stderr)
        sys.exit("paired inference aborted: SKIP_OVER_BUDGET set mismatch")

    common = list(next(iter(distinct))) if distinct else []
    if common:
        sample = ", ".join(str(idx) for idx in common[:10])
        more = "" if len(common) <= 10 else f", … (+{len(common) - 10})"
        print(
            f"INFO: {len(common)} row(s) excluded identically from all paired "
            f"inputs as {SKIP_OVER_BUDGET}: {sample}{more}",
            file=sys.stderr)
    return common


def require_complete_item_alignment(
    drops: Mapping[str, Sequence[str]], *, allow_interaction: bool
) -> None:
    """Abort on any one-sided artifact-key gap unless explicitly overridden."""
    missing = [(role, list(keys)) for role, keys in drops.items() if keys]
    if not missing:
        return
    level = "WARNING" if allow_interaction else "ERROR"
    print(f"{level}: paired input data has one-sided missing item(s):",
          file=sys.stderr)
    for role, keys in missing:
        sample = ", ".join(keys[:10])
        more = "" if len(keys) <= 10 else f", … (+{len(keys) - 10})"
        print(f"  {role} missing {len(keys)}: {sample}{more}", file=sys.stderr)
    if allow_interaction:
        print("WARNING: --allow-interaction enabled; paired inference will use "
              "only the clean item intersection", file=sys.stderr)
        return
    sys.exit(
        "paired inference aborted: input item sets differ; re-collect the "
        "missing data or pass --allow-interaction to explicitly use the intersection")


def require_min_items(scores_a, matched) -> None:
    """Fail closed: a single item makes the bootstrap CI zero-width, which
    would report spurious "significant" verdicts. Paired inference needs
    >= 2 items."""
    if len(scores_a) < 2:
        sys.exit(
            f"paired inference needs >= 2 usable items; only {len(scores_a)} usable "
            f"(matched {len(matched)} after --start/--end). "
            "Widen the item range or check the inputs.")
