#!/usr/bin/env python3
# =============================================================================
# saved_metrics_paired_compare.py
#
# Item-level PAIRED comparison of two quantization candidates from ON-THE-FLY
# metric dumps (collect_kld.py / VLMK .npz) instead of stored logits — the
# metrics-pipeline counterpart of paired_compare.py, producing the same report.
#
# Inputs are TWO collect_kld.py dirs that share the SAME reference:
#     --candidate-a <ref-vs-A metrics dir>   (metrics/*.npz + collect_meta.json)
#     --candidate-b <ref-vs-B metrics dir>
#
# Each per-token VLMK record already carries everything the paired report
# needs; per-item scores are reconstructed exactly as paired_compare computes
# them from raw logits:
#     nll            = mean nll_cand
#     kld            = mean kld                  (full-vocab by construction)
#     reversed_kld   = mean reversed_kld
#     js_kld         = mean js_kld
#     ear            = mean ear                  (VLMK v2 dumps only; Expected
#                      Acceptance Rate, sum min(p_ref, p_cand) = 1 - TV,
#                      arXiv:2605.02404. v1 dumps predate the column: `ear` is
#                      then dropped from the default metric set with a
#                      persisted warning, or hard-fails if explicitly
#                      requested via --metrics.)
#     same_top_rate  = mean(argmax_ref == argmax_cand)
#     mse_dp         = mean(((exp(-nll_cand) - exp(-nll_ref)) * 100)^2)
#     entropy        = mean entropy_cand         (candidate-only, no verdict)
# plus the DESCRIPTIVE per-token distribution ladders (max / 99.9% / 99.0% /
# 95.0% / 90.0% / median / 10.0% / 5.0% / 1.0% / 0.1% / min over ALL tokens of
# ALL items flattened, with the item@position each value came from and no
# bootstrap or CI) built by paired_compare from the per-token `kld` and `ear`
# columns this script threads through. The `ear` ladder needs the v2 column;
# a v1 dir gets the kld ladder alone.
# and handed to paired_compare.compare_items — the SAME statistics engine
# (item/token weighting, seeded paired bootstrap, verdict truth table,
# ppl/ppl_ratio/rms_dp derivation, pooled per-token ladders) and the SAME markdown
# renderer. On the validation runs the resulting report matches the
# stored-logits report to float32 metric-storage rounding (~1e-7 relative)
# with identical verdicts.
#
# Alignment & guards (all hard failures exit 1 with the reason on stderr):
#   - collect_meta.json of the two dirs must agree on the reference identity
#     (ref_model + ref_mmproj) and on every field that changes the numbers:
#     dataset/subset/split/sort_by/num_eval_tokens/image_{min,max}_tokens/
#     tf_chunk/n_batch/n_ubatch/media_wrapper. Hard fail on mismatch. When a
#     collect_meta.json is MISSING these cross-dir checks (and same-model
#     detection) cannot run — that downgrade is warned about AND persisted in
#     the report, but the per-item guards below still apply.
#   - Per item, the two dirs' vocab/npos/n_prefill/n_past_actual and `target`
#     columns must
#     match (hard fail — anything else is item misalignment).
#   - REFERENCE CONSISTENCY: the per-token nll_ref/entropy_ref/argmax_ref
#     columns must be bit-identical between the two dirs. The paired verdict
#     is only meaningful if A and B were measured against the same reference
#     realization; with a shared build/GPU/--tf-chunk the reference forward
#     passes are bit-exact, so any drift here means the premise is broken
#     (different build, GPU, or batching). Hard fail by default;
#     --allow-ref-drift downgrades it to a warning, and the drift magnitude
#     (worst per-item max |Δ nll_ref|, total argmax_ref flips) is persisted
#     in the report and JSON so the artifact can prove how (im)pure the
#     pairing was.
#   - Items missing from either dir hard-fail as missing data before inference.
#     `--allow-interaction` (alias `--allow-intersection`) explicitly permits
#     the clean artifact-key intersection and records the missing keys in the
#     report. `SKIP_OVER_BUDGET` row sets must still match exactly between the
#     manifests; a mismatch always hard-fails, even with that override.
#   - Any non-finite metric, rejected/incomplete artifact, or failed collection
#     row aborts before paired inference; numerical failures are never excluded
#     from the sample. Fails closed when fewer than 2 usable items remain.
#     Warnings are emitted when the candidates are the same model+mmproj, or a
#     candidate equals the reference.
#
# Usage:
#   python3 tools/skymizer/saved_metrics_paired_compare.py \
#       --candidate-a tmp/kld-q4km-mmf16/ \
#       --candidate-b tmp/kld-q4km-mmq80/ \
#       --out tmp/paired-q4km.md \
#       [--label-a Q4KM_MMF16] [--label-b Q4KM_MMQ80] \
#       [--metrics nll kld reversed_kld js_kld ear same_top_rate mse_dp] \
#       [--confidence-level 0.95] [--bootstrap-iters 5000] [--seed 1234] \
#       [--weighting both|item|token] [--start 0] [--end N|-1] \
#       [--num-eval-tokens N]   (compare-time cap: per item, first
#                                min(N, npos) positions; -1 = all, default) \
#       [--allow-interaction] \
#       [--allow-ref-drift] [--show-diagnostic-metrics] \
#       [--output-json tmp/paired-q4km.json]
#
# Speed/memory: metric files are ~44 KiB/item at npos=1024, so loading and
# per-item scoring run serially in seconds — no --jobs/--device machinery is
# needed (or provided). The one super-linear step is the pooled-token KLD
# tail bootstrap (item-cluster resampling over ALL tokens, --bootstrap-iters
# replicates): ~10 s at 1M tokens / 5000 iters on a desktop CPU.
#
# Pair with: collect_kld.py (produces the input dirs), paired_compare.py
# (the stored-logits counterpart whose engine/renderer this reuses).
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.kld_metrics_io import (                             # noqa: E402
    KLD_METRIC_KEYS, VLMK_VERSION, item_means, kld_metric_keys, load_kld_metrics,
)
from lib.collect_common import manifest_row_statuses              # noqa: E402
from compare.contracts import (                          # noqa: E402
    AlignmentError, DEFAULT_METRICS, POOLED_TOKEN_METRICS,
)
from compare.engine import compare_items                 # noqa: E402
from compare.render import (                             # noqa: E402
    build_execution_metadata, format_comparison_table,
)
from compare.cli_common import (                         # noqa: E402
    add_shared_output_args, add_shared_paired_args, metadata_argv,
    require_common_budget_skips, require_complete_item_alignment,
    require_min_items, resolve_item_end, resolve_metrics,
    validate_shared_paired_args, write_report_and_json,
)


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
                          "swa_full")
META_MUST_MATCH_BY_KIND = {
    "vlm_kld_metrics": ("ref_mmproj", "image_min_tokens", "image_max_tokens",
                        "media_wrapper"),
    "llm_kld_metrics": (),
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


def _side_scores(m, keep: int):
    """One candidate's per-item score dict from its (sliced) metric columns —
    the same aggregation paired_compare applies to raw logits. Returns
    (scores, token_metrics): the flat score dict of means, and the retained
    per-token columns as {name: (keep,) float32} for the pooled distribution
    ladders. `ear` (both as a mean and as a per-token column) is present only
    when the dump carries it (VLMK v2); the caller decides how to treat a
    mixed/absent-ear metric set."""
    if keep == 0:
        # No scored positions: NaN means -> a hard non-finite failure downstream,
        # mirroring paired_compare's handling of npos == 0 items. Same key
        # set as the keep > 0 branch (ear only when the dump has the column).
        nan_keys = [k for k in DEFAULT_METRICS if k != "ear" or "ear" in m]
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


def score_item(key: str, a_dir: Path, b_dir: Path, num_eval_tokens: int):
    """Load one matched item from both dirs, enforce the per-item guards, and
    return (scores_a, scores_b, token_a, token_b, keep, finite,
    ref_drift, vlmk_versions). token_a/token_b are {metric: (keep,) float32}
    dicts of the retained per-token columns ("kld" always, "ear" for VLMK
    v2+). vlmk_versions = (version_a, version_b) from
    the two dumps' headers — main() derives from them which metric columns
    each dir carries (kld_metrics_io.kld_metric_keys). ref_drift is None
    when the reference columns are bit-identical, else a dict
    {key, msg, max_dnll_ref, argmax_flips} that main() prints as an ERROR and
    exits on (or, with --allow-ref-drift, prints as a warning and persists in
    the report). Raises AlignmentError on shape/target misalignment or an
    unreadable dump."""
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

    ma, ha = _load(a_dir, "candidate-a")
    mb, hb = _load(b_dir, "candidate-b")

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


def _infer_cand_label(meta, override, fallback):
    if override:
        return override
    if meta and meta.get("cand_model"):
        return Path(meta["cand_model"]).name
    return fallback


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Item-level PAIRED comparison of two quantization candidates "
                    "from on-the-fly KLD metric dirs (collect_kld.py outputs) "
                    "sharing one reference.")
    p.add_argument("--candidate-a", required=True, type=Path,
                   help="collect_kld.py dir of the ref-vs-A run")
    p.add_argument("--candidate-b", required=True, type=Path,
                   help="collect_kld.py dir of the ref-vs-B run (same reference)")
    add_shared_paired_args(
        p,
        num_eval_tokens_help=(
            "Compare-time cap: per item, use only the first "
            "min(N, npos) stored positions. -1 = all (default)."),
        end_help=(
            "Exclusive item end after matching; default or -1 = all "
            "matched items. Values past the matched count warn and "
            "use all available items."))
    p.add_argument("--allow-ref-drift", action="store_true",
                   help="Downgrade the bit-identical-reference check to a warning "
                        "(use when the two runs crossed a build/GPU change; the "
                        "comparison is then only approximately paired).")
    add_shared_output_args(
        p,
        omit_host_metadata_help=(
            "drop host-volatile fields (RAM/CPU/platform) from "
            "the report and JSON so two runs of the same command "
            "are byte-identical; also the default when "
            "SKYMIZER_REPRODUCIBLE_REPORT=1"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv_for_metadata = metadata_argv(argv, "saved_metrics_paired_compare.py")
    args = parse_args(argv)
    validate_shared_paired_args(args)

    for flag, d in (("--candidate-a", args.candidate_a),
                    ("--candidate-b", args.candidate_b)):
        if not d.is_dir():
            sys.exit(f"{flag}: {d} is not a directory")
        if not (d / "metrics").is_dir():
            sys.exit(f"{flag}: {d} has no metrics/ subdir — not a "
                     "collect_kld.py output dir?")

    try:
        require_collection_success(args.candidate_a, "candidate-a")
        require_collection_success(args.candidate_b, "candidate-b")
        a_meta = load_kld_collect_meta(args.candidate_a)
        b_meta = load_kld_collect_meta(args.candidate_b)
        warnings = check_kld_meta_alignment(a_meta, b_meta)
    except (AlignmentError, ValueError) as e:
        sys.exit(str(e))
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    common_budget_skips = require_common_budget_skips({
        "candidate-a": args.candidate_a,
        "candidate-b": args.candidate_b,
    })


    metrics = resolve_metrics(args)

    matched, drops = find_metric_items(args.candidate_a, args.candidate_b)
    require_complete_item_alignment(
        drops, allow_interaction=args.allow_interaction)
    if not matched:
        def _describe(d):
            n_npz = len(list((Path(d) / "metrics").glob("*.npz")))
            n_bin = len(list((Path(d) / "metrics").glob("*.bin")))
            extra = (f" (+{n_bin} unconverted .bin — an interrupted/incomplete "
                     "collection; re-collect those rows into a fresh --out)") if n_bin else ""
            return f"{d}: {n_npz} .npz{extra}"
        sys.exit("no items present in both metrics dirs:\n"
                 f"  {_describe(args.candidate_a)}\n"
                 f"  {_describe(args.candidate_b)}")
    end, end_warning = resolve_item_end(args.end, len(matched))
    if end_warning:
        print(end_warning, file=sys.stderr)
    matched = matched[args.start:end]

    scores_a, scores_b, weights, used = [], [], [], []
    # Per-token columns for the pooled distribution ladders. Keyed lazily off
    # the first kept item so a dir of v1 dumps (no `ear` column) simply has no
    # EAR ladder instead of raising.
    token_a: dict[str, list[np.ndarray]] = {}
    token_b: dict[str, list[np.ndarray]] = {}
    drifts: list[dict] = []
    versions_seen = {"candidate-a": set(), "candidate-b": set()}
    for key in matched:
        try:
            sa, sb, tok_a, tok_b, keep, finite, drift, versions = score_item(
                key, args.candidate_a, args.candidate_b, args.num_eval_tokens)
        except AlignmentError as e:
            sys.exit(str(e))
        versions_seen["candidate-a"].add(versions[0])
        versions_seen["candidate-b"].add(versions[1])
        if drift is not None:
            drifts.append(drift)
            if not args.allow_ref_drift:
                continue   # collect every drifted item before failing below
        if not finite:
            bad = []
            for role, scores in (("candidate-a", sa), ("candidate-b", sb)):
                names = sorted(name for name, value in scores.items()
                               if not np.isfinite(value))
                if names:
                    bad.append(f"{role}={','.join(names)}")
            sys.exit(
                f"{key}: non-finite metric score(s) ({'; '.join(bad)}); "
                "paired inference aborted")
        scores_a.append(sa)
        scores_b.append(sb)
        cols = sorted(set(tok_a) & set(tok_b))
        if not token_a:
            token_a = {c: [] for c in cols}
            token_b = {c: [] for c in cols}
        elif sorted(token_a) != cols:
            sys.exit(f"{key}: per-token columns {cols} differ from the earlier "
                     f"items' {sorted(token_a)}; the two dirs mix VLMK versions "
                     "(re-collect so every dump carries the same columns)")
        for c in cols:
            token_a[c].append(tok_a[c])
            token_b[c].append(tok_b[c])
        weights.append(keep)
        used.append(key)

    if drifts:
        for d in drifts:
            print(f"{'WARNING' if args.allow_ref_drift else 'ERROR'}: {d['msg']}",
                  file=sys.stderr)
        if not args.allow_ref_drift:
            sys.exit(
                f"{len(drifts)} item(s) show reference drift between the two "
                "dirs: the runs do not share a bit-identical reference (different "
                "build / GPU / --tf-chunk / -ub?). The paired verdict premise is "
                "broken. Re-collect one side, or pass --allow-ref-drift to "
                "compare anyway (approximately paired).")
    used_set = set(used)
    # Persisted drift summary: the artifact must be able to distinguish 1-ulp
    # batching noise from a completely different reference model — stderr is
    # gone by the time anyone reads an archived report.
    drift_summary = None
    if drifts:
        drift_summary = {
            "n_items": len(drifts),
            "n_used": sum(1 for d in drifts if d["key"] in used_set),
            "max_abs_dnll_ref": max(d["max_dnll_ref"] for d in drifts),
            "argmax_ref_flips": sum(d["argmax_flips"] for d in drifts),
        }

    require_min_items(scores_a, matched)

    # Which metric columns each dir carries follows from its dumps' VLMK
    # versions (kld_metrics_io is the single authority on per-version
    # layouts). A requested base metric missing from any used dump is dropped
    # from the DEFAULT set with a persisted warning that names the dir(s) and
    # version(s) responsible (the rest of the report is unaffected), but is a
    # hard failure when the operator explicitly asked for it — silently
    # reporting less than requested would look like a complete answer.
    version_desc = {role: "v" + "/v".join(str(v) for v in sorted(vs))
                    for role, vs in versions_seen.items()}
    if versions_seen["candidate-a"] != versions_seen["candidate-b"] or \
            any(len(vs) > 1 for vs in versions_seen.values()):
        warnings.append(
            "mixed VLMK versions: candidate-a dumps are "
            f"{version_desc['candidate-a']}, candidate-b dumps are "
            f"{version_desc['candidate-b']}; only the columns every dump "
            "carries are compared")
        print(f"WARNING: {warnings[-1]}", file=sys.stderr)
    missing_by_metric: dict[str, list[str]] = {}
    for role, vs in versions_seen.items():
        for v in sorted(vs):
            for m in metrics:
                if m in KLD_METRIC_KEYS and m not in kld_metric_keys(v):
                    missing_by_metric.setdefault(m, []).append(f"{role} (VLMK v{v})")
    for m, where in missing_by_metric.items():
        if args.metrics and m in args.metrics:
            sys.exit(
                f"--metrics {m} requested, but the {m!r} column is absent from "
                f"{', '.join(where)}; re-collect that dir with the current "
                "llama-{vlm,llm}-kld (writes VLMK v" + str(VLMK_VERSION) + ").")
        metrics = tuple(x for x in metrics if x != m)
        warnings.append(
            f"the {m!r} metric was dropped from this report: its column is "
            f"absent from {', '.join(where)} — re-collect that dir with the "
            "current scorer to get it")
        print(f"WARNING: {warnings[-1]}", file=sys.stderr)

    a_label = _infer_cand_label(a_meta, args.label_a, "candidate-a")
    b_label = _infer_cand_label(b_meta, args.label_b, "candidate-b")
    # When both metas exist the cross-dir guard verified they agree; when one
    # is missing this is just "whichever survives" and ## Inputs reflects only
    # that side's claims (the persisted missing-meta warning says so).
    ref_meta = a_meta or b_meta
    metrics_source = _meta_kind(ref_meta) if ref_meta else "vlm_kld_metrics"
    ref_label = (Path(ref_meta["ref_model"]).name
                 if ref_meta and ref_meta.get("ref_model") else "reference")

    result = compare_items(
        scores_a, scores_b, weights, metrics=metrics,
        confidence_level=args.confidence_level, bootstrap_iters=args.bootstrap_iters,
        seed=args.seed, model_a_label=a_label, model_b_label=b_label,
        ci_method=args.ci_method,
        primary_metric=args.primary_metric,
        primary_weighting=args.primary_weighting,
        equivalence_margin=args.equivalence_margin,
        position_buckets=args.position_buckets,
        token_metrics_a=token_a, token_metrics_b=token_b, item_keys=used)
    result["reference_label"] = ref_label
    result["metrics_source"] = metrics_source
    result["vlmk_versions"] = {role.replace("-", "_"): sorted(vs)
                               for role, vs in versions_seen.items()}
    result["inputs"] = {
        "reference":   {"label": ref_label,
                        "model_path": ref_meta.get("ref_model") if ref_meta else None},
        "candidate_a": {"label": a_label,
                        "model_path": a_meta.get("cand_model") if a_meta else None},
        "candidate_b": {"label": b_label,
                        "model_path": b_meta.get("cand_model") if b_meta else None},
        "kind":    metrics_source,
        "dataset": ref_meta.get("dataset") if ref_meta else None,
        "subset":  ref_meta.get("subset") if ref_meta else None,
        "split":   ref_meta.get("split") if ref_meta else None,
        "sort_by": ref_meta.get("sort_by") if ref_meta else None,
    }
    result["execution"] = build_execution_metadata(
        args, argv_for_metadata, device="cpu", jobs=1,
        omit_host_metadata=args.omit_host_metadata)
    # The metrics were computed by the C++ dual-model scorer during collection;
    # only the lightweight paired statistics below run on CPU. Keep the two
    # stages separate so an archived report does not mislabel GPU-collected
    # metrics as a CPU evaluation. gpu_name is hardware provenance recorded by
    # the collectors, not a claim reconstructed from the comparison host.
    result["execution"]["metrics_collection"] = {
        "mode": "on-the-fly",
        "gpu_by_candidate": {
            "candidate-a": (a_meta or {}).get("gpu_name") or "unknown",
            "candidate-b": (b_meta or {}).get("gpu_name") or "unknown",
        },
    }
    result["alignment"] = {
        "n_matched": len(used),
        "drops": drops,
        "warnings": warnings,
        "allow_interaction": bool(args.allow_interaction),
        "n_common_skipped_over_budget": len(common_budget_skips),
        "common_skipped_over_budget": common_budget_skips,
        "n_ref_drift": len(drifts),
        "ref_drift_allowed": bool(args.allow_ref_drift),
    }
    if drift_summary is not None:
        result["alignment"]["ref_drift"] = drift_summary

    table = format_comparison_table(
        result, reference_label=ref_label, display_weighting=args.weighting,
        show_diagnostic_metrics=args.show_diagnostic_metrics,
        drops=drops, n_matched=len(used),
        num_eval_tokens=args.num_eval_tokens)
    collector = "collect_llm_kld.py" if metrics_source == "llm_kld_metrics" else "collect_kld.py"
    table += (f"note: computed from on-the-fly VLMK metric dumps ({collector}); "
              "no logits were stored. KLD-family metrics are full-vocab by "
              "construction.\n")
    # Persist the meta warnings: an archived report must carry its own caveats
    # (a degenerate same-model comparison reads exactly like a genuine
    # "inconclusive" win otherwise).
    for w in warnings:
        table += f"warning: {w}\n"
    if drift_summary is not None and args.allow_ref_drift:
        table += (f"note: --allow-ref-drift accepted {drift_summary['n_items']} "
                  f"item(s) whose reference columns differ between the dirs "
                  f"({drift_summary['n_used']} used in the comparison; worst "
                  f"max |Δ nll_ref| = {drift_summary['max_abs_dnll_ref']:.3e}, "
                  f"{drift_summary['argmax_ref_flips']} argmax_ref flip(s)) — "
                  "the comparison is only approximately paired.\n")
    write_report_and_json(args, table, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
