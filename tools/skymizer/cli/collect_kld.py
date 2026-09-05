#!/usr/bin/env python3
# =============================================================================
# collect_kld.py
#
# Sweep a HuggingFace VLM ground-truth dataset with a (reference, candidate)
# GGUF model pair and save per-row ON-THE-FLY fidelity metrics — no logits are
# stored. The C++ scorer (llama-vlm-kld) loads both models once, teacher-forces
# each row's answer tokens through both, and writes one 44-byte record per
# answer position (kld / reversed_kld / js_kld / nll_ref / nll_cand /
# entropy_ref / entropy_cand / ear / target / argmax_ref / argmax_cand).
#
#     dataset --collect_kld (ref=F16, cand=Q4_K_M)--> <out_a>/   (metrics only)
#     dataset --collect_kld (ref=F16, cand=OTHER) --> <out_b>/
#                                  \________ numpy/torch postprocessing ______
#
# This path needs only ~44 KiB/row at npos=1024 and computes every metric
# over the complete vocabulary, but fixes the (reference, candidate)
# pairing at collection time.
# Runs that must be comparable — e.g. ref-vs-A and ref-vs-B sharing the same
# reference — MUST use the same --tf-chunk / --n-ubatch (the reference side's
# logits depend on decode batching; FP non-associativity).
#
# Two phases per sweep (after `--sort-by` ordering):
#   1. Per row: prep_vlm_score_from_hf.prep_row() in-process (dataset +
#      tokenizer loaded once per sweep)
#   2. Once: one scorer JSONL manifest for all rows still needing metrics,
#      one llama-vlm-kld --manifest call (both models + mmprojs loaded a
#      single time for the whole sweep)
#   then per scored row:
#   3. Validates the VLMK dump (size vs header; npos/n_prefill vs the row;
#      embedded target column vs the row's input_ids; all metric columns
#      finite) and converts it to <idx>_<item_id>.npz (np.load-able without
#      this repo), deleting this run's .bin after a successful conversion.
#      A dump failing validation is kept on disk as evidence; the row is
#      recorded FAIL.
#   4. Appends row metadata to manifest.csv
#   5. Deletes the transient prep dir unless --keep-prep is set
#
# Output layout (under --out):
#     manifest.csv                       row_idx, item_id, num_images, n_prefill,
#                                        n_answer, n_eval, vocab, metrics_bytes,
#                                        wall_s, status. One terminal record
#                                        per row per --out (an artifact-less
#                                        FAIL_* row may be retried, appending
#                                        a second record; last-wins readers
#                                        resolve it);
#                                        metrics/*.npz is the authoritative
#                                        record of what exists.
#     collect_meta.json                  ref/cand model identity of this dir
#     metrics/<idx>_<item_id>.npz        per-token metric arrays (VLMK columns)
#     logs/kld_run.log                   scorer stderr for the run (prep runs
#                                        in-process; its failures go to the
#                                        console and manifest.csv)
#
# Output-root policy (no resume):
#   Every requested row is written exactly once. Before prep or scoring the
#   run takes an exclusive lock on --out, requires manifest.csv (if present)
#   to carry exactly the current schema, gates the scorer binary's VLMK
#   version, and scans the requested --start/--end window for ANY prior
#   output — a manifest record (except an artifact-less FAIL_* row, which is
#   retryable), or a metrics/ _prep/ entry with
#   the row's stem prefix (.bin, .npz, .tmp, prep dirs). One hit refuses the
#   whole run with a listing; nothing is ever deleted or overwritten.
#   Disjoint windows may be collected sequentially into one --out; a later
#   shard must match the dir's IDENTITY_FIELDS exactly (collect_meta.json).
#   A crashed run leaves its partial rows as collisions: clear them by hand
#   or use a fresh --out. Exits non-zero if any row failed.
#
# Usage:
#   python3 tools/skymizer/collect_kld.py \
#       --ref-model   /path/to/Qwen3VL-2B-Instruct-F16.gguf \
#       --ref-mmproj  /path/to/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
#       --cand-model  /path/to/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
#       --cand-mmproj /path/to/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
#       --out         tmp/kld-q4km/ \
#       --dataset     skymizer/ground-truth-mmmu-pro-vision-sampling-500 \
#       [--subset Qwen3-VL-2B-Instruct] [--split train] [--sort-by num_images] \
#       [--start 0] [--end N|-1] [--dataset-limit N|-1] \
#       [--num-eval-tokens N]   (cap scored answer positions; -1 = all, default) \
#       [--image-min-tokens N] [--image-max-tokens N] \
#       [--n-batch 2048] [--n-ubatch 2048] [--tf-chunk -1] [--n-ctx 32768] \
#       [--metric-threads -1] \
#       [--llama-vlm-kld build/bin/llama-vlm-kld] \
#       [--keep-prep]
#
# Postprocessing example (no repo imports needed):
#   import numpy as np

#   z = np.load("tmp/kld-q4km/metrics/000_test_X.npz")
#   z["kld"].mean(), (z["argmax_ref"] == z["argmax_cand"]).mean()
#
# Pair with: kld_metrics_io.py (loader), llama-vlm-kld (scorer)
# =============================================================================

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Light import: prep_vlm_score_from_hf's heavy deps (datasets/transformers)
# are lazy, so pulling the wrapper-mode constant here keeps the hermetic
# unit tests (numpy-only) working.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.prep_vlm_score_from_hf import MEDIA_WRAPPER_MODE

import lib.collect_core as collect_core
from lib.dataset_fingerprint import (
    dataset_content_hash,
    logged_file_fingerprint,
    maybe_file_fingerprint,
)

from lib.collect_common import (
    SKIP_OVER_BUDGET,
    acquire_out_lock,
    ensure_collect_meta as _ensure_collect_meta,
    check_manifest_header,
    collision_error,
    root_has_prior_output,
    manifest_row_writer,
    max_total_tokens_skip_info,
    preflight_n_ctx,
    preflight_scorer_vlmk_version,
    scan_output_collisions,
)
from lib.collect_vision import VisionBudgetReporter
from lib.collect_meta_provenance import build_collect_provenance
from lib.kld_metrics_io import (
    KLD_RECORD_DT,
    VLMK_VERSION,
    assert_kld_current_version,
    assert_kld_file_complete,
    convert_kld_bin_to_npz,
    load_kld_metrics,
)

# The float32 metric columns (kld/.../entropy_cand) — the ones screened for
# non-finite values in postprocess. Derived from the dtype so it cannot drift.
KLD_FLOAT_KEYS = tuple(k for k in KLD_RECORD_DT.names if KLD_RECORD_DT[k].kind == "f")


MANIFEST_COLUMNS = ("row_idx", "item_id", "num_images", "n_prefill", "n_answer", "n_eval",
                    "vocab", "metrics_bytes", "wall_s", "status")

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECT_LOG_PREFIX = f"[{Path(__file__).name}]"

# Immutable identity of a collection: a later, disjoint --start/--end shard
# may only be written into an existing --out when every one of these fields
# equals what the dir was created with (exact match, no legacy defaults).
# They are the settings that determine each row's metric values (models,
# dataset view, token caps, decode batching — FP non-associativity — and
# the media wrapper). Fingerprints are stored once in collect_meta.json but
# RECOMPUTED on every collision-free invocation and verified before any
# write (see ensure_collect_meta).
# NB no "sort_desc" either (recorded in the meta, guarded by both
# comparators with a legacy default of False): adding it here would make
# every dir collected before the flag existed un-extendable, exactly the
# `kind` situation above. The default (False) is the only order any
# existing dir was collected in.
IDENTITY_FIELDS = ["kind", "vlmk_version", "ref_model", "ref_mmproj", "cand_model", "cand_mmproj",
                   "dataset", "subset", "split",
                   "sort_by", "num_eval_tokens", "max_total_tokens",
                   "image_min_tokens", "image_max_tokens",
                   "tf_chunk", "n_ctx", "n_batch", "n_ubatch",
                   "n_gpu_layers", "n_threads", "metric_threads", "flash_attn",
                   "swa_full", "media_wrapper"]


def dump_stem(idx: int, item_id: str) -> str:
    """Filename stem for a row's metrics files: f"{idx:03d}_{item_id}",
    the key the collision scan and the comparators intersect on."""
    return f"{idx:03d}_{item_id}"


def build_kld_manifest_entry(prep_dir: Path, metrics_path: Path,
                             n_images: int, n_prefill: int,
                             image_files=None) -> dict:
    """One JSONL entry consumed by llama-vlm-kld --manifest. `image_files`
    (from prep's meta) names the files prep actually wrote; without it the
    historical img_{i}.png layout is assumed."""
    if image_files is None:
        image_files = [f"img_{i}.png" for i in range(n_images)]
    return {
        "images": [str(prep_dir / name) for name in image_files],
        "formatted_chat": str(prep_dir / "formatted_chat.txt"),
        "tokens_in": str(prep_dir / "tokens.bin"),
        "n_prefill": n_prefill,
        "output_metrics": str(metrics_path),
    }


def write_kld_manifest(path: Path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


_KLD_DONE_RE = re.compile(r"DONE output_metrics=(\S+) wall_s=([\d.]+)\s*$")


def parse_kld_done_line(line: str):
    """Match the per-entry timing marker llama-vlm-kld emits after each
    manifest entry. Returns (output_metrics_path, wall_s) or None."""
    m = _KLD_DONE_RE.search(line)
    if m is None:
        return None
    return m.group(1), float(m.group(2))


def check_n_past_expected(row: dict, header: dict, metrics_path,
                          require_match: bool = False):
    """llama.cpp-generated rows carry the generator's own position count
    after prefill (n_past_expected). The scorer's n_past_actual should equal
    it when both run the same mtmd preprocessing; a different llama.cpp build
    or budget can legitimately differ, and old datasets are routinely scored
    with newer scorers, so a mismatch WARNS and is recorded on the row.
    --require-n-past-match turns it into a rejection."""
    expected = row.get("n_past_expected")
    if expected is None:
        return None
    actual = header.get("n_past_actual")
    if actual is None or int(actual) == 0:
        return None          # pre-n_past_actual dump: nothing to compare
    if int(actual) == int(expected):
        row["n_past_mismatch"] = None
        return None
    msg = (f"{metrics_path}: scorer n_past_actual={int(actual)} != generator "
           f"n_past_expected={int(expected)} (the scorer's mtmd preprocessing "
           "differs from the one the reference text was generated under)")
    row["n_past_mismatch"] = (int(expected), int(actual))
    if require_match:
        raise ValueError(msg)
    print(f"{COLLECT_LOG_PREFIX} WARNING: {msg}", file=sys.stderr)
    return row["n_past_mismatch"]


def postprocess_kld_result(row: dict, num_eval_tokens: int, elapsed_s: float,
                           require_n_past_match: bool = False):
    """Validate one scored row's VLMK dump, cross-check its embedded targets
    against the row's input_ids, convert to .npz (deleting this run's .bin
    after a successful conversion), and return a manifest.csv row.

    A dump that fails validation is LEFT ON DISK as evidence (the error names
    it); the row is recorded FAIL and any later run over that row refuses as
    a collision — collectors never delete output."""
    metrics_path = row["metrics_path"]
    prep_dir = row["prep_dir"]

    try:
        # Writer-current gate (defense in depth behind main()'s binary
        # preflight): an old-version file here means the scorer on disk
        # changed under us mid-run.
        header = assert_kld_current_version(
            assert_kld_file_complete(metrics_path), metrics_path)
        npos = header["npos"]
        expected_npos = (min(num_eval_tokens, row["n_answer"])
                         if num_eval_tokens > 0 else row["n_answer"])
        if npos != expected_npos:
            raise ValueError(
                f"{metrics_path}: npos={npos} != expected n_eval={expected_npos} "
                f"(n_answer={row['n_answer']}, num_eval_tokens={num_eval_tokens})")
        if header["n_prefill"] != row["n_prefill"]:
            raise ValueError(
                f"{metrics_path}: header n_prefill={header['n_prefill']} != "
                f"row n_prefill={row['n_prefill']}")
        check_n_past_expected(row, header, metrics_path, require_n_past_match)

        # Integrity cross-check: the scorer embeds each position's target token in
        # its record; they must equal this row's input_ids[n_prefill:n_prefill+npos].
        # Catches manifest/output mix-ups end-to-end.
        metrics, _ = load_kld_metrics(metrics_path)
        tokens = np.fromfile(prep_dir / "tokens.bin", dtype=np.int32)
        expected_targets = tokens[row["n_prefill"]:row["n_prefill"] + npos]
        if expected_targets.size != npos or not np.array_equal(metrics["target"], expected_targets):
            raise ValueError(
                f"{metrics_path}: embedded target tokens do not match the row's "
                f"input_ids answer slice")

        # The metrics ARE the only artifact (no logits survive to re-derive
        # them), so a numerical blowup must fail the row loudly here, not get
        # recorded OK and poison downstream means.
        nonfinite = {k: int(n) for k in KLD_FLOAT_KEYS
                     if (n := (~np.isfinite(metrics[k])).sum())}
        if nonfinite:
            raise ValueError(
                f"{metrics_path}: non-finite metric values (count by column: "
                f"{nonfinite})")
    except Exception as e:
        # Keep the dump as evidence, but RENAME it out of the way: every
        # compare/check tool selects items by globbing *.bin / *.npz, so a
        # rejected-but-well-formed dump left under its real name would be
        # consumed as data (and convert_kld_bin_to_npz would promote it to an
        # indistinguishable .npz). The row's stem prefix is preserved, so the
        # collision scan still sees the row. This is a rename, never a delete.
        rejected = metrics_path.with_name(metrics_path.name + ".rejected")
        try:
            metrics_path.replace(rejected)
        except OSError:                      # keep the original error primary
            rejected = metrics_path
        # Say where it is without re-typing the exception: type(e)(msg) breaks
        # for exceptions whose constructor is not a single string (numpy's
        # _ArrayMemoryError) and would strip errno/filename off an OSError.
        # add_note is 3.11+, and this repo supports 3.10.
        if hasattr(e, "add_note"):
            e.add_note(f"rejected output kept at {rejected}")
        print(f"{COLLECT_LOG_PREFIX} rejected dump kept at {rejected}",
              file=sys.stderr)
        raise

    npz_path = metrics_path.with_suffix(".npz")
    convert_kld_bin_to_npz(metrics_path, npz_path)
    metrics_path.unlink()

    return [
        row["idx"], row["item_id"], row["n_images"], row["n_prefill"],
        row["n_answer"], npos,
        header["vocab"],
        npz_path.stat().st_size,
        f"{elapsed_s:.2f}",
        "OK",
    ]


MODEL_FINGERPRINT_FIELDS = ("ref_model_fingerprint", "cand_model_fingerprint",
                            "ref_mmproj_fingerprint", "cand_mmproj_fingerprint")

def build_collect_meta(args, dataset_content_hash: str | None = None, *,
                       ref_model_fingerprint: str | None = None,
                       cand_model_fingerprint: str | None = None,
                       ref_mmproj_fingerprint: str | None = None,
                       cand_mmproj_fingerprint: str | None = None) -> dict:
    """Identity of the (reference, candidate) metric collection in this --out
    dir. Model paths are resolved to absolute so ./x.gguf and /abs/x.gguf
    compare equal. dataset_content_hash fingerprints the loaded dataset's
    content (None for callers without a dataset in hand; field omitted).
    The four model/mmproj fingerprints default to sampling each file's
    CONTENT when the path exists -- a path string alone cannot see an
    in-place replacement (e.g. a requant written to the same filename);
    computed here, not only in main(), so direct library callers get the
    guard too."""
    if ref_model_fingerprint is None:
        ref_model_fingerprint = maybe_file_fingerprint(args.ref_model)
    if cand_model_fingerprint is None:
        cand_model_fingerprint = maybe_file_fingerprint(args.cand_model)
    if ref_mmproj_fingerprint is None:
        ref_mmproj_fingerprint = maybe_file_fingerprint(args.ref_mmproj)
    if cand_mmproj_fingerprint is None:
        cand_mmproj_fingerprint = maybe_file_fingerprint(args.cand_mmproj)
    meta = {
        "kind":        "vlm_kld_metrics",
        "vlmk_version": VLMK_VERSION,
        "ref_model":   str(Path(args.ref_model).resolve()),
        "ref_mmproj":  str(Path(args.ref_mmproj).resolve()),
        "cand_model":  str(Path(args.cand_model).resolve()),
        "cand_mmproj": str(Path(args.cand_mmproj).resolve()),
        "dataset": args.dataset,
        "subset":  args.subset,
        "split":   args.split,
        "sort_by": args.sort_by,
        "sort_desc": bool(getattr(args, "sort_desc", False)),
        "num_eval_tokens": args.num_eval_tokens,
        "max_total_tokens": getattr(args, "max_total_tokens", None),
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "tf_chunk":  args.tf_chunk,
        "n_ctx":     args.n_ctx,
        "n_batch":   args.n_batch,
        "n_ubatch":  args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "metric_threads": args.metric_threads,
        "flash_attn": args.flash_attn,
        "swa_full": bool(args.swa_full),
        "media_wrapper": MEDIA_WRAPPER_MODE,
        # Recorded but deliberately NOT in IDENTITY_FIELDS (the sort_desc
        # precedent): pre-existing dirs never saw this flag and must stay
        # extendable; the scorer log records every accepted mismatch.
        "allow_vocab_attr_mismatch": bool(getattr(args, "allow_vocab_attr_mismatch", False)),
    }
    if dataset_content_hash is not None:
        meta["dataset_content_hash"] = dataset_content_hash
    for field, value in (("ref_model_fingerprint", ref_model_fingerprint),
                         ("cand_model_fingerprint", cand_model_fingerprint),
                         ("ref_mmproj_fingerprint", ref_mmproj_fingerprint),
                         ("cand_mmproj_fingerprint", cand_mmproj_fingerprint)):
        if value is not None:
            meta[field] = value
    meta.update(build_collect_provenance())
    return meta


def ensure_collect_meta(out_dir: Path, current: dict) -> bool:
    """This collector's IDENTITY_FIELDS bound to the shared implementation
    (collect_common.ensure_collect_meta)."""
    return _ensure_collect_meta(out_dir, current, IDENTITY_FIELDS,
                                MODEL_FINGERPRINT_FIELDS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect per-row on-the-fly KLD metrics over a HF dataset")
    p.add_argument("--ref-model",   required=True, help="Reference GGUF LLM path (e.g. F16)")
    p.add_argument("--ref-mmproj",  required=True, help="Reference GGUF mmproj path")
    p.add_argument("--cand-model",  required=True, help="Candidate GGUF LLM path (e.g. Q4_K_M)")
    p.add_argument("--cand-mmproj", required=True, help="Candidate GGUF mmproj path")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dataset", required=True, help="HF dataset name")
    p.add_argument("--subset")
    p.add_argument("--split", default="train")
    p.add_argument("--sort-by", default="num_images")
    p.add_argument("--sort-desc", action="store_true",
                   help="Sort --sort-by descending instead of skymizer's default "
                        "ascending order. Default false preserves existing "
                        "index-keyed artifact dirs.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=None,
                   help="Exclusive end; default or -1 = dataset length. "
                        "Values past dataset length warn and use all available rows.")
    p.add_argument("--dataset-limit", type=int, default=None,
                   help="Score at most N rows starting at --start (i.e. "
                        "end = start + N, further capped by --end / dataset "
                        "length). -1 = no cap (same as omitting the flag).")
    p.add_argument("--num-eval-tokens", type=int, default=-1,
                   help="Cap scored answer positions per row; -1 = all answer "
                        "tokens (default). Clamped to each row's answer length.")
    p.add_argument("--max-total-tokens", type=int, default=None,
                   help="Dataset filter: skip rows whose GT n_prefill_tokens + "
                        "effective eval tokens exceeds this cap, recording "
                        "SKIP_OVER_BUDGET before scorer prep. GT n_prefill_tokens "
                        "already includes HF-expanded image-pad tokens, so vision "
                        "cost is included for cross-runtime row selection. This is "
                        "not a KV guarantee: mtmd's actual vision token count is "
                        "C++-side; --n-ctx remains scorer capacity.")
    p.add_argument("--image-min-tokens", type=int, default=-1,
                   help="Lower bound on per-image vision tokens forwarded to the scorer; "
                        "-1 = use model metadata (default).")
    p.add_argument("--image-max-tokens", type=int, default=-1,
                   help="Upper bound on per-image vision tokens forwarded to the scorer; "
                        "-1 = use model metadata (default).")
    p.add_argument("--n-batch", type=int, default=2048,
                   help="Scorer prefill logical batch (-b). Default 2048; must be >= --n-ubatch.")
    p.add_argument("--n-ubatch", type=int, default=2048,
                   help="Scorer prefill micro-batch + default teacher-forcing chunk (-ub). Default 2048.")
    p.add_argument("--tf-chunk", type=int, default=-1,
                   help="Scorer teacher-forcing chunk size (--tf-chunk), decoupled from --n-ubatch. "
                        "-1 = follow --n-ubatch (batched, default); 1 = per-token. Runs whose "
                        "metrics are compared against each other MUST use the same value.")
    p.add_argument("--n-ctx", type=int, default=32768,
                   help="Scorer context size (-c), per model.")
    p.add_argument("--n-gpu-layers", type=int, default=99,
                   help="Scorer GPU layer count (-ngl). Default 99.")
    p.add_argument("--n-threads", type=int, default=-1,
                   help="Scorer CPU generation/batch threads (-t); -1 = llama.cpp default.")
    p.add_argument("--metric-threads", type=int, default=-1,
                   help="Scorer --metric-threads (CPU threads for the per-position "
                        "metric kernel); -1 = hardware concurrency (default).")
    flash_group = p.add_mutually_exclusive_group()
    flash_group.add_argument("--flash-attn", dest="flash_attn", action="store_const",
                             const="enabled",
                             help="Require scorer flash attention (explicit enabled mode).")
    flash_group.add_argument("--no-flash-attn", dest="flash_attn", action="store_const",
                             const="disabled",
                             help="Disable scorer flash attention explicitly.")
    p.set_defaults(flash_attn="auto")
    p.add_argument("--swa-full", action="store_true",
                   help="Forwarded to the scorer: full-size KV cache for sliding-window "
                        "layers (the llama_context default). Off by default, like the "
                        "common llama.cpp CLI (SWA layers allocate only their window). "
                        "The two settings differ only by FP reordering; runs that must "
                        "be comparable use the same one (recorded in collect_meta).")
    p.add_argument("--allow-vocab-attr-mismatch", action="store_true",
                   help="Forwarded to the scorer: accept per-token ATTRIBUTE "
                        "differences (token type / EOG flags) between the two "
                        "vocabs; token texts must still match id by id. For "
                        "same-base-model conversions whose metadata disagrees "
                        "(e.g. which id is <eos>).")
    p.add_argument("--llama-vlm-kld",
                   default=str(REPO_ROOT / "build/bin/llama-vlm-kld"))
    p.add_argument("--require-n-past-match", action="store_true",
                   help="Reject llama.cpp-generated rows whose scorer n_past_actual "
                        "differs from the generator's llamacpp_n_past_prefill "
                        "(default: warn only; producer/consumer builds may differ).")
    p.add_argument("--keep-prep", action="store_true",
                   help="Don't delete per-row prep dir (debug)")
    return p.parse_args()


def resolve_dataset_end(end_arg, n_rows):
    if end_arg is None or end_arg == -1:
        return n_rows, None
    if end_arg > n_rows:
        return (
            n_rows,
            f"WARNING: --end {end_arg} exceeds dataset length {n_rows}; "
            f"falling back to {n_rows} (all available rows).",
        )
    return end_arg, None


def apply_dataset_limit(start: int, end: int, limit) -> int:
    """Cap the sweep to at most `limit` rows from --start. None or -1 (the
    repo-wide "all" sentinel) = no cap."""
    if limit is None or limit == -1:
        return end
    return min(end, start + limit)


def format_row_progress(idx: int, n_rows: int) -> str:
    return f"[{idx + 1:3d}/{n_rows}]"


def main():
    args = parse_args()
    if args.num_eval_tokens != -1 and args.num_eval_tokens < 1:
        sys.exit(f"--num-eval-tokens must be -1 (all) or >= 1, got {args.num_eval_tokens}")
    if args.max_total_tokens is not None and args.max_total_tokens < 1:
        sys.exit(f"--max-total-tokens must be >= 1 when set, got {args.max_total_tokens}")
    if args.n_batch < 1 or args.n_ubatch < 1:
        sys.exit(f"--n-batch and --n-ubatch must be >= 1, got {args.n_batch} / {args.n_ubatch}")
    if args.n_ubatch > args.n_batch:
        sys.exit(f"--n-ubatch ({args.n_ubatch}) must be <= --n-batch ({args.n_batch})")
    if args.tf_chunk != -1 and args.tf_chunk < 1:
        sys.exit(f"--tf-chunk must be -1 (follow --n-ubatch) or >= 1, got {args.tf_chunk}")
    if args.n_ctx < 1:
        sys.exit(f"--n-ctx must be >= 1, got {args.n_ctx}")
    if args.n_gpu_layers < 0:
        sys.exit(f"--n-gpu-layers must be >= 0, got {args.n_gpu_layers}")
    if args.n_threads != -1 and args.n_threads < 1:
        sys.exit(f"--n-threads must be -1 (default) or >= 1, got {args.n_threads}")
    if args.dataset_limit is not None and args.dataset_limit != -1 and args.dataset_limit < 1:
        sys.exit(f"--dataset-limit must be -1 (no cap) or >= 1, got {args.dataset_limit}")
    if args.metric_threads != -1 and args.metric_threads < 1:
        sys.exit(f"--metric-threads must be -1 (auto) or >= 1, got {args.metric_threads}")
    if args.start < 0:
        sys.exit(f"--start must be >= 0, got {args.start} (a negative index "
                 "would silently read from the dataset's end and stamp a "
                 "negative row_idx into artifact stems and manifest.csv)")
    if args.end is not None and args.end < -1:
        sys.exit(f"--end must be -1 (all) or >= 0, got {args.end}")

    # Fail fast on missing inputs: a typo otherwise surfaces only after the
    # dataset has already been downloaded and loaded.
    required_paths = {
        "--ref-model":     args.ref_model,
        "--ref-mmproj":    args.ref_mmproj,
        "--cand-model":    args.cand_model,
        "--cand-mmproj":   args.cand_mmproj,
        "--llama-vlm-kld": args.llama_vlm_kld,
    }
    missing = [f"  {flag} -> {path}" for flag, path in required_paths.items()
               if not Path(path).exists()]
    if missing:
        sys.exit("missing required path(s):\n" + "\n".join(missing))
    # Output-root policy: never skip, never delete, refuse on collision. Lock
    # the root, require the exact manifest schema, gate the scorer binary's
    # VLMK version, then (after the dataset load resolves the window) scan
    # every requested row for prior output before any write.
    manifest_path = args.out / "manifest.csv"
    try:
        out_lock = acquire_out_lock(args.out)
    except RuntimeError as e:
        sys.exit(str(e))
    with out_lock:   # released on every exit path (normal, sys.exit, exception)
        try:
            check_manifest_header(manifest_path, MANIFEST_COLUMNS)
        except ValueError as e:
            sys.exit(str(e))
        _run_locked(args, manifest_path)


def _load_dataset(args):
    # Imported here (not at module top) so the pure helpers stay importable
    # for unit tests without the heavy `datasets`/`transformers` dependencies
    # that prep_vlm_score_from_hf pulls in lazily.
    import cli.prep_vlm_score_from_hf as prep_lib
    ds = prep_lib.load_dataset_sorted(args.dataset, args.subset, args.split,
                                      args.sort_by, args.sort_desc)
    # _PrepState(args) is built after loading but only sees args; stash the
    # table so llama.cpp rows can be prepped from their raw image bytes.
    args.loaded_dataset = ds
    return ds


def _stamp_meta(args, ds_hash):
    fps = {field: logged_file_fingerprint(field, path) for field, path in (
        ("ref_model_fingerprint", args.ref_model),
        ("cand_model_fingerprint", args.cand_model),
        ("ref_mmproj_fingerprint", args.ref_mmproj),
        ("cand_mmproj_fingerprint", args.cand_mmproj))}
    ensure_collect_meta(args.out, build_collect_meta(args, ds_hash, **fps))


def _header_lines(args):
    return [
        f"  ref_model     = {Path(args.ref_model).name}",
        f"  ref_mmproj    = {Path(args.ref_mmproj).name}",
        f"  cand_model    = {Path(args.cand_model).name}",
        f"  cand_mmproj   = {Path(args.cand_mmproj).name}",
        f"  n_eval_tokens = {args.num_eval_tokens}",
        *( [f"  max_total_tok = {args.max_total_tokens}"]
           if args.max_total_tokens is not None else [] ),
        *( [f"  sort_by       = {args.sort_by}"
            f"{' (desc)' if args.sort_desc else ' (asc)'}"]
           if args.sort_by else [] ),
        f"  image_min_tok = {args.image_min_tokens}",
        f"  image_max_tok = {args.image_max_tokens}",
        f"  n_batch       = {args.n_batch}",
        f"  n_ubatch      = {args.n_ubatch}",
        f"  tf_chunk      = {args.tf_chunk}"
        f"{'  (per-token)' if args.tf_chunk == 1 else '  (batched)' if args.tf_chunk == -1 else ''}",
        f"  n_ctx         = {args.n_ctx}",
        f"  n_gpu_layers  = {args.n_gpu_layers}",
        f"  n_threads     = {args.n_threads}",
        f"  metric_threads= {args.metric_threads}",
        f"  flash_attn    = {args.flash_attn}",
        f"  swa_full      = {args.swa_full}",
        f"  out           = {args.out.as_posix()}",
    ]


class _PrepState:
    """Per-sweep prep state: the tokenizer cache (model_name -> tokenizer,
    loaded once and reused across rows) and the one-shot vision-budget
    comparison against the dataset's own limits."""
    def __init__(self, args):
        self.tok_cache = {}
        self.vision_budget = VisionBudgetReporter(args.image_min_tokens,
                                                  args.image_max_tokens)
        self.raw_images = None          # RawImageReader, built on first llama.cpp row
        self.dataset = getattr(args, "loaded_dataset", None)


def _prep(args, row, prep_dir, state):
    import cli.prep_vlm_score_from_hf as prep_lib
    raw_images = None
    if prep_lib.is_llamacpp_row(row):
        if state.raw_images is None:
            if state.dataset is None:
                raise prep_lib.PrepError(
                    "llama.cpp row needs the loaded dataset for raw image bytes")
            state.raw_images = prep_lib.RawImageReader(state.dataset)
        raw_images = state.raw_images.get(row["item_id"])
        tok = None          # prompt string is stored verbatim; no decode needed
    else:
        model_name = row["generation_model_name_or_path"]
        if model_name not in state.tok_cache:
            state.tok_cache[model_name] = prep_lib.load_tokenizer(model_name)
        tok = state.tok_cache[model_name]
    if raw_images is None:
        meta = prep_lib.prep_row(row, tok, prep_dir)
    else:
        meta = prep_lib.prep_row(row, tok, prep_dir, raw_images=raw_images)
    state.vision_budget.check(meta)
    return {"n_images": meta["num_images"],
            "n_prefill": meta["n_prefill"],
            "n_answer": meta["n_answer"],
            "image_files": meta.get("image_files"),
            "n_past_expected": meta.get("n_past_expected")}


def _scorer_argv(args, kld_manifest_path):
    return [
        args.llama_vlm_kld,
        "--ref-model",   args.ref_model,
        "--ref-mmproj",  args.ref_mmproj,
        "--cand-model",  args.cand_model,
        "--cand-mmproj", args.cand_mmproj,
        "--manifest", str(kld_manifest_path),
        "--num-eval-tokens", str(args.num_eval_tokens),
        "--image-min-tokens", str(args.image_min_tokens),
        "--image-max-tokens", str(args.image_max_tokens),
        "-b",                str(args.n_batch),
        "-c",                str(args.n_ctx),
        "-ub",               str(args.n_ubatch),
        "-ngl",              str(args.n_gpu_layers),
        "--tf-chunk",        str(args.tf_chunk),
        "-t",                str(args.n_threads),
        "--metric-threads",  str(args.metric_threads),
        *(["--flash-attn"] if args.flash_attn == "enabled" else
          ["--no-flash-attn"] if args.flash_attn == "disabled" else []),
        *(["--swa-full"] if args.swa_full else []),
        *(["--allow-vocab-attr-mismatch"]
          if getattr(args, "allow_vocab_attr_mismatch", False) else []),
    ]


def _spec():
    """This collector's divergence points, late-bound to THIS module so the
    tests' monkeypatch surface (dataset_content_hash, prep_lib,
    preflight_scorer_vlmk_version, ...) keeps intercepting. The lifecycle
    itself lives in collect_core.run_locked."""
    return collect_core.make_spec(
        pre_run=lambda args: preflight_scorer_vlmk_version(args.llama_vlm_kld),
        load_dataset=_load_dataset,
        limit_end=lambda args, end: apply_dataset_limit(args.start, end, args.dataset_limit),
        item_ids=lambda ds: ds["item_id"],
        dataset_hash=lambda ds: dataset_content_hash(ds),
        stamp_meta=_stamp_meta,
        resolve_dataset_end=resolve_dataset_end,
        manifest_columns=MANIFEST_COLUMNS,
        artifact_subdirs=("metrics",),
        header_lines=_header_lines,
        log_filename="kld_run.log",
        dump_stem=dump_stem,
        row_paths=lambda args, key: {
            "metrics_path": args.out / "metrics" / f"{key}.bin"},
        budget_extra=lambda row: {"num_images": int(row.get("num_images", 0))},
        prep_state=_PrepState,
        prep=_prep,
        scorer_display="llama-vlm-kld",
        kind_word="kld",
        write_manifest=write_kld_manifest,
        manifest_entry=lambda row: build_kld_manifest_entry(
            row["prep_dir"], row["metrics_path"], row["n_images"], row["n_prefill"],
            image_files=row.get("image_files")),
        scorer_argv=_scorer_argv,
        scorer_cmd=lambda args: args.llama_vlm_kld,
        parse_done=parse_kld_done_line,
        saved_line=lambda path, wall_s:
            f"[vlm-kld]  saved metrics {Path(path).name} in {wall_s:.1f}s",
        output_key="metrics_path",
        postprocess=lambda args, row, elapsed_s: postprocess_kld_result(
            row, args.num_eval_tokens, elapsed_s,
            require_n_past_match=getattr(args, "require_n_past_match", False)),
        postprocessed_line=lambda row, elapsed_s:
            f"{COLLECT_LOG_PREFIX} postprocessed {row['item_id']}: "
            f"n_img={row['n_images']} n_ans={row['n_answer']} "
            f"wall={elapsed_s:.1f}s"
            + (f" n_past_expected={row['n_past_mismatch'][0]} "
               f"n_past_actual={row['n_past_mismatch'][1]} (DRIFT)"
               if row.get("n_past_mismatch") else ""),
        log_prefix=COLLECT_LOG_PREFIX,
    )


def _run_locked(args, manifest_path):
    """Everything after the output-root lock is held."""
    collect_core.run_locked(args, manifest_path, _spec())


if __name__ == "__main__":
    main()
