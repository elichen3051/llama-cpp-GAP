#!/usr/bin/env python3
"""Collect validated VLM fidelity metrics against a fixed reference."""

import argparse
import json
import sys
from pathlib import Path


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
    add_collector_runtime_args,
    add_collector_selection_args,
    locked_manifest,
    require_paths,
    validate_collector_args,
    KLD_FLOAT_KEYS,
    apply_dataset_limit,
    dump_stem,
    finalized_kld_row,
    format_row_progress,
    parse_kld_done_line,
    resolve_dataset_end,
    write_kld_manifest,
    ensure_collect_meta as _ensure_collect_meta,
    preflight_n_ctx,
    preflight_scorer_vlmk_version,
)
from lib.collect_vision import VisionBudgetReporter
from lib.collect_meta_provenance import build_collect_provenance
from lib.kld_metrics_io import VLMK_VERSION


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
                   "swa_full", "media_wrapper", "execution_identity"]


def build_kld_manifest_entry(prep_dir: Path, metrics_path: Path,
                             n_images: int, n_prefill: int,
                             image_files=None, add_special: bool = False, reference_vocabulary=None) -> dict:
    """One JSONL entry consumed by llama-vlm-kld --manifest. `image_files`
    (from prep's meta) names the files prep actually wrote; without it the
    historical img_{i}.png layout is assumed. `add_special` (llama-server
    rows) makes the scorer tokenize the prefix with mtmd add_special=true."""
    if image_files is None:
        image_files = [f"img_{i}.png" for i in range(n_images)]
    entry = {
        "images": [str(prep_dir / name) for name in image_files],
        "formatted_chat": str(prep_dir / "formatted_chat.txt"),
        "tokens_in": str(prep_dir / "tokens.bin"),
        "n_prefill": n_prefill,
        "output_metrics": str(metrics_path),
    }
    if reference_vocabulary:
        entry["reference_vocabulary"] = reference_vocabulary
    if add_special:
        entry["add_special"] = True
    return entry


def check_n_past_expected(row: dict, header: dict, metrics_path,
                          require_match: bool = False):
    """llama.cpp-generated rows carry the generator's own position count
    after prefill (n_past_expected). The scorer's n_past_actual should equal
    it when both run the same mtmd preprocessing; a mismatch means the scored
    prefix is not the one the reference text was generated under, so by
    default (require_match=True, i.e. without --allow-prefix-drift) the row
    is rejected. With --allow-prefix-drift it WARNS and is recorded on the row."""
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
    with finalized_kld_row(row, num_eval_tokens, COLLECT_LOG_PREFIX) as header:
        check_n_past_expected(row, header, row["metrics_path"], require_n_past_match)
    return [row["idx"], row["item_id"], row["n_images"], row["n_prefill"],
            row["n_answer"], header["npos"], header["vocab"],
            row["metrics_path"].with_suffix(".npz").stat().st_size, f"{elapsed_s:.2f}", "OK"]


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
        # provenance for llama.cpp-generated datasets: whether prefix drift
        # between generator and scorer was allowed (warn) or rejected (default)
        "allow_prefix_drift": bool(getattr(args, "allow_prefix_drift", False)),
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
    meta.update(build_collect_provenance(getattr(args, "llama_vlm_kld", None)))
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
    add_collector_selection_args(p)
    p.add_argument("--max-total-tokens", type=int, default=None,
                   help="Dataset filter: skip rows whose GT n_prefill_tokens + "
                        "effective eval tokens exceeds this cap, recording "
                        "SKIP_OVER_BUDGET before scorer prep. GT n_prefill_tokens "
                        "already includes HF-expanded image-pad tokens, so vision "
                        "cost is included for cross-runtime row selection. This is "
                        "not a KV guarantee: mtmd's actual vision token count is "
                        "C++-side; --n-ctx remains scorer capacity.")
    p.add_argument("--image-min-tokens", type=int, default=None,
                   help="Lower bound on per-image vision tokens forwarded to the scorer. "
                        "-1 = the mmproj's own metadata. Omitted (with --image-max-tokens "
                        "also omitted) = adopt the budget a llama.cpp-generated dataset "
                        "recorded, else -1.")
    p.add_argument("--image-max-tokens", type=int, default=None,
                   help="Upper bound on per-image vision tokens forwarded to the scorer. "
                        "-1 = the mmproj's own metadata. Omitted (with --image-min-tokens "
                        "also omitted) = adopt the budget a llama.cpp-generated dataset "
                        "recorded, else -1.")
    add_collector_runtime_args(p)
    p.add_argument("--allow-vocab-attr-mismatch", action="store_true",
                   help="Forwarded to the scorer: accept per-token ATTRIBUTE "
                        "differences (token type / EOG flags) between the two "
                        "vocabs; token texts must still match id by id. For "
                        "same-base-model conversions whose metadata disagrees "
                        "(e.g. which id is <eos>).")
    p.add_argument("--llama-vlm-kld",
                   default=str(REPO_ROOT / "build/bin/llama-vlm-kld"))
    p.add_argument("--allow-prefix-drift", action="store_true",
                   help="Score llama.cpp-generated rows even when this scorer's mtmd image "
                        "spans / n_past differ from the generator's (warn instead of reject). "
                        "Default rejects: the paired metrics would otherwise be conditioned on a "
                        "different prefix than the reference text was generated under.")
    p.add_argument("--keep-prep", action="store_true",
                   help="Don't delete per-row prep dir (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    validate_collector_args(args)

    # Fail fast on missing inputs: a typo otherwise surfaces only after the
    # dataset has already been downloaded and loaded.
    required_paths = {
        "--ref-model":     args.ref_model,
        "--ref-mmproj":    args.ref_mmproj,
        "--cand-model":    args.cand_model,
        "--cand-mmproj":   args.cand_mmproj,
        "--llama-vlm-kld": args.llama_vlm_kld,
    }
    require_paths(required_paths)
    with locked_manifest(args.out, MANIFEST_COLUMNS) as manifest_path:
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
    resolve_image_token_budget(args, ds)
    return ds


def resolve_image_token_budget(args, ds) -> None:
    """Turn the CLI's --image-min/max-tokens into the ints the scorer takes.

    ``None`` (flag omitted) is distinct from an explicit ``-1`` (use the
    mmproj's metadata): when BOTH flags are omitted and the dataset was
    generated by llama.cpp, adopt the budget the generator recorded in
    ``image_processor_config``; otherwise every remaining ``None`` becomes -1.
    Explicit values, -1 included, are never overwritten; the
    VisionBudgetReporter still warns when they differ from the dataset's."""
    if args.image_min_tokens is None and args.image_max_tokens is None:
        adopted = _dataset_image_budget(ds)
        if adopted is not None:
            args.image_min_tokens, args.image_max_tokens = adopted
            print(f"{COLLECT_LOG_PREFIX} llama.cpp-generated dataset: adopting its recorded "
                  f"image token budget min={adopted[0]} max={adopted[1]} "
                  "(pass --image-min-tokens/--image-max-tokens explicitly to override; "
                  "-1 -1 = mmproj metadata)", file=sys.stderr)
    if args.image_min_tokens is None:
        args.image_min_tokens = -1
    if args.image_max_tokens is None:
        args.image_max_tokens = -1


def _dataset_image_budget(ds):
    """(min, max) recorded by a llama.cpp-generated dataset, else None."""
    if len(ds) == 0:
        return None
    import cli.prep_vlm_score_from_hf as prep_lib
    row = ds[0]
    if not prep_lib.is_llamacpp_row(row):
        return None
    try:
        cfg = json.loads(row["image_processor_config"])
        limits = prep_lib.derive_image_token_limits(cfg)
        return int(limits["image_min_tokens"]), int(limits["image_max_tokens"])
    except (KeyError, TypeError, ValueError, prep_lib.PrepError):
        return None


def _stamp_meta(args, ds_hash):
    fps = {field: logged_file_fingerprint(field, path) for field, path in (
        ("ref_model_fingerprint", args.ref_model),
        ("cand_model_fingerprint", args.cand_model),
        ("ref_mmproj_fingerprint", args.ref_mmproj),
        ("cand_mmproj_fingerprint", args.cand_mmproj))}
    meta = build_collect_meta(args, ds_hash, **fps)
    ensure_collect_meta(args.out, meta)
    args._recorded_execution_identity = meta["execution_identity"]


def _verify_execution(args):
    current = build_collect_provenance(args.llama_vlm_kld)["execution_identity"]
    if current != args._recorded_execution_identity:
        raise ValueError("scorer execution identity changed during collection")


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
    return {"reference_vocabulary": meta.get("reference_vocabulary"),
            "n_images": meta["num_images"],
            "n_prefill": meta["n_prefill"],
            "n_answer": meta["n_answer"],
            "image_files": meta.get("image_files"),
            "n_past_expected": meta.get("n_past_expected"),
            "add_special": bool(meta.get("add_special", False))}


def _scorer_argv(args, kld_manifest_path):
    return [
        args.llama_vlm_kld,
        "--ref-model",   args.ref_model,
        "--ref-mmproj",  args.ref_mmproj,
        "--cand-model",  args.cand_model,
        "--cand-mmproj", args.cand_mmproj,
        "--manifest", str(kld_manifest_path),
        "--num-eval-tokens", str(args.num_eval_tokens),
        *(["--image-min-tokens", str(args.image_min_tokens)] if args.image_min_tokens != -1 else []),
        *(["--image-max-tokens", str(args.image_max_tokens)] if args.image_max_tokens != -1 else []),
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
        *(["--allow-prefix-drift"]
          if getattr(args, "allow_prefix_drift", False) else []),
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
        verify_execution=_verify_execution,
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
            image_files=row.get("image_files"), add_special=row.get("add_special", False),
            reference_vocabulary=row.get("reference_vocabulary")),
        scorer_argv=_scorer_argv,
        scorer_cmd=lambda args: args.llama_vlm_kld,
        parse_done=parse_kld_done_line,
        saved_line=lambda path, wall_s:
            f"[vlm-kld]  saved metrics {Path(path).name} in {wall_s:.1f}s",
        output_key="metrics_path",
        postprocess=lambda args, row, elapsed_s: postprocess_kld_result(
            row, args.num_eval_tokens, elapsed_s,
            require_n_past_match=not getattr(args, "allow_prefix_drift", False)),
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
