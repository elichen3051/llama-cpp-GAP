#!/usr/bin/env python3
"""Collect validated LLM fidelity metrics against a fixed reference."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from lib.collect_meta_provenance import build_collect_provenance
import lib.collect_core as collect_core
from lib.dataset_fingerprint import (
    dataset_content_hash,
    logged_file_fingerprint,
    maybe_file_fingerprint,
)
from lib.kld_metrics_io import VLMK_VERSION


MANIFEST_COLUMNS = ("row_idx", "item_id", "n_prefill", "n_answer", "n_eval",
                    "vocab", "metrics_bytes", "wall_s", "status")

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECT_LOG_PREFIX = f"[{Path(__file__).name}]"

# Immutable identity of a collection: a later, disjoint --start/--end shard
# may only be written into an existing --out when every one of these fields
# equals what the dir was created with (exact match, no legacy defaults).
# They are the settings that determine each row's metric values (models,
# dataset view, token caps, decode batching — FP non-associativity).
# Fingerprints are stored once in collect_meta.json but RECOMPUTED on
# every collision-free invocation and verified before any write.
IDENTITY_FIELDS = ["kind", "vlmk_version", "ref_model", "cand_model", "dataset", "subset",
                   "split", "sort_by", "sort_desc", "num_eval_tokens",
                   "max_total_tokens", "tf_chunk", "n_ctx", "n_batch", "n_ubatch",
                   "n_gpu_layers", "n_threads", "metric_threads", "flash_attn",
                   "swa_full", "execution_identity", "perplexity_window",
                   "corpus_protocol", "corpus_windows_sha256"]


def build_kld_manifest_entry(prep_dir: Path, metrics_path: Path,
                             n_prefill: int, reference_vocabulary=None) -> dict:
    """One JSONL entry consumed by llama-llm-kld --manifest."""
    return {
        **({"reference_vocabulary": reference_vocabulary} if reference_vocabulary else {}),
        "tokens_in": str(prep_dir / "tokens.bin"),
        "n_prefill": n_prefill,
        "output_metrics": str(metrics_path),
    }


def postprocess_kld_result(row: dict, num_eval_tokens: int, elapsed_s: float):
    with finalized_kld_row(row, num_eval_tokens, COLLECT_LOG_PREFIX) as header:
        if row.get("corpus_window") and header["n_past_actual"] != 2 * (row["n_prefill"] - 1):
            raise ValueError("corpus scorer did not decode the full PPL window")
    return [row["idx"], row["item_id"], row["n_prefill"], row["n_answer"],
            header["npos"], header["vocab"],
            row["metrics_path"].with_suffix(".npz").stat().st_size, f"{elapsed_s:.2f}", "OK"]


MODEL_FINGERPRINT_FIELDS = ("ref_model_fingerprint", "cand_model_fingerprint")

def build_collect_meta(args, dataset_content_hash: str | None = None, *,
                       ref_model_fingerprint: str | None = None,
                       cand_model_fingerprint: str | None = None) -> dict:
    """Identity of the (reference, candidate) metric collection in this --out
    dir. Model paths are resolved to absolute so ./x.gguf and /abs/x.gguf
    compare equal. dataset_content_hash fingerprints the loaded dataset's
    content (None for callers without a dataset in hand; field omitted).
    The ref/cand model fingerprints default to sampling each file's CONTENT
    when the path exists -- a path string alone cannot see an in-place
    replacement (e.g. a requant written to the same filename); computed
    here, not only in main(), so direct library callers get the guard too."""
    if ref_model_fingerprint is None:
        ref_model_fingerprint = maybe_file_fingerprint(args.ref_model)
    if cand_model_fingerprint is None:
        cand_model_fingerprint = maybe_file_fingerprint(args.cand_model)
    meta = {
        "kind":        "llm_kld_metrics",
        "vlmk_version": VLMK_VERSION,
        "ref_model":   str(Path(args.ref_model).resolve()),
        "cand_model":  str(Path(args.cand_model).resolve()),
        "dataset": args.dataset,
        "subset":  args.subset,
        "split":   args.split,
        "sort_by": args.sort_by,
        "sort_desc": args.sort_desc,
        "num_eval_tokens": args.num_eval_tokens,
        "max_total_tokens": getattr(args, "max_total_tokens", None),
        "tf_chunk":  args.tf_chunk,
        "n_ctx":     args.n_ctx,
        "n_batch":   args.n_batch,
        "n_ubatch":  args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "metric_threads": args.metric_threads,
        "flash_attn": args.flash_attn,
        "swa_full": bool(args.swa_full),
        # Keep this outside IDENTITY_FIELDS for legacy collections; native logs record accepted mismatches.
        "allow_vocab_attr_mismatch": bool(getattr(args, "allow_vocab_attr_mismatch", False)),
        "perplexity_window": bool(getattr(args, "perplexity_window", False)),
        "corpus_protocol": None,
        "corpus_windows_sha256": None,
    }
    if getattr(args, "perplexity_window", False):
        from lib.text_corpus import digest_json
        meta["perplexity_window"] = True
        corpus = getattr(args, "_corpus_manifest", None)
        if corpus is not None:
            meta["corpus_protocol"] = corpus["protocol"]
            meta["corpus_windows_sha256"] = digest_json(corpus)
    if dataset_content_hash is not None:
        meta["dataset_content_hash"] = dataset_content_hash
    if ref_model_fingerprint is not None:
        meta["ref_model_fingerprint"] = ref_model_fingerprint
    if cand_model_fingerprint is not None:
        meta["cand_model_fingerprint"] = cand_model_fingerprint
    meta.update(build_collect_provenance(getattr(args, "llama_llm_kld", None)))
    return meta


def ensure_collect_meta(out_dir: Path, current: dict) -> bool:
    """This collector's IDENTITY_FIELDS bound to the shared implementation
    (collect_common.ensure_collect_meta)."""
    return _ensure_collect_meta(out_dir, current, IDENTITY_FIELDS,
                                MODEL_FINGERPRINT_FIELDS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect per-row text-only LLM KLD metrics over a HF dataset")
    p.add_argument("--ref-model",   required=True, help="Reference GGUF LLM path (e.g. F16)")
    p.add_argument("--cand-model",  required=True, help="Candidate GGUF LLM path (e.g. Q4_K_M)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dataset", default="company/llm-ground-truth-general-fix-double-BOS",
                   help="HF dataset name")
    p.add_argument("--subset", default="Qwen3-4B-Instruct-2507-vllm")
    p.add_argument("--split", default="train")
    p.add_argument("--sort-by", default="",
                   help="Optional dataset column to sort by. GAP sorts ascending; "
                        "logits repo sorts descending for its similarly named flag.")
    add_collector_selection_args(p)
    p.add_argument("--max-total-tokens", type=int, default=None,
                   help="Dataset filter: skip rows whose GT n_prefill_tokens + "
                        "effective eval tokens exceeds this cap, recording "
                        "SKIP_OVER_BUDGET before scorer prep.")
    add_collector_runtime_args(p)
    p.add_argument("--allow-vocab-attr-mismatch", action="store_true",
                   help="Forwarded to the scorer: accept per-token attribute differences; token texts must still match id by id.")
    p.add_argument("--llama-llm-kld",
                   default=str(REPO_ROOT / "build/bin/llama-llm-kld"))
    p.add_argument("--perplexity-window", action="store_true",
                   help="Decode full classic PPL windows; requires a frozen corpus dataset and n-ctx=n-batch=n-ubatch.")
    p.add_argument("--keep-prep", action="store_true",
                   help="Don't delete per-row prep dir (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    validate_collector_args(args)

    if args.perplexity_window:
        validate_perplexity_runtime(args)

    # Fail fast on missing inputs: a typo otherwise surfaces only after the
    # dataset has already been downloaded and loaded.
    required_paths = {
        "--ref-model":     args.ref_model,
        "--cand-model":    args.cand_model,
        "--llama-llm-kld": args.llama_llm_kld,
    }
    require_paths(required_paths)
    with locked_manifest(args.out, MANIFEST_COLUMNS) as manifest_path:
        _run_locked(args, manifest_path)


def validate_perplexity_runtime(args):
    if (args.n_ctx < 4 or args.n_ctx % 2
            or args.n_batch != args.n_ctx or args.n_ubatch != args.n_ctx
            or args.num_eval_tokens not in (-1, args.n_ctx // 2 - 1)
            or args.tf_chunk not in (-1, args.n_ctx)
            or args.sort_by or args.sort_desc
            or args.max_total_tokens is not None):
        raise ValueError("--perplexity-window requires even n-ctx=n-batch=n-ubatch, all half-window targets, no sorting or token-budget filtering, and tf-chunk=-1 or n-ctx")


def _load_dataset(args):
    # Imported here (not at module top) so the pure helpers stay importable
    # for unit tests without the heavy `datasets` dependency.
    import cli.prep_llm_score_from_hf as prep_lib
    ds = prep_lib.load_dataset_sorted(
        args.dataset, args.subset, args.split, args.sort_by, args.sort_desc)
    has_corpus = len(ds) > 0 and bool(ds[0].get("corpus_protocol"))
    if has_corpus != bool(getattr(args, "perplexity_window", False)):
        raise ValueError("frozen corpus rows require --perplexity-window; this mode requires frozen corpus rows")
    if has_corpus:
        from lib.text_corpus import validate_corpus_dataset
        validate_perplexity_runtime(args)
        args._corpus_manifest = validate_corpus_dataset(ds, args.n_ctx)
    return ds


def _stamp_meta(args, ds_hash):
    ref_fp = logged_file_fingerprint("ref_model_fingerprint", args.ref_model)
    cand_fp = logged_file_fingerprint("cand_model_fingerprint", args.cand_model)
    meta = build_collect_meta(args, ds_hash,
        ref_model_fingerprint=ref_fp, cand_model_fingerprint=cand_fp)
    ensure_collect_meta(args.out, meta)
    if getattr(args, "perplexity_window", False):
        from lib.reference_dataset import canonical_json
        path = args.out / "corpus_windows.json"
        payload = canonical_json(args._corpus_manifest) + "\n"
        if path.exists() and path.read_text() != payload:
            raise ValueError("stored corpus window map differs from the prepared dataset")
        if not path.exists():
            path.write_text(payload)
    args._recorded_execution_identity = meta["execution_identity"]


def _verify_execution(args):
    current = build_collect_provenance(args.llama_llm_kld)["execution_identity"]
    if current != args._recorded_execution_identity:
        raise ValueError("scorer execution identity changed during collection")


def _header_lines(args):
    return [
        f"  ref_model     = {Path(args.ref_model).name}",
        f"  cand_model    = {Path(args.cand_model).name}",
        f"  n_eval_tokens = {args.num_eval_tokens}",
        *( [f"  max_total_tok = {args.max_total_tokens}"]
           if args.max_total_tokens is not None else [] ),
        *( [f"  sort_by       = {args.sort_by}"
            f"{' (desc)' if args.sort_desc else ' (asc)'}"]
           if args.sort_by else [] ),
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


def _prep(args, row, prep_dir, state):
    import cli.prep_llm_score_from_hf as prep_lib
    meta = prep_lib.prep_row(row, prep_dir)
    return {"n_prefill": meta["n_prefill"], "n_answer": meta["n_answer"],
            "reference_vocabulary": meta.get("reference_vocabulary"),
            **({"corpus_window": meta["corpus_window"]} if "corpus_window" in meta else {})}


def _scorer_argv(args, kld_manifest_path):
    return [
        args.llama_llm_kld,
        "--ref-model",   args.ref_model,
        "--cand-model",  args.cand_model,
        "--manifest", str(kld_manifest_path),
        "--num-eval-tokens", str(args.num_eval_tokens),
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
        *(["--perplexity-window"] if getattr(args, "perplexity_window", False) else []),
    ]


def _spec():
    """This collector's divergence points, late-bound to THIS module so the
    tests' monkeypatch surface (dataset_content_hash, prep_lib,
    preflight_scorer_vlmk_version, ...) keeps intercepting. The lifecycle
    itself lives in collect_core.run_locked."""
    return collect_core.make_spec(
        pre_run=lambda args: preflight_scorer_vlmk_version(args.llama_llm_kld),
        load_dataset=_load_dataset,
        limit_end=lambda args, end: apply_dataset_limit(args.start, end, args.dataset_limit),
        item_ids=lambda ds: [str(x) for x in ds["id"]],
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
        budget_extra=lambda row: {},
        prep_state=lambda args: None,
        prep=_prep,
        scorer_display="llama-llm-kld",
        kind_word="kld",
        write_manifest=write_kld_manifest,
        manifest_entry=lambda row: build_kld_manifest_entry(
            row["prep_dir"], row["metrics_path"], row["n_prefill"], row.get("reference_vocabulary")),
        scorer_argv=_scorer_argv,
        scorer_cmd=lambda args: args.llama_llm_kld,
        parse_done=parse_kld_done_line,
        saved_line=lambda path, wall_s:
            f"[llm-kld]  saved metrics {Path(path).name} in {wall_s:.1f}s",
        output_key="metrics_path",
        postprocess=lambda args, row, elapsed_s: postprocess_kld_result(
            row, args.num_eval_tokens, elapsed_s),
        postprocessed_line=lambda row, elapsed_s:
            f"{COLLECT_LOG_PREFIX} postprocessed {row['item_id']}: "
            f"n_ans={row['n_answer']} wall={elapsed_s:.1f}s",
        log_prefix=COLLECT_LOG_PREFIX,
    )


def _run_locked(args, manifest_path):
    """Everything after the output-root lock is held."""
    collect_core.run_locked(args, manifest_path, _spec())


if __name__ == "__main__":
    main()
