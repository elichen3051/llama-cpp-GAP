"""Shared dataset, preparation, scoring and completion lifecycle for both KLD lanes."""

import os
import signal
import threading
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from lib.collection_state import CollectionAttempt, refuse_unfinished_attempts

from lib.collect_common import (
    SKIP_OVER_BUDGET,
    format_row_progress,
    collision_error,
    manifest_row_writer,
    max_total_tokens_skip_info,
    preflight_n_ctx,
    root_has_prior_output,
    scan_output_collisions,
)


def make_spec(**kw) -> SimpleNamespace:
    """A CollectorSpec is a plain namespace; see run_locked for the fields."""
    return SimpleNamespace(**kw)


def run_locked(args, manifest_path, spec):
    refuse_unfinished_attempts(args.out)
    attempt = CollectionAttempt(args.out, args.start, args.end)
    previous = None
    if threading.current_thread() is threading.main_thread():
        def terminate(signum, frame):
            raise KeyboardInterrupt(f"collection interrupted by signal {signum}")
        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        _run_attempt(args, manifest_path, spec, attempt)
        attempt.finish()
    except BaseException as error:
        attempt.stop(error)
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _run_attempt(args, manifest_path, spec, attempt):
    """Everything after the output-root lock is held. `spec` fields:

    pre_run(args)                             e.g. scorer-binary preflight; no-op default
    load_dataset(args) -> ds                  lazy prep-lib import + sort
    limit_end(args, end) -> end               e.g. --dataset-limit; identity default
    item_ids(ds) -> list[str]                 the id column, tool-specific
    dataset_hash(ds) -> str                   wraps the module's dataset_content_hash
    stamp_meta(args, ds_hash)                 fingerprints + ensure_collect_meta
    manifest_columns: tuple[str, ...]
    artifact_subdirs: tuple[str, ...]         e.g. ("metrics",)
    header_lines(args) -> list[str]           the per-tool settings block
    log_filename: str                         e.g. "score_run.log" / "kld_run.log"
    dump_stem(idx, item_id) -> str
    row_paths(args, key) -> dict              per-row artifact path entries
    budget_extra(row) -> dict                 extra SKIP_OVER_BUDGET columns
    prep_state(args) -> Any                   e.g. tokenizer cache + vision budget
    prep(args, row, prep_dir, state) -> dict  n_prefill/n_answer(/n_images); raises on failure
    scorer_display: str                       name in the "prep complete" line
    kind_word: str                            "score" / "kld" in log + error lines
    write_manifest(path, entries)             the module's JSONL writer
    manifest_entry(row) -> dict               the module's per-row JSONL entry
    scorer_argv(args, manifest_path) -> list[str]
    scorer_cmd(args) -> str                   for CalledProcessError
    parse_done(line) -> (path, wall)|None
    saved_line(path, wall) -> str
    output_key: str                           e.g. "metrics_path"
    postprocess(args, row, elapsed_s) -> list manifest.csv row
    postprocessed_line(row, elapsed_s) -> str
    log_prefix: str                           the module's COLLECT_LOG_PREFIX
    """
    spec.pre_run(args)

    print(f"loading dataset {args.dataset} | {args.subset} | {args.split}", file=sys.stderr)
    ds = spec.load_dataset(args)

    item_ids = spec.item_ids(ds)
    n_rows = len(item_ids)
    end, end_warning = spec.resolve_dataset_end(args.end, n_rows)
    if end_warning:
        print(end_warning, file=sys.stderr)
    end = spec.limit_end(args, end)
    if args.start >= end:
        sys.exit(f"--start {args.start} >= end {end}: the requested window is "
                 f"empty (dataset has {n_rows} row(s)), nothing would be "
                 "collected; check the shard bounds")
    try:
        conflicts = scan_output_collisions(args.out, range(args.start, end), manifest_path)
    except ValueError as e:
        sys.exit(str(e))
    if conflicts:
        sys.exit(collision_error(conflicts, args.start, end))

    # First shard: record identity + provenance + content fingerprints.
    # Later shard: IDENTITY_FIELDS must match exactly AND the content tiers
    # are re-derived and verified (whole-dataset hash — the shard window does
    # not affect it — and the model-file fingerprints), so a model replaced
    # IN PLACE or a dataset regenerated under the same name between shards is
    # refused. A meta that predates a fingerprint field warns and proceeds.
    if not (args.out / "collect_meta.json").exists():
        # A root with rows but no collect_meta.json cannot be given an
        # identity after the fact: refuse rather than stamp fresh metadata
        # over rows of unknown provenance.
        try:
            prior = root_has_prior_output(args.out, manifest_path)
        except ValueError as e:      # damaged manifest
            sys.exit(str(e))
        if prior is not None:
            sys.exit(f"{args.out}: holds prior output ({prior}) but no "
                     "collect_meta.json; cannot establish this root's identity "
                     "— use a fresh --out. Nothing was deleted.")
    hash_t0 = time.time()
    ds_hash = spec.dataset_hash(ds)
    print(f"  dataset_content_hash = {ds_hash[:16]}… "
          f"({time.time() - hash_t0:.2f}s)", file=sys.stderr)
    spec.stamp_meta(args, ds_hash)
    attempt.declare((idx, item_ids[idx]) for idx in range(args.start, end))
    for sub in (*spec.artifact_subdirs, "logs", "_prep"):
        (args.out / sub).mkdir(exist_ok=True)
    print(f"dataset has {n_rows} rows; sweeping [{args.start}, {end})", file=sys.stderr)
    for line in spec.header_lines(args):
        print(line, file=sys.stderr)

    score_log_path = args.out / "logs" / spec.log_filename

    with open(manifest_path, "a", newline="") as mf:
        raw_write_row = manifest_row_writer(mf, manifest_path, spec.manifest_columns)
        def write_row(row):
            raw_write_row(row)
            mf.flush()
            os.fsync(mf.fileno())
            record = row if isinstance(row, dict) else dict(zip(spec.manifest_columns, row))
            attempt.record(record)

        n_failed = 0
        n_skipped_over_budget = 0
        pending = []
        prep_start = time.time()
        state = spec.prep_state(args)
        for idx in range(args.start, end):
            item_id = item_ids[idx]
            key = spec.dump_stem(idx, item_id)
            paths = spec.row_paths(args, key)
            prep_dir = args.out / "_prep" / key

            # Budget filter first: over-budget rows never reach prep. The
            # dataset row is materialized here ONLY when the filter is on.
            row = None
            if args.max_total_tokens is not None:
                row = ds[idx]
                skip_info = max_total_tokens_skip_info(
                    row, args.num_eval_tokens, args.max_total_tokens)
                if skip_info is not None:
                    n_skipped_over_budget += 1
                    write_row({"row_idx": idx,
                               "item_id": item_id,
                               **spec.budget_extra(row),
                               "n_prefill": skip_info["n_prefill"],
                               "n_answer": skip_info["n_answer"],
                               "n_eval": skip_info["n_eval"],
                               "status": SKIP_OVER_BUDGET})
                    mf.flush()
                    print(f"{format_row_progress(idx, n_rows)} {item_id}: "
                          f"{SKIP_OVER_BUDGET} ({skip_info['error']})",
                          file=sys.stderr)
                    continue

            prep_start_t = time.time()
            try:
                # --- prep (in-process; dataset loaded once per sweep). The
                # collision scan guaranteed prep_dir does not exist. ---
                if row is None:
                    row = ds[idx]
                meta = spec.prep(args, row, prep_dir, state)
                if row.get("generation_schema_version"):
                    import json
                    attempt.reference(idx, item_id, json.loads(row["generation_metadata"]), {
                        key: row.get(key) for key in ("generation_request", "generation_sampling_params",
                            "generation_enable_thinking", "generation_chat_template_kwargs",
                            "generation_token_logprobs", "generation_decoding_stats", "finish_reason")})

                prep_wall_s = time.time() - prep_start_t
                pending.append({
                    "idx": idx,
                    "item_id": item_id,
                    "key": key,
                    "prep_dir": prep_dir,
                    **paths,
                    **meta,
                    "prep_wall_s": prep_wall_s,
                    "prep_start_t": prep_start_t,   # fallback if scorer omits per-row timing
                })

            except Exception as e:
                n_failed += 1
                # Remove OUR OWN half-built scratch before recording the FAIL,
                # so the row leaves no artifact and scan_output_collisions'
                # retryable-FAIL exemption can actually fire on the re-run.
                # prep_row mkdirs prep_dir as its first statement, so the dir
                # exists for every failure after that point and its presence
                # would otherwise collide forever (_prep is an ARTIFACT_SUBDIR).
                # Same rationale as the n_ctx preflight's rmtree. --keep-prep
                # keeps it for inspection and the row then collides on re-run,
                # by design. rmtree BEFORE write_row: a crash in between leaves
                # neither artifact nor record (a genuinely fresh row), never a
                # record plus scratch.
                if not args.keep_prep:
                    shutil.rmtree(prep_dir, ignore_errors=True)
                write_row({"row_idx": idx,
                           "item_id": item_id,
                           "wall_s": f"{time.time()-prep_start_t:.2f}",
                           "status": f"FAIL_{type(e).__name__}"})
                mf.flush()
                print(f"{format_row_progress(idx, n_rows)} {item_id}: "
                      f"FAIL {type(e).__name__}: {e}",
                      file=sys.stderr)

        if pending:
            try:
                preflight_n_ctx(
                    pending, args.n_ctx, args.num_eval_tokens,
                    n_seq_max=getattr(args, "n_seq_max", 1))
            except SystemExit:
                # The n_ctx preflight exists so the operator fixes the flag and
                # re-runs. Every _prep dir here was created by THIS run and
                # nothing else was produced, so remove our own scratch: left
                # behind it would collide with the corrected re-run. (Prior
                # output is still never touched.)
                for row in pending:
                    shutil.rmtree(row["prep_dir"], ignore_errors=True)
                raise
            print(f"prep complete: {len(pending)} row(s) ready in "
                  f"{time.time()-prep_start:.1f}s; running {spec.scorer_display}",
                  file=sys.stderr)
            score_manifest_path = args.out / "_manifest.jsonl"
            spec.write_manifest(
                score_manifest_path,
                [spec.manifest_entry(row) for row in pending],
            )

            spec.verify_execution(args)
            score_returncode = None
            score_error = None
            score_interrupted = None
            proc = None
            # Parsed "DONE output_*=... wall_s=..." markers, keyed by the exact
            # path the scorer was handed (same path we put in the manifest
            # entry, so str(output path) round-trips).
            score_wall_s_by_path = {}
            try:
                with open(score_log_path, "a") as log_f:
                    log_f.write(f"\n=== {spec.kind_word} manifest {score_manifest_path} ===\n")
                    log_f.flush()
                    # Popen + line-by-line read so per-row "DONE" timings can be
                    # parsed and a short progress line printed as each row
                    # finishes (instead of one bulk dump at the end).
                    proc = subprocess.Popen(
                        spec.scorer_argv(args, score_manifest_path),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,
                        text=True, bufsize=1, errors="replace",
                    )
                    for line in proc.stderr:
                        log_f.write(line)
                        log_f.flush()
                        parsed = spec.parse_done(line)
                        if parsed is not None:
                            path, wall_s = parsed
                            score_wall_s_by_path[path] = wall_s
                            print(spec.saved_line(path, wall_s), file=sys.stderr)
                    proc.wait()
                    score_returncode = proc.returncode
            except BaseException as e:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if isinstance(e, Exception):
                    score_error = e
                else:
                    score_interrupted = e
                    score_returncode = proc.returncode if proc is not None else -1
            finally:
                if score_interrupted is None:
                    try:
                        score_manifest_path.unlink()
                    except FileNotFoundError:
                        pass

            if score_error is not None:
                print(f"{spec.kind_word} manifest failed to launch: {score_error}  "
                      f"(see {score_log_path})", file=sys.stderr)
            elif score_returncode != 0:
                print(f"{spec.kind_word} manifest exited {score_returncode}; postprocessing any "
                      f"completed rows (see {score_log_path})", file=sys.stderr)

            for row in pending:
                try:
                    if score_error is not None:
                        raise score_error
                    if score_returncode != 0 and not row[spec.output_key].exists():
                        raise subprocess.CalledProcessError(
                            score_returncode, spec.scorer_cmd(args))

                    # wall_s in manifest.csv = this row's prep + this row's
                    # score (matches per-row mode's old meaning). Fallback to
                    # "time since prep started" if the scorer didn't emit a
                    # DONE marker (e.g. older scorer binary).
                    score_wall_s = score_wall_s_by_path.get(str(row[spec.output_key]))
                    if score_wall_s is not None:
                        elapsed_s = row["prep_wall_s"] + score_wall_s
                    else:
                        elapsed_s = time.time() - row["prep_start_t"]

                    manifest_row = spec.postprocess(args, row, elapsed_s)
                    write_row(manifest_row)
                    mf.flush()
                    # This row's OK record is now durable. Cleanup failures
                    # (ESTALE, a still-open file, a read-only mount) must NOT
                    # reach the FAIL handlers below: a second record for a row
                    # that already has one breaks the one-record-per-row
                    # invariant the whole output-root policy rests on.
                    try:
                        if not args.keep_prep:
                            shutil.rmtree(row["prep_dir"])
                        print(spec.postprocessed_line(row, elapsed_s), file=sys.stderr)
                    except Exception as cleanup_err:
                        print(f"{spec.log_prefix} {row['item_id']}: recorded OK; "
                              f"post-record cleanup failed: {cleanup_err}",
                              file=sys.stderr)

                except subprocess.CalledProcessError as e:
                    n_failed += 1
                    write_row({"row_idx": row["idx"],
                               "item_id": row["item_id"],
                               "wall_s": f"{time.time()-row['prep_start_t']:.2f}",
                               "status": f"FAIL_EXIT_{e.returncode}"})
                    mf.flush()
                    print(f"{format_row_progress(row['idx'], n_rows)} {row['item_id']}: "
                          f"FAIL exit={e.returncode}  (see {score_log_path})",
                          file=sys.stderr)
                except Exception as e:
                    n_failed += 1
                    write_row({"row_idx": row["idx"],
                               "item_id": row["item_id"],
                               "wall_s": f"{time.time()-row['prep_start_t']:.2f}",
                               "status": f"FAIL_{type(e).__name__}"})
                    mf.flush()
                    print(f"{format_row_progress(row['idx'], n_rows)} {row['item_id']}: "
                          f"FAIL {type(e).__name__}: {e}  (see {score_log_path})",
                          file=sys.stderr)

            if score_interrupted is not None:
                raise score_interrupted

    try:
        (args.out / "_prep").rmdir()
    except OSError:
        pass

    if n_skipped_over_budget:
        print(f"skipped {n_skipped_over_budget} over-budget row(s) "
              f"(max-total-tokens={args.max_total_tokens})",
              file=sys.stderr)

    # Exit non-zero when any row failed: pipeline scripts run under `set -e`
    # and must not sail past a partial collection.
    if n_failed:
        sys.exit(f"collection complete with {n_failed} failed row(s); "
                 f"manifest -> {manifest_path}")
    spec.verify_execution(args)
    print(f"collection complete; manifest -> {manifest_path}", file=sys.stderr)
