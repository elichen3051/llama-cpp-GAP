#!/usr/bin/env python3
"""Collect one model/subset/candidate using the study's archived runtime profile."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import sys
import signal
import threading
import socket

COMPANY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPANY))
from lib.reference_study import kld_runtime, study_overview, validate_reference_cohort
from lib.reference_run import atomic_json
from lib.collection_state import require_completed_attempts
from cli.run_reference_campaign import snapshot, tree_hashes, ProcessSupervisor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--size", type=int, choices=[100, 500], default=100)
    p.add_argument("--tail-400", action="store_true",
                   help="freeze a pinned materialized tail400 config with --size 500 in a NEW study")
    p.add_argument("--reference-revision", help="exact 40-character Hub commit; required with --tail-400")
    p.add_argument("--reference-cache-dir", type=Path, help="optional Hub download cache for --tail-400")
    p.add_argument("--freeze-only", action="store_true", help="validate and freeze tail400 without starting the scorer")
    p.add_argument("--metric-threads", type=int, help="CPU metric workers; default is the archived profile value")
    p.add_argument("--profiles", type=Path, required=True, help="explicit pilot100 or collect500 profile JSON from profiles/")
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--mode", required=True, choices=["instruct", "thinking"])
    p.add_argument("--candidate", required=True, help="stable directory label for this candidate")
    p.add_argument("--cand-model", required=True, type=Path)
    p.add_argument("--cand-mmproj", type=Path, help="defaults to the reference BF16 projector")
    p.add_argument("--gpu", required=True)
    p.add_argument("--models-dir", type=Path, help="model root on this scoring host; default ~/models")
    p.add_argument("--llama-vlm-kld", required=True, type=Path)
    p.add_argument("--dataset", type=Path, help="optional local native dataset; default is the published reference config")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.tail_400 and args.size != 500:
        p.error("--tail-400 requires --size 500 and its collect500 profiles")
    if args.tail_400 and args.dataset:
        p.error("--tail-400 requires a pinned materialized config; --dataset would bypass its validation")
    if args.tail_400 and not re.fullmatch(r"[0-9a-f]{40}", args.reference_revision or ""):
        p.error("--tail-400 requires --reference-revision with an exact 40-character Hub commit")
    if not args.tail_400 and (args.reference_revision or args.reference_cache_dir or args.freeze_only):
        p.error("reference freeze options require --tail-400")
    if args.freeze_only and args.dry_run:
        p.error("--freeze-only and --dry-run are mutually exclusive")
    if args.metric_threads is not None and args.metric_threads < 1:
        p.error("--metric-threads must be positive")
    if not args.gpu or "," in args.gpu or args.gpu == "-1":
        p.error("--gpu must identify one GPU")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.candidate) is None:
        p.error("--candidate must be a filesystem-safe label")
    return args


def build_command(args, plan, profiles, scripts):
    validate_reference_cohort(profiles, plan["size"])
    if plan.get("reference_tail_400") and (plan["size"] != 500 or args.dataset
            or not re.fullmatch(r"[0-9a-f]{40}", args.reference_revision or "")):
        raise ValueError("tail400 studies require a pinned materialized config and collect500 profile")
    for value, key in ((args.model, "models"), (args.source, "sources"), (args.mode, "modes")):
        if value not in plan[key]:
            raise ValueError(f"{value!r} is not in the study's {key}")
    runtime = kld_runtime(profiles, args.model, args.mode, plan["hardware"])
    if "metric_threads" in plan:
        runtime["metric_threads"] = plan["metric_threads"]
    profile = profiles["models"][args.model]
    model_root = (args.models_dir or Path(plan["models_dir"])).expanduser().resolve()
    cohort = "tail-400" if plan.get("reference_tail_400") else f"subsample-{plan['size']}"
    subset = f"{args.source}-{cohort}-" + ("ins" if args.mode == "instruct" else "think")
    out = args.study.expanduser().resolve() / "artifacts" / args.model / subset / "kld" / args.candidate
    dataset = str(args.dataset.expanduser().resolve()) if args.dataset else (
        f"user-company/{args.model}-" + ("collect-400" if plan.get("reference_tail_400") else
                                           "pilot" if plan["size"] == 100 else "collect-500"))
    if plan.get("reference_tail_400"):
        dataset = str(args.study.expanduser().resolve() / "references" / args.model / subset / "dataset")
    command = [sys.executable, str(scripts / "cli/collect_kld.py"),
               "--dataset", dataset, "--subset", "" if args.dataset or plan.get("reference_tail_400") else subset, "--split", "train",
               "--ref-model", str(model_root / profile["model"]), "--ref-mmproj", str(model_root / profile["mmproj"]),
               "--cand-model", str(args.cand_model.expanduser().resolve()),
               "--cand-mmproj", str(args.cand_mmproj.expanduser().resolve() if args.cand_mmproj else model_root / profile["mmproj"]),
               "--llama-vlm-kld", str(args.llama_vlm_kld.expanduser().resolve()), "--out", str(out),
               "--sort-by", "num_images", "--flash-attn"]
    for key in ("n_ctx", "n_batch", "n_ubatch", "tf_chunk", "n_gpu_layers", "n_threads", "metric_threads", "num_eval_tokens"):
        command += ["--" + key.replace("_", "-"), str(runtime[key])]
    if runtime["allow_vocab_attr_mismatch"]:
        command.append("--allow-vocab-attr-mismatch")
    return command, out


@contextmanager
def study_lock(study):
    study.parent.mkdir(parents=True, exist_ok=True)
    with (study.parent / (study.name + ".metadata.lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            study.mkdir(parents=True, exist_ok=True)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def archived_dispatch(study):
    archive = study / "scripts"
    entry = archive / "company/cli/collect_model_kld.py"
    if entry.is_file() and entry.resolve() != Path(__file__).resolve():
        if json.loads((archive / "manifest.json").read_text()) != tree_hashes(archive):
            raise ValueError("archived scripts changed; use the original snapshot")
        os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])


def prepare_study(args):
    study = args.study.expanduser().resolve()
    model_root = (args.models_dir or Path.home() / "models").expanduser().resolve()
    with study_lock(study):
        existing = study / "scripts/gap/cli/collect_model_kld.py"
        if existing.is_file() and existing.resolve() != Path(__file__).resolve():
            return study, existing.parents[1], None, None
        scripts = snapshot(study, args.profiles)
        profiles = json.loads((scripts / "profiles/reference_model_profiles.json").read_text())
        validate_reference_cohort(profiles, args.size)
        plan = {"stage": "kld", "hardware": "pro6000", "size": args.size, "num_samples": None,
                "models_dir": str(model_root), "models": list(profiles["models"]),
                "sources": profiles["sources"], "modes": ["instruct", "thinking"]}
        if args.tail_400:
            plan["reference_tail_400"] = True
        if args.metric_threads is not None:
            plan["metric_threads"] = args.metric_threads
        path = study / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("study settings differ; use a new KLD study directory")
        atomic_json(path, plan)
        atomic_json(study / "study.json", study_overview(profiles, plan))
    return study, scripts, profiles, plan


def record_status(study, out, status, **extra):
    with study_lock(study):
        path = study / "status.json"
        value = json.loads(path.read_text()) if path.exists() else {"stage": "kld", "collections": {}}
        value["collections"][str(out.relative_to(study))] = {
            "status": status, "pid": os.getpid(), "host": socket.gethostname(), **extra}
        value["status"] = "running" if any(job["status"] == "running" for job in value["collections"].values()) else "idle"
        atomic_json(path, value)


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--study", type=Path)
    location, _ = bootstrap.parse_known_args()
    if location.study is not None:
        archived_dispatch(location.study.expanduser().resolve())
    args = parse_args()
    study, scripts, profiles, plan = prepare_study(args)
    archived_dispatch(study)
    command, out = build_command(args, plan, profiles, scripts)
    print(shlex.join([f"CUDA_VISIBLE_DEVICES={args.gpu}", *command]), flush=True)
    if args.dry_run:
        if plan.get("reference_tail_400"):
            print(f"dry run: reference {args.reference_revision} has not been frozen or validated", flush=True)
        return 0
    receipt = None
    if plan.get("reference_tail_400"):
        from lib.reference_freeze import freeze_reference
        subset = out.parents[1].name
        receipt = freeze_reference(scripts / "profiles/reference_model_profiles.json", args.model,
                                   f"user-company/{args.model}-collect-400", args.reference_revision,
                                   subset, study / "references" / args.model / subset, args.reference_cache_dir)
        if args.freeze_only:
            print(json.dumps(receipt), flush=True)
            return 0
    out.mkdir(parents=True, exist_ok=False)
    if receipt is not None:
        atomic_json(out / "reference-freeze.json", receipt)
    stopped = threading.Event()
    previous = {s: signal.signal(s, lambda *_: stopped.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    code, state, error = 1, "failed", None
    try:
        record_status(study, out, "running", gpu=args.gpu)
        code = ProcessSupervisor(stopped).execute(command, out / "collect.log", gpu=args.gpu, timeout=172800)
        if code == 0:
            require_completed_attempts(out)
            state = "complete"
        else:
            error = f"collector exited with code {code}"
    except InterruptedError as exc:
        code, state, error = 130, "interrupted", str(exc)
    except Exception as exc:
        code, error = 1, str(exc)
    finally:
        record_status(study, out, state, gpu=args.gpu, exit_code=code, error=error)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if error:
        print(error, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
