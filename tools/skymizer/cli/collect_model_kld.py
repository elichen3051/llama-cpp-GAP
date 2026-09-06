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

SKYMIZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYMIZER))
from lib.reference_study import kld_runtime, study_overview
from lib.reference_run import atomic_json
from lib.collection_state import require_completed_attempts
from cli.run_reference_campaign import snapshot, tree_hashes, ProcessSupervisor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--size", type=int, choices=[100, 500], default=100)
    p.add_argument("--profiles", type=Path, default=SKYMIZER / "scripts/reference_model_profiles.json")
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
    if not args.gpu or "," in args.gpu or args.gpu == "-1":
        p.error("--gpu must identify one GPU")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.candidate) is None:
        p.error("--candidate must be a filesystem-safe label")
    return args


def build_command(args, plan, profiles, scripts):
    for value, key in ((args.model, "models"), (args.source, "sources"), (args.mode, "modes")):
        if value not in plan[key]:
            raise ValueError(f"{value!r} is not in the study's {key}")
    runtime = kld_runtime(profiles, args.model, args.mode, plan["hardware"])
    profile = profiles["models"][args.model]
    model_root = (args.models_dir or Path(plan["models_dir"])).expanduser().resolve()
    subset = f"{args.source}-subsample-{plan['size']}-" + ("ins" if args.mode == "instruct" else "think")
    out = args.study.expanduser().resolve() / "artifacts" / args.model / subset / "kld" / args.candidate
    dataset = str(args.dataset.expanduser().resolve()) if args.dataset else (
        f"elichen-skymizer/{args.model}-" + ("pivot" if plan["size"] == 100 else "collect-500"))
    command = [sys.executable, str(scripts / "cli/collect_kld.py"),
               "--dataset", dataset, "--subset", "" if args.dataset else subset, "--split", "train",
               "--ref-model", str(model_root / profile["model"]), "--ref-mmproj", str(model_root / profile["mmproj"]),
               "--cand-model", str(args.cand_model.expanduser().resolve()),
               "--cand-mmproj", str(args.cand_mmproj.expanduser().resolve() if args.cand_mmproj else model_root / profile["mmproj"]),
               "--llama-vlm-kld", str(args.llama_vlm_kld.expanduser().resolve()), "--out", str(out),
               "--sort-by", "num_images", "--flash-attn"]
    for key in ("n_ctx", "n_batch", "n_ubatch", "tf_chunk", "n_gpu_layers", "n_threads", "metric_threads", "num_eval_tokens"):
        command += ["--" + key.replace("_", "-"), str(runtime[key])]
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
    entry = archive / "skymizer/cli/collect_model_kld.py"
    if entry.is_file() and entry.resolve() != Path(__file__).resolve():
        if json.loads((archive / "manifest.json").read_text()) != tree_hashes(archive):
            raise ValueError("archived scripts changed; use the original snapshot")
        os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])


def prepare_study(args):
    study = args.study.expanduser().resolve()
    model_root = (args.models_dir or Path.home() / "models").expanduser().resolve()
    with study_lock(study):
        existing = study / "scripts/skymizer/cli/collect_model_kld.py"
        if existing.is_file() and existing.resolve() != Path(__file__).resolve():
            return study, existing.parents[1], None, None
        scripts = snapshot(study, args.profiles)
        profiles = json.loads((scripts / "scripts/reference_model_profiles.json").read_text())
        plan = {"stage": "kld", "hardware": "pro6000", "size": args.size, "num_samples": None,
                "models_dir": str(model_root), "models": list(profiles["models"]),
                "sources": profiles["sources"], "modes": ["instruct", "thinking"]}
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
        return 0
    out.mkdir(parents=True, exist_ok=False)
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
