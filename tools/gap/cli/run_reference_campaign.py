#!/usr/bin/env python3
"""Run a shared GPU queue with archived evidence and immutable cohort uploads.

Resume verifies completed artifacts. A host restart or cancelled Python job reruns
that whole job in a new attempt directory; native crash recovery preserves rows
inside one driver invocation. Completed cohorts with row failures remain final.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

COMPANY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPANY))
from lib.collect_meta_provenance import execution_identity
from lib.reference_dataset import sha256_file
from lib.reference_run import atomic_json, read_records
from lib.reference_study import model_modes, study_overview, validate_reference_cohort


@contextmanager
def campaign_owner(out, resume):
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking a lock can give two processes different locks.
    with (out.parent / (out.name + ".lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"campaign already has an active owner: {out}") from error
        try:
            if out.exists() and not resume:
                raise ValueError("campaign directory exists; use --resume to preserve its evidence")
            out.mkdir(parents=True, exist_ok=True)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tree_hashes(directory):
    return {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and p != directory / "manifest.json"}


def snapshot(out, profiles):
    archive = out / "scripts"
    if archive.exists():
        if json.loads((archive / "manifest.json").read_text()) != tree_hashes(archive):
            raise ValueError("archived scripts or provenance changed; restore the original snapshot")
        return archive / "company"
    staging = Path(tempfile.mkdtemp(prefix=".scripts-", dir=out))
    try:
        directory = staging / "company"
        directory.mkdir()
        for name in ("core", "cli", "lib", "stats", "scripts", "profiles"):
            shutil.copytree(COMPANY / name, directory / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for pattern in ("pyproject.toml", "uv.lock", ".python-version", "CMakeLists.txt"):
            for path in COMPANY.glob(pattern):
                shutil.copyfile(path, directory / path.name)
        shutil.copyfile(profiles, directory / "profiles/reference_model_profiles.json")
        provenance = staging / "provenance"
        provenance.mkdir()
        repo = COMPANY.parents[1]
        commands = {
            "source-base.txt": ["git", "rev-parse", "HEAD"],
            "source-diff.patch": ["git", "diff", "--binary", "HEAD"],
            "source-status.txt": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            "submodules.txt": ["git", "submodule", "status", "--recursive"],
            "gpu.txt": ["nvidia-smi"],
        }
        for name, command in commands.items():
            with (provenance / name).open("w") as log:
                try:
                    result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                    if result.returncode and name.startswith("source-"):
                        raise ValueError(f"cannot capture source provenance: {name}")
                except (OSError, subprocess.TimeoutExpired) as error:
                    if name.startswith("source-"):
                        raise
                    log.write(str(error) + "\n")
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo, timeout=60)
        atomic_json(provenance / "untracked-sha256.json", {
            name: sha256_file(repo / name) for name in os.fsdecode(untracked).split("\0") if name and (repo / name).is_file()})
        atomic_json(provenance / "python.json", {"executable": sys.executable, "version": sys.version})
        atomic_json(provenance / "python-packages.json", sorted(
            ({"name": dist.metadata["Name"], "version": dist.version} for dist in distributions()),
            key=lambda item: (item["name"].lower(), item["version"])))
        atomic_json(staging / "manifest.json", tree_hashes(staging))
        staging.replace(archive)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return archive / "company"


def require_current_uploader(scripts):
    relative = Path("cli/upload_reference.py")
    if sha256_file(scripts / relative) != sha256_file(COMPANY / relative):
        raise ValueError("archived uploader differs from this release; private publication requires a new campaign snapshot or the current standalone uploader with the original run and archived profile; preserve the old archive")


class ProcessSupervisor:
    """Each child owns a process group, including all generation wrappers."""

    def __init__(self, stopped, grace=10):
        self.stopped = stopped
        self.grace = grace

    @staticmethod
    def group_alive(pid):
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def cleanup(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self.grace
        while self.group_alive(proc.pid) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(0.05)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def execute(self, command, log_path, gpu=None, timeout=3600):
        if self.stopped.is_set():
            raise InterruptedError("campaign stopped")
        with log_path.open("x") as log:
            atomic_json(log_path.with_suffix(".command.json"), command)
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, **({"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else {})})
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if self.stopped.is_set():
                        raise InterruptedError("campaign stopped")
                    code = proc.poll()
                    if code is not None:
                        return code
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"whole subprocess timeout after {timeout} seconds")
                    self.stopped.wait(0.1)
            finally:
                # Reap descendants even when their wrapper already exited.
                self.cleanup(proc)


def artifact_hashes(run):
    required = ("complete.json", "metadata.json", "run_state.json", "run_start.json", "excluded.jsonl", "failures.jsonl")
    hashes = {name: sha256_file(run / name) for name in required}
    if (run / "native-attempts.json").exists():
        hashes["native-attempts.json"] = sha256_file(run / "native-attempts.json")
    completion = json.loads((run / "complete.json").read_text())
    state = json.loads((run / "run_state.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    cohort = completion["cohort"]
    if (completion["status"] not in ("complete", "complete_with_failures")
            or state["status"] != completion["status"] or state["cohort"] != cohort or metadata["cohort"] != cohort):
        raise ValueError("generation completion, state and metadata disagree")
    requested = cohort["requested_ids"]
    partition = cohort["eligible_ids"] + cohort["excluded_ids"] + cohort["failed_ids"]
    if len(set(requested)) != len(requested) or len(partition) != len(requested) or set(partition) != set(requested):
        raise ValueError("generation cohort is not an exact partition")
    for key in ("requested", "eligible", "excluded", "failed"):
        if type(cohort[key]) is not int or cohort[key] != len(cohort[key + "_ids"]):
            raise ValueError(f"generation cohort count mismatch: {key}")
    if type(completion["rows"]) is not int or completion["rows"] != cohort["eligible"]:
        raise ValueError("generation eligible row count mismatch")
    if (completion["status"] == "complete") != (cohort["failed"] == 0):
        raise ValueError("generation status disagrees with failed row count")
    for name, key in (("excluded.jsonl", "excluded_ids"), ("failures.jsonl", "failed_ids")):
        if [row["id"] for row in read_records(run / name)] != cohort[key]:
            raise ValueError(f"generation audit IDs mismatch: {name}")
    if completion["rows"]:
        if "native-attempts.json" not in hashes:
            raise ValueError("completed generation is missing native-attempts.json")
        if not (run / "dataset/state.json").is_file() or not (run / "dataset/dataset_info.json").is_file():
            raise ValueError("completed dataset is missing")
        dataset_state = json.loads((run / "dataset/state.json").read_text())
        for item in dataset_state["_data_files"]:
            path = run / "dataset" / item["filename"]
            if path.resolve().parent != (run / "dataset").resolve() or not path.is_file():
                raise ValueError("saved dataset shard is missing or outside its directory")
        if not dataset_state["_data_files"]:
            raise ValueError("saved dataset has no shards")
        hashes.update({"dataset/" + name: digest for name, digest in tree_hashes(run / "dataset").items()})
    elif (run / "dataset").exists():
        raise ValueError("zero-row completion unexpectedly has a dataset")
    return hashes


def validate_receipt(run, model, subset, size):
    manifest = json.loads((run / "upload/manifest.json").read_text())
    receipt = json.loads((run / "upload/receipt.json").read_text())
    completion = json.loads((run / "complete.json").read_text())
    repo = f"user-company/{model}-" + ("pilot" if size == 100 else "collect-500")
    expected = {"repo": repo, "subset": subset, "rows": completion["rows"]}
    if any(manifest.get(k) != v or receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("upload receipt does not describe this job")
    if (receipt.get("status") != "verified" or receipt.get("private") is not True or re.fullmatch(r"[0-9a-f]{40,64}", receipt.get("commit", "")) is None
            or manifest.get("cohort") != completion["cohort"] or manifest.get("split") != "train"
            or manifest.get("metadata_sha256") != sha256_file(run / "metadata.json")
            or manifest.get("parquet_sha256") != sha256_file(run / "upload/train-00000-of-00001.parquet")
            or receipt.get("parquet_sha256") != manifest.get("parquet_sha256")):
        raise ValueError("upload receipt or exported artifacts failed verification")
    audit_files = ("metadata.json", "complete.json", "run_start.json", "excluded.jsonl", "failures.jsonl", "native-attempts.json")
    audit_hashes = {name: sha256_file(run / name) for name in audit_files}
    if manifest.get("audit_sha256") != audit_hashes or receipt.get("audit_sha256") != audit_hashes:
        raise ValueError("upload receipt audit hashes differ from the complete local audit")
    return receipt


def next_number(directory, pattern):
    return 1 + max((int(re.search(r"(\d+)(?:\.log)?$", p.name)[1]) for p in directory.glob(pattern)
                    if re.search(r"(\d+)(?:\.log)?$", p.name)), default=0)


def run_campaign(args):
    out = args.out.expanduser().resolve()
    stopped = threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, lambda signum, frame: stopped.set())
    try:
        with campaign_owner(out, args.resume):
            return run_owned_campaign(args, out, stopped)
    finally:
        stopped.set()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def run_owned_campaign(args, out, stopped):
    scripts = snapshot(out, args.profiles)
    if args.upload:
        require_current_uploader(scripts)
    profiles_path = scripts / "profiles/reference_model_profiles.json"
    profiles = json.loads(profiles_path.read_text())
    validate_reference_cohort(profiles, args.size)
    models = args.models or list(profiles["models"])
    sources = args.sources or profiles["sources"]
    for name, values in (("models", models), ("sources", sources), ("modes", args.modes), ("gpus", args.gpus)):
        if not values or len(values) != len(set(values)) or any(not v.strip() for v in values):
            raise ValueError(f"{name} must be nonempty and unique")
    if set(models) - set(profiles["models"]) or set(sources) - set(profiles["sources"]):
        raise ValueError("unknown model or source")
    if set(args.modes) - {"instruct", "thinking"} or "-1" in args.gpus:
        raise ValueError("invalid mode or GPU")
    if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", v) is None for v in models + sources):
        raise ValueError("model and source names must be filesystem-safe")
    for name in ("job_timeout", "upload_timeout", "kill_grace"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    binary = args.llama_reference.expanduser().resolve()
    identity = execution_identity(binary)[0]
    settings = {"models": models, "sources": sources, "modes": args.modes, "size": args.size,
                "hardware": args.hardware, "gpus": args.gpus, "num_samples": args.num_samples,
                "models_dir": str(args.models_dir.expanduser().resolve()), "binary": str(binary),
                "execution_identity": identity, "model_identities": {m: profiles["models"][m].get("identity") for m in models},
                "upload": args.upload, "job_timeout": args.job_timeout,
                "upload_timeout": args.upload_timeout, "kill_grace": args.kill_grace}
    plan_file = out / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != settings:
        raise ValueError("resume arguments or executable/backend identity differ from the frozen campaign plan")
    atomic_json(plan_file, settings)
    atomic_json(out / "study.json", study_overview(profiles, settings))
    priority = ["gemma-4-31b-it", "qwen3.6-35b-a3b", "gemma-4-26b-a4b-it", "glm-4.6v-flash", "gemma-4-e4b-it", "qwen3.5-4b"]
    jobs = [(model, f"{source}-subsample-{args.size}-" + ("ins" if mode == "instruct" else "think"), source, mode)
            for model in sorted(models, key=lambda x: (priority.index(x) if x in priority else len(priority), x))
            for mode in model_modes(profiles["models"][model], args.modes) for source in sources]
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "plan": settings, "archive": str(out)}, indent=2))
        return 130 if stopped.is_set() else 0
    work = queue.Queue()
    for job in jobs:
        work.put(job)
    results, uploads = {}, []
    result_lock, event_lock = threading.Lock(), threading.Lock()
    supervisor = ProcessSupervisor(stopped, args.kill_grace)

    def event(value):
        value["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with event_lock:
            try:
                with (out / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(value) + "\n")
            except OSError as error:
                print(f"cannot persist campaign event: {error}", file=sys.stderr, flush=True)
            print(json.dumps(value), flush=True)

    def record(directory, state):
        try:
            atomic_json(directory / "job.json", state)
        except OSError as error:
            state["status"], state["persistence_error"] = "failed", str(error)
        with result_lock:
            results[str(directory.relative_to(out))] = state
            atomic_json(out / "status.json", {"status": "running", "total_jobs": len(jobs), "jobs": results})
        event({"job": str(directory.relative_to(out)), **state})

    def verify_generation(directory, run, model, source, mode, gpu, saved=None):
        if run.resolve().parent != directory.resolve() or re.fullmatch(r"attempt-\d+", run.name) is None:
            raise ValueError("generation run points outside its job directory")
        hashes = artifact_hashes(run)
        if saved is not None and saved.get("artifacts") != hashes:
            raise ValueError("completed generation artifacts changed; retain the evidence and investigate")
        completion = json.loads((run / "complete.json").read_text())
        metadata = json.loads((run / "metadata.json").read_text())
        source_info = metadata["dataset_source"]
        expected_size = min(args.size, args.num_samples) if args.num_samples else args.size
        if (source_info["path"] != profiles["dataset"]["repo"] or source_info["revision"] != profiles["dataset"]["revision"]
                or source_info["subset"] != f"{source}-subsample-{args.size}" or source_info["split"] != "train"
                or source_info["num_rows"] != expected_size or completion["cohort"]["requested"] != expected_size
                or metadata["requested_enable_thinking"] is not (mode == "thinking")):
            raise ValueError("completed generation differs from the frozen source/mode plan")
        if completion["rows"] or metadata.get("execution_identity") is not None:
            expected_identity = {**identity, "environment": {**identity["environment"], "CUDA_VISIBLE_DEVICES": gpu}}
            if metadata.get("execution_identity") != expected_identity:
                raise ValueError("generated binary/backend identity differs from the campaign plan")
            if Path(metadata["model_path"]).resolve() != Path(settings["models_dir"], profiles["models"][model]["model"]).resolve():
                raise ValueError("generated model differs from the campaign plan")
            pinned = settings["model_identities"][model]
            if pinned:
                actual = {Path(row["path"]).name: (row["size"], row["sha256"]) for row in metadata["model_files"]}
                expected = {row["name"]: (row["size"], row["sha256"]) for row in pinned["files"] if row["role"] == "llm"}
                projector = [row for row in pinned["files"] if row["role"] == "mmproj"]
                if (actual != expected or len(projector) != 1
                        or Path(metadata["mmproj_path"]).name != projector[0]["name"]
                        or metadata.get("mmproj_sha256") != projector[0]["sha256"]):
                    raise ValueError("generated model/projector hashes differ from the pinned identities")
                if metadata.get("decoding", {}).get("mtp", {}).get("head_source") == "sidecar":
                    heads = metadata["decoding"]["mtp"]["head_files"]
                    if {Path(row["path"]).name: (row["size"], row["sha256"]) for row in heads} != {
                            row["name"]: (row["size"], row["sha256"]) for row in pinned.get("head_files", [])}:
                        raise ValueError("generated MTP head differs from the pinned identity")
        return completion, hashes

    def upload(directory, run, model, subset, mode):
        command = [sys.executable, str(scripts / "cli/upload_reference.py"), "--run", str(run),
                   "--model", model, "--mode", mode, "--profiles", str(profiles_path), "--private"]
        for _ in range(3):
            if stopped.is_set():
                return
            try:
                require_current_uploader(scripts)
                number = next_number(directory, "upload-*.log")
                code = supervisor.execute(command, directory / f"upload-{number:04d}.log", timeout=args.upload_timeout)
                if code:
                    raise RuntimeError(f"upload process failed ({code})")
                receipt = validate_receipt(run, model, subset, args.size)
                atomic_json(directory / "upload-status.json", {"status": "verified", "run": str(run), "receipt": receipt})
                event({"job": str(directory.relative_to(out)), "phase": "upload", "status": "verified"})
                return
            except Exception as error:
                event({"job": str(directory.relative_to(out)), "phase": "upload", "error": str(error)})
        try:
            atomic_json(directory / "upload-status.json", {"status": "failed", "run": str(run)})
        except OSError as error:
            event({"job": str(directory.relative_to(out)), "phase": "upload", "error": str(error)})

    upload_pool = ThreadPoolExecutor(max_workers=2)
    pool = ThreadPoolExecutor(max_workers=len(args.gpus))

    def worker(gpu):
        unavailable = set()
        while not stopped.is_set():
            try:
                model, subset, source, mode = work.get_nowait()
            except queue.Empty:
                return
            directory = out / "artifacts" / model / subset
            run, state = None, {}
            try:
                directory.mkdir(parents=True, exist_ok=True)
                state_path = directory / "job.json"
                try:
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                except (ValueError, UnicodeError):
                    shutil.copyfile(state_path, directory / f"job-corrupt-{time.time_ns()}.json")
                    raise ValueError("corrupt job state preserved; this job failed without stopping the queue")
                if state.get("status") == "invalid_artifacts":
                    raise ValueError(state["error"])
                if state.get("status") in ("complete", "complete_with_failures"):
                    run = Path(state["run"])
                    completion, hashes = verify_generation(directory, run, model, source, mode, state["gpu"], state)
                elif (model, mode) in unavailable:
                    record(directory, {"status": "skipped_model_startup_failure", "gpu": gpu, "mode": mode})
                    continue
                else:
                    number = max(next_number(directory, "attempt-*"), next_number(directory, "generation-*.log"))
                    run = directory / f"attempt-{number:04d}"
                    command = [sys.executable, str(scripts / "cli/generate_model_reference.py"),
                               "--model", model, "--mode", mode, "--source", source, "--size", str(args.size),
                               "--hardware", args.hardware, "--gpu", gpu, "--models-dir", settings["models_dir"],
                               "--profiles", str(profiles_path), "--llama-reference", str(binary), "--out", str(run)]
                    if args.num_samples:
                        command += ["--num-samples", str(args.num_samples)]
                    record(directory, {"status": "running", "run": str(run), "gpu": gpu})
                    if execution_identity(binary)[0] != identity:
                        raise ValueError("executable/backend identity changed before dispatch")
                    code = supervisor.execute(command, directory / f"generation-{number:04d}.log", gpu, args.job_timeout)
                    if code:
                        raise RuntimeError(f"generation process failed ({code}); partial evidence: {run}")
                    completion, hashes = verify_generation(directory, run, model, source, mode, gpu)
                    state = {"status": completion["status"], "run": str(run), "gpu": gpu,
                             "cohort": completion["cohort"], "artifacts": hashes}
                record(directory, state)
                if not completion["rows"]:
                    if any(row.get("status") == "unattempted_startup_failure" for row in read_records(run / "failures.jsonl")):
                        unavailable.add((model, mode))
                elif args.upload:
                    atomic_json(directory / "upload-status.json", {"status": "queued", "run": str(run)})
                    uploads.append(upload_pool.submit(upload, directory, run, model, subset, mode))
            except Exception as error:
                status = "invalid_artifacts" if state.get("status") in ("complete", "complete_with_failures", "invalid_artifacts") else "failed"
                record(directory, {"status": "interrupted" if stopped.is_set() else status, "error": str(error),
                                   "gpu": gpu, "run": str(run) if run else None})
            finally:
                work.task_done()

    try:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
        for future in uploads:
            future.result()
    finally:
        # Setting the event also covers unexpected errors outside a worker.
        failed_or_cancelled = stopped.is_set() or sys.exc_info()[0] is not None
        if failed_or_cancelled:
            stopped.set()
        pool.shutdown(wait=True, cancel_futures=failed_or_cancelled)
        upload_pool.shutdown(wait=True, cancel_futures=failed_or_cancelled)
    complete = not stopped.is_set()
    expected_jobs = {str((out / "artifacts" / model / subset).relative_to(out)) for model, subset, _, _ in jobs}
    observed_jobs = {str(p.relative_to(out)) for p in (out / "artifacts").glob("*/*") if p.is_dir()}
    if observed_jobs != expected_jobs:
        complete = False
    for model, subset, source, mode in jobs:
        directory = out / "artifacts" / model / subset
        key = str(directory.relative_to(out))
        try:
            state = json.loads((directory / "job.json").read_text())
            if state.get("status") not in ("complete", "complete_with_failures"):
                raise ValueError(state.get("error", state.get("status", "missing job state")))
            run = Path(state["run"])
            completion, _ = verify_generation(directory, run, model, source, mode, state["gpu"], state)
            if args.upload:
                if not completion["rows"]:
                    raise ValueError("no eligible references to publish; cohort audit retained")
                receipt = validate_receipt(run, model, subset, args.size)
                upload_state = json.loads((directory / "upload-status.json").read_text())
                if upload_state != {"status": "verified", "run": str(run), "receipt": receipt}:
                    raise ValueError("missing or stale upload verification")
            results[key] = state
            complete = complete and state["status"] == "complete"
        except Exception as error:
            complete = False
            previous = results.get(key, {})
            status = previous.get("status", "failed")
            if status in ("complete", "complete_with_failures", "running"):
                status = "failed"
            results[key] = {**previous, "status": "interrupted" if stopped.is_set() else status, "error": str(error)}
    status = "interrupted" if stopped.is_set() else ("complete" if complete else "complete_with_failures")
    atomic_json(out / "status.json", {"status": status, "jobs": results,
                "unexpected_jobs": sorted(observed_jobs - expected_jobs), "missing_jobs": sorted(expected_jobs - observed_jobs)})
    return 130 if stopped.is_set() else (0 if complete else 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, choices=[100, 500], required=True)
    p.add_argument("--gpus", required=True, help="comma-separated GPU ordinals/UUIDs; one worker per GPU")
    p.add_argument("--models", nargs="+")
    p.add_argument("--sources", nargs="+")
    p.add_argument("--modes", nargs="+", choices=["instruct", "thinking"], default=["instruct", "thinking"])
    p.add_argument("--hardware", choices=["pro6000", "h100"], default="pro6000")
    p.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    p.add_argument("--profiles", type=Path, required=True, help="explicit pilot100 or collect500 profile JSON from profiles/")
    p.add_argument("--llama-reference", type=Path, default=COMPANY.parents[1] / "build/bin/llama-reference")
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, help="diagnostics only; requires --no-upload")
    p.add_argument("--job-timeout", type=float, default=172800, help="whole generation job deadline, including downloads, in seconds")
    p.add_argument("--upload-timeout", type=float, default=3600, help="deadline for each of three upload attempts, in seconds")
    p.add_argument("--kill-grace", type=float, default=10, help="seconds before killing remaining child process groups")
    p.add_argument("--resume", action="store_true", help="verify completed jobs; rerun interrupted whole jobs in new directories")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if args.num_samples is not None and (args.num_samples <= 0 or args.upload):
        p.error("--num-samples must be positive and requires --no-upload")
    try:
        raise SystemExit(run_campaign(args))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
