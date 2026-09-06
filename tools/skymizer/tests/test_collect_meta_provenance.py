import argparse
import json
import importlib
import subprocess

import pytest

import cli.collect_kld as collect_kld
import cli.collect_llm_kld as collect_llm_kld
def _provenance_module():
    try:
        return importlib.import_module("lib.collect_meta_provenance")
    except ModuleNotFoundError:
        pytest.fail("collect_meta_provenance module is missing")


def _completed(cmd, stdout):
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


def test_build_collect_provenance_records_git_commit(monkeypatch):
    cmp = _provenance_module()
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "git":
            return _completed(cmd, "abc123\n")
        if cmd[0] == "nvidia-smi":
            return _completed(cmd, "NVIDIA GeForce RTX 3090\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cmp.subprocess, "run", fake_run)

    meta = cmp.build_collect_provenance()

    assert meta["source_checkout_commit"] == "abc123"
    assert meta["llama_cpp_build_commit"] == "unknown"
    assert calls[0] == ["git", "-C", str(cmp.REPO_ROOT), "rev-parse", "HEAD"]


def test_build_collect_provenance_uses_unknown_commit_on_failure(monkeypatch):
    cmp = _provenance_module()

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "git":
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "nvidia-smi":
            return _completed(cmd, "NVIDIA GeForce RTX 3090\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cmp.subprocess, "run", fake_run)

    assert cmp.build_collect_provenance()["llama_cpp_build_commit"] == "unknown"


def test_build_collect_provenance_records_gpu_from_nvidia_smi(monkeypatch):
    cmp = _provenance_module()
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "git":
            return _completed(cmd, "abc123\n")
        if cmd[0] == "nvidia-smi":
            return _completed(cmd, "NVIDIA GeForce RTX 3090\nNVIDIA A100-SXM4-80GB\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cmp.subprocess, "run", fake_run)

    meta = cmp.build_collect_provenance()

    assert meta["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert calls[1] == ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]


def test_build_collect_provenance_uses_unknown_gpu_on_failure(monkeypatch):
    cmp = _provenance_module()

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "git":
            return _completed(cmd, "abc123\n")
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cmp.subprocess, "run", fake_run)

    assert cmp.build_collect_provenance()["gpu_name"] == "unknown"


def _vlm_kld_args(tmp_path):
    return argparse.Namespace(
        ref_model=str(tmp_path / "ref.gguf"),
        ref_mmproj=str(tmp_path / "ref.mmproj.gguf"),
        cand_model=str(tmp_path / "cand.gguf"),
        cand_mmproj=str(tmp_path / "cand.mmproj.gguf"),
        dataset="dataset",
        subset="subset",
        split="train",
        sort_by="num_images",
        num_eval_tokens=-1,
        image_min_tokens=-1,
        image_max_tokens=-1,
        tf_chunk=1,
        n_ctx=32768,
        n_batch=2048,
        n_ubatch=2048,
        n_gpu_layers=99,
        n_threads=8,
        metric_threads=8,
        flash_attn="enabled",
        swa_full=False,
    )


def _llm_kld_args(tmp_path):
    return argparse.Namespace(
        ref_model=str(tmp_path / "ref.gguf"),
        cand_model=str(tmp_path / "cand.gguf"),
        dataset="dataset",
        subset="subset",
        split="train",
        sort_by="",
        sort_desc=False,
        num_eval_tokens=-1,
        tf_chunk=1,
        n_ctx=32768,
        n_batch=2048,
        n_ubatch=2048,
        n_gpu_layers=99,
        n_threads=8,
        metric_threads=8,
        flash_attn="enabled",
        swa_full=False,
    )


@pytest.mark.parametrize(
    ("collector", "args_factory"),
    [
        (collect_kld, _vlm_kld_args),
        (collect_llm_kld, _llm_kld_args),
    ],
)
def test_collectors_ignore_checkout_provenance_but_bind_execution_identity(
    tmp_path, monkeypatch, collector, args_factory,
):
    provenance_keys = {"llama_cpp_build_commit", "gpu_name"}
    assert provenance_keys.isdisjoint(collector.IDENTITY_FIELDS)

    monkeypatch.setattr(
        collector,
        "build_collect_provenance",
        lambda *args: {"llama_cpp_build_commit": "commit-a", "gpu_name": "GPU A", "execution_identity": {"binary_sha256": "a" * 64}},
    )
    first = collector.build_collect_meta(args_factory(tmp_path))
    assert first["llama_cpp_build_commit"] == "commit-a"
    assert first["gpu_name"] == "GPU A"

    out_dir = tmp_path / collector.__name__
    out_dir.mkdir()
    collector.ensure_collect_meta(out_dir, first)

    monkeypatch.setattr(
        collector,
        "build_collect_provenance",
        lambda *args: {"llama_cpp_build_commit": "commit-b", "gpu_name": "GPU B", "execution_identity": {"binary_sha256": "a" * 64}},
    )
    second = collector.build_collect_meta(args_factory(tmp_path))

    # a later shard from a different build/GPU is still the same collection
    assert collector.ensure_collect_meta(out_dir, second) is False
    stored = json.loads((out_dir / "collect_meta.json").read_text())
    assert stored["llama_cpp_build_commit"] == "commit-a"   # never rewritten


@pytest.mark.parametrize(("collector", "args_factory"), [(collect_kld, _vlm_kld_args), (collect_llm_kld, _llm_kld_args)])
def test_append_rejects_different_executed_binary(tmp_path, monkeypatch, collector, args_factory):
    from fakes import EXECUTION_IDENTITY
    monkeypatch.setattr(collector, "build_collect_provenance", lambda *args: {"execution_identity": EXECUTION_IDENTITY})
    out = tmp_path / "out"
    out.mkdir()
    collector.ensure_collect_meta(out, collector.build_collect_meta(args_factory(tmp_path)))
    monkeypatch.setattr(collector, "build_collect_provenance", lambda *args: {
        "execution_identity": {**EXECUTION_IDENTITY, "binary_sha256": "b" * 64}})
    with pytest.raises(SystemExit, match="execution_identity"):
        collector.ensure_collect_meta(out, collector.build_collect_meta(args_factory(tmp_path)))



def test_execution_identity_hashes_actual_binary_and_loaded_libraries(tmp_path, monkeypatch):
    import hashlib
    cmp = _provenance_module()
    binary = tmp_path / "scorer"
    library = tmp_path / "backend.so"
    binary.write_bytes(b"actual binary")
    library.write_bytes(b"actual backend")
    def run(cmd, **kwargs):
        if cmd[0] == str(binary):
            return _completed(cmd, json.dumps({"build": "compiled-source", "contracts": ["reference-vocabulary-v1"],
                                              "loaded_libraries": [str(library)]}))
        return _completed(cmd, "checkout-or-gpu")
    monkeypatch.setattr(cmp.subprocess, "run", run)
    first = cmp.build_collect_provenance(binary)
    identity = first["execution_identity"]
    assert identity["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert first["llama_cpp_build_commit"] == "compiled-source"
    assert first["source_checkout_commit"] == "checkout-or-gpu"
    library.write_bytes(b"changed backend")
    assert cmp.build_collect_provenance(binary)["execution_identity"] != identity
    library.write_bytes(b"actual backend")
    assert cmp.build_collect_provenance(binary)["execution_identity"] == identity


@pytest.mark.parametrize("key", ["GGML_CUDA_CUBLAS_COMPUTE_TYPE", "GGML_CUDA_DISABLE_FUSION",
                                "GGML_CPU_DISABLE_FUSION", "GGML_CUDA_DEVICES", "NVIDIA_TF32_OVERRIDE"])
def test_execution_identity_rejects_changed_arithmetic_environment(tmp_path, monkeypatch, key):
    cmp = _provenance_module()
    binary = tmp_path / "scorer"
    library = tmp_path / "backend.so"
    binary.write_bytes(b"scorer")
    library.write_bytes(b"backend")
    def run(cmd, **kwargs):
        if cmd[0] == str(binary):
            return _completed(cmd, json.dumps({"build": "build", "contracts": ["reference-vocabulary-v1"],
                                              "loaded_libraries": [str(library)]}))
        return _completed(cmd, "GPU-0, uuid-0, driver\nGPU-1, uuid-1, driver\n")
    monkeypatch.setattr(cmp.subprocess, "run", run)
    monkeypatch.delenv(key, raising=False)
    first, _ = cmp.execution_identity(binary)
    assert len(first["gpu"]) == 2
    monkeypatch.setenv(key, "bf16" if key.endswith("COMPUTE_TYPE") else "1")
    second, _ = cmp.execution_identity(binary)
    assert first != second
    assert second["environment"][key] == ("bf16" if key.endswith("COMPUTE_TYPE") else "1")
    with pytest.raises(ValueError, match="identity differs"):
        cmp.require_execution_alignment({"execution_identity": first}, {"execution_identity": second})


def test_execution_environment_keeps_compute_controls_without_api_credentials(monkeypatch):
    cmp = _provenance_module()
    monkeypatch.setenv("LLAMA_ATTN_ROT_DISABLE", "1")
    monkeypatch.setenv("LLAMA_ARG_API_KEY", "test-credential")
    monkeypatch.setenv("HF_TOKEN", "test-hub-credential")
    environment = cmp._execution_environment()
    assert environment["LLAMA_ATTN_ROT_DISABLE"] == "1"
    assert "LLAMA_ARG_API_KEY" not in environment
    assert "HF_TOKEN" not in environment


@pytest.fixture
def campaign_fixture(tmp_path, monkeypatch):
    import cli.run_reference_campaign as campaign
    from pathlib import Path

    binary, library = tmp_path / "binary", tmp_path / "library"
    binary.write_bytes(b"fake binary")
    library.write_bytes(b"fake library")

    def identity(_binary):
        return {"binary_sha256": campaign.sha256_file(binary), "libraries": [{"name": "library", "sha256": campaign.sha256_file(library)}],
                "environment": {}, "gpu": ["fake GPUs"]}, {}

    profiles = {"models": {"qwen3.5-4b": {"model": "model.gguf"}}, "sources": [f"s{i}" for i in range(12)],
                "dataset": {"repo": "test/prepared", "revision": "a" * 40}, "fake_identity": identity(binary)[0],
                "events": str(tmp_path / "process-events.jsonl"), "barrier": 0}
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profiles))
    generator = r'''
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("out", "profiles", "models-dir", "source", "model", "mode", "gpu", "size", "num-samples"):
    p.add_argument("--" + name)
a, _ = p.parse_known_args()
profile = json.loads(Path(a.profiles).read_text())
run = Path(a.out)
run.mkdir(parents=True)
events = Path(profile["events"])
def event(phase):
    with events.open("a") as f:
        f.write(json.dumps({"phase": phase, "source": a.source, "mode": a.mode, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.monotonic()}) + "\n")
event("start")
if profile.get("barrier"):
    markers = events.parent / "markers"
    markers.mkdir(exist_ok=True)
    (markers / a.gpu).touch()
    deadline = time.monotonic() + 5
    while len(list(markers.iterdir())) < profile["barrier"]:
        if time.monotonic() > deadline:
            raise RuntimeError("GPU workers did not run concurrently")
        time.sleep(.02)
if a.source == profile.get("fail"):
    event("failed")
    sys.exit(7)
if a.source == profile.get("hang"):
    time.sleep(60)
time.sleep(.03)
count = int(a.num_samples or a.size)
ids = [a.source + "-" + str(i) for i in range(count)]
startup = a.mode == profile.get("startup_mode")
cohort = {"requested": count, "eligible": 0 if startup else count, "excluded": 0, "failed": count if startup else 0,
          "requested_ids": ids, "eligible_ids": [] if startup else ids, "excluded_ids": [], "failed_ids": ids if startup else []}
status = "complete_with_failures" if startup else "complete"
metadata = {"cohort": cohort, "dataset_source": {**profile["dataset"], "path": profile["dataset"]["repo"],
            "subset": a.source + "-subsample-" + a.size, "split": "train", "num_rows": count},
            "requested_enable_thinking": a.mode == "thinking", "model_path": str(Path(a.models_dir) / "model.gguf")}
if not startup:
    metadata["execution_identity"] = {**profile["fake_identity"], "environment": {"CUDA_VISIBLE_DEVICES": a.gpu}}
for name, value in (("complete.json", {"status": status, "rows": cohort["eligible"], "cohort": cohort}),
                    ("metadata.json", metadata), ("run_state.json", {"status": status, "cohort": cohort}), ("run_start.json", {})):
    (run / name).write_text(json.dumps(value))
(run / "excluded.jsonl").write_text("")
(run / "native-attempts.json").write_text("[]")
(run / "failures.jsonl").write_text("".join(json.dumps({"id": i, "status": "unattempted_startup_failure"}) + "\n" for i in ids) if startup else "")
if not startup:
    dataset = run / "dataset"
    dataset.mkdir()
    (dataset / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data.arrow"}]}))
    (dataset / "dataset_info.json").write_text("{}")
    (dataset / "data.arrow").write_bytes(b"fake dataset shard")
event("end")
'''
    uploader = r'''
import argparse, hashlib, json, time
from pathlib import Path
p = argparse.ArgumentParser()
for key in ("run", "model", "mode", "profiles"):
    p.add_argument("--" + key)
a = p.parse_args()
run = Path(a.run)
profile = json.loads(Path(a.profiles).read_text())
if profile.get("upload_hang"):
    Path(profile["events"]).with_suffix(".upload-started").touch()
    time.sleep(60)
if profile.get("upload_fail"):
    raise SystemExit(7)
metadata = json.loads((run / "metadata.json").read_text())
completion = json.loads((run / "complete.json").read_text())
export = run / "upload"
export.mkdir(exist_ok=True)
parquet = export / "train-00000-of-00001.parquet"
parquet.write_bytes(b"fake parquet")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
subset = metadata["dataset_source"]["subset"] + ("-think" if a.mode == "thinking" else "-ins")
manifest = {"repo": "elichen-skymizer/" + a.model + "-pivot", "subset": subset, "rows": completion["rows"], "split": "train",
            "cohort": completion["cohort"], "metadata_sha256": sha(run / "metadata.json"), "parquet_sha256": sha(parquet),
            "audit_sha256": {name: sha(run / name) for name in ("metadata.json", "complete.json", "run_start.json", "excluded.jsonl", "failures.jsonl", "native-attempts.json")}}
(export / "manifest.json").write_text(json.dumps(manifest))
receipt = {**manifest, "status": "verified", "commit": "b" * 40}
(export / "receipt.json").write_text(json.dumps(receipt))
'''

    def snapshot(out, path):
        archived = out / "scripts/skymizer"
        if not archived.exists():
            (archived / "cli").mkdir(parents=True)
            (archived / "scripts").mkdir()
            (archived / "scripts/reference_model_profiles.json").write_text(path.read_text())
            (archived / "cli/generate_model_reference.py").write_text(generator)
            (archived / "cli/upload_reference.py").write_text(uploader)
        return archived

    monkeypatch.setattr(campaign, "snapshot", snapshot)
    monkeypatch.setattr(campaign, "execution_identity", identity)
    args = argparse.Namespace(out=tmp_path / "campaign", profiles=profile_path, models=["qwen3.5-4b"], sources=["s0", "s1"],
        modes=["instruct"], gpus=["0"], size=100, hardware="pro6000", num_samples=1, models_dir=tmp_path / "models",
        llama_reference=binary, upload=False, resume=False, dry_run=False, job_timeout=10, upload_timeout=10, kill_grace=.1)

    def configure(**values):
        profiles.update(values)
        profile_path.write_text(json.dumps(profiles))
        archived = args.out / "scripts/skymizer/scripts/reference_model_profiles.json"
        if archived.exists():
            archived.write_text(json.dumps(profiles))

    def events():
        path = Path(profiles["events"])
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return campaign, args, configure, events, library


@pytest.mark.parametrize("workers", [4, 6])
def test_campaign_shared_queue_uses_every_gpu_and_advances_after_failure(campaign_fixture, workers):
    campaign, args, configure, events, _ = campaign_fixture
    args.gpus = [str(i) for i in range(workers)]
    args.sources = [f"s{i}" for i in range(workers * 2)]
    configure(barrier=workers, fail="s0")
    assert campaign.run_campaign(args) == 2
    starts = [e for e in events() if e["phase"] == "start"]
    assert {e["gpu"] for e in starts} == set(args.gpus)
    assert sorted(e["source"] for e in starts) == sorted(args.sources)
    status = json.loads((args.out / "status.json").read_text())
    assert len(status["jobs"]) == workers * 2
    assert sum(s["status"] == "complete" for s in status["jobs"].values()) == workers * 2 - 1
    assert any(e["phase"] == "end" and e["source"] != "s0" for e in events())


def test_campaign_resume_keeps_completed_jobs_and_detects_tampering(campaign_fixture):
    campaign, args, _, events, _ = campaign_fixture
    assert campaign.run_campaign(args) == 0
    count = len(events())
    args.resume = True
    assert campaign.run_campaign(args) == 0
    assert len(events()) == count
    shard = next(args.out.glob("artifacts/*/s0-*/attempt-*/dataset/data.arrow"))
    shard.write_bytes(b"changed")
    assert campaign.run_campaign(args) == 2
    assert len(events()) == count
    assert campaign.run_campaign(args) == 2
    assert len(events()) == count


def test_campaign_corrupt_job_state_is_preserved_and_other_job_runs(campaign_fixture):
    campaign, args, _, events, _ = campaign_fixture
    assert campaign.run_campaign(args) == 0
    first = next(args.out.glob("artifacts/*/s0-*/job.json"))
    second = next(args.out.glob("artifacts/*/s1-*/job.json"))
    first.write_text("{torn")
    second.write_text(json.dumps({"status": "running"}))
    args.resume = True
    assert campaign.run_campaign(args) == 2
    assert len(list(first.parent.glob("job-corrupt-*.json"))) == 1
    assert len([e for e in events() if e["source"] == "s1" and e["phase"] == "start"]) == 2
    assert len(list(second.parent.glob("attempt-*"))) == 2


def test_campaign_freezes_library_identity_and_models_directory(campaign_fixture):
    campaign, args, _, _, library = campaign_fixture
    args.dry_run = True
    assert campaign.run_campaign(args) == 0
    args.resume = True
    library.write_bytes(b"replaced backend")
    with pytest.raises(ValueError, match="frozen campaign plan"):
        campaign.run_campaign(args)
    library.write_bytes(b"fake library")
    args.models_dir = args.models_dir / "different"
    with pytest.raises(ValueError, match="frozen campaign plan"):
        campaign.run_campaign(args)


def test_campaign_duplicate_jobs_and_second_owner_are_rejected(campaign_fixture):
    campaign, args, _, _, _ = campaign_fixture
    with campaign.campaign_owner(args.out, False):
        with pytest.raises(ValueError, match="active owner"):
            campaign.run_campaign(args)
    args.resume = True
    args.sources = ["s0", "s0"]
    with pytest.raises(ValueError, match="unique"):
        campaign.run_campaign(args)


def test_campaign_timeout_advances_queue_and_startup_cache_is_mode_specific(campaign_fixture):
    campaign, args, configure, events, _ = campaign_fixture
    args.job_timeout = .3
    configure(hang="s0")
    assert campaign.run_campaign(args) == 2
    assert any(e["phase"] == "end" and e["source"] == "s1" for e in events())
    args.out = args.out.with_name("mode-cache")
    args.job_timeout = 10
    args.modes = ["instruct", "thinking"]
    configure(hang=None, startup_mode="instruct")
    assert campaign.run_campaign(args) == 2
    status = json.loads((args.out / "status.json").read_text())["jobs"]
    assert status["artifacts/qwen3.5-4b/s1-subsample-100-ins"]["status"] == "skipped_model_startup_failure"
    assert status["artifacts/qwen3.5-4b/s0-subsample-100-think"]["status"] == "complete"
    assert status["artifacts/qwen3.5-4b/s1-subsample-100-think"]["status"] == "complete"


def test_campaign_upload_resume_retains_logs_and_rejects_stale_success(campaign_fixture):
    campaign, args, configure, events, _ = campaign_fixture
    args.upload, args.num_samples = True, None
    args.sources = ["s0"]
    assert campaign.run_campaign(args) == 0
    directory = next(args.out.glob("artifacts/*/s0-*"))
    original_log = (directory / "upload-0001.log").read_bytes()
    args.resume = True
    assert campaign.run_campaign(args) == 0
    assert (directory / "upload-0001.log").read_bytes() == original_log
    assert (directory / "upload-0002.log").exists()
    count = len(events())
    configure(upload_fail=True)
    assert campaign.run_campaign(args) == 2
    assert len(list(directory.glob("upload-*.log"))) == 5
    assert len(events()) == count
    assert json.loads((directory / "upload-status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("signum", [2, 15])
@pytest.mark.parametrize("phase", ["generation", "upload"])
def test_campaign_signals_cancel_generation_and_upload(campaign_fixture, signum, phase):
    import os
    import threading
    import time
    from pathlib import Path
    campaign, args, configure, events, _ = campaign_fixture
    args.sources = ["s0"]
    args.upload, args.num_samples = phase == "upload", None if phase == "upload" else 1
    configure(hang="s0" if phase == "generation" else None, upload_hang=phase == "upload")
    done = threading.Event()

    def interrupt():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not done.is_set():
            ready = bool(events()) if phase == "generation" else (args.out.parent / "process-events.upload-started").exists()
            if ready:
                os.kill(os.getpid(), signum)
                return
            time.sleep(.02)

    thread = threading.Thread(target=interrupt)
    thread.start()
    try:
        started = time.monotonic()
        assert campaign.run_campaign(args) == 130
        assert time.monotonic() - started < 5
        assert json.loads((args.out / "status.json").read_text())["status"] == "interrupted"
    finally:
        done.set()
        thread.join()


def test_campaign_supervisor_kills_descendant_after_wrapper_exit(tmp_path):
    import os
    import sys
    import threading
    import time
    from pathlib import Path
    import cli.run_reference_campaign as campaign
    child_pid = tmp_path / "child.pid"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("import subprocess, sys, time\nfrom pathlib import Path\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(p.pid))\ntime.sleep(.2)\n")
    supervisor = campaign.ProcessSupervisor(threading.Event(), grace=.1)
    assert supervisor.execute([sys.executable, str(wrapper)], tmp_path / "wrapper.log", timeout=3) == 0
    pid = int(child_pid.read_text())
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("descendant survived process-group cleanup")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_campaign_snapshot_is_atomic_and_verifies_full_source_provenance(tmp_path, monkeypatch):
    from pathlib import Path
    import cli.run_reference_campaign as campaign
    root = tmp_path / "repo"
    skymizer = root / "tools/skymizer"
    for name in ("cli", "lib", "scripts"):
        (skymizer / name).mkdir(parents=True)
        (skymizer / name / "sample.py").write_text("pass\n")
    profiles = skymizer / "scripts/reference_model_profiles.json"
    profiles.write_text("{}")
    (root / "new.c").write_text("untracked source")
    out = tmp_path / "campaign"
    out.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("diff --git a/src/llama.cpp b/src/llama.cpp\n" if command[:2] == ["git", "diff"] else "captured\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign, "SKYMIZER", skymizer)
    monkeypatch.setattr(campaign.subprocess, "run", run)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: b"new.c\0")
    directory = campaign.snapshot(out, profiles)
    packages = json.loads((directory.parent / "provenance/python-packages.json").read_text())
    assert any(p["name"].lower() == "pytest" for p in packages)
    assert ["git", "diff", "--binary", "HEAD"] in calls
    assert json.loads((out / "scripts/provenance/untracked-sha256.json").read_text())["new.c"] == campaign.sha256_file(root / "new.c")
    assert not list(out.glob(".scripts-*"))
    assert campaign.snapshot(out, profiles) == directory
    (directory / "cli/sample.py").write_text("tampered")
    with pytest.raises(ValueError, match="archived scripts"):
        campaign.snapshot(out, profiles)
    other = tmp_path / "failed-snapshot"
    other.mkdir()
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("failed capture")))
    with pytest.raises(OSError, match="failed capture"):
        campaign.snapshot(other, profiles)
    assert not (other / "scripts").exists()
    assert not list(other.glob(".scripts-*"))


def test_campaign_reconciles_missing_dataset_and_unexpected_job(campaign_fixture):
    campaign, args, _, events, _ = campaign_fixture
    assert campaign.run_campaign(args) == 0
    count = len(events())
    state = next(args.out.glob("artifacts/*/s0-*/attempt-*/dataset/state.json"))
    state.unlink()
    unexpected = args.out / "artifacts/unplanned/source-ins"
    unexpected.mkdir(parents=True)
    args.resume = True
    assert campaign.run_campaign(args) == 2
    assert len(events()) == count
    summary = json.loads((args.out / "status.json").read_text())
    assert summary["unexpected_jobs"] == ["artifacts/unplanned/source-ins"]
    assert "dataset is missing" in summary["jobs"]["artifacts/qwen3.5-4b/s0-subsample-100-ins"]["error"]


def test_campaign_receipt_is_bound_to_job_and_exported_bytes(campaign_fixture):
    campaign, args, _, _, _ = campaign_fixture
    args.sources = ["s0"]
    args.upload, args.num_samples = True, None
    assert campaign.run_campaign(args) == 0
    run = next(args.out.glob("artifacts/*/s0-*/attempt-*"))
    receipt = run / "upload/receipt.json"
    original = receipt.read_text()
    value = json.loads(original)
    value["subset"] = "s1-subsample-100-ins"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not describe this job"):
        campaign.validate_receipt(run, "qwen3.5-4b", "s0-subsample-100-ins", 100)
    receipt.write_text(original)
    (run / "upload/train-00000-of-00001.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        campaign.validate_receipt(run, "qwen3.5-4b", "s0-subsample-100-ins", 100)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_campaign_requires_finite_positive_deadlines(campaign_fixture, value):
    campaign, args, _, _, _ = campaign_fixture
    args.job_timeout = value
    with pytest.raises(ValueError, match="must be positive"):
        campaign.run_campaign(args)


@pytest.mark.parametrize("corruption", ["missing_map", "missing_entry", "extra_entry", "wrong_receipt", "changed_audit", "changed_both_maps"])
def test_campaign_receipt_requires_exact_full_audit_hashes(campaign_fixture, corruption):
    campaign, args, _, _, _ = campaign_fixture
    args.sources = ["s0"]
    args.upload, args.num_samples = True, None
    assert campaign.run_campaign(args) == 0
    run = next(args.out.glob("artifacts/*/s0-*/attempt-*"))
    manifest_path, receipt_path = run / "upload/manifest.json", run / "upload/receipt.json"
    manifest, receipt = json.loads(manifest_path.read_text()), json.loads(receipt_path.read_text())
    if corruption == "missing_map":
        manifest.pop("audit_sha256")
        receipt.pop("audit_sha256")
    elif corruption == "missing_entry":
        manifest["audit_sha256"].pop("run_start.json")
        receipt["audit_sha256"].pop("run_start.json")
    elif corruption == "extra_entry":
        manifest["audit_sha256"]["other.json"] = "f" * 64
        receipt["audit_sha256"]["other.json"] = "f" * 64
    elif corruption == "wrong_receipt":
        receipt["audit_sha256"]["failures.jsonl"] = "f" * 64
    elif corruption == "changed_audit":
        (run / "native-attempts.json").write_text("[{}]")
    elif corruption == "changed_both_maps":
        manifest["audit_sha256"]["excluded.jsonl"] = "f" * 64
        receipt["audit_sha256"]["excluded.jsonl"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="audit hashes"):
        campaign.validate_receipt(run, "qwen3.5-4b", "s0-subsample-100-ins", 100)


def test_campaign_nonempty_generation_requires_native_attempt_audit(campaign_fixture):
    campaign, args, _, _, _ = campaign_fixture
    args.sources = ["s0"]
    assert campaign.run_campaign(args) == 0
    run = next(args.out.glob("artifacts/*/s0-*/attempt-*"))
    (run / "native-attempts.json").unlink()
    with pytest.raises(ValueError, match="missing native-attempts.json"):
        campaign.artifact_hashes(run)
