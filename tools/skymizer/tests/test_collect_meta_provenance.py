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

    assert meta["llama_cpp_build_commit"] == "abc123"
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
        image_resize_backend="pillow",
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
def test_collectors_record_provenance_but_shard_identity_ignores_it(
    tmp_path, monkeypatch, collector, args_factory,
):
    provenance_keys = {"llama_cpp_build_commit", "gpu_name"}
    assert provenance_keys.isdisjoint(collector.IDENTITY_FIELDS)

    monkeypatch.setattr(
        collector,
        "build_collect_provenance",
        lambda: {"llama_cpp_build_commit": "commit-a", "gpu_name": "GPU A"},
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
        lambda: {"llama_cpp_build_commit": "commit-b", "gpu_name": "GPU B"},
    )
    second = collector.build_collect_meta(args_factory(tmp_path))

    # a later shard from a different build/GPU is still the same collection
    assert collector.ensure_collect_meta(out_dir, second) is False
    stored = json.loads((out_dir / "collect_meta.json").read_text())
    assert stored["llama_cpp_build_commit"] == "commit-a"   # never rewritten
