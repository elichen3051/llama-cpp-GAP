"""Collection checks and stage commands for the per-article contract."""

import copy
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from article_fakes import (
    CHECKS, GAP, MODELS, REPO, RUNTIME, VOCABULARY, offline_env,
    prepared_articles, require_cli, run_cli, runtime_flags, sha256, write_executable,
)
from fakes import EXECUTION_IDENTITY, make_records, write_vlmk
from lib.kld_metrics_io import KLD_RECORD_DT, convert_kld_bin_to_npz, load_kld_metrics


def write_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_idx", "item_id", "n_prefill", "n_answer", "n_eval", "vocab", "metrics_bytes", "wall_s", "status"])
        writer.writeheader()
        writer.writerows(rows)


def synthetic_collection(root, receipt, rows):
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True)
    manifest = []
    paths = []
    for i, (article, row) in enumerate(zip(receipt["articles"], rows)):
        count = article["n_targets"]
        records = make_records(npos=count, vocab=VOCABULARY["size"], seed=i)
        records["target"] = row["input_ids"][1:]
        records["kld"] = 0
        records["reversed_kld"] = 0
        records["nll_ref"][0] = 0.0
        records["entropy_ref"][0] = 0.0
        records["nll_cand"] = records["nll_ref"]
        records["entropy_cand"] = records["entropy_ref"]
        records["argmax_cand"] = records["argmax_ref"]
        path = metrics_dir / f"{i:03d}_{article['id']}.npz"
        binary = root / f"fixture-{i}.bin"
        write_vlmk(binary, records, vocab=VOCABULARY["size"], n_prefill=1, n_past_actual=1)
        convert_kld_bin_to_npz(binary, path)
        binary.unlink()
        paths.append(path)
        manifest.append({"row_idx": i, "item_id": article["id"], "n_prefill": 1, "n_answer": count, "n_eval": count,
                         "vocab": VOCABULARY["size"], "metrics_bytes": path.stat().st_size, "wall_s": "0.01", "status": "OK"})
    write_manifest(root / "manifest.csv", manifest)
    meta = {
        **RUNTIME, "kind": "llm_kld_metrics", "perplexity_window": False, "num_eval_tokens": -1,
        "flash_attn": "enabled", "swa_full": False, "ref_model_fingerprint": "sha256:" + "c" * 64,
        "execution_identity": copy.deepcopy(EXECUTION_IDENTITY),
    }
    (root / "collect_meta.json").write_text(json.dumps(meta))
    return {"root": root, "paths": paths, "manifest": manifest, "meta": meta}


def verification_inputs(tmp_path):
    require_cli("verify_article_collection")
    prepared, receipt, rows, _ = prepared_articles(tmp_path)
    candidate = synthetic_collection(tmp_path / "candidate" / "llm-kld", receipt, rows)
    anchor = synthetic_collection(tmp_path / "anchor" / "llm-kld", receipt, rows)
    return prepared / "corpus_articles.json", candidate, anchor


def edit_npz(path, key, value, index=None):
    with np.load(path) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    if index is None:
        arrays[key] = np.asarray(value, dtype=arrays[key].dtype)
    else:
        arrays[key][index] = value
    np.savez(path, **arrays)


def read_checks(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["check", "passed", "detail"]
        rows = list(reader)
    assert [row["check"] for row in rows] == list(CHECKS)
    for row in rows:
        assert row["passed"] in ("true", "false")
        assert row["detail"] is not None
    return rows


def verify(tmp_path, articles, candidate, anchor=None, zero=False, expected=(), exact=True, checks_name="verify-checks.csv"):
    checks = tmp_path / checks_name
    args = ["--articles", articles, "--collection", candidate["root"], *runtime_flags(), "--checks-out", checks]
    if anchor is not None:
        args += ["--parity-with", anchor["root"]]
    if zero:
        args += ["--expect-zero-kld"]
    result = run_cli("verify_article_collection", args, tmp_path)
    assert checks.is_file(), result.stderr
    assert result.returncode == (1 if expected else 0), result.stderr
    rows = read_checks(checks)
    failed = {row["check"] for row in rows if row["passed"] == "false"}
    if exact:
        assert failed == set(expected), rows
    else:
        assert set(expected) <= failed, rows
    assert all(row["detail"].strip() for row in rows if row["check"] in failed)
    return rows


@pytest.mark.parametrize("requested", [False, True])
def test_good_collection_passes_all_checks(tmp_path, requested):
    articles, candidate, anchor = verification_inputs(tmp_path)
    rows = verify(tmp_path, articles, candidate, anchor if requested else None, zero=requested)
    if not requested:
        assert {row["check"]: row["detail"] for row in rows if row["check"] in ("reference_parity", "zero_kld")} == {
            "reference_parity": "not requested", "zero_kld": "not requested",
        }


@pytest.mark.parametrize("mutation,check", [
    ("missing-npz", "metrics_files"), ("extra-file", "metrics_files"), ("wrong-n-eval", "lengths"),
    ("nonfinite", "finite"), ("wrong-target", "targets"), ("wrong-runtime", "runtime"),
    ("reference-column", "reference_parity"), ("nonzero-kld", "zero_kld"),
])
def test_collection_fault_fails_exactly_its_check(tmp_path, mutation, check):
    articles, candidate, anchor = verification_inputs(tmp_path)
    path = candidate["paths"][0]
    if mutation == "missing-npz":
        path.unlink()
    elif mutation == "extra-file":
        (path.parent / "unexpected.txt").write_text("extra")
    elif mutation == "wrong-n-eval":
        candidate["manifest"][0]["n_eval"] += 1
        write_manifest(candidate["root"] / "manifest.csv", candidate["manifest"])
    elif mutation == "nonfinite":
        edit_npz(path, "entropy_cand", np.nan, 0)
    elif mutation == "wrong-target":
        edit_npz(path, "target", 511, 0)
    elif mutation == "wrong-runtime":
        candidate["meta"]["n_ubatch"] = 2048
        (candidate["root"] / "collect_meta.json").write_text(json.dumps(candidate["meta"]))
    elif mutation == "reference-column":
        edit_npz(path, "nll_ref", 0.25, 0)
    else:
        edit_npz(path, "kld", np.nextafter(np.float32(0), np.float32(1)), 0)
    verify(tmp_path, articles, candidate, anchor if mutation in ("missing-npz", "reference-column") else None,
           zero=mutation in ("missing-npz", "nonzero-kld"), expected=[check])


def test_missing_npz_does_not_skip_checks_for_present_articles(tmp_path):
    articles, candidate, anchor = verification_inputs(tmp_path)
    candidate["paths"][0].unlink()
    path = candidate["paths"][1]
    edit_npz(path, "n_prefill", 2)
    edit_npz(path, "entropy_cand", np.nan, 0)
    edit_npz(path, "target", 511, 0)
    edit_npz(anchor["paths"][1], "entropy_ref", -0.0, 0)
    edit_npz(path, "kld", 0.5, 0)
    verify(tmp_path, articles, candidate, anchor, zero=True,
           expected=["metrics_files", "lengths", "finite", "targets", "reference_parity", "zero_kld"])


def damage_npz(path, damage):
    if damage == "truncated":
        payload = path.read_bytes()
        path.write_bytes(payload[:len(payload) // 2])
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "non-zip":
        path.write_bytes(b"not a zip archive\n")
    else:
        assert damage == "missing-array"
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files if key != "ear"}
        np.savez(path, **arrays)


@pytest.mark.parametrize("damage", ["truncated", "empty", "non-zip", "missing-array"])
@pytest.mark.parametrize("requested", [False, True])
def test_unreadable_candidate_npz_fails_all_requested_content_checks(tmp_path, damage, requested):
    articles, candidate, anchor = verification_inputs(tmp_path)
    path = candidate["paths"][0]
    damage_npz(path, damage)
    expected = ["lengths", "finite", "targets"]
    if requested:
        expected += ["reference_parity", "zero_kld"]
    rows = verify(tmp_path, articles, candidate, anchor if requested else None, zero=requested, expected=expected)
    detail = {row["check"]: row["detail"] for row in rows}
    assert path.name in detail["lengths"]
    if not requested:
        assert detail["reference_parity"] == detail["zero_kld"] == "not requested"


def test_unreadable_anchor_npz_fails_only_reference_parity(tmp_path):
    articles, candidate, anchor = verification_inputs(tmp_path)
    path = anchor["paths"][0]
    original = path.read_bytes()
    for damage in ("non-zip", "truncated", "empty", "missing-array"):
        path.write_bytes(original)
        damage_npz(path, damage)
        verify(tmp_path, articles, candidate, anchor, zero=True, expected=["reference_parity"], checks_name=f"{damage}.csv")


@pytest.mark.parametrize("mutation", ["failed-status", "reordered", "missing-row", "duplicate-row", "wrong-id", "wrong-index"])
def test_manifest_requires_one_ok_row_per_article_in_order(tmp_path, mutation):
    articles, candidate, _ = verification_inputs(tmp_path)
    rows = candidate["manifest"]
    if mutation == "failed-status":
        rows[0]["status"] = "FAIL_EXIT_7"
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "missing-row":
        rows.pop()
    elif mutation == "duplicate-row":
        rows.append(dict(rows[0]))
    else:
        rows[0]["item_id" if mutation == "wrong-id" else "row_idx"] = "wrong" if mutation == "wrong-id" else 9
    write_manifest(candidate["root"] / "manifest.csv", rows)
    checks = tmp_path / "verify-checks.csv"
    result = run_cli("verify_article_collection", ["--articles", articles, "--collection", candidate["root"], *runtime_flags(), "--checks-out", checks], tmp_path)
    assert result.returncode == 1, result.stderr
    assert read_checks(checks)[0]["passed"] == "false"


@pytest.mark.parametrize("field", ["manifest-n_prefill", "manifest-n_answer", "n_prefill", "npos", "vocab", "array-length"])
def test_lengths_validate_manifest_header_and_metric_arrays(tmp_path, field):
    articles, candidate, _ = verification_inputs(tmp_path)
    if field.startswith("manifest-"):
        candidate["manifest"][0][field.removeprefix("manifest-")] += 1
        write_manifest(candidate["root"] / "manifest.csv", candidate["manifest"])
    elif field == "array-length":
        metrics, _ = load_kld_metrics(candidate["paths"][0])
        edit_npz(candidate["paths"][0], "entropy_cand", metrics["entropy_cand"][:-1])
    else:
        _, header = load_kld_metrics(candidate["paths"][0])
        edit_npz(candidate["paths"][0], field, header[field] + 1)
    verify(tmp_path, articles, candidate, expected=["lengths"], exact=field not in ("npos", "array-length"))


@pytest.mark.parametrize("field", [key for key in KLD_RECORD_DT.names if KLD_RECORD_DT[key].kind == "f"])
def test_finite_checks_every_float_metric(tmp_path, field):
    articles, candidate, _ = verification_inputs(tmp_path)
    edit_npz(candidate["paths"][0], field, np.inf, 0)
    verify(tmp_path, articles, candidate, expected=["finite"])


@pytest.mark.parametrize("field,value", [
    *[(key, value + 1) for key, value in RUNTIME.items()],
    ("kind", "vlm_kld_metrics"), ("perplexity_window", True), ("num_eval_tokens", 1), ("flash_attn", "auto"), ("swa_full", True),
])
def test_runtime_checks_every_required_field(tmp_path, field, value):
    articles, candidate, _ = verification_inputs(tmp_path)
    candidate["meta"][field] = value
    (candidate["root"] / "collect_meta.json").write_text(json.dumps(candidate["meta"]))
    verify(tmp_path, articles, candidate, expected=["runtime"])


@pytest.mark.parametrize("field", [
    "nll_ref", "argmax_ref", "entropy_ref", "ref_model_fingerprint", "execution_identity", *RUNTIME,
    "kind", "perplexity_window", "num_eval_tokens", "flash_attn", "swa_full",
])
def test_reference_parity_checks_columns_and_identity_bitwise(tmp_path, field):
    articles, candidate, anchor = verification_inputs(tmp_path)
    if field in ("nll_ref", "entropy_ref"):
        edit_npz(anchor["paths"][0], field, -0.0, 0)
    elif field == "argmax_ref":
        metrics, _ = load_kld_metrics(anchor["paths"][0])
        edit_npz(anchor["paths"][0], field, (int(metrics[field][0]) + 1) % VOCABULARY["size"], 0)
    else:
        original = anchor["meta"][field]
        if field == "execution_identity":
            changed = {**EXECUTION_IDENTITY, "binary_sha256": "d" * 64}
        elif field == "ref_model_fingerprint":
            changed = "sha256:" + "d" * 64
        elif isinstance(original, bool):
            changed = not original
        elif isinstance(original, int):
            changed = original + 1
        else:
            changed = "different"
        anchor["meta"][field] = changed
        (anchor["root"] / "collect_meta.json").write_text(json.dumps(anchor["meta"]))
    verify(tmp_path, articles, candidate, anchor, expected=["reference_parity"])


def test_reference_parity_allows_candidate_columns_to_differ(tmp_path):
    articles, candidate, anchor = verification_inputs(tmp_path)
    edit_npz(candidate["paths"][0], "nll_cand", 1.0, 0)
    edit_npz(candidate["paths"][0], "kld", 0.125, 0)
    verify(tmp_path, articles, candidate, anchor)


def gpu_identity():
    return {
        **copy.deepcopy(EXECUTION_IDENTITY),
        "libraries": [{"name": "libfixture.so", "sha256": "e" * 64}],
        "gpu": [
            "Fixture accelerator A, GPU-11111111-1111-1111-1111-111111111111, 600.10",
            "Fixture accelerator B, GPU-22222222-2222-2222-2222-222222222222, 600.10",
        ],
        "environment": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8"},
        "fixture_key": "must match",
    }


def set_identity(collection, identity):
    collection["meta"]["execution_identity"] = copy.deepcopy(identity)
    (collection["root"] / "collect_meta.json").write_text(json.dumps(collection["meta"]))


def test_reference_parity_ignores_environment_but_binds_every_other_identity_key(tmp_path):
    articles, candidate, anchor = verification_inputs(tmp_path)
    identity = gpu_identity()
    set_identity(candidate, identity)
    changed = copy.deepcopy(identity)
    changed["environment"] = {"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "12", "FIXTURE_FLAG": "different"}
    set_identity(anchor, changed)
    verify(tmp_path, articles, candidate, anchor, checks_name="environment-only.csv")

    mutations = {
        "scheme": "different-scheme", "binary_sha256": "f" * 64,
        "libraries": [{"name": "libfixture.so", "sha256": "f" * 64}],
        "gpu-model": [identity["gpu"][0].replace("accelerator A", "accelerator C"), identity["gpu"][1]],
        "gpu-driver": [identity["gpu"][0].replace("600.10", "600.11"), identity["gpu"][1]],
        "fixture_key": "different",
    }
    for field, value in mutations.items():
        differing = copy.deepcopy(changed)
        differing["gpu" if field.startswith("gpu-") else field] = value
        set_identity(anchor, differing)
        verify(tmp_path, articles, candidate, anchor, expected=["reference_parity"], checks_name=f"{field}.csv")
    for field in ("fixture_key", "environment"):
        differing = copy.deepcopy(changed)
        del differing[field]
        set_identity(anchor, differing)
        verify(tmp_path, articles, candidate, anchor, expected=[] if field == "environment" else ["reference_parity"],
               checks_name=f"missing-{field}.csv")
    set_identity(anchor, changed)
    edit_npz(anchor["paths"][0], "nll_ref", -0.0, 0)
    verify(tmp_path, articles, candidate, anchor, expected=["reference_parity"], checks_name="reference-bits.csv")


@pytest.mark.parametrize("change", ["uuid", "order", "uuid-and-order"])
def test_reference_parity_normalizes_gpu_uuid_and_order_only(tmp_path, change):
    articles, candidate, anchor = verification_inputs(tmp_path)
    identity = gpu_identity()
    identity["gpu"] += ["CPU fixture", "Fixture accelerator, local device, 600.10"]
    changed = copy.deepcopy(identity)
    if "uuid" in change:
        changed["gpu"][0] = changed["gpu"][0].replace("11111111-1111-1111-1111-111111111111", "abcdef01-2345-6789-abcd-0123456789ab")
        changed["gpu"][1] = changed["gpu"][1].replace("22222222-2222-2222-2222-222222222222", "abcdef02-2345-6789-abcd-0123456789ab")
    if "order" in change:
        changed["gpu"].reverse()
    set_identity(candidate, identity)
    set_identity(anchor, changed)
    verify(tmp_path, articles, candidate, anchor, checks_name="normalized.csv")
    for i, entry in enumerate(("different CPU fixture", "Fixture accelerator, other local device, 600.10")):
        differing = copy.deepcopy(changed)
        original = identity["gpu"][i + 2]
        differing["gpu"][differing["gpu"].index(original)] = entry
        set_identity(anchor, differing)
        verify(tmp_path, articles, candidate, anchor, expected=["reference_parity"], checks_name=f"literal-{i}.csv")
    differing = copy.deepcopy(changed)
    differing["gpu"].append(differing["gpu"][0])
    set_identity(anchor, differing)
    verify(tmp_path, articles, candidate, anchor, expected=["reference_parity"], checks_name="gpu-count.csv")


def test_zero_kld_requires_bitwise_equal_nll(tmp_path):
    articles, candidate, _ = verification_inputs(tmp_path)
    edit_npz(candidate["paths"][0], "nll_cand", -0.0, 0)
    verify(tmp_path, articles, candidate, zero=True, expected=["zero_kld"])


@pytest.mark.parametrize("field", ["reversed_kld", "entropy_cand", "argmax_cand"])
def test_zero_kld_checks_reverse_entropy_and_argmax_but_ignores_js(tmp_path, field):
    articles, candidate, _ = verification_inputs(tmp_path)
    path = candidate["paths"][0]
    residue = np.nextafter(np.float32(0), np.float32(1))
    edit_npz(path, "js_kld", residue, 0)
    verify(tmp_path, articles, candidate, zero=True, checks_name="js-residue.csv")
    metrics, _ = load_kld_metrics(path)
    value = {"reversed_kld": residue, "entropy_cand": -0.0,
             "argmax_cand": (int(metrics["argmax_ref"][0]) + 1) % VOCABULARY["size"]}[field]
    edit_npz(path, field, value, 0)
    verify(tmp_path, articles, candidate, zero=True, expected=["zero_kld"])


def test_verifier_evaluates_later_checks_after_failures(tmp_path):
    articles, candidate, _ = verification_inputs(tmp_path)
    candidate["manifest"][0]["status"] = "FAIL_EXIT_7"
    write_manifest(candidate["root"] / "manifest.csv", candidate["manifest"])
    edit_npz(candidate["paths"][0], "entropy_cand", np.nan, 0)
    edit_npz(candidate["paths"][1], "target", 511, 0)
    edit_npz(candidate["paths"][1], "kld", 0.5, 0)
    verify(tmp_path, articles, candidate, zero=True, expected=["manifest", "finite", "targets", "zero_kld"])


@pytest.mark.parametrize("case", ["missing-runtime", "malformed-integer", "unknown-flag"])
def test_verifier_usage_errors_write_no_csv(tmp_path, case):
    require_cli("verify_article_collection")
    checks = tmp_path / "checks.csv"
    flags = runtime_flags()
    if case == "missing-runtime":
        del flags[:2]
    elif case == "malformed-integer":
        flags[1] = "invalid"
    else:
        flags.append("--unknown-flag")
    result = run_cli("verify_article_collection", ["--articles", tmp_path / "articles.json", "--collection", tmp_path / "collection", *flags, "--checks-out", checks], tmp_path)
    assert result.returncode == 2, result.stderr
    assert not checks.exists()


def test_verifier_refuses_existing_checks_output(tmp_path):
    articles, candidate, _ = verification_inputs(tmp_path)
    checks = tmp_path / "checks.csv"
    checks.write_bytes(b"evidence\n")
    result = run_cli("verify_article_collection", ["--articles", articles, "--collection", candidate["root"], *runtime_flags(), "--checks-out", checks], tmp_path)
    assert result.returncode == 2, result.stderr
    assert checks.read_bytes() == b"evidence\n"


def stage_script(name):
    path = GAP / "scripts" / "text-article" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing implementation: scripts/text-article/{name}")
    return path


def stage_env(tmp_path):
    env = offline_env(tmp_path)
    env.update(FORK_REPO=str(REPO), RUNTIME_DIR=str(tmp_path / "runtime"), TEXT_BIN=str(tmp_path / "runtime/bin"), PYTHON=sys.executable, DRY_RUN="1")
    env["TEXT_LIB"] = str(tmp_path / "runtime/lib")
    env.pop("ALLOW_VOCAB_ATTR_MISMATCH", None)
    return env


def run_stage(name, args, tmp_path, env=None):
    return subprocess.run(["bash", str(stage_script(name)), *map(str, args)], cwd=tmp_path,
                          env=env or stage_env(tmp_path), capture_output=True, text=True, timeout=20)


def printed_commands(result, filename):
    commands = []
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if line.startswith("+"):
            words = shlex.split(line[1:])
            if any(Path(word).name == filename for word in words):
                commands.append(words)
    return commands


def assert_flags(command, values):
    for flag, value in values.items():
        assert command.count(flag) == 1, (flag, command)
        assert command[command.index(flag) + 1] == str(value), (flag, command)


@pytest.mark.parametrize("name", ["env.sh", "prepare_corpus.sh", "llm_kld.sh", "run_segment.sh"])
def test_stage_scripts_pass_bash_syntax(tmp_path, name):
    result = subprocess.run(["bash", "-n", str(stage_script(name))], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model", MODELS)
def test_llm_stage_dry_run_has_collection_and_verifier_arguments(tmp_path, model):
    ref, cand, prepared, out, anchor = [tmp_path / name for name in ("ref.gguf", "cand.gguf", "prepared", "candidate", "anchor")]
    result = run_stage("llm_kld.sh", [model, ref, cand, prepared, out, "--allow-vocab-attr-mismatch", "--parity-with", anchor, "--expect-zero-kld"], tmp_path)
    assert result.returncode == 0, result.stderr
    [collect] = printed_commands(result, "collect_llm_kld.py")
    [verify_command] = printed_commands(result, "verify_article_collection.py")
    ubatch = 2048 if model == "gemma-4-31b-it" else 512
    runtime = {"--" + key.replace("_", "-"): value for key, value in {**RUNTIME, "n_ubatch": ubatch}.items()}
    assert_flags(collect, {
        **runtime, "--ref-model": ref, "--cand-model": cand, "--dataset": prepared / "dataset", "--subset": "",
        "--out": out / "llm-kld", "--llama-llm-kld": tmp_path / "runtime/bin/llama-llm-kld",
        "--start": 0, "--end": -1, "--num-eval-tokens": -1,
    })
    assert "--flash-attn" in collect and "--allow-vocab-attr-mismatch" in collect
    assert "--perplexity-window" not in collect and "--swa-full" not in collect
    assert str(out / "llm.log") in collect
    assert_flags(verify_command, {**runtime, "--collection": out / "llm-kld", "--checks-out": out / "verify-checks.csv", "--parity-with": anchor / "llm-kld"})
    assert "--expect-zero-kld" in verify_command
    assert verify_command[verify_command.index("--articles") + 1] in (str(prepared / "corpus_articles.json"), str(out / "llm-kld/corpus_articles.json"))
    assert not out.exists()


def test_prepare_stage_dry_run_arguments(tmp_path):
    corpus, index, ref, out = [tmp_path / name for name in ("corpus.txt", "index.json", "ref.gguf", "prepared")]
    result = run_stage("prepare_corpus.sh", [corpus, "pg-full-rss-article", index, ref, out], tmp_path)
    assert result.returncode == 0, result.stderr
    [command] = printed_commands(result, "prepare_article_corpus.py")
    assert_flags(command, {
        "--corpus": corpus, "--corpus-name": "pg-full-rss-article", "--article-index": index, "--ref-model": ref,
        "--llama-tokenize": tmp_path / "runtime/bin/llama-tokenize", "--llama-llm-kld": tmp_path / "runtime/bin/llama-llm-kld", "--out": out,
    })
    assert not out.exists()


def test_llm_stage_refuses_unknown_model(tmp_path):
    result = run_stage("llm_kld.sh", ["unknown-model", tmp_path / "ref", tmp_path / "cand", tmp_path / "prepared", tmp_path / "candidate"], tmp_path)
    assert result.returncode != 0
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("stage", ["prepare_corpus.sh", "llm_kld.sh"])
def test_stage_refuses_existing_output(tmp_path, stage):
    stage_script(stage)
    candidate = tmp_path / "candidate"
    out = tmp_path / "prepared" if stage == "prepare_corpus.sh" else candidate / "llm-kld"
    out.mkdir(parents=True)
    (out / "evidence").write_bytes(b"preserve")
    args = ([tmp_path / "corpus", "pg-full-rss-article", tmp_path / "index", tmp_path / "ref", out] if stage == "prepare_corpus.sh" else
            ["qwen3.5-4b", tmp_path / "ref", tmp_path / "cand", tmp_path / "prepared", candidate])
    result = run_stage(stage, args, tmp_path)
    assert result.returncode != 0
    assert {path.name: path.read_bytes() for path in out.iterdir()} == {"evidence": b"preserve"}
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not printed_commands(result, "prepare_article_corpus.py")


@pytest.mark.parametrize("model", [*MODELS, "unknown-model"])
def test_stage_env_constants_and_ubatch_for(tmp_path, model):
    script = stage_script("env.sh")
    command = 'source "$1"; printf "%s\\n" "$N_CTX" "$N_BATCH" "$TF_CHUNK" "$N_THREADS" "$METRIC_THREADS" "$N_GPU_LAYERS"; ubatch_for "$2"'
    result = subprocess.run(["bash", "-c", command, "article-test", str(script), model],
                            cwd=tmp_path, env=stage_env(tmp_path), capture_output=True, text=True, timeout=10)
    assert result.stdout.splitlines()[:6] == ["32768", "2048", "2048", "8", "16", "-2"]
    if model == "unknown-model":
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[6:] == ["2048" if model == "gemma-4-31b-it" else "512"]


@pytest.mark.parametrize("initial", ["unset", "other-library", "already-prefixed"])
def test_stage_env_library_path_prepend_is_idempotent(tmp_path, initial):
    env = stage_env(tmp_path)
    library = env["TEXT_LIB"]
    other = str(tmp_path / "other/lib")
    if initial == "unset":
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["LD_LIBRARY_PATH"] = other if initial == "other-library" else library + ":" + other
    command = 'source "$1"; printf "%s\\n" "$LD_LIBRARY_PATH"; source "$1"; printf "%s\\n" "$LD_LIBRARY_PATH"'
    result = subprocess.run(["bash", "-c", command, "article-test", str(stage_script("env.sh"))],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    first, second = result.stdout.splitlines()
    assert first == second
    assert first == (library if initial == "unset" else library + ":" + other)


def test_stage_env_vocab_waiver_for_all_models(tmp_path):
    command = r'''
source "$1"
shift
declare -F needs_vocab_waiver >/dev/null || exit 99
for model in "$@"; do
    if needs_vocab_waiver "$model"; then printf '%s yes\n' "$model"; else printf '%s no\n' "$model"; fi
done
'''
    models = (*MODELS, "unknown-model")
    result = subprocess.run(["bash", "-c", command, "article-test", str(stage_script("env.sh")), *models],
                            cwd=tmp_path, env=stage_env(tmp_path), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [model + (" yes" if model in MODELS[:2] else " no") for model in models]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
def test_llm_stage_vocab_preflight_and_waiver_dry_run(tmp_path, model, explicit):
    env = stage_env(tmp_path)
    binary_dir = Path(env["TEXT_BIN"])
    binary_dir.mkdir(parents=True)
    write_executable(binary_dir / "llama-llm-kld", r'''
from pathlib import Path
(Path(__file__).parent / "unexpected-native-call").touch()
raise SystemExit(91)
''')
    ref, cand, prepared, out = [tmp_path / name for name in ("ref.gguf", "cand.gguf", "prepared", "candidate")]
    args = [model, ref, cand, prepared, out]
    if explicit:
        args.append("--allow-vocab-attr-mismatch")
    result = run_stage("llm_kld.sh", args, tmp_path, env)
    assert result.returncode == 0, result.stderr
    [collect] = printed_commands(result, "collect_llm_kld.py")
    assert collect.count("--allow-vocab-attr-mismatch") == int(explicit or model in MODELS[:2])
    native = [command for command in printed_commands(result, "llama-llm-kld") if "--vocab-identity" in command]
    assert len(native) == 1, result.stderr
    assert_flags(native[0], {"--vocab-identity": ref})
    assert str(binary_dir / "llama-llm-kld") in native[0]
    assert "CUDA_VISIBLE_DEVICES=" in native[0]
    assert not (binary_dir / "unexpected-native-call").exists()
    assert not out.exists()


@pytest.mark.parametrize("field", list(VOCABULARY))
def test_llm_stage_vocab_mismatch_stops_before_collector_or_candidate_writes(tmp_path, field):
    prepared, receipt, _, inputs = prepared_articles(tmp_path)
    env = stage_env(tmp_path)
    env.update(DRY_RUN="0", CUDA_VISIBLE_DEVICES="fixture-visible-device")
    binary_dir = Path(env["TEXT_BIN"])
    binary_dir.mkdir(parents=True)
    fake = write_executable(binary_dir / "llama-llm-kld", r'''
import json, os, sys
from pathlib import Path
base = Path(__file__).parent
with (base / "identity-calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
assert len(sys.argv) == 3 and sys.argv[1] == "--vocab-identity"
print((base / "vocabulary.json").read_text())
''')
    wrapper = write_executable(tmp_path / "python-wrapper", r'''
import os, sys
from pathlib import Path
if any(Path(arg).name == "collect_llm_kld.py" for arg in sys.argv[1:]):
    (Path(__file__).parent / "collector-started").touch()
    sys.exit(77)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    env["PYTHON"] = str(wrapper)
    vocabulary = receipt["protocol"]["vocabulary"]
    changed = vocabulary[field] + 1 if isinstance(vocabulary[field], int) else "f" * 64 if field in ("mapping", "attributes") else "different-scheme"
    (fake.parent / "vocabulary.json").write_text(json.dumps({**vocabulary, field: changed}))
    cand = tmp_path / "cand.gguf"
    cand.write_bytes(b"candidate fixture")
    out = tmp_path / "candidate"
    args = ["qwen3.5-4b", inputs["ref_model"], cand, prepared, out]
    result = run_stage("llm_kld.sh", args, tmp_path, env)
    assert result.returncode != 0
    assert not (tmp_path / "collector-started").exists(), "vocabulary mismatch reached the collector"
    assert not out.exists(), result.stderr
    assert not printed_commands(result, "collect_llm_kld.py")
    calls = [json.loads(line) for line in (fake.parent / "identity-calls.jsonl").read_text().splitlines()]
    assert calls == [{"args": ["--vocab-identity", str(inputs["ref_model"])], "cuda": ""}]

    # Object equality permits different JSON key order and whitespace.
    (fake.parent / "vocabulary.json").write_text(json.dumps(dict(reversed(list(vocabulary.items()))), indent=2) + "\n")
    args[4] = tmp_path / "matching-candidate"
    result = run_stage("llm_kld.sh", args, tmp_path, env)
    assert (tmp_path / "collector-started").exists(), result.stderr


def segment_args(tmp_path, labels, model="qwen3.5-4b"):
    return [tmp_path / "ref.gguf", tmp_path / "prepared", tmp_path / "runs", model, "wikitext-2-test-article",
            *[f"{label}={tmp_path / (str(i) + '.gguf')}" for i, label in enumerate(labels)]]


def candidate_dir(tmp_path, label, model="qwen3.5-4b"):
    return tmp_path / "runs/our-llm-kld-records" / model / "wikitext-2-test-article/candidates" / label


def passing_checks(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["check", "passed", "detail"])
        writer.writerows((check, "true", "fixture") for check in CHECKS)


def parity_checks(candidate, anchor):
    meta = anchor / "llm-kld/collect_meta.json"
    fingerprint = sha256(meta.read_bytes())[:12] if meta.is_file() else "missing"
    return candidate / f"parity-{anchor.name}-{fingerprint}.csv"


def assert_segment_collections(result, directories, anchor):
    stages = printed_commands(result, "llm_kld.sh")
    if stages:
        assert len(stages) == len(directories), stages
        for command, directory in zip(stages, directories):
            assert str(directory) in command
            if directory == anchor:
                assert "--parity-with" not in command
            else:
                assert_flags(command, {"--parity-with": anchor})
    else:
        commands = printed_commands(result, "collect_llm_kld.py")
        assert len(commands) == len(directories), commands
        verifiers = printed_commands(result, "verify_article_collection.py")
        for command, directory in zip(commands, directories):
            assert_flags(command, {"--out": directory / "llm-kld"})
            [check] = [entry for entry in verifiers if entry[entry.index("--collection") + 1] == str(directory / "llm-kld")]
            if directory == anchor:
                assert "--parity-with" not in check
            else:
                assert_flags(check, {"--parity-with": anchor / "llm-kld"})


def test_segment_dry_run_preserves_candidate_order_and_forwards_waiver(tmp_path):
    labels = ["candidate--user--q4", "candidate--company--q8"]
    env = stage_env(tmp_path)
    env["ALLOW_VOCAB_ATTR_MISMATCH"] = "1"
    result = run_stage("run_segment.sh", segment_args(tmp_path, labels), tmp_path, env)
    assert result.returncode == 0, result.stderr
    stages = printed_commands(result, "llm_kld.sh")
    commands = stages or printed_commands(result, "collect_llm_kld.py")
    assert len(commands) == 2
    for i, (label, command) in enumerate(zip(labels, commands)):
        if stages:
            assert str(tmp_path / f"{i}.gguf") in command
            assert str(candidate_dir(tmp_path, label)) in command
        else:
            assert_flags(command, {"--cand-model": tmp_path / f"{i}.gguf", "--out": candidate_dir(tmp_path, label) / "llm-kld"})
        assert "--allow-vocab-attr-mismatch" in command
    directories = [candidate_dir(tmp_path, label) for label in labels]
    assert_segment_collections(result, directories, directories[0])
    assert any("dry-run" in line.lower() and "corpus_articles.json" in line for line in result.stderr.splitlines())
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("model", ["qwen3.5-4b", "gemma-4-31b-it"])
@pytest.mark.parametrize("parity_receipt", ["absent", "passing", "failing"])
def test_segment_prescans_anchor_and_rechecks_resumed_candidates(tmp_path, model, parity_receipt):
    prepared, _, _, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q4", "candidate--user--q8", "candidate--company--q6", "candidate--company--q5"]
    a, b, c, d = [candidate_dir(tmp_path, label, model) for label in labels]
    for directory in (b, c):
        passing_checks(directory / "verify-checks.csv")
        (directory / "llm-kld").mkdir()
        (directory / "llm-kld/corpus_articles.json").write_bytes((prepared / "corpus_articles.json").read_bytes())
    parity = parity_checks(c, b)
    log = parity.with_suffix(".log")
    if parity_receipt != "absent":
        passing_checks(parity)
        if parity_receipt == "failing":
            parity.write_text(parity.read_text().replace("reference_parity,true,", "reference_parity,false,"))
    before = {path: path.read_bytes() for directory in (b, c) for path in directory.rglob("*") if path.is_file()}
    args = segment_args(tmp_path, labels, model)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path)
    if parity_receipt == "failing":
        assert result.returncode != 0, result.stderr
    else:
        assert result.returncode == 0, result.stderr
    verifiers = printed_commands(result, "verify_article_collection.py")
    rechecks = [command for command in verifiers if command[command.index("--collection") + 1] == str(c / "llm-kld")]
    assert not any(command[command.index("--collection") + 1] == str(b / "llm-kld") for command in verifiers)
    if parity_receipt == "absent":
        assert len(rechecks) == 1, result.stderr
        [command] = rechecks
        assert_flags(command, {
            **dict(zip(runtime_flags()[::2], runtime_flags()[1::2])),
            "--n-ubatch": 2048 if model == "gemma-4-31b-it" else 512,
            "--articles": c / "llm-kld/corpus_articles.json", "--collection": c / "llm-kld",
            "--parity-with": b / "llm-kld", "--checks-out": parity,
        })
        assert str(log) in command
    else:
        assert not rechecks
    if parity_receipt == "failing":
        assert not any(str(d) in word for command in printed_commands(result, "llm_kld.sh") + printed_commands(result, "collect_llm_kld.py") for word in command)
    else:
        assert_segment_collections(result, [a, d], b)
    assert {path: path.read_bytes() for directory in (b, c) for path in directory.rglob("*") if path.is_file()} == before
    assert not a.exists() and not d.exists()
    assert not log.exists()


def test_segment_refuses_corpus_name_mismatch_before_collection(tmp_path):
    _, _, _, inputs = prepared_articles(tmp_path)
    args = segment_args(tmp_path, ["candidate--user--q4"])
    args[0] = inputs["ref_model"]
    args[4] = "pg-full-rss-article"
    result = run_stage("run_segment.sh", args, tmp_path)
    assert result.returncode != 0, result.stderr
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not (tmp_path / "runs").exists()


def test_segment_failing_parity_recheck_stops_before_next_resumed_candidate(tmp_path):
    prepared, receipt, rows, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q8", "candidate--company--q6", "candidate--company--q5"]
    directories = [candidate_dir(tmp_path, label) for label in labels]
    collections = []
    for directory in directories:
        collection = synthetic_collection(directory / "llm-kld", receipt, rows)
        (collection["root"] / "corpus_articles.json").write_bytes((prepared / "corpus_articles.json").read_bytes())
        passing_checks(directory / "verify-checks.csv")
        collections.append(collection)
    edit_npz(collections[1]["paths"][0], "nll_ref", -0.0, 0)
    before = [(directory / "verify-checks.csv").read_bytes() for directory in directories]
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode != 0, result.stderr
    anchor, candidate, later = directories
    parity_name = parity_checks(candidate, anchor).stem
    checks = candidate / f"{parity_name}.csv"
    assert checks.is_file(), result.stderr
    assert {row["check"] for row in read_checks(checks) if row["passed"] == "false"} == {"reference_parity"}
    assert (candidate / f"{parity_name}.log").is_file()
    [command] = printed_commands(result, "verify_article_collection.py")
    assert_flags(command, {
        **dict(zip(runtime_flags()[::2], runtime_flags()[1::2])),
        "--articles": candidate / "llm-kld/corpus_articles.json", "--collection": candidate / "llm-kld",
        "--parity-with": anchor / "llm-kld", "--checks-out": checks,
    })
    assert not list(anchor.glob("parity-*"))
    assert not list(later.glob("parity-*"))
    assert [(directory / "verify-checks.csv").read_bytes() for directory in directories] == before
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")


def test_segment_resumes_passing_candidate_and_uses_it_as_parity_anchor(tmp_path):
    stage_script("run_segment.sh")
    labels = ["candidate--user--q4", "candidate--company--q8"]
    anchor = candidate_dir(tmp_path, labels[0])
    passing_checks(anchor / "verify-checks.csv")
    before = (anchor / "verify-checks.csv").read_bytes()
    result = run_stage("run_segment.sh", segment_args(tmp_path, labels), tmp_path)
    assert result.returncode == 0, result.stderr
    stages = printed_commands(result, "llm_kld.sh")
    if stages:
        [command] = stages
        assert str(candidate_dir(tmp_path, labels[1])) in command
        assert_flags(command, {"--parity-with": anchor})
    else:
        [collect] = printed_commands(result, "collect_llm_kld.py")
        assert_flags(collect, {"--out": candidate_dir(tmp_path, labels[1]) / "llm-kld"})
        [command] = printed_commands(result, "verify_article_collection.py")
        assert_flags(command, {"--parity-with": anchor / "llm-kld"})
    assert (anchor / "verify-checks.csv").read_bytes() == before


@pytest.mark.parametrize("receipt", [None, "check,passed,detail\nmanifest,false,failed\n"])
def test_segment_stops_on_existing_unverified_candidate(tmp_path, receipt):
    stage_script("run_segment.sh")
    labels = ["candidate--user--q4", "candidate--company--q8"]
    incomplete = candidate_dir(tmp_path, labels[0])
    incomplete.mkdir(parents=True)
    if receipt is not None:
        (incomplete / "verify-checks.csv").write_text(receipt)
    result = run_stage("run_segment.sh", segment_args(tmp_path, labels), tmp_path)
    assert result.returncode != 0
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not candidate_dir(tmp_path, labels[1]).exists()


@pytest.mark.parametrize("field,value", [
    (3, "../model"), (3, "."), (4, "corpus/child"), (4, ".."),
    (5, "../candidate--user--q4=model.gguf"), (5, "candidate--user--q4/child=model.gguf"),
    (5, "invalid=model.gguf"), (5, "candidate----q4=model.gguf"), (5, "candidate--user--=model.gguf"),
])
def test_segment_refuses_invalid_path_components_and_labels(tmp_path, field, value):
    args = segment_args(tmp_path, ["candidate--user--q4"])
    args[field] = value
    result = run_stage("run_segment.sh", args, tmp_path)
    assert result.returncode != 0
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not (tmp_path / "runs").exists()


def test_unparsable_manifest_writes_failed_manifest_and_lengths_checks(tmp_path):
    articles, candidate, anchor = verification_inputs(tmp_path)
    candidate["manifest"][0]["item_id"] = "x" * (csv.field_size_limit() + 1)
    write_manifest(candidate["root"] / "manifest.csv", candidate["manifest"])
    verify(tmp_path, articles, candidate, anchor, zero=True, expected=["manifest", "lengths"])


@pytest.mark.parametrize("overrides", [False, True], ids=["runtime-defaults", "explicit-bin-and-lib"])
def test_stage_env_exports_relative_runtime_paths_as_absolute(tmp_path, overrides):
    env = stage_env(tmp_path)
    relative = {"RUNTIME_DIR": "runtime", "TEXT_BIN": "runtime/bin", "TEXT_LIB": "runtime/lib"}
    env["RUNTIME_DIR"] = relative["RUNTIME_DIR"]
    if overrides:
        relative.update(TEXT_BIN="native tools/bin", TEXT_LIB="native tools/lib")
        env.update(relative)
    else:
        env.pop("TEXT_BIN")
        env.pop("TEXT_LIB")
    for path in relative.values():
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    command = 'source "$1"; "$PYTHON" -c "$2"'
    inspect = 'import json, os; print(json.dumps({key: os.environ[key] for key in ("RUNTIME_DIR", "TEXT_BIN", "TEXT_LIB")}))'
    result = subprocess.run(["bash", "-c", command, "article-test", str(stage_script("env.sh")), inspect],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {key: str(tmp_path / path) for key, path in relative.items()}


def test_segment_resumes_with_relative_prepared_and_runs_paths(tmp_path):
    prepared, receipt, rows, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q8", "candidate--company--q6"]
    anchor, candidate = [candidate_dir(tmp_path, label) for label in labels]
    for directory in (anchor, candidate):
        collection = synthetic_collection(directory / "llm-kld", receipt, rows)
        (collection["root"] / "corpus_articles.json").write_bytes((prepared / "corpus_articles.json").read_bytes())
        verify(directory, prepared / "corpus_articles.json", collection)
    before = {path: path.read_bytes() for directory in (anchor, candidate) for path in directory.rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[:3] = [inputs["ref_model"], prepared.relative_to(tmp_path), (tmp_path / "runs").relative_to(tmp_path)]
    assert not tmp_path.resolve().is_relative_to(REPO.resolve())
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode == 0, result.stderr
    parity = parity_checks(candidate, anchor)
    assert parity.is_file(), result.stderr
    assert all(row["passed"] == "true" for row in read_checks(parity))
    assert parity.with_suffix(".log").is_file()
    assert "8/8 checks passed" in parity.with_suffix(".log").read_text()
    [command] = printed_commands(result, "verify_article_collection.py")
    assert_flags(command, {
        "--articles": candidate / "llm-kld/corpus_articles.json", "--collection": candidate / "llm-kld",
        "--parity-with": anchor / "llm-kld", "--checks-out": parity,
    })
    assert str(parity.with_suffix(".log")) in command
    assert not list(anchor.glob("parity-*"))
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert {path: path.read_bytes() for path in before} == before


def stderr_commands(result):
    return [shlex.split(line[1:]) for line in result.stderr.splitlines() if line.startswith("+")]


@pytest.mark.parametrize("parity_receipt", ["absent", "failing"])
def test_segment_parity_preflight_precedes_first_collection(tmp_path, parity_receipt):
    prepared, _, _, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q4", "candidate--user--q8", "candidate--company--q6", "candidate--company--q5"]
    a, b, c, d = [candidate_dir(tmp_path, label) for label in labels]
    for directory in (b, c):
        passing_checks(directory / "verify-checks.csv")
        (directory / "llm-kld").mkdir()
        (directory / "llm-kld/corpus_articles.json").write_bytes((prepared / "corpus_articles.json").read_bytes())
    parity = parity_checks(c, b)
    if parity_receipt == "failing":
        passing_checks(parity)
        parity.write_text(parity.read_text().replace("reference_parity,true,", "reference_parity,false,"))
    before = {path: path.read_bytes() for directory in (b, c) for path in directory.rglob("*") if path.is_file()}
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path)
    commands = stderr_commands(result)
    collections = [i for i, command in enumerate(commands) if any(Path(word).name in ("llm_kld.sh", "collect_llm_kld.py") for word in command)]
    if parity_receipt == "failing":
        assert result.returncode != 0, result.stderr
        assert str(parity) in result.stderr
        assert not collections, result.stderr
        assert not printed_commands(result, "llm_kld.sh")
        assert not printed_commands(result, "collect_llm_kld.py")
        assert not printed_commands(result, "verify_article_collection.py")
    else:
        assert result.returncode == 0, result.stderr
        rechecks = [i for i, command in enumerate(commands) if "--checks-out" in command and command[command.index("--checks-out") + 1] == str(parity)]
        assert len(rechecks) == 1, result.stderr
        assert collections, result.stderr
        assert rechecks[0] < min(collections), result.stderr
        assert_flags(commands[rechecks[0]], {"--collection": c / "llm-kld", "--parity-with": b / "llm-kld"})
        assert_segment_collections(result, [a, d], b)
    assert not a.exists() and not d.exists()
    assert {path: path.read_bytes() for directory in (b, c) for path in directory.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("receipt", [None, "check,passed,detail\nmanifest,false,failed\n"], ids=["missing-receipt", "failing-receipt"])
def test_segment_prescan_stops_before_new_candidate_when_later_candidate_is_unverified(tmp_path, receipt):
    _, _, _, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q4", "candidate--company--q8"]
    new, incomplete = [candidate_dir(tmp_path, label) for label in labels]
    incomplete.mkdir(parents=True)
    if receipt is not None:
        (incomplete / "verify-checks.csv").write_text(receipt)
    before = {path: path.read_bytes() for path in incomplete.rglob("*") if path.is_file()}
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path)
    assert result.returncode != 0, result.stderr
    assert str(incomplete) in result.stderr
    assert not stderr_commands(result), result.stderr
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not new.exists()
    assert {path: path.read_bytes() for path in incomplete.rglob("*") if path.is_file()} == before


def test_segment_refuses_duplicate_labels_before_printing_commands(tmp_path):
    _, _, _, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q4", "candidate--company--q8", "candidate--user--q4"]
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = run_stage("run_segment.sh", args, tmp_path)
    assert result.returncode != 0, result.stderr
    assert not result.stdout
    assert not stderr_commands(result), result.stderr
    assert not (tmp_path / "runs").exists()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("mismatch", [1, 3], ids=["anchor", "later-resumed-candidate"])
def test_segment_refuses_nonidentical_corpus_copy_before_any_recheck_or_collection(tmp_path, mismatch):
    prepared, _, _, inputs = prepared_articles(tmp_path)
    labels = ["candidate--user--q4", "candidate--user--q8", "candidate--company--q6", "candidate--company--q5", "candidate--user--q3"]
    directories = [candidate_dir(tmp_path, label) for label in labels]
    original = (prepared / "corpus_articles.json").read_bytes()
    for directory in directories[1:4]:
        passing_checks(directory / "verify-checks.csv")
        (directory / "llm-kld").mkdir()
        (directory / "llm-kld/corpus_articles.json").write_bytes(original)
    different = directories[mismatch] / "llm-kld/corpus_articles.json"
    different.write_bytes(original + b" \n")
    assert different.read_bytes() != original
    assert json.loads(different.read_bytes()) == json.loads(original)
    before = {path: path.read_bytes() for directory in directories[1:4] for path in directory.rglob("*") if path.is_file()}
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path)
    assert result.returncode != 0, result.stderr
    assert not stderr_commands(result), result.stderr
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert not printed_commands(result, "verify_article_collection.py")
    assert not directories[0].exists() and not directories[-1].exists()
    assert {path: path.read_bytes() for directory in directories[1:4] for path in directory.rglob("*") if path.is_file()} == before


def verified_candidates(tmp_path, labels):
    prepared, receipt, rows, inputs = prepared_articles(tmp_path)
    directories = [candidate_dir(tmp_path, label) for label in labels]
    collections = []
    for i, directory in enumerate(directories):
        collection = synthetic_collection(directory / "llm-kld", receipt, rows)
        collection["meta"]["created"] = f"2026-01-01T00:00:{i:02d}Z"
        (collection["root"] / "collect_meta.json").write_text(json.dumps(collection["meta"], indent=2) + "\n")
        (collection["root"] / "corpus_articles.json").write_bytes((prepared / "corpus_articles.json").read_bytes())
        passing_checks(directory / "verify-checks.csv")
        collections.append(collection)
    return inputs, directories, collections


@pytest.mark.parametrize("name", ["label-only", "other-fingerprint"])
@pytest.mark.parametrize("passed", [True, False], ids=["passing", "failing"])
def test_segment_ignores_parity_receipts_for_other_anchor_collections(tmp_path, name, passed):
    labels = ["candidate--user--q8", "candidate--company--q6"]
    inputs, (anchor, candidate), _ = verified_candidates(tmp_path, labels)
    stale_name = f"parity-{labels[0]}"
    if name == "other-fingerprint":
        meta = anchor / "llm-kld/collect_meta.json"
        original = meta.read_bytes()
        stale_name += f"-{sha256(original)[:12]}"
        meta.write_bytes(original + b" \n")
    parity = parity_checks(candidate, anchor)
    stale = candidate / f"{stale_name}.csv"
    assert stale != parity
    passing_checks(stale)
    if not passed:
        stale.write_text(stale.read_text().replace("reference_parity,true,", "reference_parity,false,"))
    stale.with_suffix(".log").write_text("previous parity log\n")
    before = {path: path.read_bytes() for directory in (anchor, candidate) for path in directory.rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode == 0, result.stderr
    [command] = printed_commands(result, "verify_article_collection.py")
    assert_flags(command, {
        "--collection": candidate / "llm-kld", "--parity-with": anchor / "llm-kld", "--checks-out": parity,
    })
    assert str(parity.with_suffix(".log")) in command
    assert all(row["passed"] == "true" for row in read_checks(parity))
    assert "8/8 checks passed" in parity.with_suffix(".log").read_text()
    assert set(candidate.glob("parity-*")) == {stale, stale.with_suffix(".log"), parity, parity.with_suffix(".log")}
    assert not list(anchor.glob("parity-*"))
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("passed", [True, False], ids=["passing-skips", "failing-stops"])
def test_segment_honors_receipt_for_current_anchor_collection(tmp_path, passed):
    labels = ["candidate--user--q8", "candidate--company--q6"]
    inputs, (anchor, candidate), _ = verified_candidates(tmp_path, labels)
    parity = parity_checks(candidate, anchor)
    passing_checks(parity)
    if not passed:
        parity.write_text(parity.read_text().replace("reference_parity,true,", "reference_parity,false,"))
        labels = ["candidate--user--q4", *labels, "candidate--company--q5"]
    parity.with_suffix(".log").write_text("previous parity log\n")
    before = {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert (result.returncode == 0) == passed, result.stderr
    if not passed:
        assert str(parity) in result.stderr
        assert not candidate_dir(tmp_path, labels[0]).exists()
        assert not candidate_dir(tmp_path, labels[-1]).exists()
    assert not stderr_commands(result), result.stderr
    assert {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()} == before


def test_segment_recollected_anchor_cannot_reuse_old_passing_parity(tmp_path):
    labels = ["candidate--user--q8", "candidate--company--q6"]
    inputs, (anchor, candidate), (anchor_collection, candidate_collection) = verified_candidates(tmp_path, labels)
    articles = tmp_path / "prepared/corpus_articles.json"
    old_parity = parity_checks(candidate, anchor)
    verify(candidate, articles, candidate_collection, anchor_collection, checks_name=old_parity.name)
    legacy_parity = candidate / f"parity-{labels[0]}.csv"
    legacy_parity.write_bytes(old_parity.read_bytes())
    anchor_collection["meta"]["created"] = "2026-01-02T00:00:00Z"
    (anchor / "llm-kld/collect_meta.json").write_text(json.dumps(anchor_collection["meta"], indent=2) + "\n")
    for column in ("nll_ref", "nll_cand"):
        edit_npz(anchor_collection["paths"][0], column, -0.0, 0)
    verify(tmp_path, articles, anchor_collection, zero=True, checks_name="recollected-anchor.csv")
    parity = parity_checks(candidate, anchor)
    assert parity != old_parity
    before = {path: path.read_bytes() for directory in (anchor, candidate) for path in directory.rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode != 0, result.stderr
    assert {row["check"] for row in read_checks(parity) if row["passed"] == "false"} == {"reference_parity"}
    assert "reference_parity" in parity.with_suffix(".log").read_text()
    [command] = printed_commands(result, "verify_article_collection.py")
    assert_flags(command, {"--collection": candidate / "llm-kld", "--parity-with": anchor / "llm-kld", "--checks-out": parity})
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("dry_run", [False, True], ids=["real-stop", "dry-run-missing"])
def test_segment_missing_anchor_meta_precedes_rechecks_and_collection(tmp_path, dry_run):
    verified_labels = ["candidate--user--q8", "candidate--company--q6"]
    inputs, (anchor, candidate), _ = verified_candidates(tmp_path, verified_labels)
    meta = anchor / "llm-kld/collect_meta.json"
    meta.unlink()
    labels = ["candidate--user--q4", *verified_labels, "candidate--company--q5"]
    before = {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "1" if dry_run else "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert (result.returncode == 0) == dry_run, result.stderr
    assert any(str(meta) in line and "missing" in line.replace(str(tmp_path), "").lower() for line in result.stderr.splitlines() if not line.startswith("+")), result.stderr
    if dry_run:
        parity = candidate / f"parity-{anchor.name}-missing.csv"
        [command] = [command for command in printed_commands(result, "verify_article_collection.py") if command[command.index("--collection") + 1] == str(candidate / "llm-kld")]
        assert_flags(command, {"--parity-with": anchor / "llm-kld", "--checks-out": parity})
        assert str(parity.with_suffix(".log")) in command
        commands = stderr_commands(result)
        first_collection = next(i for i, entry in enumerate(commands) if any(Path(word).name in ("llm_kld.sh", "collect_llm_kld.py") for word in entry))
        assert commands.index(command) < first_collection
        assert_segment_collections(result, [candidate_dir(tmp_path, labels[0]), candidate_dir(tmp_path, labels[-1])], anchor)
    else:
        assert not stderr_commands(result), result.stderr
    assert not candidate_dir(tmp_path, labels[0]).exists()
    assert not candidate_dir(tmp_path, labels[-1]).exists()
    assert {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("python", ["relative-path", "bare-command"])
def test_stage_env_exports_absolute_fork_and_python_paths(tmp_path, python):
    fork = tmp_path / "fork tree"
    fork.mkdir()
    executable = tmp_path / "python tools/python"
    executable.parent.mkdir()
    executable.symlink_to(sys.executable)
    env = stage_env(tmp_path)
    env["FORK_REPO"] = "./fork tree"
    env["PYTHON"] = "./python tools/python" if python == "relative-path" else "python3"
    command = 'source "$1"; "$2" -c "$3"'
    inspect = 'import json, os; print(json.dumps({key: os.environ[key] for key in ("FORK_REPO", "PYTHON")}))'
    result = subprocess.run(["bash", "-c", command, "article-test", str(stage_script("env.sh")), sys.executable, inspect],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "FORK_REPO": str(fork), "PYTHON": str(executable) if python == "relative-path" else "python3",
    }


@pytest.mark.parametrize("missing_index", [0, 1], ids=["anchor", "later-resumed-candidate"])
@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_segment_missing_corpus_copy_has_own_diagnostic_before_commands(tmp_path, missing_index, dry_run):
    verified_labels = ["candidate--user--q8", "candidate--company--q6"]
    inputs, directories, _ = verified_candidates(tmp_path, verified_labels)
    missing = directories[missing_index] / "llm-kld/corpus_articles.json"
    missing.unlink()
    labels = ["candidate--user--q4", *verified_labels, "candidate--company--q5"]
    before = {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "1" if dry_run else "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode != 0, result.stderr
    assert not stderr_commands(result), result.stderr
    assert any(str(missing) in line and "missing" in line.replace(str(tmp_path), "").lower() for line in result.stderr.splitlines()), result.stderr
    assert "differs" not in result.stderr.lower()
    assert "another preparation" not in result.stderr.lower()
    assert not candidate_dir(tmp_path, labels[0]).exists()
    assert not candidate_dir(tmp_path, labels[-1]).exists()
    assert {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("reverse", [False, True], ids=["short-first", "long-first"])
def test_segment_distinguishes_prefix_labels_and_names_duplicates(tmp_path, reverse):
    labels = ["candidate--a--q4", "candidate--a--q4_k"]
    if reverse:
        labels.reverse()
    inputs, (anchor, candidate), _ = verified_candidates(tmp_path, labels)
    before = {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()}
    env = stage_env(tmp_path)
    env["DRY_RUN"] = "0"
    args = segment_args(tmp_path, labels)
    args[0] = inputs["ref_model"]
    duplicate = run_stage("run_segment.sh", [*args, args[5]], tmp_path, env)
    assert duplicate.returncode != 0, duplicate.stderr
    assert labels[0] in shlex.split(duplicate.stderr)
    assert not duplicate.stdout
    assert not stderr_commands(duplicate), duplicate.stderr
    assert {path: path.read_bytes() for path in (tmp_path / "runs").rglob("*") if path.is_file()} == before
    result = run_stage("run_segment.sh", args, tmp_path, env)
    assert result.returncode == 0, result.stderr
    parity = parity_checks(candidate, anchor)
    [command] = printed_commands(result, "verify_article_collection.py")
    assert_flags(command, {"--collection": candidate / "llm-kld", "--parity-with": anchor / "llm-kld", "--checks-out": parity})
    assert all(row["passed"] == "true" for row in read_checks(parity))
    assert "8/8 checks passed" in parity.with_suffix(".log").read_text()
    assert not list(anchor.glob("parity-*"))
    assert not printed_commands(result, "llm_kld.sh")
    assert not printed_commands(result, "collect_llm_kld.py")
    assert {path: path.read_bytes() for path in before} == before
