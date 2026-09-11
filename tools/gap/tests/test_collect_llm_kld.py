"""Tests for collect_llm_kld.py pure helpers.

Hermetic: synthesizes VLMK fixtures; no models, datasets, or GPU.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import cli.collect_llm_kld as ck
import lib.kld_metrics_io as kio
import cli.prep_llm_score_from_hf as prep_lib

import lib.collect_common as common
from test_collect_kld import _make_row
from fakes import make_records, write_vlmk, EXECUTION_IDENTITY


@pytest.fixture(autouse=True)
def _skip_scorer_preflight(monkeypatch):
    monkeypatch.setattr(ck, "build_collect_provenance", lambda *args: {"execution_identity": EXECUTION_IDENTITY})
    """main() preflights the scorer binary's --vlmk-version before anything
    else; most main() tests use fake scorers that do not implement the flag,
    so stub the gate by default. The preflight's own tests below call
    collect_common.preflight_scorer_vlmk_version directly."""
    monkeypatch.setattr(ck, "preflight_scorer_vlmk_version",
                        lambda path: kio.VLMK_VERSION)


def _args(tmp_path, **overrides):
    base = dict(
        ref_model=str(tmp_path / "ref.gguf"),
        cand_model=str(tmp_path / "cand.gguf"),
        dataset="company/llm-ground-truth-general-fix-double-BOS",
        subset="Qwen3-4B-Instruct-2507-vllm",
        split="train",
        sort_by="",
        sort_desc=False,
        num_eval_tokens=-1,
        max_total_tokens=None,
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
    base.update(overrides)
    return argparse.Namespace(**base)


def test_scorer_argv_forwards_explicit_execution_protocol(tmp_path):
    args = _args(
        tmp_path, llama_llm_kld="/bin/llama-llm-kld",
        n_ctx=32768, n_batch=2048, n_ubatch=512, tf_chunk=2048,
        n_gpu_layers=99, n_threads=8, metric_threads=8, flash_attn="enabled",
    )
    manifest = tmp_path / "manifest.jsonl"

    assert ck._scorer_argv(args, manifest) == [
        "/bin/llama-llm-kld",
        "--ref-model", args.ref_model,
        "--cand-model", args.cand_model,
        "--manifest", str(manifest),
        "--num-eval-tokens", "-1",
        "-b", "2048", "-c", "32768", "-ub", "512", "-ngl", "99",
        "--tf-chunk", "2048", "-t", "8", "--metric-threads", "8",
        "--flash-attn",
    ]


def test_scorer_argv_forwards_swa_full_only_when_requested(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    off = _args(tmp_path, llama_llm_kld="/bin/llama-llm-kld")
    on = _args(tmp_path, llama_llm_kld="/bin/llama-llm-kld", swa_full=True)
    assert "--swa-full" not in ck._scorer_argv(off, manifest)
    assert ck._scorer_argv(on, manifest)[-1] == "--swa-full"


def test_manifest_entry_has_text_only_shape(tmp_path):
    entry = ck.build_kld_manifest_entry(tmp_path / "prep", tmp_path / "m.bin", n_prefill=42)

    assert entry == {
        "tokens_in": str(tmp_path / "prep" / "tokens.bin"),
        "n_prefill": 42,
        "output_metrics": str(tmp_path / "m.bin"),
    }


def test_build_collect_meta_is_llm_kind_and_has_no_vlm_fields(tmp_path):
    meta = ck.build_collect_meta(_args(tmp_path))

    assert meta["kind"] == "llm_kld_metrics"
    assert "kind" in ck.IDENTITY_FIELDS
    assert "ref_mmproj" not in meta
    assert "cand_mmproj" not in meta
    assert "image_min_tokens" not in meta
    assert "image_max_tokens" not in meta
    assert "media_wrapper" not in meta


def test_parse_args_defaults_to_llm_dataset_and_sort_natural(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_llm_kld.py",
            "--ref-model",
            "ref.gguf",
            "--cand-model",
            "cand.gguf",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    args = ck.parse_args()

    assert args.dataset == "company/llm-ground-truth-general-fix-double-BOS"
    assert args.subset == "Qwen3-4B-Instruct-2507-vllm"
    assert args.sort_by == ""
    assert args.sort_desc is False
    assert Path(args.llama_llm_kld) == Path(__file__).resolve().parents[3] / "build/bin/llama-llm-kld"
    assert args.n_gpu_layers == 99
    assert args.n_threads == -1
    assert args.metric_threads == -1
    assert args.flash_attn == "auto"
    assert args.swa_full is False
    assert ck.build_collect_meta(args)["swa_full"] is False
    assert "swa_full" in ck.IDENTITY_FIELDS


def test_parse_args_documents_sort_direction(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["collect_llm_kld.py", "--help"])

    with pytest.raises(SystemExit):
        ck.parse_args()

    out = capsys.readouterr().out
    help_text = " ".join(out.split())
    assert "company sorts ascending" in help_text
    assert "logits repo sorts descending" in help_text
    assert "--sort-desc" in out


@pytest.mark.parametrize("allow_attributes", [False, True])
def test_vocab_attr_mismatch_is_opt_in_and_preserves_legacy_meta(tmp_path, monkeypatch, allow_attributes):
    argv = ["collect_llm_kld.py", "--ref-model", "ref.gguf", "--cand-model", "cand.gguf", "--out", str(tmp_path)]
    if allow_attributes:
        argv.append("--allow-vocab-attr-mismatch")
    monkeypatch.setattr("sys.argv", argv)

    args = ck.parse_args()
    command = ck._scorer_argv(args, tmp_path / "manifest.jsonl")
    meta = ck.build_collect_meta(args)
    assert args.allow_vocab_attr_mismatch is allow_attributes
    assert ("--allow-vocab-attr-mismatch" in command) is allow_attributes
    assert meta["allow_vocab_attr_mismatch"] is allow_attributes

    legacy = dict(meta)
    legacy.pop("allow_vocab_attr_mismatch")
    assert ck.ensure_collect_meta(tmp_path, legacy) is True
    stored = (tmp_path / "collect_meta.json").read_bytes()
    assert ck.ensure_collect_meta(tmp_path, meta) is False
    assert (tmp_path / "collect_meta.json").read_bytes() == stored


def test_sort_desc_is_guarded_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_llm_kld.py",
            "--ref-model", "ref.gguf",
            "--cand-model", "cand.gguf",
            "--out", str(tmp_path / "out"),
            "--sort-by", "generated_tokens_len",
            "--sort-desc",
        ],
    )

    args = ck.parse_args()
    meta = ck.build_collect_meta(args)

    assert "sort_desc" in ck.IDENTITY_FIELDS
    assert args.sort_desc is True
    assert meta["sort_desc"] is True


def test_max_total_tokens_is_an_identity_field_and_missing_field_refuses(tmp_path):
    current = ck.build_collect_meta(_args(tmp_path, max_total_tokens=None))
    assert "max_total_tokens" in ck.IDENTITY_FIELDS
    assert ck.ensure_collect_meta(tmp_path, current) is True
    assert ck.ensure_collect_meta(tmp_path, current) is False
    changed = ck.build_collect_meta(_args(tmp_path, max_total_tokens=10))
    with pytest.raises(SystemExit, match="max_total_tokens"):
        ck.ensure_collect_meta(tmp_path, changed)
    legacy = dict(current)
    legacy.pop("max_total_tokens")
    (tmp_path / "collect_meta.json").write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(SystemExit, match="max_total_tokens: missing"):
        ck.ensure_collect_meta(tmp_path, current)


def test_n_ctx_preflight_for_kld_uses_uncapped_answer_tokens():
    pending = [
        {"idx": 1, "item_id": "short", "n_prefill": 8, "n_answer": 5},
        {"idx": 2, "item_id": "long", "n_prefill": 4, "n_answer": 20},
    ]

    from lib.collect_common import compute_n_ctx_requirement
    requirement = compute_n_ctx_requirement(pending, num_eval_tokens=-1)

    assert requirement == {
        "idx": 2,
        "item_id": "long",
        "n_prefill": 4,
        "n_answer": 20,
        "n_eval": 20,
        "row_need": 24,
        "n_seq_max": 1,
        "need": 24,
    }
    ck.preflight_n_ctx(pending, n_ctx=24, num_eval_tokens=-1)
    with pytest.raises(SystemExit) as exc:
        ck.preflight_n_ctx(pending, n_ctx=23, num_eval_tokens=-1)
    msg = str(exc.value)
    assert "row_idx=2" in msg
    assert "item_id=long" in msg
    assert "n_answer=20" in msg
    assert "suggested --n-ctx 24" in msg


@pytest.mark.parametrize("field,value", [("kind", "vlm_kld_metrics"), ("dataset", "other/ds"),
                                         ("swa_full", True)])
def test_ensure_collect_meta_fails_on_identity_mismatch(tmp_path, field, value):
    ck.ensure_collect_meta(tmp_path, ck.build_collect_meta(_args(tmp_path)))
    changed = ck.build_collect_meta(_args(tmp_path))
    changed[field] = value

    with pytest.raises(SystemExit, match=field):
        ck.ensure_collect_meta(tmp_path, changed)


def test_postprocess_manifest_row_omits_num_images(tmp_path):
    row = _make_row(tmp_path, npos=3, n_prefill=4, n_answer=4)

    manifest_row = ck.postprocess_kld_result(row, num_eval_tokens=3, elapsed_s=2.5)

    assert manifest_row[:6] == [0, "item", 4, 4, 3, 11]
    assert manifest_row[7:] == ["2.50", "OK"]


def test_write_kld_manifest_jsonl_has_no_media_keys(tmp_path):
    path = tmp_path / "_manifest.jsonl"
    entry = ck.build_kld_manifest_entry(tmp_path / "prep", tmp_path / "out.bin", 5)

    ck.write_kld_manifest(path, [entry])

    [loaded] = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == entry
    assert "images" not in loaded
    assert "formatted_chat" not in loaded


def _paths(tmp_path):
    return tmp_path / "row.bin", tmp_path / "row.npz"


# ---------------------------------------------------------------------------
# scorer --vlmk-version preflight
# ---------------------------------------------------------------------------

def _write_version_scorer(path: Path, body: str):
    path.write_text("#!/usr/bin/env python3\nimport sys\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_preflight_accepts_current_version_scorer(tmp_path):
    scorer = tmp_path / "scorer.py"
    _write_version_scorer(
        scorer,
        "assert sys.argv[1:] == ['--vlmk-version'], sys.argv\n"
        f"print({kio.VLMK_VERSION})\n")
    assert common.preflight_scorer_vlmk_version(scorer) == kio.VLMK_VERSION


def test_preflight_rejects_older_version_scorer(tmp_path):
    scorer = tmp_path / "scorer.py"
    _write_version_scorer(scorer, f"print({kio.VLMK_VERSION - 1})\n")
    with pytest.raises(SystemExit, match=rf"writes VLMK v{kio.VLMK_VERSION - 1} .* rebuild"):
        common.preflight_scorer_vlmk_version(scorer)


def test_preflight_rejects_scorer_without_the_flag(tmp_path):
    """A pre-v2 binary rejects --vlmk-version as an unknown argument (exit
    1, nothing on stdout) — that alone must fail the preflight."""
    scorer = tmp_path / "scorer.py"
    _write_version_scorer(
        scorer, "print('unknown argument: --vlmk-version', file=sys.stderr)\nsys.exit(1)\n")
    with pytest.raises(SystemExit, match="outdated build"):
        common.preflight_scorer_vlmk_version(scorer)


def test_preflight_rejects_missing_scorer(tmp_path):
    with pytest.raises(SystemExit, match="cannot run"):
        common.preflight_scorer_vlmk_version(tmp_path / "nope")


def test_postprocess_rejects_npos_mismatch(tmp_path):
    row = _make_row(tmp_path, npos=5, n_prefill=4, n_answer=5)

    with pytest.raises(ValueError, match="npos"):
        ck.postprocess_kld_result(row, num_eval_tokens=3, elapsed_s=0.0)


def test_postprocess_rejects_target_mismatch_and_keeps_bin_as_evidence(tmp_path):
    row = _make_row(tmp_path, npos=3, n_prefill=4, n_answer=4, seed=2)
    other = _make_row(tmp_path / "other", npos=3, n_prefill=4, n_answer=4, seed=9)
    row["metrics_path"].unlink()
    other["metrics_path"].rename(row["metrics_path"])
    before = row["metrics_path"].read_bytes()

    with pytest.raises(ValueError, match="target") as excinfo:
        ck.postprocess_kld_result(row, num_eval_tokens=3, elapsed_s=0.0)
    # the ORIGINAL exception type/message survives (the manifest status is
    # derived from it); the kept path rides along as a note
    assert excinfo.type is ValueError
    assert any("rejected output kept at" in n for n in excinfo.value.__notes__)
    rejected = row["metrics_path"].with_name(row["metrics_path"].name + ".rejected")
    assert rejected.read_bytes() == before                # kept, never deleted
    assert not row["metrics_path"].exists()               # but out of glob reach


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if key == "id":
            return [row["id"] for row in self._rows]
        return self._rows[key]


def _fake_row(item_id, tokens, n_prefill=2):
    tokens = list(tokens)
    return {
        "id": item_id,
        "input_ids": tokens,
        "input_tokens_len": len(tokens),
        "n_prefill_tokens": n_prefill,
        "generated_tokens_len": len(tokens) - n_prefill,
        "labels": [-100] * n_prefill + tokens[n_prefill:],
        "source": "unit",
        "category": "test",
        "generated_texts": "",
        "seed": 0,
        "question": "",
    }


def _fake_prep_row(row, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(row["input_ids"], dtype=np.int32).tofile(out_dir / "tokens.bin")
    return {
        "item_id": str(row["id"]),
        "n_prefill": int(row["n_prefill_tokens"]),
        "n_answer": int(row["generated_tokens_len"]),
    }


def _write_called_marker_scorer(path: Path, marker: Path, exit_code: int):
    path.write_text(
        f"""#!/usr/bin/env python3
from pathlib import Path
import sys

Path({str(marker)!r}).write_text("called", encoding="utf-8")
sys.exit({exit_code})
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_main_n_ctx_preflight_exits_before_launching_llm_kld(tmp_path, monkeypatch):
    ref_model = tmp_path / "ref.gguf"
    cand_model = tmp_path / "cand.gguf"
    ref_model.write_bytes(b"ref")
    cand_model.write_bytes(b"cand")
    scorer_marker = tmp_path / "scorer_called"
    scorer = tmp_path / "fake_llm_kld.py"
    _write_called_marker_scorer(scorer, scorer_marker, 0)
    out = tmp_path / "out"

    ds = _FakeDataset([_fake_row(17, [10, 11, 2, 3])])
    monkeypatch.setattr(prep_lib, "load_dataset_sorted", lambda *args: ds)
    monkeypatch.setattr(prep_lib, "prep_row", _fake_prep_row)
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_llm_kld.py",
            "--ref-model", str(ref_model),
            "--cand-model", str(cand_model),
            "--llama-llm-kld", str(scorer),
            "--out", str(out),
            "--start", "0",
            "--end", "1",
            "--n-ctx", "3",
        ],
    )

    with pytest.raises(SystemExit, match="suggested --n-ctx 4"):
        ck.main()

    assert not scorer_marker.exists()


def test_main_over_budget_then_collision_then_disjoint_v2_shard(
    tmp_path, monkeypatch, capsys,
):
    """Window [0,1): over-budget -> SKIP row, no prep, no scorer. Same
    window again: refused before prep/scorer, files byte-identical. Window
    [1,2): the fake scorer writes a current-version VLMK dump that is
    validated, converted and appended under the same manifest header."""
    ref_model = tmp_path / "ref.gguf"
    cand_model = tmp_path / "cand.gguf"
    ref_model.write_bytes(b"ref")
    cand_model.write_bytes(b"cand")
    out = tmp_path / "out"
    prep_calls = []

    tokens2 = np.array([10, 11, 2, 3], dtype=np.int32)
    rec = make_records(npos=2, vocab=11, seed=9)
    rec["target"] = tokens2[2:4]
    staged = tmp_path / "v2_output.bin"
    write_vlmk(staged, rec, vocab=11, n_prefill=2)
    scorer = tmp_path / "fake_llm_kld.py"
    scorer_marker = tmp_path / "scorer_called"
    scorer.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        f"open({str(scorer_marker)!r}, 'a').write('x')\n"
        "i = sys.argv.index('--manifest')\n"
        "entry = json.loads(open(sys.argv[i + 1]).readline())\n"
        f"open(entry['output_metrics'], 'wb').write(open({str(staged)!r}, 'rb').read())\n"
        "print('DONE output_metrics=' + entry['output_metrics'] + ' wall_s=0.5')\n"
        "sys.exit(0)\n", encoding="utf-8")
    scorer.chmod(0o755)

    ds = _FakeDataset([_fake_row(17, [0] * 12, n_prefill=10),
                       _fake_row(18, tokens2.tolist())])

    def fake_prep(row, out_dir):
        prep_calls.append(row["id"])
        return _fake_prep_row(row, out_dir)

    monkeypatch.setattr(prep_lib, "load_dataset_sorted", lambda *args: ds)
    monkeypatch.setattr(prep_lib, "prep_row", fake_prep)

    def argv(start, end):
        return ["collect_llm_kld.py", "--ref-model", str(ref_model),
                "--cand-model", str(cand_model), "--llama-llm-kld", str(scorer),
                "--out", str(out), "--start", str(start), "--end", str(end),
                "--num-eval-tokens", "2", "--max-total-tokens", "11"]

    monkeypatch.setattr("sys.argv", argv(0, 1))
    ck.main()
    assert prep_calls == [] and not scorer_marker.exists()
    manifest = out / "manifest.csv"
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["row_idx"], r["status"]) for r in rows] == [("0", "SKIP_OVER_BUDGET")]
    assert rows[0]["n_prefill"] == "10" and rows[0]["n_answer"] == "2" and rows[0]["n_eval"] == "2"
    assert "skipped 1 over-budget row(s) (max-total-tokens=11)" in capsys.readouterr().err
    snapshot = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}

    with pytest.raises(SystemExit) as excinfo:
        ck.main()
    assert "row 0: manifest.csv status=SKIP_OVER_BUDGET" in str(excinfo.value)
    assert not scorer_marker.exists() and prep_calls == []
    assert all(p.read_bytes() == content for p, content in snapshot.items())
    assert any(json.loads(p.read_text())["state"] == "aborted" for p in (out / ".attempts").glob("*/state.json"))

    monkeypatch.setattr("sys.argv", argv(1, 2))
    ck.main()
    assert prep_calls == [18] and scorer_marker.exists()
    npz = out / "metrics" / "001_18.npz"
    assert npz.exists() and not npz.with_suffix(".bin").exists()
    assert kio.read_kld_header(npz)["version"] == kio.VLMK_VERSION
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["row_idx"], r["status"]) for r in rows] == [("0", "SKIP_OVER_BUDGET"), ("1", "OK")]
    assert manifest.read_text().count("row_idx,item_id") == 1

    # and now row 1's artifacts collide too (any status, any artifact)
    with pytest.raises(SystemExit) as excinfo:
        ck.main()
    assert "row 1: metrics/001_18.npz" in str(excinfo.value)


def test_main_rejects_nonpositive_max_total_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_llm_kld.py",
            "--ref-model", "ref.gguf",
            "--cand-model", "cand.gguf",
            "--out", str(tmp_path / "out"),
            "--max-total-tokens", "0",
        ],
    )

    with pytest.raises(SystemExit, match="--max-total-tokens"):
        ck.main()


def test_main_records_fail_exit_row_with_fake_scorer(tmp_path, monkeypatch):
    ref_model = tmp_path / "ref.gguf"
    cand_model = tmp_path / "cand.gguf"
    ref_model.write_bytes(b"ref")
    cand_model.write_bytes(b"cand")
    scorer_marker = tmp_path / "scorer_called"
    scorer = tmp_path / "fake_llm_kld.py"
    _write_called_marker_scorer(scorer, scorer_marker, 7)
    out = tmp_path / "out"

    ds = _FakeDataset([_fake_row(17, [10, 11, 2, 3])])
    monkeypatch.setattr(prep_lib, "load_dataset_sorted", lambda *args: ds)
    monkeypatch.setattr(prep_lib, "prep_row", _fake_prep_row)
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_llm_kld.py",
            "--ref-model", str(ref_model),
            "--cand-model", str(cand_model),
            "--llama-llm-kld", str(scorer),
            "--out", str(out),
            "--start", "0",
            "--end", "1",
        ],
    )

    with pytest.raises(SystemExit, match="1 failed row"):
        ck.main()

    assert scorer_marker.exists()
    with open(out / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "FAIL_EXIT_7"
    assert rows[0]["item_id"] == "17"


def test_main_rejects_negative_start_and_bad_end(tmp_path, monkeypatch):
    """Finding 11 (LLM KLD lane): negative --start rejected before any
    output dir is created."""
    argv = ["collect_llm_kld.py", "--ref-model", "r.gguf",
            "--cand-model", "c.gguf", "--out", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", [*argv, "--start", "-1"])
    with pytest.raises(SystemExit, match="--start must be >= 0"):
        ck.main()
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr("sys.argv", [*argv, "--end", "-2"])
    with pytest.raises(SystemExit, match="--end must be -1"):
        ck.main()


def test_main_refuses_meta_less_root_that_already_holds_rows(tmp_path, monkeypatch):
    """The meta-less-root refusal is a copy-pasted preamble in all four
    collectors; cover it through main() on the KLD lane too (test_collect_logits
    covers the logits lane)."""
    out = tmp_path / "out"
    (out / "metrics").mkdir(parents=True)
    (out / "metrics" / "000_17.npz").write_bytes(b"not really an npz")
    ref_model = tmp_path / "ref.gguf"
    cand_model = tmp_path / "cand.gguf"
    ref_model.write_bytes(b"ref")
    cand_model.write_bytes(b"cand")
    scorer = tmp_path / "fake_llm_kld.py"
    _write_version_scorer(scorer, f"print({kio.VLMK_VERSION})\n")
    ds = _FakeDataset([_fake_row(17, [10, 11, 2, 3]), _fake_row(18, [10, 11, 4, 5])])
    monkeypatch.setattr(prep_lib, "load_dataset_sorted", lambda *args: ds)
    monkeypatch.setattr(prep_lib, "prep_row", lambda *a: pytest.fail("no prep"))
    monkeypatch.setattr("sys.argv", [
        "collect_llm_kld.py", "--ref-model", str(ref_model),
        "--cand-model", str(cand_model), "--llama-llm-kld", str(scorer),
        "--out", str(out), "--start", "1", "--end", "2"])
    with pytest.raises(SystemExit) as excinfo:
        ck.main()
    assert "no collect_meta.json" in str(excinfo.value)
    assert not (out / "collect_meta.json").exists()



def test_sigterm_append_preserves_finished_metrics_and_records_interruption(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    company = Path(__file__).resolve().parents[1]
    scorer = tmp_path / "fake-scorer.py"
    scorer.write_text("#!/usr/bin/env python3\n" + r"""
import json, os, pathlib, sys, time
manifest = pathlib.Path(sys.argv[sys.argv.index('--manifest') + 1])
for entry in map(json.loads, manifest.read_text().splitlines()):
    output = pathlib.Path(entry['output_metrics'])
    data = pathlib.Path(__file__).with_name('fixture.bin').read_bytes()
    output.write_bytes(data)
    if output.name.startswith('003_'):
        pathlib.Path(__file__).with_name('child.pid').write_text(str(os.getpid()))
        pathlib.Path(__file__).with_name('ready').write_text('ready')
        time.sleep(60)
""")
    scorer.chmod(0o755)
    records = make_records(npos=2, vocab=11)
    records['target'] = [2, 3]
    write_vlmk(tmp_path / 'fixture.bin', records, n_prefill=2)
    (tmp_path / 'ref.gguf').write_bytes(b'ref')
    (tmp_path / 'cand.gguf').write_bytes(b'cand')
    worker = tmp_path / 'worker.py'
    worker.write_text(f"import sys\nsys.path[:0] = {[str(company), str(company / 'tests')]!r}\n" + r"""
from pathlib import Path
from test_collect_llm_kld import _FakeDataset, _fake_row, _fake_prep_row
from fakes import EXECUTION_IDENTITY
import cli.collect_llm_kld as collector
import cli.prep_llm_score_from_hf as prep
base = Path(__file__).parent
prep.load_dataset_sorted = lambda *args: _FakeDataset([_fake_row(str(i), [0, 1, 2, 3]) for i in range(5)])
prep.prep_row = _fake_prep_row
collector.preflight_scorer_vlmk_version = lambda *args: 4
collector.build_collect_provenance = lambda *args: {'execution_identity': EXECUTION_IDENTITY}
start, end = sys.argv[1:]
sys.argv = ['collector', '--ref-model', str(base/'ref.gguf'), '--cand-model', str(base/'cand.gguf'),
            '--llama-llm-kld', str(base/'fake-scorer.py'), '--out', str(base/'out'),
            '--start', start, '--end', end, '--num-eval-tokens', '2']
collector.main()
""")
    subprocess.run([sys.executable, str(worker), '0', '3'], capture_output=True, text=True, check=True, timeout=20)
    before = {p.name: p.read_bytes() for p in (tmp_path/'out/metrics').glob('*.npz')}
    proc = subprocess.Popen([sys.executable, str(worker), '3', '5'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path/'ready').exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path/'ready').exists(), proc.communicate(timeout=2)
        os.kill(proc.pid, signal.SIGTERM)
        _out, err = proc.communicate(timeout=12)
        assert proc.returncode != 0, err
        child_pid = int((tmp_path/'child.pid').read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    metrics = tmp_path/'out/metrics'
    assert all((metrics/name).read_bytes() == content for name, content in before.items())
    assert (metrics/'003_3.npz').exists()
    assert not (metrics/'004_4.npz').exists()
    states = [json.loads(p.read_text())['state'] for p in (tmp_path/'out/.attempts').glob('*/state.json')]
    assert sorted(states) == ['completed', 'interrupted']
    from lib.collection_state import require_completed_attempts
    with pytest.raises(ValueError, match='interrupted'):
        require_completed_attempts(tmp_path/'out')


def test_perplexity_runtime_and_mode_are_explicit(tmp_path):
    args = _args(tmp_path, perplexity_window=True, n_ctx=512, n_batch=512,
                 n_ubatch=512, tf_chunk=-1, llama_llm_kld="/bin/llama-llm-kld")
    ck.validate_perplexity_runtime(args)
    assert "--perplexity-window" in ck._scorer_argv(args, tmp_path / "manifest.jsonl")
    assert ck.build_collect_meta(args)["perplexity_window"] is True
    for field, value in (("n_ctx", 511), ("n_batch", 2048), ("n_ubatch", 256),
                         ("num_eval_tokens", 254), ("tf_chunk", 1), ("sort_by", "id"),
                         ("sort_desc", True), ("max_total_tokens", 512)):
        changed = argparse.Namespace(**{**vars(args), field: value})
        with pytest.raises(ValueError, match="perplexity-window"):
            ck.validate_perplexity_runtime(changed)
