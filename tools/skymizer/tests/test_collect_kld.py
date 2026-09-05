"""Tests for collect_kld.py's pure helpers: manifest building, DONE-marker
parsing, postprocess validation, and the shard-identity check.

Hermetic: synthesizes VLMK fixtures; no GPU, no models, no built binary.
Run from tools/skymizer:  python3 -m pytest tests/test_collect_kld.py -v
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import cli.collect_kld as ck
import lib.collect_common as ccommon
import lib.kld_metrics_io as kio

from fakes import make_records, write_vlmk


@pytest.fixture(autouse=True)
def _skip_scorer_preflight(monkeypatch):
    """main() preflights the scorer binary's --vlmk-version before anything
    else; the main() tests here point --llama-vlm-kld at placeholder files,
    so stub the gate (it has its own tests in test_collect_llm_kld.py)."""
    monkeypatch.setattr(ck, "preflight_scorer_vlmk_version",
                        lambda path: kio.VLMK_VERSION)


# ---------------------------------------------------------------------------
# manifest building + DONE parsing
# ---------------------------------------------------------------------------

def test_build_kld_manifest_entry(tmp_path):
    entry = ck.build_kld_manifest_entry(tmp_path / "prep", tmp_path / "m.bin",
                                        n_images=2, n_prefill=42)
    assert entry == {
        "images": [str(tmp_path / "prep" / "img_0.png"),
                   str(tmp_path / "prep" / "img_1.png")],
        "formatted_chat": str(tmp_path / "prep" / "formatted_chat.txt"),
        "tokens_in": str(tmp_path / "prep" / "tokens.bin"),
        "n_prefill": 42,
        "output_metrics": str(tmp_path / "m.bin"),
    }


def test_write_kld_manifest_is_one_json_object_per_line(tmp_path):
    p = tmp_path / "m.jsonl"
    ck.write_kld_manifest(p, [{"a": 1}, {"b": "x"}])
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [{"a": 1}, {"b": "x"}]


def test_parse_kld_done_line():
    line = "[row] DONE output_metrics=/tmp/out/metrics/000_x.bin wall_s=12.345\n"
    assert ck.parse_kld_done_line(line) == ("/tmp/out/metrics/000_x.bin", 12.345)
    assert ck.parse_kld_done_line("scored slots 100 / 1023") is None
    assert ck.parse_kld_done_line("DONE output_logits=/tmp/x.bin wall_s=1.0") is None


# ---------------------------------------------------------------------------
# postprocess_kld_result
# ---------------------------------------------------------------------------

def _make_row(tmp_path, npos=5, n_prefill=7, n_answer=5, vocab=11, seed=2):
    """A scored row: prep dir with tokens.bin + a VLMK .bin whose embedded
    targets equal input_ids[n_prefill:n_prefill+npos]."""
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=n_prefill + n_answer).astype(np.int32)
    tokens.tofile(prep_dir / "tokens.bin")

    rec = make_records(npos=npos, vocab=vocab, seed=seed)
    rec["target"] = tokens[n_prefill:n_prefill + npos]
    metrics_path = tmp_path / "000_item.bin"
    write_vlmk(metrics_path, rec, vocab=vocab, n_prefill=n_prefill)

    return {
        "idx": 0, "item_id": "item", "key": "000_item",
        "prep_dir": prep_dir, "metrics_path": metrics_path,
        "n_images": 1, "n_prefill": n_prefill, "n_answer": n_answer,
    }


def test_postprocess_converts_and_reports(tmp_path):
    row = _make_row(tmp_path)
    out = ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=3.5)
    npz_path = row["metrics_path"].with_suffix(".npz")
    assert npz_path.exists()
    assert not row["metrics_path"].exists()      # .bin deleted after conversion
    assert out[:7] == [0, "item", 1, 7, 5, 5, 11]
    assert out[7] == npz_path.stat().st_size
    assert out[9] == "OK"


def test_postprocess_accepts_capped_npos(tmp_path):
    row = _make_row(tmp_path, npos=3, n_answer=5)
    out = ck.postprocess_kld_result(row, num_eval_tokens=3, elapsed_s=0.0)
    assert out[5] == 3


def test_postprocess_accepts_cap_larger_than_answer(tmp_path):
    """num_eval_tokens > n_answer: the scorer clamps to n_answer, and the
    expected-npos check must clamp the same way."""
    row = _make_row(tmp_path, npos=5, n_answer=5)
    out = ck.postprocess_kld_result(row, num_eval_tokens=10, elapsed_s=0.0)
    assert out[5] == 5


def test_postprocess_rejects_npos_mismatch(tmp_path):
    """Scorer wrote fewer/more positions than the cap demands -> loud failure."""
    row = _make_row(tmp_path, npos=5, n_answer=5)
    with pytest.raises(ValueError, match="npos"):
        ck.postprocess_kld_result(row, num_eval_tokens=3, elapsed_s=0.0)


def test_postprocess_rejects_truncated_bin(tmp_path):
    row = _make_row(tmp_path)
    row["metrics_path"].write_bytes(row["metrics_path"].read_bytes()[:-12])
    with pytest.raises(ValueError, match="size"):
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)


def test_postprocess_rejects_target_mismatch(tmp_path):
    """Embedded targets != the row's input_ids slice = output/manifest mix-up."""
    row = _make_row(tmp_path)
    other = _make_row(tmp_path / "other_dir", seed=9)
    row["metrics_path"].unlink()
    other["metrics_path"].rename(row["metrics_path"])
    with pytest.raises(ValueError, match="target"):
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)


def test_postprocess_rejects_n_prefill_mismatch(tmp_path):
    row = _make_row(tmp_path)
    row["n_prefill"] = 8   # disagrees with the header echo (7)
    with pytest.raises(ValueError, match="n_prefill"):
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)


def test_postprocess_rejects_nonfinite_metrics(tmp_path):
    """The metrics are the ONLY artifact; a numerical blowup must fail the
    row, not be recorded OK and poison downstream means."""
    row = _make_row(tmp_path)
    rec = np.fromfile(row["metrics_path"], dtype=kio.KLD_RECORD_DT, offset=24)
    rec["kld"][1] = np.nan
    write_vlmk(row["metrics_path"], rec, n_prefill=row["n_prefill"])
    with pytest.raises(ValueError, match="non-finite"):
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)


def test_postprocess_keeps_rejected_dump_as_evidence(tmp_path):
    """A dump that failed validation stays on disk (collectors never delete
    output); the error names it and the row is recorded FAIL — any later run
    over the row refuses it as a collision."""
    row = _make_row(tmp_path)
    other = _make_row(tmp_path / "other_dir", seed=9)
    row["metrics_path"].unlink()
    other["metrics_path"].rename(row["metrics_path"])
    before = row["metrics_path"].read_bytes()
    with pytest.raises(ValueError) as excinfo:
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    # kept, but renamed out of the way so no *.bin/*.npz glob consumes it as
    # data; the row's stem prefix still makes the row a collision
    rejected = row["metrics_path"].with_name(row["metrics_path"].name + ".rejected")
    assert rejected.read_bytes() == before
    assert not row["metrics_path"].exists()
    assert not row["metrics_path"].with_suffix(".npz").exists()
    assert any("rejected output kept at" in n for n in excinfo.value.__notes__)
    assert ccommon.row_stem_index(rejected.name) == row["idx"]


def test_postprocess_keeps_valid_dump_until_converted(tmp_path):
    """A valid dump converts to .npz (this run's .bin is removed after)."""
    row = _make_row(tmp_path)
    ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    assert row["metrics_path"].with_suffix(".npz").exists()


def test_kld_float_keys_are_the_fourteen_metric_columns():
    assert ck.KLD_FLOAT_KEYS == ("kld", "reversed_kld", "js_kld", "nll_ref",
                                 "nll_cand", "entropy_ref", "entropy_cand",
                                 "ear", "ear_20", "ear_10", "ear_5",
                                 "ear_20_normalized", "ear_10_normalized", "ear_5_normalized")


def test_dump_stem_contract():
    """The collision scan and the comparators key on the f"{idx:03d}_{item}"
    stem prefix — the stem format must not drift."""
    assert ck.dump_stem(0, "test_X") == "000_test_X"
    assert ck.dump_stem(42, "a b") == "042_a b"


def _meta_args(tmp_path, **overrides):
    class A:
        ref_model = str(tmp_path / "ref.gguf")
        ref_mmproj = str(tmp_path / "ref-mm.gguf")
        cand_model = str(tmp_path / "cand.gguf")
        cand_mmproj = str(tmp_path / "cand-mm.gguf")
        dataset = "d"; subset = "s"; split = "train"; sort_by = "num_images"
        num_eval_tokens = 1024
        max_total_tokens = None
        image_min_tokens = -1; image_max_tokens = -1
        tf_chunk = 16; n_ctx = 32768; n_batch = 2048; n_ubatch = 2048
        n_gpu_layers = 99; n_threads = 8; metric_threads = 8
        flash_attn = "enabled"
        swa_full = False
    a = A()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a

def test_scorer_argv_forwards_explicit_execution_protocol(tmp_path):
    args = _meta_args(
        tmp_path, llama_vlm_kld="/bin/llama-vlm-kld",
        n_ctx=32768, n_batch=2048, n_ubatch=512, tf_chunk=2048,
        n_gpu_layers=99, n_threads=8, metric_threads=8, flash_attn="enabled",
    )
    manifest = tmp_path / "manifest.jsonl"

    assert ck._scorer_argv(args, manifest) == [
        "/bin/llama-vlm-kld",
        "--ref-model", args.ref_model, "--ref-mmproj", args.ref_mmproj,
        "--cand-model", args.cand_model, "--cand-mmproj", args.cand_mmproj,
        "--manifest", str(manifest),
        "--num-eval-tokens", "1024",
        "--image-min-tokens", "-1", "--image-max-tokens", "-1",
        "-b", "2048", "-c", "32768", "-ub", "512", "-ngl", "99",
        "--tf-chunk", "2048", "-t", "8", "--metric-threads", "8",
        "--flash-attn",
    ]


def test_scorer_argv_forwards_swa_full_only_when_requested(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    off = _meta_args(tmp_path, llama_vlm_kld="/bin/llama-vlm-kld")
    on = _meta_args(tmp_path, llama_vlm_kld="/bin/llama-vlm-kld", swa_full=True)
    assert "--swa-full" not in ck._scorer_argv(off, manifest)
    assert ck._scorer_argv(on, manifest)[-1] == "--swa-full"


def test_ensure_collect_meta_writes_then_accepts_same(tmp_path):
    meta = ck.build_collect_meta(_meta_args(tmp_path))
    ck.ensure_collect_meta(tmp_path, meta)
    assert json.loads((tmp_path / "collect_meta.json").read_text())["kind"] == "vlm_kld_metrics"
    ck.ensure_collect_meta(tmp_path, meta)   # same identity: no exit


@pytest.mark.parametrize("field,value", [
    ("cand_model", "other.gguf"),
    ("ref_mmproj", "other-mm.gguf"),
    ("tf_chunk", 1),
    ("num_eval_tokens", 512),
    ("dataset", "other-dataset"),
    ("subset", "other-subset"),
    ("swa_full", True),
])
def test_ensure_collect_meta_fails_fast_on_mismatch(tmp_path, field, value):
    ck.ensure_collect_meta(tmp_path, ck.build_collect_meta(_meta_args(tmp_path)))
    changed = ck.build_collect_meta(_meta_args(tmp_path, **{field: value}))
    with pytest.raises(SystemExit, match=field):
        ck.ensure_collect_meta(tmp_path, changed)


def test_max_total_tokens_is_an_identity_field_and_missing_field_refuses(tmp_path):
    current = ck.build_collect_meta(_meta_args(tmp_path, max_total_tokens=None))
    assert "max_total_tokens" in ck.IDENTITY_FIELDS
    assert ck.ensure_collect_meta(tmp_path, current) is True
    assert ck.ensure_collect_meta(tmp_path, current) is False
    with pytest.raises(SystemExit, match="max_total_tokens"):
        ck.ensure_collect_meta(
            tmp_path, ck.build_collect_meta(_meta_args(tmp_path, max_total_tokens=10)))
    legacy = dict(current)
    legacy.pop("max_total_tokens")
    (tmp_path / "collect_meta.json").write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(SystemExit, match="max_total_tokens: missing"):
        ck.ensure_collect_meta(tmp_path, current)


# ---------------------------------------------------------------------------
# dataset range helpers
# ---------------------------------------------------------------------------

def test_resolve_dataset_end_defaults_to_all():
    assert ck.resolve_dataset_end(None, 100) == (100, None)
    assert ck.resolve_dataset_end(-1, 100) == (100, None)


def test_resolve_dataset_end_warns_past_length():
    end, warning = ck.resolve_dataset_end(500, 100)
    assert end == 100 and "WARNING" in warning


def test_apply_dataset_limit():
    assert ck.apply_dataset_limit(0, 100, None) == 100
    assert ck.apply_dataset_limit(0, 100, -1) == 100    # -1 = the "all" sentinel
    assert ck.apply_dataset_limit(0, 100, 50) == 50
    assert ck.apply_dataset_limit(10, 100, 50) == 60
    assert ck.apply_dataset_limit(80, 100, 50) == 100   # capped by end


def test_collect_kld_rejects_nonpositive_max_total_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_kld.py",
            "--ref-model", "ref.gguf",
            "--ref-mmproj", "ref-mmproj.gguf",
            "--cand-model", "cand.gguf",
            "--cand-mmproj", "cand-mmproj.gguf",
            "--out", str(tmp_path / "out"),
            "--dataset", "some/ds",
            "--max-total-tokens", "0",
        ],
    )

    with pytest.raises(SystemExit, match="--max-total-tokens"):
        ck.main()


# ---------------------------------------------------------------------------
# main() flow (output-root policy)
# ---------------------------------------------------------------------------

class _FakeVLMDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if key == "item_id":
            return [row["item_id"] for row in self._rows]
        return self._rows[key]


def test_main_over_budget_rows_skip_and_collide_on_rerun(tmp_path, monkeypatch, capsys):
    """Over-budget rows are recorded SKIP_OVER_BUDGET once without prep; that
    record collides on a same-window rerun (refused, manifest byte-identical)
    while a disjoint window appends under the same header."""
    import cli.prep_vlm_score_from_hf as vlm_prep_lib

    out = tmp_path / "out"
    paths = {}
    for name in ("ref.gguf", "ref-mmproj.gguf", "cand.gguf", "cand-mmproj.gguf",
                 "fake_vlm_kld.py"):
        path = tmp_path / name
        path.write_bytes(b"fake")
        paths[name] = path
    ds = _FakeVLMDataset([
        {"item_id": "q1", "num_images": 1, "n_prefill_tokens": 20, "generated_tokens_len": 4},
        {"item_id": "q2", "num_images": 1, "n_prefill_tokens": 25, "generated_tokens_len": 4}])
    monkeypatch.setattr(vlm_prep_lib, "load_dataset_sorted", lambda *args: ds)
    monkeypatch.setattr(vlm_prep_lib, "load_tokenizer",
                        lambda *args: pytest.fail("over-budget row must not load tokenizer"))
    monkeypatch.setattr(vlm_prep_lib, "prep_row",
                        lambda *args: pytest.fail("over-budget row must not be prepped"))

    def argv(start, end):
        return ["collect_kld.py",
                "--ref-model", str(paths["ref.gguf"]), "--ref-mmproj", str(paths["ref-mmproj.gguf"]),
                "--cand-model", str(paths["cand.gguf"]), "--cand-mmproj", str(paths["cand-mmproj.gguf"]),
                "--llama-vlm-kld", str(paths["fake_vlm_kld.py"]),
                "--dataset", "fake-ds", "--out", str(out),
                "--start", str(start), "--end", str(end),
                "--num-eval-tokens", "2", "--max-total-tokens", "21"]

    monkeypatch.setattr("sys.argv", argv(0, 1))
    ck.main()
    manifest = out / "manifest.csv"
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["row_idx"], r["status"]) for r in rows] == [("0", "SKIP_OVER_BUDGET")]
    assert rows[0]["num_images"] == "1" and rows[0]["n_prefill"] == "20"
    assert rows[0]["n_answer"] == "4" and rows[0]["n_eval"] == "2"
    assert "skipped 1 over-budget row(s) (max-total-tokens=21)" in capsys.readouterr().err
    first = manifest.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        ck.main()
    assert "row 0: manifest.csv status=SKIP_OVER_BUDGET" in str(excinfo.value)
    assert manifest.read_bytes() == first

    monkeypatch.setattr("sys.argv", argv(1, 2))
    ck.main()
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["row_idx"], r["status"]) for r in rows] == [
        ("0", "SKIP_OVER_BUDGET"), ("1", "SKIP_OVER_BUDGET")]
    assert manifest.read_text().count("row_idx,item_id") == 1


def test_main_rejects_negative_start_and_bad_end(tmp_path, monkeypatch):
    """Finding 11 (VLM KLD lane): negative --start rejected before any
    output dir is created."""
    argv = ["collect_kld.py", "--ref-model", "r.gguf", "--ref-mmproj", "rm.gguf",
            "--cand-model", "c.gguf", "--cand-mmproj", "cm.gguf",
            "--out", str(tmp_path / "out"), "--dataset", "d"]
    monkeypatch.setattr("sys.argv", [*argv, "--start", "-1"])
    with pytest.raises(SystemExit, match="--start must be >= 0"):
        ck.main()
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr("sys.argv", [*argv, "--end", "-2"])
    with pytest.raises(SystemExit, match="--end must be -1"):
        ck.main()


def test_vlmk_version_is_part_of_a_kld_dirs_identity(tmp_path):
    """A v1 metrics dir must not be extendable with a v2 shard (the report
    would silently lose `ear`): the record version is recorded and guarded."""
    current = ck.build_collect_meta(_meta_args(tmp_path))
    assert current["vlmk_version"] == kio.VLMK_VERSION
    assert "vlmk_version" in ck.IDENTITY_FIELDS
    assert ck.ensure_collect_meta(tmp_path, current) is True
    assert ck.ensure_collect_meta(tmp_path, current) is False
    older = dict(current, vlmk_version=1)
    (tmp_path / "collect_meta.json").write_text(json.dumps(older), encoding="utf-8")
    with pytest.raises(SystemExit, match="vlmk_version"):
        ck.ensure_collect_meta(tmp_path, current)


def test_ensure_collect_meta_rejects_non_object_json(tmp_path):
    """`echo null > collect_meta.json` (or a disk-full truncation) parses fine
    but is not a mapping: exit with the policy message, not a TypeError."""
    (tmp_path / "collect_meta.json").write_text("null", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable .*expected a JSON object"):
        ck.ensure_collect_meta(tmp_path, ck.build_collect_meta(_meta_args(tmp_path)))


def test_main_n_ctx_preflight_exits_before_launching_vlm_kld(tmp_path, monkeypatch):
    """D1, KLD lane: same pre-scorer n_ctx gate; both sides of the pair get
    their own context of the same size, so the requirement is the
    single-sequence number."""
    import cli.prep_vlm_score_from_hf as vlm_prep_lib

    out = tmp_path / "out"
    paths = {}
    for name in ("ref.gguf", "ref-mmproj.gguf", "cand.gguf", "cand-mmproj.gguf",
                 "fake_vlm_kld.py"):
        p = tmp_path / name
        p.write_bytes(b"fake")
        paths[name] = p
    scorer_marker = tmp_path / "scorer_called"
    paths["fake_vlm_kld.py"].write_text(
        "#!/usr/bin/env python3\nimport pathlib\n"
        f"pathlib.Path({str(scorer_marker)!r}).write_text('x')\n")

    rows = [{"item_id": "q1", "num_images": 1, "n_prefill_tokens": 30,
             "generated_tokens_len": 6,
             "generation_model_name_or_path": "fake/model"}]
    ds = _FakeVLMDataset(rows)
    monkeypatch.setattr(vlm_prep_lib, "load_dataset_sorted", lambda *a: ds)
    monkeypatch.setattr(vlm_prep_lib, "load_tokenizer", lambda *a: object())

    def fake_prep_row(row, tok, prep_dir):
        prep_dir.mkdir(parents=True)
        (prep_dir / "scratch.txt").write_text("x")
        return {"num_images": 1, "n_prefill": 30, "n_answer": 6}

    monkeypatch.setattr(vlm_prep_lib, "prep_row", fake_prep_row)
    monkeypatch.setattr("sys.argv", [
        "collect_kld.py",
        "--ref-model", str(paths["ref.gguf"]),
        "--ref-mmproj", str(paths["ref-mmproj.gguf"]),
        "--cand-model", str(paths["cand.gguf"]),
        "--cand-mmproj", str(paths["cand-mmproj.gguf"]),
        "--llama-vlm-kld", str(paths["fake_vlm_kld.py"]),
        "--out", str(out), "--dataset", "fake-ds",
        "--start", "0", "--end", "1",
        "--num-eval-tokens", "4", "--n-ctx", "33"])

    with pytest.raises(SystemExit, match="suggested --n-ctx 34"):
        ck.main()

    assert not scorer_marker.exists()
    assert list((out / "_prep").iterdir()) == []
    with open(out / "manifest.csv", newline="") as f:
        assert list(csv.DictReader(f)) == []


def test_sort_desc_is_recorded_but_not_an_identity_field(tmp_path, monkeypatch):
    """Same policy as collect_logits: recorded, guarded by the comparators
    (legacy default False), not an identity field (existing dirs stay
    extendable)."""
    paths = {}
    for name in ("r.gguf", "rp.gguf", "c.gguf", "cp.gguf"):
        p = tmp_path / name; p.write_bytes(b"x"); paths[name] = p
    base = ["collect_kld.py", "--ref-model", str(paths["r.gguf"]), "--ref-mmproj", str(paths["rp.gguf"]),
            "--cand-model", str(paths["c.gguf"]), "--cand-mmproj", str(paths["cp.gguf"]),
            "--out", str(tmp_path / "out"), "--dataset", "d"]
    monkeypatch.setattr("sys.argv", base + ["--sort-desc"])
    args = ck.parse_args()
    assert args.sort_desc is True
    assert ck.build_collect_meta(args)["sort_desc"] is True
    assert "sort_desc" not in ck.IDENTITY_FIELDS
    monkeypatch.setattr("sys.argv", base)
    default_args = ck.parse_args()
    assert Path(default_args.llama_vlm_kld) == Path(__file__).resolve().parents[3] / "build/bin/llama-vlm-kld"
    assert ck.build_collect_meta(default_args)["sort_desc"] is False
    assert default_args.n_gpu_layers == 99
    assert default_args.n_threads == -1
    assert default_args.metric_threads == -1
    assert default_args.flash_attn == "auto"
    assert default_args.swa_full is False
    assert ck.build_collect_meta(default_args)["swa_full"] is False
    assert "swa_full" in ck.IDENTITY_FIELDS


# ---------------------------------------------------------------------------
# llama.cpp-generated rows: generator n_past vs scorer n_past_actual
# ---------------------------------------------------------------------------

def _make_llamacpp_row(tmp_path, n_past_actual, n_past_expected):
    row = _make_row(tmp_path)
    rec = make_records(npos=5, vocab=11, seed=2)
    tokens = np.fromfile(row["prep_dir"] / "tokens.bin", dtype=np.int32)
    rec["target"] = tokens[7:12]
    write_vlmk(row["metrics_path"], rec, vocab=11, n_prefill=7, n_past_actual=n_past_actual)
    row["n_past_expected"] = n_past_expected
    return row


def test_postprocess_llamacpp_row_matching_n_past_is_clean(tmp_path):
    row = _make_llamacpp_row(tmp_path, n_past_actual=41, n_past_expected=41)
    out = ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    assert out[9] == "OK"
    assert row["n_past_mismatch"] is None


def test_postprocess_llamacpp_row_n_past_drift_warns_by_default(tmp_path, capsys):
    row = _make_llamacpp_row(tmp_path, n_past_actual=43, n_past_expected=41)
    out = ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    assert out[9] == "OK"                       # scored, not rejected
    assert row["n_past_mismatch"] == (41, 43)
    assert "n_past_actual=43 != generator n_past_expected=41" in capsys.readouterr().err


def test_postprocess_llamacpp_row_n_past_drift_rejects_when_required(tmp_path):
    row = _make_llamacpp_row(tmp_path, n_past_actual=43, n_past_expected=41)
    with pytest.raises(ValueError, match="n_past_actual=43"):
        ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0,
                                  require_n_past_match=True)
    assert row["metrics_path"].with_name(row["metrics_path"].name + ".rejected").exists()


def test_postprocess_llamacpp_row_without_n_past_actual_skips_check(tmp_path):
    row = _make_llamacpp_row(tmp_path, n_past_actual=0, n_past_expected=41)
    out = ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    assert out[9] == "OK"
    assert "n_past_mismatch" not in row


def test_postprocess_vllm_rows_ignore_n_past(tmp_path):
    row = _make_row(tmp_path)
    out = ck.postprocess_kld_result(row, num_eval_tokens=-1, elapsed_s=0.0)
    assert out[9] == "OK"
    assert "n_past_mismatch" not in row


def test_build_kld_manifest_entry_uses_prep_image_files(tmp_path):
    entry = ck.build_kld_manifest_entry(tmp_path / "prep", tmp_path / "m.bin",
                                        n_images=2, n_prefill=42,
                                        image_files=["img_0.jpg", "img_1.png"])
    assert entry["images"] == [str(tmp_path / "prep" / "img_0.jpg"),
                               str(tmp_path / "prep" / "img_1.png")]


def test_build_kld_manifest_entry_add_special_only_when_true(tmp_path):
    base = ck.build_kld_manifest_entry(tmp_path / "p", tmp_path / "m.bin", n_images=1, n_prefill=3)
    assert "add_special" not in base
    entry = ck.build_kld_manifest_entry(tmp_path / "p", tmp_path / "m.bin", n_images=1, n_prefill=3,
                                        add_special=True)
    assert entry["add_special"] is True


def test_resolve_image_token_budget_distinguishes_omitted_from_explicit_minus_one(capsys):
    import types
    cfg = json.dumps({"engine": "llama.cpp", "image_min_tokens": 64, "image_max_tokens": 16384})
    ds = [{"generation_engine": "llama.cpp", "image_processor_config": cfg}]

    # both flags omitted -> adopt the dataset's recorded budget
    args = types.SimpleNamespace(image_min_tokens=None, image_max_tokens=None)
    ck.resolve_image_token_budget(args, ds)
    assert (args.image_min_tokens, args.image_max_tokens) == (64, 16384)
    assert "adopting its recorded image token budget" in capsys.readouterr().err

    # explicit -1 -1 (mmproj metadata) is NOT the same as omitted: kept verbatim
    args = types.SimpleNamespace(image_min_tokens=-1, image_max_tokens=-1)
    ck.resolve_image_token_budget(args, ds)
    assert (args.image_min_tokens, args.image_max_tokens) == (-1, -1)
    assert "adopting" not in capsys.readouterr().err

    # one explicit value disables adoption; the omitted one falls back to -1
    args = types.SimpleNamespace(image_min_tokens=8, image_max_tokens=None)
    ck.resolve_image_token_budget(args, ds)
    assert (args.image_min_tokens, args.image_max_tokens) == (8, -1)

    # non-llama.cpp rows: omitted -> -1 -1, nothing adopted
    args = types.SimpleNamespace(image_min_tokens=None, image_max_tokens=None)
    ck.resolve_image_token_budget(args, [{"generation_engine": None, "image_processor_config": "{}"}])
    assert (args.image_min_tokens, args.image_max_tokens) == (-1, -1)

    # empty dataset: omitted -> -1 -1
    args = types.SimpleNamespace(image_min_tokens=None, image_max_tokens=None)
    ck.resolve_image_token_budget(args, [])
    assert (args.image_min_tokens, args.image_max_tokens) == (-1, -1)
