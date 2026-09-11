"""Tests for saved_metrics_paired_compare.py: per-item score reconstruction
from VLMK metric dumps, the two-dir alignment guards (incl. the bit-identical
reference consistency check), and the end-to-end report.

Hermetic: synthesizes metrics dirs via kld_metrics_io; no GPU, no models.
Run from tools/gap:
    python3 -m pytest tests/test_saved_metrics_paired_compare.py -v
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import lib.kld_metrics_io as kio
from stats.contracts import POOLED_LADDER
import stats.cli.saved_metrics_paired_compare as smpc
from stats import collection_io as paired_io

from fakes import write_vlmk, completed_collection, EXECUTION_IDENTITY


BASE_META = {
    "execution_identity": EXECUTION_IDENTITY,
    "kind": "vlm_kld_metrics",
    "ref_model": "/m/ref.gguf", "ref_mmproj": "/m/ref-mm.gguf",
    "cand_model": "/m/cand-a.gguf", "cand_mmproj": "/m/ref-mm.gguf",
    "dataset": "d", "subset": "s", "split": "train", "sort_by": "num_images",
    "num_eval_tokens": -1, "image_min_tokens": -1, "image_max_tokens": -1,
    "tf_chunk": 16, "n_batch": 2048, "n_ubatch": 2048, "swa_full": False,
    "media_wrapper": "stripped",
    "dataset_content_hash": "ds-v2:same",
    "ref_model_fingerprint": "gguf-sampled-v1:refsame",
    "ref_mmproj_fingerprint": "gguf-sampled-v1:mmsame",
}


def _make_item(npos=4, vocab=11, seed=0, ref_seed=100,
               version=kio.VLMK_VERSION):
    """One item's records: the reference-side columns derive from ref_seed
    only, so two dirs built with the same ref_seed are bit-identical on the
    reference side (the shared-reference premise). version=1 builds genuine
    40-byte legacy records without the ear column."""
    ref_rng = np.random.default_rng(ref_seed)
    cand_rng = np.random.default_rng(seed)
    rec = np.zeros(npos, dtype=kio.kld_record_dt(version))
    rec["nll_ref"] = ref_rng.uniform(0.1, 2.0, npos).astype(np.float32)
    rec["entropy_ref"] = ref_rng.uniform(0.1, 3.0, npos).astype(np.float32)
    rec["argmax_ref"] = ref_rng.integers(0, vocab, npos).astype(np.int32)
    rec["target"] = ref_rng.integers(0, vocab, npos).astype(np.int32)
    for k in ("kld", "reversed_kld", "js_kld"):
        rec[k] = cand_rng.uniform(0.0, 0.2, npos).astype(np.float32)
    rec["nll_cand"] = cand_rng.uniform(0.1, 2.5, npos).astype(np.float32)
    rec["entropy_cand"] = cand_rng.uniform(0.1, 3.0, npos).astype(np.float32)
    if "ear" in rec.dtype.names:
        rec["ear"] = cand_rng.uniform(0.5, 1.0, npos).astype(np.float32)
    rec["argmax_cand"] = cand_rng.integers(0, vocab, npos).astype(np.int32)
    return rec


def _write_metrics_dir(d: Path, items: dict, meta: dict | None = BASE_META,
                       n_past_actual=0, skipped=()):
    (d / "metrics").mkdir(parents=True)
    for key, rec in items.items():
        if int(key.split("_", 1)[0]) in skipped:
            continue
        bin_path = d / "metrics" / f"{key}.bin"
        version = next(v for v in kio.VLMK_SUPPORTED_VERSIONS
                       if rec.dtype == kio.kld_record_dt(v))
        write_vlmk(bin_path, rec, vocab=11, n_prefill=7, version=version,
                   n_past_actual=n_past_actual)
        kio.convert_kld_bin_to_npz(bin_path, bin_path.with_suffix(".npz"))
        bin_path.unlink()
    completed_collection(d, items, skipped)
    if meta is not None:
        (d / "collect_meta.json").write_text(json.dumps(meta))


def _make_pair(tmp_path, n_items=3, meta_b_overrides=None, mutate_b=None,
               npos_list=None, version=kio.VLMK_VERSION, version_b=None, n_items_b=None, skipped_a=(), skipped_b=()):
    """Two metrics dirs sharing bit-identical reference columns per item.
    mutate_b(key, records) may edit candidate-b's records before writing;
    npos_list gives per-item position counts (default: uniform 4);
    version=1 builds legacy pre-ear dirs (version_b overrides b's alone, for
    mixed-version pairs)."""
    version_b = version if version_b is None else version_b
    items_a, items_b = {}, {}
    for i in range(n_items):
        npos = npos_list[i] if npos_list else 4
        key = f"{i:03d}_item{i}"
        items_a[key] = _make_item(npos=npos, seed=10 + i, ref_seed=100 + i,
                                  version=version)
        rb = _make_item(npos=npos, seed=20 + i, ref_seed=100 + i,
                        version=version_b)
        if mutate_b:
            mutate_b(key, rb)
        items_b[key] = rb
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    meta_b = dict(BASE_META, cand_model="/m/cand-b.gguf",
                  **(meta_b_overrides or {}))
    if n_items_b is not None:
        items_b = dict(list(items_b.items())[:n_items_b])
    _write_metrics_dir(a_dir, items_a, skipped=skipped_a)
    _write_metrics_dir(b_dir, items_b, meta_b, skipped=skipped_b)
    return a_dir, b_dir


def test_score_item_rejects_post_prefill_position_drift(tmp_path):
    """The two dirs' n_prefill headers agree by construction (both echo the
    HF ground-truth length from the manifest), so they cannot see a
    vision-token change. n_past_actual is llama.cpp's own count and does
    move; it is the content-level backstop for the image-bounds meta guard."""
    rec = _make_item(npos=4, seed=1, ref_seed=2)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    _write_metrics_dir(a_dir, {"000_x": rec}, n_past_actual=512)
    _write_metrics_dir(b_dir, {"000_x": rec}, dict(BASE_META,
                                                   cand_model="/m/b.gguf"),
                       n_past_actual=640)
    with pytest.raises(smpc.AlignmentError,
                       match="post-prefill position count"):
        paired_io.score_item("000_x", a_dir, b_dir, -1)


def test_score_item_skips_the_position_guard_on_legacy_dumps(tmp_path):
    """0 means "not recorded", not "zero positions": a dump written before the
    field existed must not be compared against one that has it."""
    rec = _make_item(npos=4, seed=1, ref_seed=2)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    _write_metrics_dir(a_dir, {"000_x": rec}, n_past_actual=0)
    _write_metrics_dir(b_dir, {"000_x": rec}, dict(BASE_META,
                                                   cand_model="/m/b.gguf"),
                       n_past_actual=640)
    assert paired_io.score_item("000_x", a_dir, b_dir, -1)[4] == 4


def _run_main(a_dir, b_dir, tmp_path, *extra):
    out = tmp_path / "report.md"
    js = tmp_path / "report.json"
    argv = ["--candidate-a", str(a_dir), "--candidate-b", str(b_dir),
            "--out", str(out), "--output-json", str(js),
            "--bootstrap-iters", "800", "--seed", "7", *extra]
    rc = smpc.main(argv)
    return rc, out, js


# ---------------------------------------------------------------------------
# per-item score reconstruction
# ---------------------------------------------------------------------------

def test_side_scores_closed_form():
    # The nll columns must have DIFFERENT means (ref: (ln2+ln4)/2 = ln(8)/2,
    # cand: ln4), so an aggregation that swapped nll_ref/nll_cand cannot
    # survive by symmetry. (Careful when editing: geometric pairs like
    # ref=[ln2, ln8] vs cand=[ln4, ln4] have EQUAL means and hide the swap.)
    rec = np.zeros(2, dtype=kio.KLD_RECORD_DT)
    rec["kld"] = [0.5, 1.5]
    rec["reversed_kld"] = [0.2, 0.4]
    rec["js_kld"] = [0.1, 0.3]
    rec["nll_ref"] = [np.log(2.0), np.log(4.0)]    # p_ref(target) = 1/2, 1/4
    rec["nll_cand"] = [np.log(4.0), np.log(4.0)]   # p_cand(target) = 1/4, 1/4
    rec["entropy_cand"] = [1.0, 3.0]
    rec["argmax_ref"] = [3, 5]
    rec["argmax_cand"] = [3, 6]
    rec["ear"] = [0.9, 0.7]
    m = {k: rec[k] for k in kio.KLD_METRIC_KEYS}
    s, tok = paired_io._side_scores(m, keep=2)
    assert s["kld"] == pytest.approx(1.0)
    assert s["reversed_kld"] == pytest.approx(0.3)
    assert s["js_kld"] == pytest.approx(0.2)
    assert s["ear"] == pytest.approx(0.8)
    # the retained per-token columns feed the pooled distribution ladders
    assert sorted(tok) == ["dp", "ear", "kld", "target"]
    np.testing.assert_allclose(tok["kld"], [0.5, 1.5])
    np.testing.assert_allclose(tok["ear"], [0.9, 0.7])
    # dp is SIGNED (p_cand - p_ref at the target, pp): 1/4-1/2 then 1/4-1/4
    np.testing.assert_allclose(tok["dp"], [-25.0, 0.0], atol=1e-5)
    assert s["mean_dp"] == pytest.approx(-12.5, abs=1e-5)
    assert s["nll_ref"] == pytest.approx((np.log(2.0) + np.log(4.0)) / 2)
    assert not [k for k in s if k.startswith("kld_")]
    assert s["nll"] == pytest.approx(np.log(4.0))           # the CAND mean ...
    ref_mean = (np.log(2.0) + np.log(4.0)) / 2
    assert abs(s["nll"] - ref_mean) > 1e-3                   # ... not the ref mean
    assert s["same_top_rate"] == pytest.approx(0.5)
    assert s["entropy"] == pytest.approx(2.0)
    # dp = (1/4 - 1/2)*100 = -25pp then (1/4 - 1/4)*100 = 0pp
    # mse = (625 + 0)/2 = 312.5
    assert s["mse_dp"] == pytest.approx(312.5, rel=1e-6)


def test_side_scores_respects_cap():
    rec = np.zeros(4, dtype=kio.KLD_RECORD_DT)
    rec["kld"] = [1.0, 1.0, 9.0, 9.0]
    m = {k: rec[k] for k in kio.KLD_METRIC_KEYS}
    assert paired_io._side_scores(m, keep=2)[0]["kld"] == pytest.approx(1.0)
    assert paired_io._side_scores(m, keep=4)[0]["kld"] == pytest.approx(5.0)
    np.testing.assert_allclose(paired_io._side_scores(m, keep=2)[1]["kld"], [1.0, 1.0])


def test_side_scores_zero_keep_is_nonfinite():
    rec = np.zeros(0, dtype=kio.KLD_RECORD_DT)
    m = {k: rec[k] for k in kio.KLD_METRIC_KEYS}
    s, tok = paired_io._side_scores(m, keep=0)
    assert not any(np.isfinite(v) for v in s.values())
    assert sorted(tok) == ["dp", "ear", "kld", "target"]
    assert all(col.size == 0 for col in tok.values())


# ---------------------------------------------------------------------------
# meta alignment guard
# ---------------------------------------------------------------------------

def test_meta_alignment_passes_and_warns_nothing_on_clean_pair():
    a = dict(BASE_META)
    b = dict(BASE_META, cand_model="/m/cand-b.gguf")
    assert paired_io.check_kld_meta_alignment(a, b) == []


@pytest.mark.parametrize("field,value", [
    ("tf_chunk", 1),
    ("ref_model", "/m/other-ref.gguf"),
    ("num_eval_tokens", 512),
    ("media_wrapper", "doubled"),
    ("swa_full", True),
])
def test_meta_alignment_hard_fails_on_mismatch(field, value):
    a = dict(BASE_META)
    b = dict(BASE_META, cand_model="/m/cand-b.gguf", **{field: value})
    with pytest.raises(smpc.AlignmentError, match=field):
        paired_io.check_kld_meta_alignment(a, b)


def test_meta_alignment_guards_sort_desc_with_legacy_default_false():
    # sort_desc mismatch = same stems index different dataset orders; must be
    # NAMED, not surfaced downstream as mysterious per-item drops.
    a = dict(BASE_META, sort_desc=False)
    b = dict(BASE_META, cand_model="/m/cand-b.gguf", sort_desc=True)
    with pytest.raises(smpc.AlignmentError, match="sort_desc"):
        paired_io.check_kld_meta_alignment(a, b)

    # Legacy dirs (pre-F2.8 / all VLM dirs) lack the field entirely: missing
    # normalizes to False, so legacy-vs-explicit-ascending pairs stay clean.
    legacy = {k: v for k, v in BASE_META.items() if k != "sort_desc"}
    explicit_asc = dict(BASE_META, cand_model="/m/cand-b.gguf", sort_desc=False)
    assert "sort_desc" not in legacy
    assert paired_io.check_kld_meta_alignment(legacy, explicit_asc) == []


def test_meta_alignment_swa_full_has_no_legacy_default():
    """Dirs written before swa_full was recorded may have used either KV
    cache mode, so a legacy dir pairs only with another legacy dir."""
    legacy_a = {k: v for k, v in BASE_META.items() if k != "swa_full"}
    legacy_b = dict(legacy_a, cand_model="/m/cand-b.gguf")
    assert paired_io.check_kld_meta_alignment(legacy_a, legacy_b) == []

    recorded = dict(BASE_META, cand_model="/m/cand-b.gguf", swa_full=False)
    with pytest.raises(smpc.AlignmentError, match="swa_full"):
        paired_io.check_kld_meta_alignment(legacy_a, recorded)


def test_meta_alignment_guards_max_total_tokens_with_legacy_none():
    a = dict(BASE_META, max_total_tokens=None)
    b_legacy = {k: v for k, v in BASE_META.items() if k != "max_total_tokens"}
    b_legacy["cand_model"] = "/m/cand-b.gguf"

    assert "max_total_tokens" in paired_io.META_MUST_MATCH
    assert paired_io.check_kld_meta_alignment(a, b_legacy) == []

    b_budgeted = dict(b_legacy, max_total_tokens=4096)
    with pytest.raises(smpc.AlignmentError, match="max_total_tokens"):
        paired_io.check_kld_meta_alignment(a, b_budgeted)


def test_meta_alignment_warns_on_same_candidate_and_missing_meta():
    a = dict(BASE_META)
    warnings = paired_io.check_kld_meta_alignment(a, dict(a))
    assert any("SAME model" in w for w in warnings)
    warnings = paired_io.check_kld_meta_alignment(a, None)
    assert any("collect_meta.json missing" in w for w in warnings)


def test_meta_alignment_warns_on_wrong_kind():
    a = dict(BASE_META, kind="something_else")
    b = dict(BASE_META, cand_model="/m/b.gguf", kind="something_else")
    assert any("collect_meta kind" in w for w in paired_io.check_kld_meta_alignment(a, b))


def test_meta_alignment_hard_fails_on_kind_mix():
    a = dict(BASE_META)
    b = dict(BASE_META, cand_model="/m/b.gguf", kind="llm_kld_metrics")
    with pytest.raises(smpc.AlignmentError, match="kind"):
        paired_io.check_kld_meta_alignment(a, b)


def test_meta_alignment_allows_llm_without_vlm_fields_and_ignores_media_wrapper():
    a = {
        "kind": "llm_kld_metrics",
        "ref_model": "/m/ref.gguf",
        "cand_model": "/m/cand-a.gguf",
        "dataset": "d",
        "subset": "s",
        "split": "train",
        "sort_by": "",
        "num_eval_tokens": -1,
        "tf_chunk": 16,
        "n_batch": 2048,
        "n_ubatch": 2048,
        "media_wrapper": "stripped",
        "dataset_content_hash": "ds-v2:same",
        "ref_model_fingerprint": "gguf-sampled-v1:refsame",
    }
    b = dict(a, cand_model="/m/cand-b.gguf", media_wrapper="doubled")

    assert paired_io.check_kld_meta_alignment(a, b) == []


@pytest.mark.parametrize("field", [
    "dataset_content_hash", "ref_model_fingerprint", "ref_mmproj_fingerprint",
])
def test_meta_alignment_content_fingerprints_three_tier(field):
    """codex round-2 finding 2: the shared-reference premise rides on these
    fields -- the stored nll_ref/entropy_ref/argmax_ref columns are not a
    unique fingerprint of the reference distribution. Present-and-different
    hard-fails; a legacy dir missing the field warns instead."""
    a = dict(BASE_META)
    b = dict(BASE_META, cand_model="/m/cand-b.gguf")
    assert paired_io.check_kld_meta_alignment(a, b) == []

    b_drifted = dict(b, **{field: "other-value"})
    with pytest.raises(smpc.AlignmentError, match=field):
        paired_io.check_kld_meta_alignment(a, b_drifted)

    b_legacy = {k: v for k, v in b.items() if k != field}
    warnings = paired_io.check_kld_meta_alignment(a, b_legacy)
    assert any(field in w and "candidate-b" in w for w in warnings)


def test_meta_alignment_llm_kind_skips_mmproj_fingerprint():
    # ref_mmproj_fingerprint only exists on the VLM lane; an LLM pair must
    # not be warned about a field its collector never writes.
    a = dict(BASE_META, kind="llm_kld_metrics")
    for m in (a,):
        m.pop("ref_mmproj", None)
        m.pop("cand_mmproj", None)
        m.pop("image_min_tokens", None)
        m.pop("image_max_tokens", None)
        m.pop("media_wrapper", None)
        m.pop("ref_mmproj_fingerprint", None)
    b = dict(a, cand_model="/m/cand-b.gguf")
    assert paired_io.check_kld_meta_alignment(a, b) == []


def test_meta_alignment_llm_same_model_warning_omits_mmproj():
    a = {
        "kind": "llm_kld_metrics",
        "ref_model": "/m/ref.gguf",
        "cand_model": "/m/same.gguf",
        "dataset": "d",
        "subset": "s",
        "split": "train",
        "sort_by": "",
        "num_eval_tokens": -1,
        "tf_chunk": 16,
        "n_batch": 2048,
        "n_ubatch": 2048,
    }
    b = dict(a)

    warnings = paired_io.check_kld_meta_alignment(a, b)

    assert any("SAME model" in w for w in warnings)
    assert not any("mmproj" in w for w in warnings)


def test_meta_alignment_warns_when_candidate_equals_reference():
    a = dict(BASE_META, cand_model=BASE_META["ref_model"],
             cand_mmproj=BASE_META["ref_mmproj"])
    b = dict(BASE_META, cand_model="/m/cand-b.gguf")
    assert any("equals the reference" in w
               for w in paired_io.check_kld_meta_alignment(a, b))


# ---------------------------------------------------------------------------
# item matching
# ---------------------------------------------------------------------------

def test_find_metric_items_reports_drops(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    (b_dir / "metrics" / "002_item2.npz").unlink()
    matched, drops = paired_io.find_metric_items(a_dir, b_dir)
    assert matched == ["000_item0", "001_item1"]
    assert drops == {"candidate-a": [], "candidate-b": ["002_item2"]}


def test_find_metric_items_numeric_stem_order(tmp_path):
    # find_metric_items only globs stems, so bare .npz files suffice. String
    # sort would return [1000, 1001, 998, 999] past row 999 (3-digit padding).
    for role in ("a", "b"):
        d = tmp_path / role / "metrics"
        d.mkdir(parents=True)
        for key in ("998_w", "999_x", "1000_y", "1001_z"):
            (d / f"{key}.npz").touch()
    matched, drops = paired_io.find_metric_items(tmp_path / "a", tmp_path / "b")
    assert matched == ["998_w", "999_x", "1000_y", "1001_z"]
    assert drops == {"candidate-a": [], "candidate-b": []}


# ---------------------------------------------------------------------------
# per-item guards (score_item)
# ---------------------------------------------------------------------------

def test_score_item_consistent_pair_has_no_drift(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    sa, sb, tok_a, tok_b, keep, finite, drift, versions = paired_io.score_item(
        "000_item0", a_dir, b_dir, num_eval_tokens=-1)
    assert drift is None and finite and keep == 4
    assert sa != sb   # different candidates
    assert versions == (kio.VLMK_VERSION, kio.VLMK_VERSION)


@pytest.mark.parametrize("col", ["nll_ref", "entropy_ref", "argmax_ref"])
def test_score_item_detects_drift_in_each_ref_column(tmp_path, col):
    """nll_ref equality does NOT imply argmax_ref equality (near-tie top-1
    flips) — every reference column must be checked independently."""
    def bump(key, rec):
        if rec[col].dtype.kind == "f":
            rec[col][0] += np.float32(1e-4)
        else:
            rec[col][0] = (rec[col][0] + 1) % 11
    a_dir, b_dir = _make_pair(tmp_path, mutate_b=bump)
    *_rest, drift, _versions = paired_io.score_item(
        "000_item0", a_dir, b_dir, num_eval_tokens=-1)
    assert drift is not None and col in drift["msg"]
    if col == "argmax_ref":
        assert drift["argmax_flips"] == 1
    else:
        assert drift["argmax_flips"] == 0


def test_score_item_hard_fails_on_vocab_mismatch(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    rec = _make_item(seed=20, ref_seed=100)
    p = b_dir / "metrics" / "000_item0"
    write_vlmk(p.with_suffix(".bin"), rec, vocab=12, n_prefill=7)
    kio.convert_kld_bin_to_npz(p.with_suffix(".bin"), p.with_suffix(".npz"))
    with pytest.raises(smpc.AlignmentError, match="vocab"):
        paired_io.score_item("000_item0", a_dir, b_dir, num_eval_tokens=-1)


def test_score_item_hard_fails_on_target_mismatch(tmp_path):
    def flip_target(key, rec):
        rec["target"][0] = (rec["target"][0] + 1) % 11
    a_dir, b_dir = _make_pair(tmp_path, mutate_b=flip_target)
    with pytest.raises(smpc.AlignmentError, match="target"):
        paired_io.score_item("000_item0", a_dir, b_dir, num_eval_tokens=-1)


def test_score_item_hard_fails_on_npos_mismatch(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    short = _make_item(npos=2, seed=20, ref_seed=100)
    p = b_dir / "metrics" / "000_item0"
    write_vlmk(p.with_suffix(".bin"), short, vocab=11, n_prefill=7)
    kio.convert_kld_bin_to_npz(p.with_suffix(".bin"), p.with_suffix(".npz"))
    with pytest.raises(smpc.AlignmentError, match="npos"):
        paired_io.score_item("000_item0", a_dir, b_dir, num_eval_tokens=-1)


# ---------------------------------------------------------------------------
# end-to-end main()
# ---------------------------------------------------------------------------

def test_main_writes_report_and_json(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    md = out.read_text()
    assert "# Paired comparison:" in md
    assert "on-the-fly VLMK metric dumps" in md
    # VLMK metric dumps store no logits: the report must say so instead of
    # claiming a logits storage format.
    assert "Logits format" not in md
    assert "no logits were stored" in md      # the report's own trailing note
    result = json.loads(js.read_text())
    assert result["n_items"] == 3
    assert result["metrics_source"] == "vlm_kld_metrics"
    assert set(result["metrics"]) >= {"nll", "kld", "ppl", "ppl_ratio", "rms_dp"}
    assert result["metrics"]["kld"]["item_weighted"]["decision"]["verdict"]
    assert result["candidate_metrics"]["entropy"]["candidate_a"]["label"] == "cand-a.gguf"
    assert result["alignment"]["n_ref_drift"] == 0


def test_main_distinguishes_gpu_collection_from_cpu_statistics(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    for d in (a_dir, b_dir):
        meta_path = d / "collect_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["gpu_name"] = "Unit Test GPU"
        meta_path.write_text(json.dumps(meta))

    rc, out, js = _run_main(a_dir, b_dir, tmp_path)

    assert rc == 0
    md = out.read_text()
    assert "stored model-generated answer trajectory" in md
    assert "GPU recorded on metrics-collection host: Unit Test GPU" in md
    assert "Paired statistics backend: cpu; jobs=1" in md
    assert "Resolved backend: cpu" not in md
    execution = json.loads(js.read_text())["execution"]
    assert execution["metrics_collection"]["gpu_by_candidate"] == {
        "candidate-a": "Unit Test GPU", "candidate-b": "Unit Test GPU"}


def test_main_with_llm_meta_persists_llm_metrics_source(tmp_path):
    llm_meta_a = {
        "execution_identity": EXECUTION_IDENTITY,
        "kind": "llm_kld_metrics",
        "ref_model": "/m/ref.gguf",
        "cand_model": "/m/cand-a.gguf",
        "dataset": "d",
        "subset": "s",
        "split": "train",
        "sort_by": "",
        "num_eval_tokens": -1,
        "tf_chunk": 16,
        "n_batch": 2048,
        "n_ubatch": 2048,
    }
    llm_meta_b = dict(llm_meta_a, cand_model="/m/cand-b.gguf")
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    (a_dir / "collect_meta.json").write_text(json.dumps(llm_meta_a))
    (b_dir / "collect_meta.json").write_text(json.dumps(llm_meta_b))

    rc, out, js = _run_main(a_dir, b_dir, tmp_path)

    assert rc == 0
    md = out.read_text()
    result = json.loads(js.read_text())
    assert result["metrics_source"] == "llm_kld_metrics"
    assert result["inputs"]["kind"] == "llm_kld_metrics"
    assert "collect_llm_kld.py" in md


def test_main_reports_ear_and_tail_metrics_on_v2_dumps(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    md = out.read_text()
    result = json.loads(js.read_text())
    assert "| ear" in md
    assert result["metrics"]["ear"]["score_direction"] == "higher_is_better"
    manual = []
    for p in sorted((a_dir / "metrics").glob("*.npz")):
        with np.load(p) as z:
            manual.append(float(z["ear"].astype(np.float64).mean()))
    got = result["metrics"]["ear"]["item_weighted"]["baseline_mean"]
    assert got == pytest.approx(np.mean(manual), abs=1e-12)
    # tails are a DESCRIPTIVE pooled-token block, not verdict rows
    assert not [k for k in result["metrics"] if k.startswith("kld_")]
    pooled = result["pooled_token_distribution"]
    assert sorted(pooled) == ["dp", "ear", "kld"]
    # the reference's own likelihood is reportable at last (PPL(base))
    ref_m = result["reference_metrics"]
    assert ref_m["max_abs_side_difference"] == pytest.approx(0.0, abs=1e-12)
    assert ref_m["ppl"]["item_weighted"] == pytest.approx(
        math.exp(ref_m["nll"]["item_weighted_mean"]))
    assert "## Reference corpus likelihood" in md and "PPL(reference)" in md
    # and the SIGNED mean delta-p, which mse_dp/rms_dp cannot show
    assert "mean_dp" in result["candidate_metrics"]
    for block in pooled.values():
        assert block["inference"] == "none"
        for side in ("candidate_a", "candidate_b"):
            names = [L["name"] for L in block[side]["levels"]]
            assert names == [n for n, _ in POOLED_LADDER]
            # every level names the token it came from
            assert all("item_key" in L["witness"] for L in block[side]["levels"])
    assert "NO bootstrap" in md
    # the pooled kld max must equal the max over every token of every dump
    every = np.concatenate([
        np.load(p)["kld"].astype(np.float64)
        for p in sorted((a_dir / "metrics").glob("*.npz"))])
    by_name = {L["name"]: L["value"]
               for L in pooled["kld"]["candidate_a"]["levels"]}
    assert by_name["max"] == pytest.approx(float(every.max()))
    assert by_name["p99"] == pytest.approx(float(np.quantile(every, 0.99)))


def test_main_drops_ear_with_persisted_warning_on_v1_dumps(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, version=1)
    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    md = out.read_text()
    result = json.loads(js.read_text())
    assert "ear" not in result["metrics"]
    assert "| ear " not in md
    assert "VLMK v1" in md                       # persisted warning
    # the kld and dp ladders ride on columns v1 dumps do have; the EAR ladder
    # needs the v2 per-token column and is simply absent
    assert sorted(result["pooled_token_distribution"]) == ["dp", "kld"]


def test_main_explicit_ear_request_on_v1_dumps_exits(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, version=1)
    with pytest.raises(SystemExit, match="ear.*candidate-a \\(VLMK v1\\), candidate-b"):
        _run_main(a_dir, b_dir, tmp_path, "--metrics", "ear", "kld")


def test_main_mixed_version_pair_names_the_v1_side_precisely(tmp_path):
    """candidate-a at v1, candidate-b at v2: accepted (non-ear metrics are
    comparable), but the persisted warnings must say exactly which dir lacks
    the column — not blame 'the input dumps' wholesale — and the JSON must
    record both dirs' versions."""
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, version=1,
                              version_b=kio.VLMK_VERSION)
    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    md = out.read_text()
    result = json.loads(js.read_text())
    assert "ear" not in result["metrics"]
    assert result["vlmk_versions"] == {"candidate_a": [1],
                                       "candidate_b": [kio.VLMK_VERSION]}
    warnings = "\n".join(result["alignment"]["warnings"])
    assert "candidate-a (VLMK v1)" in warnings
    assert "candidate-b (VLMK" not in warnings       # b is not blamed
    assert "mixed VLMK versions" in warnings
    assert f"candidate-b dumps are v{kio.VLMK_VERSION}" in warnings
    assert "candidate-a (VLMK v1)" in md


def test_side_scores_keep_zero_omits_ear_for_v1_records():
    """The npos == 0 NaN dict must carry the same key set as the scored
    branch: no phantom 'ear' for a dump without the column."""
    def columns(rec):   # load_kld_metrics' shape: dict of 1-d arrays
        return {k: np.ascontiguousarray(rec[k]) for k in rec.dtype.names}
    m1 = columns(_make_item(npos=2, seed=1, ref_seed=2, version=1))
    m2 = columns(_make_item(npos=2, seed=1, ref_seed=2))
    nan1, _ = paired_io._side_scores(m1, 0)
    nan2, _ = paired_io._side_scores(m2, 0)
    assert "ear" not in nan1 and "ear" in nan2
    assert not (set(kio.VERSIONED_METRIC_KEYS) & set(nan1))
    assert set(nan1) | set(kio.VERSIONED_METRIC_KEYS) == set(nan2)
    assert set(nan2) == set(paired_io._side_scores(m2, 2)[0])


def test_main_matches_manual_item_means(tmp_path):
    """The report's per-side means must equal a direct numpy aggregation of
    the npz columns (no hidden re-weighting)."""
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path)
    result = json.loads(js.read_text())
    manual = []
    for p in sorted((a_dir / "metrics").glob("*.npz")):
        with np.load(p) as z:
            manual.append(float(z["kld"].astype(np.float64).mean()))
    got = result["metrics"]["kld"]["item_weighted"]["baseline_mean"]
    assert got == pytest.approx(np.mean(manual), abs=1e-12)


def test_main_is_deterministic(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=4)
    _rc, _out, js1 = _run_main(a_dir, b_dir, tmp_path / "r1")
    _rc, _out, js2 = _run_main(a_dir, b_dir, tmp_path / "r2")
    m1 = json.loads(js1.read_text())["metrics"]
    m2 = json.loads(js2.read_text())["metrics"]
    assert m1 == m2


def test_main_with_omit_host_metadata_is_byte_identical(tmp_path):
    """The whole-file oracle: SAME command run twice with --omit-host-metadata
    must produce byte-identical .md and .json (same output paths, so the
    recorded command line is identical too). Without the flag the JSON embeds
    MemAvailable, which moves between runs."""
    a_dir, b_dir = _make_pair(tmp_path, n_items=4)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    _rc, out, js = _run_main(a_dir, b_dir, run_dir, "--omit-host-metadata")
    md1, j1 = out.read_bytes(), js.read_bytes()
    _rc, out, js = _run_main(a_dir, b_dir, run_dir, "--omit-host-metadata")
    assert out.read_bytes() == md1
    assert js.read_bytes() == j1
    assert "ram_available_bytes" not in j1.decode()


def test_main_hard_fails_on_meta_mismatch(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, meta_b_overrides={"tf_chunk": 1})
    with pytest.raises(SystemExit, match="tf_chunk"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_hard_fails_on_ref_drift_by_default(tmp_path):
    def bump_ref(key, rec):
        rec["nll_ref"][0] += 1e-4
    a_dir, b_dir = _make_pair(tmp_path, mutate_b=bump_ref)
    with pytest.raises(SystemExit, match="reference drift"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_allow_ref_drift_compares_anyway_and_persists_magnitude(tmp_path):
    def bump_ref(key, rec):
        rec["nll_ref"][0] += np.float32(1e-4)
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, mutate_b=bump_ref)
    rc, out, js = _run_main(a_dir, b_dir, tmp_path, "--allow-ref-drift")
    assert rc == 0
    md = out.read_text()
    assert "approximately paired" in md
    # The artifact must quantify the drift — stderr is gone when a reader
    # audits an archived report.
    assert "max |Δ nll_ref|" in md
    result = json.loads(js.read_text())
    assert result["n_items"] == 3
    assert result["alignment"]["n_ref_drift"] == 3
    drift = result["alignment"]["ref_drift"]
    assert drift["n_items"] == 3 and drift["n_used"] == 3
    assert 0 < drift["max_abs_dnll_ref"] < 1e-3
    assert drift["argmax_ref_flips"] == 0


def test_main_partial_drift_counts_only_drifted_items(tmp_path):
    def bump_one(key, rec):
        if key == "001_item1":
            rec["nll_ref"][0] += np.float32(1e-4)
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, mutate_b=bump_one)
    with pytest.raises(SystemExit, match="1 item"):
        _run_main(a_dir, b_dir, tmp_path / "strict")
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path / "loose", "--allow-ref-drift")
    assert rc == 0
    result = json.loads(js.read_text())
    assert result["n_items"] == 3
    assert result["alignment"]["n_ref_drift"] == 1
    assert result["alignment"]["ref_drift"]["n_used"] == 1


def test_main_b_side_scores_come_from_b_columns(tmp_path):
    """A refactor reusing A's reference columns for B ('they must be identical
    anyway') corrupts exactly the --allow-ref-drift degraded mode — pin B's
    mse_dp to a manual computation from B's OWN npz columns."""
    def shift_ref(key, rec):
        rec["nll_ref"] += np.float32(0.5)
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, mutate_b=shift_ref)
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path, "--allow-ref-drift")
    assert rc == 0
    result = json.loads(js.read_text())
    manual = []
    for p in sorted((b_dir / "metrics").glob("*.npz")):
        with np.load(p) as z:
            nr = z["nll_ref"].astype(np.float64)
            nc = z["nll_cand"].astype(np.float64)
            dp = (np.exp(-nc) - np.exp(-nr)) * 100.0
            manual.append(float((dp * dp).mean()))
    got = result["metrics"]["mse_dp"]["item_weighted"]["candidate_mean"]
    assert got == pytest.approx(np.mean(manual), abs=1e-9)


def test_main_aborts_on_nonfinite_item(tmp_path):
    def poison(key, rec):
        if key == "000_item0":
            rec["kld"][0] = np.nan
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, mutate_b=poison)
    with pytest.raises(SystemExit, match="000_item0.*non-finite.*aborted"):
        _run_main(a_dir, b_dir, tmp_path)


@pytest.mark.parametrize("suffix, message", [
    (".bin", "unconverted metric artifact"),
    (".bin.tmp", "incomplete temporary artifact"),
])
def test_unfinished_artifact_aborts_without_manifest(tmp_path, suffix, message):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    (b_dir / "metrics" / f"000_item0{suffix}").write_bytes(b"unfinished")
    with pytest.raises(SystemExit, match=message + ".*aborted"):
        _run_main(a_dir, b_dir, tmp_path)


def test_collection_failure_aborts_before_item_intersection(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    (a_dir / "manifest.csv").write_text(
        "row_idx,status\n0,OK\n1,FAIL_EXIT_1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="failed manifest row.*aborted"):
        _run_main(a_dir, b_dir, tmp_path)


def test_rejected_artifact_aborts_without_manifest(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3)
    (b_dir / "metrics" / "000_item0.bin.rejected").write_bytes(b"evidence")
    with pytest.raises(SystemExit, match="rejected metric artifact.*aborted"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_fails_closed_below_two_items(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=2, n_items_b=1)
    with pytest.raises(SystemExit, match=">= 2 usable items"):
        _run_main(a_dir, b_dir, tmp_path, "--allow-interaction")


def test_main_num_eval_tokens_cap_changes_scores(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=2)
    _rc, _out, js_all = _run_main(a_dir, b_dir, tmp_path / "all")
    _rc, _out, js_cap = _run_main(a_dir, b_dir, tmp_path / "cap",
                                  "--num-eval-tokens", "2")
    full = json.loads(js_all.read_text())["metrics"]["kld"]["item_weighted"]["baseline_mean"]
    capped = json.loads(js_cap.read_text())["metrics"]["kld"]["item_weighted"]["baseline_mean"]
    assert full != capped


def test_main_rejects_unknown_metric(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    with pytest.raises(SystemExit, match="unknown --metrics"):
        _run_main(a_dir, b_dir, tmp_path, "--metrics", "nll", "bogus")


# ---------------------------------------------------------------------------
# weights / token-weighting (heterogeneous npos — uniform npos makes
# token-weighted ≡ item-weighted and would leave the weights wiring untested)
# ---------------------------------------------------------------------------

def _manual_token_mean(d, metric, cap=-1):
    means, ws = [], []
    for p in sorted((Path(d) / "metrics").glob("*.npz")):
        with np.load(p) as z:
            keep = int(z["npos"]) if cap == -1 else min(cap, int(z["npos"]))
            means.append(float(z[metric][:keep].astype(np.float64).mean()))
            ws.append(keep)
    return float(np.average(means, weights=ws)), ws


def test_main_token_weighting_uses_per_item_npos(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=2, npos_list=[4, 12])
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    result = json.loads(js.read_text())
    expected, ws = _manual_token_mean(a_dir, "kld")
    assert ws == [4, 12]
    got = result["metrics"]["kld"]["token_weighted"]["baseline_mean"]
    assert got == pytest.approx(expected, abs=1e-12)
    assert got != result["metrics"]["kld"]["item_weighted"]["baseline_mean"]


def test_main_cap_caps_weights_and_scores_per_item(tmp_path):
    """--num-eval-tokens 6 with npos [4, 12] must weight items [4, 6] and
    score only each item's first keep positions — pins weights.append(keep)
    including the min() clamp, on BOTH sides."""
    a_dir, b_dir = _make_pair(tmp_path, n_items=2, npos_list=[4, 12])
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path, "--num-eval-tokens", "6")
    assert rc == 0
    result = json.loads(js.read_text())
    expected_a, ws = _manual_token_mean(a_dir, "kld", cap=6)
    assert ws == [4, 6]
    assert result["metrics"]["kld"]["token_weighted"]["baseline_mean"] == \
        pytest.approx(expected_a, abs=1e-12)
    expected_b, _ = _manual_token_mean(b_dir, "kld", cap=6)
    assert result["metrics"]["kld"]["token_weighted"]["candidate_mean"] == \
        pytest.approx(expected_b, abs=1e-12)


def test_main_cap_larger_than_npos_is_noop(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=2, npos_list=[4, 12])
    _rc, _o, js_all = _run_main(a_dir, b_dir, tmp_path / "all")
    _rc, _o, js_cap = _run_main(a_dir, b_dir, tmp_path / "cap",
                                "--num-eval-tokens", "100")
    assert json.loads(js_all.read_text())["metrics"] == \
        json.loads(js_cap.read_text())["metrics"]


# ---------------------------------------------------------------------------
# main(): range selection, missing metas, artifact-persisted diagnostics
# ---------------------------------------------------------------------------

def test_main_start_end_selects_subset(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=4)
    rc, _out, js = _run_main(a_dir, b_dir, tmp_path, "--start", "1", "--end", "3")
    assert rc == 0
    result = json.loads(js.read_text())
    assert result["n_items"] == 2
    manual = []
    for stem in ("001_item1", "002_item2"):
        with np.load(a_dir / "metrics" / f"{stem}.npz") as z:
            manual.append(float(z["kld"].astype(np.float64).mean()))
    assert result["metrics"]["kld"]["item_weighted"]["baseline_mean"] == \
        pytest.approx(np.mean(manual), abs=1e-12)


def test_main_rejects_missing_execution_metadata(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    (a_dir / "collect_meta.json").unlink()
    (b_dir / "collect_meta.json").unlink()
    with pytest.raises(SystemExit, match="executed scorer identity missing"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_rejects_one_missing_execution_metadata(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    (a_dir / "collect_meta.json").unlink()
    with pytest.raises(SystemExit, match="executed scorer identity missing"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_persists_meta_warnings_in_artifacts(tmp_path):
    """A degenerate same-model comparison reads exactly like a genuine
    'no sig. diff.' — the warning must live in the archived report, not
    only on stderr."""
    a_dir, b_dir = _make_pair(tmp_path)
    (b_dir / "collect_meta.json").write_text(json.dumps(BASE_META))  # cand == a's
    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    assert "SAME model" in out.read_text()
    assert any("SAME model" in w
               for w in json.loads(js.read_text())["alignment"]["warnings"])


def test_main_hard_fails_on_one_sided_missing_item(tmp_path, capsys):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, n_items_b=2)

    with pytest.raises(SystemExit, match="input item sets differ"):
        _run_main(a_dir, b_dir, tmp_path)

    stderr = capsys.readouterr().err
    assert "ERROR: paired input data has one-sided missing item" in stderr
    assert "candidate-b missing 1: 002_item2" in stderr
    assert not (tmp_path / "report.md").exists()


def test_main_allow_interaction_renders_missing_items(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, n_items_b=2)

    rc, out, js = _run_main(
        a_dir, b_dir, tmp_path, "--allow-interaction")
    assert rc == 0
    md = out.read_text()
    assert "Compared on the 2-item intersection" in md
    assert "candidate-b missing 1: 002_item2" in md
    assert "`--allow-interaction` explicitly enabled" in md
    result = json.loads(js.read_text())
    assert result["alignment"]["allow_interaction"] is True
    assert result["alignment"]["common_skipped_over_budget"] == []


def test_main_reports_identical_budget_skip_set(tmp_path, capsys):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, skipped_a=(2,), skipped_b=(2,))

    rc, out, js = _run_main(a_dir, b_dir, tmp_path)
    assert rc == 0
    assert "excluded identically" in capsys.readouterr().err
    assert "Common SKIP_OVER_BUDGET: 1 row(s): 2" in out.read_text()
    result = json.loads(js.read_text())
    assert result["alignment"]["common_skipped_over_budget"] == [2]
    assert result["alignment"]["n_common_skipped_over_budget"] == 1


def test_main_budget_skip_mismatch_is_fatal_even_with_interaction(
    tmp_path, capsys,
):
    a_dir, b_dir = _make_pair(tmp_path, n_items=3, skipped_a=(2,))

    with pytest.raises(SystemExit, match="SKIP_OVER_BUDGET set mismatch"):
        _run_main(a_dir, b_dir, tmp_path, "--allow-interaction")

    stderr = capsys.readouterr().err
    assert "SKIP_OVER_BUDGET sets differ" in stderr
    assert "candidate-a: [2]" in stderr
    assert "candidate-b: []" in stderr
    assert not (tmp_path / "report.md").exists()


# ---------------------------------------------------------------------------
# main(): failure ergonomics
# ---------------------------------------------------------------------------

def test_main_corrupt_npz_error_names_key_dir_and_path(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    p = b_dir / "metrics" / "001_item1.npz"
    p.write_bytes(p.read_bytes()[:50])
    with pytest.raises(SystemExit) as ei:
        _run_main(a_dir, b_dir, tmp_path)
    msg = str(ei.value)
    assert "001_item1" in msg and "candidate-b" in msg and str(p) in msg


def test_main_rejects_missing_dir_and_missing_metrics_subdir(tmp_path):
    a_dir, _b_dir = _make_pair(tmp_path)
    with pytest.raises(SystemExit, match="not a directory"):
        _run_main(a_dir, tmp_path / "nope", tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="metrics/ subdir"):
        _run_main(a_dir, empty, tmp_path)


def test_main_aborts_on_unconverted_bins_before_item_selection(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path, n_items=2)
    for p in (b_dir / "metrics").glob("*.npz"):
        p.rename(p.with_suffix(".bin"))   # simulate un-postprocessed leftovers
    with pytest.raises(SystemExit, match="unconverted metric artifact.*aborted"):
        _run_main(a_dir, b_dir, tmp_path)


def test_main_rejects_bad_range_and_confidence(tmp_path):
    a_dir, b_dir = _make_pair(tmp_path)
    with pytest.raises(SystemExit, match="--start"):
        _run_main(a_dir, b_dir, tmp_path, "--start", "-1")
    with pytest.raises(SystemExit, match="--end"):
        _run_main(a_dir, b_dir, tmp_path, "--end", "-2")
    with pytest.raises(SystemExit, match="confidence"):
        _run_main(a_dir, b_dir, tmp_path, "--confidence-level", "0")
    # Finding 14: 0 iterations previously reached numpy quantile on an empty
    # array deep in the statistics engine (native tool AND this one).
    with pytest.raises(SystemExit, match="--bootstrap-iters"):
        _run_main(a_dir, b_dir, tmp_path, "--ci-method", "studentized",
                  "--bootstrap-iters", "0")


@pytest.mark.parametrize("state", ["running", "interrupted"])
def test_completed_prefix_cannot_hide_an_unfinished_append(tmp_path, state):
    from lib.collection_state import CollectionAttempt
    a, b = _make_pair(tmp_path)
    for directory in (a, b):
        attempt = CollectionAttempt(directory, 3, 4)
        attempt.declare([(3, "next")])
        if state == "interrupted":
            attempt.stop(KeyboardInterrupt())
    with pytest.raises(SystemExit, match=f"attempt is {state}"):
        _run_main(a, b, tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_comparison_refuses_active_writer_even_with_complete_old_rows(tmp_path):
    from lib.collect_common import acquire_out_lock
    a, b = _make_pair(tmp_path)
    with acquire_out_lock(a):
        with pytest.raises(SystemExit, match="collector is still active"):
            _run_main(a, b, tmp_path)


def test_comparison_requires_exact_binary_identity_despite_identical_reference(tmp_path):
    a, b = _make_pair(tmp_path)
    meta = json.loads((b / "collect_meta.json").read_text())
    meta["execution_identity"]["binary_sha256"] = "b" * 64
    (b / "collect_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="scorer/build/backend identity differs"):
        _run_main(a, b, tmp_path)


def test_intersection_option_cannot_hide_a_missing_declared_metric(tmp_path):
    a, b = _make_pair(tmp_path)
    (b / "metrics/002_item2.npz").unlink()
    with pytest.raises(SystemExit, match="metric artifacts differ"):
        _run_main(a, b, tmp_path, "--allow-interaction")


def test_comparison_rejects_rename_before_terminal_record(tmp_path):
    from lib.collection_state import CollectionAttempt
    a, b = _make_pair(tmp_path)
    for directory in (a, b):
        attempt = CollectionAttempt(directory, 3, 4)
        attempt.declare([(3, "new")])
        (directory / "metrics/003_new.npz").write_bytes((directory / "metrics/000_item0.npz").read_bytes())
    with pytest.raises(SystemExit, match="attempt is running"):
        _run_main(a, b, tmp_path)


def test_failed_root_sync_cannot_publish_completed_attempt(tmp_path, monkeypatch):
    import lib.collection_state as state
    root = tmp_path / "collection"
    root.mkdir()
    attempt = state.CollectionAttempt(root, 0, 1)
    attempt.declare([(0, "item")])
    attempt.record({"row_idx": 0, "item_id": "item", "status": "OK"})
    original = state.fsync_directory
    def fail_root(path):
        if Path(path) == root:
            raise OSError("simulated directory sync failure")
        original(path)
    monkeypatch.setattr(state, "fsync_directory", fail_root)
    with pytest.raises(OSError, match="directory sync failure"):
        attempt.finish()
    with pytest.raises(ValueError, match="attempt is running"):
        state.require_completed_attempts(root)


@pytest.mark.parametrize("version_b", [4, 5])
def test_v4_compare_omits_ear64_without_fabricating_columns(tmp_path, version_b):
    a, b = _make_pair(tmp_path, version=4, version_b=version_b)
    rc, md, js = _run_main(a, b, tmp_path)
    assert rc == 0
    result = json.loads(js.read_text())
    assert result["vlmk_versions"] == {"candidate_a": [4], "candidate_b": [version_b]}
    warnings = "\n".join(result["alignment"]["warnings"])
    for key in ("ear_64", "ear_64_normalized"):
        assert key not in result["metrics"]
        assert key in warnings and key in md.read_text()
        with pytest.raises(SystemExit, match=key + ".*candidate-a.*VLMK v4"):
            _run_main(a, b, tmp_path, "--metrics", "kld", key)
    assert "candidate-a (VLMK v4)" in warnings
    if version_b == 5:
        assert "candidate-b (VLMK" not in warnings
    assert "ear_20" in result["metrics"]
    assert "ear_20_normalized" in result["metrics"]


def test_ear64_report_uses_saved_prefix_means_and_higher_is_better(tmp_path):
    a, b = _make_pair(tmp_path, npos_list=[2, 3, 5])
    for side, offset in ((a, 0.0), (b, 0.1)):
        for path in (side / "metrics").glob("*.npz"):
            with np.load(path) as z:
                payload = {key: z[key] for key in z.files}
            npos = int(payload["npos"])
            payload["ear_64"] = np.linspace(0.25 + offset, 0.75 + offset, npos, dtype=np.float32)
            payload["ear_64_normalized"] = np.linspace(0.5 + offset, 0.8 + offset, npos, dtype=np.float32)
            np.savez(path, **payload)
    rc, md, js = _run_main(a, b, tmp_path, "--num-eval-tokens", "3")
    assert rc == 0
    result = json.loads(js.read_text())
    assert result["multiplicity"]["family_size"] == 14
    for key in ("ear_64", "ear_64_normalized"):
        block = result["metrics"][key]
        assert block["score_direction"] == "higher_is_better"
        assert key in md.read_text()
        means, weights = [], []
        for path in sorted((a / "metrics").glob("*.npz")):
            with np.load(path) as z:
                prefix = z[key][:3].astype(np.float64)
                means.append(float(prefix.mean()))
                weights.append(len(prefix))
        assert block["item_weighted"]["baseline_mean"] == pytest.approx(np.mean(means), abs=1e-12)
        assert block["token_weighted"]["baseline_mean"] == pytest.approx(np.average(means, weights=weights), abs=1e-12)
        assert block["item_weighted"]["delta_candidate_minus_baseline"] > 0
        assert block["item_weighted"]["decision"]["verdict"] == "B closer"


def _make_corpus_pair(tmp_path):
    from lib.text_corpus import corpus_row, digest_json
    from test_prep_llm_score_from_hf import _corpus_protocol
    protocol = _corpus_protocol()
    from lib.text_corpus import token_digest
    protocol.update(stream_tokens=35, stream_sha256=token_digest(list(range(1, 9)) * 4 + [9, 10, 1]),
                    vocabulary={**protocol["vocabulary"], "size": 11})
    assignments = ([0, 0, 1], [1, 1, 1], [1, 2, 2], [2, 2, 2])
    rows = [corpus_row(protocol, [1, 2, 3, 4, 5, 6, 7, 8], i, list(ids)) for i, ids in enumerate(assignments)]
    corpus = {"protocol": protocol, "windows": [json.loads(row["corpus_window"]) for row in rows]}
    directories = []
    for role, seed in (("a", 10), ("b", 20)):
        directory = tmp_path / role
        (directory / "metrics").mkdir(parents=True)
        keys = []
        for i, row in enumerate(rows):
            key = f"{i:03d}_{row['id']}"
            keys.append(key)
            records = _make_item(npos=3, seed=seed + i, ref_seed=100 + i)
            records["target"] = row["input_ids"][5:]
            records["kld"] = np.arange(i * 3, i * 3 + 3) / 100 + (0.01 if role == "b" else 0)
            path = directory / "metrics" / f"{key}.bin"
            write_vlmk(path, records, vocab=11, n_prefill=5, n_past_actual=8)
            kio.convert_kld_bin_to_npz(path, path.with_suffix(".npz"))
            path.unlink()
        completed_collection(directory, keys)
        meta = {**BASE_META, "kind": "llm_kld_metrics", "sort_by": "", "cand_model": f"/m/{role}.gguf",
                "n_ctx": 8, "n_batch": 8, "n_ubatch": 8, "tf_chunk": -1, "n_threads": 8, "metric_threads": 8,
                "perplexity_window": True, "corpus_protocol": protocol,
                "corpus_windows_sha256": digest_json(corpus)}
        (directory / "collect_meta.json").write_text(json.dumps(meta))
        (directory / "corpus_windows.json").write_text(json.dumps(corpus))
        directories.append(directory)
    return directories


@pytest.mark.parametrize("unit,n_units", [("window", 4), ("article", 3), ("block", 2)])
def test_corpus_aggregation_preserves_pooled_targets_and_labels(tmp_path, unit, n_units):
    a, b = _make_corpus_pair(tmp_path)
    rc, report, path = _run_main(a, b, tmp_path, "--unit", unit, "--block-windows", "2", "--metrics", "kld")
    assert rc == 0
    result = json.loads(path.read_text())
    assert result["n_items"] == n_units
    assert result["sampling"]["n_windows"] == 4
    assert result["sampling"]["unit"] == unit
    assert "model-generated answer trajectory" not in report.read_text()
    assert "may remain dependent" in report.read_text()
    assert "independent sampling units" in result["sampling"]["scope"]
    assert "conditional fixed-corpus" in result["multiplicity"]["policy"]
    assert result["metrics"]["kld"]["token_weighted"]["baseline_mean"] == pytest.approx(0.055)
    if unit == "article":
        expected = np.mean([0.005, 0.04, 0.09])
        assert result["metrics"]["kld"]["item_weighted"]["baseline_mean"] == pytest.approx(expected)
    if unit != "window":
        assert "position_strata" not in result


def test_corpus_aggregation_rejects_partial_population_and_changed_targets(tmp_path):
    a, b = _make_corpus_pair(tmp_path)
    with pytest.raises(SystemExit, match="complete corpus window set"):
        _run_main(a, b, tmp_path, "--unit", "article", "--end", "3")
    for directory in (a, b):
        path = next((directory / "metrics").glob("*.npz"))
        with np.load(path) as data:
            values = dict(data)
        values["target"][-1] = 9
        np.savez(path, **values)
    with pytest.raises(SystemExit, match="targets or runtime shape"):
        _run_main(a, b, tmp_path, "--unit", "block", "--block-windows", "2")


@pytest.mark.parametrize("field", ["n_ctx", "n_threads", "metric_threads", "n_gpu_layers", "flash_attn"])
def test_pairing_rejects_runtime_changes_even_with_identical_reference_columns(tmp_path, field):
    a, b = _make_pair(tmp_path, meta_b_overrides={field: "changed"})
    with pytest.raises(SystemExit, match=field):
        _run_main(a, b, tmp_path)


def test_perplexity_bridge_verifies_binary_tokens_and_quantized_reference(tmp_path):
    import argparse
    import struct
    from cli.verify_perplexity_bridge import read_ppl_header, verify
    a, _ = _make_corpus_pair(tmp_path)
    meta = json.loads((a / "collect_meta.json").read_text())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(json.dumps({"protocol": meta["corpus_protocol"]}))
    stream = list(range(1, 9)) * 4 + [9, 10, 1]
    (prepared / "stream.json").write_text(json.dumps(stream))
    records = [kio.load_kld_metrics(path)[0] for path in sorted((a / "metrics").glob("*.npz"))]
    encoded = np.zeros((4, 3, 16), dtype="<u2")
    quantized = []
    for index, row in enumerate(records):
        params = np.tile(np.array([0.001, -2.0], dtype="<f4"), (3, 1))
        encoded[index, :, :4] = params.view("<u2")
        codes = np.rint((2.0 - row["nll_ref"]) / 0.001).astype("<u2")
        encoded[index, np.arange(3), row["target"] + 4] = codes
        quantized.extend(-(params[:, 0] * codes + params[:, 1]))
    path = tmp_path / "ppl.bin"
    path.write_bytes(b"_logits_" + struct.pack("<III", 8, 11, 4)
                     + np.array(stream[:32], dtype="<i4").tobytes() + encoded.tobytes())
    assert read_ppl_header(path)["tokens"].reshape(-1).tolist() == stream[:32]
    ref_mean = np.concatenate([row["nll_ref"] for row in records]).astype(float).mean()
    cand_mean = np.concatenate([row["nll_cand"] for row in records]).astype(float).mean()
    kld_mean = np.concatenate([row["kld"] for row in records]).astype(float).mean()
    shape = "n_ctx=8, batch_size=8, n_seq=1\nn_threads = 8 (n_threads_batch = 8)\nmetric_threads = 8\n"
    ref_log, cand_log = tmp_path / "ref.log", tmp_path / "cand.log"
    ref_log.write_text(shape + f"Final estimate: PPL = {math.exp(ref_mean):.4f}\n")
    cand_log.write_text(shape + f"Mean PPL(Q) : {math.exp(cand_mean):.6f}\nMean PPL(base) : {math.exp(np.mean(quantized)):.6f}\nMean KLD: {kld_mean:.6f}\n")
    args = argparse.Namespace(llm_collection=a, prepared=prepared, ppl_logits=path,
                              ppl_reference_log=ref_log, ppl_candidate_log=cand_log, nll_tolerance=1e-5)
    from lib.collect_common import acquire_out_lock
    with acquire_out_lock(a):
        with pytest.raises(ValueError, match="collector is still active"):
            verify(args)
    result = verify(args)
    assert result["status"] == "passed" and result["targets"] == 12
    assert result["quantized_reference"]["max_abs_target_error"] < 0.00051
    raw = path.read_bytes()
    for bad in (raw[:-1], raw + b"x", b"badmagic" + raw[8:]):
        path.write_bytes(bad)
        with pytest.raises(ValueError):
            read_ppl_header(path)
    path.write_bytes(raw)
    cand_log.write_text(cand_log.read_text().replace("metric_threads = 8", "metric_threads = 4"))
    with pytest.raises(ValueError, match="threads must agree"):
        verify(args)


def test_grouped_allowed_drift_counts_the_contributing_source_window(tmp_path):
    a, b = _make_corpus_pair(tmp_path)
    path = next((b / "metrics").glob("*.npz"))
    with np.load(path) as data:
        values = dict(data)
    values["nll_ref"][0] += np.float32(0.001)
    np.savez(path, **values)
    rc, _, report = _run_main(a, b, tmp_path, "--unit", "article", "--allow-ref-drift", "--metrics", "kld")
    assert rc == 0
    result = json.loads(report.read_text())
    assert result["alignment"]["ref_drift"]["n_used"] == 1
    assert result["alignment"]["ref_drift"]["n_items"] == 1


def test_perplexity_header_checks_size_before_allocating_tokens(tmp_path, monkeypatch):
    import struct
    from cli.verify_perplexity_bridge import read_ppl_header
    path = tmp_path / "bad-ppl.bin"
    path.write_bytes(b"_logits_" + struct.pack("<III", 512, 262144, 2**32 - 1))
    def forbidden_read(*args, **kwargs):
        pytest.fail("unvalidated header drove an allocation")
    monkeypatch.setattr(np, "fromfile", forbidden_read)
    with pytest.raises(ValueError, match="incomplete or has trailing"):
        read_ppl_header(path)


def test_self_paired_smoke_consumes_descriptive_token_schema(tmp_path, monkeypatch):
    from verify_and_validation_scripts import smoke_kld

    a, _ = _make_pair(tmp_path, npos_list=[2, 3, 4])
    rc, _, report_path = _run_main(a, a, tmp_path)
    assert rc == 0
    report = json.loads(report_path.read_text())
    for name in ("paired", "self-paired"):
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    collection = {"sample.npz": {"npos": np.array(3), "kld": np.zeros(3)}}
    monkeypatch.setattr(smoke_kld, "read_collection", lambda *args: collection)
    monkeypatch.setattr("sys.argv", ["smoke_kld.py", "--verify-only", "--lane", "llm",
                                    "--ref-model", "ref", "--cand-a-model", "a", "--cand-b-model", "b",
                                    "--dataset", "dataset", "--out", str(tmp_path)])
    smoke_kld.main()
    assert json.loads((tmp_path / "summary.json").read_text())["self_paired_zero_deltas"] is True
    report["metrics"]["kld"]["token_weighted"]["p_value"] = 0.5
    (tmp_path / "self-paired.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        smoke_kld.main()


def _campaign_inputs(n=24, variants=9, cells=2):
    plan = {"schema_version": "company-campaign-plan-v1", "family_id": "global", "metric": "kld", "weighting": "item", "sampling_plan": "fixed_n", "prespecification_status": "user_declared", "alpha": .05, "correction": "holm", "cells": []}
    loaded = {}
    rng = np.random.default_rng(812)
    for c in range(cells):
        cid = f"cell{c}"
        roster = [{"item_id": f"item{i}", "item_key": f"{i:03}_item{i}", "cluster_id": f"image{i}", "source_id": "source", "image_hashes": [f"hash{i}"]} for i in range(n)]
        plan["cells"].append({"cell_id": cid, "dataset_id": "dataset", "dataset_content_hash": "ds-v2:same", "mode": "instruct", "model_family": f"family{c}", "model_id": f"model{c}", "fixed_analysis_n": n, "roster": roster, "variants": [{"quant_id": f"q{k}", "collection_dir": f"{cid}/q{k}"} for k in range(variants)]})
        scores = 1 + rng.normal(0, .02, (n, variants)) + .1*np.arange(variants)
        loaded[cid] = {"scores": scores, "provenance": [{"quant_id": f"q{k}", "collection_dir": f"/{cid}/q{k}", "collect_meta_sha256": f"{cid}/meta{k}", "reference_fingerprint_sha256": f"{cid}/ref"} for k in range(variants)]}
    return plan, loaded


def test_campaign_global_family_and_simultaneous_best_set():
    from stats.campaign import analyze_campaign
    from stats.multiplicity import adjust_pvalues
    plan, loaded = _campaign_inputs()
    result = analyze_campaign(plan, loaded)
    assert result["family_size"] == 72
    assert len(result["pairs"]) == 72
    assert [p["p_adjusted"] for p in result["pairs"]] == pytest.approx(adjust_pvalues([p["p_value"] for p in result["pairs"]]))
    assert all(c["unique_best_by_simultaneous_intervals"] == "q0" for c in result["cells"])
    assert all(p["ci_bonferroni"][0] <= p["ci_pointwise"][0] <= p["ci_pointwise"][1] <= p["ci_bonferroni"][1] for p in result["pairs"])
    plan["weighting"] = "token"
    with pytest.raises(ValueError, match="weighting=item"):
        analyze_campaign(plan, loaded)


def test_clustered_mean_preserves_item_weights_and_scalar_covariance():
    from stats.campaign import clustered_mean_test
    from scipy.stats import t
    d = np.array([1., 1., 1., 8.])
    result = clustered_mean_test(d, [0, 0, 0, 1], .05, 36)
    assert result["estimate"] == pytest.approx(2.75)
    assert result["standard_error"] == pytest.approx(2.625)
    assert result["degrees_of_freedom"] == 1
    assert result["p_value"] == pytest.approx(2 * t.sf(2.75/2.625, 1))
    iid = clustered_mean_test(d, list(range(4)), .05, 1)
    assert iid["standard_error"] == pytest.approx(d.std(ddof=1)/2)
    for bad_alpha, bad_family in [(float("nan"), 1), (.05, 0), (.05, 1.2)]:
        with pytest.raises(ValueError):
            clustered_mean_test(d, list(range(4)), bad_alpha, bad_family)
    with pytest.raises(ValueError, match="unresolved"):
        clustered_mean_test(np.array([1, 1, 3, 3])*np.nextafter(0., 1.), list(range(4)), .05, 1)


def test_campaign_rejects_missing_cells_bad_clusters_and_undeclared_bh():
    from stats.campaign import analyze_campaign
    plan, loaded = _campaign_inputs()
    with pytest.raises(ValueError, match="exactly"):
        analyze_campaign(plan, {"cell0": loaded["cell0"]})
    plan["cells"][0]["roster"][1]["image_hashes"] = ["hash0"]
    with pytest.raises(ValueError, match="sharing an image"):
        analyze_campaign(plan, loaded)
    plan["cells"][0]["roster"][1]["cluster_id"] = "image0"
    plan["correction"] = "fdr_bh"
    with pytest.raises(ValueError, match="BH requires"):
        analyze_campaign(plan, loaded)
    plan["dependence_assumption"] = "independent_or_prds"
    assert analyze_campaign(plan, loaded)["error_control"] == "FDR"


def test_campaign_loader_cli_and_frozen_prefix(tmp_path):
    from stats.campaign import analyze_campaign, load_campaign
    from stats.cli.campaign_compare import main
    a, b = _make_pair(tmp_path, n_items=8)
    for root in (a, b):
        path = root / "collect_meta.json"
        meta = json.loads(path.read_text())
        meta.update(cand_model_fingerprint=f"candidate-{root.name}", cand_mmproj_fingerprint="projector")
        path.write_text(json.dumps(meta))
    plan, _ = _campaign_inputs(n=8, variants=2, cells=1)
    cell = plan["cells"][0]
    cell["variants"][0]["collection_dir"] = str(a)
    cell["variants"][1]["collection_dir"] = str(b)
    cell["analysis_item_ids"] = [r["item_id"] for r in cell["roster"][:4]]
    cell["fixed_analysis_n"] = 4
    loaded = load_campaign(plan)
    result = analyze_campaign(plan, loaded)
    assert result["pairs"][0]["n_items"] == 4
    assert len(result["cells"][0]["input_provenance"]) == 2
    manifest = tmp_path / "plan.json"
    manifest.write_text(json.dumps(plan))
    out, output_json = tmp_path / "report.md", tmp_path / "result.json"
    assert main(["analyze", "--manifest", str(manifest), "--out", str(out), "--output-json", str(output_json)]) == 0
    assert json.loads(output_json.read_text())["family_size"] == 1
    assert "Clusters" in out.read_text()
    meta = json.loads((b / "collect_meta.json").read_text())
    missing_fingerprint = dict(meta)
    missing_fingerprint.pop("cand_model_fingerprint")
    (b / "collect_meta.json").write_text(json.dumps(missing_fingerprint))
    with pytest.raises(ValueError, match="candidate model/projector fingerprints"):
        load_campaign(plan)
    meta["dataset_content_hash"] = "changed"
    (b / "collect_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_campaign(plan)


def test_selection_receipt_fixes_pool_and_withheld_families():
    import copy
    from stats.campaign import analyze_campaign, content_hash, select_benchmark
    plan, loaded = _campaign_inputs(n=10, variants=3)
    rule = {"algorithm": "greedy_backward_maximin_snr_v1", "size": 6, "training_families": ["family0", "family1"], "heldout_families": ["family2", "family3"], "source_min_counts": {"source": 6}}
    receipt = select_benchmark(plan, loaded, rule)
    assert len(receipt["selected_item_ids"]) == 6
    assert receipt["candidate_subsets_evaluated"] == 10+9+8+7
    assert receipt == select_benchmark(plan, loaded, rule)
    validation = copy.deepcopy(plan)
    validation["selection_receipt"] = receipt
    for i, cell in enumerate(validation["cells"]):
        cell["model_family"] = f"family{i+2}"
        cell["analysis_item_ids"] = receipt["selected_item_ids"]
        cell["fixed_analysis_n"] = 6
    with pytest.raises(ValueError, match="reuse the same resolved path"):
        analyze_campaign(validation, loaded)
    loaded = copy.deepcopy(loaded)
    for value in loaded.values():
        for record in value["provenance"]:
            for field in ("collection_dir", "collect_meta_sha256", "reference_fingerprint_sha256"):
                record[field] += "heldout"
    assert analyze_campaign(validation, loaded)["pairs"][0]["n_items"] == 6
    bad = copy.deepcopy(validation)
    bad["cells"] = bad["cells"][:1]
    with pytest.raises(ValueError, match="every declared heldout"):
        analyze_campaign(bad, {"cell0": loaded["cell0"]})
    bad = copy.deepcopy(validation)
    bad["cells"][0]["mode"] = "thinking"
    with pytest.raises(ValueError, match="mode"):
        analyze_campaign(bad, loaded)
    bad = copy.deepcopy(validation)
    bad["selection_receipt"]["selected_item_ids_sha256"] = "wrong"
    bad["selection_receipt"]["receipt_sha256"] = content_hash({k:v for k,v in bad["selection_receipt"].items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="hash"):
        analyze_campaign(bad, loaded)
    bad = copy.deepcopy(plan)
    bad["cells"][0]["model_family"] = "family2"
    with pytest.raises(ValueError, match="training families"):
        select_benchmark(bad, loaded, rule)


def test_selector_does_not_turn_cancelling_deltas_into_absolute_effects():
    from stats.campaign import select_benchmark
    plan, loaded = _campaign_inputs(n=8, variants=2, cells=1)
    d = np.array([-1., 1., -2., 2., -3., 3., -4., 4.])
    loaded["cell0"]["scores"] = np.column_stack([np.full(8, 5.), 5+d])
    rule = {"algorithm": "greedy_backward_maximin_snr_v1", "size": 4, "training_families": ["family0"], "heldout_families": ["heldout"]}
    result = select_benchmark(plan, loaded, rule)
    idx = [int(key.removeprefix("item")) for key in result["selected_item_ids"]]
    assert result["training_objective"] == pytest.approx(abs(d[idx].mean())/d[idx].std(ddof=1))


def test_selected_benchmark_can_run_on_only_selected_rows_and_blocks_copied_provenance():
    import copy
    from stats.campaign import analyze_campaign, content_hash, select_benchmark
    plan, loaded = _campaign_inputs(n=9, variants=2, cells=1)
    loaded["cell0"]["provenance"] = [{"quant_id": f"q{k}", "collection_dir": f"/training/{k}", "collect_meta_sha256": f"trainingmeta{k}", "reference_fingerprint_sha256": "trainingref"} for k in range(2)]
    rule = {"algorithm": "greedy_backward_maximin_snr_v1", "size": 5, "training_families": ["family0"], "heldout_families": ["heldout"]}
    receipt = select_benchmark(plan, loaded, rule)
    validation = copy.deepcopy(plan)
    validation["selection_receipt"] = receipt
    cell = validation["cells"][0]
    cell["model_family"] = "heldout"
    idx = [i for i,r in enumerate(cell["roster"]) if r["item_id"] in receipt["selected_item_ids"]]
    cell["roster"] = [cell["roster"][i] for i in idx]
    cell["fixed_analysis_n"] = 5
    heldout = {"cell0": {"scores": loaded["cell0"]["scores"][idx], "provenance": [{"quant_id": f"q{k}", "collection_dir": f"/heldout/{k}", "collect_meta_sha256": f"heldoutmeta{k}", "reference_fingerprint_sha256": "heldoutref"} for k in range(2)]}}
    assert analyze_campaign(validation, heldout)["pairs"][0]["n_items"] == 5
    heldout["cell0"]["provenance"][0]["reference_fingerprint_sha256"] = "trainingref"
    with pytest.raises(ValueError, match="share model provenance"):
        analyze_campaign(validation, heldout)


def test_selector_resolves_small_subset_after_removing_large_outlier():
    from stats.campaign import select_benchmark
    plan, loaded = _campaign_inputs(n=4, variants=2, cells=1)
    loaded["cell0"]["scores"] = np.column_stack([np.zeros(4), [1e-20, 2e-20, 3e-20, 1.]])
    rule = {"algorithm": "greedy_backward_maximin_snr_v1", "size": 3, "training_families": ["family0"], "heldout_families": ["heldout"]}
    result = select_benchmark(plan, loaded, rule)
    assert result["selected_item_ids"] == ["item0", "item1", "item2"]
    assert result["training_objective"] == pytest.approx(2.)
    loaded["cell0"]["provenance"] = []
    with pytest.raises(ValueError, match="complete observed"):
        select_benchmark(plan, loaded, rule)


def test_campaign_cli_defaults_json_without_overwriting_plan_and_preflights_selection(tmp_path, monkeypatch):
    import stats.cli.campaign_compare as cli
    plan, loaded = _campaign_inputs(n=8, variants=2, cells=1)
    manifest = tmp_path / "campaign.json"
    original = json.dumps(plan)
    manifest.write_text(original)
    monkeypatch.setattr(cli, "load_campaign", lambda *args: loaded)
    out = tmp_path / "campaign.md"
    assert cli.main(["analyze", "--manifest", str(manifest), "--out", str(out)]) == 0
    assert manifest.read_text() == original
    assert (tmp_path / "campaign.md.json").is_file()
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--manifest", str(manifest), "--out", str(out), "--output-json", str(manifest)])
    def forbidden_load(*args):
        pytest.fail("heldout scores must not be loaded by selection")
    monkeypatch.setattr(cli, "load_campaign", forbidden_load)
    plan["selection_rule"] = {"algorithm": "greedy_backward_maximin_snr_v1", "size": 4, "training_families": ["training"], "heldout_families": ["family0"]}
    manifest.write_text(json.dumps(plan))
    with pytest.raises(SystemExit):
        cli.main(["select", "--manifest", str(manifest), "--out", str(out)])


def test_campaign_rejects_reassigned_artifact_to_stable_item_binding():
    from stats.campaign import validate_plan
    plan, _ = _campaign_inputs(n=8, variants=2, cells=1)
    roster = plan["cells"][0]["roster"]
    roster[0]["item_key"], roster[1]["item_key"] = roster[1]["item_key"], roster[0]["item_key"]
    with pytest.raises(ValueError, match="collected ID"):
        validate_plan(plan)


def test_campaign_rejects_complex_scores_and_differences():
    from stats.campaign import analyze_campaign, clustered_mean_test
    plan, loaded = _campaign_inputs(n=8, variants=2, cells=1)
    loaded["cell0"]["scores"] = loaded["cell0"]["scores"].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        analyze_campaign(plan, loaded)
    with pytest.raises(ValueError, match="real"):
        clustered_mean_test(np.array([1+1j, 2+1j, 3+1j]), [0, 1, 2], .05, 1)


@pytest.mark.parametrize("invalid", ["nonfinite", "same_path"])
def test_shared_report_writer_validates_before_replacing_output(tmp_path, invalid):
    from argparse import Namespace
    from stats.cli.common import write_report_and_json
    out = tmp_path / "report.md"
    out.write_text("existing report")
    output_json = out if invalid == "same_path" else tmp_path / "result.json"
    payload = {"p": float("nan") if invalid == "nonfinite" else .5}
    with pytest.raises(ValueError):
        write_report_and_json(Namespace(out=out, output_json=output_json), "replacement", payload)
    assert out.read_text() == "existing report"
    if invalid == "nonfinite":
        assert not output_json.exists()
