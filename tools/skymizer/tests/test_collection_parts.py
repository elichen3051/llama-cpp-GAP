"""Composition contracts and an independent concatenated-item statistical oracle."""

import json

import numpy as np
import pytest
from scipy import stats

from lib.reference_cohort import canonical_digest, tail_cohort
from stats.cli import saved_metrics_paired_compare as cli
from test_saved_metrics_paired_compare import BASE_META, _make_item, _write_metrics_dir


def cohort(size, eligible):
    requested = [f"id{i}" for i in range(size)]
    result = {}
    for name, ids in (("requested", requested), ("eligible", eligible),
                      ("excluded", [x for x in requested if x not in eligible]), ("failed", [])):
        result.update({name: len(ids), name + "_ids": ids,
                       name + "_ids_sha256": canonical_digest(ids)})
    result.update(generated=size, native_generated=size)
    return result


def generator(size, eligible):
    return {"schema_version": "skymizer-reference-v2", "cohort": cohort(size, eligible),
            "dataset_source": {"path": "org/prepared", "revision": "a" * 40,
                               "split": "train", "subset": f"source-subsample-{size}", "num_rows": size},
            "n_predict": 8 if size == 100 else 4, "model_files": [{"sha256": "b" * 64}],
            "mmproj_sha256": "c" * 64, "sampling": {"seed": 1234},
            "vocab_size": 11, "vocabulary": {"scheme": "llama-vocabulary-sha256-v1",
                "size": 11, "type": 2, "mapping": "d" * 64, "attributes": "e" * 64},
            "decoding": {"method": "autoregressive", "logprob_source": "target_raw_logits",
                         "token_source": "target_accepted"},
            "chat_template": "中文 template", "requested_enable_thinking": False,
            "chat_template_source": "gguf", "chat_template_kwargs": {}, "system_prompt": "",
            "default_enable_thinking": False, "supports_enable_thinking": True,
            "thinking_column": None, "jinja": True, "add_bos": False, "bos_token_id": 0,
            "eos_token_id": 1, "eot_token_id": 1, "media_marker": "image", "image_placeholder_id": -1,
            "image_token_budget_source": "mtmd_init_params", "n_ctx": 100, "n_ctx_per_seq": 100,
            "n_batch": 32, "n_ubatch": 32, "n_threads": 8, "n_threads_batch": 8, "total_slots": 1,
            "execution_identity": {"libraries": [{"name": "libggml.so", "sha256": "a" * 64}]},
            "image_min_tokens": -1, "image_max_tokens": -1,
            "binary_sha256": "f" * 64, "repetition_detector": {"enabled": True}}


def write_native(root, ids, gen, side, part, skipped=()):
    records = {}
    # Different part means and sample sizes catch averaging the two part means.
    for i, item_id in enumerate(ids):
        rec = _make_item(npos=4, seed=i, ref_seed=i)
        rec["kld"] = 1 if side == "a" else (1.1 + i * .04 + (part == "tail") * .4)
        records[f"{i:03d}_{item_id}"] = rec
    meta = dict(BASE_META, dataset=part, subset=part, dataset_content_hash=f"ds-v3:{part}",
                cand_model=f"/m/{side}.gguf", cand_model_fingerprint=f"fp:{side}",
                cand_mmproj_fingerprint="fp:mm", max_total_tokens=None, n_ctx=8192,
                n_gpu_layers=99, n_threads=8, metric_threads=8, flash_attn=True,
                allow_vocab_attr_mismatch=False, allow_prefix_drift=False)
    _write_metrics_dir(root, records, meta, skipped=skipped)
    attempt = next((root / ".attempts").iterdir())
    (attempt / "generators").mkdir()
    digest = canonical_digest(gen)
    (attempt / "generators" / f"{digest}.json").write_text(json.dumps(gen))
    lines = [{"row_idx": i, "item_id": item, "generator_sha256": digest, "row_settings": {
                "generation_request": json.dumps({"id": item, "enable_thinking": False}),
                "generation_enable_thinking": False, "generation_sampling_params": json.dumps(gen["sampling"]),
                "generation_chat_template_kwargs": json.dumps({"enable_thinking": "false"}),
                "generation_token_logprobs": [-1.] * 4}}
             for i, item in enumerate(ids) if i not in skipped]
    (attempt / "references.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))


@pytest.fixture
def combined(tmp_path):
    ids_p, ids_t = ["id0", "id2"], ["id100", "id102", "id105"]
    roots = {}
    for part, ids, size in (("pilot", ids_p, 100), ("tail", ids_t, 500)):
        gen = generator(size, ids if size == 100 else ["id0", *ids])
        for side in ("a", "b"):
            roots[part + side] = root = tmp_path / (part + side)
            write_native(root, ids, gen, side, part)
    args = ["--candidate-a", str(roots["taila"]), "--candidate-b", str(roots["tailb"]),
            "--pilot-candidate-a", str(roots["pilota"]), "--pilot-candidate-b", str(roots["pilotb"]),
            "--num-eval-tokens", "4", "--bootstrap-iters", "100", "--omit-host-metadata",
            "--out", str(tmp_path / "result.txt"), "--output-json", str(tmp_path / "result.json")]
    return roots, args, tmp_path / "result.json"


def test_combine_once_equal_item_oracle(combined, monkeypatch):
    roots, args, output = combined
    engine = cli.compare_items
    calls = []
    def check(a, b, weights, **kwargs):
        calls.append((a, b, weights, kwargs))
        delta = np.array([y["kld"] - x["kld"] for x,y in zip(a,b)])
        expected = np.array([.1,.14,.5,.54,.58])
        np.testing.assert_allclose(delta, expected, atol=1e-7)
        assert len(delta) == 5
        assert kwargs["item_keys"] == ["pilot:000_id0", "pilot:001_id2", "tail:000_id100", "tail:001_id102", "tail:002_id105"]
        actual = stats.ttest_1samp(delta, 0)
        assert actual.statistic == pytest.approx(delta.mean() / (delta.std(ddof=1) / np.sqrt(5)))
        return engine(a,b,weights,**kwargs)
    monkeypatch.setattr(cli, "compare_items", check)
    assert cli.main(args) == 0
    assert len(calls) == 1
    result = json.loads(output.read_text())
    assert result["alignment"]["n_statistical_units"] == 5
    assert result["inputs"]["dataset"] == "pilot100+tail400"
    assert [p["generation_cap"] for p in result["inputs"]["parts"]] == [8,4]
    assert [p["cohort"]["eligible"] for p in result["inputs"]["parts"]] == [2,3]
    a,b,_,_ = calls[0]
    delta = np.array([y["kld"]-x["kld"] for x,y in zip(a,b)])
    oracle = stats.ttest_1samp(delta, 0)
    item = result["metrics"]["kld"]["item_weighted"]
    assert item["p_value"] == pytest.approx(oracle.pvalue)
    assert item["delta_candidate_minus_baseline"] == pytest.approx(delta.mean())
    assert item["ci_delta"]["lower"] == pytest.approx(oracle.confidence_interval().low)
    token = result["metrics"]["kld"]["token_weighted"]
    assert token["role"] == "descriptive" and "p_value" not in token and "ci_delta" not in token


@pytest.mark.parametrize("field,value", [("cand_model_fingerprint", "different"),
    ("cand_mmproj_fingerprint", None), ("num_eval_tokens", 2), ("tf_chunk", 32)])
def test_reject_cross_part_candidate_and_scoring_drift(combined, field, value):
    roots,args,_ = combined
    for side in ("a", "b"):
        path = roots["pilot"+side] / "collect_meta.json"
        meta = json.loads(path.read_text()); meta[field] = value
        path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="pilot/tail|scoring capacity"): cli.main(args)


def test_reject_tampered_generator(combined):
    roots,args,_ = combined
    path = next((roots["pilota"] / ".attempts").glob("*/generators/*.json"))
    gen = json.loads(path.read_text()); gen["n_predict"] = 9
    path.write_text(json.dumps(gen))
    with pytest.raises(SystemExit, match="SHA256"): cli.main(args)


def test_reject_missing_eligible_metric_even_when_both_pairs_complete(combined):
    roots,args,_ = combined
    for side in ("a", "b"):
        gen = generator(500, ["id100", "id102", "id105", "id107"])
        root = roots["tail"+side]
        import shutil
        shutil.rmtree(root)
        write_native(root, ["id100", "id102", "id105"], gen, side, "tail")
    with pytest.raises(SystemExit, match="exact native eligible"): cli.main(args)


def test_reject_missing_pilot_flag_and_all_tokens(combined):
    _,args,_ = combined
    with pytest.raises(SystemExit, match="required together"): cli.main(args[:6] + args[8:])
    args[args.index("--num-eval-tokens") + 1] = "-1"
    with pytest.raises(SystemExit, match="explicit positive"): cli.main(args)


def test_tail_uses_requested_positions_and_preserves_exclusions():
    tail = tail_cohort(cohort(100, ["id2"]), cohort(500, ["id0", "id100", "id499"]))
    assert tail["eligible_ids"] == ["id100", "id499"]
    assert tail["excluded"] == 398


@pytest.mark.parametrize("flag", ["--out", "--output-json"])
def test_comparison_cannot_overwrite_source(combined, flag):
    roots,args,_ = combined
    path = roots["taila"] / "collect_meta.json"
    original = path.read_bytes()
    args[args.index(flag)+1] = str(path)
    with pytest.raises(SystemExit, match="outside all source"): cli.main(args)
    assert path.read_bytes() == original


def test_common_budget_skip_is_reported_not_silently_dropped(combined):
    roots,args,out = combined
    import shutil
    for side in ("a", "b"):
        root = roots["tail"+side]; shutil.rmtree(root)
        write_native(root, ["id100", "id102", "id105"], generator(500, ["id100", "id102", "id105"]), side, "tail", skipped=(1,))
    assert cli.main(args) == 0
    result=json.loads(out.read_text())
    assert result["n_items"] == 4
    assert result["alignment"]["common_skipped_over_budget"] == ["tail:1"]


def test_reject_both_sides_truncated_metrics(combined):
    roots,args,_ = combined
    for side in ("a", "b"):
        path = roots["tail"+side] / "metrics/000_id100.npz"
        with np.load(path) as data:
            arrays={k: data[k] for k in data.files}
        # The archived format's records and npos must agree internally, yet
        # composition must detect disagreement with the native generation length.
        for key,value in arrays.items():
            if value.ndim and len(value) == 4: arrays[key] = value[:2]
        if "npos" in arrays: arrays["npos"] = np.array(2)
        np.savez(path, **arrays)
    with pytest.raises(SystemExit, match="stored header differs"): cli.main(args)


def test_reject_same_wrong_vocabulary_on_both_pilot_sides(combined):
    roots,args,_ = combined
    for side in ("a","b"):
        path=roots["pilot"+side]/"metrics/000_id0.npz"
        with np.load(path) as data: arrays={k:data[k] for k in data.files}
        arrays["vocab"]=np.array(22)
        np.savez(path,**arrays)
    with pytest.raises(SystemExit,match="stored header differs"): cli.main(args)


@pytest.fixture
def cross_host_combined(combined):
    roots, args, output = combined
    for name, root in roots.items():
        path = root / "collect_meta.json"
        meta = json.loads(path.read_text())
        digit = "1" if name.startswith("pilot") else "2"
        meta["execution_identity"].update(
            libraries=[{"name": "libggml.so", "sha256": "a" * 64}],
            gpu=[f"NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-{digit * 8}-1111-1111-1111-111111111111, 595.71.05"])
        path.write_text(json.dumps(meta))
    return roots, args, output


def test_cross_host_composition_requires_opt_in_and_retains_raw_metadata(cross_host_combined):
    roots, args, output = cross_host_combined
    before = {name: (root / "collect_meta.json").read_bytes() for name, root in roots.items()}
    with pytest.raises(SystemExit, match="identity differs"):
        cli.main(args)
    assert cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"]) == 0
    result = json.loads(output.read_text())
    assert result["inputs"]["cross_part_execution_policy"] == "same-gpu-model-v1"
    for part in result["inputs"]["parts"]:
        for side in ("a", "b"):
            assert part[f"candidate_{side}_meta"] == json.loads(before[part["part"] + side])
    assert "does not establish numerical or bitwise equivalence" in output.with_suffix(".txt").read_text()
    assert before == {name: (root / "collect_meta.json").read_bytes() for name, root in roots.items()}


def test_cross_host_opt_in_never_relaxes_within_part_pairing(cross_host_combined):
    roots, args, _ = cross_host_combined
    path = roots["tailb"] / "collect_meta.json"
    meta = json.loads(path.read_text())
    meta["execution_identity"]["gpu"][0] = meta["execution_identity"]["gpu"][0].replace("22222222", "33333333")
    path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="identity differs"):
        cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"])


@pytest.mark.parametrize("field,value", [("ref_model", "/relocated/ref.gguf"), ("ref_mmproj", "/relocated/mmproj.gguf"),
                                       ("tf_chunk", 32), ("n_threads", 12), ("cand_model_fingerprint", "changed")])
def test_cross_host_opt_in_keeps_path_model_and_scoring_guards(cross_host_combined, field, value):
    roots, args, _ = cross_host_combined
    for side in ("a", "b"):
        path = roots["tail" + side] / "collect_meta.json"
        meta = json.loads(path.read_text())
        meta[field] = value
        path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="pilot/tail"):
        cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"])


def test_cross_host_policy_requires_pilot_collections(combined):
    _, args, _ = combined
    args = [*args[:4], *args[8:], "--cross-part-execution-policy", "same-gpu-model-v1"]
    with pytest.raises(SystemExit, match="requires both pilot"):
        cli.main(args)


def test_cross_part_metric_worker_counts_require_opt_in(cross_host_combined):
    roots, args, output = cross_host_combined
    for side in ("a", "b"):
        pilot = json.loads((roots["pilot" + side] / "collect_meta.json").read_text())
        path = roots["tail" + side] / "collect_meta.json"
        meta = json.loads(path.read_text())
        meta["execution_identity"] = pilot["execution_identity"]
        meta["metric_threads"] = 12
        path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="scoring configuration differs: metric_threads"):
        cli.main(args)
    assert cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"]) == 0
    parts = json.loads(output.read_text())["inputs"]["parts"]
    assert [p["candidate_a_meta"]["metric_threads"] for p in parts] == [8, 12]


@pytest.mark.parametrize("count", [-1, 0, True, "12"])
def test_cross_part_metric_worker_counts_must_be_positive_integers(cross_host_combined, count):
    roots, args, _ = cross_host_combined
    for side in ("a", "b"):
        path = roots["tail" + side] / "collect_meta.json"
        meta = json.loads(path.read_text())
        meta["metric_threads"] = count
        path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="requires positive metric_threads"):
        cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"])


def test_cross_part_metric_worker_exception_never_relaxes_within_part_pairing(cross_host_combined):
    roots, args, _ = cross_host_combined
    path = roots["tailb"] / "collect_meta.json"
    meta = json.loads(path.read_text())
    meta["metric_threads"] = 12
    path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="collect_meta mismatch on 'metric_threads'"):
        cli.main([*args, "--cross-part-execution-policy", "same-gpu-model-v1"])
