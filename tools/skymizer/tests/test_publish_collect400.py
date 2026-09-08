"""Native cohort publication filters and mutation-free stage preflight checks."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from cli.publish_collect400 import publish, validate_stage
from lib.reference_cohort import tail_cohort
from test_collection_parts import generator


@pytest.fixture
def staged(tmp_path):
    directory = tmp_path / "stage"
    entry = {"config": "source-tail-400-ins", "parent_config": "source-subsample-500-ins",
             "pilot_config": "source-subsample-100-ins", "path": "source-tail-400-ins/parent500.parquet",
             "source_path": "source-subsample-500-ins/train-00000-of-00001.parquet", "parquet_sha256": "1" * 64}
    plan = {"schema": "skymizer-collect400-view-v1", "destination": "elichen-skymizer/model-collect-400",
            "parent_repo": "elichen-skymizer/model-collect-500", "parent_revision": "a" * 40,
            "pilot_repo": "elichen-skymizer/model-pilot", "pilot_revision": "b" * 40, "configs": [entry]}
    for label,size,ids in (("parent",500,["id0","id100","id105"]),("pilot",100,["id0"])):
        gen=generator(size,ids)
        gen["dataset_source"]["path"]="elichen-skymizer/vlm-prepared-dataset"
        raw=json.dumps(gen).encode(); digest=hashlib.sha256(raw).hexdigest()
        manifest={"repo":plan[label+"_repo"], "subset":entry[label+"_config"], "split":"train", "rows":len(ids),
                  "cohort":gen["cohort"], "source":gen["dataset_source"], "metadata_sha256":digest,
                  "audit_sha256":{"metadata.json":digest}, "parquet_sha256":"1"*64}
        entry[label+"_audit"]=manifest; entry[label+"_generation_cap"]=gen["n_predict"]
        out=directory/"audit"/entry["config"]/label; out.mkdir(parents=True)
        (out/"manifest.json").write_text(json.dumps(manifest)); (out/"metadata.json").write_bytes(raw)
    entry["source"]=entry["parent_audit"]["source"]
    entry["cohort"]=tail_cohort(entry["pilot_audit"]["cohort"],entry["parent_audit"]["cohort"])
    card={"viewer":False,"configs":[{"config_name":entry["config"],"data_files":[{"split":"train","path":entry["path"]}],
                                    "filters":[["item_id","in",entry["cohort"]["requested_ids"]]]}]}
    (directory/"README.md").write_text("---\n"+yaml.safe_dump(card)+"---\nTail400 filtered view.\n")
    (directory/"composition.json").write_text(json.dumps(plan))
    return directory,plan


def test_real_datasets_loader_honors_filter_normal_and_streaming(staged):
    from datasets import Dataset, load_dataset
    directory,plan=staged
    assert len(validate_stage(directory,plan)) == 6
    # Standalone local dataset card with the exact published builder parameters.
    entry=plan["configs"][0]
    path=directory/entry["path"]; path.parent.mkdir()
    Dataset.from_dict({"item_id":["id0","id100","id105"],"value":[1,2,3]}).to_parquet(path)
    normal=load_dataset(str(directory),entry["config"],split="train")
    streamed=load_dataset(str(directory),entry["config"],split="train",streaming=True,columns=["item_id"])
    assert normal["item_id"] == ["id100","id105"]
    assert [r["item_id"] for r in streamed] == ["id100","id105"]


@pytest.mark.parametrize("mutation", ["destination","revision","source","cap","tail","filters","extra","symlink"])
def test_staged_edits_fail_before_any_hub_write(staged,mutation):
    directory,plan=staged
    entry=plan["configs"][0]
    if mutation=="destination": plan["destination"]=plan["parent_repo"]
    if mutation=="revision": plan["parent_revision"]="main"
    if mutation=="source": entry["source"]={**entry["source"],"revision":"c"*40}
    if mutation=="cap": entry["pilot_generation_cap"]=999
    if mutation=="tail": entry["cohort"]["eligible_ids"].append("id0")
    if mutation=="filters":
        path=directory/"README.md"; path.write_text(path.read_text().replace("filters:","ignored_filters:"))
    if mutation=="extra": (directory/"private-notes.txt").write_text("unrelated")
    if mutation=="symlink":
        path=directory/"audit"/entry["config"]/"pilot/metadata.json"
        other=directory.parent/"external.json"; other.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(other)
    with pytest.raises(ValueError): validate_stage(directory,plan)


def test_existing_destination_is_never_modified(staged,monkeypatch):
    directory,_=staged
    import huggingface_hub
    calls=[]
    class API:
        def __init__(self,**kwargs): pass
        def dataset_info(self,repo): calls.append(repo); return SimpleNamespace(private=True)
        def create_repo(self,*args,**kwargs): pytest.fail("must not mutate an existing destination")
    monkeypatch.setattr(huggingface_hub,"HfApi",API)
    with pytest.raises(ValueError,match="refusing to overwrite"): publish(directory,"test-token")
    assert calls==["elichen-skymizer/model-collect-400"]


def test_tail_launcher_uses_500_profile_and_new_repo(tmp_path):
    from pathlib import Path
    from cli import collect_model_kld as launch
    from lib.reference_study import study_overview
    profiles=json.loads((Path(__file__).resolve().parents[1]/"profiles/small-collect500.json").read_text())
    model=next(iter(profiles["models"]))
    args=launch.parse_args(["--study",str(tmp_path/"new-study"),"--size","500","--tail-400",
        "--reference-revision", "a" * 40,
        "--profiles","profiles.json","--model",model,"--source","mmstar","--mode","instruct",
        "--candidate","Q4","--cand-model","candidate.gguf","--llama-vlm-kld","scorer","--gpu","0"])
    plan={"stage":"kld","size":500,"reference_tail_400":True,"num_samples":None,"hardware":"pro6000",
          "models":[model],"sources":["mmstar"],"modes":["instruct"],"models_dir":str(tmp_path)}
    command,out=launch.build_command(args,plan,profiles,tmp_path)
    assert command[command.index("--dataset")+1]==str(tmp_path/"new-study/references"/model/"mmstar-tail-400-ins/dataset")
    assert command[command.index("--subset")+1]==""
    assert "mmstar-tail-400-ins" in str(out)
    overview=study_overview(profiles,plan)
    assert overview["source"]["requested_per_job"]==400
    assert overview["models"][model]["hf_repo"].endswith("-collect-400")
    args.dataset=tmp_path/"full500"
    with pytest.raises(ValueError,match="pinned materialized config"):
        launch.build_command(args,plan,profiles,tmp_path)


@pytest.mark.parametrize("revision", [None, "main", "a" * 39, "a" * 41])
def test_tail_launcher_requires_immutable_revision(tmp_path, revision):
    from cli import collect_model_kld as launch
    argv = ["--study", str(tmp_path), "--size", "500", "--tail-400", "--profiles", "profile.json",
            "--model", "qwen3.5-4b", "--source", "mmstar", "--mode", "instruct",
            "--candidate", "Q4", "--cand-model", "candidate.gguf", "--llama-vlm-kld", "scorer", "--gpu", "0"]
    if revision is not None:
        argv += ["--reference-revision", revision]
    with pytest.raises(SystemExit):
        launch.parse_args(argv)


def test_tail_freeze_only_validates_before_scoring(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from cli import collect_model_kld as launch
    profiles = json.loads((Path(__file__).resolve().parents[1] / "profiles/small-collect500.json").read_text())
    model = next(iter(profiles["models"]))
    argv = ["--study", str(tmp_path), "--size", "500", "--tail-400", "--reference-revision", "a" * 40,
            "--freeze-only", "--metric-threads", "12", "--profiles", "profile.json", "--model", model,
            "--source", "mmstar", "--mode", "instruct", "--candidate", "Q4", "--cand-model", "candidate.gguf",
            "--llama-vlm-kld", "scorer", "--gpu", "0"]
    plan = {"size": 500, "reference_tail_400": True, "hardware": "pro6000", "metric_threads": 12,
            "models": [model], "sources": ["mmstar"], "modes": ["instruct"], "models_dir": str(tmp_path)}
    scripts = tmp_path / "scripts/skymizer"
    monkeypatch.setattr(sys, "argv", ["collect_model_kld.py", *argv])
    monkeypatch.setattr(launch, "prepare_study", lambda args: (tmp_path, scripts, profiles, plan))
    monkeypatch.setattr(launch, "archived_dispatch", lambda study: None)
    monkeypatch.setattr(launch, "ProcessSupervisor", lambda *args: pytest.fail("freeze-only must not score"))
    calls = []
    def freeze(*args):
        calls.append(args)
        return {"revision": args[3], "dataset": "dataset"}
    monkeypatch.setitem(sys.modules, "lib.reference_freeze", SimpleNamespace(freeze_reference=freeze))
    assert launch.main() == 0
    assert calls == [(scripts / "profiles/reference_model_profiles.json", model,
                      f"elichen-skymizer/{model}-collect-400", "a" * 40, "mmstar-tail-400-ins",
                      tmp_path / "references" / model / "mmstar-tail-400-ins", None)]
    assert not (tmp_path / "artifacts").exists()
    command, _ = launch.build_command(launch.parse_args(argv), plan, profiles, scripts)
    assert command[command.index("--metric-threads") + 1] == "12"
    assert command[command.index("--n-threads") + 1] == "8"
    def fail(*args):
        raise ValueError("Parquet SHA256 mismatch")
    monkeypatch.setattr(sys.modules["lib.reference_freeze"], "freeze_reference", fail)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launch.main()
    assert not (tmp_path / "artifacts").exists()


@pytest.fixture
def materialized_freeze(staged, tmp_path, monkeypatch):
    from copy import deepcopy
    from urllib.parse import unquote
    import huggingface_hub
    from datasets import Dataset
    from cli.upload_reference import DETECTOR
    from lib.reference_dataset import reference_features, sha256_file
    from test_gap_sample_prep import native_row

    directory, plan = staged
    plan["schema"] = "skymizer-collect400-materialized-v1"
    entry = plan["configs"][0]
    entry["path"] = f"{entry['config']}/train-00000-of-00001.parquet"
    entry["source_parquet_sha256"] = entry["parquet_sha256"]
    native = native_row()
    native_metadata = json.loads(native["generation_metadata"])
    for label in ("parent", "pilot"):
        base = directory / "audit" / entry["config"] / label
        metadata = {**native_metadata, **json.loads((base / "metadata.json").read_text()),
                    "model_path": "/models/test.gguf", "mmproj_path": "/models/projector.gguf",
                    "model_files": [{"path": "/models/test.gguf", "size": 32, "sha256": "b" * 64}],
                    "chat_template_kwargs": {"preserve_reasoning": "true"}, "repetition_detector": DETECTOR}
        raw = json.dumps(metadata).encode()
        manifest = entry[label + "_audit"]
        manifest["metadata_sha256"] = manifest["audit_sha256"]["metadata.json"] = hashlib.sha256(raw).hexdigest()
        (base / "metadata.json").write_bytes(raw)
        (base / "manifest.json").write_text(json.dumps(manifest))
        if label == "parent":
            parent = metadata
    rows = []
    for item_id in entry["cohort"]["eligible_ids"]:
        row = deepcopy(native)
        row.update(id=item_id, item_id=item_id, generation_metadata=json.dumps(parent),
                   generation_request=json.dumps({"id": item_id, "question": row["question"], "enable_thinking": False}),
                   generation_chat_template_kwargs=json.dumps({"preserve_reasoning": "true", "enable_thinking": "false"}))
        rows.append(row)
    parquet = directory / entry["path"]
    parquet.parent.mkdir()
    Dataset.from_list(rows, features=reference_features()).to_parquet(parquet)
    entry["parquet_sha256"] = sha256_file(parquet)
    (directory / "composition.json").write_text(json.dumps(plan))
    profile = {"model": "test.gguf", "identity": {"files": [
        {"role": "llm", "name": "test.gguf", "size": 32, "sha256": "b" * 64},
        {"role": "mmproj", "name": "projector.gguf", "size": 12, "sha256": "c" * 64}]},
        "effective_sampling": {"instruct": parent["sampling"]},
        "runtime": {"pro6000": {"instruct": {"ctx": 100, "batch": 32, "ubatch": 32, "threads": 8}}}}
    profiles = {"cohort_size": 500, "sources": ["source"], "models": {"model": profile}, "seed": 1234,
                "generation_caps": {"instruct": 4},
                "dataset": {"repo": parent["dataset_source"]["path"], "revision": "a" * 40}}
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profiles))
    revision = "d" * 40
    calls = []
    def download(repo, name, **kwargs):
        assert repo == plan["destination"]
        assert kwargs["revision"] == revision and kwargs["repo_type"] == "dataset"
        calls.append(name)
        return str(directory / name)
    def file_metadata(url):
        name = unquote(url.split(f"/resolve/{revision}/", 1)[1])
        raw = (directory / name).read_bytes()
        return SimpleNamespace(commit_hash=revision, etag=hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", file_metadata)
    kwargs = {"profiles_path": profile_path, "model": "model", "repo": plan["destination"], "revision": revision,
              "subset": entry["config"], "out": tmp_path / "freeze", "cache_dir": tmp_path / "cache"}
    return kwargs, directory, plan, calls


def test_materialized_freeze_preserves_parent_rows_and_reuses_offline(materialized_freeze, tmp_path, monkeypatch):
    import shutil
    import huggingface_hub
    from datasets import load_from_disk
    from lib.reference_freeze import freeze_reference
    kwargs, directory, plan, calls = materialized_freeze
    receipt = freeze_reference(**kwargs)
    dataset = load_from_disk(str(kwargs["out"] / receipt["dataset"]))
    assert list(dataset["id"]) == ["id100", "id105"]
    assert receipt["cohort"]["requested"] == 400 and receipt["cohort"]["excluded"] == 398
    assert json.loads(dataset[0]["generation_metadata"])["cohort"]["requested"] == 500
    assert json.loads(dataset[0]["generation_metadata"])["n_predict"] == 4
    assert receipt["dataset"] == "dataset" and len(calls) == 6
    assert all("parent500.parquet" not in name for name in calls)
    from cli.prep_vlm_score_from_hf import load_dataset_sorted
    sorted_dataset = load_dataset_sorted(str(kwargs["out"] / "dataset"), None, "train", "id", sort_desc=True)
    assert list(sorted_dataset["id"]) == ["id105", "id100"]
    assert freeze_reference(**kwargs) == receipt
    moved = tmp_path / "new-machine"
    shutil.copytree(kwargs["out"], moved)
    kwargs["out"] = moved
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("reuse must work offline"))
    assert freeze_reference(**kwargs) == receipt


@pytest.mark.parametrize("mutation", ["revision", "profile", "arrow", "missing", "extra", "symlink", "partial"])
def test_freeze_reuse_rejects_incomplete_or_changed_files(materialized_freeze, mutation):
    from lib.reference_freeze import freeze_reference
    kwargs, _, _, _ = materialized_freeze
    freeze_reference(**kwargs)
    out = kwargs["out"]
    if mutation == "revision": kwargs["revision"] = "e" * 40
    if mutation == "profile": kwargs["profiles_path"].write_text(kwargs["profiles_path"].read_text() + "\n")
    if mutation == "arrow":
        with next((out / "dataset").glob("*.arrow")).open("ab") as stream: stream.write(b"changed")
    if mutation == "missing": (out / "dataset/dataset_info.json").unlink()
    if mutation == "extra": (out / "dataset/untracked.arrow").write_bytes(b"changed")
    if mutation == "symlink":
        path = out / "dataset/dataset_info.json"
        target = out.parent / "outside.json"
        target.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(target)
    if mutation == "partial": (out / "freeze-receipt.json").unlink()
    with pytest.raises(ValueError): freeze_reference(**kwargs)


@pytest.mark.parametrize("mutation", ["parquet_hash", "ids", "image", "tokens", "metadata", "cap", "view"])
def test_materialized_freeze_rejects_invalid_source(materialized_freeze, mutation):
    from datasets import Dataset
    import pyarrow.parquet as pq
    from lib.reference_dataset import reference_features, sha256_file
    from lib.reference_freeze import freeze_reference
    kwargs, directory, plan, _ = materialized_freeze
    entry = plan["configs"][0]
    path = directory / entry["path"]
    if mutation in ("ids", "image", "tokens", "metadata"):
        rows = pq.read_table(path).to_pylist()
        if mutation == "ids": rows.reverse()
        if mutation == "image": rows[0]["images"][0]["bytes"] += b"changed"
        if mutation == "tokens": rows[0]["input_ids"][-1] = 9999
        if mutation == "metadata":
            metadata = json.loads(rows[0]["generation_metadata"])
            metadata["n_predict"] = 400
            rows[0]["generation_metadata"] = json.dumps(metadata)
        Dataset.from_list(rows, features=reference_features()).to_parquet(path)
        entry["parquet_sha256"] = sha256_file(path)
    if mutation == "parquet_hash": entry["parquet_sha256"] = "f" * 64
    if mutation == "cap":
        profiles = json.loads(kwargs["profiles_path"].read_text())
        profiles["generation_caps"]["instruct"] = 400
        kwargs["profiles_path"].write_text(json.dumps(profiles))
    if mutation == "view": plan["schema"] = "skymizer-collect400-view-v1"
    (directory / "composition.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError): freeze_reference(**kwargs)
    assert not (kwargs["out"] / "freeze-receipt.json").exists()


def test_freeze_rejects_cached_audit_with_wrong_hub_blob(materialized_freeze, monkeypatch):
    import huggingface_hub
    from lib.reference_freeze import freeze_reference
    kwargs, _, _, _ = materialized_freeze
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda url:
                        SimpleNamespace(commit_hash=kwargs["revision"], etag="e" * 40))
    with pytest.raises(ValueError, match="pinned audit content mismatch"):
        freeze_reference(**kwargs)
    assert not (kwargs["out"] / "freeze-receipt.json").exists()


@pytest.mark.parametrize("mutation", ["request_mode", "request_mode_type", "request_mode_missing", "request_mode_null", "over_cap"])
def test_freeze_rejects_rows_incompatible_with_composition(materialized_freeze, mutation):
    from datasets import Dataset
    import pyarrow.parquet as pq
    from lib.reference_dataset import reference_features, sha256_file, validate_reference_row
    from lib.reference_freeze import freeze_reference
    kwargs, directory, plan, _ = materialized_freeze
    entry = plan["configs"][0]
    path = directory / entry["path"]
    rows = pq.read_table(path).to_pylist()
    row = rows[0]
    if mutation.startswith("request_mode"):
        request = json.loads(row["generation_request"])
        if mutation == "request_mode_missing":
            request.pop("enable_thinking")
        else:
            request["enable_thinking"] = {"request_mode": True, "request_mode_type": 0, "request_mode_null": None}[mutation]
        row["generation_request"] = json.dumps(request)
        error = "request thinking mode differs"
    else:
        cap = json.loads(row["generation_metadata"])["n_predict"]
        extra = cap + 1 - row["generated_tokens_len"]
        row["input_ids"] += [row["input_ids"][-1]] * extra
        row["labels"] += [row["labels"][-1]] * extra
        row["generation_token_logprobs"] += [row["generation_token_logprobs"][-1]] * extra
        row["input_tokens_len"] += extra
        row["generated_tokens_len"] += extra
        error = "generated length exceeds"
    validate_reference_row(row)
    Dataset.from_list(rows, features=reference_features()).to_parquet(path)
    entry["parquet_sha256"] = sha256_file(path)
    (directory / "composition.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=error):
        freeze_reference(**kwargs)
    assert not (kwargs["out"] / "freeze-receipt.json").exists()
