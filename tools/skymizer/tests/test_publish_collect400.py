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
        "--profiles","profiles.json","--model",model,"--source","mmstar","--mode","instruct",
        "--candidate","Q4","--cand-model","candidate.gguf","--llama-vlm-kld","scorer","--gpu","0"])
    plan={"stage":"kld","size":500,"reference_tail_400":True,"num_samples":None,"hardware":"pro6000",
          "models":[model],"sources":["mmstar"],"modes":["instruct"],"models_dir":str(tmp_path)}
    command,out=launch.build_command(args,plan,profiles,tmp_path)
    assert command[command.index("--dataset")+1]==f"elichen-skymizer/{model}-collect-400"
    assert command[command.index("--subset")+1]=="mmstar-tail-400-ins"
    assert "mmstar-tail-400-ins" in str(out)
    overview=study_overview(profiles,plan)
    assert overview["source"]["requested_per_job"]==400
    assert overview["models"][model]["hf_repo"].endswith("-collect-400")
    args.dataset=tmp_path/"full500"
    with pytest.raises(ValueError,match="published filtered config"):
        launch.build_command(args,plan,profiles,tmp_path)
