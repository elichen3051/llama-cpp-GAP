"""Prep-contract tests over REAL GAP collection rows.

tests/data/gap-mmmu-pro-standard-10-qwen3.5-4b-ins/ holds the first two rows
of `elichen-skymizer/GAP-mmmu-pro-standard-10 :: qwen3.5-4b-ins-gen-2048`;
tests/data/gap-ocrbench-v1-kimi-vl-a3b-ins/ holds three rows of
`elichen-skymizer/kimi-vl-a3b-ins-full-ref-text-rec-dec ::
ocrbench-v1-subsample-100-ins` (revision 4790137, 2026-09-03): the two
smallest-image rows (ocr-00081, ocr-00034: 3 and 8 vision tokens, at or
below mtmd's built-in kimivl floor of 8) and the smallest row whose HF
vision-token count exceeds the derived cap (ocr-00382: 1089 > 1024).
Both vendored verbatim with datasets.Dataset.save_to_disk. Test data is real
collection samples, not script-generated look-alikes: these rows carry the
actual producer schema (frozen input_ids/labels, n_prefill_tokens, the
per-row image_processor_config, real images), so the prep contract is pinned
against what collection runs actually consume -- once per registered
image-block scheme (Qwen-style wrappers, Kimi-VL's media block).

The schema/meta tests are hermetic (no tokenizer, no network). The full
prep_row test additionally needs the generating model's tokenizer and is
marked external_model (self-skips when the tokenizer cannot be loaded).
"""

import json
import math
import re
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

import cli.prep_vlm_score_from_hf as prep
from lib.dataset_fingerprint import dataset_content_hash

DATA = Path(__file__).parent / "data"
SAMPLES = ["gap-mmmu-pro-standard-10-qwen3.5-4b-ins", "gap-ocrbench-v1-kimi-vl-a3b-ins"]

# Documented (derive_image_token_limits) slack for families whose HF resize
# pads the grid past the derived cap; every other family's cap bounds rows.
CAP_SLACK = {"moonshotai/Kimi-VL": 1.2}


class Sample(NamedTuple):
    ds: object
    rows: list


@pytest.fixture(scope="module", params=SAMPLES, ids=SAMPLES)
def sample(request):
    datasets = pytest.importorskip("datasets")
    ds = datasets.load_from_disk(str(DATA / request.param), keep_in_memory=True)
    return Sample(ds, [ds[i] for i in range(len(ds))])


def cap_slack_for(model_name):
    return next((v for k, v in CAP_SLACK.items() if model_name.startswith(k)), 1.0)


def test_sample_is_present_and_nonempty(sample):
    assert len(sample.rows) >= 2
    ids = [r["item_id"] for r in sample.rows]
    assert len(set(ids)) == len(ids)


def test_real_rows_pass_the_prep_sanity_contract(sample):
    """input_ids/labels/n_prefill_tokens relations (R4-R6) on real producer
    rows — the contract collect_kld.py enforces before scoring anything."""
    for row in sample.rows:
        prep.sanity_check_row(row)   # raises PrepError on violation
        n_pre = row["n_prefill_tokens"]
        assert 0 < n_pre < row["input_tokens_len"]
        assert all(int(x) == -100 for x in row["labels"][:n_pre])
        assert [int(x) for x in row["labels"][n_pre:]] == \
               [int(x) for x in row["input_ids"][n_pre:]]


def test_real_rows_have_a_registered_model_family(sample):
    for row in sample.rows:
        name = row["generation_model_name_or_path"]
        assert prep.image_block_regex_for(name) is not None
        assert prep.image_pad_token_for(name)


def test_real_rows_pad_runs_match_the_vision_token_counts(sample):
    """Tokenizer-free structural check of the producer's prefill layout: the
    id with the longest run is the image pad; it occurs in num_images
    maximal runs totalling sum_vision_tokens, the longest being
    max_vision_tokens; and it is the pad id the producer recorded, when it
    did. (The family regex that collapses those runs is exercised with a
    tokenizer in the external_model test.)"""
    for row in sample.rows:
        ids = [int(x) for x in row["input_ids"][: row["n_prefill_tokens"]]]
        runs = [(k, sum(1 for _ in g)) for k, g in groupby(ids)]
        pad = max(runs, key=lambda kv: kv[1])[0]
        pad_runs = [n for k, n in runs if k == pad]
        assert len(pad_runs) == row["num_images"], row["item_id"]
        assert sum(pad_runs) == row["sum_vision_tokens"], row["item_id"]
        assert max(pad_runs) == row["max_vision_tokens"], row["item_id"]
        cfg = json.loads(row["image_processor_config"])
        declared = cfg.get("kimi_vl_processor_config", {}).get("media_pad_token_id")
        if declared is not None:
            assert pad == declared, row["item_id"]


def test_build_meta_derives_image_token_limits(sample):
    for row in sample.rows:
        meta = prep.build_meta(row)
        assert meta["n_prefill"] == row["n_prefill_tokens"]
        assert meta["n_answer"] == row["generated_tokens_len"]
        assert meta["num_images"] == row["num_images"] == len(row["images"])
        limits = meta["image_token_limits"]
        # -1 = mtmd's own floor; otherwise a real, ordered bound
        assert (limits["image_min_tokens"] == -1
                or 1 <= limits["image_min_tokens"] <= limits["image_max_tokens"])
        # the row's own vision-token count sits inside the derived budget,
        # up to the family's documented grid-padding slack
        slack = cap_slack_for(row["generation_model_name_or_path"])
        assert row["max_vision_tokens"] <= math.ceil(limits["image_max_tokens"] * slack)


def test_kimi_vl_rows_do_exceed_the_derived_cap(sample):
    """Pins the documented Kimi-VL property (HF pads the grid past
    in_token_limit) on a real row, so the slack above is not a blanket
    relaxation: a Kimi-VL fixture must contain a row over the cap, and no
    other family's fixture may."""
    is_kimi = [r["generation_model_name_or_path"].startswith("moonshotai/Kimi-VL")
               for r in sample.rows]
    over = [r["item_id"] for r in sample.rows
            if r["max_vision_tokens"]
            > prep.build_meta(r)["image_token_limits"]["image_max_tokens"]]
    assert bool(over) == any(is_kimi), over


def test_dataset_content_hash_is_stable_on_the_real_rows(sample):
    """ds-v2 content hashing works on real rows and is order-sensitive —
    the identity guard's premise (NUMERICAL_CONTRACT #7)."""
    ds = sample.ds
    h1 = dataset_content_hash(ds)
    h2 = dataset_content_hash(ds)
    assert h1 == h2 and h1.startswith("ds-v3:")
    reversed_ds = ds.select([1, 0])
    assert dataset_content_hash(reversed_ds) != h1


@pytest.mark.external_model
def test_prep_row_end_to_end_on_real_rows(sample, tmp_path):
    """Full prep on the vendored rows: needs the generating model's tokenizer
    (HF cache or network); skips when it cannot be loaded."""
    name = sample.rows[0]["generation_model_name_or_path"]
    try:
        tok = prep.load_tokenizer(name)
    except Exception as e:
        pytest.skip(f"tokenizer {name} unavailable: {e}")
    for i, row in enumerate(sample.rows):
        out = tmp_path / f"row{i}"
        meta = prep.prep_row(row, tok, out)
        toks = np.fromfile(out / "tokens.bin", dtype=np.int32)
        assert toks.tolist() == [int(x) for x in row["input_ids"]]
        chat = (out / "formatted_chat.txt").read_text(encoding="utf-8")
        assert chat.count("<__media__>") == row["num_images"]
        # no family image-pad block survives the collapse
        block_re = prep.image_block_regex_for(name)
        assert not re.search(block_re, chat)
        assert len(list(out.glob("img_*.png"))) == row["num_images"]
        stored = json.loads((out / "meta.json").read_text())
        assert stored["n_prefill"] == meta["n_prefill"] == row["n_prefill_tokens"]


def native_row(with_image=True):
    from lib.reference_dataset import build_row
    prefix = [1, -1, -1, 2] if with_image else [1, 2]
    chunks = ([{"type": "text", "start": 0, "n_tokens": 1, "n_pos": 1, "tokens": [1]},
               {"type": "image", "start": 1, "n_tokens": 2, "n_pos": 1, "grid_x": 2, "grid_y": 1, "grid_t": 1},
               {"type": "text", "start": 3, "n_tokens": 1, "n_pos": 1, "tokens": [2]}]
              if with_image else [{"type": "text", "start": 0, "n_tokens": 2, "n_pos": 2, "tokens": prefix}])
    request = {"id": "image" if with_image else "text", "question": "What color?", "images": []}
    result = {
        "id": request["id"], "input_ids": prefix + [3, 4], "n_prefill_tokens": len(prefix),
        "n_past_prefill": 3 if with_image else 2,
        "prompt_layout": {"n_tokens": len(prefix), "n_pos": 3 if with_image else 2, "chunks": chunks},
        "content": "red", "finish_reason": "stop", "sampling": {"seed": 1234},
        "chat_template_kwargs": {}, "enable_thinking": False, "token_logprobs": [-0.5, -0.1],
        "prompt": "<__media__>What color?" if with_image else "What color?",
        "add_special": True, "stripped_leading_bos": False, "stop_type": "eos", "stopping_word": "",
    }
    metadata = {"model_path": "/models/test.gguf", "build_info": "test-build", "image_min_tokens": -1,
                "image_max_tokens": -1, "image_token_budget_source": "mtmd_init_params", "media_marker": "<__media__>",
                "vocab_size": 8,
                "vocabulary": {"scheme": "llama-vocabulary-sha256-v1", "size": 8, "type": 2,
                               "mapping": "a" * 64, "attributes": "b" * 64},
                "decoding": {"method": "autoregressive", "token_source": "target_accepted", "logprob_source": "target_raw_logits"}}
    import io
    from PIL import Image
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    return build_row({}, request, result, metadata, [image.getvalue()] if with_image else [])


@pytest.mark.parametrize("tiles_per_image", [(3,), (5, 13)])
@pytest.mark.parametrize("tile_separators", [False, True])
def test_native_contract_distinguishes_images_from_tiles(tiles_per_image, tile_separators):
    from copy import deepcopy
    from lib.reference_contract import GTContractError
    from lib.reference_dataset import validate_reference_row

    row = native_row()
    chunks, prefix = [], []

    def append_chunk(kind, count):
        chunk = {"type": kind, "start": len(prefix), "n_tokens": count, "n_pos": count}
        if kind == "text":
            chunk["tokens"] = [1] * count
        else:
            chunk.update(grid_x=16, grid_y=16, grid_t=1)
        prefix.extend(chunk.get("tokens", [-1] * count))
        chunks.append(chunk)

    append_chunk("text", 1)
    for count in tiles_per_image:
        for tile in range(count):
            if tile and tile_separators:
                append_chunk("text", 1)
            append_chunk("image", 256)
        append_chunk("text", 1)
    n_pre = len(prefix)
    n_tiles = sum(tiles_per_image)
    row.update(input_ids=prefix + [3, 4], labels=[-100] * n_pre + [3, 4],
               input_tokens_len=n_pre + 2, n_prefill_tokens=n_pre,
               llamacpp_n_past_prefill=n_pre, llamacpp_tokens_evaluated=n_pre,
               llamacpp_prompt_layout=json.dumps({"n_tokens": n_pre, "n_pos": n_pre, "chunks": chunks}),
               per_image_vision_token_counts=[256] * n_tiles, per_image_n_pos=[256] * n_tiles,
               per_image_grid=[[16, 16]] * n_tiles, sum_vision_tokens=256 * n_tiles,
               max_vision_tokens=256, num_images=len(tiles_per_image),
               images=row["images"] * len(tiles_per_image),
               image_bytes_sha256=row["image_bytes_sha256"] * len(tiles_per_image),
               llamacpp_prompt_string="<__media__>" * len(tiles_per_image) + "What color?")
    validate_reference_row(row)
    for field, value in (
        ("num_images", len(tiles_per_image) + 1),
        ("llamacpp_prompt_string", "What color?"),
        ("image_bytes_sha256", []),
        ("per_image_vision_token_counts", [256] * (n_tiles - 1)),
        ("input_ids", [7] + row["input_ids"][1:]),
    ):
        corrupted = deepcopy(row)
        corrupted[field] = value
        with pytest.raises(GTContractError):
            validate_reference_row(corrupted)


@pytest.mark.parametrize("corruption", ["bytes", "layout", "position", "vocab", "logprobs", "version"])
def test_native_contract_rejects_replay_corruption(corruption):
    from lib.reference_dataset import validate_reference_row
    from lib.reference_contract import GTContractError
    row = native_row()
    if corruption == "bytes":
        row["images"][0]["bytes"] += b"changed"
    elif corruption in ("layout", "position"):
        layout = json.loads(row["llamacpp_prompt_layout"])
        layout["chunks"][0]["tokens" if corruption == "layout" else "n_pos"] = [7] if corruption == "layout" else 9
        row["llamacpp_prompt_layout"] = json.dumps(layout)
    elif corruption == "vocab":
        row["input_ids"][-1] = row["labels"][-1] = 100
    elif corruption == "logprobs":
        row["generation_token_logprobs"][0] = float("nan")
    else:
        row["generation_schema_version"] = "future-version"
    with pytest.raises(GTContractError):
        validate_reference_row(row)


def test_native_dataset_storage_and_both_preppers_without_hf_tokenizer(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    from lib.reference_dataset import reference_features, validate_reference_row
    from cli import prep_llm_score_from_hf as llm_prep
    rows = [native_row(False), native_row(True)]
    path = tmp_path / "dataset"
    datasets.Dataset.from_list(rows, features=reference_features()).save_to_disk(str(path))
    monkeypatch.setattr(prep, "load_tokenizer", lambda *_: pytest.fail("HF tokenizer must not load"))
    ds = llm_prep.load_dataset_sorted(str(path), None, "train", "")
    for row in ds:
        validate_reference_row(row)
    llm_prep.prep_row(ds[0], tmp_path / "llm")
    row = ds[1]
    raw = [im["bytes"] for im in row["images"]]
    meta = prep.prep_row(row, None, tmp_path / "vlm", raw_images=raw)
    assert (tmp_path / "vlm" / meta["image_files"][0]).read_bytes() == raw[0]
    assert (tmp_path / "vlm/formatted_chat.txt").read_text() == row["llamacpp_prompt_string"]
    assert np.fromfile(tmp_path / "llm/tokens.bin", dtype=np.int32).tolist() == rows[0]["input_ids"]
    assert np.fromfile(tmp_path / "vlm/tokens.bin", dtype=np.int32).tolist() == row["input_ids"]
    with pytest.raises(llm_prep.PrepError, match="text-only"):
        llm_prep.prep_row(row, tmp_path / "wrong-lane")


@pytest.mark.parametrize("source", ["embedded", "sidecar"])
def test_mtp_reference_preserves_target_vocabulary_without_loading_head(tmp_path, source):
    from lib.reference_dataset import validate_reference_row
    from cli import prep_llm_score_from_hf as llm_prep
    row = native_row(False)
    metadata = json.loads(row["generation_metadata"])
    files = [{"path": "/not-present/head.gguf", "sha256": "c" * 64, "size": 12}]
    metadata["model_files"] = files
    metadata["decoding"] = {"method": "mtp", "token_source": "target_accepted", "logprob_source": "target_raw_logits",
                            "mtp": {"head_source": source, "head_files": files, "settings": {"n_max": 3}}}
    row["generation_metadata"] = json.dumps(metadata)
    validate_reference_row(row)
    meta = llm_prep.prep_row(row, tmp_path / "prep")
    assert meta["reference_vocabulary"] == metadata["vocabulary"]
    assert meta["reference_generation"]["decoding"] == metadata["decoding"]
    metadata["decoding"]["logprob_source"] = "draft_logits"
    row["generation_metadata"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="raw target logprobs"):
        validate_reference_row(row)


def test_native_reference_requires_vocabulary_mapping_digest():
    from lib.reference_dataset import validate_reference_row
    row = native_row(False)
    metadata = json.loads(row["generation_metadata"])
    del metadata["vocabulary"]["mapping"]
    row["generation_metadata"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="target vocabulary identity"):
        validate_reference_row(row)


@pytest.mark.parametrize("corruption", ["missing", "null", "positive", "short", "boolean", "string"])
def test_native_reference_requires_complete_nonpositive_logprobs(tmp_path, corruption):
    from lib.reference_dataset import validate_reference_row
    from lib.reference_contract import GTContractError
    from cli import prep_llm_score_from_hf as llm_prep
    row = native_row(False)
    if corruption == "missing":
        row.pop("generation_token_logprobs")
    elif corruption == "null":
        row["generation_token_logprobs"] = None
    elif corruption == "short":
        row["generation_token_logprobs"] = []
    else:
        row["generation_token_logprobs"][0] = {"positive": 1.0, "boolean": False, "string": "-1.0"}[corruption]
    with pytest.raises(GTContractError, match="generation_token_logprobs"):
        validate_reference_row(row)
    with pytest.raises(llm_prep.PrepError, match="generation_token_logprobs"):
        llm_prep.prep_row(row, tmp_path / "prep")


def test_native_decoding_statistics_survive_dataset_storage(tmp_path):
    from datasets import Dataset, load_from_disk
    from lib.reference_dataset import reference_features, validate_reference_row
    row = native_row(False)
    stats = {"drafted_tokens": 0, "accepted_draft_tokens": 0, "verification_batches": 0, "rollback_batches": 0, "checkpoint_replays": 0}
    row["generation_decoding_stats"] = json.dumps(stats)
    ds = Dataset.from_list([row], features=reference_features())
    ds.save_to_disk(str(tmp_path / "dataset"))
    restored = load_from_disk(str(tmp_path / "dataset"))[0]
    validate_reference_row(restored)
    assert json.loads(restored["generation_decoding_stats"]) == stats


@pytest.mark.parametrize("with_image", [False, True])
def test_native_generator_accepts_empty_question_only_with_images(tmp_path, monkeypatch, with_image):
    from datasets import Dataset
    from cli import generate_reference as gen
    binary = tmp_path / "llama-reference"
    binary.write_bytes(b"unused")
    images = native_row(True)["images"] if with_image else []
    ds = Dataset.from_list([{"item_id": "image-only", "question": "", "images": images}])
    monkeypatch.setattr(gen, "load_source", lambda args: ds)
    def at_native_launch(path):
        raise RuntimeError("native launch reached")
    monkeypatch.setattr(gen, "execution_identity", at_native_launch)
    args = gen.parse_args(["--dataset", "unused", "--out", str(tmp_path / "out"),
                           "--llama-reference", str(binary), "--", "-m", "unused.gguf"])
    if with_image:
        with pytest.raises(RuntimeError, match="native launch reached"):
            gen.generate(args)
        request = json.loads((args.out / "requests.jsonl").read_text())
        assert request["question"] == ""
        assert len(request["images"]) == 1
        assert Path(request["images"][0]).read_bytes() == images[0]["bytes"]
    else:
        gen.generate(args)
        completion = json.loads((args.out / "complete.json").read_text())
        assert completion["status"] == "complete_with_failures"
        assert completion["cohort"]["failed_ids"] == ["image-only"]
        assert completion["rows"] == 0



def test_native_generator_pins_hub_dataset_revision(monkeypatch):
    from datasets import Dataset
    import datasets
    from cli import generate_reference as gen
    calls = []
    def load(repo, subset, **kwargs):
        calls.append((repo, subset, kwargs))
        return Dataset.from_list([{"question": "one"}, {"question": "two"}])
    monkeypatch.setattr(datasets, "load_dataset", load)
    args = gen.parse_args(["--dataset", "org/source", "--subset", "cohort", "--revision", "pinned-sha",
                           "--num-samples", "1", "--out", "unused", "--", "-m", "unused.gguf"])
    assert len(gen.load_source(args)) == 1
    assert calls == [("org/source", "cohort", {"split": "train", "revision": "pinned-sha"})]


def test_model_reference_launcher_keeps_gpu_source_and_mode_explicit(tmp_path):
    from cli import generate_model_reference as launch
    args = launch.parse_args(["--profiles", str(tmp_path / "profiles.json"), "--model", "qwen", "--mode", "thinking", "--source", "image-only",
                              "--gpu", "GPU-example", "--out", str(tmp_path / "out"), "--dry-run"])
    runtime = {"ctx": 32768, "batch": 2048, "ubatch": 512, "threads": 8, "threads_batch": 8, "draft_max": 3}
    profiles = {"sources": ["image-only"], "dataset": {"repo": "org/data", "revision": "pinned-sha"},
                "seed": 1234, "generation_caps": {"thinking": 16384}, "models": {"qwen": {
                    "model": "qwen/bf16.gguf", "mmproj": "qwen/mmproj.gguf", "mtp": "embedded",
                    "runtime": {"pro6000": {"thinking": runtime}},
                    "sampling_args": {"thinking": ["--temp", "1.0", "--min-p", "0"]}}}}
    command = launch.build_command(args, profiles)
    assert command[command.index("--subset") + 1] == "image-only-subsample-100"
    profiles["cohort_size"] = 100
    assert launch.build_command(args, profiles) == command
    for wrong_size in (500, "100", True, None):
        profiles["cohort_size"] = wrong_size
        with pytest.raises(ValueError, match="profile cohort_size"):
            launch.build_command(args, profiles)
    del profiles["cohort_size"]
    assert command[command.index("--revision") + 1] == "pinned-sha"
    assert "--enable-thinking" in command
    assert command[command.index("-n") + 1] == "16384"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("-ngl") + 1] == "all"
    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("-c") + 1] == "32768"
    assert "--image-max-tokens" not in command
    assert "--system-prompt" not in command and "--chat-template-kwargs" not in command
    model = profiles["models"]["qwen"]
    model["system_prompt"] = {"thinking": "Exact reasoning prompt\nKeep this text."}
    model["chat_template_kwargs"] = {"thinking": {"reasoning_strength": "high", "current_date": "2026-09-06", "extra": [True, None, 2]}}
    command = launch.build_command(args, profiles)
    assert command.index("--system-prompt") > command.index("--")
    assert command[command.index("--system-prompt") + 1] == model["system_prompt"]["thinking"]
    assert command.index("--chat-template-kwargs") > command.index("--")
    assert json.loads(command[command.index("--chat-template-kwargs") + 1]) == {"preserve_reasoning": True, **model["chat_template_kwargs"]["thinking"]}
    model["semantic_modes"] = ["instruct"]
    with pytest.raises(ValueError, match="unsupported semantic mode"):
        launch.build_command(args, profiles)
    model["semantic_modes"] = ["thinking"]
    model["chat_template_kwargs"]["thinking"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="enable_thinking is selected"):
        launch.build_command(args, profiles)
    del model["chat_template_kwargs"]["thinking"]["enable_thinking"]
    for key in ("system_prompt", "chat_template_kwargs"):
        original = model[key]
        model[key] = None
        with pytest.raises(ValueError, match="must be mode maps"):
            launch.build_command(args, profiles)
        model[key] = original
    runtime["parallel"] = 2
    with pytest.raises(ValueError, match="MTP reference generation requires one sequence"):
        launch.build_command(args, profiles)
    args.mtp = "off"
    command = launch.build_command(args, profiles)
    assert "--spec-type" not in command
    assert command[command.index("-np") + 1] == "2"
    assert command[command.index("-c") + 1] == "65536"
    args.parallel = 3
    args.ctx = 24576
    command = launch.build_command(args, profiles)
    assert command[command.index("-np") + 1] == "3"
    assert command[command.index("-c") + 1] == "73728"
    args.max_new_tokens = 24576
    with pytest.raises(ValueError, match="generation cap"):
        launch.build_command(args, profiles)
    args.max_new_tokens = None
    args.ctx = None
    args.parallel = None
    for invalid in (0, -1, 1.5, "2", True):
        runtime["parallel"] = invalid
        with pytest.raises(ValueError, match="positive integer"):
            launch.build_command(args, profiles)
    runtime["parallel"] = 1
    for invalid in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            launch.parse_args(["--profiles", str(tmp_path / "profiles.json"), "--model", "qwen", "--mode", "thinking", "--source", "image-only",
                               "--gpu", "0", "--out", str(tmp_path / "out"), "--parallel", invalid])
    args.mtp = "3"
    profiles["models"]["qwen"]["mtp"] = None
    with pytest.raises(ValueError, match="no supported local MTP head"):
        launch.build_command(args, profiles)
    args.max_new_tokens = 32768
    with pytest.raises(ValueError, match="generation cap"):
        launch.build_command(args, profiles)


@pytest.mark.parametrize("scenario", ["crash", "crash_always", "timeout", "result_before_exit", "data_error", "startup_partial", "startup_always"])
def test_native_reference_supervisor_preserves_rows_and_bounds_retries(tmp_path, scenario):
    from lib.reference_run import run_native
    binary = tmp_path / "fake-native"
    binary.write_text('''#!/usr/bin/env python3
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--requests');p.add_argument('--out-dir');p.add_argument('--continue-on-error',action='store_true');p.add_argument('--scenario')
a=p.parse_args();d=Path(a.out_dir);d.mkdir()
state=d.parents[2]/'fake-state'
if a.scenario=='startup_always' or (a.scenario=='startup_partial' and not state.exists()):
 state.write_text('once');(d/'metadata.json').write_text('{');os._exit(7)
(d/'metadata.json').write_text(json.dumps({'model':'fixed','seed':1234}))
e=(d/'events.jsonl').open('w');o=(d/'generations.jsonl').open('w')
def event(x): e.write(json.dumps(x)+'\\n');e.flush()
for r in map(json.loads,Path(a.requests).read_text().splitlines()):
 i=r['id'];event({'event':'row_started','id':i})
 if i=='first' and a.scenario=='data_error':
  event({'event':'row_failed','id':i,'error':'bad image','retryable':False});continue
 if i=='first' and a.scenario=='timeout':time.sleep(30)
 if i=='first' and (a.scenario=='crash_always' or (a.scenario=='crash' and not state.exists())):
  state.write_text('once');o.write('{');o.flush();os._exit(8)
 o.write(json.dumps({'id':i,'valid':True})+'\\n');o.flush()
 if i=='first' and a.scenario=='result_before_exit':os._exit(9)
 event({'event':'row_succeeded','id':i})
''')
    binary.chmod(0o755)
    out = tmp_path / "run"
    out.mkdir()
    rows = [{"id": "first"}, {"id": "second"}]
    results, failures, metadata = run_native(binary, ["--scenario", scenario], rows, out, row_timeout=0.1 if scenario == "timeout" else 1800)
    if scenario == "startup_always":
        assert not results and metadata is None
        assert all(f["status"] == "unattempted_startup_failure" and f["attempts"] == 0 for f in failures.values())
        assert len(json.loads((out / "native-attempts.json").read_text())) == 2
        return
    assert "second" in results
    assert set(results) | set(failures) == {"first", "second"}
    assert bool(failures) is (scenario in ("data_error", "crash_always", "timeout"))
    if scenario in ("crash_always", "timeout"):
        assert failures["first"]["attempts"] == 2
    assert metadata == {"model": "fixed", "seed": 1234}
    attempts = json.loads((out / "native-attempts.json").read_text())
    assert len(attempts) == (1 if scenario == "data_error" else 2)
    if scenario == "result_before_exit":
        assert attempts[1]["requested_ids"] == ["second"]
    if scenario == "crash":
        assert attempts[1]["requested_ids"] == ["second", "first"]


@pytest.mark.parametrize("scenario", ["terminal", "result_without_start"])
def test_native_reference_supervisor_rejects_corrupt_terminal_journal(tmp_path, scenario):
    from lib.reference_run import run_native
    binary = tmp_path / "fake-native"
    binary.write_text('''#!/usr/bin/env python3
import json,sys
from pathlib import Path
a=sys.argv;d=Path(a[a.index('--out-dir')+1]);d.mkdir()
(d/'metadata.json').write_text('{}')
if a[a.index('--scenario')+1]=='terminal':
 (d/'events.jsonl').write_text(json.dumps({'event':'row_failed','id':'a','error':'bad','retryable':False})+'\\n')
else:
 (d/'generations.jsonl').write_text(json.dumps({'id':'a'})+'\\n')
''')
    binary.chmod(0o755)
    message = "terminal event" if scenario == "terminal" else "result without a row start"
    with pytest.raises(ValueError, match=message):
        run_native(binary, ["--scenario", scenario], [{"id": "a"}], tmp_path / "run")


@pytest.mark.parametrize("flags", [[], ["--seed", "-1"], ["-s", "-1"], ["--seed=-1"], ["--seed=4294967295"]])
def test_reference_driver_resolves_one_seed_before_any_retry(flags):
    from cli import generate_reference as gen
    args = gen.parse_args(["--dataset", "x", "--out", "y", "--", "-m", "model.gguf", *flags])
    resolved = next((flag.split("=", 1)[1] for flag in args.llama_args if flag.startswith("--seed=")), None)
    if resolved is None:
        index = next(i for i, flag in enumerate(args.llama_args) if flag in ("--seed", "-s"))
        resolved = args.llama_args[index + 1]
    assert 0 <= int(resolved) < 4294967295


def test_reference_driver_filters_and_reconciles_out_of_order_results(tmp_path, monkeypatch):
    from datasets import Dataset, load_from_disk
    from cli import generate_reference as gen
    from lib.reference_dataset import canonical_json
    import hashlib
    fixture = native_row(False)
    metadata = json.loads(fixture["generation_metadata"])
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    metadata.update(model_path=str(model), mmproj_path="", schema_version="skymizer-reference-v2")
    ds = Dataset.from_list([{"item_id": i, "question": "What color?"} for i in ("first", "repeat", "bad", "last")])
    monkeypatch.setattr(gen, "load_source", lambda args: ds)
    monkeypatch.setattr(gen, "execution_identity", lambda path: ({"binary_sha256": "f" * 64}, []))
    def native(binary, native_args, requests, out, **kwargs):
        results = {}
        for row_id in ("last", "repeat", "first"):
            n = 96 if row_id == "repeat" else 2
            result = {"id": row_id, "input_ids": [1, 2] + [3] * n, "n_prefill_tokens": 2, "n_past_prefill": 2,
                      "prompt_layout": json.loads(fixture["llamacpp_prompt_layout"]), "content": "red", "finish_reason": "stop",
                      "sampling": {"seed": 1234}, "chat_template_kwargs": {}, "enable_thinking": False,
                      "token_logprobs": [-0.5] * n, "prompt": "What color?", "add_special": True,
                      "stripped_leading_bos": False, "stop_type": "eos", "stopping_word": ""}
            if row_id == "repeat":
                result["repetition"] = {"start": 0, "unit_len": 1, "repeats": 96, "repeated_len": 96}
            results[row_id] = result
        (out / "native").mkdir()
        (out / "native/generations.jsonl").write_text("".join(canonical_json(r) + "\n" for r in results.values()))
        return results, {"bad": {"id": "bad", "status": "failed", "error": "poison row"}}, metadata
    monkeypatch.setattr(gen, "run_native", native)
    args = gen.parse_args(["--dataset", "unused", "--out", str(tmp_path / "out"), "--llama-reference", str(model), "--", "-m", str(model)])
    gen.generate(args)
    completion = json.loads((args.out / "complete.json").read_text())
    cohort = completion["cohort"]
    assert completion["status"] == "complete_with_failures"
    assert cohort["requested"] == 4 and cohort["generated"] == 3
    assert cohort["eligible_ids"] == ["first", "last"]
    assert cohort["excluded_ids"] == ["repeat"] and cohort["failed_ids"] == ["bad"]
    assert (args.out / "scripts/lib/reference_study.py").read_bytes() == (gen.SKYMIZER / "lib/reference_study.py").read_bytes()
    assert list(load_from_disk(str(args.out / "dataset"))["id"]) == ["first", "last"]
    assert len((args.out / "native/generations.jsonl").read_text().splitlines()) == 3
    assert cohort["eligible_ids_sha256"] == hashlib.sha256(canonical_json(["first", "last"]).encode()).hexdigest()
    assert json.loads((args.out / "excluded.jsonl").read_text())["id"] == "repeat"
    assert json.loads((args.out / "failures.jsonl").read_text())["id"] == "bad"


@pytest.fixture
def upload_run(tmp_path):
    from copy import deepcopy
    from datasets import Dataset
    from cli import upload_reference as upload
    from lib.reference_dataset import canonical_json, reference_features
    from lib.reference_run import atomic_json
    import hashlib
    run = tmp_path / 'run'
    run.mkdir()
    profiles = {'dataset': {'repo': 'elichen-skymizer/vlm-prepared-dataset', 'revision': 'pinned'},
                'seed': 1234, 'sources': ['mmmu-pro-vision'], 'generation_caps': {'instruct': 8192, 'thinking': 16384},
                'models': {'test-model': {'model': 'test.gguf', 'mmproj': 'mmproj.gguf', 'mtp': None,
                    'effective_sampling': {'instruct': {'seed': 1234}, 'thinking': {'seed': 1234}},
                    'identity': {'files': [{'name': 'test.gguf', 'size': 12, 'sha256': 'c' * 64, 'role': 'llm'},
                                           {'name': 'mmproj.gguf', 'size': 6, 'sha256': 'd' * 64, 'role': 'mmproj'}]}}}}
    def write(size=100, mode='instruct', change=None, template=None, row_change=None):
        import shutil
        if (run / 'dataset').exists():
            shutil.rmtree(run / 'dataset')
        ids = [f'row-{i}' for i in range(size)]
        cohort = {'requested': size, 'eligible': 2, 'excluded': 1, 'failed': size - 3,
                  'native_generated': 3, 'generated': 3, 'requested_ids': ids,
                  'eligible_ids': ids[:2], 'excluded_ids': ids[2:3], 'failed_ids': ids[3:]}
        for key in ('requested_ids', 'eligible_ids', 'excluded_ids', 'failed_ids'):
            cohort[key + '_sha256'] = hashlib.sha256(canonical_json(cohort[key]).encode()).hexdigest()
        row = native_row(True)
        metadata = json.loads(row['generation_metadata'])
        metadata.update(cohort=cohort, sampling={'seed': 1234},
                        chat_template_source='gguf', jinja=True, system_prompt='', chat_template_kwargs={'preserve_reasoning': 'true'}, model_files=[{'path': '/models/test.gguf', 'size': 12, 'sha256': 'c' * 64}],
                        mmproj_path='/models/mmproj.gguf', mmproj_sha256='d' * 64,
                        dataset_source={'path': profiles['dataset']['repo'], 'revision': 'pinned', 'split': 'train',
                                        'subset': f'mmmu-pro-vision-subsample-{size}', 'num_rows': size},
                        requested_enable_thinking=mode == 'thinking', thinking_column=None,
                        n_predict=profiles['generation_caps'][mode], repetition_detector=deepcopy(upload.DETECTOR))
        metadata.update(template or {})
        if change:
            change(metadata)
        for name, value in [('metadata.json', metadata), ('run_start.json', {'status': 'running'}),
                            ('native-attempts.json', {'attempts': []}),
                            ('run_state.json', {'status': 'complete_with_failures', 'cohort': cohort}),
                            ('complete.json', {'status': 'complete_with_failures', 'cohort': cohort, 'rows': 2})]:
            atomic_json(run / name, value)
        rows = []
        for item in ids[:2]:
            r = deepcopy(row)
            r['generation_chat_template_kwargs'] = json.dumps({**metadata['chat_template_kwargs'], 'enable_thinking': 'true' if mode == 'thinking' else 'false'})
            r.update(id=item, item_id=item, generation_enable_thinking=mode == 'thinking', generation_metadata=canonical_json(metadata))
            if row_change:
                row_change(r)
            rows.append(r)
        Dataset.from_list(rows, features=reference_features()).save_to_disk(str(run / 'dataset'))
        (run / 'excluded.jsonl').write_text(json.dumps({'id': ids[2], 'reason': 'repetition'}) + '\n')
        (run / 'failures.jsonl').write_text(''.join(json.dumps({'id': i, 'status': 'preparation_failed'}) + '\n' for i in ids[3:]))
        return run, profiles
    return write


@pytest.mark.parametrize('size,mode', [(100, 'instruct'), (100, 'thinking'), (500, 'instruct'), (500, 'thinking')])
def test_upload_cohort_names_and_image_parquet_roundtrip(upload_run, size, mode):
    from cli import upload_reference as upload
    from datasets import Dataset
    from lib.reference_dataset import raw_images
    run, profiles = upload_run(size, mode)
    manifest, parquet = upload.prepare(run, 'test-model', mode, profiles)
    assert manifest['repo'] == 'elichen-skymizer/test-model-' + ('pilot' if size == 100 else 'collect-500')
    assert manifest['subset'] == f'mmmu-pro-vision-subsample-{size}-' + ('ins' if mode == 'instruct' else 'think')
    assert manifest['rows'] == 2 and manifest['cohort']['requested'] == size
    ds = Dataset.from_parquet(str(parquet))
    assert raw_images(ds[0]) == raw_images(native_row(True))
    assert ds[0]['generation_token_logprobs'] == [-0.5, -0.1]


@pytest.mark.parametrize("size,profile_size", [(100, 500), (500, 100)])
def test_upload_rejects_profile_size_mismatch(upload_run, size, profile_size):
    from cli import upload_reference as upload
    run, profiles = upload_run(size=size)
    profiles["cohort_size"] = profile_size
    with pytest.raises(ValueError, match="profile cohort_size"):
        upload.prepare(run, "test-model", "instruct", profiles)
    profiles["cohort_size"] = size
    assert upload.prepare(run, "test-model", "instruct", profiles)[0]["cohort"]["requested"] == size


@pytest.mark.parametrize('scenario', ['fallback', 'explicit', 'metadata_prompt', 'metadata_kwargs', 'row_prompt', 'row_kwargs', 'unsupported_mode'])
def test_upload_uses_profile_template_contract(upload_run, scenario):
    from cli import upload_reference as upload
    from lib.reference_study import reference_template
    profile = {'semantic_modes': ['thinking'], 'system_prompt': {'thinking': 'Exact reasoning prompt'},
               'chat_template_kwargs': {'thinking': {'reasoning_strength': 'high', 'current_date': '2026-09-06'}}}
    template = reference_template(profile, 'thinking')
    assert template['chat_template_kwargs'] == {'preserve_reasoning': 'true', 'reasoning_strength': '"high"', 'current_date': '"2026-09-06"'}
    def change(metadata):
        if scenario == 'metadata_prompt':
            metadata['system_prompt'] = 'wrong'
        if scenario == 'metadata_kwargs':
            metadata['chat_template_kwargs']['reasoning_strength'] = '"low"'
    def row_change(row):
        request = json.loads(row['generation_request'])
        request.pop('system_prompt', None)
        if scenario in ('explicit', 'row_prompt'):
            request['system_prompt'] = 'wrong' if scenario == 'row_prompt' else profile['system_prompt']['thinking']
        row['generation_request'] = json.dumps(request)
        if scenario == 'row_kwargs':
            kwargs = json.loads(row['generation_chat_template_kwargs'])
            kwargs['current_date'] = '"2026-09-07"'
            row['generation_chat_template_kwargs'] = json.dumps(kwargs)
    run, profiles = upload_run(mode='thinking', template=template, change=change, row_change=row_change)
    profiles['models']['test-model'].update(profile)
    if scenario == 'unsupported_mode':
        profiles['models']['test-model']['semantic_modes'] = ['instruct']
    if scenario in ('fallback', 'explicit'):
        assert upload.prepare(run, 'test-model', 'thinking', profiles)[0]['rows'] == 2
    else:
        with pytest.raises(ValueError):
            upload.prepare(run, 'test-model', 'thinking', profiles)


@pytest.mark.parametrize('change', [
    lambda m: m['cohort'].update(eligible=3),
    lambda m: m['cohort'].update(native_generated=4),
    lambda m: m['dataset_source'].update(split='test'),
    lambda m: m.update(n_predict=192),
    lambda m: m['sampling'].update(seed=999),
    lambda m: m['sampling'].update(temperature=0),
    lambda m: m.update(chat_template_source='cli_override'),
    lambda m: m.update(system_prompt='changed'),
    lambda m: m.update(image_max_tokens=256),
    lambda m: m.update(requested_enable_thinking=True),
    lambda m: m['repetition_detector'].update(min_repeated_tokens=999),
    lambda m: m['repetition_detector'].update(enabled=False),
    lambda m: m['model_files'][0].update(sha256='e' * 64),
    lambda m: m.update(mmproj_sha256='e' * 64),
    lambda m: m['cohort']['eligible_ids'].reverse(),
])
def test_upload_rejects_changed_cohort_or_protocol(upload_run, change):
    from cli import upload_reference as upload
    run, profiles = upload_run(change=change)
    with pytest.raises(ValueError):
        upload.prepare(run, 'test-model', 'instruct', profiles)


def test_upload_rejects_short_smoke(upload_run):
    from cli import upload_reference as upload
    run, profiles = upload_run(size=4)
    with pytest.raises(ValueError, match='full subsample'):
        upload.prepare(run, 'test-model', 'instruct', profiles)


@pytest.fixture
def fake_reference_hub(monkeypatch, tmp_path):
    import huggingface_hub as hub
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    from types import SimpleNamespace
    import httpx
    import yaml
    class FakeHub:
        def __init__(self):
            self.files = {}
            self.head = '0'
            self.history = {'0': {}}
            self.conflict_once = False
            self.commits = 0
            self.private = None
            self.creation_privacies = []
            self.privacy_updates = []
        def save(self):
            self.head = str(int(self.head) + 1)
            self.history[self.head] = self.files.copy()
        def create_repo(self, *args, private, **kwargs):
            self.creation_privacies.append(private)
            if self.private is None:
                self.private = private
        def update_repo_settings(self, *args, private, **kwargs):
            self.privacy_updates.append(private)
            self.private = private
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha=self.head, private=self.private)
        def list_repo_files(self, *args, revision, **kwargs):
            return list(self.history[revision])
        def download(self, repo, path, revision, **kwargs):
            data = self.history[revision].get(path)
            if data is None:
                raise EntryNotFoundError('missing')
            local = tmp_path / 'hub' / revision / path
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            return str(local)
        def create_commit(self, *args, operations, parent_commit, **kwargs):
            if self.conflict_once:
                self.conflict_once = False
                other = {'config_name': 'other-ins', 'data_files': [{'split': 'train', 'path': 'other-ins/train.parquet'}]}
                self.files['README.md'] = ('---\n' + yaml.safe_dump({'configs': [other]}) + '---\nBody\n').encode()
                self.files['other-ins/train.parquet'] = b'other data'
                self.save()
                response = httpx.Response(409, request=httpx.Request('POST', 'https://example.test/commit'))
                raise HfHubHTTPError('conflict', response=response)
            assert parent_commit == self.head
            for operation in operations:
                content = operation.path_or_fileobj
                self.files[operation.path_in_repo] = content if isinstance(content, bytes) else Path(content).read_bytes()
            self.commits += 1
            self.save()
            return SimpleNamespace(oid=self.head)
    api = FakeHub()
    monkeypatch.setattr(hub, 'HfApi', lambda: api)
    monkeypatch.setattr(hub, 'hf_hub_download', api.download)
    return api


def test_upload_atomic_cas_preserves_other_configs_and_idempotent(upload_run, fake_reference_hub):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    api.conflict_once = True
    receipt = upload.publish(run, manifest, parquet)
    assert receipt['status'] == 'verified' and not receipt['already_present']
    assert api.files['other-ins/train.parquet'] == b'other data'
    assert 'other-ins' in api.files['README.md'].decode()
    again = upload.publish(run, manifest, parquet)
    assert again['already_present'] and api.commits == 1


@pytest.mark.parametrize('kind', ['parquet', 'audit'])
def test_upload_rejects_orphan_remote_namespace(upload_run, fake_reference_hub, kind):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    path = f"{manifest['subset']}/{parquet.name}" if kind == 'parquet' else f"audit/{manifest['subset']}/metadata.json"
    api = fake_reference_hub
    api.files[path] = b'preexisting'
    api.save()
    with pytest.raises(ValueError, match='already occupied'):
        upload.publish(run, manifest, parquet)
    assert api.commits == 0 and api.files[path] == b'preexisting'


@pytest.mark.parametrize('corruption', ['metadata', 'readme', 'parquet', 'extra'])
def test_upload_idempotency_checks_all_remote_artifacts(upload_run, fake_reference_hub, corruption):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    upload.publish(run, manifest, parquet)
    api = fake_reference_hub
    subset = manifest['subset']
    if corruption == 'metadata':
        api.files[f'audit/{subset}/metadata.json'] = b'{}'
    elif corruption == 'readme':
        api.files['README.md'] = b'---\nconfigs: []\n---\nWrong card\n'
    elif corruption == 'parquet':
        api.files[f'{subset}/{parquet.name}'] = b'changed'
    else:
        api.files[f'{subset}/orphan.parquet'] = b'extra'
    api.save()
    with pytest.raises(ValueError):
        upload.publish(run, manifest, parquet)
    assert api.commits == 1


@pytest.mark.parametrize('path', ['train.parquet', 'data/train.parquet'])
def test_upload_preserves_implicit_existing_default(upload_run, fake_reference_hub, path):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    api.files[path] = b'existing default data'
    api.files['README.md'] = b'# Existing default dataset\n'
    api.save()
    before = api.files.copy()
    with pytest.raises(ValueError, match='implicit default'):
        upload.publish(run, manifest, parquet)
    assert api.commits == 0 and api.files == before


@pytest.mark.parametrize('config', [
    {'config_name': 'existing', 'data_files': '**/*.parquet'},
    {'config_name': 'existing', 'data_files': '*/**/*.parquet'},
    {'config_name': 'existing', 'data_files': '**/**/*.parquet'},
    {'config_name': 'existing', 'data_dir': '.', 'data_files': '**/*.parquet'},
    {'config_name': 'existing', 'data_dir': './', 'data_files': '**/*.parquet'},
    {'config_name': 'existing'},
    {'config_name': 'existing', 'data_files': '*.parquet'},
])
def test_upload_preserves_existing_config_membership(upload_run, fake_reference_hub, config):
    from cli import upload_reference as upload
    import yaml
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    api.files['README.md'] = ('---\n' + yaml.safe_dump({'configs': [config]}) + '---\nExisting\n').encode()
    api.files['data/train.parquet'] = b'existing'
    api.save()
    before = api.files.copy()
    with pytest.raises(ValueError, match='existing config'):
        upload.publish(run, manifest, parquet)
    assert api.commits == 0 and api.files == before


@pytest.mark.parametrize('scenario', ['plan', 'restore_and_skip', 'existing_mismatch', 'download_mismatch', 'head_plan', 'head_restore_and_skip', 'head_existing_mismatch', 'head_download_mismatch', 'head_missing_identity', 'head_unsafe_path'])
def test_reference_model_restore_keeps_verified_nested_paths(tmp_path, monkeypatch, scenario):
    from cli import restore_reference_models as restore
    import hashlib
    import sys
    root = tmp_path / 'models'
    data = b'weights'
    files = [{'name': n, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'role': role}
             for n, role in [('part1.gguf', 'llm'), ('part2.gguf', 'llm'), ('projector.gguf', 'mmproj')]]
    profile = {'models': {'test': {'model': 'test/nested/part1.gguf', 'mmproj': 'test/projector.gguf',
                                 'identity': {'manifest': 's3://research-kld-benchmark/reference_model/test/manifest.json', 'files': files}}}}
    model = profile['models']['test']
    with_head = scenario.startswith('head_')
    if with_head:
        model.update(mtp='sidecar', head='test/mtp/head.gguf')
        model['identity']['head_files'] = [{'name': 'head.gguf', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]
    if scenario == 'head_missing_identity':
        model['identity']['head_files'] = []
    elif scenario == 'head_unsafe_path':
        model['head'] = '../head.gguf'
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps(profile))
    command = ['restore', '--profiles', str(path), '--models-dir', str(root)]
    calls = []
    def download(args, check):
        assert args[:3] == ['aws', 's3', 'cp'] and check
        calls.append(args[3])
        bad = scenario == 'download_mismatch' or (scenario == 'head_download_mismatch' and args[3].endswith('/head.gguf'))
        Path(args[4]).write_bytes(b'bad' if bad else data)
    monkeypatch.setattr(restore.subprocess, 'run', download)
    if scenario not in ('plan', 'head_plan'):
        command.append('--download')
    monkeypatch.setattr(sys, 'argv', command)
    if scenario in ('head_missing_identity', 'head_unsafe_path'):
        with pytest.raises(ValueError, match='MTP head|must stay under'):
            restore.main()
        assert not calls and not list(root.rglob('*.gguf'))
    elif scenario in ('existing_mismatch', 'head_existing_mismatch'):
        existing = root / ('test/mtp/head.gguf' if scenario == 'head_existing_mismatch' else 'test/projector.gguf')
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b'existing')
        with pytest.raises(ValueError, match='existing file differs'):
            restore.main()
        assert existing.read_bytes() == b'existing' and not calls
    elif scenario in ('download_mismatch', 'head_download_mismatch'):
        with pytest.raises(ValueError, match='full-file size/SHA256'):
            restore.main()
        if scenario == 'download_mismatch':
            assert not list(root.rglob('*.gguf'))
        else:
            assert len(calls) == 4 and not (root / 'test/mtp/head.gguf').exists()
            assert (root / 'test/projector.gguf').read_bytes() == data
    else:
        restore.main()
        if scenario in ('plan', 'head_plan'):
            assert not calls and not list(root.rglob('*.gguf'))
        else:
            names = ['test/nested/part1.gguf', 'test/nested/part2.gguf', 'test/projector.gguf']
            if with_head:
                names.append('test/mtp/head.gguf')
            assert len(calls) == len(names)
            for name in names:
                assert (root / name).read_bytes() == data
            restore.main()
            assert len(calls) == len(names)


@pytest.mark.parametrize("mode,cap", [("instruct", 2048), ("thinking", 8192)])
@pytest.mark.parametrize("model,ubatch", [("qwen3.5-4b", 512), ("qwen3.6-35b-a3b", 512),
    ("gemma-4-e4b-it", 512), ("glm-4.6v-flash", 512), ("internvl3.5-30b-a3b", 512), ("gemma-4-31b-it", 2048)])
def test_kld_launcher_uses_matching_runtime_without_generation_or_analysis(tmp_path, model, ubatch, mode, cap):
    from cli import collect_model_kld as launch
    from lib.reference_study import study_overview
    profiles = json.loads((Path(__file__).resolve().parents[1] / "profiles/small-pilot100.json").read_text())
    for group in ("snr", "final"):
        profiles["models"].update(json.loads((Path(__file__).resolve().parents[1] / f"profiles/{group}-pilot100.json").read_text())["models"])
    plan = {"stage": "kld", "models": [model], "sources": ["mmmu-pro-vision"], "modes": [mode],
            "hardware": "pro6000", "size": 100, "num_samples": None, "models_dir": str(tmp_path / "models")}
    args = launch.parse_args(["--profiles", str(tmp_path / "profiles.json"), "--study", str(tmp_path / "study"), "--model", model, "--mode", mode,
        "--source", "mmmu-pro-vision", "--candidate", "Q4_K_M", "--cand-model", str(tmp_path / "candidate.gguf"),
        "--llama-vlm-kld", str(tmp_path / "llama-vlm-kld"), "--gpu", "0", "--dry-run"])
    command, out = launch.build_command(args, plan, profiles, tmp_path / "archived")
    flag = lambda key: command[command.index(key) + 1]
    assert flag("--n-ctx") == "32768" and flag("--n-batch") == "2048"
    assert flag("--n-ubatch") == str(ubatch) and flag("--tf-chunk") == "2048"
    assert flag("--n-gpu-layers") == "-2" and flag("--num-eval-tokens") == str(cap)
    assert flag("--ref-mmproj") == flag("--cand-mmproj")
    assert "--image-max-tokens" not in command and "--max-total-tokens" not in command
    assert "--spec-type" not in command and "--model-draft" not in command
    assert command[1] == str(tmp_path / "archived/cli/collect_kld.py")
    subset = "mmmu-pro-vision-subsample-100-" + ("ins" if mode == "instruct" else "think")
    assert flag("--subset") == subset and flag("--dataset") == f"elichen-skymizer/{model}-pilot"
    assert out == tmp_path / "study/artifacts" / model / subset / "kld/Q4_K_M"
    overview = study_overview(profiles, plan)
    assert overview["models"][model]["kld_runtime"][mode]["n_ubatch"] == ubatch
    assert overview["models"][model]["reference_runtime"][mode]["ubatch"] == ubatch
    assert "trial" not in json.dumps(overview)
    assert "--allow-vocab-attr-mismatch" not in command
    profiles["models"][model]["allow_vocab_attr_mismatch"] = True
    compatible, _ = launch.build_command(args, plan, profiles, tmp_path / "archived")
    assert compatible == command + ["--allow-vocab-attr-mismatch"]
    assert study_overview(profiles, plan)["models"][model]["kld_runtime"][mode]["allow_vocab_attr_mismatch"] is True
    profiles["models"][model]["allow_vocab_attr_mismatch"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        launch.build_command(args, plan, profiles, tmp_path / "archived")
    profiles["models"][model].pop("allow_vocab_attr_mismatch")
    args.dataset = tmp_path / "local-dataset"
    command, _ = launch.build_command(args, plan, profiles, tmp_path / "archived")
    assert command[command.index("--subset") + 1] == ""
    args.source = "not-in-study"
    with pytest.raises(ValueError, match="not in the study"):
        launch.build_command(args, plan, profiles, tmp_path / "archived")


@pytest.mark.parametrize("scenario", ["parallel_crash", "parallel_partial", "parallel_timeout", "too_many_slots"])
def test_native_reference_supervisor_recovers_parallel_rows(tmp_path, scenario):
    from lib.reference_run import run_native
    binary = tmp_path / "parallel-native"
    binary.write_text('''#!/usr/bin/env python3
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--requests');p.add_argument('--out-dir');p.add_argument('--continue-on-error',action='store_true');p.add_argument('--scenario')
a=p.parse_args();d=Path(a.out_dir);d.mkdir()
state=d.parents[2]/'parallel-state'
(d/'metadata.json').write_text(json.dumps({'model':'fixed','total_slots':1 if a.scenario=='too_many_slots' else 2}))
e=(d/'events.jsonl').open('w');o=(d/'generations.jsonl').open('w')
def event(kind,i):e.write(json.dumps({'event':kind,'id':i})+'\\n');e.flush()
def result(i):o.write(json.dumps({'id':i})+'\\n');o.flush();event('row_succeeded',i)
rows=[r['id'] for r in map(json.loads,Path(a.requests).read_text().splitlines())]
if state.exists():
 for i in rows:event('row_started',i);result(i)
else:
 state.write_text('once')
 event('row_started',rows[0]);event('row_started',rows[1])
 if a.scenario=='parallel_partial':result(rows[1])
 if a.scenario=='parallel_timeout':
  result(rows[1]);time.sleep(1.2)
  event('row_started',rows[2]);result(rows[2]);time.sleep(30)
 os._exit(7)
''')
    binary.chmod(0o755)
    out = tmp_path / "run"
    out.mkdir()
    rows = [{"id": name} for name in ("first", "second", "third")]
    if scenario == "too_many_slots":
        with pytest.raises(ValueError, match="declared parallel slots"):
            run_native(binary, ["--scenario", scenario], rows, out)
        return
    results, failures, metadata = run_native(binary, ["--scenario", scenario], rows, out,
                                            row_timeout=0.1 if scenario == "parallel_timeout" else 1800)
    assert set(results) == {"first", "second", "third"} and not failures
    assert metadata["total_slots"] == 2
    merged = [json.loads(line)["id"] for line in (out / "native/generations.jsonl").read_text().splitlines()]
    assert merged == ["first", "second", "third"]
    attempts = json.loads((out / "native-attempts.json").read_text())
    assert len(attempts) == 2
    expected_retry = {"parallel_crash": ["third", "first", "second"],
                      "parallel_partial": ["third", "first"], "parallel_timeout": ["first"]}
    assert attempts[1]["requested_ids"] == expected_retry[scenario]
    assert attempts[0]["timed_out"] is (scenario == "parallel_timeout")



def test_study_overview_uses_each_checkpoint_mode_and_sequence_count(tmp_path):
    import copy
    from cli import collect_model_kld as launch
    from lib.reference_study import study_overview
    profiles = json.loads((Path(__file__).resolve().parents[1] / "profiles/small-pilot100.json").read_text())
    original = profiles["models"]["qwen3.5-4b"]
    profiles["models"] = {}
    for mode, parallel in (("instruct", 4), ("thinking", 1)):
        profile = copy.deepcopy(original)
        profile["semantic_modes"] = [mode]
        for hardware, settings in profile["runtime"].items():
            profile["runtime"][hardware] = {mode: settings[mode]}
            profile["runtime"][hardware][mode]["parallel"] = parallel
        profiles["models"][mode + "-only"] = profile
    profiles["cohort_size"] = 100
    plan = {"stage": "kld", "models": list(profiles["models"]), "sources": ["mmmu-pro-vision"],
            "modes": ["instruct", "thinking"], "hardware": "pro6000", "size": 100,
            "num_samples": None, "models_dir": str(tmp_path / "models")}
    overview = study_overview(profiles, plan)
    assert overview["generation_jobs"] == 2
    for mode, parallel in (("instruct", 4), ("thinking", 1)):
        model = overview["models"][mode + "-only"]
        assert model["modes"] == [mode]
        assert list(model["kld_runtime"]) == [mode]
        assert model["reference_runtime"][mode]["parallel"] == parallel
    args = launch.parse_args(["--profiles", str(tmp_path / "profiles.json"), "--study", str(tmp_path / "study"), "--model", "thinking-only", "--mode", "instruct",
        "--source", "mmmu-pro-vision", "--candidate", "Q4", "--cand-model", str(tmp_path / "candidate.gguf"),
        "--llama-vlm-kld", str(tmp_path / "llama-vlm-kld"), "--gpu", "0", "--dry-run"])
    with pytest.raises(ValueError, match="unsupported semantic mode"):
        launch.build_command(args, plan, profiles, tmp_path / "archive")
    args.mode = "thinking"
    launch.build_command(args, plan, profiles, tmp_path / "archive")
    plan["size"] = 500
    with pytest.raises(ValueError, match="cohort_size"):
        launch.build_command(args, plan, profiles, tmp_path / "archive")


@pytest.mark.parametrize('existing_private', [None, False, True])
def test_upload_private_default_checks_new_and_existing_repositories(upload_run, fake_reference_hub, existing_private):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    api.private = existing_private
    receipt = upload.publish(run, manifest, parquet)
    assert receipt['private'] is True and api.private is True
    assert api.creation_privacies == [True]
    assert api.privacy_updates == ([True] if existing_private is False else [])


@pytest.mark.parametrize('change', ['ignored', 'denied'])
def test_upload_refuses_private_setting_failure_before_writing(upload_run, fake_reference_hub, monkeypatch, change):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    api.private = False
    def update(*args, **kwargs):
        if change == 'denied':
            raise PermissionError('setting denied')
    monkeypatch.setattr(api, 'update_repo_settings', update)
    with pytest.raises((ValueError, PermissionError), match='private|denied'):
        upload.publish(run, manifest, parquet)
    assert api.commits == 0 and not api.files
    assert not (run / 'upload/receipt.json').exists()


def test_upload_refuses_success_receipt_if_privacy_changes(upload_run, fake_reference_hub, monkeypatch):
    from cli import upload_reference as upload
    run, profiles = upload_run()
    manifest, parquet = upload.prepare(run, 'test-model', 'instruct', profiles)
    api = fake_reference_hub
    original = api.create_commit
    def commit(*args, **kwargs):
        result = original(*args, **kwargs)
        api.private = False
        return result
    monkeypatch.setattr(api, 'create_commit', commit)
    with pytest.raises(ValueError, match='not private'):
        upload.publish(run, manifest, parquet)
    assert not (run / 'upload/receipt.json').exists()


@pytest.mark.parametrize('flags', [[], ['--private']])
def test_upload_cli_defaults_to_private(upload_run, monkeypatch, flags):
    import sys
    from cli import upload_reference as upload
    run, profiles = upload_run()
    profile_path = run / 'profiles.json'
    profile_path.write_text(json.dumps(profiles))
    observed = []
    monkeypatch.setattr(upload, 'publish', lambda run, manifest, parquet, private: observed.append(private) or {'private': private})
    monkeypatch.setattr(sys, 'argv', ['upload_reference.py', '--run', str(run), '--model', 'test-model', '--mode', 'instruct', '--profiles', str(profile_path), *flags])
    upload.main()
    assert observed == [True]
