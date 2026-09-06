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
    assert h1 == h2 and h1.startswith("ds-v2:")
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
                "vocab_size": 8}
    import io
    from PIL import Image
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    return build_row({}, request, result, metadata, [image.getvalue()] if with_image else [])


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
