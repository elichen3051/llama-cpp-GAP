"""Tests for prep_vlm_score_from_hf.py helpers.

Hermetic: no `datasets`, `transformers`, `torch`, or PIL imports. Images and
tokenizer behavior are tiny local fakes.
"""

import json
import re

import numpy as np
import pytest

import cli.prep_vlm_score_from_hf as prep
from fakes import FakeImage, FakeTok


_IMG_CFG = {
    "hf_image_processor": {
        "image_processor_type": "Qwen2VLImageProcessor",
        "patch_size": 16,
        "merge_size": 2,
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    },
    "vllm_mm_processor_kwargs": {},
}


def _row(**overrides):
    # Prefill mirrors real Qwen3-VL data: every <|image_pad|> run is wrapped in
    # <|vision_start|> (15) ... <|vision_end|> (16), as the HF processor emits.
    row = {
        "item_id": "test_item",
        "source": "mmmu",
        "category": "Physics",
        "generation_model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "input_ids": [10, 15, 11, 11, 16, 12, 20, 21],
        "labels": [-100, -100, -100, -100, -100, -100, 20, 21],
        "input_tokens_len": 8,
        "n_prefill_tokens": 6,
        "generated_tokens_len": 2,
        "num_images": 1,
        "sum_vision_tokens": 256,
        "max_vision_tokens": 256,
        "image_processor_config_hash": "hash123",
        "image_processor_config": json.dumps(_IMG_CFG),
        "ref_answer": "A",
        "generated_texts": "generated answer",
        "seed": 7,
        "images": [FakeImage()],
    }
    row.update(overrides)
    return row


def _tok():
    return FakeTok({
        10: "Question ",
        11: "<|image_pad|>",
        12: " prompt",
        13: " between ",
        14: " after",
        15: "<|vision_start|>",
        16: "<|vision_end|>",
        20: "A",
        21: "B",
    })


def _gemma_tok():
    return FakeTok({
        10: "Question ",
        12: " prompt",
        17: "<|image>",
        18: "<|image|>",
        19: "<image|>",
        20: "A",
        21: "B",
    })


def _kimi_tok():
    # Real Kimi-VL ids: 163602 <|media_start|>, 4017 "image",
    # 163603 <|media_content|>, 163605 <|media_pad|>, 163604 <|media_end|>.
    return FakeTok({
        10: "Question ",
        12: " prompt",
        4017: "image",
        163602: "<|media_start|>",
        163603: "<|media_content|>",
        163604: "<|media_end|>",
        163605: "<|media_pad|>",
        20: "A",
        21: "B",
    })


def test_prep_row_writes_files_and_returns_meta_matching_disk(tmp_path):
    row = _row()

    meta = prep.prep_row(row, _tok(), tmp_path)

    assert (tmp_path / "img_0.png").read_bytes() == b"PNG"
    assert np.fromfile(tmp_path / "tokens.bin", dtype=np.int32).tolist() == row["input_ids"]
    assert (tmp_path / "formatted_chat.txt").read_text(encoding="utf-8") == (
        prep.build_formatted_chat(row, _tok())
    )
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")) == meta


def test_build_formatted_chat_collapses_wrapped_image_pad_run():
    """The whole <|vision_start|>(<|image_pad|>)+<|vision_end|> block becomes
    one <__media__>: mtmd re-adds the wrapper pair itself at eval time, so the
    wrappers must NOT survive into the formatted chat (double-wrapping)."""
    formatted = prep.build_formatted_chat(_row(), _tok())

    assert formatted == "Question <__media__> prompt"
    assert formatted.count("<__media__>") == 1
    assert "<|vision_start|>" not in formatted
    assert "<|vision_end|>" not in formatted


def test_build_formatted_chat_collapses_two_wrapped_runs_to_two_markers():
    row = _row(
        input_ids=[10, 15, 11, 11, 16, 13, 15, 11, 11, 16, 14, 20],
        input_tokens_len=12,
        n_prefill_tokens=11,
        generated_tokens_len=1,
        num_images=2,
        images=[FakeImage(), FakeImage()],
    )

    formatted = prep.build_formatted_chat(row, _tok())

    assert formatted == "Question <__media__> between <__media__> after"
    assert formatted.count("<__media__>") == 2


def test_build_formatted_chat_rejects_bare_pad_run_without_wrappers():
    """A pad run NOT wrapped in <|vision_start|>/<|vision_end|> is malformed
    data (the HF processor always wraps); it must produce no marker and fail
    the count check rather than be silently collapsed."""
    row = _row(
        input_ids=[10, 11, 11, 12, 20, 21],
        input_tokens_len=6,
        n_prefill_tokens=4,
    )

    with pytest.raises(prep.PrepError, match="marker count 0 != num_images 1"):
        prep.build_formatted_chat(row, _tok())


def test_build_formatted_chat_collapses_gemma4_image_block():
    row = _row(
        generation_model_name_or_path="google/gemma-4-E4B-it",
        input_ids=[10, 17, 18, 18, 19, 12, 20],
        input_tokens_len=7,
        n_prefill_tokens=6,
        generated_tokens_len=1,
    )

    formatted = prep.build_formatted_chat(row, _gemma_tok())

    assert formatted == "Question <__media__> prompt"
    assert formatted.count("<__media__>") == 1
    assert "<|image>" not in formatted
    assert "<image|>" not in formatted


def test_build_formatted_chat_collapses_adjacent_gemma4_image_blocks():
    row = _row(
        generation_model_name_or_path="google/gemma-4-E4B-it",
        input_ids=[10, 17, 18, 19, 17, 18, 18, 19, 12, 20],
        input_tokens_len=10,
        n_prefill_tokens=9,
        generated_tokens_len=1,
        num_images=2,
        images=[FakeImage(), FakeImage()],
    )

    formatted = prep.build_formatted_chat(row, _gemma_tok())

    assert formatted == "Question <__media__><__media__> prompt"
    assert formatted.count("<__media__>") == 2


def test_build_formatted_chat_rejects_bare_gemma4_pad_run():
    row = _row(
        generation_model_name_or_path="google/gemma-4-E4B-it",
        input_ids=[10, 18, 18, 12, 20],
        input_tokens_len=5,
        n_prefill_tokens=4,
        generated_tokens_len=1,
    )

    with pytest.raises(prep.PrepError, match="marker count 0 != num_images 1"):
        prep.build_formatted_chat(row, _gemma_tok())


def test_build_formatted_chat_collapses_kimi_vl_image_block():
    """The whole HF block, prelude included, folds to one marker: mtmd's
    kimivl path re-adds only <|media_start|>/<|media_end|>, never the
    `image<|media_content|>` tokens, so leaving them in would double them."""
    row = _row(
        generation_model_name_or_path="moonshotai/Kimi-VL-A3B-Instruct",
        input_ids=[10, 163602, 4017, 163603, 163605, 163605, 163604, 12, 20],
        input_tokens_len=9,
        n_prefill_tokens=8,
        generated_tokens_len=1,
    )

    formatted = prep.build_formatted_chat(row, _kimi_tok())

    assert formatted == "Question <__media__> prompt"
    assert formatted.count("<__media__>") == 1
    assert "<|media_" not in formatted
    assert "image" not in formatted


def test_build_formatted_chat_collapses_adjacent_kimi_vl_image_blocks():
    row = _row(
        generation_model_name_or_path="moonshotai/Kimi-VL-A3B-Instruct",
        input_ids=[10, 163602, 4017, 163603, 163605, 163604,
                   163602, 4017, 163603, 163605, 163605, 163604, 12, 20],
        input_tokens_len=14,
        n_prefill_tokens=13,
        generated_tokens_len=1,
        num_images=2,
        images=[FakeImage(), FakeImage()],
    )

    formatted = prep.build_formatted_chat(row, _kimi_tok())

    assert formatted == "Question <__media__><__media__> prompt"
    assert formatted.count("<__media__>") == 2


@pytest.mark.parametrize(
    "input_ids",
    [
        [10, 163605, 163605, 12, 20],                     # bare pad run
        [10, 163602, 163605, 163605, 163604, 12, 20],     # wrappers, no prelude
        [10, 163602, 4017, 163603, 163604, 12, 20],       # prelude, zero pads
    ],
    ids=["bare-pads", "no-prelude", "zero-pads"],
)
def test_build_formatted_chat_rejects_incomplete_kimi_vl_block(input_ids):
    """Only the complete HF block collapses; anything else must fail the
    marker count check rather than be silently accepted."""
    row = _row(
        generation_model_name_or_path="moonshotai/Kimi-VL-A3B-Instruct",
        input_ids=input_ids,
        input_tokens_len=len(input_ids),
        n_prefill_tokens=len(input_ids) - 1,
        generated_tokens_len=1,
    )

    with pytest.raises(prep.PrepError, match="marker count 0 != num_images 1"):
        prep.build_formatted_chat(row, _kimi_tok())


def test_build_formatted_chat_accepts_qwen35_model():
    """Qwen3.5 uses the same wrapper token *strings* as Qwen3-VL (the IDs
    differ — 248053+ vs 151652+ — but prep works on decoded text), so the
    family registry maps it to the same collapse regex."""
    row = _row(generation_model_name_or_path="Qwen/Qwen3.5-2B")

    formatted = prep.build_formatted_chat(row, _tok())

    assert formatted == "Question <__media__> prompt"


def test_build_formatted_chat_accepts_qwen36_model():
    """Qwen3.6 rows (GAP qwen3.6-35b-a3b) decode each image to the same
    <|vision_start|>(<|image_pad|>)+<|vision_end|> block as Qwen3-VL/Qwen3.5
    and its mmproj is qwen3vl_merger, so the family registry reuses the
    Qwen collapse regex and stripped-wrapper mode."""
    row = _row(generation_model_name_or_path="Qwen/Qwen3.6-35B-A3B")

    formatted = prep.build_formatted_chat(row, _tok())

    assert formatted == "Question <__media__> prompt"


def test_build_formatted_chat_rejects_unregistered_family():
    row = _row(generation_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct")

    with pytest.raises(prep.PrepError, match="unsupported model family"):
        prep.build_formatted_chat(row, _tok())


def test_build_formatted_chat_rejects_non_qwen_model():
    row = _row(generation_model_name_or_path="deepseek-ai/DeepSeek-V2-Lite")

    with pytest.raises(prep.PrepError, match="unsupported model family"):
        prep.build_formatted_chat(row, _tok())


def test_image_block_regex_for_prefers_longest_prefix(monkeypatch):
    """Overlapping registry prefixes must resolve by longest match, not dict
    insertion order — a hypothetical 'Qwen/Qwen3.5-VL' entry added after the
    shorter 'Qwen/Qwen3.5' must still win for its own models."""
    sentinel = re.compile("sentinel")
    monkeypatch.setitem(prep.MODEL_FAMILIES, "Qwen/Qwen3.5-VL", sentinel)

    assert prep.image_block_regex_for("Qwen/Qwen3.5-VL-7B") is sentinel
    assert (prep.image_block_regex_for("Qwen/Qwen3.5-2B")
            is prep.MODEL_FAMILIES["Qwen/Qwen3.5"])


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("Qwen/Qwen3-VL-4B-Instruct", "<|image_pad|>"),
        ("Qwen/Qwen3.5-2B", "<|image_pad|>"),
        ("Qwen/Qwen3.6-35B-A3B", "<|image_pad|>"),
        ("google/gemma-4-E4B-it", "<|image|>"),
        ("moonshotai/Kimi-VL-A3B-Instruct", "<|media_pad|>"),
    ],
)
def test_image_pad_token_for_registered_families(model_name, expected):
    assert prep.image_pad_token_for(model_name) == expected


def test_family_registries_agree_on_prefixes():
    """One family = one prefix in every registry: a prefix registered for
    the collapse regex must also have a pad token, and only registered
    families may opt into remote-code tokenizers."""
    assert set(prep.MODEL_FAMILIES) == set(prep.IMAGE_PAD_TOKENS)
    assert set(prep.REMOTE_CODE_TOKENIZER_PREFIXES) <= set(prep.MODEL_FAMILIES)
    for prefix, token in prep.IMAGE_PAD_TOKENS.items():
        # the pad token is what the family's regex collapses
        assert re.escape(token) in prep.MODEL_FAMILIES[prefix].pattern


def test_image_pad_token_for_rejects_unknown_family():
    with pytest.raises(prep.PrepError, match="unsupported model family"):
        prep.image_pad_token_for("Qwen/Qwen2.5-VL-7B-Instruct")


def test_image_pad_token_for_prefers_longest_prefix(monkeypatch):
    """Overlapping image-pad prefixes use the same longest-match rule."""
    monkeypatch.setitem(
        prep.IMAGE_PAD_TOKENS, "Qwen/Qwen3.5-VL", "<|specific_image_pad|>")

    assert (prep.image_pad_token_for("Qwen/Qwen3.5-VL-7B")
            == "<|specific_image_pad|>")
    assert (prep.image_pad_token_for("Qwen/Qwen3.5-2B")
            == "<|image_pad|>")


@pytest.mark.parametrize(
    ("model_name", "token"),
    [
        ("Qwen/Qwen3-VL-4B-Instruct", "<|image_pad|>"),
        ("Qwen/Qwen3.6-35B-A3B", "<|image_pad|>"),
        ("google/gemma-4-E4B-it", "<|image|>"),
        ("moonshotai/Kimi-VL-A3B-Instruct", "<|media_pad|>"),
    ],
)
def test_resolve_image_pad_id_round_trips(model_name, token):
    tok = FakeTok({11: token})

    assert prep.resolve_image_pad_id(tok, model_name) == 11


def test_resolve_image_pad_id_rejects_tokenizer_without_pad_token():
    """HF fast tokenizers return the unk id (not None) for a missing token;
    the round-trip check must catch that instead of silently using unk."""
    tok = FakeTok({5: "hello"})

    with pytest.raises(prep.PrepError, match="image_pad"):
        prep.resolve_image_pad_id(tok, "Qwen/Qwen3-VL-4B-Instruct")


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("moonshotai/Kimi-VL-A3B-Instruct", True),
        ("moonshotai/Kimi-VL-A3B-Thinking", True),
        ("Qwen/Qwen3-VL-4B-Instruct", False),
        ("Qwen/Qwen3.6-35B-A3B", False),
        ("google/gemma-4-E4B-it", False),
        ("some-org/unregistered-model", False),
    ],
)
def test_tokenizer_trusts_remote_code_only_for_registered_families(model_name, expected):
    assert prep.tokenizer_trusts_remote_code(model_name) is expected


@pytest.fixture
def from_pretrained_calls(monkeypatch):
    """Stand-in `transformers` module so load_tokenizer's lazy import is
    observable without the real package (the suite stays hermetic). Returns
    the (name, kwargs) list the fake AutoTokenizer.from_pretrained records;
    the fake is built per test so nothing leaks between tests."""
    import sys
    import types
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return f"tok:{name}"

    mod = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return calls


def test_load_tokenizer_passes_trust_remote_code_for_kimi_vl(from_pretrained_calls):
    tok = prep.load_tokenizer("moonshotai/Kimi-VL-A3B-Instruct")

    assert tok == "tok:moonshotai/Kimi-VL-A3B-Instruct"
    assert from_pretrained_calls == [
        ("moonshotai/Kimi-VL-A3B-Instruct", {"trust_remote_code": True})]


def test_load_tokenizer_keeps_remote_code_off_for_other_families(from_pretrained_calls):
    prep.load_tokenizer("Qwen/Qwen3.6-35B-A3B")
    prep.load_tokenizer("google/gemma-4-E4B-it")

    assert [kw for _, kw in from_pretrained_calls] == [
        {"trust_remote_code": False}, {"trust_remote_code": False}]


def test_build_formatted_chat_rejects_marker_count_mismatch():
    row = _row(num_images=2)

    with pytest.raises(prep.PrepError, match="marker count 1 != num_images 2"):
        prep.build_formatted_chat(row, _tok())


def test_sanity_check_row_rejects_input_length_mismatch():
    row = _row(input_tokens_len=7)

    with pytest.raises(prep.PrepError, match="input_tokens_len"):
        prep.sanity_check_row(row)


@pytest.mark.parametrize("n_prefill", [0, 8])
def test_sanity_check_row_rejects_n_prefill_out_of_range(n_prefill):
    row = _row(n_prefill_tokens=n_prefill)

    with pytest.raises(prep.PrepError, match="out of range"):
        prep.sanity_check_row(row)


def test_sanity_check_row_rejects_prefill_plus_answer_mismatch():
    row = _row(generated_tokens_len=1)

    with pytest.raises(prep.PrepError, match="n_prefill"):
        prep.sanity_check_row(row)


def test_sanity_check_row_rejects_labels_length_mismatch():
    row = _row(labels=[-100] * 6 + [20])

    with pytest.raises(prep.PrepError, match="len\\(labels\\)"):
        prep.sanity_check_row(row)


def test_sanity_check_row_rejects_unmasked_prefill_labels():
    row = _row(labels=[-100] * 5 + [12, 20, 21])

    with pytest.raises(prep.PrepError, match="must all be -100"):
        prep.sanity_check_row(row)


def test_sanity_check_row_rejects_labels_answer_divergence():
    """Rule 6: stored labels must equal the input_ids answer suffix — the C++
    scorer targets input_ids[n_prefill:], so a divergent row would be scored
    against different tokens than the producer's labels-based eval."""
    row = _row(labels=[-100] * 6 + [20, 99])

    with pytest.raises(prep.PrepError, match="must equal input_ids"):
        prep.sanity_check_row(row)


def test_derive_image_token_limits_inverts_pixel_area():
    # size.{shortest,longest}_edge are pixel *areas* (min_pixels / max_pixels).
    # mtmd: tokens = pixels / (patch_size**2 * merge_size**2) = pixels / 1024.
    cfg = {
        "hf_image_processor": {
            "patch_size": 16,
            "merge_size": 2,
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        },
    }

    limits = prep.derive_image_token_limits(cfg)

    assert limits == {
        "patch_size": 16,
        "merge_size": 2,
        "min_pixels": 65536,
        "max_pixels": 16777216,
        "image_min_tokens": 64,
        "image_max_tokens": 16384,
    }
    for v in limits.values():
        assert type(v) is int


def test_derive_image_token_limits_prefers_explicit_min_max_pixels():
    # Older transformers Qwen2VLImageProcessor emits top-level min_pixels /
    # max_pixels instead of size.{shortest,longest}_edge; accept either.
    cfg = {
        "hf_image_processor": {
            "patch_size": 14,
            "merge_size": 2,
            "min_pixels": 3136,
            "max_pixels": 12845056,
        },
    }

    limits = prep.derive_image_token_limits(cfg)

    assert limits["min_pixels"] == 3136
    assert limits["max_pixels"] == 12845056
    assert limits["image_min_tokens"] == 3136 // (14 * 14 * 2 * 2)
    assert limits["image_max_tokens"] == 12845056 // (14 * 14 * 2 * 2)


@pytest.mark.parametrize(
    "cfg",
    [
        {
            "hf_image_processor": {
                "image_processor_type": "Gemma4ImageProcessor",
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280,
            },
        },
        {
            "gemma4_vision_token_config": {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280,
            },
        },
    ],
)
def test_derive_image_token_limits_gemma4_schema(cfg):
    limits = prep.derive_image_token_limits(cfg)

    assert limits == {
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "max_soft_tokens": 280,
        "image_min_tokens": -1,
        "image_max_tokens": 280,
    }
    for v in limits.values():
        assert type(v) is int


# The real per-row config of elichen-skymizer/kimi-vl-a3b-ins-full-ref-text-*
# (hash c3287470bd...), minus the auto_map / mean / std provenance keys.
_KIMI_IMG_CFG = {
    "hf_image_processor": {
        "image_processor_type": "KimiVLImageProcessor",
        "in_token_limit": 4096,
        "merge_kernel_size": [2, 2],
        "num_pooled_tokens": 1024,
        "pad_input": True,
        "patch_size": 14,
    },
    "kimi_vl_processor_config": {
        "image_token": "<|media_pad|>",
        "media_pad_token_id": 163605,
    },
    "kimi_vl_vision_token_config": {
        "in_token_limit": 4096,
        "merge_kernel_size": [2, 2],
        "num_pooled_tokens": 1024,
        "patch_size": 14,
    },
    "kimi_vl_vision_tokens_override_by_vllm": False,
    "vllm_mm_processor_kwargs": {},
}


@pytest.mark.parametrize(
    "cfg",
    [
        _KIMI_IMG_CFG,
        {"hf_image_processor": _KIMI_IMG_CFG["hf_image_processor"]},
        {"kimi_vl_vision_token_config": _KIMI_IMG_CFG["kimi_vl_vision_token_config"]},
    ],
    ids=["full", "hf-only", "vision-cfg-only"],
)
def test_derive_image_token_limits_kimi_vl_schema(cfg):
    """in_token_limit caps PRE-merge patches; one vision token per 2x2 merge
    kernel, so 4096 // 4 = 1024 (== mtmd's kimivl default upper bound). The
    HF processor never upscales, so the floor is 1 token (NOT -1, which
    would let mtmd's built-in kimivl floor of 8 upscale small images)."""
    limits = prep.derive_image_token_limits(cfg)

    assert limits == {
        "patch_size": 14,
        "merge_kernel_size": [2, 2],
        "in_token_limit": 4096,
        "image_min_tokens": 1,
        "image_max_tokens": 1024,
    }
    for key in ("image_min_tokens", "image_max_tokens"):
        assert type(limits[key]) is int


@pytest.mark.parametrize(
    "broken",
    [
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "merge_kernel_size": [2, 2]},                              # no limit
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096},                                   # no kernel
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096, "merge_kernel_size": [2]},         # 1-d kernel
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": "many", "merge_kernel_size": [2, 2]},    # non-int
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096, "merge_kernel_size": [0, 2]},      # zero
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096, "merge_kernel_size": "22"},        # str kernel
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096, "merge_kernel_size": [2, 2, 2]},   # 3-d kernel
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096.0, "merge_kernel_size": [2, 2]},    # float
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 4096, "merge_kernel_size": [True, True]},  # bool
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": "14",
         "in_token_limit": 4096, "merge_kernel_size": [2, 2]},      # numeric str
        {"image_processor_type": "KimiVLImageProcessor", "patch_size": 14,
         "in_token_limit": 3, "merge_kernel_size": [2, 2]},         # cap -> 0
    ],
    ids=["no-limit", "no-kernel", "1d-kernel", "non-int", "zero-kernel",
         "str-kernel", "3d-kernel", "float", "bool", "numeric-str", "cap-zero"],
)
def test_derive_image_token_limits_rejects_invalid_kimi_vl_schema(broken):
    with pytest.raises(prep.PrepError,
                       match="invalid Kimi-VL image processor config schema"):
        prep.derive_image_token_limits({"hf_image_processor": broken})


def test_derive_image_token_limits_rejects_unknown_schema():
    cfg = {
        "hf_image_processor": {
            "image_processor_type": "UnknownImageProcessor",
        },
    }

    with pytest.raises(prep.PrepError,
                       match="unsupported image processor config schema"):
        prep.derive_image_token_limits(cfg)


def test_build_meta_contains_expected_keys_and_int_fields():
    row = _row(
        input_tokens_len="6",
        n_prefill_tokens="4",
        generated_tokens_len="2",
        num_images="1",
        sum_vision_tokens="256",
        max_vision_tokens="128",
        seed="7",
    )

    meta = prep.build_meta(row)

    assert meta == {
        "item_id": "test_item",
        "source": "mmmu",
        "category": "Physics",
        "n_prefill": 4,
        "n_answer": 2,
        "input_tokens_len": 6,
        "num_images": 1,
        "sum_vision_tokens": 256,
        "max_vision_tokens": 128,
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "image_processor_config_hash": "hash123",
        "image_processor_config": _IMG_CFG,
        "image_token_limits": {
            "patch_size": 16,
            "merge_size": 2,
            "min_pixels": 65536,
            "max_pixels": 16777216,
            "image_min_tokens": 64,
            "image_max_tokens": 16384,
        },
        "ref_answer": "A",
        "generated_texts_preview": "generated answer",
        "dataset_seed": 7,
        # vLLM rows: no llama.cpp provenance (schema v1.4 keys present, null)
        "generation_engine": None,
        "n_past_expected": None,
        "per_image_vision_token_counts": None,
        "add_special": False,
    }
    for key in [
        "n_prefill",
        "n_answer",
        "input_tokens_len",
        "num_images",
        "sum_vision_tokens",
        "max_vision_tokens",
        "dataset_seed",
    ]:
        assert type(meta[key]) is int


def test_load_dataset_sorted_can_sort_descending(monkeypatch):
    """D2: the VLM loader gains sort_desc with the LLM loader's semantics;
    the default keeps the ascending sort every existing dir was built on."""
    calls = []

    class FakeDataset:
        def sort(self, keys, reverse=False):
            calls.append(("sort", list(keys), reverse))
            return self

    def fake_load_dataset(dataset, subset=None, split=None):
        calls.append(("load", dataset, subset, split))
        return FakeDataset()

    import sys
    monkeypatch.setitem(sys.modules, "datasets",
                        type("D", (), {"load_dataset": fake_load_dataset}))
    prep.load_dataset_sorted("ds", "", "train", "num_images")
    prep.load_dataset_sorted("ds", "", "train", "num_images", sort_desc=True)
    assert calls == [("load", "ds", None, "train"), ("sort", ["num_images"], False),
                     ("load", "ds", None, "train"), ("sort", ["num_images"], True)]
# ---------------------------------------------------------------------------
# llama.cpp-generated rows (llm_reference_generator schema v1.4)
# ---------------------------------------------------------------------------

_LLAMACPP_IMG_CFG = {
    "engine": "llama.cpp",
    "llama_cpp_build": "b10820-f41f902cf",
    "mmproj_file": "mmproj-BF16.gguf",
    "mmproj_sha256": "ab" * 32,
    "clip": {"clip.vision.patch_size": 16, "clip.vision.spatial_merge_size": 2},
    "image_min_tokens": -1,
    "image_max_tokens": -1,
    "media_marker": "<__media__>",
}

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 8


def _llamacpp_row(**overrides):
    import hashlib

    row = _row(
        generation_engine="llama.cpp",
        generation_model_name_or_path="Qwen/Qwen3.5-4B",
        llamacpp_prompt_string="Question <__media__> prompt",
        llamacpp_media_marker="<__media__>",
        llamacpp_prompt_layout=json.dumps({"n_tokens": 6, "n_pos": 4, "chunks": []}),
        per_image_vision_token_counts=[2],
        per_image_n_pos=[1],
        llamacpp_n_past_prefill=4,
        llamacpp_tokens_evaluated=6,
        image_bytes_sha256=[hashlib.sha256(_JPG_BYTES).hexdigest()],
        image_processor_config=json.dumps(_LLAMACPP_IMG_CFG),
    )
    row.update(overrides)
    return row


def test_llamacpp_row_formatted_chat_is_stored_prompt_verbatim():
    row = _llamacpp_row()
    # No tokenizer needed: the decode-and-collapse path is bypassed entirely.
    assert prep.build_formatted_chat(row, None) == "Question <__media__> prompt"


def test_llamacpp_row_normalizes_server_media_marker():
    row = _llamacpp_row(
        llamacpp_prompt_string="Q <__media_r4nd0m__> p",
        llamacpp_media_marker="<__media_r4nd0m__>",
    )
    assert prep.build_formatted_chat(row, None) == "Q <__media__> p"


def test_llamacpp_row_marker_count_must_match_num_images():
    row = _llamacpp_row(llamacpp_prompt_string="no marker here")
    with pytest.raises(prep.PrepError, match="marker count 0 != num_images 1"):
        prep.build_formatted_chat(row, None)


def test_llamacpp_row_missing_prompt_string_is_an_error():
    row = _llamacpp_row(llamacpp_prompt_string=None)
    with pytest.raises(prep.PrepError, match="llamacpp_prompt_string"):
        prep.build_formatted_chat(row, None)


def test_derive_image_token_limits_llamacpp_engine_passes_recorded_budget():
    limits = prep.derive_image_token_limits(
        dict(_LLAMACPP_IMG_CFG, image_min_tokens=64, image_max_tokens=16384)
    )
    assert limits == {
        "engine": "llama.cpp",
        "image_min_tokens": 64,
        "image_max_tokens": 16384,
        "llama_cpp_build": "b10820-f41f902cf",
        "mmproj_sha256": "ab" * 32,
    }


def test_image_extension_sniffs_common_formats():
    assert prep.image_extension(_PNG_BYTES) == "png"
    assert prep.image_extension(_JPG_BYTES) == "jpg"
    assert prep.image_extension(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert prep.image_extension(b"GIF89a\x00") == "gif"
    assert prep.image_extension(b"junk") == "bin"


def test_prep_row_llamacpp_writes_raw_bytes_and_records_n_past(tmp_path):
    row = _llamacpp_row(llamacpp_add_special=True)

    meta = prep.prep_row(row, None, tmp_path, raw_images=[_JPG_BYTES])

    assert (tmp_path / "img_0.jpg").read_bytes() == _JPG_BYTES
    assert not (tmp_path / "img_0.png").exists()
    assert meta["image_files"] == ["img_0.jpg"]
    assert meta["n_past_expected"] == 4
    assert meta["generation_engine"] == "llama.cpp"
    assert meta["add_special"] is True
    assert meta["per_image_vision_token_counts"] == [2]
    assert meta["image_token_limits"]["engine"] == "llama.cpp"
    assert (tmp_path / "formatted_chat.txt").read_text(
        encoding="utf-8"
    ) == "Question <__media__> prompt"
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")) == meta


def test_prep_row_llamacpp_rejects_raw_bytes_that_do_not_match_sha256(tmp_path):
    row = _llamacpp_row()
    with pytest.raises(prep.PrepError, match="image_bytes_sha256"):
        prep.prep_row(row, None, tmp_path, raw_images=[_PNG_BYTES])


def test_prep_row_llamacpp_rejects_raw_image_count_mismatch(tmp_path):
    row = _llamacpp_row()
    with pytest.raises(prep.PrepError, match="raw images"):
        prep.prep_row(row, None, tmp_path, raw_images=[_JPG_BYTES, _JPG_BYTES])


def test_prep_row_vllm_rows_keep_png_reencode_and_null_n_past(tmp_path):
    row = _row()
    meta = prep.prep_row(row, _tok(), tmp_path)
    assert meta["image_files"] == ["img_0.png"]
    assert meta["n_past_expected"] is None
    assert meta["generation_engine"] is None
    assert (tmp_path / "img_0.png").read_bytes() == b"PNG"


def test_prep_row_llamacpp_requires_raw_images(tmp_path):
    with pytest.raises(prep.PrepError, match="raw image bytes"):
        prep.prep_row(_llamacpp_row(), None, tmp_path)


def test_llamacpp_rows_without_add_special_column_default_false():
    assert prep.build_meta(_llamacpp_row())["add_special"] is False


def test_load_dataset_sorted_reads_a_local_save_to_disk_dir(tmp_path):
    """A generator --save_path directory is scored as written: loaded with
    load_from_disk (no Hub round trip), sorted like a Hub dataset, --subset
    rejected because it has no meaning for a local dir."""
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"id": ["b", "a", "c", "d", "e"], "num_images": [2, 1, 3, 1, None]})
    ds.save_to_disk(str(tmp_path / "gt"))
    before = {p.name: p.read_bytes() for p in (tmp_path / "gt").iterdir()}
    got = prep.load_dataset_sorted(str(tmp_path / "gt"), None, "train", "num_images")
    assert got["id"] == ["a", "d", "b", "c", "e"]
    got = prep.load_dataset_sorted(str(tmp_path / "gt"), None, "train", "num_images", sort_desc=True)
    assert got["id"] == ["c", "b", "a", "d", "e"]
    assert {p.name: p.read_bytes() for p in (tmp_path / "gt").iterdir()} == before
    with pytest.raises(ValueError, match="--subset"):
        prep.load_dataset_sorted(str(tmp_path / "gt"), "smoke7-x", "train", "num_images")
    datasets.DatasetDict({"train": ds}).save_to_disk(str(tmp_path / "gtdict"))
    got = prep.load_dataset_sorted(str(tmp_path / "gtdict"), None, "train", None)
    assert got["id"] == ["b", "a", "c", "d", "e"]
