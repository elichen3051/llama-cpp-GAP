"""Tests for the derived-corpus helpers: thinking split/render (lib.thinking_trajectory) and row contract
(lib.derived_corpus). Hermetic: synthetic rows, no models, no `datasets`."""

import json

import numpy as np
import pytest

from lib import derived_corpus as dc
from lib import thinking_trajectory as tt
from lib.reference_dataset import canonical_json as canonical

QWEN_PROMPT = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"
GEMMA_PROMPT = "<|turn>user\nq<turn|>\n<|turn>model\n"


def source_row(text, *, model_terminal, stop=("stop", "eos"), prompt=QWEN_PROMPT, row_id="item-1", source="mmstar"):
    last = model_terminal if stop == ("stop", "eos") else 7
    return {"id": row_id, "source": source, "generated_texts": text, "finish_reason": stop[0],
            "llamacpp_stop_type": stop[1], "input_ids": [1, 2, last], "llamacpp_prompt_string": prompt,
            "generation_enable_thinking": True}


def merged(segments):
    return [(s["text"], s["parse_special"]) for s in segments]


def test_qwen_source_drops_newlines_around_think_close_and_renders_for_gemma():
    row = source_row("step one\n</think>\n\nThe answer is B.", model_terminal=248046)
    parts = tt.split_trajectory(row, "qwen3.5-4b", "mmstar-subsample-100-think")
    assert (parts["reasoning"], parts["answer"], parts["source_separator"]) == ("step one", "The answer is B.", "\n</think>\n\n")
    tt.validate_source_parts(parts)
    assert merged(tt.render_trajectory(parts, "gemma-4-e4b-it", GEMMA_PROMPT)) == [
        ("<|channel>", True), ("thought\nstep one\n", False), ("<channel|>", True), ("The answer is B.", False),
        ("<turn|>", True)]


def test_gemma_source_open_reasoning_at_length_limit_stays_open_for_qwen():
    row = source_row("<|channel>thought\nstill thinking ", model_terminal=106, stop=("length", "limit"), prompt=GEMMA_PROMPT)
    parts = tt.split_trajectory(row, "gemma-4-31b-it", "mmstar-subsample-100-think")
    assert not parts["has_answer_boundary"] and parts["reasoning"] == "still thinking "
    tt.validate_source_parts(parts)
    assert merged(tt.render_trajectory(parts, "qwen3.5-4b", QWEN_PROMPT)) == [("still thinking ", False)]
    # Gemma target: the channel opens, nothing closes it.
    assert merged(tt.render_trajectory(parts, "gemma-4-e4b-it", GEMMA_PROMPT)) == [
        ("<|channel>", True), ("thought\nstill thinking ", False)]


def test_gemma_direct_answer_gets_an_empty_qwen_reasoning_block():
    row = source_row("Just B.", model_terminal=106, prompt=GEMMA_PROMPT)
    parts = tt.split_trajectory(row, "gemma-4-e4b-it", "mmstar-subsample-100-think")
    assert not parts["source_has_reasoning"] and parts["answer"] == "Just B."
    tt.validate_source_parts(parts)
    assert merged(tt.render_trajectory(parts, "qwen3.6-35b-a3b", QWEN_PROMPT)) == [
        ("\n", False), ("</think>", True), ("\n\nJust B.", False), ("<|im_end|>", True)]


def test_reviewed_source_split_is_byte_exact():
    row = source_row("\u25c1think\u25b7 r \u25c1/think\u25b7 a ", model_terminal=163586)
    parts = tt.split_trajectory(row, "kimi-vl-a3b-thinking-2506", "mmstar-subsample-100-think")
    assert (parts["reasoning"], parts["answer"]) == (" r ", " a ")
    tt.validate_source_parts(parts)


@pytest.mark.parametrize("text, stop, message", [
    ("no opener", ("stop", "eos"), "without native opener"),
    ("<think>open", ("stop", "eos"), "EOS with open reasoning"),
    ("<think>r</think><|eom|>a", ("stop", "eos"), "literal source marker"),
])
def test_reviewed_source_needs_registered_exception(text, stop, message):
    row = source_row(text, model_terminal=151336, stop=stop)
    with pytest.raises(ValueError, match=message):
        tt.split_trajectory(row, "glm-4.6v-flash", "mmstar-subsample-100-think")


def test_pilot_rejects_unreviewed_marker_and_terminal_mismatch():
    with pytest.raises(ValueError, match="unreviewed thinking marker"):
        tt.split_trajectory(source_row("r</think>a<think>", model_terminal=248046), "qwen3.5-4b", "x-think")
    row = source_row("r</think>a", model_terminal=248046)
    row["input_ids"][-1] = 5
    with pytest.raises(ValueError, match="terminal token"):
        tt.split_trajectory(row, "qwen3.5-4b", "x-think")
    with pytest.raises(ValueError, match="no thinking split rule"):
        tt.split_trajectory(row, "unknown-model", "x-think")


def test_source_parts_must_rebuild_the_source_text():
    parts = tt.split_trajectory(source_row("r\n</think>\na", model_terminal=248046), "qwen3.5-4b", "x-think")
    tt.validate_source_parts(parts)
    for key, value in (("reasoning", "R"), ("source_separator", "</think>!"), ("source_exception", {"action": "x"})):
        with pytest.raises(ValueError):
            tt.validate_source_parts({**parts, key: value})


def test_native_continuation_follows_stop_condition():
    complete = {"has_answer_boundary": True, "source_stop_type": "eos", "reasoning": "r"}
    assert tt.native_continuation({"suffix": "x<|im_end|>\n"}, complete, "qwen3.5-4b") == "x<|im_end|>"
    assert tt.native_continuation({"suffix": "x<|im_end|>\n"}, {**complete, "source_stop_type": "limit"}, "qwen3.5-4b") == "x"
    open_parts = {"has_answer_boundary": False, "source_stop_type": "limit", "reasoning": "partial"}
    suffix = "<|channel>thought\n" + tt.INCOMPLETE_SENTINEL + "<channel|><turn|>\n"
    assert tt.native_continuation({"suffix": suffix}, open_parts, "gemma-4-e4b-it") == "<|channel>thought\npartial"


def test_literal_tokenization_only_for_qwen_targets_with_reviewed_flag():
    allowed = next(r for r in tt.EXCEPTIONS["source_rows"] if r.get("allow_qwen_literal_special"))
    parts = {"source_exception": allowed}
    assert tt.literal_tokenization_allowed(parts, "qwen3.5-4b")
    assert not tt.literal_tokenization_allowed(parts, "gemma-4-31b-it")
    assert not tt.literal_tokenization_allowed({"source_exception": None}, "qwen3.5-4b")


def test_exception_registry_covers_known_sources_only():
    for entry in tt.EXCEPTIONS["source_rows"]:
        assert tt.SOURCES[entry["model"]]["policy"] == "reviewed" and entry["subset"].endswith("-think")
    for entry in tt.EXCEPTIONS["pilot_literal_markers"]:
        assert tt.SOURCES[entry["model"]]["policy"] == "pilot"


VOCAB = {"scheme": "llama-vocabulary-sha256-v1", "size": 100, "type": 2, "mapping": "0" * 64, "attributes": "1" * 64}


def target_peer():
    layout = {"n_tokens": 5, "n_pos": 4, "chunks": [
        {"type": "text", "start": 0, "n_tokens": 2, "n_pos": 2, "tokens": [5, 6]},
        {"type": "image", "start": 2, "n_tokens": 2, "n_pos": 1},
        {"type": "text", "start": 4, "n_tokens": 1, "n_pos": 1, "tokens": [7]}]}
    return {"id": "item-1", "item_id": "item-1", "source": "mmstar", "question": "q", "images": [b"img"],
            "num_images": 1, "llamacpp_prompt_string": "a<__media__>" + QWEN_PROMPT,
            "llamacpp_prompt_layout": json.dumps(layout), "llamacpp_n_past_prefill": 4, "llamacpp_add_special": False,
            "image_bytes_sha256": ["ab" * 32], "n_prefill_tokens": 5, "input_ids": [5, 6, -1, -1, 7, 90, 91]}


def same_family_row(answer=(10, 11, 12)):
    protocol = {"schema": dc.SCHEMA_SAME_FAMILY, "mode": "thinking", "source_vocabulary": VOCAB,
                "tokens_transferred_without_retokenization": True, "source_ids_sha256": tt.text_sha(canonical(list(answer)))}
    return dc.derived_row(target_peer(), list(answer), "source text", protocol, VOCAB)


def test_same_family_row_keeps_target_prefix_and_source_tokens():
    row = same_family_row()
    assert row["input_ids"] == [5, 6, -1, -1, 7, 10, 11, 12]
    assert row["labels"] == [-100] * 5 + [10, 11, 12]
    assert dc.is_derived_row(row) and dc.validate_row(row)["schema"] == dc.SCHEMA_SAME_FAMILY


@pytest.mark.parametrize("change, message", [
    (lambda r: r.update(input_ids=r["input_ids"][:-1] + [13], labels=r["labels"][:-1] + [13]), "answer token IDs changed"),
    (lambda r: r.update(labels=[-100] * len(r["labels"])), "labels differ"),
    (lambda r: r.update(input_ids=[5, 6, -1, -1, 8, 10, 11, 12]), "prefix token IDs changed"),
    (lambda r: r.update(llamacpp_prompt_string="changed"), "prompt changed"),
    (lambda r: r.update(generated_texts="other"), "answer text changed"),
])
def test_derived_row_tampering_is_rejected(change, message):
    row = same_family_row()
    change(row)
    with pytest.raises(ValueError, match=message):
        dc.validate_row(row)


def test_same_family_requires_identical_vocabulary():
    protocol = {"schema": dc.SCHEMA_SAME_FAMILY, "mode": "thinking", "source_vocabulary": {**VOCAB, "mapping": "f" * 64},
                "tokens_transferred_without_retokenization": True, "source_ids_sha256": tt.text_sha(canonical([10]))}
    with pytest.raises(ValueError, match="identical source and target vocabulary"):
        dc.derived_row(target_peer(), [10], "t", protocol, VOCAB)


def cross_family_protocol(peer, segments, encoded):
    parts = tt.split_trajectory(source_row("<|channel>thought\nr<channel|>a", model_terminal=106, prompt=GEMMA_PROMPT),
                                "gemma-4-31b-it", "mmstar-subsample-100-think")
    text = "".join(s["text"] for s in segments)
    return {"schema": dc.SCHEMA_CROSS_FAMILY, "mode": "thinking", "target": "qwen3.5-4b", "source_parts": parts,
            "canonical_tokenization_exact": True, "trajectory_segments": segments,
            "encoded_trajectory_segments": encoded, "trajectory_sha256": tt.text_sha(text),
            "target_prompt": peer["llamacpp_prompt_string"],
            "native_template_oracle": {"continuation_sha256": tt.text_sha(text), "native_prefix_exact": True}}


def test_cross_family_row_rechecks_the_rendered_trajectory():
    peer = target_peer()
    parts = tt.split_trajectory(source_row("<|channel>thought\nr<channel|>a", model_terminal=106, prompt=GEMMA_PROMPT),
                                "gemma-4-31b-it", "mmstar-subsample-100-think")
    segments = tt.render_trajectory(parts, "qwen3.5-4b", peer["llamacpp_prompt_string"])
    fake_ids = {"r\n": [20, 21], "</think>": [30], "\n\na": [22, 23], "<|im_end|>": [31]}
    encoded = [{**s, "tokens": fake_ids[s["text"]]} for s in segments]
    answer = [t for s in encoded for t in s["tokens"]]
    text = "".join(s["text"] for s in segments)
    row = dc.derived_row(peer, answer, text, cross_family_protocol(peer, segments, encoded), VOCAB)
    assert dc.validate_row(row)["target"] == "qwen3.5-4b"
    bad = [{**s, "tokens": [29, 30]} if s["parse_special"] else s for s in encoded]
    with pytest.raises(ValueError, match="encoded segments differ|one native special token"):
        dc.derived_row(peer, [t for s in bad for t in s["tokens"]], text, cross_family_protocol(peer, segments, bad), VOCAB)


def test_write_prep_writes_scorer_inputs(tmp_path):
    import hashlib
    row = same_family_row()
    data = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    row["image_bytes_sha256"] = [hashlib.sha256(data).hexdigest()]
    protocol = json.loads(row["corpus_protocol"])
    protocol["image_bytes_sha256"] = row["image_bytes_sha256"]
    row["corpus_protocol"] = canonical(protocol)
    meta = dc.write_prep(row, tmp_path / "prep", [data])
    assert np.fromfile(tmp_path / "prep" / "tokens.bin", dtype="<i4").tolist() == row["input_ids"]
    assert (meta["n_prefill"], meta["n_answer"], meta["n_past_expected"]) == (5, 3, 4)
    assert (tmp_path / "prep" / meta["image_files"][0]).read_bytes() == data
    with pytest.raises(ValueError, match="raw image hashes differ"):
        dc.write_prep(row, tmp_path / "prep2", [b"other"])


def test_legacy_reference_schema_needs_opt_in(monkeypatch):
    import hashlib
    from lib import reference_dataset as rd
    monkeypatch.setattr(rd, "LEGACY_SCHEMA_SHA256", hashlib.sha256(b"old-reference-v2").hexdigest())
    monkeypatch.delenv(rd.LEGACY_SCHEMA_ENV, raising=False)
    assert rd.schema_supported(rd.SCHEMA_VERSION) and not rd.schema_supported("old-reference-v2")
    monkeypatch.setenv(rd.LEGACY_SCHEMA_ENV, "1")
    assert rd.schema_supported("old-reference-v2") and not rd.schema_supported("other-reference-v2")
