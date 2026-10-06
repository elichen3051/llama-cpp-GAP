"""Move a native thinking reference answer into another model's thinking format.

A source answer is split into reasoning and final answer at the source model's native markers
(split_trajectory). The two texts are then written in the tested model's format (render_trajectory):

  Qwen:   {reasoning}\\n</think>\\n\\n{answer}<|im_end|>        (the prompt already ends with "<think>\\n")
  Gemma:  <|channel>thought\\n{reasoning}\\n<channel|>{answer}<turn|>

Only the structural markers are special tokens; reasoning and answer bytes are kept. The end-of-turn
token is added only when the source stopped at EOS. Reasoning cut by the length limit stays open: no
closing marker and no answer are made up. Rows that need an exception are listed in
thinking_source_exceptions.json, keyed by the SHA256 of the source text.
"""

import hashlib
import json
from pathlib import Path

INCOMPLETE_SENTINEL = "CROSS_FAMILY_UNCLOSED_REASONING_804439CDA6"
STOPS = (("stop", "eos"), ("length", "limit"))

QWEN_THINKING_PROMPT = "<think>\n"
GEMMA_OPENING = "<|channel>thought\n"
PILOT_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")
REVIEWED_MARKERS = PILOT_MARKERS + ("\u25c1think\u25b7", "\u25c1/think\u25b7", "<|eom|>", "<|start|>", "<|message|>")

# policy "pilot": the Qwen/Gemma pilot sources, split as in the first cross-family campaign
# (a Qwen source drops the newlines around "</think>", which the Qwen template adds back).
# policy "reviewed": later sources, split byte-exact; anything unusual needs a registered exception.
SOURCES = {
    "qwen3.5-4b":                {"policy": "pilot", "opening": "", "closing": "</think>", "terminal": 248046},
    "qwen3.6-35b-a3b":           {"policy": "pilot", "opening": "", "closing": "</think>", "terminal": 248046},
    "gemma-4-31b-it":            {"policy": "pilot", "opening": GEMMA_OPENING, "closing": "<channel|>", "terminal": 106},
    "gemma-4-e4b-it":            {"policy": "pilot", "opening": GEMMA_OPENING, "closing": "<channel|>", "terminal": 106},
    "kimi-vl-a3b-thinking-2506": {"policy": "reviewed", "opening": "\u25c1think\u25b7", "closing": "\u25c1/think\u25b7", "terminal": 163586},
    "glm-4.6v-flash":            {"policy": "reviewed", "opening": "<think>", "closing": "</think>", "terminal": 151336},
    "internvl3.5-30b-a3b":       {"policy": "reviewed", "opening": "<think>", "closing": "</think>", "terminal": 151645},
    "muse-glimmer-30b":          {"policy": "reviewed", "opening": " to=self<|message|>",
                                  "closing": "<|eom|><|start|>assistant to=user<|message|>", "terminal": 200008},
}

EXCEPTIONS_PATH = Path(__file__).with_name("thinking_source_exceptions.json")
EXCEPTIONS = json.loads(EXCEPTIONS_PATH.read_text())

QWEN_STRUCTURAL_IDS = (248045, 248046, 248068, 248069)  # <|im_start|> <|im_end|> <think> </think>


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def target_family(target):
    for family in ("qwen", "gemma"):
        if target.startswith(family):
            return family
    raise ValueError(f"no thinking format for tested model: {target}")


def source_rule(model):
    if model not in SOURCES:
        raise ValueError(f"no thinking split rule for source model: {model}")
    return SOURCES[model]


def _reviewed_exception(model, subset, row):
    matches = [r for r in EXCEPTIONS["source_rows"] if (r["model"], r["subset"], r["id"]) == (model, subset, row["id"])]
    if not matches:
        return None
    if len(matches) != 1 or matches[0]["text_sha256"] != text_sha(row["generated_texts"]):
        raise ValueError("source exception identity changed: " + row["id"])
    if matches[0]["stop"] != [row["finish_reason"], row["llamacpp_stop_type"]]:
        raise ValueError("source exception stop condition changed: " + row["id"])
    return matches[0]


def _pilot_exception(model, row):
    matches = [r for r in EXCEPTIONS["pilot_literal_markers"]
               if (r["model"], r["source"], r["id"], r["source_text_sha256"])
               == (model, row["source"], row["id"], text_sha(row["generated_texts"]))]
    if len(matches) != 1:
        raise ValueError("unreviewed thinking marker in content: " + row["id"])
    return matches[0]


def _split_pilot(row, model, rule, stop):
    text = row["generated_texts"]
    qwen = rule["opening"] == ""
    if qwen:
        if not row["llamacpp_prompt_string"].endswith(QWEN_THINKING_PROMPT):
            raise ValueError("Qwen native thinking prompt lacks its opening marker")
        opening, body = "", text
    else:
        opening = rule["opening"] if text.startswith(rule["opening"]) else ""
        body = text[len(opening):]
    closing = rule["closing"]
    separator = ""
    if not qwen and not opening:
        if any(m in text for m in PILOT_MARKERS):
            raise ValueError("unsupported native channel format")
        reasoning, answer, boundary = "", text, True
    elif body.count(closing) > 1:
        raise ValueError("multiple reasoning/answer boundaries")
    elif closing not in body:
        if stop != ("length", "limit"):
            raise ValueError("unclosed reasoning without a length stop")
        reasoning, answer, boundary = body, "", False
    else:
        reasoning, answer = body.split(closing)
        boundary, separator = True, closing
        if qwen:
            if reasoning.endswith("\n"):
                reasoning, separator = reasoning[:-1], "\n" + separator
            for newlines in ("\n\n", "\n"):
                if answer.startswith(newlines):
                    answer, separator = answer[len(newlines):], separator + newlines
                    break
    exception = None
    if any(m in reasoning or m in answer for m in PILOT_MARKERS):
        exception = _pilot_exception(model, row)
        if any(m in answer for m in PILOT_MARKERS) or any(m in reasoning for m in PILOT_MARKERS if m != exception["marker"]):
            raise ValueError("unreviewed thinking marker in content: " + row["id"])
    return opening, separator, reasoning, answer, boundary, exception


def _split_reviewed(row, model, subset, rule, stop):
    text = row["generated_texts"]
    exception = _reviewed_exception(model, subset, row)
    action = exception["action"] if exception else None
    opening, closing, separator = rule["opening"], rule["closing"], ""
    if not text.startswith(opening):
        if action != "direct_answer" or any(m in text for m in REVIEWED_MARKERS):
            raise ValueError("unreviewed thinking text without native opener: " + row["id"])
        opening, reasoning, answer, boundary = "", "", text, True
    else:
        body = text[len(opening):]
        boundary = closing in body
        if boundary:
            reasoning, answer = body.split(closing, 1)
            separator = closing
        else:
            reasoning, answer = body, ""
            if stop != ("length", "limit") and action != "open_reasoning_eos":
                raise ValueError("unreviewed EOS with open reasoning: " + row["id"])
    if any(m in reasoning or m in answer for m in REVIEWED_MARKERS) and action != "literal_markers":
        raise ValueError("unreviewed literal source marker: " + row["id"])
    return opening, separator, reasoning, answer, boundary, exception


def split_trajectory(row, model, subset):
    """Source reasoning and final answer of one native thinking row, with what is needed to rebuild the source text."""
    rule = source_rule(model)
    if not subset.endswith("-think") or row["generation_enable_thinking"] is not True:
        raise ValueError("thinking split requires a thinking source subset")
    stop = (row["finish_reason"], row["llamacpp_stop_type"])
    if stop not in STOPS:
        raise ValueError("unsupported source stop condition")
    if (row["input_ids"][-1] == rule["terminal"]) != (stop == ("stop", "eos")):
        raise ValueError("source terminal token disagrees with its stop condition")
    if rule["policy"] == "pilot":
        opening, separator, reasoning, answer, boundary, exception = _split_pilot(row, model, rule, stop)
    else:
        opening, separator, reasoning, answer, boundary, exception = _split_reviewed(row, model, subset, rule, stop)
    return {"source_model": model, "source_subset": subset, "source_id": row["id"], "source_benchmark": row["source"],
            "source_policy": rule["policy"], "source_opening": opening, "source_separator": separator,
            "reasoning": reasoning, "answer": answer, "has_answer_boundary": boundary,
            "source_has_reasoning": bool(opening) or rule["opening"] == "",
            "source_finish_reason": stop[0], "source_stop_type": stop[1],
            "source_terminal_token": rule["terminal"] if stop[1] == "eos" else None,
            "source_text_sha256": text_sha(row["generated_texts"]), "source_exception": exception}


def literal_tokenization_allowed(parts, target):
    """A reviewed literal "</think>" in the content is split into ordinary Qwen tokens."""
    exception = parts.get("source_exception") or {}
    return (target_family(target) == "qwen" and exception.get("action") == "literal_markers"
            and exception.get("allow_qwen_literal_special") is True)


def validate_source_parts(parts):
    rule = source_rule(parts["source_model"])
    if parts["source_policy"] != rule["policy"]:
        raise ValueError("source split policy changed")
    exception = parts["source_exception"]
    if exception is not None:
        if rule["policy"] == "pilot":
            identity = (parts["source_model"], parts["source_benchmark"], parts["source_id"], parts["source_text_sha256"])
            if exception not in EXCEPTIONS["pilot_literal_markers"] or identity != (
                    exception["model"], exception["source"], exception["id"], exception["source_text_sha256"]):
                raise ValueError("source exception does not match the registry")
        else:
            identity = (parts["source_model"], parts["source_subset"], parts["source_id"], parts["source_text_sha256"])
            if exception not in EXCEPTIONS["source_rows"] or identity != (
                    exception["model"], exception["subset"], exception["id"], exception["text_sha256"]):
                raise ValueError("source exception does not match the registry")
            if exception["stop"] != [parts["source_finish_reason"], parts["source_stop_type"]]:
                raise ValueError("source stop differs from the registered exception")
    action = (exception or {}).get("action")
    stop = (parts["source_finish_reason"], parts["source_stop_type"])
    if stop not in STOPS or parts["source_terminal_token"] != (rule["terminal"] if stop[1] == "eos" else None):
        raise ValueError("source stop or EOG mapping changed")
    opening, separator = parts["source_opening"], parts["source_separator"]
    if opening not in ("", rule["opening"]):
        raise ValueError("source opening marker changed")
    direct = rule["opening"] != "" and opening == ""
    if direct:
        allowed = rule["policy"] == "pilot" or action == "direct_answer"
        if not allowed or parts["reasoning"] or separator or not parts["has_answer_boundary"]:
            raise ValueError("unapproved direct thinking answer")
    if parts["has_answer_boundary"] and not direct:
        closing = rule["closing"]
        valid = {closing}
        if rule["policy"] == "pilot" and rule["opening"] == "":
            valid = {before + closing + after for before in ("", "\n") for after in ("", "\n", "\n\n")}
        if separator not in valid:
            raise ValueError("source reasoning/answer separator changed")
    if not parts["has_answer_boundary"]:
        if separator or parts["answer"] or (stop != ("length", "limit") and action != "open_reasoning_eos"):
            raise ValueError("unapproved open reasoning ending")
    original = opening + parts["reasoning"] + separator + parts["answer"]
    if text_sha(original) != parts["source_text_sha256"]:
        raise ValueError("reasoning and answer do not rebuild the source text")


def render_trajectory(parts, target, prompt):
    """Continuation segments in the tested model's format; parse_special marks structural tokens."""
    qwen = target_family(target) == "qwen"
    if not prompt.endswith(QWEN_THINKING_PROMPT if qwen else "<|turn>model\n"):
        raise ValueError("unexpected tested-model thinking prompt suffix")
    complete = parts["has_answer_boundary"]
    terminal = "<|im_end|>" if qwen else "<turn|>"
    reasoning = parts["reasoning"].strip() if qwen and complete else parts["reasoning"]
    answer = parts["answer"].strip() if complete else ""
    segments = []
    if not qwen and (reasoning or not complete):
        segments += [{"text": "<|channel>", "parse_special": True}, {"text": "thought\n", "parse_special": False}]
    segments.append({"text": reasoning, "parse_special": False})
    if complete:
        if qwen:
            segments += [{"text": "\n", "parse_special": False}, {"text": "</think>", "parse_special": True},
                         {"text": "\n\n", "parse_special": False}]
        elif reasoning:
            segments += [{"text": "\n", "parse_special": False}, {"text": "<channel|>", "parse_special": True}]
        segments.append({"text": answer, "parse_special": False})
    if parts["source_stop_type"] == "eos":
        segments.append({"text": terminal, "parse_special": True})
    merged = []
    for segment in segments:
        if not segment["text"]:
            continue
        if merged and not segment["parse_special"] and not merged[-1]["parse_special"]:
            merged[-1]["text"] += segment["text"]
        else:
            merged.append(dict(segment))
    return merged


def native_continuation(rendered, parts, target):
    """What the tested model's own template writes after the prompt for this reasoning and answer."""
    suffix = rendered["suffix"]
    terminal = "<|im_end|>" if target_family(target) == "qwen" else "<turn|>"
    if not suffix.endswith(terminal + "\n"):
        raise ValueError("native completed template has an unexpected terminal")
    eos = parts["source_stop_type"] == "eos"
    if parts["has_answer_boundary"]:
        return suffix[:-1] if eos else suffix[:-len(terminal + "\n")]
    if suffix.count(INCOMPLETE_SENTINEL) != 1:
        raise ValueError("incomplete template sentinel is not unique")
    return suffix.split(INCOMPLETE_SENTINEL)[0] + parts["reasoning"] + (terminal if eos else "")


def validate_trajectory(protocol, text, answer_ids):
    """Check a derived row's continuation against its recorded source parts and native template oracle."""
    parts = protocol["source_parts"]
    validate_source_parts(parts)
    if protocol["canonical_tokenization_exact"] != (not literal_tokenization_allowed(parts, protocol["target"])):
        raise ValueError("unapproved literal tokenization exception")
    segments = protocol["trajectory_segments"]
    if "".join(s["text"] for s in segments) != text or protocol["trajectory_sha256"] != text_sha(text):
        raise ValueError("rendered trajectory changed")
    if segments != render_trajectory(parts, protocol["target"], protocol["target_prompt"]):
        raise ValueError("trajectory does not use the tested model format")
    oracle = protocol["native_template_oracle"]
    if text_sha(text) != oracle["continuation_sha256"] or not oracle["native_prefix_exact"]:
        raise ValueError("trajectory differs from the tested-model template")
    encoded = protocol["encoded_trajectory_segments"]
    if [{k: s[k] for k in ("text", "parse_special")} for s in encoded] != segments:
        raise ValueError("encoded structural/content boundaries changed")
    if [t for s in encoded for t in s["tokens"]] != list(answer_ids):
        raise ValueError("encoded segments differ from the continuation tokens")
    for segment in encoded:
        if segment["parse_special"] and len(segment["tokens"]) != 1:
            raise ValueError("structural marker is not one native special token")
        if (not protocol["canonical_tokenization_exact"] and not segment["parse_special"]
                and any(t in QWEN_STRUCTURAL_IDS for t in segment["tokens"])):
            raise ValueError("literal content contains structural tokens")
