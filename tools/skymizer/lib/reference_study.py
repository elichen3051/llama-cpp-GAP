"""Shared generation and collection settings for an operator's study overview."""

import json
from pathlib import Path


def validate_reference_cohort(profiles, size):
    expected = profiles.get("cohort_size", size)
    if type(expected) is not int or expected != size:
        raise ValueError(f"profile cohort_size={expected!r} does not match requested size={size}")


def reference_template(profile, mode):
    modes = profile.get("semantic_modes", profile.get("sampling_args", profile.get("effective_sampling", {})))
    if mode not in ("instruct", "thinking") or mode not in modes:
        raise ValueError(f"unsupported semantic mode: {mode}")
    prompts = profile.get("system_prompt", {})
    template_kwargs = profile.get("chat_template_kwargs", {})
    if not isinstance(prompts, dict) or not isinstance(template_kwargs, dict):
        raise ValueError("profile system_prompt and chat_template_kwargs must be mode maps")
    prompt = prompts.get(mode, "")
    kwargs = template_kwargs.get(mode, {})
    if not isinstance(prompt, str) or not isinstance(kwargs, dict) or any(not isinstance(k, str) for k in kwargs):
        raise ValueError("profile template requires a string system prompt and a JSON object of kwargs")
    if "enable_thinking" in kwargs:
        raise ValueError("enable_thinking is selected by the semantic mode, not profile template kwargs")
    return {
        "system_prompt": prompt,
        "chat_template_kwargs": {key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                 for key, value in {"preserve_reasoning": True, **kwargs}.items()},
    }


def kld_runtime(profiles, model, mode, hardware):
    generation = profiles["models"][model]["runtime"][hardware][mode]
    return {
        "n_ctx": generation["ctx"], "n_batch": generation["batch"],
        "n_ubatch": generation["ubatch"], "tf_chunk": generation["batch"],
        "n_gpu_layers": -2, "n_threads": generation["threads"], "metric_threads": 8,
        "flash_attn": "enabled", "swa_full": False, "mtp": False,
        "num_eval_tokens": profiles["kld_eval_tokens"][mode],
        "image_token_budget": "inherit reference dataset", "cache_type_k": "f16", "cache_type_v": "f16",
    }


def study_overview(profiles, plan):
    model_root = Path(plan["models_dir"])
    modes = plan["modes"]
    suffix = "pilot" if plan["size"] == 100 else "collect-500"
    models = {}
    for name in plan["models"]:
        profile = profiles["models"][name]
        runtime = {mode: {key: value for key, value in profile["runtime"][plan["hardware"]][mode].items()
                          if key not in ("trial", "validation")} for mode in modes}
        models[name] = {
            "ref_model": str(model_root / profile["model"]),
            "ref_mmproj": str(model_root / profile["mmproj"]),
            "reference_runtime": runtime,
            "kld_runtime": {mode: kld_runtime(profiles, name, mode, plan["hardware"]) for mode in modes},
            "sampling_args": {mode: profile["sampling_args"][mode] for mode in modes},
            "hf_repo": f"elichen-skymizer/{name}-{suffix}",
        }
        if profile.get("head") and any(runtime[m]["draft_max"] for m in modes):
            models[name]["mtp_head"] = str(model_root / profile["head"])
    return {
        "schema": "skymizer-reference-study-v1", "stage": plan.get("stage", "reference"), "hardware": plan["hardware"],
        "source": {**profiles["dataset"], "split": "train", "requested_per_job": plan["num_samples"] or plan["size"]},
        "subsets": [f"{source}-subsample-{plan['size']}-" + ("ins" if mode == "instruct" else "think")
                    for source in plan["sources"] for mode in modes],
        "generation_caps": {mode: profiles["generation_caps"][mode] for mode in modes},
        "reference_common": {"gpu_layers": "all", "flash_attn": "on", "cache_type_k": "f16",
                             "cache_type_v": "f16", "sequences": 1, "fit": "off", "seed": profiles["seed"],
                             "image_token_budget": profiles["image_token_budget"]},
        "models": models, "generation_jobs": len(models) * len(plan["sources"]) * len(modes),
        "layout": {"scripts": "scripts/skymizer", "script_hashes": "scripts/manifest.json",
                   "provenance": "scripts/provenance", "status": "status.json",
                   "reference": "artifacts/<model>/<subset>/attempt-<number>/dataset",
                   "kld": "artifacts/<model>/<subset>/kld/<candidate>/metrics/*.npz",
                   "kld_row_status": "artifacts/<model>/<subset>/kld/<candidate>/manifest.csv"},
        "settings_source": "scripts/skymizer/scripts/reference_model_profiles.json",
        "note": "Derived overview; launchers use the archived profile. Change protocols only in a new study.",
    }
