#!/usr/bin/env python3
"""Generate one model/mode/subset reference job from the measured model profiles."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SKYMIZER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYMIZER))
from lib.reference_study import reference_template, validate_reference_cohort


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", type=Path, default=SKYMIZER / "scripts/reference_model_profiles.json")
    p.add_argument("--model", required=True, help="model directory name under --models-dir")
    p.add_argument("--mode", required=True, choices=["instruct", "thinking"])
    p.add_argument("--source", required=True, help="prepared dataset source, for example mmmu-pro-vision")
    p.add_argument("--size", type=int, choices=[100, 500], default=100)
    p.add_argument("--hardware", choices=["pro6000", "h100"], default="pro6000")
    p.add_argument("--gpu", required=True, help="one CUDA device ordinal or GPU UUID")
    p.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llama-reference", type=Path, default=SKYMIZER.parents[1] / "build/bin/llama-reference")
    p.add_argument("--num-samples", type=int, help="optional first N rows for a small smoke")
    p.add_argument("--ctx", type=int, help="override the context size per sequence")
    p.add_argument("--parallel", type=int, help="parallel sequences (default: runtime profile value or 1)")
    p.add_argument("--max-new-tokens", type=int, help="override the generation cap")
    p.add_argument("--mtp", choices=["profile", "off", "3", "5", "8"], default="profile")
    p.add_argument("--dry-run", action="store_true", help="print the command without loading data or weights")
    args = p.parse_args(argv)
    if not args.gpu or "," in args.gpu or args.gpu == "-1":
        p.error("--gpu must select exactly one GPU")
    for name in ("ctx", "parallel", "max_new_tokens", "num_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return args


def build_command(args, profiles):
    validate_reference_cohort(profiles, args.size)
    if args.source not in profiles["sources"]:
        raise ValueError(f"unknown source {args.source!r}; choose from {profiles['sources']}")
    if args.model not in profiles["models"]:
        raise ValueError(f"unknown production model {args.model!r}; choose from {list(profiles['models'])}")
    model = profiles["models"][args.model]
    template = reference_template(model, args.mode)
    settings = model["runtime"][args.hardware][args.mode]
    ctx = args.ctx or settings["ctx"]
    parallel = args.parallel if args.parallel is not None else settings.get("parallel", 1)
    if type(parallel) is not int or parallel <= 0:
        raise ValueError("parallel must be a positive integer")
    cap = args.max_new_tokens or profiles["generation_caps"][args.mode]
    if cap >= ctx:
        raise ValueError("generation cap must be smaller than context; image/prompt tokens also need room")
    draft = settings["draft_max"] if args.mtp == "profile" else (0 if args.mtp == "off" else int(args.mtp))
    if draft and not model["mtp"]:
        raise ValueError(f"{args.model} has no supported local MTP head")
    if draft and parallel != 1:
        raise ValueError("MTP reference generation requires one sequence; use --parallel 1 or --mtp off")
    paths = {key: args.models_dir.expanduser().resolve() / model[key] for key in ("model", "mmproj")}
    if draft and model["mtp"] == "sidecar":
        paths["head"] = args.models_dir.expanduser().resolve() / model["head"]
    for path in paths.values():
        if not args.dry_run and not path.is_file():
            raise ValueError(f"missing model file: {path}")
    command = [sys.executable, str(SKYMIZER / "cli/generate_reference.py"),
               "--dataset", profiles["dataset"]["repo"], "--revision", profiles["dataset"]["revision"],
               "--subset", f"{args.source}-subsample-{args.size}", "--split", "train",
               "--out", str(args.out.expanduser().resolve()), "--llama-reference", str(args.llama_reference.resolve()),
               "--enable-thinking" if args.mode == "thinking" else "--no-enable-thinking"]
    if args.num_samples is not None:
        command += ["--num-samples", str(args.num_samples)]
    command += ["--", "-m", str(paths["model"]), "--mmproj", str(paths["mmproj"]),
                "-ngl", "all", "-c", str(ctx * parallel), "-b", str(settings["batch"]), "-ub", str(settings["ubatch"]),
                "-t", str(settings["threads"]), "-tb", str(settings["threads_batch"]),
                "-fa", "on", "-ctk", "f16", "-ctv", "f16", "-np", str(parallel), "--fit", "off",
                "-n", str(cap), "--seed", str(profiles["seed"]), *model["sampling_args"][args.mode]]
    if template["system_prompt"]:
        command += ["--system-prompt", template["system_prompt"]]
    if template["chat_template_kwargs"] != {"preserve_reasoning": "true"}:
        command += ["--chat-template-kwargs", json.dumps({key: json.loads(value) for key, value in template["chat_template_kwargs"].items()}, ensure_ascii=False)]
    if draft:
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft),
                    "--spec-draft-n-min", "0", "--spec-draft-p-min", "0", "-ngld", "all"]
        if "head" in paths:
            command += ["--model-draft", str(paths["head"])]
    return command


def main():
    args = parse_args()
    try:
        profiles = json.loads(args.profiles.read_text())
        command = build_command(args, profiles)
        print(shlex.join([f"CUDA_VISIBLE_DEVICES={args.gpu}", *command]), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu})
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
