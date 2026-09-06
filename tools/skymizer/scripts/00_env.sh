#!/usr/bin/env bash
# =============================================================================
# 00_env.sh — shared configuration for the numbered pipeline in scripts/.
#
# Sourced by every 0N_*.sh script here; not runnable on its own. Edit the
# "VLM lane" / "LLM lane" blocks per machine / experiment — every value is an
# environment override (VAR=... ./scripts/01_save_vlm_kld.sh). All relative
# paths are resolved from tools/skymizer/ (the scripts cd there first).
#
# The two lanes are independent: the VLM lane needs (model, mmproj) pairs and
# an image ground-truth dataset; the LLM lane needs text models and a text
# ground-truth dataset tokenized by the SAME model family. Skip a lane by
# simply not running its collect step.
# =============================================================================

# --- LD_LIBRARY_PATH hygiene: AMI shell profiles often inject /usr/local/cuda*
# here, which can shadow CUDA libraries bundled inside pip wheels and mix two
# toolkit versions in one process. Strip the system-CUDA entries; nothing in
# this pipeline needs them (llama.cpp binaries resolve via RPATH).
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -v '^/usr/local/cuda' | paste -sd: - || true)
    if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
fi

# --- python resolution: a RELATIVE $PYTHON override (e.g.
# PYTHON=.venv/bin/python) is resolved against the caller's cwd BEFORE we cd;
# bare names (python3) stay PATH lookups.
SKYMIZER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${PYTHON:-}" ]; then
    PYTHON=$SKYMIZER_DIR/../../.venv/bin/python
    [ -x "$PYTHON" ] || PYTHON=python3
fi
case "$PYTHON" in
    /*) ;;
    */*)
        PYTHON_DIR=$(cd -- "$(dirname -- "$PYTHON")" 2>/dev/null && pwd) || {
            echo "error: PYTHON=$PYTHON not found (relative to $PWD)" >&2
            exit 1
        }
        PYTHON=$PYTHON_DIR/$(basename -- "$PYTHON")
        ;;
esac

# cd into tools/skymizer/ regardless of where the caller invoked the script.
cd "$SKYMIZER_DIR"

VLM_KLD_BIN=${VLM_KLD_BIN:-../../build/bin/llama-vlm-kld}
LLM_KLD_BIN=${LLM_KLD_BIN:-../../build/bin/llama-llm-kld}

# ---------------------------------------------------------------------------
# VLM lane (edit per machine / experiment).
#
# Default A/B design — LLM quantization isolated: every role shares the SAME
# bf16 mmproj, only the LLM GGUF varies. To measure the projector instead,
# keep the LLM GGUF identical across A and B and vary the mmproj
# (see docs/compare.md "What gets compared").
# ---------------------------------------------------------------------------
VLM_MODEL_DIR=${VLM_MODEL_DIR:-$HOME/models/qwen3.5-4b/bartowski}

VLM_REF_MODEL=${VLM_REF_MODEL:-$VLM_MODEL_DIR/Qwen_Qwen3.5-4B-bf16.gguf}
VLM_REF_MMPROJ=${VLM_REF_MMPROJ:-$VLM_MODEL_DIR/mmproj-Qwen_Qwen3.5-4B-bf16.gguf}

VLM_CAND_A_MODEL=${VLM_CAND_A_MODEL:-$VLM_MODEL_DIR/Qwen_Qwen3.5-4B-Q4_K_M.gguf}
VLM_CAND_A_MMPROJ=${VLM_CAND_A_MMPROJ:-$VLM_REF_MMPROJ}

VLM_CAND_B_MODEL=${VLM_CAND_B_MODEL:-$VLM_MODEL_DIR/Qwen_Qwen3.5-4B-Q4_1.gguf}
VLM_CAND_B_MMPROJ=${VLM_CAND_B_MMPROJ:-$VLM_REF_MMPROJ}

VLM_LABEL_A=${VLM_LABEL_A:-Q4_K_M}
VLM_LABEL_B=${VLM_LABEL_B:-Q4_1}

# Image ground-truth dataset (the GAP collection; any config whose generating
# model family matches the GGUFs above works).
VLM_DATASET=${VLM_DATASET:-elichen-skymizer/GAP-mmmu-pro-standard-10}
VLM_SUBSET=${VLM_SUBSET:-qwen3.5-4b-ins-gen-2048}
VLM_SPLIT=${VLM_SPLIT:-train}

# Per-image vision-token bounds forwarded to mtmd (-1 = GGUF model metadata).
IMAGE_MIN_TOKENS=${IMAGE_MIN_TOKENS:--1}
IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS:--1}

# ---------------------------------------------------------------------------
# LLM lane (optional; text-only twin of the VLM lane).
#
# The dataset's input_ids are consumed directly, so it must have been
# tokenized by the same model family as the GGUFs — a locally prepared
# ground-truth directory containing parquet files works as --dataset too.
# ---------------------------------------------------------------------------
LLM_MODEL_DIR=${LLM_MODEL_DIR:-$HOME/models/qwen3.5-4b/bartowski}

LLM_REF_MODEL=${LLM_REF_MODEL:-$LLM_MODEL_DIR/Qwen_Qwen3.5-4B-bf16.gguf}
LLM_CAND_A_MODEL=${LLM_CAND_A_MODEL:-$LLM_MODEL_DIR/Qwen_Qwen3.5-4B-Q4_K_M.gguf}
LLM_CAND_B_MODEL=${LLM_CAND_B_MODEL:-$LLM_MODEL_DIR/Qwen_Qwen3.5-4B-Q4_1.gguf}
LLM_LABEL_A=${LLM_LABEL_A:-Q4_K_M}
LLM_LABEL_B=${LLM_LABEL_B:-Q4_1}

# Text ground-truth dataset (hub name or a local prepared dir).
LLM_DATASET=${LLM_DATASET:-}          # REQUIRED for the LLM lane; no default
LLM_SUBSET=${LLM_SUBSET:-}
LLM_SPLIT=${LLM_SPLIT:-train}

# ---------------------------------------------------------------------------
# Shared knobs (compare guards enforce consistency across runs you compare).
# ---------------------------------------------------------------------------
# Cap on teacher-forced answer positions per row (-1 = all).
N_EVAL_TOKENS=${N_EVAL_TOKENS:-2048}

# Teacher-forcing chunk size. Batched decode (chunk > 1) shifts logits by FP
# non-associativity, so EVERY collect run you want to compare must share this
# value (enforced via collect_meta.json + the report's reference bit-identity
# check). Lower it if VRAM is tight; 1 = bit-exact per-token numerics (slow).
TF_CHUNK=${TF_CHUNK:-16}

# KV context per model (NOT a compare-guard field on its own — but it is part
# of the dir identity, so shards into one --out must match). The KLD path
# keeps TWO models (+ mmprojs on the VLM lane) resident.
N_CTX=${N_CTX:-32768}

# Row range. Metrics cost ~44 B/position, so the whole dataset is collected
# by default. Collectors refuse a window that overlaps rows already
# collected: EXTEND a dir by moving START, not END.
KLD_START=${KLD_START:-0}
KLD_END=${KLD_END:--1}                # -1 = all rows

# ---------------------------------------------------------------------------
# Output layout (everything under outputs/ is gitignored).
# ---------------------------------------------------------------------------
OUT_ROOT=${OUT_ROOT:-outputs}
VLM_OUT_KLD_A=$OUT_ROOT/vlm-kld-ref-vs-a
VLM_OUT_KLD_B=$OUT_ROOT/vlm-kld-ref-vs-b
LLM_OUT_KLD_A=$OUT_ROOT/llm-kld-ref-vs-a
LLM_OUT_KLD_B=$OUT_ROOT/llm-kld-ref-vs-b

# Resolve one lane's variables into the generic names the later steps use.
# Usage: pick_lane [vlm|llm]   (default vlm)
pick_lane() {
    LANE=${1:-vlm}
    case "$LANE" in
        vlm) OUT_KLD_A=$VLM_OUT_KLD_A; OUT_KLD_B=$VLM_OUT_KLD_B
             LABEL_A=$VLM_LABEL_A;     LABEL_B=$VLM_LABEL_B ;;
        llm) OUT_KLD_A=$LLM_OUT_KLD_A; OUT_KLD_B=$LLM_OUT_KLD_B
             LABEL_A=$LLM_LABEL_A;     LABEL_B=$LLM_LABEL_B ;;
        *)   echo "error: unknown lane '$LANE' (vlm|llm)" >&2; exit 1 ;;
    esac
}
