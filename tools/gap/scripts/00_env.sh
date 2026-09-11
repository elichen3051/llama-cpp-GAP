#!/usr/bin/env bash
# Shared wrapper settings. Model and dataset paths are relative to tools/gap.
# COMPANY_WORK selects build/venv/output defaults; relative work paths use the caller cwd.

company_clean_cuda_path() {
    # Keep system CUDA paths from shadowing wheel libraries; native binaries use RPATH.
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
            | grep -v '^/usr/local/cuda' | paste -sd: - || true)
        if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
    fi
}

company_check_scorer() {
    local scorer=$1 target=$2
    if [ ! -x "$scorer" ]; then
        echo "error: $scorer not found." >&2
        printf 'build it first: cmake --build %q --target %q -j\n' "${BUILD_DIR:-../../build}" "$target" >&2
        return 1
    fi
    "$scorer" --self-test
}

company_clean_cuda_path
# Bootstrap needs the helpers without changing cwd or resolving the wrapper Python.
if [[ "${1:-}" == --setup-only ]]; then
    return 0
fi

COMPANY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "${COMPANY_WORK:-}" in
    ""|/*) ;;
    *) COMPANY_WORK=$PWD/$COMPANY_WORK ;;
esac
PYTHON=${PYTHON:-${COMPANY_PYTHON:-}}
if [ -z "$PYTHON" ]; then
    PYTHON=${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-${COMPANY_WORK:+$COMPANY_WORK/venv}}}
    PYTHON=${PYTHON:-$COMPANY_DIR/../../.venv}/bin/python
    [ -x "$PYTHON" ] || PYTHON=python3
fi
# Resolve a relative Python path before cd; bare names remain PATH lookups.
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
cd "$COMPANY_DIR"

BUILD_DIR=${BUILD_DIR:-${COMPANY_WORK:+$COMPANY_WORK/build}}
BUILD_DIR=${BUILD_DIR:-../../build}
VLM_KLD_BIN=${VLM_KLD_BIN:-$BUILD_DIR/bin/llama-vlm-kld}
LLM_KLD_BIN=${LLM_KLD_BIN:-$BUILD_DIR/bin/llama-llm-kld}
REFERENCE_BIN=${REFERENCE_BIN:-$BUILD_DIR/bin/llama-reference}
PPL_BIN=${PPL_BIN:-$BUILD_DIR/bin/llama-perplexity}
TOKENIZE_BIN=${TOKENIZE_BIN:-$BUILD_DIR/bin/llama-tokenize}

# Select exact model files; keep one projector when comparing LLM quantizations.
VLM_REF_MODEL=${VLM_REF_MODEL:-}
VLM_REF_MMPROJ=${VLM_REF_MMPROJ:-}
VLM_CAND_A_MODEL=${VLM_CAND_A_MODEL:-}
VLM_CAND_A_MMPROJ=${VLM_CAND_A_MMPROJ:-$VLM_REF_MMPROJ}
VLM_CAND_B_MODEL=${VLM_CAND_B_MODEL:-}
VLM_CAND_B_MMPROJ=${VLM_CAND_B_MMPROJ:-$VLM_REF_MMPROJ}
VLM_LABEL_A=${VLM_LABEL_A:-A}
VLM_LABEL_B=${VLM_LABEL_B:-B}
VLM_DATASET=${VLM_DATASET:-}
VLM_SUBSET=${VLM_SUBSET:-}
VLM_SPLIT=${VLM_SPLIT:-train}
# Omit image bounds to preserve the native dataset's recorded budget.
IMAGE_MIN_TOKENS=${IMAGE_MIN_TOKENS:-}
IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS:-}

LLM_REF_MODEL=${LLM_REF_MODEL:-}
LLM_CAND_A_MODEL=${LLM_CAND_A_MODEL:-}
LLM_CAND_B_MODEL=${LLM_CAND_B_MODEL:-}
LLM_LABEL_A=${LLM_LABEL_A:-A}
LLM_LABEL_B=${LLM_LABEL_B:-B}
LLM_DATASET=${LLM_DATASET:-}
LLM_SUBSET=${LLM_SUBSET:-}
LLM_SPLIT=${LLM_SPLIT:-train}

# Freeze the same runtime for every candidate in a comparison.
N_EVAL_TOKENS=${N_EVAL_TOKENS:--1}
TF_CHUNK=${TF_CHUNK:-2048}
N_CTX=${N_CTX:-32768}
N_BATCH=${N_BATCH:-2048}
N_UBATCH=${N_UBATCH:-512}
N_THREADS=${N_THREADS:-8}
METRIC_THREADS=${METRIC_THREADS:-8}
N_GPU_LAYERS=${N_GPU_LAYERS:--2}
FLASH_ATTN=${FLASH_ATTN:-on}
KLD_START=${KLD_START:-0}
KLD_END=${KLD_END:--1}

company_collect_args() {
    COLLECT_ARGS=(
        --start "$KLD_START" --end "$KLD_END"
        --num-eval-tokens "$N_EVAL_TOKENS" --tf-chunk "$TF_CHUNK"
        --n-ctx "$N_CTX" --n-batch "$N_BATCH" --n-ubatch "$N_UBATCH"
        --n-threads "$N_THREADS" --metric-threads "$METRIC_THREADS"
        --n-gpu-layers "$N_GPU_LAYERS"
    )
    case "$FLASH_ATTN" in
        on) COLLECT_ARGS+=(--flash-attn) ;;
        off) COLLECT_ARGS+=(--no-flash-attn) ;;
        auto) ;;
        *) echo "error: FLASH_ATTN must be on, off or auto (got '$FLASH_ATTN')." >&2; return 2 ;;
    esac
}

OUT_ROOT=${OUT_ROOT:-${COMPANY_WORK:+$COMPANY_WORK/outputs}}
OUT_ROOT=${OUT_ROOT:-outputs}
VLM_OUT_KLD_A=${VLM_OUT_KLD_A:-$OUT_ROOT/vlm-kld-ref-vs-a}
VLM_OUT_KLD_B=${VLM_OUT_KLD_B:-$OUT_ROOT/vlm-kld-ref-vs-b}
LLM_OUT_KLD_A=${LLM_OUT_KLD_A:-$OUT_ROOT/llm-kld-ref-vs-a}
LLM_OUT_KLD_B=${LLM_OUT_KLD_B:-$OUT_ROOT/llm-kld-ref-vs-b}

pick_lane() {
    LANE=${1:-vlm}
    case "$LANE" in
        vlm) OUT_KLD_A=$VLM_OUT_KLD_A; OUT_KLD_B=$VLM_OUT_KLD_B
             LABEL_A=$VLM_LABEL_A;     LABEL_B=$VLM_LABEL_B ;;
        text) OUT_KLD_A=${TEXT_WORK:-$OUT_ROOT/text-bridge}/llm-a; OUT_KLD_B=${TEXT_WORK:-$OUT_ROOT/text-bridge}/llm-b
              LABEL_A=$LLM_LABEL_A; LABEL_B=$LLM_LABEL_B ;;
        llm) OUT_KLD_A=$LLM_OUT_KLD_A; OUT_KLD_B=$LLM_OUT_KLD_B
             LABEL_A=$LLM_LABEL_A;     LABEL_B=$LLM_LABEL_B ;;
        *)   echo "error: unknown lane '$LANE' (vlm|text|llm)" >&2; exit 1 ;;
    esac
}
