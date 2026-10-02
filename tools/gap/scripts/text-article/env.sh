#!/usr/bin/env bash
# Runtime of the per-article text collection. Sourced by lib.sh (and directly by tests); prints nothing.
# The collection settings follow the VLM main collection (collect_meta.json of every artifacts-collect-400 collection):
# n_ctx 32768, n_batch 2048, tf_chunk 2048, 8 threads, 12 metric threads, all layers on the GPU, flash attention, no full
# SWA cache, every answer token scored; n_ubatch 2048 for gemma-4-31b-it and 512 for every other model (ubatch_for).

_TA_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Paths are made absolute against the caller's working directory when this file is sourced: the stage scripts later
# run the collector and the verifier after `cd FORK_REPO`.
_ta_abs() { if [[ -z "$1" || "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$PWD" "${1#./}"; fi; }

# The fork source tree: four levels above this directory (tools/gap/scripts/text-article), unless FORK_REPO is set.
_TA_REPO=$(cd -- "$_TA_SCRIPTS/../../../.." 2>/dev/null && pwd || true)
[[ -n "$_TA_REPO" && -f "$_TA_REPO/tools/gap/cli/collect_llm_kld.py" ]] || _TA_REPO=
export FORK_REPO=$(_ta_abs "${FORK_REPO:-$_TA_REPO}")

# Native binaries: RUNTIME_DIR/{bin,lib} when set, else the fork's build/bin.
export RUNTIME_DIR=$(_ta_abs "${RUNTIME_DIR:-}")
if [[ -n "$RUNTIME_DIR" ]]; then
    export TEXT_BIN=$(_ta_abs "${TEXT_BIN:-$RUNTIME_DIR/bin}")
    export TEXT_LIB=$(_ta_abs "${TEXT_LIB:-$RUNTIME_DIR/lib}")
else
    export TEXT_BIN=$(_ta_abs "${TEXT_BIN:-${BUILD_DIR:-${_TA_REPO:-.}/build}/bin}")
    export TEXT_LIB=$(_ta_abs "${TEXT_LIB:-$TEXT_BIN}")
fi
# Prepend TEXT_LIB once: sourcing this file again (run_segment.sh -> llm_kld.sh) must not change the environment.
if [[ "${LD_LIBRARY_PATH:-}" != "$TEXT_LIB" && "${LD_LIBRARY_PATH:-}" != "$TEXT_LIB":* ]]; then
    export LD_LIBRARY_PATH=$TEXT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

export PYTHON=${PYTHON:-python3}
[[ "$PYTHON" != */* ]] || PYTHON=$(_ta_abs "$PYTHON")   # a bare command name stays on PATH
export PYTHONDONTWRITEBYTECODE=1

export N_CTX=32768          # longest article is about 18k tokens
export N_BATCH=2048
export TF_CHUNK=2048
export N_THREADS=8
export METRIC_THREADS=12
export N_GPU_LAYERS=-2      # llm-kld spelling of "all layers"

# ubatch_for MODEL -> the VLM main collection's n_ubatch of that checkpoint
ubatch_for() {
    case "$1" in
        gemma-4-31b-it) echo 2048 ;;
        gemma-4-e4b-it|glm-4.6v-flash|internvl3.5-30b-a3b|kimi-vl-a3b-instruct|kimi-vl-a3b-thinking-2506|\
        muse-glimmer-30b|qwen3.5-4b|qwen3.6-35b-a3b) echo 512 ;;
        *) echo "error: no n_ubatch for model '$1' (not one of the nine GAP checkpoints)" >&2; return 1 ;;
    esac
}

# needs_vocab_waiver MODEL -> success for the Gemma family, whose candidates carry different token attribute metadata
# (the old campaign's uniform --allow-vocab-attr-mismatch waiver; token texts must still match id by id)
needs_vocab_waiver() {
    case "$1" in
        gemma-4-31b-it|gemma-4-e4b-it) return 0 ;;
        *) return 1 ;;
    esac
}
