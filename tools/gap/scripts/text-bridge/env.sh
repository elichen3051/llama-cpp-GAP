#!/usr/bin/env bash
# Single shared configuration for the text-bridge collection scripts (llama-perplexity + llama-llm-kld).
# Source this file (lib.sh does it for you). Override any value from the environment before sourcing.
#
#   RUNTIME_DIR    one directory with bin/{llama-perplexity,llama-tokenize,llama-llm-kld} and lib/*.so
#                  (optional; the release ships no binaries, their identities are in runtime.json)
#   TEXT_BIN       binaries directory   } set these two instead of RUNTIME_DIR for a fresh llama.cpp build,
#   TEXT_LIB       shared libraries dir } where both live in build/bin (default when run inside the fork)
#   FORK_REPO  llama.cpp fork source tree providing tools/gap (auto-detected when this directory is
#                  tools/gap/scripts/text-bridge of that checkout). Needed by prepare_corpus.sh,
#                  llm_kld.sh and verify_bridge.sh only.
#   PYTHON         interpreter with tools/gap's locked dependencies (datasets, numpy, pyarrow, huggingface-hub)
#   CHUNKS         -1 = every window (production). A positive number limits BOTH tools to that many windows (smoke).
#   DRY_RUN=1      print the exact command lines instead of running them (no validation).

_TB_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Two homes: <release>/scripts (give RUNTIME_DIR or TEXT_BIN/TEXT_LIB), or <fork>/tools/gap/scripts/text-bridge
# (defaults to the fork's build/bin).
_TB_REPO=$(cd -- "$_TB_SCRIPTS/../../../.." 2>/dev/null && pwd || true)
[[ -n "$_TB_REPO" && -f "$_TB_REPO/tools/gap/cli/collect_llm_kld.py" ]] || _TB_REPO=
export FORK_REPO=${FORK_REPO:-$_TB_REPO}

if [[ -n "${RUNTIME_DIR:-}" ]]; then
    :
elif [[ -d "$_TB_SCRIPTS/../runtime/bin" ]]; then
    RUNTIME_DIR=$(cd -- "$_TB_SCRIPTS/../runtime" && pwd)
else
    RUNTIME_DIR=
fi
export RUNTIME_DIR
if [[ -n "$RUNTIME_DIR" ]]; then
    export TEXT_BIN=${TEXT_BIN:-$RUNTIME_DIR/bin}
    export TEXT_LIB=${TEXT_LIB:-$RUNTIME_DIR/lib}
else
    export TEXT_BIN=${TEXT_BIN:-${BUILD_DIR:-${_TB_REPO:-.}/build}/bin}
    export TEXT_LIB=${TEXT_LIB:-$TEXT_BIN}
fi
export LD_LIBRARY_PATH=$TEXT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export PYTHON=${PYTHON:-python3}

# Frozen protocol. Not tuning knobs: every run under runs/ used exactly these values.
export N_CTX=512            # one 512-token window per sequence (n_seq=1)
export N_BATCH=512
export N_UBATCH=512
export N_THREADS=8          # inference + batch threads: -t 8 -tb 8 (PPL), --n-threads 8 (llm-kld)
export METRIC_THREADS=8     # metric kernel threads: llama-perplexity follows -t; llm-kld --metric-threads 8
export N_GPU_LAYERS_PPL=all # -ngl all
export N_GPU_LAYERS_KLD=-2  # llm-kld spelling of "all layers"
export CHUNKS=${CHUNKS:--1}

export PYTHONDONTWRITEBYTECODE=1
