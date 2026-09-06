#!/usr/bin/env bash
# env_setup.sh — one-shot environment bootstrap for the tools/skymizer pipeline.
#
# Sets up everything the scripts in this directory need:
#   1. apt build prerequisites (build-essential, cmake, git, OpenSSL dev, ccache)
#   2. uv (installed to ~/.local/bin if missing)
#   3. llama.cpp build of the two Skymizer KLD targets
#      (CUDA backend when an NVIDIA driver + nvcc are present, CPU otherwise)
#   4. Developer Python env at <repo>/.venv via `uv sync --group dev` — runtime
#      dependencies plus pytest/scipy
#   5. Smoke checks: scorer binary runs, python imports, and the hermetic
#      pytest suite (no GPU/model/dataset needed)
#
# Idempotent — safe to rerun, e.g. after a driver/toolkit upgrade.
#
# Overrides (env vars):
#   BUILD_DIR=<path>     cmake build dir              (default: <repo>/build)
#   JOBS=<n>             parallel build jobs          (default: nproc)
#   PYTHON_VERSION=<v>   venv python                  (default: 3.12)
#   FORCE_CPU=1          CPU scorer build
#   SKIP_APT=1           never run apt-get (die instead if tools are missing)
#   SKIP_BUILD=1         skip the cmake build
#   SKIP_TESTS=1         skip the pytest smoke run
set -euo pipefail

# --- LD_LIBRARY_PATH hygiene: AMI shell profiles often inject /usr/local/cuda*
# here, which can shadow CUDA libraries bundled inside pip wheels and mix two
# toolkit versions in one process. Strip the system-CUDA entries; nothing in
# this pipeline needs them (llama.cpp binaries resolve via RPATH).
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' \
        | grep -v '^/usr/local/cuda' | paste -sd: - || true)
    if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
VENV_DIR="$REPO_ROOT/.venv"   # README's test recipe references ../../.venv
JOBS="${JOBS:-$(nproc)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log()  { printf '\033[1;32m[env_setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[env_setup] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[env_setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. apt prerequisites --------------------------------------------------
# libssl-dev enables HTTPS model downloads in upstream llama.cpp.
# ccache: optional; ggml picks it up automatically and makes rebuilds cheap.
if [[ "${SKIP_APT:-0}" != 1 ]] && command -v apt-get &>/dev/null; then
    missing=()
    command -v cmake  &>/dev/null || missing+=(cmake)
    command -v g++    &>/dev/null || missing+=(build-essential)
    command -v git    &>/dev/null || missing+=(git)
    command -v curl   &>/dev/null || missing+=(curl)
    command -v ccache &>/dev/null || missing+=(ccache)
    pkg-config --exists openssl 2>/dev/null || missing+=(pkg-config libssl-dev)
    if (( ${#missing[@]} )); then
        log "installing apt packages: ${missing[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${missing[@]}"
    fi
fi
command -v cmake &>/dev/null || die "cmake not found (install it, or rerun without SKIP_APT=1)"
command -v g++   &>/dev/null || die "g++ not found (install build-essential)"

# ---- 2. uv -----------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv install failed (is ~/.local/bin on PATH?)"

# ---- 3. CUDA detection -----------------------------------------------------
# DRIVER_CUDA = max CUDA version the installed NVIDIA *driver* supports
# (nvidia-smi banner). nvcc governs whether the CUDA scorer backend can be
# *built*.
DRIVER_CUDA=""
if [[ "${FORCE_CPU:-0}" != 1 ]] && command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1 || true)"
fi

GGML_CUDA=OFF
if [[ -n "$DRIVER_CUDA" ]] && command -v nvcc &>/dev/null; then
    GGML_CUDA=ON
elif [[ -n "$DRIVER_CUDA" ]]; then
    warn "NVIDIA driver found (CUDA $DRIVER_CUDA) but nvcc is missing — building the scorer CPU-only."
    warn "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads) and rerun for the CUDA build."
fi
log "driver CUDA: ${DRIVER_CUDA:-none} | scorer build: GGML_CUDA=$GGML_CUDA"

# ---- 4. build Skymizer C++ scorers -----------------------------------------
SCORER_TARGETS=(llama-reference llama-vlm-kld llama-llm-kld)
PRIMARY_BIN="$BUILD_DIR/bin/llama-vlm-kld"
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release "-DGGML_CUDA=$GGML_CUDA")
    if ! pkg-config --exists openssl 2>/dev/null; then
        warn "OpenSSL dev files not found; building with -DLLAMA_OPENSSL=OFF (HTTPS downloads disabled)"
        CMAKE_ARGS+=(-DLLAMA_OPENSSL=OFF)
    fi
    log "configuring: cmake -S $REPO_ROOT -B $BUILD_DIR ${CMAKE_ARGS[*]}"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
    log "building ${SCORER_TARGETS[*]} with $JOBS jobs (the first CUDA build takes a while)"
    cmake --build "$BUILD_DIR" --target "${SCORER_TARGETS[@]}" -j "$JOBS"
fi

# ---- 5. python env ---------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
log "uv sync --project $SCRIPT_DIR --python $PYTHON_VERSION --group dev"
uv sync --project "$SCRIPT_DIR" --python "$PYTHON_VERSION" --group dev

# ---- 6. smoke checks -------------------------------------------------------
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    for target in "${SCORER_TARGETS[@]}"; do
        [[ -x "$BUILD_DIR/bin/$target" ]] || die "expected binary missing: $BUILD_DIR/bin/$target"
    done
    "$BUILD_DIR/bin/llama-vlm-kld" --self-test || die "llama-vlm-kld self-test failed"
    "$BUILD_DIR/bin/llama-llm-kld" --self-test || die "llama-llm-kld self-test failed"
fi

log "verifying python env"
"$VENV_DIR/bin/python" - <<'PY'
import numpy, PIL, datasets
print(f"  numpy {numpy.__version__} | pillow {PIL.__version__} | "
      f"datasets {datasets.__version__}")
PY

if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
    log "running the hermetic test suite"
    (cd "$SCRIPT_DIR" && uv run --project "$SCRIPT_DIR" --python "$VENV_DIR/bin/python" --no-sync \
        python -m pytest -q -p no:cacheprovider tests/)
fi

log "done."
cat <<EOF
  primary scorer: $PRIMARY_BIN
  python venv   : $VENV_DIR    (activate: source $VENV_DIR/bin/activate)
  next step     : the canonical collect/compare recipe is in
                  $SCRIPT_DIR/README.md (TL;DR section)
EOF
