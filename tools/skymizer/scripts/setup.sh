#!/usr/bin/env bash
# Install prerequisites, build native tools, sync the locked Python env and run CPU checks.
# SKYMIZER_WORK sets build/venv/tmp/cache defaults; explicit path overrides take priority.
# Overrides: BUILD_DIR, VENV_DIR (then UV_PROJECT_ENVIRONMENT), TMPDIR, UV_CACHE_DIR, CCACHE_DIR, CCACHE_TEMPDIR, CUDA_CACHE_PATH, HF_HOME, XDG_CACHE_HOME.
# Other controls: JOBS, PYTHON_VERSION, FORCE_CPU=1, SKIP_APT=1, SKIP_BUILD=1, SKIP_TESTS=1.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Set SKYMIZER_WORK to a writable work volume. Overrides: BUILD_DIR, VENV_DIR, TMPDIR, UV_CACHE_DIR, CCACHE_DIR, CCACHE_TEMPDIR, CUDA_CACHE_PATH, HF_HOME, XDG_CACHE_HOME, JOBS, PYTHON_VERSION.'
    echo 'Controls: FORCE_CPU=1, SKIP_APT=1, SKIP_BUILD=1, SKIP_TESTS=1. See docs/workflows.md.'
    exit 0
fi
: "${SKYMIZER_WORK:?set SKYMIZER_WORK to a writable work volume}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/00_env.sh" --setup-only
SKYMIZER_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SKYMIZER_DIR/../.." &>/dev/null && pwd)"
BUILD_DIR="${BUILD_DIR:-${SKYMIZER_WORK:-$REPO_ROOT}/build}"
VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-${SKYMIZER_WORK:+$SKYMIZER_WORK/venv}}}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TMPDIR="${TMPDIR:-${SKYMIZER_WORK:-$BUILD_DIR}/tmp}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${SKYMIZER_WORK:-$BUILD_DIR}/uv-cache}"
CCACHE_DIR="${CCACHE_DIR:-$SKYMIZER_WORK/ccache}"
CCACHE_TEMPDIR="${CCACHE_TEMPDIR:-$TMPDIR/ccache}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$SKYMIZER_WORK/cuda-cache}"
HF_HOME="${HF_HOME:-$SKYMIZER_WORK/hf}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SKYMIZER_WORK/xdg-cache}"
# uv resolves relative environment paths from its project; bind overrides to the caller.
for setup_path in BUILD_DIR VENV_DIR TMPDIR UV_CACHE_DIR CCACHE_DIR CCACHE_TEMPDIR CUDA_CACHE_PATH HF_HOME XDG_CACHE_HOME; do
    [[ "${!setup_path}" == /* ]] || printf -v "$setup_path" '%s/%s' "$PWD" "${!setup_path}"
done
export TMPDIR UV_CACHE_DIR CCACHE_DIR CCACHE_TEMPDIR CUDA_CACHE_PATH HF_HOME XDG_CACHE_HOME PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
mkdir -p -- "$TMPDIR" "$UV_CACHE_DIR" "$CCACHE_DIR" "$CCACHE_TEMPDIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$XDG_CACHE_HOME"
JOBS="${JOBS:-$(nproc)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# OpenSSL enables HTTPS downloads; ccache speeds up rebuilds.
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

# Install uv if missing.
if ! command -v uv &>/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv install failed (is ~/.local/bin on PATH?)"

# A CUDA build requires both the driver and nvcc.
DRIVER_CUDA=""
if [[ "${FORCE_CPU:-0}" != 1 ]] && command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1 || true)"
fi

GGML_CUDA=OFF
if [[ -n "$DRIVER_CUDA" ]] && command -v nvcc &>/dev/null; then
    GGML_CUDA=ON
elif [[ -n "$DRIVER_CUDA" ]]; then
    warn "NVIDIA driver found (CUDA $DRIVER_CUDA) but nvcc is missing; building the scorer CPU-only."
    warn "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads) and rerun for the CUDA build."
fi
log "driver CUDA: ${DRIVER_CUDA:-none} | scorer build: GGML_CUDA=$GGML_CUDA"

# Build reference generation, collectors and the text PPL bridge.
SCORER_TARGETS=(llama-reference llama-vlm-kld llama-llm-kld llama-perplexity llama-tokenize)
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

log "uv sync --project $SKYMIZER_DIR --python $PYTHON_VERSION --locked --group dev"
uv sync --project "$SKYMIZER_DIR" --python "$PYTHON_VERSION" --locked --group dev

# Check binaries and the Python environment.
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    for target in "${SCORER_TARGETS[@]}"; do
        [[ -x "$BUILD_DIR/bin/$target" ]] || die "expected binary missing: $BUILD_DIR/bin/$target"
    done
    skymizer_check_scorer "$BUILD_DIR/bin/llama-vlm-kld" llama-vlm-kld || die "llama-vlm-kld self-test failed"
    skymizer_check_scorer "$BUILD_DIR/bin/llama-llm-kld" llama-llm-kld || die "llama-llm-kld self-test failed"
fi

log "verifying python env"
"$VENV_DIR/bin/python" - <<'PY'
import numpy, PIL, datasets
print(f"  numpy {numpy.__version__} | pillow {PIL.__version__} | "
      f"datasets {datasets.__version__}")
PY

if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
    log "running the hermetic test suite"
    (cd "$SKYMIZER_DIR" && SKYMIZER_TEST_BIN="$BUILD_DIR/bin" uv run --project "$SKYMIZER_DIR" --python "$VENV_DIR/bin/python" --no-sync \
        python -m pytest -q -p no:cacheprovider -m "not external_model" --basetemp "$TMPDIR/skymizer-pytest" tests/)
fi

log "done."
cat <<EOF
  primary scorer: $PRIMARY_BIN
  python venv   : $VENV_DIR    (activate: source $VENV_DIR/bin/activate)
  next step     : the canonical collect/compare recipe is in
                  $SKYMIZER_DIR/README.md
EOF
