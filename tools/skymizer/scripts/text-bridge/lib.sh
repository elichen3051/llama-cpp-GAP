#!/usr/bin/env bash
# Shared helpers for the text-bridge scripts. Sourced by every stage script; sources env.sh.
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

die()  { echo "error: $*" >&2; exit 1; }
log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
# Missing inputs are fatal, except under DRY_RUN=1 where earlier stages did not produce their outputs.
_missing()     { if [[ "${DRY_RUN:-0}" == 1 ]]; then log "dry-run, would fail: $*"; else die "$*"; fi; }
require_file() { [[ -f "$1" ]] || _missing "missing file: $1"; }
require_dir()  { [[ -d "$1" ]] || _missing "missing directory: $1"; }
require_exe()  { [[ -x "$1" ]] || _missing "missing executable: $1 (set RUNTIME_DIR; see runtime.json)"; }
require_new()  { [[ ! -e "$1" ]] || die "already exists, refusing to overwrite evidence (use a new output): $1"; }
require_skymizer() {
    [[ -n "$SKYMIZER_REPO" ]] || die "set SKYMIZER_REPO to the llama.cpp checkout that contains tools/skymizer"
    require_file "$SKYMIZER_REPO/tools/skymizer/cli/collect_llm_kld.py"
    command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON not found"
}
abspath() { "$PYTHON" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
sha256_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }
file_sig()    { if stat -c '%s %Y' "$1" >/dev/null 2>&1; then stat -c '%s %Y' "$1"; else stat -f '%z %m' "$1"; fi; }
file_size()   { file_sig "$1" | cut -d' ' -f1; }
# path recorded in a "sha256sum" style receipt ("<hex>  <path>")
sha_receipt_path() { sed -E 's/^[0-9a-f]+ +//' "$1"; }
sha_receipt_hex()  { cut -d' ' -f1 "$1"; }

# run_logged LOGFILE cmd args...  -> prints the copy-pasteable command, then runs it with stdout+stderr
# appended to LOGFILE. DRY_RUN=1 prints only.
run_logged() {
    local logfile=$1; shift
    { printf '+'; printf ' %q' "$@"; printf ' >> %q 2>&1\n' "$logfile"; } >&2
    [[ "${DRY_RUN:-0}" == 1 ]] && return 0
    "$@" >> "$logfile" 2>&1
}

# grep_once ERE FILE -> the first capture group of the single matching line (exactly one line must match)
grep_once() {
    local n
    n=$(grep -cE "$1" "$2" || true)
    [[ "$n" == 1 ]] || die "expected exactly one line matching /$1/ in $2, found $n"
    sed -nE "s/.*$1.*/\\1/p" "$2"
}

# Exact argument lists used for every production run.
PPL_ARGS=(-c "$N_CTX" -b "$N_BATCH" -ub "$N_UBATCH" -t "$N_THREADS" -tb "$N_THREADS" -ngl "$N_GPU_LAYERS_PPL"
          -fa on -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks "$CHUNKS")
KLD_ARGS=(--perplexity-window --n-ctx "$N_CTX" --n-batch "$N_BATCH" --n-ubatch "$N_UBATCH" --tf-chunk -1
          --n-threads "$N_THREADS" --metric-threads "$METRIC_THREADS" --n-gpu-layers "$N_GPU_LAYERS_KLD" --flash-attn
          --start 0 --end -1 --dataset-limit "$CHUNKS" --num-eval-tokens -1)

# prepared_info PREPARED_DIR -> "windows vocab window_size targets_per_window" from the preparer's manifest.json
prepared_info() {
    require_file "$1/manifest.json"
    "$PYTHON" - "$1/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
p = m["protocol"]
print(len(m["windows"]), p["vocabulary"]["size"], p["window_size"], p["targets_per_window"])
PY
}
# expected_windows TOTAL -> windows actually scored under CHUNKS
expected_windows() {
    if [[ "$CHUNKS" == -1 ]] || (( CHUNKS >= $1 )); then echo "$1"; else echo "$CHUNKS"; fi
}
# ppl_base_bytes WINDOWS VOCAB WINDOW_SIZE TARGETS -> exact size of the --save-all-logits base
ppl_base_bytes() {
    "$PYTHON" -c 'import math, sys
w, v, ws, t = map(int, sys.argv[1:])
print(20 + w * ws * 4 + w * t * (2 * math.ceil(v / 2) + 4) * 2)' "$@"
}
