#!/usr/bin/env bash
# Shared helpers for the per-article stage scripts. Sourced by every stage script; sources env.sh.
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

die()  { echo "error: $*" >&2; exit 1; }
log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

# Missing inputs are fatal, except under DRY_RUN=1 where earlier stages did not produce their outputs.
_missing()     { if [[ "${DRY_RUN:-0}" == 1 ]]; then log "dry-run, would fail: $*"; else die "$*"; fi; }
require_file() { [[ -f "$1" ]] || _missing "missing file: $1"; }
require_dir()  { [[ -d "$1" ]] || _missing "missing directory: $1"; }
require_exe()  { [[ -x "$1" ]] || _missing "missing executable: $1 (set RUNTIME_DIR or TEXT_BIN)"; }
require_new()  { [[ ! -e "$1" ]] || die "already exists, refusing to overwrite evidence (use a new output): $1"; }
make_dir()     { [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p -- "$@"; }

require_fork() {
    [[ -n "$FORK_REPO" ]] || die "set FORK_REPO to the llama.cpp fork source tree that contains tools/gap"
    require_file "$FORK_REPO/tools/gap/cli/collect_llm_kld.py"
    command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON not found"
}

abspath() { "$PYTHON" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

# run_logged LOGFILE cmd args...  -> prints the copy-pasteable command, then runs it with stdout+stderr
# appended to LOGFILE. DRY_RUN=1 prints only.
run_logged() {
    local logfile=$1; shift
    { printf '+'; printf ' %q' "$@"; printf ' >> %q 2>&1\n' "$logfile"; } >&2
    [[ "${DRY_RUN:-0}" == 1 ]] && return 0
    "$@" >> "$logfile" 2>&1
}

# Runtime arguments shared by the collector and the verifier (n_ubatch comes from ubatch_for).
RUNTIME_ARGS=(--n-ctx "$N_CTX" --n-batch "$N_BATCH" --tf-chunk "$TF_CHUNK" --n-threads "$N_THREADS"
              --metric-threads "$METRIC_THREADS" --n-gpu-layers "$N_GPU_LAYERS")

# checks_pass FILE -> success when FILE is a verify-checks.csv whose eight checks all passed
checks_pass() {
    [[ -f "$1" ]] || return 1
    "$PYTHON" - "$1" <<'PY'
import csv, sys
CHECKS = ["manifest", "metrics_files", "lengths", "finite", "targets", "runtime", "reference_parity", "zero_kld"]
try:
    with open(sys.argv[1], newline="") as stream:
        rows = list(csv.reader(stream))
    ok = (rows[0] == ["check", "passed", "detail"] and [row[0] for row in rows[1:]] == CHECKS
          and all(row[1] == "true" for row in rows[1:]))
except (IndexError, OSError, csv.Error):
    ok = False
sys.exit(0 if ok else 1)
PY
}
