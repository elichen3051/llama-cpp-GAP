#!/usr/bin/env bash
# Stage 5 (CPU only): verify that llama-perplexity and llama-llm-kld scored the same tokens/targets and agree
# on uncompressed mean NLL (1e-5 nats); reports both KLD definitions separately.
#
# usage: verify_bridge.sh PREPARED_DIR SEGMENT_DIR CAND_DIR [PPL_BASE]
#   SEGMENT_DIR  holds reference-ppl.log and reference-ppl.sha256 (from reference_ppl.sh)
#   CAND_DIR     holds llm-kld/ (from llm_kld.sh) and, unless PPL_LOG is set, ppl.log; writes bridge.log, bridge.json
#   PPL_BASE     base path (default: the path recorded in SEGMENT_DIR/reference-ppl.sha256)
#   env PPL_LOG  candidate ppl.log when it lives in the llama-perplexity-records tree (default CAND_DIR/ppl.log)
#
# Production command (after `sha256sum -c reference-ppl.sha256`, CPU queue, nice 10):
#   CUDA_VISIBLE_DEVICES= python tools/gap/cli/verify_perplexity_bridge.py --prepared PREPARED \
#     --llm-collection CAND_DIR/llm-kld --ppl-logits PPL_BASE --ppl-reference-log SEGMENT_DIR/reference-ppl.log \
#     --ppl-candidate-log CAND_DIR/ppl.log --out CAND_DIR/bridge.json
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 3 && $# -le 4 ]] || usage
require_fork

PREPARED=$(abspath "$1"); SEG=$(abspath "$2"); CDIR=$(abspath "$3")
SHA=$SEG/reference-ppl.sha256
PPL_LOG=${PPL_LOG:-$CDIR/ppl.log}
require_file "$SHA"; require_file "$SEG/reference-ppl.log"; require_file "$PPL_LOG"; require_dir "$CDIR/llm-kld"
if [[ $# -eq 4 ]]; then BASE=$(abspath "$4")
elif [[ -f "$SHA" ]]; then BASE=$(sha_receipt_path "$SHA")
else die "no PPL_BASE given and $SHA is missing"; fi
require_file "$BASE"
read -r W V WS T <<<"$(prepared_info "$PREPARED")"
WE=$(expected_windows "$W")
OUT=$CDIR/bridge.json; LOG=$CDIR/bridge.log
require_new "$OUT"; require_new "$LOG"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    log "checking base checksum ($(file_size "$BASE") bytes, full read): $BASE"
    ACTUAL=$(sha256_file "$BASE" | cut -d' ' -f1)
    [[ "$ACTUAL" == "$(sha_receipt_hex "$SHA")" ]] || die "reference base checksum mismatch: $BASE"
    echo "$BASE: OK" >> "$LOG"
fi
( cd -- "$FORK_REPO" && run_logged "$LOG" env CUDA_VISIBLE_DEVICES= nice -n 10 \
    "$PYTHON" "$FORK_REPO/tools/gap/cli/verify_perplexity_bridge.py" \
    --prepared "$PREPARED" --llm-collection "$CDIR/llm-kld" --ppl-logits "$BASE" \
    --ppl-reference-log "$SEG/reference-ppl.log" --ppl-candidate-log "$PPL_LOG" --out "$OUT" ) \
    || die "bridge verification failed; see $LOG"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

"$PYTHON" - "$OUT" "$WE" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if r.get("status") != "passed":
    sys.exit(f"bridge status {r.get('status')!r}, not 'passed'")
if r["windows"] != int(sys.argv[2]):
    sys.exit(f"bridge windows {r['windows']} != expected {sys.argv[2]}")
print(f"bridge passed: windows={r['windows']} targets={r['targets']} "
      f"llm_kld.mean_kld={r['llm_kld']['mean_kld']:.6f} ppl.mean_kld={r['ppl']['mean_kld']:.6f}")
PY
log "bridge verified: $OUT"
