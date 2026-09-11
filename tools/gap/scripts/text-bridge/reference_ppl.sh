#!/usr/bin/env bash
# Stage 2: reference perplexity and its reusable saved-logits base.
#
# usage: reference_ppl.sh REF_MODEL PREPARED_DIR SEGMENT_DIR [PPL_BASE]
#   REF_MODEL     full-precision reference GGUF (shard 1 for split models)
#   PREPARED_DIR  output of prepare_corpus.sh for this reference model + corpus
#   SEGMENT_DIR   output directory for this (checkpoint, corpus) segment:
#                 reference-ppl.log, reference-ppl.sha256, reference-ppl.receipt.json
#   PPL_BASE      where to save the base (default SEGMENT_DIR/reference-ppl.bin).
#                 Size = 20 + W*512*4 + W*255*(2*ceil(V/2)+4)*2 bytes (W windows, V vocab): 60-200 GB per segment.
#
# Production command (identical for all 18 segments):
#   llama-perplexity -m REF -f PREPARED/corpus.txt -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on \
#     -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks -1 --save-all-logits PPL_BASE
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 3 && $# -le 4 ]] || usage

REF=$(abspath "$1"); PREPARED=$(abspath "$2"); SEG=$3; BASE=${4:-$SEG/reference-ppl.bin}
require_file "$REF"; require_file "$PREPARED/corpus.txt"; require_exe "$TEXT_BIN/llama-perplexity"
read -r W V WS T <<<"$(prepared_info "$PREPARED")"
WE=$(expected_windows "$W")
EXPECTED_BYTES=$(ppl_base_bytes "$WE" "$V" "$WS" "$T")
mkdir -p -- "$SEG" "$(dirname -- "$BASE")"
LOG=$SEG/reference-ppl.log
require_new "$LOG"; require_new "$BASE"; require_new "$SEG/reference-ppl.sha256"

log "reference PPL: $W windows in corpus, scoring $WE; vocab $V; base $EXPECTED_BYTES bytes -> $BASE"
run_logged "$LOG" "$TEXT_BIN/llama-perplexity" -m "$REF" -f "$PREPARED/corpus.txt" "${PPL_ARGS[@]}" \
    --save-all-logits "$BASE" \
    || die "llama-perplexity failed; see $LOG (delete the partial $BASE before retrying)"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

# Validation, as in the production dispatcher: one final estimate, window count, exact base size.
PPL=$(grep_once 'Final estimate: PPL = ([0-9.eE+-]+)' "$LOG")
RUN_CHUNKS=$(grep_once 'calculating perplexity over ([0-9]+) chunks' "$LOG")
[[ "$RUN_CHUNKS" == "$WE" ]] || die "log reports $RUN_CHUNKS chunks, prepared corpus expects $WE"
SIZE=$(file_size "$BASE")
[[ "$SIZE" == "$EXPECTED_BYTES" ]] || die "base size $SIZE != expected $EXPECTED_BYTES"
chmod a-w "$BASE"
sha256_file "$BASE" > "$SEG/reference-ppl.sha256"
DIGEST=$(sha_receipt_hex "$SEG/reference-ppl.sha256")
REF="$REF" BASE="$BASE" SIZE="$SIZE" DIGEST="$DIGEST" WE="$WE" PPL="$PPL" PREPARED="$PREPARED" \
    "$PYTHON" - "$SEG/reference-ppl.receipt.json" <<'PY'
import datetime, json, os, sys
e = os.environ
json.dump({"base_path": e["BASE"], "bytes": int(e["SIZE"]), "sha256": e["DIGEST"], "windows": int(e["WE"]),
           "chunks": int(e.get("CHUNKS", "-1")), "reference_ppl": float(e["PPL"]), "reference_model": e["REF"],
           "prepared": e["PREPARED"], "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
          open(sys.argv[1], "x"), indent=1, sort_keys=True)
PY
log "reference PPL = $PPL; base sha256 = $DIGEST; receipt $SEG/reference-ppl.receipt.json"
