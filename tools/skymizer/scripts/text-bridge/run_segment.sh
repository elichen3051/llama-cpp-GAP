#!/usr/bin/env bash
# One (reference model, prepared corpus) segment: reference PPL once, then every candidate in order.
# Reproduces runs/<checkpoint>/<corpus>/ for any model and any prepared dataset.
#
# usage: run_segment.sh REF_MODEL PREPARED_DIR SEGMENT_DIR LABEL=CAND_MODEL [LABEL=CAND_MODEL ...]
#   env PPL_BASE                     base path (default SEGMENT_DIR/reference-ppl.bin; needs 60-200 GB)
#   env ALLOW_VOCAB_ATTR_MISMATCH=1  pass --allow-vocab-attr-mismatch to LLM-KLD (Gemma family)
#   env CHUNKS=2                     two-window smoke for every stage (production: -1)
# reference_ppl.sh is skipped when SEGMENT_DIR/reference-ppl.receipt.json already exists (resume).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 4 ]] || usage

REF=$1; PREPARED=$2; SEG=$3; shift 3
for spec in "$@"; do [[ "$spec" == *=* ]] || die "candidate must be LABEL=PATH: $spec"; done
BASE=${PPL_BASE:-$SEG/reference-ppl.bin}
WAIVER=()
[[ "${ALLOW_VOCAB_ATTR_MISMATCH:-0}" != 1 ]] || WAIVER=(--allow-vocab-attr-mismatch)

if [[ -f "$SEG/reference-ppl.receipt.json" ]]; then
    BASE=$(sha_receipt_path "$SEG/reference-ppl.sha256")
    log "reference PPL receipt present, reusing base: $BASE"
else
    "$HERE/reference_ppl.sh" "$REF" "$PREPARED" "$SEG" "$BASE"
fi

for spec in "$@"; do
    LABEL=${spec%%=*}; CAND=${spec#*=}
    PPL_BASE=$BASE "$HERE/run_candidate.sh" "$REF" "$CAND" "$PREPARED" "$SEG" "$LABEL" ${WAIVER[@]+"${WAIVER[@]}"}
done
log "segment complete: $SEG"
