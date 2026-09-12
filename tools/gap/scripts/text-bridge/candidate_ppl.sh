#!/usr/bin/env bash
# Stage 3: candidate perplexity + KLD against the saved reference base (llama-perplexity --kl-divergence).
#
# usage: candidate_ppl.sh CAND_MODEL PPL_BASE CAND_DIR
#   CAND_MODEL  quantized candidate GGUF of the SAME checkpoint as the base's reference
#   PPL_BASE    reference-ppl.bin written by reference_ppl.sh (read-only; verified unchanged afterwards)
#   CAND_DIR    output directory for this candidate: ppl.log
#
# Production command (identical for all 254 candidate runs; no -f, the base carries the tokens):
#   llama-perplexity -m CAND -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on -ctk f16 -ctv f16 \
#     --fit off --no-escape --ppl-stride 0 --chunks -1 --kl-divergence --kl-divergence-base PPL_BASE
# Both --kl-divergence AND --kl-divergence-base are required: the base option alone aliases the
# --save-all-logits writer and would overwrite the reference base.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -eq 3 ]] || usage

CAND=$(abspath "$1"); BASE=$(abspath "$2"); CDIR=$3
require_file "$CAND"; require_file "$BASE"; require_exe "$TEXT_BIN/llama-perplexity"
make_dir "$CDIR"
LOG=$CDIR/ppl.log
require_new "$LOG"
BEFORE=; [[ "${DRY_RUN:-0}" == 1 ]] || BEFORE=$(file_sig "$BASE")

log "candidate PPL: $(basename -- "$CAND") vs base $BASE"
run_logged "$LOG" "$TEXT_BIN/llama-perplexity" -m "$CAND" "${PPL_ARGS[@]}" \
    --kl-divergence --kl-divergence-base "$BASE" \
    || die "llama-perplexity failed; see $LOG"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

# Validation, as in the production dispatcher: exactly one of each summary line, base untouched.
PPL_Q=$(grep_once 'Mean PPL\(Q\)[[:space:]]*:[[:space:]]*([0-9.eE+-]+)' "$LOG")
PPL_B=$(grep_once 'Mean PPL\(base\)[[:space:]]*:[[:space:]]*([0-9.eE+-]+)' "$LOG")
KLD=$(grep_once 'Mean[[:space:]]+KLD:[[:space:]]*([0-9.eE+-]+)' "$LOG")
[[ "$(file_sig "$BASE")" == "$BEFORE" ]] || die "reference base changed during candidate PPL: $BASE"
log "candidate PPL(Q) = $PPL_Q; saved-base PPL = $PPL_B; mean KLD (PPL tool, quantized reference) = $KLD"
