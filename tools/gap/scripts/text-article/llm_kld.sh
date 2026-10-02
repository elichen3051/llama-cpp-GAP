#!/usr/bin/env bash
# Stage 2: full-vocabulary LLM-KLD of one candidate over every prepared article, then its verification.
#
# usage: llm_kld.sh MODEL REF_MODEL CAND_MODEL PREPARED_DIR CAND_DIR [--allow-vocab-attr-mismatch]
#                   [--parity-with OTHER_CAND_DIR] [--expect-zero-kld]
#   MODEL                        checkpoint name; selects n_ubatch (ubatch_for in env.sh)
#   PREPARED_DIR                 output of prepare_corpus.sh for this reference model and corpus
#   CAND_DIR                     new output: llm.log, llm-kld/ (+ corpus_articles.json copy), verify.log, verify-checks.csv
#   --allow-vocab-attr-mismatch  token attribute metadata may differ; token texts must match id by id (added automatically for
#                                the Gemma family, see needs_vocab_waiver in env.sh)
#   --parity-with OTHER_CAND_DIR a verified candidate of the same model and corpus: reference columns must be bit-identical
#   --expect-zero-kld            the candidate is the reference itself: every KLD must be exactly 0
# Before anything is written, the reference's vocabulary identity must equal the prepared corpus's (wrong PREPARED_DIR guard).
# Fails (non-zero) when that check, the collector or any check of verify_article_collection.py fails.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 5 ]] || usage
MODEL=$1
UBATCH=$(ubatch_for "$MODEL") || die "unknown model: $MODEL"
require_fork
REF=$(abspath "$2"); CAND=$(abspath "$3"); PREPARED=$(abspath "$4"); CDIR=$(abspath "$5"); shift 5
WAIVER=(); PARITY=(); ZERO=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-vocab-attr-mismatch) WAIVER=(--allow-vocab-attr-mismatch); shift ;;
        --parity-with) [[ $# -ge 2 ]] || usage; PARITY=(--parity-with "$(abspath "$2")/llm-kld"); shift 2 ;;
        --expect-zero-kld) ZERO=(--expect-zero-kld); shift ;;
        *) usage ;;
    esac
done
! needs_vocab_waiver "$MODEL" || WAIVER=(--allow-vocab-attr-mismatch)

OUT=$CDIR/llm-kld; LOG=$CDIR/llm.log; VLOG=$CDIR/verify.log; CHECKS=$CDIR/verify-checks.csv
for path in "$OUT" "$LOG" "$VLOG" "$CHECKS"; do require_new "$path"; done
require_file "$REF"; require_file "$CAND"; require_dir "$PREPARED/dataset"; require_file "$PREPARED/corpus_articles.json"
require_exe "$TEXT_BIN/llama-llm-kld"

# The prepared corpus must come from this reference: same vocabulary identity (CPU only).
{ printf '+'; printf ' %q' CUDA_VISIBLE_DEVICES= "$TEXT_BIN/llama-llm-kld" --vocab-identity "$REF"; printf '\n'; } >&2
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    IDENTITY=$(CUDA_VISIBLE_DEVICES= "$TEXT_BIN/llama-llm-kld" --vocab-identity "$REF") || die "llama-llm-kld --vocab-identity failed for $REF"
    "$PYTHON" - "$PREPARED/corpus_articles.json" "$IDENTITY" <<'PY' || die "the vocabulary of $REF differs from $PREPARED/corpus_articles.json (wrong PREPARED_DIR?)"
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    prepared = json.load(stream)["protocol"]["vocabulary"]
sys.exit(0 if prepared == json.loads(sys.argv[2]) else 1)
PY
fi
make_dir "$CDIR"

log "LLM-KLD per article: $MODEL $(basename -- "$CAND") (n_ubatch $UBATCH) -> $OUT"
( cd -- "$FORK_REPO" && run_logged "$LOG" "$PYTHON" "$FORK_REPO/tools/gap/cli/collect_llm_kld.py" \
    --ref-model "$REF" --cand-model "$CAND" --dataset "$PREPARED/dataset" --subset "" --out "$OUT" \
    --llama-llm-kld "$TEXT_BIN/llama-llm-kld" "${RUNTIME_ARGS[@]}" --n-ubatch "$UBATCH" --flash-attn \
    --start 0 --end -1 --num-eval-tokens -1 ${WAIVER[@]+"${WAIVER[@]}"} ) \
    || die "collect_llm_kld.py failed; see $LOG"
run_logged "$LOG" cp -- "$PREPARED/corpus_articles.json" "$OUT/corpus_articles.json" || die "could not copy corpus_articles.json"
( cd -- "$FORK_REPO" && run_logged "$VLOG" "$PYTHON" "$FORK_REPO/tools/gap/cli/verify_article_collection.py" \
    --articles "$OUT/corpus_articles.json" --collection "$OUT" "${RUNTIME_ARGS[@]}" --n-ubatch "$UBATCH" \
    ${PARITY[@]+"${PARITY[@]}"} ${ZERO[@]+"${ZERO[@]}"} --checks-out "$CHECKS" ) \
    || die "verification failed; see $CHECKS and $VLOG"
[[ "${DRY_RUN:-0}" == 1 ]] || log "verified: $CHECKS"
