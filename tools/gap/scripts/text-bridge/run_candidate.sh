#!/usr/bin/env bash
# One candidate end to end: candidate PPL (GPU) -> LLM-KLD (GPU) -> bridge verification (CPU), written into the
# two record trees of the archive:
#   RUNS_ROOT/llama-perplexity-records/MODEL/CORPUS/candidates/LABEL/ppl.log
#   RUNS_ROOT/our-llm-kld-records/MODEL/CORPUS/candidates/LABEL/{llm.log,llm-kld/,bridge.log,bridge.json}
#
# usage: run_candidate.sh REF_MODEL CAND_MODEL PREPARED_DIR RUNS_ROOT MODEL CORPUS LABEL [--allow-vocab-attr-mismatch]
#   RUNS_ROOT/llama-perplexity-records/MODEL/CORPUS must already hold reference-ppl.sha256 (run reference_ppl.sh first)
#   LABEL        candidate--<provider>--<quant>, e.g. candidate--bartowski--IQ4_NL
#   env PPL_BASE overrides the base path recorded in reference-ppl.sha256
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 7 && $# -le 8 ]] || usage

REF=$1; CAND=$2; PREPARED=$3; ROOT=$4; MODEL=$5; CORPUS=$6; LABEL=$7; shift 7
for part in "$MODEL" "$CORPUS" "$LABEL"; do
    [[ -n "$part" && "$part" == "${part//\//}" ]] || die "MODEL, CORPUS and LABEL must be single path components"
done
PPL_SEG=$ROOT/llama-perplexity-records/$MODEL/$CORPUS
KLD_SEG=$ROOT/our-llm-kld-records/$MODEL/$CORPUS
require_file "$PPL_SEG/reference-ppl.sha256"
if [[ -n "${PPL_BASE:-}" ]]; then BASE=$PPL_BASE
elif [[ -f "$PPL_SEG/reference-ppl.sha256" ]]; then BASE=$(sha_receipt_path "$PPL_SEG/reference-ppl.sha256")
else die "set PPL_BASE or run reference_ppl.sh first"; fi
PPL_CDIR=$PPL_SEG/candidates/$LABEL
KLD_CDIR=$KLD_SEG/candidates/$LABEL
require_new "$PPL_CDIR"; require_new "$KLD_CDIR"
make_dir "$PPL_CDIR" "$KLD_CDIR"

"$HERE/candidate_ppl.sh" "$CAND" "$BASE" "$PPL_CDIR"
"$HERE/llm_kld.sh" "$REF" "$CAND" "$PREPARED" "$KLD_CDIR" "$@"
PPL_LOG=$PPL_CDIR/ppl.log "$HERE/verify_bridge.sh" "$PREPARED" "$PPL_SEG" "$KLD_CDIR" "$BASE"
