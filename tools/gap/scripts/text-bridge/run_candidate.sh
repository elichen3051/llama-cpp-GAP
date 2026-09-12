#!/usr/bin/env bash
# One candidate end to end: candidate PPL (GPU) -> LLM-KLD (GPU) -> bridge verification (CPU).
# Output layout matches runs/<checkpoint>/<corpus>/candidates/<label>/.
#
# usage: run_candidate.sh REF_MODEL CAND_MODEL PREPARED_DIR SEGMENT_DIR LABEL [--allow-vocab-attr-mismatch]
#   SEGMENT_DIR  must already hold reference-ppl.sha256 (run reference_ppl.sh first)
#   LABEL        candidate directory name, e.g. candidate-019--bartowski--IQ4_NL
#   env PPL_BASE overrides the base path recorded in SEGMENT_DIR/reference-ppl.sha256
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
usage() { sed -n '2,9p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 5 && $# -le 6 ]] || usage

REF=$1; CAND=$2; PREPARED=$3; SEG=$4; LABEL=$5; shift 5
[[ "$LABEL" == "${LABEL//\//}" && -n "$LABEL" ]] || die "LABEL must be a single path component"
require_file "$SEG/reference-ppl.sha256"
if [[ -n "${PPL_BASE:-}" ]]; then BASE=$PPL_BASE
elif [[ -f "$SEG/reference-ppl.sha256" ]]; then BASE=$(sha_receipt_path "$SEG/reference-ppl.sha256")
else die "set PPL_BASE or run reference_ppl.sh first"; fi
CDIR=$SEG/candidates/$LABEL
require_new "$CDIR"
make_dir "$CDIR"

"$HERE/candidate_ppl.sh" "$CAND" "$BASE" "$CDIR"
"$HERE/llm_kld.sh" "$REF" "$CAND" "$PREPARED" "$CDIR" "$@"
"$HERE/verify_bridge.sh" "$PREPARED" "$SEG" "$CDIR" "$BASE"
