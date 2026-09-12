#!/usr/bin/env bash
# One (reference model, prepared corpus) segment: reference PPL once, then every candidate in order.
# Reproduces the archive layout runs/{llama-perplexity-records,our-llm-kld-records}/MODEL/CORPUS/.
#
# usage: run_segment.sh REF_MODEL PREPARED_DIR RUNS_ROOT MODEL CORPUS LABEL=CAND_MODEL [LABEL=CAND_MODEL ...]
#   RUNS_ROOT                        archive root, e.g. runs
#   MODEL / CORPUS                   directory names, e.g. qwen3.5-4b / wikitext-2-test
#   LABEL                            candidate--<provider>--<quant>
#   env PPL_BASE                     base path (default RUNS_ROOT/llama-perplexity-records/MODEL/CORPUS/reference-ppl.bin; 60-200 GB)
#   env ALLOW_VOCAB_ATTR_MISMATCH=1  pass --allow-vocab-attr-mismatch to LLM-KLD (Gemma family)
#   env CHUNKS=2                     two-window smoke for every stage (production: -1)
# reference_ppl.sh is skipped when the segment's reference-ppl.receipt.json already exists (resume).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 6 ]] || usage

REF=$1; PREPARED=$2; ROOT=$3; MODEL=$4; CORPUS=$5; shift 5
for spec in "$@"; do [[ "$spec" == *=* ]] || die "candidate must be LABEL=PATH: $spec"; done
PPL_SEG=$ROOT/llama-perplexity-records/$MODEL/$CORPUS
BASE=${PPL_BASE:-$PPL_SEG/reference-ppl.bin}
WAIVER=()
[[ "${ALLOW_VOCAB_ATTR_MISMATCH:-0}" != 1 ]] || WAIVER=(--allow-vocab-attr-mismatch)

if [[ -f "$PPL_SEG/reference-ppl.receipt.json" ]]; then
    BASE=$(sha_receipt_path "$PPL_SEG/reference-ppl.sha256")
    log "reference PPL receipt present, reusing base: $BASE"
else
    "$HERE/reference_ppl.sh" "$REF" "$PREPARED" "$PPL_SEG" "$BASE"
fi

for spec in "$@"; do
    LABEL=${spec%%=*}; CAND=${spec#*=}
    PPL_BASE=$BASE "$HERE/run_candidate.sh" "$REF" "$CAND" "$PREPARED" "$ROOT" "$MODEL" "$CORPUS" "$LABEL" ${WAIVER[@]+"${WAIVER[@]}"}
done
log "segment complete: $ROOT/{llama-perplexity-records,our-llm-kld-records}/$MODEL/$CORPUS"
