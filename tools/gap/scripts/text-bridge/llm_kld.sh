#!/usr/bin/env bash
# Stage 4: full-vocabulary LLM-KLD over the same 512-token windows (collect_llm_kld.py -> llama-llm-kld).
#
# usage: llm_kld.sh REF_MODEL CAND_MODEL PREPARED_DIR CAND_DIR [--allow-vocab-attr-mismatch]
#   REF_MODEL / CAND_MODEL   same files as the PPL stages
#   PREPARED_DIR             output of prepare_corpus.sh (its dataset/ is the frozen window set)
#   CAND_DIR                 output directory for this candidate: llm.log and llm-kld/
#                            (llm-kld/metrics/*.npz, manifest.csv, collect_meta.json, corpus_windows.json, logs/)
#   --allow-vocab-attr-mismatch   Gemma-family candidates only: token attribute metadata may differ between
#                            the bf16 reference and the candidate; token texts must still match id by id.
#
# Production command (all 254 runs; Gemma runs add --allow-vocab-attr-mismatch):
#   python tools/gap/cli/collect_llm_kld.py --ref-model REF --cand-model CAND \
#     --dataset PREPARED/dataset --subset "" --out CAND_DIR/llm-kld --llama-llm-kld BIN/llama-llm-kld \
#     --perplexity-window --n-ctx 512 --n-batch 512 --n-ubatch 512 --tf-chunk -1 --n-threads 8 \
#     --metric-threads 8 --n-gpu-layers -2 --flash-attn --start 0 --end -1 --dataset-limit -1 --num-eval-tokens -1
# The collector prepares one tokens.bin per window and runs the scorer once with a JSONL manifest
# (rows: {"tokens_in", "n_prefill": 257, "output_metrics"}), i.e.:
#   llama-llm-kld --ref-model REF --cand-model CAND --manifest CAND_DIR/llm-kld/_manifest.jsonl \
#     --num-eval-tokens -1 -b 512 -c 512 -ub 512 -ngl -2 --tf-chunk -1 -t 8 --metric-threads 8 \
#     --flash-attn [--allow-vocab-attr-mismatch] --perplexity-window
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 4 && $# -le 5 ]] || usage
require_fork

REF=$(abspath "$1"); CAND=$(abspath "$2"); PREPARED=$(abspath "$3"); CDIR=$4; shift 4
WAIVER=0; EXTRA=()
for a in "$@"; do
    [[ "$a" == --allow-vocab-attr-mismatch ]] || usage
    WAIVER=1; EXTRA=(--allow-vocab-attr-mismatch)
done
require_file "$REF"; require_file "$CAND"; require_dir "$PREPARED/dataset"; require_exe "$TEXT_BIN/llama-llm-kld"
read -r W V WS T <<<"$(prepared_info "$PREPARED")"
WE=$(expected_windows "$W")
mkdir -p -- "$CDIR"
CDIR=$(abspath "$CDIR")
OUT=$CDIR/llm-kld; LOG=$CDIR/llm.log
require_new "$LOG"; require_new "$OUT"

log "LLM-KLD: $(basename -- "$REF") vs $(basename -- "$CAND"), $WE windows, waiver=$WAIVER -> $OUT"
( cd -- "$FORK_REPO" && run_logged "$LOG" "$PYTHON" "$FORK_REPO/tools/gap/cli/collect_llm_kld.py" \
    --ref-model "$REF" --cand-model "$CAND" --dataset "$PREPARED/dataset" --subset "" --out "$OUT" \
    --llama-llm-kld "$TEXT_BIN/llama-llm-kld" "${KLD_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ) \
    || die "collect_llm_kld.py failed; see $LOG"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

# Validation, as in the production dispatcher: metric count, reconciled collection state, recorded flags.
N=$(ls "$OUT/metrics"/*.npz 2>/dev/null | wc -l | tr -d ' ')
[[ "$N" == "$WE" ]] || die "$N metric files under $OUT/metrics, expected $WE"
( cd -- "$FORK_REPO/tools/gap" && CUDA_VISIBLE_DEVICES= "$PYTHON" - "$OUT" "$WAIVER" <<'PY'
import json, pathlib, sys
sys.path.insert(0, ".")
from stats.collection_io import require_collection_success
out = pathlib.Path(sys.argv[1])
require_collection_success(out, "llm-kld")
meta = json.load(open(out / "collect_meta.json"))
if meta.get("perplexity_window") is not True:
    sys.exit("collect_meta lacks perplexity_window")
if bool(meta.get("allow_vocab_attr_mismatch")) != (sys.argv[2] == "1"):
    sys.exit("collect_meta allow_vocab_attr_mismatch does not match the flag given to this script")
print("collection ok:", meta.get("execution_identity", {}).get("llama_cpp_build_commit", "?"))
PY
) || die "LLM-KLD collection did not reconcile; see $OUT"
log "LLM-KLD complete: $N windows in $OUT/metrics"
