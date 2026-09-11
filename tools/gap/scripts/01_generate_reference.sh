#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    echo 'Set COHORT_PROFILE, CHECKPOINT, SOURCE, MODE, COHORT_SIZE; optional REFERENCE_OUT, MODELS_DIR, GPU, HARDWARE.'
    exec "$PYTHON" cli/generate_model_reference.py --help
fi
: "${COHORT_PROFILE:?set COHORT_PROFILE to a reviewed cohort profile}"
: "${CHECKPOINT:?set CHECKPOINT to a profile model key}"
: "${SOURCE:?set SOURCE to a profile source key}"
: "${MODE:?set MODE to instruct or thinking}"
: "${COHORT_SIZE:?set COHORT_SIZE to 100 or 500}"
"$PYTHON" cli/generate_model_reference.py \
    --profiles "$COHORT_PROFILE" --model "$CHECKPOINT" --source "$SOURCE" \
    --mode "$MODE" --size "$COHORT_SIZE" --gpu "${GPU:-0}" --hardware "${HARDWARE:-pro6000}" \
    --models-dir "${MODELS_DIR:-$HOME/models}" --out "${REFERENCE_OUT:-$OUT_ROOT/reference}" \
    --llama-reference "$REFERENCE_BIN" "$@"
