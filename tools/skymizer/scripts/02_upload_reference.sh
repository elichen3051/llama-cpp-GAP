#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    echo 'Set COHORT_PROFILE, CHECKPOINT, MODE and REFERENCE_RUN (or REFERENCE_OUT). Use --dry-run to validate and stage locally.'
    exec "$PYTHON" cli/upload_reference.py --help
fi
: "${COHORT_PROFILE:?set COHORT_PROFILE to the exact generation profile}"
: "${CHECKPOINT:?set CHECKPOINT to the generated checkpoint}"
: "${MODE:?set MODE to the generated mode}"
"$PYTHON" cli/upload_reference.py --private \
    --profiles "$COHORT_PROFILE" --model "$CHECKPOINT" --mode "$MODE" \
    --run "${REFERENCE_RUN:-${REFERENCE_OUT:-$OUT_ROOT/reference}}" "$@"
