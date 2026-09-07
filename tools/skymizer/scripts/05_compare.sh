#!/usr/bin/env bash
set -euo pipefail
# Paired A/B report from completed metric directories; reference columns must match.
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    exec "$PYTHON" stats/cli/saved_metrics_paired_compare.py --help
fi
pick_lane "${1:-vlm}"
[[ $# -eq 0 ]] || shift

REPORT=$OUT_ROOT/paired-$LANE-kld-${LABEL_A}-vs-${LABEL_B}

"$PYTHON" stats/cli/saved_metrics_paired_compare.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --label-a     "$LABEL_A" \
    --label-b     "$LABEL_B" \
    --out         "${REPORT}.md" \
    --output-json "${REPORT}.json" "$@"
echo "comparison complete"
echo "next (optional planning): scripts/06_power_analysis.sh $LANE"
