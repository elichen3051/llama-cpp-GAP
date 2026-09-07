#!/usr/bin/env bash
set -euo pipefail
# Paired A/B report from completed metric directories; reference columns must match.
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
pick_lane "${1:-vlm}"

REPORT=$OUT_ROOT/paired-$LANE-kld-${LABEL_A}-vs-${LABEL_B}

"$PYTHON" cli/saved_metrics_paired_compare.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --label-a     "$LABEL_A" \
    --label-b     "$LABEL_B" \
    --out         "${REPORT}.md" \
    --output-json "${REPORT}.json"
echo "report: ${REPORT}.md"
echo "next (optional planning): scripts/04_power_analysis.sh $LANE"
