#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# 03_paired_test_kld.sh [vlm|llm] — paired A-vs-B report from the stored KLD
# metric dirs (01 / 02). CPU-only, seconds. Hard-fails unless both dirs share
# a bit-identical reference (per-item nll_ref/entropy_ref/argmax_ref columns
# must match exactly).
#
#   ./scripts/03_paired_test_kld.sh          # VLM lane (default)
#   ./scripts/03_paired_test_kld.sh llm      # LLM lane
# =============================================================================
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh"
pick_lane "${1:-vlm}"

REPORT=$OUT_ROOT/paired-$LANE-kld-${LABEL_A}-vs-${LABEL_B}

"$PYTHON" cli/saved_metrics_paired_compare.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --label-a     "$LABEL_A" \
    --label-b     "$LABEL_B" \
    --out         "${REPORT}.md" \
    --output-json "${REPORT}.json"
# Other knobs (defaults shown):
#   --metrics nll kld reversed_kld js_kld ear same_top_rate mse_dp
#   --primary-metric kld --primary-weighting item
#   --ci-method t            (studentized | bca | percentile need
#                             --bootstrap-iters 5000 --seed 1234)
#   --confidence-level 0.95
#   --weighting both --num-eval-tokens -1 --start 0 --end -1
#   --allow-ref-drift          accept non-bit-identical reference (approx. paired)
#   --show-diagnostic-metrics

echo "report: ${REPORT}.md"
echo "next (optional planning): scripts/04_power_analysis.sh $LANE"
