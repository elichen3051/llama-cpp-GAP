#!/usr/bin/env bash
set -euo pipefail
# Prospective design, observed cap diagnostics and finite-pilot verdict stability.
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    exec "$PYTHON" stats/cli/power_analysis.py --help
fi
pick_lane "${1:-vlm}"

METRIC=${METRIC:-kld}

# Prospective simulation samples complete items with replacement.
: "${TOKEN_CAPS:?set TOKEN_CAPS to a prefix grid no larger than the stored collection cap}"
DESIGN_SIZES=${DESIGN_SIZES:-"25 50 75 100 150 200 300 500"}
REPRO_SIZES=${REPRO_SIZES:-"25 50 75 100"}
POWER_REPS=${POWER_REPS:-2000}
OUTER_REPS=${OUTER_REPS:-200}
REPRO_REPS=${REPRO_REPS:-100}

# SESOI is a chosen B-minus-A effect; empty means precision/MDE only.
SESOI=${SESOI:-}
EFFECT_PROFILE=${EFFECT_PROFILE:-flat}
REFERENCE_CAP=${REFERENCE_CAP:-}

echo "=== sequential-prefix design (lane=$LANE, metric=$METRIC) ==="
POWER_ARGS=(
    --candidate-a "$OUT_KLD_A"
    --candidate-b "$OUT_KLD_B"
    --metric "$METRIC"
    --weighting item
    --effect-profile "$EFFECT_PROFILE"
    --reps "$POWER_REPS"
    --outer-reps "$OUTER_REPS"
    --out "$OUT_ROOT/$LANE-seq-power-$METRIC.md"
    --output-json "$OUT_ROOT/$LANE-seq-power-$METRIC.json"
)
if [[ -n "$SESOI" ]]; then
    POWER_ARGS+=(--sesoi "$SESOI")
fi
if [[ "$EFFECT_PROFILE" == pilot ]]; then
    if [[ -z "$REFERENCE_CAP" ]]; then
        echo "ERROR: EFFECT_PROFILE=pilot requires REFERENCE_CAP" >&2
        exit 2
    fi
    POWER_ARGS+=(--reference-cap "$REFERENCE_CAP")
fi
# shellcheck disable=SC2086  # TOKEN_CAPS/DESIGN_SIZES are deliberate word-split grids
"$PYTHON" stats/cli/power_analysis.py \
    "${POWER_ARGS[@]}" \
    --token-caps   $TOKEN_CAPS \
    --sample-sizes $DESIGN_SIZES

echo

case "$METRIC" in
kld|reversed_kld|js_kld|nll)
echo "=== variance decomposition (lane=$LANE, metric=$METRIC) ==="
"$PYTHON" stats/cli/variance_decomposition.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --metric      "$METRIC" \
    --output-json "$OUT_ROOT/$LANE-variance-decomposition-$METRIC.json"
;;
*) echo "variance decomposition skipped: metric $METRIC is unsupported" ;;
esac

echo
# Reproducibility samples this finite pilot without replacement.
echo "=== verdict reproducibility curve (reps=$REPRO_REPS, sizes=$REPRO_SIZES) ==="
# shellcheck disable=SC2086  # REPRO_SIZES is a deliberate word-split list
"$PYTHON" stats/cli/random_subsample_power.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --mode        reproducibility \
    --sizes       $REPRO_SIZES \
    --reps        "$REPRO_REPS" \
    --out         "$OUT_ROOT/$LANE-reproducibility-curve.md" \
    --output-json "$OUT_ROOT/$LANE-reproducibility-curve.json"

echo
echo "reproducibility: $OUT_ROOT/$LANE-reproducibility-curve.md"
echo "seq design:      $OUT_ROOT/$LANE-seq-power-$METRIC.md"
