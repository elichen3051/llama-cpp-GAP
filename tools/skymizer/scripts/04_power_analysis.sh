#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# 04_power_analysis.sh [vlm|llm] — OPTIONAL sample-size / power study on the
# KLD metric dirs (01 / 02). CPU only, runs on already-collected .npz dumps
# in seconds-to-minutes.
#
# 1. power_analysis.py — prospective N x token-cap design surface. Tokens
#    remain inside their ordered item prefix; only complete items are sampled.
#    SESOI=<signed effect> enables prospective power. Without SESOI the tool
#    emits precision / MDE only and never substitutes the pilot effect.
#
# 2. variance_decomposition.py — observed cap-profile diagnostic only:
#    mu(K), empirical Var[d_i(K)], sign flips, saturation and the failure of
#    the sigma_b^2 + sigma_w^2/K approximation. Its observed-effect N columns
#    are not prospective sample-size recommendations.
#
# 3. random_subsample_power.py --mode reproducibility — optional stability
#    diagnostic on this finite pilot. It is deliberately not called power.
# =============================================================================
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh"
pick_lane "${1:-vlm}"

# Metric for the variance decomposition (kld | reversed_kld | js_kld | nll | ...).
METRIC=${METRIC:-kld}

# Design grid. Future item counts may exceed the pilot size because prospective
# simulation samples from the empirical item distribution with replacement.
TOKEN_CAPS=${TOKEN_CAPS:-"16 32 64 128 256 512 1024 2048"}
SIZES=${SIZES:-"25 50 75 100 150 200 300 500"}
POWER_REPS=${POWER_REPS:-2000}
OUTER_REPS=${OUTER_REPS:-200}
REPRO_REPS=${REPRO_REPS:-100}

# Signed candidate-B minus candidate-A effect in METRIC units. Leave unset for
# precision/MDE-only output. Deciding this threshold is a domain decision; the
# script intentionally has no observed-effect fallback.
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
# shellcheck disable=SC2086  # TOKEN_CAPS/SIZES are deliberate word-split grids
"$PYTHON" cli/power_analysis.py \
    "${POWER_ARGS[@]}" \
    --token-caps   $TOKEN_CAPS \
    --sample-sizes $SIZES

echo

echo "=== variance decomposition (lane=$LANE, metric=$METRIC) ==="
"$PYTHON" cli/variance_decomposition.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --metric      "$METRIC" \
    --output-json "$OUT_ROOT/$LANE-variance-decomposition-$METRIC.json"
# Other knobs: --num-eval-tokens -1  --confidence-level 0.95  --power-target 0.80

echo
# This finite-pool curve answers a different question: whether the pilot's own
# verdict is stable under without-replacement subsampling. It is not power.
echo "=== verdict reproducibility curve (reps=$REPRO_REPS, sizes=$SIZES) ==="
# shellcheck disable=SC2086  # SIZES is a deliberate word-split list
"$PYTHON" cli/random_subsample_power.py \
    --candidate-a "$OUT_KLD_A" \
    --candidate-b "$OUT_KLD_B" \
    --mode        reproducibility \
    --sizes       $SIZES \
    --reps        "$REPRO_REPS" \
    --out         "$OUT_ROOT/$LANE-reproducibility-curve.md" \
    --output-json "$OUT_ROOT/$LANE-reproducibility-curve.json"

echo
echo "reproducibility: $OUT_ROOT/$LANE-reproducibility-curve.md"
echo "seq design:      $OUT_ROOT/$LANE-seq-power-$METRIC.md"
echo "variance:     $OUT_ROOT/$LANE-variance-decomposition-$METRIC.json"
