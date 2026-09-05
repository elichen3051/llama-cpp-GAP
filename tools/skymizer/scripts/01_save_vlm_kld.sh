#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# 01_save_vlm_kld.sh — VLM collection step: load (reference, candidate)
# (model, mmproj) pairs together, teacher-force the same rows through both,
# store per-token metrics only (44 B/position — whole-dataset scale, no logit
# storage).
#
# Two runs against the SAME reference pair (bit-identity verified by 03):
#   outputs/vlm-kld-ref-vs-a   metrics/*.npz   (candidate A)
#   outputs/vlm-kld-ref-vs-b                   (candidate B)
#
# Both models + both mmprojs are resident at once. Re-running over a window
# that already has output is REFUSED (no skip, no delete): fresh OUT_ROOT or
# a disjoint KLD_START/KLD_END shard.
# =============================================================================
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh"

if [ ! -x "$VLM_KLD_BIN" ]; then
    echo "error: $VLM_KLD_BIN not found." >&2
    echo "build it first:  cmake --build ../../build --target llama-vlm-kld -j" >&2
    exit 1
fi

# Metric-kernel sanity check: closed-form + naive-reference cross-check.
# No models needed, < 1 s.
"$VLM_KLD_BIN" --self-test

collect() {
    local cand_model=$1 cand_mmproj=$2 out_dir=$3
    "$PYTHON" cli/collect_kld.py \
        --ref-model         "$VLM_REF_MODEL" \
        --ref-mmproj        "$VLM_REF_MMPROJ" \
        --cand-model        "$cand_model" \
        --cand-mmproj       "$cand_mmproj" \
        --out               "$out_dir" \
        --dataset           "$VLM_DATASET" \
        --subset            "$VLM_SUBSET" \
        --split             "$VLM_SPLIT" \
        --start             "$KLD_START" \
        --end               "$KLD_END" \
        --num-eval-tokens   "$N_EVAL_TOKENS" \
        --tf-chunk          "$TF_CHUNK" \
        --image-min-tokens  "$IMAGE_MIN_TOKENS" \
        --image-max-tokens  "$IMAGE_MAX_TOKENS" \
        --n-ctx             "$N_CTX" \
        --llama-vlm-kld     "$VLM_KLD_BIN"
    # Other knobs (defaults shown):
    #   --dataset-limit -1             row cap before the window
    #   --n-batch 2048 --n-ubatch 2048
    #   --swa-full                     full-size SWA KV cache (pre-2026-09-02 behaviour)
    #   --metric-threads -1            CPU threads for the metric kernel
    #   --keep-prep
}

collect "$VLM_CAND_A_MODEL" "$VLM_CAND_A_MMPROJ" "$VLM_OUT_KLD_A"
collect "$VLM_CAND_B_MODEL" "$VLM_CAND_B_MMPROJ" "$VLM_OUT_KLD_B"

echo
echo "metric dirs:"
du -sh "$VLM_OUT_KLD_A" "$VLM_OUT_KLD_B"
echo "next: scripts/03_paired_test_kld.sh vlm"
