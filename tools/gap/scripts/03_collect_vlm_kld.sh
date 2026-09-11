#!/usr/bin/env bash
set -euo pipefail
# Collect both candidates against one reference pair on the same frozen rows.
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    exec "$PYTHON" cli/collect_kld.py --help
fi

if [ -z "$VLM_DATASET" ]; then
    echo "error: VLM_DATASET is not set (hub name or a local prepared dir)." >&2
    exit 1
fi
: "${VLM_REF_MODEL:?set VLM_REF_MODEL to the exact GGUF file}"
: "${VLM_CAND_A_MODEL:?set VLM_CAND_A_MODEL to the exact GGUF file}"
: "${VLM_CAND_B_MODEL:?set VLM_CAND_B_MODEL to the exact GGUF file}"
: "${VLM_REF_MMPROJ:?set VLM_REF_MMPROJ to the reference projector}"
company_collect_args
company_check_scorer "$VLM_KLD_BIN" llama-vlm-kld

IMAGE_ARGS=()
[[ -z "$IMAGE_MIN_TOKENS" ]] || IMAGE_ARGS+=(--image-min-tokens "$IMAGE_MIN_TOKENS")
[[ -z "$IMAGE_MAX_TOKENS" ]] || IMAGE_ARGS+=(--image-max-tokens "$IMAGE_MAX_TOKENS")

collect() {
    local cand_model=$1 cand_mmproj=$2 out_dir=$3
    "$PYTHON" cli/collect_kld.py \
        --ref-model "$VLM_REF_MODEL" --ref-mmproj "$VLM_REF_MMPROJ" \
        --cand-model "$cand_model" --cand-mmproj "$cand_mmproj" \
        --out "$out_dir" \
        --dataset "$VLM_DATASET" --subset "$VLM_SUBSET" --split "$VLM_SPLIT" \
        "${COLLECT_ARGS[@]}" "${IMAGE_ARGS[@]}" \
        --llama-vlm-kld "$VLM_KLD_BIN"
}

collect "$VLM_CAND_A_MODEL" "$VLM_CAND_A_MMPROJ" "$VLM_OUT_KLD_A"
collect "$VLM_CAND_B_MODEL" "$VLM_CAND_B_MMPROJ" "$VLM_OUT_KLD_B"

echo
echo "metric dirs:"
du -sh "$VLM_OUT_KLD_A" "$VLM_OUT_KLD_B"
echo "next: scripts/05_compare.sh vlm"
