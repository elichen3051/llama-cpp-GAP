#!/usr/bin/env bash
set -euo pipefail
# Text datasets must contain frozen input_ids from the same model family.
# Classic PPL windows use the direct commands in docs/text-bridge.md.
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    exec "$PYTHON" cli/collect_llm_kld.py --help
fi

if [ -z "$LLM_DATASET" ]; then
    echo "error: LLM_DATASET is not set (hub name or a local prepared dir)." >&2
    echo "The LLM lane is optional; skip this step if you only run the VLM lane." >&2
    exit 1
fi
: "${LLM_REF_MODEL:?set LLM_REF_MODEL to the exact GGUF file}"
: "${LLM_CAND_A_MODEL:?set LLM_CAND_A_MODEL to the exact GGUF file}"
: "${LLM_CAND_B_MODEL:?set LLM_CAND_B_MODEL to the exact GGUF file}"
company_collect_args
company_check_scorer "$LLM_KLD_BIN" llama-llm-kld

collect() {
    local cand_model=$1 out_dir=$2
    "$PYTHON" cli/collect_llm_kld.py \
        --ref-model "$LLM_REF_MODEL" --cand-model "$cand_model" \
        --out "$out_dir" \
        --dataset "$LLM_DATASET" --subset "$LLM_SUBSET" --split "$LLM_SPLIT" \
        "${COLLECT_ARGS[@]}" --llama-llm-kld "$LLM_KLD_BIN"
}

collect "$LLM_CAND_A_MODEL" "$LLM_OUT_KLD_A"
collect "$LLM_CAND_B_MODEL" "$LLM_OUT_KLD_B"

echo
echo "metric dirs:"
du -sh "$LLM_OUT_KLD_A" "$LLM_OUT_KLD_B"
echo "next: scripts/05_compare.sh llm"
