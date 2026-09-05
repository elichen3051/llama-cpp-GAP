#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# 02_save_llm_kld.sh — OPTIONAL text-only collection step: the LLM twin of
# 01. Loads (reference, candidate) together, teacher-forces the dataset's
# frozen input_ids through both, stores per-token metrics only. No images,
# no mmproj, no chat-template work — the dataset must be tokenized by the
# same model family as the GGUFs (a local parquet directory works
# as LLM_DATASET too).
#
# Two runs against the SAME reference (bit-identity verified by 03):
#   outputs/llm-kld-ref-vs-a   metrics/*.npz   (candidate A)
#   outputs/llm-kld-ref-vs-b                   (candidate B)
# =============================================================================
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh"

if [ -z "$LLM_DATASET" ]; then
    echo "error: LLM_DATASET is not set (hub name or a local prepared dir)." >&2
    echo "The LLM lane is optional — skip this step if you only run the VLM lane." >&2
    exit 1
fi
if [ ! -x "$LLM_KLD_BIN" ]; then
    echo "error: $LLM_KLD_BIN not found." >&2
    echo "build it first:  cmake --build ../../build --target llama-llm-kld -j" >&2
    exit 1
fi

# Metric-kernel sanity check: closed-form + naive-reference cross-check.
# No models needed, < 1 s.
"$LLM_KLD_BIN" --self-test

collect() {
    local cand_model=$1 out_dir=$2
    "$PYTHON" cli/collect_llm_kld.py \
        --ref-model        "$LLM_REF_MODEL" \
        --cand-model       "$cand_model" \
        --out              "$out_dir" \
        --dataset          "$LLM_DATASET" \
        --subset           "$LLM_SUBSET" \
        --split            "$LLM_SPLIT" \
        --start            "$KLD_START" \
        --end              "$KLD_END" \
        --num-eval-tokens  "$N_EVAL_TOKENS" \
        --tf-chunk         "$TF_CHUNK" \
        --n-ctx            "$N_CTX" \
        --llama-llm-kld    "$LLM_KLD_BIN"
    # Other knobs (defaults shown):
    #   --dataset-limit -1             row cap before the window
    #   --n-batch 2048 --n-ubatch 2048
    #   --swa-full                     full-size SWA KV cache (pre-2026-09-02 behaviour)
    #   --metric-threads -1            CPU threads for the metric kernel
    #   --keep-prep
}

collect "$LLM_CAND_A_MODEL" "$LLM_OUT_KLD_A"
collect "$LLM_CAND_B_MODEL" "$LLM_OUT_KLD_B"

echo
echo "metric dirs:"
du -sh "$LLM_OUT_KLD_A" "$LLM_OUT_KLD_B"
echo "next: scripts/03_paired_test_kld.sh llm"
