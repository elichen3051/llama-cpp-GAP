#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/00_env.sh" --pipeline
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Prepare a frozen corpus, save reference PPL, collect candidates A/B and verify both bridges.
Required: CORPUS, CORPUS_NAME, LLM_REF_MODEL, LLM_CAND_A_MODEL, LLM_CAND_B_MODEL.
Optional: ARTICLE_INDEX (required for PG article boundaries), TEXT_WORK, TEXT_CHUNKS (-1 for all, 2 for a smoke).
Binary overrides: PPL_BIN, TOKENIZE_BIN, LLM_KLD_BIN. TEXT_WORK must be new.
Runtime is fixed at 512 context/batch/microbatch, 8 threads, F16 KV, all GPU layers and flash attention.
HELP
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'error: use environment overrides; see --help' >&2; exit 2; }
: "${CORPUS:?set CORPUS to the frozen corpus file}"
: "${CORPUS_NAME:?set CORPUS_NAME to wikitext-2-test or pg-full-rss}"
: "${LLM_REF_MODEL:?set LLM_REF_MODEL to the exact reference GGUF}"
: "${LLM_CAND_A_MODEL:?set LLM_CAND_A_MODEL to the exact candidate GGUF}"
: "${LLM_CAND_B_MODEL:?set LLM_CAND_B_MODEL to the exact candidate GGUF}"
TEXT_WORK=${TEXT_WORK:-$OUT_ROOT/text-bridge}
TEXT_CHUNKS=${TEXT_CHUNKS:--1}
if [[ "$TEXT_CHUNKS" != -1 ]] && ! [[ "$TEXT_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'error: TEXT_CHUNKS must be -1 or a positive window count' >&2
    exit 2
fi
for binary in "$PPL_BIN" "$TOKENIZE_BIN" "$LLM_KLD_BIN"; do
    [[ -x "$binary" ]] || { echo "error: missing binary $binary" >&2; exit 1; }
done
[[ -f "$CORPUS" ]] || { echo "error: missing corpus $CORPUS" >&2; exit 1; }
ARTICLE_ARGS=()
[[ -z "${ARTICLE_INDEX:-}" ]] || ARTICLE_ARGS+=(--article-index "$ARTICLE_INDEX")
if [[ "$CORPUS_NAME" == pg-full-rss && ${#ARTICLE_ARGS[@]} -eq 0 ]]; then
    echo 'error: PG requires ARTICLE_INDEX from the frozen corpus acquisition' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$TEXT_WORK")"
mkdir -- "$TEXT_WORK"
PREPARED=$TEXT_WORK/prepared
"$PYTHON" cli/prepare_perplexity_corpus.py \
    --corpus "$CORPUS" --corpus-name "$CORPUS_NAME" --ref-model "$LLM_REF_MODEL" \
    --llama-tokenize "$TOKENIZE_BIN" --llama-llm-kld "$LLM_KLD_BIN" \
    --out "$PREPARED" "${ARTICLE_ARGS[@]}"
PPL_ARGS=(-c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on
          -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks "$TEXT_CHUNKS")
"$PPL_BIN" -m "$LLM_REF_MODEL" -f "$PREPARED/corpus.txt" "${PPL_ARGS[@]}" \
    --save-all-logits "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/reference-ppl.log" 2>&1
chmod a-w "$TEXT_WORK/reference-ppl.bin"
sha256sum "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/reference-ppl.sha256"
LIMIT_ARGS=()
[[ "$TEXT_CHUNKS" == -1 ]] || LIMIT_ARGS+=(--dataset-limit "$TEXT_CHUNKS")
collect() {
    local label=$1 candidate=$2
    "$PPL_BIN" -m "$candidate" "${PPL_ARGS[@]}" \
        --kl-divergence --kl-divergence-base "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/candidate-$label-ppl.log" 2>&1
    sha256sum -c "$TEXT_WORK/reference-ppl.sha256"
    "$PYTHON" cli/collect_llm_kld.py \
        --ref-model "$LLM_REF_MODEL" --cand-model "$candidate" \
        --dataset "$PREPARED/dataset" --subset '' --out "$TEXT_WORK/llm-$label" \
        --llama-llm-kld "$LLM_KLD_BIN" --perplexity-window \
        --n-ctx 512 --n-batch 512 --n-ubatch 512 --tf-chunk -1 \
        --n-threads 8 --metric-threads 8 --n-gpu-layers -2 --flash-attn "${LIMIT_ARGS[@]}"
    "$PYTHON" cli/verify_perplexity_bridge.py \
        --prepared "$PREPARED" --llm-collection "$TEXT_WORK/llm-$label" \
        --ppl-logits "$TEXT_WORK/reference-ppl.bin" --ppl-reference-log "$TEXT_WORK/reference-ppl.log" \
        --ppl-candidate-log "$TEXT_WORK/candidate-$label-ppl.log" --out "$TEXT_WORK/bridge-$label.json"
}
collect a "$LLM_CAND_A_MODEL"
collect b "$LLM_CAND_B_MODEL"
echo "next: scripts/05_compare.sh text --unit window"
