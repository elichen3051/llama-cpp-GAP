#!/usr/bin/env bash
# Stage 1: freeze a corpus into 512-token windows with the reference model's native tokenizer (no GPU inference).
# Run once per (reference model, corpus); every candidate of that checkpoint reuses the same prepared directory.
#
# usage: prepare_corpus.sh CORPUS_FILE CORPUS_NAME REF_MODEL OUT_DIR [ARTICLE_INDEX]
#   CORPUS_FILE    raw UTF-8 text (e.g. wiki.test.raw). Passed byte-exact; one final newline is stripped for tokenization.
#   CORPUS_NAME    label recorded in the protocol (production: wikitext-2-test, pg-full-rss)
#   REF_MODEL      reference GGUF whose tokenizer/vocabulary defines the windows
#   OUT_DIR        new directory: corpus.txt, effective.txt, dataset/ (HF saved dataset), manifest.json, stream.json
#   ARTICLE_INDEX  JSON with byte_start/byte_end_exclusive article spans (required for PG; WikiText-2 used the
#                  verified 60-article index). Only affects article attribution, not tokens/windows/targets.
#
# Production command:
#   python tools/gap/cli/prepare_perplexity_corpus.py --corpus CORPUS --corpus-name NAME \
#     [--article-index INDEX] --ref-model REF --llama-tokenize BIN/llama-tokenize \
#     --llama-llm-kld BIN/llama-llm-kld --out OUT_DIR
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -ge 4 && $# -le 5 ]] || usage
require_fork

CORPUS=$(abspath "$1"); NAME=$2; REF=$(abspath "$3"); OUT=$4
require_file "$CORPUS"; require_file "$REF"
require_exe "$TEXT_BIN/llama-tokenize"; require_exe "$TEXT_BIN/llama-llm-kld"
require_new "$OUT"
mkdir -p -- "$(dirname -- "$OUT")"
OUT=$(abspath "$OUT")
INDEX_ARGS=()
if [[ $# -eq 5 ]]; then require_file "$5"; INDEX_ARGS=(--article-index "$(abspath "$5")"); fi
[[ "$NAME" != pg-full-rss || ${#INDEX_ARGS[@]} -gt 0 ]] || die "pg-full-rss requires ARTICLE_INDEX"
LOG=$OUT.prepare.log
require_new "$LOG"

log "preparing $NAME with $(basename -- "$REF") -> $OUT"
( cd -- "$FORK_REPO" && run_logged "$LOG" "$PYTHON" "$FORK_REPO/tools/gap/cli/prepare_perplexity_corpus.py" \
    --corpus "$CORPUS" --corpus-name "$NAME" ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"} --ref-model "$REF" \
    --llama-tokenize "$TEXT_BIN/llama-tokenize" --llama-llm-kld "$TEXT_BIN/llama-llm-kld" --out "$OUT" ) \
    || die "prepare_perplexity_corpus.py failed; see $LOG"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
read -r W V WS T <<<"$(prepared_info "$OUT")"
log "prepared $W windows x $T targets (window $WS, vocab $V): $OUT"
