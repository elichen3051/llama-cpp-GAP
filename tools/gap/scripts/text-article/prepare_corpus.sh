#!/usr/bin/env bash
# Stage 1: freeze a corpus as one native-tokenized sequence per article (CPU only, no GPU inference).
# Run once per (reference model, corpus); every candidate of that checkpoint reuses the prepared directory.
#
# usage: prepare_corpus.sh CORPUS_FILE CORPUS_NAME ARTICLE_INDEX REF_MODEL OUT_DIR
#   CORPUS_FILE    raw UTF-8 text (wiki.test.raw, pg.txt); every article is tokenized from its exact bytes
#   CORPUS_NAME    protocol label (production: wikitext-2-test-article, pg-full-rss-article)
#   ARTICLE_INDEX  wikitext-2.articles.json or pg.manifest.json (byte spans that partition the corpus)
#   REF_MODEL      reference GGUF whose tokenizer and vocabulary define the tokens (BOS only if the vocabulary adds one)
#   OUT_DIR        new directory: dataset/, logs/, corpus_articles.json (written last); log in OUT_DIR.prepare.log
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2; }
[[ $# -eq 5 ]] || usage
require_fork

CORPUS=$(abspath "$1"); NAME=$2; INDEX=$(abspath "$3"); REF=$(abspath "$4"); OUT=$(abspath "$5")
LOG=$OUT.prepare.log
require_new "$OUT"; require_new "$LOG"
require_file "$CORPUS"; require_file "$INDEX"; require_file "$REF"
require_exe "$TEXT_BIN/llama-tokenize"; require_exe "$TEXT_BIN/llama-llm-kld"
make_dir "$(dirname -- "$OUT")"

log "preparing $NAME per article with $(basename -- "$REF") -> $OUT"
( cd -- "$FORK_REPO" && run_logged "$LOG" "$PYTHON" "$FORK_REPO/tools/gap/cli/prepare_article_corpus.py" \
    --corpus "$CORPUS" --corpus-name "$NAME" --article-index "$INDEX" --ref-model "$REF" \
    --llama-tokenize "$TEXT_BIN/llama-tokenize" --llama-llm-kld "$TEXT_BIN/llama-llm-kld" --out "$OUT" ) \
    || die "prepare_article_corpus.py failed; see $LOG"
[[ "${DRY_RUN:-0}" == 1 ]] || log "prepared: $OUT/corpus_articles.json"
