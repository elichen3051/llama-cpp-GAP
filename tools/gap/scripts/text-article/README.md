# Per-article text collection

Full-vocabulary LLM-KLD of quantized GGUF candidates against their BF16 reference on plain-text corpora, with **one sequence
per article** and **every token after position 0 scored**. It complements the classic text bridge in `../text-bridge/`, which
concatenates a corpus into one token stream, cuts non-overlapping 512-token windows and scores only their second half (49.8 %
of the tokens; windows cross article boundaries). The old protocol and its tools are unchanged.

## Protocol

- Each article of the index (`wikitext-2.articles.json`: 60 WikiText-2 test articles; `pg.manifest.json`: 217 Paul Graham essays)
  is tokenized on its own from its exact bytes (no newline stripping, `--no-escape --no-parse-special`).
- BOS as llama-perplexity: the reference vocabulary's `add_bos` decides; with it, every sequence starts with BOS and every
  article token is scored; without it, nothing is prepended and the article's first token is context only. A tokenizer that
  appends anything else (EOS) is refused.
- Scoring through the generic `cli/collect_llm_kld.py` path with `n_prefill = 1`: the logits of position 0 score token 1,
  teacher forcing scores tokens 2 … L − 1, so an article of L tokens gives L − 1 records.
- Runtime of the VLM main collection: `n_ctx 32768`, `n_batch 2048`, `tf_chunk 2048`, 8 threads, all layers on the GPU,
  flash attention, no full SWA cache, `n_ubatch` 2048 for `gemma-4-31b-it` and 512 otherwise (`env.sh`); one exception,
  16 metric threads instead of the VLM main collection's 12 (user decision 2026-10-03).

## Stages

```sh
# once per (reference model, corpus), CPU only
prepare_corpus.sh CORPUS_FILE CORPUS_NAME ARTICLE_INDEX REF_MODEL PREPARED_DIR
# every candidate of a checkpoint (GPU)
run_segment.sh REF_MODEL PREPARED_DIR RUNS_ROOT MODEL CORPUS LABEL=CAND_MODEL [LABEL=CAND_MODEL ...]
# or one candidate
llm_kld.sh MODEL REF_MODEL CAND_MODEL PREPARED_DIR CAND_DIR [--allow-vocab-attr-mismatch] [--parity-with OTHER_CAND_DIR] [--expect-zero-kld]
```

Production corpus names: `wikitext-2-test-article`, `pg-full-rss-article`. `DRY_RUN=1` prints every command without running
it. Outputs are never overwritten: an existing output stops the stage.

- `prepare_corpus.sh` runs `cli/prepare_article_corpus.py` and writes `PREPARED_DIR/{dataset/,logs/,corpus_articles.json}`.
  `corpus_articles.json` (written last) holds the protocol and, per article, its span, SHA256, token count and token digests;
  it contains no token ids and no corpus text, so it can ship with the records. The prepared `dataset/` must not ship: its token
  ids reproduce the text.
- `llm_kld.sh` first checks that `llama-llm-kld --vocab-identity REF_MODEL` (CPU) equals the vocabulary recorded in
  `PREPARED_DIR/corpus_articles.json`, so a prepared directory of another reference is refused before anything is written. It
  adds `--allow-vocab-attr-mismatch` automatically for `gemma-4-31b-it` and `gemma-4-e4b-it` (the old campaign's uniform Gemma
  waiver: token attribute metadata may differ, token texts must match id by id). It then
  collects one candidate into `CAND_DIR/llm-kld/` (log `CAND_DIR/llm.log`), copies `corpus_articles.json` into it,
  then runs `cli/verify_article_collection.py` (log `CAND_DIR/verify.log`), which writes `CAND_DIR/verify-checks.csv`
  (`check,passed,detail`) and fails the stage unless every check passes:
  `manifest`, `metrics_files`, `lengths`, `finite`, `targets`, `runtime`, `reference_parity` (with `--parity-with`: same
  reference fingerprint and runtime, same binaries, libraries, GPU model and driver, bit-identical `nll_ref`, `argmax_ref`,
  `entropy_ref`; environment variables and GPU UUIDs are recorded but not compared), `zero_kld` (with `--expect-zero-kld`, for
  the reference scored against itself: `kld` and `reversed_kld` exactly 0, candidate columns bitwise equal to the reference
  columns). An unreadable npz fails the checks that need it; the CSV is always written.
- `run_segment.sh` writes `RUNS_ROOT/our-llm-kld-records/MODEL/CORPUS/candidates/LABEL/` for each candidate in order, after
  checking that `CORPUS` is the `corpus_name` of `PREPARED_DIR/corpus_articles.json`.
  - Parity anchor: the first candidate of the list whose `verify-checks.csv` passes; when none does, the first candidate
    collected in this run. Every newly collected candidate is verified against it.
  - Resume: a candidate whose `verify-checks.csv` passes is not collected again; before any GPU work it is re-checked against
    the anchor once, into `CAND_DIR/parity-<anchor label>-<F>.csv` (log `.log`), where `F` is the first 12 hex digits of the
    SHA256 of the anchor's `llm-kld/collect_meta.json`. A re-collected anchor with the same label has a new `F`, so receipts
    written against an earlier anchor are never reused or trusted.
  - The segment stops before any re-check or collection when a `LABEL` appears twice in the list, when an existing candidate
    has no passing `verify-checks.csv`, when a verified candidate's `llm-kld/corpus_articles.json` is missing or differs from
    `PREPARED_DIR/corpus_articles.json` (it was collected on another preparation of the corpus), or when the anchor's
    `collect_meta.json` is missing. It stops before any collection when a parity receipt for the current anchor has failed or a
    re-check fails (re-checks of candidates listed earlier may already have written their receipts, which remain valid).

## Recovery

Outputs are evidence and are never overwritten or deleted by these scripts. When a candidate fails (collector error, failed
check, interrupted run), inspect `CAND_DIR/llm.log`, `CAND_DIR/verify.log` and `CAND_DIR/verify-checks.csv`, move the whole
`CAND_DIR` aside (for example to a `failed/` directory next to `candidates/`), fix the cause and rerun the segment: verified
candidates are skipped and re-checked against the anchor, the moved candidate is collected again. A failing
`parity-<anchor label>-<F>.csv` means the candidate does not match the anchor (reference columns not bit-identical, a different
runtime, binary or GPU model, or a defect in the candidate's own collection; its `detail` column says which); move the
offending candidate aside (or, if the anchor itself is wrong, every candidate verified against it) and rerun. Receipts of an
earlier anchor stay where they are as evidence; they are ignored because their `F` differs. A verified candidate on another
corpus preparation: rerun with the original `PREPARED_DIR`, or move those candidates aside and collect them again. A verified
candidate without `llm-kld/corpus_articles.json`, or an anchor without `llm-kld/collect_meta.json`, is incomplete evidence: move
it aside and collect it again.

Two limits of these checks: a receipt trusts the anchor collection identified by its `collect_meta.json`, so never edit or
replace files inside a collected `llm-kld/` by hand; and labels that differ only in letter case name the same directory on a
case-insensitive filesystem, so run the collection on a case-sensitive one (the Linux GPU host).

Parity does not compare environment variables or GPU UUIDs (only binaries, libraries, GPU model and driver, plus bit-identical
reference columns), so run a whole campaign on one host image, driver and GPU model; `collect_meta.json` still records the
environment and the host's GPU inventory.

## Tests

From `tools/gap`, with the pinned environment (`uv sync --group dev`):

```sh
python -m pytest tests/test_article_corpus_prepare.py tests/test_collect_llm_kld_articles.py \
  tests/test_llm_kld_article_scoring.py tests/test_article_collection_verify.py -q --checks-csv PATH_OUTSIDE_THE_TREE.csv
```

`test_llm_kld_article_scoring.py` compiles `core/llm-kld.cpp` against stubs (needs `g++`, marked `integration`); the other
suites are hermetic.
