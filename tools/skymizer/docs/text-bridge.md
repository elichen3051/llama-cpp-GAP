# Classic PPL and full-logit text KLD

This workflow feeds the same frozen text, native GGUF vocabulary, BOS policy, 512-token windows and 255 targets to `llama-perplexity` and `llama-llm-kld`. It is separate from scoring generated prompt/answer trajectories. The [shared setup](workflows.md) builds the required tokenizer, PPL and LLM scorer binaries.

`llama-llm-kld` computes full-vocabulary metrics from uncompressed logits in memory. `llama-perplexity` saves clipped uint16 reference log probabilities. Their input protocol and uncompressed NLL should agree; the two KLD outputs are separate measurements because their references differ in precision. Keep original reference PPL, reconstructed saved-base PPL and candidate PPL separate in results.

## Freeze the corpus

Supply the exact corpus bytes at `CORPUS`. Use a separate work directory per checkpoint and corpus. WikiText-2 top-level headers define articles; the whitespace-only prelude belongs to the first article and nested headings do not start new articles. PG requires a saved article index with an `articles` or `items` list containing contiguous `byte_start` and `byte_end_exclusive` spans over the complete corpus bytes. Preserve article order, titles/URLs, raw acquisition inputs, conversion tool versions and SHA256 receipts.

The accepted September 6, 2026 inputs had these identities:

| Corpus | Bytes | SHA256 |
| --- | ---: | --- |
| WikiText-2 test | 1290590 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| Full PG RSS, 217 articles | 3179044 | `26db5717f58a11a8ed9c24dab9acffc557bcb9d7697b733f44039f13cca4e082` |

These inputs are external data, not bundled repository assets. The WikiText snapshot came from `ggml-org/ci@927b3642933080f1b0e811e2f916e14c292992f9`. The accepted PG conversion used html5lib 1.1 and native html2text 2.4.0, followed by `get-pg.sh`'s `tail`, `sed` and `fmt -w 80` operations in `C.UTF-8`. Re-running a live RSS feed or a different converter creates different input. Do not substitute the historical partial 107-article PG corpus for the complete frozen corpus.

## Run the bridge

Set exact reference and candidate files from one checkpoint. For split models, point at shard 1. The wrapper creates a new `TEXT_WORK`, prepares the corpus once and reuses its dataset and PPL base for candidates A and B.

```bash
export CORPUS=/path/to/frozen/wiki.test.raw
export CORPUS_NAME=wikitext-2-test
export LLM_REF_MODEL=/path/to/reference-bf16.gguf
export LLM_CAND_A_MODEL=/path/to/candidate-a.gguf
export LLM_CAND_B_MODEL=/path/to/candidate-b.gguf
export LLM_LABEL_A=Q4_K_M
export LLM_LABEL_B=Q4_1
export TEXT_WORK="$SKYMIZER_WORK/qwen35-wikitext"

tools/skymizer/scripts/04_collect_text_bridge.sh
```

For PG, select `CORPUS_NAME=pg-full-rss`, its frozen `CORPUS` and `ARTICLE_INDEX`, and a new `TEXT_WORK`. A bounded smoke sets `TEXT_CHUNKS=2` for both PPL and KLD; use a separate output directory and omit it for full-corpus collection. Article/block aggregation refuses that partial smoke. No models or corpus files are downloaded by the wrapper.

The fixed runtime is context/batch/microbatch 512, one sequence, inference/batch/metric threads 8, all supported GPU layers, flash attention on, F16 K/V, fit off and MTP off. SWA full-cache allocation stays off without disabling architectural sliding-window attention. Both uncompressed models remain loaded together for KLD. A successful single-model PPL run does not prove pair capacity. Preserve the binary, loaded libraries, GPU/driver, relevant environment and all runtime settings across candidates.

The wrapper performs these stages using the existing CLIs:

1. `cli/prepare_perplexity_corpus.py` verifies native tokenization and writes `prepared/dataset`, corpus/window/article hashes, `stream.json`, original `corpus.txt` and `effective.txt`.
2. `llama-perplexity --save-all-logits` writes reference PPL and its reusable base, which is then made read-only and hashed.
3. Each candidate PPL command uses both `--kl-divergence` and `--kl-divergence-base FILE`; the latter alone aliases the writer option and can overwrite the base. The wrapper rechecks its SHA after each candidate.
4. `cli/collect_llm_kld.py --perplexity-window` writes completed full-logit metric collections at `llm-a` and `llm-b`.
5. `cli/verify_perplexity_bridge.py` verifies each pair of PPL/KLD outputs and writes `bridge-a.json` or `bridge-b.json`.

The verifier checks exact saved tokens/targets, complete file size, window count and decode shape, explicit thread agreement and uncompressed mean NLL within 1e-5 nats plus printed-PPL rounding. It reports saved-reference NLL error and both KLD means separately. Completion requires verified summaries and reconciled collection state, not only a process exit code. Keep failed outputs and use a new `TEXT_WORK` for another attempt.

The PPL base is large: bytes = `20 + windows*512*4 + windows*255*(2*ceil(vocab/2)+4)*2`, roughly 127.5 MiB per window for a 262144-token vocabulary. Full-logit KLD instead stores 76 bytes per target plus headers and provenance. Check storage capacity before production collection.

## Exact decode contract

1. Remove one final newline from the original corpus when present, matching common's file handling. Do not interpret escapes or parse literal special-token spellings.
2. Tokenize the full effective text using the reference GGUF's normal leading-BOS policy. The preparer checks native `llama-tokenize` output with and without `--no-bos`.
3. Split into non-overlapping 512-token windows, keeping all complete windows and dropping the incomplete tail. Classic PPL requires at least 1024 input tokens.
4. When BOS is required, replace token 0 in every window; do not insert an extra token or retokenize at article boundaries.
5. Clear KV before each window and decode all 512 tokens in one batch at positions 0..511, sequence 0. Request logits at positions 256..511, including the unused final output.
6. Score logits 256..510 against targets 257..511: exactly 255 targets. `n_prefill=257` identifies the first target; `n_past_actual=512` records the full decoded window.

Both vocabularies must use `add_eos=false`. The prepared field `generated_tokens_len=255` describes teacher-forced targets, not generated answers. Pass the original `prepared/corpus.txt` to PPL, since it applies its own final-newline handling; do not pass `effective.txt` instead.

## Paired windows, articles and blocks

The wrapper's `text` lane finds the two completed collections under `TEXT_WORK`. Choose grouping and weighting before inspecting candidate results:

```bash
tools/skymizer/scripts/05_compare.sh text \
  --unit article --metrics kld nll --primary-metric kld --primary-weighting token
```

Use `--unit window` for windows or `--unit block --block-windows 8` for fixed contiguous blocks. Article/block modes require every original window and stored target. They validate paired windows first, then merge full per-token records without retokenization, BOS insertion or additional scoring. Cross-article windows assign each target to its article; articles with no scored targets contribute no group. The final shorter block is retained. Grouping precedes t statistics/bootstrap, so group count determines degrees of freedom and resampling.

Article boundaries use the longest stable native-token prefix at the exact byte boundary. The first token affected by a cut belongs to the following article; affected prefixes are recorded. `item` weighting gives equal weight to each selected group; `token` weighting is the pooled scored-token estimand corresponding to corpus PPL.

Adjacent groups can remain correlated. A fixed corpus is not a random panel of independent questions, and choosing a block size does not prove independence. Residual dependence can invalidate CI coverage and p-values. Declare article/block sensitivity analyses and save distinct report paths when comparing them. The strict comparator requires completed collections, execution/runtime identity, matching corpus/window hashes, exact targets and shared reference columns. See [statistical definitions](compare.md).

Historical acceptance covered two windows per candidate/corpus comparison. It established exercised protocol and NLL parity, not full-corpus candidate quality, SNR saturation or nominal coverage for dependent articles. High raw-text reference PPL alone does not identify a broken quantized candidate; clipping in the saved uint16 reference can also change reconstructed PPL substantially.
