# AWS text bridge: llama-perplexity and llm-kld

Updated 2026-09-06. The implementation, full CPU regression suite and sparse GPU parity checks are complete. Production corpus collection remains an operator task; the acceptance scope below is two windows per candidate/corpus comparison.

This lane uses the same frozen corpus, native GGUF vocabulary, 512-token windows, BOS policy and 255 targets in both tools. llm-kld computes full-vocabulary metrics while both uncompressed logit rows are in memory. llama-perplexity saves clipped, uint16-encoded reference log probabilities. Always pass both `--kl-divergence` and `--kl-divergence-base FILE` for candidate PPL: the filename option alone aliases `--save-all-logits` and can overwrite the reference file. Check completed PPL/KLD summaries, not only the exit code. Their KLD results are separate measurements; agreement is required for their uncompressed NLL and input protocol, not for the two KLD definitions.

A family means quantizations of one base checkpoint. Kimi Instruct and Thinking-2506 are separate checkpoints. Use only architectures supported by this checkout's upstream base, `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`; this work changes evaluation tools, not model implementations. The candidate roster is subject to the separate integrity and numerical audit. The [reference handover](reference-runpod-handover.md) defines the SNR and final-evaluation groups; finish the SNR group before the final group. VLM KLD uses its own capacity-driven context and batch settings.

## Frozen inputs

All paths below are under `/opt/dlami/nvme/skymizer-text-bridge-20260906/corpora`. Preserve these bytes when moving to another AWS host.

| Corpus | File | Bytes | SHA256 |
| --- | --- | ---: | --- |
| WikiText-2 test | `wikitext-2/wikitext-2-raw/wiki.test.raw` | 1290590 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| Full PG RSS | `pg-normalized-html5lib1.1-html2text2.4.0/pg.txt` | 3179044 | `26db5717f58a11a8ed9c24dab9acffc557bcb9d7697b733f44039f13cca4e082` |

WikiText comes from `ggml-org/ci@927b3642933080f1b0e811e2f916e14c292992f9`, the source used by `scripts/get-wikitext-2.sh`. Its top-level headers identify 62 articles. The whitespace-only prelude belongs to the first article; nested section headers do not start articles.

PG contains all 217 URLs in the frozen RSS order used by `scripts/get-pg.sh`. The earlier estimate of 127 was not a cap. The original system html2text failed on valid Founder Visa HTML; the frozen conversion uniformly uses html5lib 1.1, native html2text 2.4.0, then the script's original `tail -n +4`, `sed`, and `fmt -w 80` operations in `C.UTF-8`. All 217 articles and byte spans were verified. No body loss was observed in the audit of removed first-three-line prefixes; titles are retained separately as RSS metadata and are not guaranteed in the corpus text. Do not use the historical, partial `pg/pg.txt` containing 107 articles.

Transfer `corpus-inputs.json`, `corpus-freeze-receipt.json`, `SHA256SUMS`, and the PG normalized directory's `manifest.json` with the two files. The complete acquisition directory preserves raw HTML, conversion steps, pinned tools and every intermediate SHA. Re-running a live RSS or a different converter does not reproduce this frozen corpus.

## Exact scoring protocol

1. Match common's file handling by removing one final newline, when present. Do not interpret backslash escapes or parse literal special-token spellings.
2. Tokenize the full effective text using the reference GGUF's native tokenizer, with its normal leading-BOS behavior. The preparer compares `llama-tokenize` with and without `--no-bos` to verify this behavior.
3. Split the resulting stream into non-overlapping 512-token windows. Keep all complete windows; drop the incomplete tail. The classic tool requires at least 1024 input tokens.
4. If the vocabulary requires BOS, replace token 0 of each window with BOS. Do not insert an additional token or reset tokenization at article boundaries.
5. Clear KV data before every window. Decode all 512 tokens in one batch, with positions 0..511 and sequence 0. Request logits at positions 256..511, including the final unused output.
6. Score logits at positions 256..510 against targets 257..511. There are exactly 255 targets. In the prepared row, `n_prefill_tokens=257`; its legacy `generated_tokens_len=255` field means teacher-forced corpus targets, not generated answers.

The opt-in `--perplexity-window` mode implements this decode shape in llama-llm-kld. Its `n_past_actual` header is 512 because the entire window has already been decoded; `n_prefill` remains 257, the first target index. The ordinary prompt/answer teacher-forcing path is unchanged. Both vocabularies must have `add_eos=false`, as required by classic PPL.

| Setting | llama-perplexity | llm-kld |
| --- | --- | --- |
| Context, logical batch, microbatch | `-c 512 -b 512 -ub 512` | `--n-ctx 512 --n-batch 512 --n-ubatch 512` |
| Sequence count | Derived `n_seq=1` | Fixed 1 |
| Inference and batch threads | `-t 8 -tb 8` | `--n-threads 8` controls both |
| Metric threads | 8, follows `-t` and is logged | `--metric-threads 8` |
| GPU layers | `-ngl all` | `--n-gpu-layers -2` |
| Flash attention | `-fa on` | `--flash-attn` |
| KV types | `-ctk f16 -ctv f16` | F16 defaults |
| Automatic fit | `--fit off` | No auto-fit path |
| SWA full cache | Omit `--swa-full` | Omit `--swa-full` |
| MTP | Off | Off |

SWA full-cache allocation being off does not disable architectural sliding-window attention. Use one scorer at a time on the RTX PRO 6000 Blackwell Server Edition, 96 GiB. llm-kld loads reference and candidate together; a successful single-model PPL run does not prove dual-model capacity. Freeze binary, backend libraries, GPU/driver, relevant environment, corpus and every runtime setting across candidates of the same base.

## Build and prepare

Run from the repository root. Put builds, environments, caches, datasets, logits, metrics and logs on NVMe.

```bash
set -euo pipefail
export TEXT_WORK=/opt/dlami/nvme/text-bridge
export TMPDIR="$TEXT_WORK/tmp"
export HF_HOME="$TEXT_WORK/hf"
export UV_CACHE_DIR="$TEXT_WORK/uv-cache"
export UV_PROJECT_ENVIRONMENT="$TEXT_WORK/venv"
mkdir -p "$TMPDIR"
cmake -S . -B "$TEXT_WORK/build" -DGGML_CUDA=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build "$TEXT_WORK/build" --target llama-tokenize llama-perplexity llama-llm-kld -j8
uv sync --project tools/skymizer --frozen
export TEXT_PYTHON="$TEXT_WORK/venv/bin/python"
export TEXT_BIN="$TEXT_WORK/build/bin"
```

Set `REF`, `CAND_A` and `CAND_B` to exact audited GGUF paths from the same base checkpoint. For a split model, use shard 1. Set `CORPUS`, `CORPUS_NAME` and `PREPARED` for one corpus; keep WikiText and PG in separate collections and reports. Existing prepared datasets are under `/opt/dlami/nvme/skymizer-text-bridge-20260906/prepared/<checkpoint>/<corpus>/dataset`.

To prepare WikiText:

```bash
"$TEXT_PYTHON" tools/skymizer/cli/prepare_perplexity_corpus.py \
  --corpus "$CORPUS" --corpus-name wikitext-2-test \
  --ref-model "$REF" --llama-tokenize "$TEXT_BIN/llama-tokenize" \
  --llama-llm-kld "$TEXT_BIN/llama-llm-kld" --out "$PREPARED"
```

For PG, use `--corpus-name pg-full-rss` and add `--article-index` pointing to the frozen normalized PG `manifest.json`. Prepare once per checkpoint and corpus, then reuse the same dataset for all its candidates. The output must be new. `manifest.json` records corpus and vocabulary hashes, the complete window map, article boundaries and dropped tail; `stream.json` preserves the pre-replacement native token stream. `corpus.txt` is the original file for PPL, while `effective.txt` documents the one-newline handling. Do not pass `effective.txt` to PPL, which applies its own file handling.

## Collect all windows

Reference PPL and its reusable low-precision base:

```bash
test ! -e "$TEXT_WORK/reference-ppl.bin"
"$TEXT_BIN/llama-perplexity" -m "$REF" -f "$PREPARED/corpus.txt" \
  -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on \
  -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks -1 \
  --save-all-logits "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/reference-ppl.log" 2>&1
chmod a-w "$TEXT_WORK/reference-ppl.bin"
sha256sum "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/reference-ppl.sha256"
```

For each candidate, use a new output directory and log path:

```bash
"$TEXT_BIN/llama-perplexity" -m "$CAND_A" \
  -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on \
  -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks -1 \
  --kl-divergence --kl-divergence-base "$TEXT_WORK/reference-ppl.bin" > "$TEXT_WORK/candidate-a-ppl.log" 2>&1
sha256sum -c "$TEXT_WORK/reference-ppl.sha256"

"$TEXT_PYTHON" tools/skymizer/cli/collect_llm_kld.py \
  --ref-model "$REF" --cand-model "$CAND_A" \
  --dataset "$PREPARED/dataset" --subset "" --out "$TEXT_WORK/llm-a" \
  --llama-llm-kld "$TEXT_BIN/llama-llm-kld" --perplexity-window \
  --n-ctx 512 --n-batch 512 --n-ubatch 512 --tf-chunk -1 \
  --n-threads 8 --metric-threads 8 --n-gpu-layers -2 --flash-attn

"$TEXT_PYTHON" tools/skymizer/cli/verify_perplexity_bridge.py \
  --prepared "$PREPARED" --llm-collection "$TEXT_WORK/llm-a" \
  --ppl-logits "$TEXT_WORK/reference-ppl.bin" \
  --ppl-reference-log "$TEXT_WORK/reference-ppl.log" \
  --ppl-candidate-log "$TEXT_WORK/candidate-a-ppl.log" --out "$TEXT_WORK/bridge-a.json"
```

Repeat with candidate B and distinct `llm-b`, log and verification paths. Keep the same reference base. The verifier checks exact saved tokens and target IDs, full file size, window count/shape, explicit thread alignment, and uncompressed mean NLL within 1e-5 nats plus printed-PPL rounding. It reports quantized-reference NLL error and both KLD means separately. It does not treat the two KLD definitions as identical.

A two-window smoke uses `--chunks 2` for both PPL commands and `--dataset-limit 2` for llm-kld. Restore full-corpus settings for production. Do not use article or block aggregation on that partial smoke collection.

The PPL base can be large: bytes = `20 + windows*512*4 + windows*255*(2*ceil(vocab/2)+4)*2`. This is roughly 127.5 MiB per window for a 262144-token vocabulary. llm-kld keeps only 76 bytes per scored target plus headers and provenance. Use a new work directory per checkpoint/corpus and avoid overwriting an existing PPL base or log.

## Paired statistics: windows, articles and contiguous blocks

Use the existing comparator with `--unit window`, `--unit article`, or `--unit block --block-windows 8`:

```bash
"$TEXT_PYTHON" tools/skymizer/cli/saved_metrics_paired_compare.py \
  --candidate-a "$TEXT_WORK/llm-a" --candidate-b "$TEXT_WORK/llm-b" \
  --unit article --metrics kld nll --primary-metric kld \
  --primary-weighting token --out "$TEXT_WORK/articles.md" \
  --output-json "$TEXT_WORK/articles.json"
```

Article and block modes require the complete corpus window set and all stored targets. They validate the original paired windows before merging full per-token records. They do not retokenize, insert BOS, shift windows or score additional targets. Every group mean is recomputed from its targets. Each article with at least one scored target is one group; articles that fall entirely in unscored context or the dropped tail contribute no group; a cross-article window contributes its targets to the respective articles. Blocks group consecutive original windows; the final shorter block is retained. Grouping happens before t statistics or bootstrap, so degrees of freedom and resampling use the number of groups.

Article cuts use the longest stable native-token prefix at the exact byte boundary. The first token affected by the cut belongs to the following article. The manifest records every affected prefix token; this makes boundary-spanning tokens explicit. In the initial small-model checks, all WikiText cuts were exact, while 29 PG boundaries each affected one token.

`item` weighting means equal weight per selected window/article/block. `token` weighting gives the pooled scored-token estimand and is the direct corpus-PPL aggregation. Windows and adjacent groups can remain correlated. These fixed corpora are not an independently sampled item panel; CI and test calculations treat the selected groups as independent sampling units. Residual dependence can invalidate coverage and p-values; fixed block size does not prove independence. Choose the primary metric, weighting, grouping and block size before inspecting candidate results. Compare article and block analyses as declared sensitivity analyses, with separate reports.

The main comparator checks complete collection state, exact execution identity, the full runtime, corpus/window hashes, embedded targets and reference columns. Invalid/nonfinite data and fewer than two groups fail explicitly. `--allow-ref-drift` is an explicit approximate-pairing option, not part of the strict protocol above. The legacy power and variance loaders now use the same collection guards; they must not silently drop failed or nonfinite rows. See [statistical definitions](compare.md) for formulas and practical-equivalence semantics.


## Acceptance scope

All nine checkpoints were checked on the RTX PRO 6000: Gemma 31B, Gemma E4B, Kimi Instruct, Kimi Thinking-2506, Qwen3.5-4B, Qwen3.6-35B-A3B, InternVL3.5-30B-A3B, GLM-4.6V-Flash and Muse Glimmer. All 18 checkpoint/corpus datasets were prepared and checked with native tokenization. The GPU checks covered 22 candidate/corpus comparisons and 11,220 paired targets; the two small checkpoints each used two candidates, while the other seven used their largest candidate by file size. The same windows reused across candidates are not independent observations.

All exact window/target checks and uncompressed NLL parity checks passed. The largest candidate mean-NLL difference was below 8e-8 nats; reference differences were below 7e-6 nats, including PPL log rounding. Both models fit simultaneously at the specified text runtime. The existing CPU suite passed 823 tests, with two external-model tests excluded. Statistical review included formulas, failure handling, paired data identity, numerical goldens and article/block aggregation. Two-window GPU smokes do not establish corpus-wide candidate quality, SNR saturation or CI coverage for correlated articles.

The acceptance also exposed a material precision difference for Gemma 31B on the first two WikiText windows:

| Measurement | PPL |
| --- | ---: |
| Original BF16 reference | 9357.5633 |
| Q4_1 candidate | 8301.195369 |
| uint16 saved reference base | 3195.379909 |

The low-precision base had 120 of 510 target entries at its quantization floor and a mean target-NLL error of -1.07448 nats. A floor entry alone does not prove how far its original value was clipped. The verifier reports this separately from the original reference likelihood. Both uncompressed tools agree even when baseline raw-text PPL is high; this observation alone does not identify a broken quantized checkpoint. PG showed the same distinction at smaller magnitude. Keep original reference PPL, reconstructed saved-base PPL and candidate PPL as separate fields in paper tables.

Local evidence is under `/opt/dlami/nvme/skymizer-text-bridge-20260906`: `gpu-acceptance/combined-acceptance-summary.json`, the r2 and extension reports, `independent-review/`, and `tests/final-cpu-suite.log`. The failed initial command and its overwritten 12-byte base are retained separately; only the corrected runs with both candidate KL flags contribute to acceptance.
