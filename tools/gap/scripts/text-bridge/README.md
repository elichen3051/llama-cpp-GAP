# Text bridge collection scripts: `llama-perplexity` + `llama-llm-kld`

These scripts are the exact commands and parameters that produced every collection under `runs/`
(18 checkpoint/corpus segments, 254 candidate runs), lifted out of the campaign dispatcher and made
reusable for any reference model, any candidate quantization and any text corpus.

Each candidate run is three GPU/CPU stages over the same frozen 512-token windows:

| stage | script | tool | writes |
| --- | --- | --- | --- |
| 1 | `prepare_corpus.sh` | `prepare_perplexity_corpus.py` (uses `llama-tokenize`, `llama-llm-kld --vocab-identity`) | `PREPARED/{corpus.txt,dataset/,manifest.json,stream.json}` |
| 2 | `reference_ppl.sh` | `llama-perplexity --save-all-logits` | `SEG/reference-ppl.{log,sha256,receipt.json}` + the base `.bin` |
| 3 | `candidate_ppl.sh` | `llama-perplexity --kl-divergence --kl-divergence-base` | `SEG/candidates/LABEL/ppl.log` |
| 4 | `llm_kld.sh` | `collect_llm_kld.py --perplexity-window` → `llama-llm-kld --manifest` | `SEG/candidates/LABEL/{llm.log,llm-kld/}` |
| 5 | `verify_bridge.sh` | `verify_perplexity_bridge.py` (CPU) | `SEG/candidates/LABEL/{bridge.log,bridge.json}` |

`run_candidate.sh` chains 3→4→5 for one candidate; `run_segment.sh` runs stage 2 once and then
`run_candidate.sh` for every candidate. Stage 1 runs once per (reference model, corpus) and its output
is reused by all candidates of that checkpoint.

## Requirements

1. **Native runtime**: `llama-perplexity`, `llama-tokenize`, `llama-llm-kld` and their shared libraries.
   No binaries ship with the release; build them from the fork below with
   `cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` and targets `llama-tokenize llama-perplexity llama-llm-kld`,
   then point `TEXT_BIN`/`TEXT_LIB` at `build/bin` (or `RUNTIME_DIR` at a directory with `bin/` + `lib/`).
   `runtime.json` records the two builds that produced `runs/`: source commit, CUDA, GPU and the SHA256 of
   every binary and library.
2. **Fork checkout** (`FORK_REPO`): <https://github.com/user/llama-cpp-GAP>, branch
   `text-bridge-production-scripts`. Its `tools/gap/cli/{prepare_perplexity_corpus,collect_llm_kld,
   verify_perplexity_bridge}.py` are byte-identical to the state used for `runs/` (upstream commit
   `ca3dc958533d768953ac92a310b0ae8c6039e51f` plus `llm-vocab-attribute-waiver.patch`, SHA256 `0b188745…585b19`,
   now committed in the fork). Auto-detected when these scripts run from `tools/gap/scripts/text-bridge`
   inside that checkout. Needed by stages 1, 4 and 5 only.
3. **Python** (`PYTHON`): the fork's locked environment, `uv sync --project tools/gap --python 3.12.3 --locked`
   (Python 3.12.3, datasets 5.0.1, numpy 2.5.2, pyarrow 25.0.1, huggingface-hub 1.30.0).
4. **Models**: reference GGUF (bf16; shard 1 for split files) and candidate GGUFs of the same checkpoint.
5. **Corpus**: raw UTF-8 text. Production used WikiText-2 test (`wiki.test.raw`, SHA256 `173c87a5…7dd08`)
   and the 217-article Project Gutenberg RSS corpus (`pg.txt`, SHA256 `26db5717…a082`) with an article index.
6. **Disk**: the reference base is `20 + W*512*4 + W*255*(2*ceil(V/2)+4)*2` bytes (W windows, V vocab),
   e.g. 73 GB for Qwen3.5-4B/WikiText-2 and 199 GB for Gemma-4/PG. Keep one base per segment and reuse it.

Only one scorer may use the GPU at a time; `llama-llm-kld` loads reference and candidate together.

## Quick start

```bash
export TEXT_BIN=/path/to/build/bin TEXT_LIB=/path/to/build/bin   # or RUNTIME_DIR=/path/with/bin+lib
export FORK_REPO=/path/to/llama-cpp-GAP        # not needed when running from inside the fork
export PYTHON=/path/to/venv/bin/python
cd scripts                                     # or tools/gap/scripts/text-bridge in the fork

# 1. freeze the corpus once per reference model
./prepare_corpus.sh wiki.test.raw wikitext-2-test ref-bf16.gguf work/prepared/qwen3.5-4b/wikitext-2-test \
    articles.json                                                     # article index optional for WikiText, required for PG

# 2..5 reference PPL, then every candidate (PPL, LLM-KLD, bridge check)
PPL_BASE=/fast-disk/qwen3.5-4b/wikitext-2-test/reference-ppl.bin \
./run_segment.sh ref-bf16.gguf work/prepared/qwen3.5-4b/wikitext-2-test runs/qwen3.5-4b/wikitext-2-test \
    candidate-019--bartowski--IQ4_NL=cand-IQ4_NL.gguf \
    candidate-020--bartowski--IQ4_XS=cand-IQ4_XS.gguf

# Gemma-family checkpoints: token attribute metadata differs between bf16 and quantized files
ALLOW_VOCAB_ATTR_MISMATCH=1 ./run_segment.sh gemma-bf16.gguf work/prepared/gemma-4-e4b-it/pg-full-rss \
    runs/gemma-4-e4b-it/pg-full-rss supplemental-google-02--google--Q4_0=gemma-4-E4B_q4_0-it.gguf
```

`DRY_RUN=1` prints every command line without running anything. `CHUNKS=2` turns every stage into the
two-window smoke used before production; leave it at `-1` for real collections.

## Exact commands

Reference PPL and base (stage 2):

```bash
llama-perplexity -m REF -f PREPARED/corpus.txt \
  -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks -1 \
  --save-all-logits PPL_BASE
```

Candidate PPL + KLD against the saved base (stage 3). Both `--kl-divergence` and `--kl-divergence-base` are
required; the base option alone aliases the writer and would overwrite the base.

```bash
llama-perplexity -m CAND \
  -c 512 -b 512 -ub 512 -t 8 -tb 8 -ngl all -fa on -ctk f16 -ctv f16 --fit off --no-escape --ppl-stride 0 --chunks -1 \
  --kl-divergence --kl-divergence-base PPL_BASE
```

Full-vocabulary LLM-KLD (stage 4). Gemma-family runs append `--allow-vocab-attr-mismatch`.

```bash
python tools/gap/cli/collect_llm_kld.py \
  --ref-model REF --cand-model CAND --dataset PREPARED/dataset --subset "" --out CAND_DIR/llm-kld \
  --llama-llm-kld BIN/llama-llm-kld --perplexity-window \
  --n-ctx 512 --n-batch 512 --n-ubatch 512 --tf-chunk -1 --n-threads 8 --metric-threads 8 --n-gpu-layers -2 --flash-attn \
  --start 0 --end -1 --dataset-limit -1 --num-eval-tokens -1
```

The collector writes one `tokens.bin` per window plus a JSONL manifest (`{"tokens_in", "n_prefill": 257,
"output_metrics"}` per row) and invokes the native scorer once:

```bash
llama-llm-kld --ref-model REF --cand-model CAND --manifest CAND_DIR/llm-kld/_manifest.jsonl --num-eval-tokens -1 \
  -b 512 -c 512 -ub 512 -ngl -2 --tf-chunk -1 -t 8 --metric-threads 8 --flash-attn [--allow-vocab-attr-mismatch] --perplexity-window
```

Bridge verification (stage 5, CPU):

```bash
sha256sum -c SEG/reference-ppl.sha256
CUDA_VISIBLE_DEVICES= python tools/gap/cli/verify_perplexity_bridge.py \
  --prepared PREPARED --llm-collection CAND_DIR/llm-kld --ppl-logits PPL_BASE \
  --ppl-reference-log SEG/reference-ppl.log --ppl-candidate-log CAND_DIR/ppl.log --out CAND_DIR/bridge.json
```

## Frozen protocol

| setting | `llama-perplexity` | `llama-llm-kld` |
| --- | --- | --- |
| context / logical batch / micro-batch | `-c 512 -b 512 -ub 512` | `--n-ctx 512 --n-batch 512 --n-ubatch 512` |
| sequences | derived `n_seq=1` | fixed `n_seq=1` |
| inference + batch threads | `-t 8 -tb 8` | `--n-threads 8` |
| metric threads | follows `-t` (logged `metric_threads = 8`) | `--metric-threads 8` |
| GPU layers | `-ngl all` | `--n-gpu-layers -2` |
| flash attention / KV | `-fa on -ctk f16 -ctv f16` | `--flash-attn`, F16 default |
| auto-fit / SWA full cache / MTP | `--fit off`, omitted, off | none, omitted, off |
| corpus handling | `--no-escape --ppl-stride 0 --chunks -1` | `--perplexity-window --tf-chunk -1 --start 0 --end -1 --dataset-limit -1 --num-eval-tokens -1` |

Decode contract shared by both tools: strip one final newline, tokenize with the reference model's native
tokenizer and its normal leading-BOS behaviour, cut non-overlapping 512-token windows (drop the tail),
replace token 0 with BOS when the vocabulary requires it, decode each window in one batch, score logits
256..510 against targets 257..511 (255 targets per window). `llama-perplexity` saves a clipped uint16
reference distribution; `llama-llm-kld` keeps both full-precision logit rows. Their KLD numbers are
therefore separate measurements; only the uncompressed NLL and the input protocol must agree, which is
what `verify_bridge.sh` checks (tolerance 1e-5 nats).

## Output layout (identical to `runs/`)

```
SEG/                                  runs/<checkpoint>/<corpus>/
  reference-ppl.log                   llama-perplexity output, "Final estimate: PPL = …"
  reference-ppl.sha256                sha256sum line of the base (base itself is large and lives on PPL_BASE)
  reference-ppl.receipt.json          bytes, sha256, windows, reference PPL, reference model
  candidates/LABEL/
    ppl.log                           "Mean PPL(Q)", "Mean PPL(base)", "Mean KLD" …
    llm.log                           collector log
    llm-kld/metrics/NNN_<window-id>.npz   per-window, per-target full-vocabulary metrics
    llm-kld/{manifest.csv,collect_meta.json,corpus_windows.json,logs/kld_run.log,.attempts/}
    bridge.log, bridge.json           cross-tool verification ("status": "passed")
```

Every script validates its output the way the production dispatcher did before marking a job done:
exactly one summary line per metric, window count equal to the prepared corpus, exact base size, base
unchanged after each candidate, `metrics/*.npz` count, reconciled collection state, `collect_meta.json`
flags, and `bridge.json` status. Outputs are never overwritten; retry into a new directory.

## Provenance

| build | llama.cpp build | used for | `llama-perplexity` | `llama-llm-kld` |
| --- | --- | --- | --- | --- |
| `runtime/` | b10835-4dc671b98, CUDA 13.2.51 | 250 runs, 2026-09-07..10 | `3176db81…7ee3` | `e0de3e19…5404` |
| `runtime-20260911/` | b10845-ca3dc9585, CUDA 13.2.51 | 4 google Q4_0 runs, 2026-09-11 | `97379fda…37bf` | `a60f9ff7…60bb` |

Both were built from source commit `ca3dc958` + the waiver patch and run on an RTX PRO 6000 Blackwell
Server Edition (driver 595.71.05). The rebuilt runtime reproduced the original bit for bit (reference PPL,
base SHA256, candidate PPL/KLD and every LLM-KLD npz); the parity report is
`campaign/runtime-parity-20260911.parity-report.json`. Only `runtime/` ships with the release; full file
lists with SHA256 for both builds are in `runtime.json`.

The maintained copy of this directory is `tools/gap/scripts/text-bridge/` in the fork; the copy in the
release package is the snapshot that documents `runs/`.

## Anonymization

The release was anonymized after collection with a fixed string mapping (company name → `company`,
maintainer handle → `user`, tool directory `tools/<company>` → `tools/gap`). Consequences:

- Protocol labels inside the data were renamed too (`company-perplexity-corpus-v1`,
  `company-collection-attempt-v1`, `company-execution-sha256-v2`). Stored digests that were computed over the
  original labels are kept as opaque identifiers: `protocol_sha256`, the window ids and `metrics/*.npz` names
  derived from it, `corpus_windows_sha256` and `dataset_content_hash`. They are internally consistent but do
  not recompute from the renamed protocol dict. Set `LEGACY_CORPUS_DIGEST=1` when running the fork's stats or
  bridge tools on these collections; every structural check (window order, ids, token/target digests, metric
  shapes) still applies. New collections made with the renamed tools need no flag.
- Checksum receipts computed before anonymization over text files that contain renamed strings no longer
  match those files (`campaign/environment*.SHA256SUMS`, `campaign/runner-scripts.*.sha256`,
  `campaign/overlay-patch.sha256`, `campaign/bundle/SHA256SUMS`). Receipts over binary artifacts
  (`runs/**/metrics/*.npz`, `backup/*/manifest.json` sizes and MD5s, model SHA256s) are unaffected.
- No native binaries ship; `runtime.json` keeps their identities.
