# Text bridge collection scripts: `llama-perplexity` + `llama-llm-kld`

These scripts are the exact commands and parameters that produced every collection under `runs/`
(18 checkpoint/corpus segments, 254 candidate runs), lifted out of the campaign dispatcher and made
reusable for any reference model, any candidate quantization and any text corpus.

Each candidate run is three GPU/CPU stages over the same frozen 512-token windows. Outputs go to two record
trees: `PPL_SEG` = `runs/llama-perplexity-records/<checkpoint>/<corpus>` for everything `llama-perplexity`
produced, `KLD_SEG` = `runs/our-llm-kld-records/<checkpoint>/<corpus>` for the LLM-KLD collections and the
cross-tool bridge check.

| stage | script | tool | writes |
| --- | --- | --- | --- |
| 0 | `get_corpora.sh` | `verify`: SHA256 of the shipped corpora; download modes need `curl`, `unzip`, for PG also `html5lib`, C++ `html2text`, GNU `fmt` | `corpora/wiki.test.raw`, `corpora/pg.txt` |
| 1 | `prepare_corpus.sh` | `prepare_perplexity_corpus.py` (uses `llama-tokenize`, `llama-llm-kld --vocab-identity`) | `PREPARED/{corpus.txt,dataset/,manifest.json,stream.json}` |
| 2 | `reference_ppl.sh` | `llama-perplexity --save-all-logits` | `PPL_SEG/reference-ppl.{log,sha256,receipt.json}` + the base `.bin` |
| 3 | `candidate_ppl.sh` | `llama-perplexity --kl-divergence --kl-divergence-base` | `PPL_SEG/candidates/LABEL/ppl.log` |
| 4 | `llm_kld.sh` | `collect_llm_kld.py --perplexity-window` → `llama-llm-kld --manifest` | `KLD_SEG/candidates/LABEL/{llm.log,llm-kld/}` |
| 5 | `verify_bridge.sh` | `verify_perplexity_bridge.py` (CPU) | `KLD_SEG/candidates/LABEL/{bridge.log,bridge.json}` |

`run_candidate.sh` chains 3→4→5 for one candidate; `run_segment.sh` runs stage 2 once and then
`run_candidate.sh` for every candidate. Stage 1 runs once per (reference model, corpus) and its output
is reused by all candidates of that checkpoint.

## The artifact archive

The collections are delivered as a separate, artifacts-only archive (no code):

```
runs/llama-perplexity-records/   reference PPL + saved-base receipts and candidate PPL logs (18 segments)
runs/our-llm-kld-records/        the 254 LLM-KLD collections and bridge checks      see "Output layout" below
corpora/         the frozen corpora: wiki.test.raw + wikitext-2.articles.json (WikiText-2 test, 60-article index),
                 pg.txt + pg.manifest.json (217 Paul Graham essays, byte-span manifest)
campaign/        models.json (reference model file and candidate labels per checkpoint), environment/ (frozen
                 execution environment of the build host), README.md (protocol notes), VLM_REVIEW.md
runtime.json     identity of the two native runtimes behind runs/ (copy of the file in this directory)
README.md
```

## Requirements

1. **Native runtime**: `llama-perplexity`, `llama-tokenize`, `llama-llm-kld` and their shared libraries.
   No binaries ship with the release; build them from the fork below with
   `cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` and targets `llama-tokenize llama-perplexity llama-llm-kld`,
   then point `TEXT_BIN`/`TEXT_LIB` at `build/bin` (or `RUNTIME_DIR` at a directory with `bin/` + `lib/`).
   `runtime.json` records the two builds that produced `runs/`: source commit, CUDA, GPU and the SHA256 of
   every binary and library.
2. **Fork source tree** (`FORK_REPO`): this llama.cpp fork (the archive itself contains no code).
   Its `tools/gap/cli/{prepare_perplexity_corpus,collect_llm_kld,verify_perplexity_bridge}.py` are the tools
   that produced `runs/` (the fork's pre-anonymization commit `ca3dc958`, whose collector already carries the
   Gemma `--allow-vocab-attr-mismatch` waiver), with the anonymization string mapping applied and, in
   `collect_llm_kld.py`, one reworded help string (codename). Shipped SHA256:
   `prepare_perplexity_corpus.py` `93e2843d…e2e7`, `collect_llm_kld.py` `485c43b9…f443`,
   `verify_perplexity_bridge.py` `5c84bf2c…7b6c`. Auto-detected when these scripts run from
   `tools/gap/scripts/text-bridge` inside that tree. Needed by stages 1, 4 and 5 only.
3. **Python** (`PYTHON`): the fork's locked environment, `uv sync --project tools/gap --python 3.12.3 --locked`
   (Python 3.12.3, datasets 5.0.1, numpy 2.5.2, pyarrow 25.0.1, huggingface-hub 1.30.0).
4. **Models**: reference GGUF (bf16; shard 1 for split files) and candidate GGUFs of the same checkpoint.
5. **Corpus**: raw UTF-8 text. Production used WikiText-2 test (`corpora/wiki.test.raw`, SHA256 `173c87a5…7dd08`)
   and 217 Paul Graham essays from the `pgessays` RSS feed (`corpora/pg.txt`, SHA256 `26db5717…a082`), both
   shipped; `get_corpora.sh verify` checks them. See "Corpora" below.
6. **Disk**: the reference base is `20 + W*512*4 + W*255*(2*ceil(V/2)+4)*2` bytes (W windows, V vocab),
   e.g. 73 GB for Qwen3.5-4B/WikiText-2 and 199 GB for Gemma-4/PG. Keep one base per segment and reuse it.

Only one scorer may use the GPU at a time; `llama-llm-kld` loads reference and candidate together.

## Quick start

```bash
export TEXT_BIN=/path/to/build/bin TEXT_LIB=/path/to/build/bin   # or RUNTIME_DIR=/path/with/bin+lib
export FORK_REPO=/path/to/llama-cpp-GAP        # not needed when running from inside the fork
export PYTHON=/path/to/venv/bin/python
cd tools/gap/scripts/text-bridge

# 0. corpora: check the archive's files against the frozen SHA256s
./get_corpora.sh --out /path/to/archive/corpora verify

# 1. freeze the corpus once per reference model
./prepare_corpus.sh corpora/wiki.test.raw wikitext-2-test ref-bf16.gguf work/prepared/qwen3.5-4b/wikitext-2-test \
    corpora/wikitext-2.articles.json
./prepare_corpus.sh corpora/pg.txt pg-full-rss ref-bf16.gguf work/prepared/qwen3.5-4b/pg-full-rss corpora/pg.manifest.json

# 2..5 reference PPL, then every candidate (PPL, LLM-KLD, bridge check)
PPL_BASE=/fast-disk/qwen3.5-4b/wikitext-2-test/reference-ppl.bin \
./run_segment.sh ref-bf16.gguf work/prepared/qwen3.5-4b/wikitext-2-test runs qwen3.5-4b wikitext-2-test \
    candidate--bartowski--IQ4_NL=cand-IQ4_NL.gguf \
    candidate--bartowski--IQ4_XS=cand-IQ4_XS.gguf

# Gemma-family checkpoints: token attribute metadata differs between bf16 and quantized files
ALLOW_VOCAB_ATTR_MISMATCH=1 ./run_segment.sh gemma-bf16.gguf work/prepared/gemma-4-e4b-it/pg-full-rss \
    runs gemma-4-e4b-it pg-full-rss candidate--google--QAT-Q4_0=gemma-4-E4B_q4_0-it.gguf
```

`DRY_RUN=1` prints every command line without running anything. `CHUNKS=2` turns every stage into the
two-window smoke used before production; leave it at `-1` for real collections.

## Corpora

```bash
./get_corpora.sh --out /path/to/archive/corpora verify   # SHA256 + size of the four shipped corpus files
./get_corpora.sh --out dl wikitext             # re-download WikiText-2 from the pinned ggml-org/ci snapshot and verify
./get_corpora.sh --out dl pg                   # best-effort reconstruction of pg.txt (Linux: C++ html2text 2.4.0, GNU fmt, html5lib 1.1)
```

| file | source | frozen identity (SHA256, bytes) |
| --- | --- | --- |
| `corpora/wiki.test.raw` (shipped) | `wikitext-2-raw-v1.zip` from the `ggml-org/ci` HF dataset, commit `927b3642` (same archive as llama.cpp `scripts/get-wikitext-2.sh`; zip SHA256 `ef7edb56…5a11`); re-downloaded and verified identical on 2026-09-12 | `173c87a5…7dd08`, 1,290,590 |
| `corpora/pg.txt` (shipped) | the 217 essays of the `pgessays` RSS feed, llama.cpp `scripts/get-pg.sh` with n=217 plus html5lib 1.1 normalization before C++ `html2text` 2.4.0, then `tail -n +4 \| sed -E 's/^[[:space:]]+//g' \| fmt -w 80` under `LC_ALL=C.UTF-8` (2026-09-06) | `26db5717…a082`, 3,179,044 |
| `corpora/wikitext-2.articles.json` (shipped) | explicit 60-article index of the WikiText-2 test split (byte spans, titles, per-article SHA256); pass it as `ARTICLE_INDEX` | `63216bfa…0bf6b72`, 10,095 |
| `corpora/pg.manifest.json` (shipped) | byte-span manifest of the 217 essays; required `ARTICLE_INDEX` for `pg-full-rss` | `0abeb0d4…a198f`, 373,445 |

With these four files `prepare_corpus.sh` rebuilds both preparations exactly as used for `runs/` (same tokens,
windows, targets and article attribution; `corpus_windows.json` in every collection records the protocol). The
`pg` download mode is a best-effort reconstruction for anyone without the frozen files: the feed is live and GNU
vs BSD `fmt` differ; the script exits non-zero on a SHA256 mismatch.

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
runs/llama-perplexity-records/<checkpoint>/<corpus>/        PPL_SEG
  reference-ppl.log                   llama-perplexity output, "Final estimate: PPL = …"
  reference-ppl.sha256                sha256sum line of the base (the base itself is large and lives on PPL_BASE)
  reference-ppl.receipt.json          bytes, sha256, windows, reference PPL, reference model
  reference-ppl.*superseded-*, reference-ppl.regeneration-*.json   (4 gemma segments whose base was regenerated)
  candidates/LABEL/ppl.log            "Mean PPL(Q)", "Mean PPL(base)", "Mean KLD" …

runs/our-llm-kld-records/<checkpoint>/<corpus>/             KLD_SEG
  candidates/LABEL/
    llm.log                           collector log
    llm-kld/metrics/NNN_<window-id>.npz   per-window, per-target full-vocabulary metrics
    llm-kld/{manifest.csv,collect_meta.json,corpus_windows.json,kld_run.log}
    bridge.log, bridge.json           cross-tool verification ("status": "passed")
    llm-kld.failed-1-<ts>/, llm.failed-1-<ts>.log   (3 kept failed first attempts)
```

`LABEL` is `candidate--<provider>--<quant>`, the quantization spelled as in the GGUF file name (mradermacher
imatrix files carry their `i1-` marker, e.g. `candidate--mradermacher--i1-IQ4_XS` next to
`candidate--mradermacher--IQ4_XS`). `campaign/models.json` lists the reference model file and the candidate labels
of each checkpoint; the exact candidate file of every collection is recorded in its `collect_meta.json`.

252 of the 254 candidates have a passing `bridge.json`. The two `candidate--unsloth--UD-Q3_K_XL` runs of
`gemma-4-31b-it` (both corpora) have `bridge.log` only: their collections are complete, but the candidate's
degenerate logits make the two tools' mean NLL differ by more than the 1e-5 nats tolerance, so the check is
recorded as failed; both collections are complete and usable.

Every script validates its output the way the production dispatcher did before marking a job done:
exactly one summary line per metric, window count equal to the prepared corpus, exact base size, base
unchanged after each candidate, `metrics/*.npz` count, reconciled collection state, `collect_meta.json`
flags, and `bridge.json` status. Outputs are never overwritten; retry into a new directory.

## Provenance

| build | llama.cpp build | used for | `llama-perplexity` | `llama-llm-kld` |
| --- | --- | --- | --- | --- |
| `runtime/` | b10835-4dc671b98, CUDA 13.2.51 | 250 runs, 2026-09-07..10 | `3176db81…7ee3` | `e0de3e19…5404` |
| `runtime-20260911/` | b10845-ca3dc9585, CUDA 13.2.51 | 4 google Q4_0 runs, 2026-09-11 | `97379fda…37bf` | `a60f9ff7…60bb` |

Both were built from the fork's commit `ca3dc958` (waiver included) and run on an RTX PRO 6000 Blackwell
Server Edition (driver 595.71.05). The rebuilt runtime reproduced the original bit for bit (reference PPL,
base SHA256, candidate PPL/KLD and every LLM-KLD npz); the verdict is recorded in `runtime.json`. No binaries
ship with the release; full file lists with SHA256 for both builds are in `runtime.json`, and
`campaign/environment/{runtime-files,resolved-libraries}.sha256` are the build host's own receipts for them.

The archive ships artifacts only; these scripts live here in the fork.

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
- The collectors' `.attempts/` directories (declared rows and terminal statuses, i.e. scheduling state) are not
  shipped; each of the 254 collections was checked before removal (state `completed`, all rows `OK`, statuses equal
  to `manifest.csv`, npz count equal to the `OK` rows). The three kept failed first attempts
  (`llm-kld.failed-1-*`) retain theirs. Set `LEGACY_ATTEMPT_RECORDS=1` as well when loading these collections
  with the fork's tools; completeness is then verified from `manifest.csv` and `metrics/` alone.
- The checksum receipts shipped under `campaign/environment/` cover binaries and still verify against the original
  files.
- Beyond the string mapping, S3 bucket names read `<bucket>`.
  The campaign's operational records (scheduler state, worker logs, S3 backup receipts, the dispatcher source, the
  full candidate roster with GGUF SHA256s) are not part of this archive.
- Candidate directory names were normalized to `candidate--<provider>--<quant>`: the campaign's running
  numbers and its `supplemental-*` prefixes were dropped, and the 16 mradermacher imatrix files whose campaign
  label omitted the `i1-` marker now carry it. The same rename was applied inside the logs and in
  `campaign/models.json`.
- No native binaries ship; `runtime.json` keeps their identities.
