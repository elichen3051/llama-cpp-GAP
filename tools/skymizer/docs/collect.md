# Collecting per-token metrics

`collect_kld.py` prepares VLM rows and runs `llama-vlm-kld`; `collect_llm_kld.py` prepares text rows and runs `llama-llm-kld`. Each invocation evaluates one reference/candidate pair. The shared collector owns locking, output identity, row preparation, one native manifest process, validation and completion records. Each lane keeps its own input and prefill contract.

Use the [AWS VLM handover](kld-aws-handover.md) for model/runtime choices and the [text handover](perplexity-llm-kld-aws-handover.md) for classic PPL windows. This page describes the common mechanics.

## Inputs

Prefer a frozen local Hugging Face `save_to_disk` directory, passed as `--dataset PATH --subset ''`. For a Hub dataset, preserve its exact revision separately and materialize a local snapshot before a formal run; a moving Hub name is not a frozen input. Reference and candidate must share the same token-ID mapping. Model file fingerprints and dataset content hashes are part of collection identity.

| Row type | Preparation |
| --- | --- |
| Native reference trajectory | Uses the saved GGUF prompt, tokens, original ordered image bytes and native producer identity. No HF tokenizer is needed. |
| Legacy HF/vLLM trajectory | Uses the declared model's tokenizer and supported image-wrapper collapse rules. Install the optional `hf-tokenizer` extra. It does not prove equality with the original HF image processor. |
| Prepared text corpus window | Uses the frozen native full-stream tokenization and per-window metadata from `prepare_perplexity_corpus.py`. Requires `--perplexity-window` and its strict runtime. |

`prep_vlm_score_from_hf.py` and `prep_llm_score_from_hf.py` can inspect individual rows with `--row N` or `--item-id ID`. Collector row ranges use the selected sort order. Preparation checks lengths, label masking, answer targets and image markers. Native vocabulary provenance must match the scorer; unsupported legacy wrapper reconstruction does not imply that native upstream model execution is unsupported.

VLM preparation writes `tokens.bin`, `formatted_chat.txt`, ordered image files and `meta.json`. The prompt contains one `<__media__>` marker per source image. An image may expand into several tiles and chunks. The native scorer consumes these files, not the dataset directly:

| Input | Meaning |
| --- | --- |
| `tokens.bin` | Frozen int32 token IDs; `tokens[n_prefill:]` supplies targets. |
| `formatted_chat.txt` | Saved prompt with media markers for mtmd tokenization. |
| Image paths | Original image order, embedded by each side's projector. |
| `n_prefill` | Index of the first scored target in the frozen token stream. |
| Vocabulary identity | Native token-ID map checked before weight loading. |

Ordinary VLM prefill uses mtmd and records `n_past_actual`, its native position count. It need not equal the frozen sequential `n_prefill`, especially for vision positions. Both scorer sides must have matching native prefill positions unless an explicit diagnostic override is used. Text corpus full-window mode instead records `n_past_actual=512` and `n_prefill=257`; see its separate protocol.

## Runtime identity

The paired comparison requires matching context, logical batch, microbatch, teacher-forcing chunk, inference/metric threads, offload, flash attention, SWA cache policy, image bounds, scored horizon and executed binary/backend identity. A matching checkout commit alone is insufficient. Both models, and both projectors in VLM, are resident together with their KV and work buffers.

The supported scorer sequence count is one. `--n-threads` sets inference and batch threads; `--metric-threads` controls metric workers. With image-token flags omitted, the collector first adopts a native dataset's recorded bounds. If those bounds are default, it omits the native flags and uses the selected mmproj defaults. Legacy rows without a native policy also use mmproj defaults; their HF processor metadata is diagnostic. `--swa-full` controls full-cache allocation, not whether architectural sliding-window attention exists. No scorer auto-fit policy reduces the requested runtime.

`--tf-chunk` changes ordinary teacher-forcing batch shape. Changing it can change reference logits through floating-point reduction order. The effective final batch also depends on target count. Freeze the largest intended scoring horizon once and derive shorter prefixes from saved metrics; separately rescoring each cap can introduce boundary differences.

The native tools check vocabulary size/type and every token text and attribute. `--allow-vocab-attr-mismatch` relaxes only attribute equality and records a warning; it never permits different token texts. Use an override only for an examined conversion difference, and keep that choice identical across the family.

## Selection and commands

The direct VLM form is:

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/collect_kld.py \
  --dataset "$REFERENCE_DATASET" --subset '' \
  --ref-model "$REF" --ref-mmproj "$MMPROJ" \
  --cand-model "$CANDIDATE" --cand-mmproj "$MMPROJ" \
  --llama-vlm-kld "$SKYMIZER_WORK/build/bin/llama-vlm-kld" \
  --n-ctx 32768 --n-batch 2048 --n-ubatch 512 --tf-chunk 2048 \
  --n-threads 8 --metric-threads 8 --n-gpu-layers -2 --flash-attn \
  --num-eval-tokens "$SCORING_CAP" --out "$SKYMIZER_WORK/collection-a"
```

These capacity values are an example, not a universal model setting. The handover specifies Gemma 31B's larger microbatch and the measured candidate capacity. For ordinary text trajectories, use `collect_llm_kld.py`, omit projectors and select `--llama-llm-kld`; classic PPL requires the text handover's full-window command instead.

| Option | Selection rule |
| --- | --- |
| `--start`, `--end` | Half-open row range after sorting; end -1 means all available rows. |
| `--dataset-limit` | Caps the number of rows from `--start`, without extending `--end`. |
| `--num-eval-tokens` | Caps scored answer positions per row; -1 uses all available targets. |
| `--max-total-tokens` | Trajectory budget filter that records `SKIP_OVER_BUDGET`; it does not replace an answer scoring cap. |
| `--keep-prep` | Retains preparation artifacts for inspection. |

Short answers keep their actual available targets; collectors do not pad or generate replacements. Full-corpus article/block analysis disallows partial window collections, token caps and reordered rows. CLI `--help` is the source for current defaults and diagnostic flags.

## Completion and artifacts

Each output contains `collect_meta.json`, `manifest.csv`, `metrics/*.npz`, native logs and per-attempt completion records. VLMK v5 stores 76 bytes per target plus a 24-byte header; NPZ conversion preserves dtype, shape, header and values. The exact layout and older reader versions are in [formats.md](formats.md).

A collector holds an exclusive output lock. It refuses overlapping row output, rejected dumps, stale artifacts and unfinished attempts. A completed collection may receive a disjoint append with the same identity; increasing `--end` without moving `--start` repeats existing rows and is refused. Use a new output directory after an interrupted or failed attempt and retain the original evidence.

Native completion is validated before conversion: header version/vocabulary/counts, lane-specific prefill, exact target IDs and finite metric fields. A bad dump is renamed to `.bin.rejected` and cannot be read as valid metrics. Conversion, manifest recording and attempt completion have separate durable steps. A surviving NPZ without its terminal row record is not a completed collection and is not repaired by editing metadata.

Normal attempt states are `completed` or `failed`; handled interrupts record `interrupted`, while a killed process can leave `running`. Pre-work refusals are `aborted`. Formal readers hold shared locks and require all declared work to reconcile with terminal statuses and metric artifacts. Common declared budget exclusions can be paired; one-sided gaps cannot silently disappear. `--allow-ref-drift` does not bypass completion or executable identity checks.

For numerical comparisons, collect A and B against the same reference, then run [saved-metrics comparison](compare.md). For failures and verification commands, see [troubleshooting.md](troubleshooting.md).

The high-level `collect_model_kld.py` reads the optional checkpoint profile boolean `allow_vocab_attr_mismatch` (default false) and forwards the native compatibility flag uniformly. Freeze this only after auditing the complete vocabulary mapping and metadata; see the [VLM handover](kld-aws-handover.md#vocabulary-compatibility). The direct collector flag alone does not prove compatibility.
