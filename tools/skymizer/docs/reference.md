# Native reference datasets

`llama-reference` generates reference trajectories directly with llama.cpp.
It loads a local GGUF model once, renders its chat template with `common/chat`,
tokenizes with the GGUF vocabulary, and processes images with mtmd. The Python
entry point only reads source datasets, preserves encoded images, validates
rows, and writes an Arrow dataset. Neither component needs llama-server,
Transformers, an HF processor/tokenizer, or the parent reference-generator repo.

## Build and generate

Run from the llama.cpp checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-reference llama-vlm-kld llama-llm-kld -j4
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project tools/skymizer --python 3.12

.venv/bin/python tools/skymizer/cli/generate_reference.py \
  --dataset /path/to/prepared-dataset --out /path/to/new-reference \
  --no-enable-thinking -- \
  -m /models/model-bf16.gguf --mmproj /models/mmproj.gguf \
  -ngl 99 -c 8192 -n 128 --seed 1234
```

For a text-only dataset, omit `--mmproj`. A run with a projector can mix text,
single-image, and multi-image rows. All native options follow `--`; use
`build/bin/llama-reference --help` for llama.cpp flags. The tool uses a context
of 8192 and a generation cap of 128 unless overridden. It requires one sequence.

Source input can be a Hub dataset (`--subset`, `--split`, `--revision`) or a local
`Dataset.save_to_disk` / `DatasetDict.save_to_disk` directory. It needs a
`question` string and optionally a list of encoded `images`. Image-only rows may use an empty question; rows with neither text nor images are rejected.
`--question-column`, `--images-column`, and `--id-column` select other columns.
IDs default to `item_id`, then `id`, then a generated row index; IDs must be
unique and filesystem-safe. `--num-samples` selects the first N rows.
HF Datasets is a storage/input dependency, with no tokenizer alignment step.

Images stay byte-exact from source through dataset to collector. Image order
is preserved and images precede the question in the user message. Optional
`--system-prompt` adds a system message. Arbitrary multi-turn conversations,
audio, and video are not supported by this schema.

## Sampling, thinking, and metadata

Sampling uses upstream `common` defaults, including sampling metadata read
from the GGUF. Explicit llama.cpp flags take precedence over those values;
for example `--temp 1.0 --top-p 0.95 --top-k 64`. The Markdown model-card
recommendations are not automatically read. There is no `vllm-like` policy or
HF parity mode in this producer. The sampler sees the prompt's text tokens;
image embedding placeholders are excluded from penalty history.

`--enable-thinking` / `--no-enable-thinking` are Python options before `--`.
They pass a boolean to the native chat template. Omitting them preserves the
native reasoning/template defaults. Native `--reasoning on|off|auto` and
`--chat-template-kwargs` are also available after `--`; a per-request thinking
value takes precedence. `--thinking-column COLUMN` reads a boolean mode per
source row and overrides the global Python option, so one model load can
generate both modes. Thinking support depends on the GGUF template. Raw
thinking tokens remain in the trajectory and count toward the generation cap.

The equivalent of querying server `/props` is:

```bash
build/bin/llama-reference -m /models/model-bf16.gguf \
  --mmproj /models/mmproj.gguf -ngl 99 -c 8192 --describe > props.json
```

This loads the model/projector and reports effective sampling, actual context
and batch sizes, vocabulary IDs/size, GGUF metadata, chat template, template
kwargs, thinking support/default, image budget settings, and build information.
Logs go to stderr. `image_min_tokens` / `image_max_tokens` equal `-1` when the
projector supplies the default; the per-row mtmd layout records actual image
token and position counts.

Every generation run writes this metadata. The Python driver additionally
records SHA-256 for its binary, all model shards, the projector, and each image,
plus source dataset fingerprint and invocation. `execution_identity` records the producer executable, loaded backend libraries, and numerical runtime environment; it is checked again after generation. Hashing large files on EBS
can take time after generation completes. Per-row sampling metadata records
the actual seed and sampler chain, including model-derived defaults. The Python driver resolves a random seed once when none is specified, records it before the first attempt, and preserves it across retries. A run resets the sampler for each row; direct JSONL requests can
provide a per-row seed override.

## Outputs and replay contract

The output directory must be new:

- `dataset/`: eligible validated reference rows in source order, loadable with `datasets.load_from_disk`; absent when no rows are eligible.
- `metadata.json`: effective settings and provenance, also embedded in rows.
- `requests.jsonl`: exact prepared requests; `attempts/NNNN/` preserves each native command, request subset, log, exit status, row journal, and raw results.
- `native/metadata.json`, `native/generations.jsonl`: direct C++ output; each row is flushed.
- `inputs/`: original encoded images; `scripts/`: copies of the Python/native producer sources and profiles used.
- `run_start.json`, `run_state.json`, `progress.json`: source/execution identity, lifecycle status, and remaining IDs.
- `excluded.jsonl`, `failures.jsonl`: repetition evidence and classified row failures.
- `complete.json`: written atomically after every requested ID has an eligible, excluded, or failed outcome; `complete_with_failures` does not mean every row generated successfully.

The schema version is `skymizer-reference-v2`; its implementation and validation
live in `lib/reference_dataset.py` and `lib/reference_contract.py`.
`input_ids` contains the complete GGUF prompt and generated trajectory.
`n_prefill_tokens` separates the prompt from the answer, and `labels` masks
prompt positions with `-100`. Image spans use `LLAMA_TOKEN_NULL` (`-1`), never
an HF image-pad token. Their exact mtmd chunks, token counts, grids, and position
counts are saved in `llamacpp_prompt_layout` and the per-image columns.
`llamacpp_n_past_prefill` can differ from `n_prefill_tokens` with multimodal RoPE.

`generation_token_logprobs` records the sampled tokens' raw-model log-probability
before sampling filters or penalties, suitable for checking teacher-forced
replay. EOG tokens are included in the trajectory and log-probs. Model/template
stops set `finish_reason=stop`; hitting the cap sets `length` and
`truncated_by_cap=true`. There is no silent prompt truncation or context shift:
prompt plus generation cap must fit the context. A context-budget or invalid-image error is recorded for that row and processing continues. Backend failures restart the native process with pending IDs; completed rows are retained. There is no overwrite of existing output directories.
Target backend sampling, reasoning budgets, custom reverse prompts, LoRA, and speculative methods other than MTP are rejected. See below for the MTP driver.

## Collect KLD

The existing collectors consume the local output directly. VLM example:

```bash
.venv/bin/python tools/skymizer/cli/collect_kld.py \
  --dataset /path/to/new-reference/dataset --out /path/to/new-kld \
  --ref-model /models/model-bf16.gguf --ref-mmproj /models/mmproj.gguf \
  --cand-model /models/model-q4.gguf --cand-mmproj /models/mmproj.gguf \
  --n-ctx 8192 --num-eval-tokens 16
```

For text-only rows:

```bash
.venv/bin/python tools/skymizer/cli/collect_llm_kld.py \
  --dataset /path/to/text-reference/dataset --subset '' --out /path/to/new-llm-kld \
  --ref-model /models/model-bf16.gguf --cand-model /models/model-q4.gguf \
  --n-ctx 8192 --num-eval-tokens 16
```

The VLM collector reuses the saved prompt and original image bytes, adopts the
recorded image budget, and rejects prefix/position drift by default. The LLM
collector takes the recorded token IDs directly and rejects image rows. Both
validate the native schema before writing scorer inputs. No HF tokenizer is
loaded. The producer records a versioned SHA-256 of the target vocabulary type,
size, and ordered ID-to-token text mapping, plus a separate attribute digest.
Both scorers check native manifests against the reference and candidate GGUF
vocabularies before loading model weights. Equal sizes alone do not pass.
The candidate's existing attribute-only override does not relax token mapping
or the generator-to-reference attribute check. Old native v1 rows must be regenerated.
Legacy HF/vLLM VLM datasets remain supported with `uv sync --extra hf-tokenizer`.


Native v2 requires one finite, non-positive raw target logprob for each generated
token. Missing, null, partial, or positive logprob vectors are rejected.

## Generate with MTP

MTP belongs only to reference generation. KLD remains ordinary teacher forcing
through the main reference/candidate models and does not load an MTP assistant.
Native v2 keeps `generation_metadata.vocabulary` tied to the target model.
`decoding.token_source` must be `target_accepted`, and `logprob_source` must be
`target_raw_logits`. Draft proposals or draft-head probabilities are not valid
reference targets/probabilities.

Autoregressive generation remains the default. To enable an embedded MTP head (for example, Qwen3.5 or Qwen3.6 GGUFs with next-N layers), add these native options after `--`:

```bash
--spec-type draft-mtp
```

For a separate MTP assistant GGUF (for example, Gemma 4 26B-A4B or 31B), also supply the matching sidecar:

```bash
--spec-type draft-mtp --model-draft /models/mtp-model-Q8_0.gguf -ngld 99
```

`--spec-draft-n-max`, `--spec-draft-n-min`, and `--spec-draft-p-min` use upstream defaults unless overridden. The maximum must be positive and smaller than both `-b` and `-ub`; the driver shortens each proposal to fit the remaining generation cap. Only local target/head files are accepted. Sidecars must have MTP layers and match the target's hidden width and ordered token mapping. Synthetic acceptance is forbidden.

The driver uses upstream `common/speculative` for proposals. It samples sequentially from the target, accepts matching proposals, and discards the rest at the first mismatch. Only the committed prefix updates the MTP carry state. Each row resets target/draft memory and samplers. Images are evaluated by mtmd in the target context; as in the upstream MTP implementation, the draft head processes the text batches and requires a text suffix after the final image.

MTP runs emit `decoding.method=mtp`, `mtp.head_source` (`embedded` or `sidecar`), and effective `mtp.settings`. Embedded heads are covered by all `model_files` hashes; sidecars have complete `mtp.head_files` hashes. `generation_decoding_stats` preserves each row's drafted tokens, accepted draft tokens, verification batches, rollback batches, and checkpoint replays. `accepted_draft_tokens` counts draft inputs retained in the verified context prefix; the last emitted token stays pending, including at a stop. `verification_batches + accepted_draft_tokens` equals generated length minus one: the first token comes directly from target prefill.

MTP verification changes target batch sizes. GPU floating-point results can therefore differ from autoregressive generation or teacher forcing with `--tf-chunk 1`, even with the same target weights. Saved logprobs are the raw target logits used at generation time, before target sampling filters; they are not recomputed to force agreement with a later collector. A controlled Qwen3.5 BF16 probe reproduced the observed MTP/AR discrepancy with ordinary teacher forcing at chunk sizes 1 versus 4 (maximum absolute logprob difference about 0.0126). This is not a claim of bitwise trajectory or logprob equivalence. Keep the same teacher-forcing settings for both KLD candidates; KLD never loads or runs MTP.

Collectors preserve generator metadata and row sampling/logprob/decoding-stat records under
`.attempts/<id>/generators/` and `references.jsonl`, even without `--keep-prep`.
The dataset fingerprint includes generation metadata and sampling/request fields.
The consumer verifies target vocabulary identity; it does not require the
reference generator's executable or MTP head to match the KLD executable.


## Repetition and recovery

Repetition handling lives entirely in `tools/skymizer`; no external detector checkout is needed at runtime. The native implementation ports the reversed-KMP tail detector and exact consecutive-block logic from `repetition_curse_detector` commit `d4163fc1c328fe39310465680647b465cf96c4af`. It requires at least 3 consecutive repetitions and a repeated span of at least 96 generated tokens. Prompt tokens, image placeholders, and unaccepted MTP proposals are excluded from detection.

Every 32 committed output tokens, the online detector examines the last 2048 tokens. A hit stops generation with `finish_reason=stop` and `stop_type=repetition`. A final exact scan checks the entire generated sequence, including repetitions before closing tokens or inside the answer. The online window bounds the periods it can catch before completion; the final scan also covers longer repeated blocks. Ordinary short punctuation, equations, or small repeated phrases do not meet the 96-token gate. This conservative exact detector is not a semantic/fuzzy repetition classifier; retain a later review pass for near-duplicates and legitimate repeated material.

Native results keep repetition offsets, unit length, repeat count, repeated span, detection position, and stage. Python retains all completed native results but excludes flagged rows from `dataset/`, preserving their text and evidence in `excluded.jsonl`. IDs never change after filtering. `metadata.cohort` records the ordered requested/eligible/excluded/failed IDs, their digests, and counts. `generated` counts validated eligible plus excluded records; `native_generated` also includes any raw result that failed validation. Both KLD candidates must consume the same saved eligible dataset. Do not regenerate/filter separately and intersect their successful rows.

`--row-retries` defaults to 1 retry for a crashed/backend-failed row. Invalid input/context-budget rows are terminal failures. `--startup-retries` defaults to 1; persistent model startup failure leaves the remaining IDs explicitly unattempted. Pending untouched rows run before the crashed row is retried. A flushed `row_started` journal entry identifies the current row; durable output wins if a crash occurred before the success journal entry. A model startup crash is never attributed to the first unstarted row.

`--row-timeout` defaults to 1800 seconds without a row-journal update, including model startup. The supervisor kills and waits for a timed-out native process before restarting. `--native-timeout` optionally limits total time per native process. Intentional SIGINT/SIGTERM records interruption and stops the child; it does not trigger infinite retries. Every attempt and partial trailing record remains available for diagnosis. Interior journal corruption, changed execution/settings, or irreconcilable IDs fail the job rather than silently accepting uncertain data.

Native `--no-repetition-stop` disables both online and final repetition checks for controlled diagnostics. Production profiles use the default enabled policy. Native `--continue-on-error` enables recoverable row continuation; the Python driver supplies it automatically. The standalone binary's `--help` also lists ordinary llama.cpp options.
