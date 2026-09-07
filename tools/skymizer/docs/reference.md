# Native reference datasets

`llama-reference` generates reference trajectories directly with llama.cpp.
It loads a local GGUF model once, renders its chat template with `common/chat`,
tokenizes with the GGUF vocabulary, and processes images with mtmd. The Python
entry point only reads source datasets, preserves encoded images, validates
rows, and writes an Arrow dataset. Neither component needs llama-server,
Transformers, an HF processor/tokenizer, or the parent reference-generator repo.

## Build and generate

Use the [README environment and build commands](workflows.md), then run from the repository root. This short example illustrates the low-level interface; use a validated cohort profile for formal generation.

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/generate_reference.py \
  --llama-reference "$SKYMIZER_WORK/build/bin/llama-reference" \
  --dataset /path/to/prepared-dataset --out "$SKYMIZER_WORK/reference-example" \
  --no-enable-thinking -- \
  -m /models/model-bf16.gguf --mmproj /models/mmproj.gguf \
  -ngl all -c 8192 -b 2048 -ub 512 -t 8 -tb 8 -fa on --fit off \
  -ctk f16 -ctv f16 -n 128 --seed 1234
```

For a text-only dataset, omit `--mmproj`. A run with a projector can mix text,
single-image, and multi-image rows. All native options follow `--`; use
`"$SKYMIZER_WORK/build/bin/llama-reference" --help` for llama.cpp flags. The tool uses a context
of 8192 and a generation cap of 128 unless overridden. Autoregressive generation supports `-np N` parallel sequences in one model/context; MTP currently requires `-np 1`.

With the default separate KV allocation, native `-c` is total context: use `-np 4 -c 131072` to retain 32768 positions per sequence. The high-level `generate_model_reference.py --parallel 4` instead treats profile `ctx` and `--ctx` as per-sequence capacity and multiplies by the slot count. Check actual `n_ctx`, `n_ctx_per_seq` and `total_slots` in metadata. Each row has its own sampler, seed, repetition history, positions and token/logprob buffers; image prefill is serial, and active rows share batched autoregressive decode. Completion order may differ from request order; the Python driver restores source order. Different batch shapes can change floating-point results and sampled trajectories, even with the same seed.

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
"$SKYMIZER_WORK/build/bin/llama-reference" -m /models/model-bf16.gguf \
  --mmproj /models/mmproj.gguf -ngl all -c 8192 -b 2048 -ub 512 \
  -t 8 -tb 8 -fa on --fit off -ctk f16 -ctv f16 --describe > "$SKYMIZER_WORK/props.json"
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
- `inputs/`: original encoded images; `scripts/`: copies of the Python/native producer sources. Preserve the selected profile and its checksum with the run; a campaign also archives its selected profile.
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

`--row-retries` defaults to 1 retry for a crashed/backend-failed row. Invalid input/context-budget rows are terminal failures. `--startup-retries` defaults to 1; persistent model startup failure leaves the remaining IDs explicitly unattempted. Pending untouched rows run before interrupted rows are retried. Each active ID has one flushed `row_started` entry; durable output wins if a crash occurred before its success journal entry. Recovery retains completed rows and applies the retry budget to every interrupted row, while checking the declared slot limit. A model startup crash is never attributed to the first unstarted row.

`--row-timeout` defaults to 1800 seconds for each active row, measured from its observed start journal entry, or 1800 seconds without journal progress during startup/idle periods. Other rows completing do not extend a stalled row's deadline. The supervisor kills and waits for a timed-out native process before restarting. `--native-timeout` optionally limits total time per native process. Intentional SIGINT/SIGTERM records interruption and stops the child; it does not trigger infinite retries. Every attempt and partial trailing record remains available for diagnosis. Interior journal corruption, changed execution/settings, or irreconcilable IDs fail the job rather than silently accepting uncertain data.

Native `--no-repetition-stop` disables both online and final repetition checks for controlled diagnostics. Production profiles use the default enabled policy. Native `--continue-on-error` enables recoverable row continuation; the Python driver supplies it automatically. The standalone binary's `--help` also lists ordinary llama.cpp options.

## Production profiles, shared GPU queue, and publication

Use a cohort-specific profile from the [portable reference guide](runpod-reference.md). The 100-question pilot uses instruct/thinking caps of 2048/8192; the 500-question cohort uses 1024/4096. The six profiles in `profiles/` separate small SNR, remaining SNR and final-evaluation checkpoints at each cohort size. High-level generation, collection and campaign CLIs require an explicit `--profiles` selection. A profile declaring `cohort_size` is rejected when used with another size.

`generate_model_reference.py` selects one checkpoint, source subset and semantic mode. The profile binds the source revision, complete reference/projector/MTP identities, effective sampling, image policy and per-mode runtime. `run_reference_campaign.py` enumerates only each model's supported modes; Muse has thinking only. The overview records actual per-mode parallel counts and does not assign a global generation sequence count.

Run from the repository root after the [README environment setup](workflows.md):

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/run_reference_campaign.py \
  --profiles "$COHORT_PROFILE" --models "$CHECKPOINT" \
  --out "$SKYMIZER_WORK/reference-pilot" --size 100 --gpus 0 --hardware pro6000 \
  --llama-reference "$SKYMIZER_WORK/build/bin/llama-reference"
```

Use the size500 profile, `--size 500` and a new directory for the full cohort. A diagnostic run can select one source and use `--num-samples 1 --no-upload`; short runs cannot be published into formal configs. Finish the SNR-decision group before launching final-evaluation models for each cohort. The campaign has no cross-host group gate or global GPU lock; the operator controls disjoint GPU assignments and group order.

One worker per selected GPU takes generation jobs from a shared queue, while upload workers handle completed jobs. Reference generation can use a validated parallel profile; MTP requires one sequence in this implementation. Complete non-causal image chunks must fit both batch and microbatch. The producer records a row failure instead of splitting such a chunk into causal batches. Use the validated capacity in the later VLM scorer as well.

`--resume` checks the frozen plan, archived source, binaries/libraries, completed artifacts and upload receipts. The archive includes `core/`, `cli/`, `lib/`, `stats/`, scripts and profiles, plus git source/diff, dependency and GPU provenance. It refuses changed archived files. A directory lock excludes a second owner of that campaign, but does not reserve the GPU against other jobs.

Generation jobs have a 48-hour process timeout and uploads have a one-hour timeout with three attempts; configure deadlines before freezing the plan. Cancellation terminates child process groups and kills remaining descendants after the grace period. Completed rows survive native retries within one invocation; a restarted Python job uses a new attempt. Failed/excluded rows remain in the audit and receive no replacements. Ordinary job failures do not stop independent queued jobs.

`upload_reference.py` validates cohort reconciliation, identities, source revision, templates, image policy, mode, caps and sampling before staging parquet. Publication uses `elichen-skymizer/<checkpoint>-pilot` or `<checkpoint>-collect-500`; each source config gains `-ins` or `-think`, with split `train`. Only eligible rows enter the dataset; complete audit files record failures and exclusions.

Publication defaults to private, including the campaign path. Existing public destinations are changed to private and verified before data upload. Privacy is checked again before accepting a receipt. Permission errors or a destination that remains public stop publication. `--private` remains accepted explicitly; `--dry-run` only validates and exports local files. Hugging Face ignores `create_repo(private=True)` for an existing repository, so changing its visibility requires `update_repo_settings`; see the [Hub API contract](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo).

Parquet, dataset-card config mapping and audit files are added in one compare-and-swap Hub commit. Existing conflicting data/config namespaces are refused. Repeated uploads verify the pinned remote files and hashes. The local `upload/receipt.json` records the verified revision and private status. A receipt records the observed visibility at verification time; an external owner can subsequently change repository settings.

Campaign exit0 requires every planned job to complete without row failures and all requested uploads to verify. Exit2 means completed processing with failures; exit130 means interruption. Preserve `status.json`, attempt logs and exact Hub revisions. An upload-enabled campaign requires the archived uploader to match this release before any work is dispatched, and checks again before upload. Legacy archives are preserved and refused for automatic publication. Use a new campaign directory, or upload an existing completed run with the current standalone `upload_reference.py --run OLD_RUN --profiles OLD_ARCHIVED_PROFILE --model CHECKPOINT --mode MODE --private`. This revalidates the local artifacts and destination; an old receipt alone is not sufficient. Do not replace files in an immutable archive.

`restore_reference_models.py --profiles "$COHORT_PROFILE"` plans exact S3 object downloads for all reference shards, projectors and declared MTP sidecars. Add `--download` to fetch missing files. It verifies complete SHA256 and size, preserves verified files, rejects mismatches and installs downloads without overwriting another writer. Exact object keys do not require bucket listing. Set the S3 bucket's region and the selected model root as described in the [reference guide](runpod-reference.md).
