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

Source input can be a Hub dataset (`--subset`, `--split`) or a local
`Dataset.save_to_disk` / `DatasetDict.save_to_disk` directory. It needs a
nonempty `question` string and optionally a list of encoded `images`.
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
plus source dataset fingerprint and invocation. Hashing large files on EBS
can take time after generation completes. Per-row sampling metadata records
the actual seed and sampler chain, including model-derived defaults. A run
chooses one seed and resets the sampler for each row; direct JSONL requests can
provide a per-row seed override.

## Outputs and replay contract

The output directory must be new:

- `dataset/`: validated reference rows, loadable with `datasets.load_from_disk`.
- `metadata.json`: effective settings and provenance, also embedded in rows.
- `requests.jsonl`, `command.json`, `native.log`: exact inputs, command, and logs.
- `native/metadata.json`, `native/generations.jsonl`: direct C++ output; each row is flushed.
- `inputs/`: original encoded images; `reference.arrow`: intermediate Arrow stream.
- `complete.json`: written only after all rows validate and the dataset is saved.

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
prompt plus generation cap must fit the context. Errors fail the run and leave
partial artifacts for diagnosis; there is no automatic resume or overwrite.
Backend sampling, reasoning budgets, custom reverse prompts, and MTP/speculative flags are rejected by the current producer. Its MTP driver has not been implemented.

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

## MTP reference provenance

MTP belongs only to reference generation. KLD remains ordinary teacher forcing
through the main reference/candidate models and does not load an MTP assistant.
Native v2 keeps `generation_metadata.vocabulary` tied to the target model.
`decoding.token_source` must be `target_accepted`, and `logprob_source` must be
`target_raw_logits`. Draft proposals or draft-head probabilities are not valid
reference targets/probabilities.

The current producer emits `decoding.method=autoregressive`. A producer that
implements MTP must emit `method=mtp`, `mtp.head_source` (`embedded` or
`sidecar`), and `mtp.settings`. Embedded heads are covered by the complete
`model_files` hashes; sidecars require complete `mtp.head_files` hashes. This
contract support is not an MTP runtime implementation or an equivalence claim.

Collectors preserve generator metadata and row sampling/logprob records under
`.attempts/<id>/generators/` and `references.jsonl`, even without `--keep-prep`.
The dataset fingerprint includes generation metadata and sampling/request fields.
The consumer verifies target vocabulary identity; it does not require the
reference generator's executable or MTP head to match the KLD executable.
