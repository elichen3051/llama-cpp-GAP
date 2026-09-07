# AWS: collect VLM KLD

Updated 2026-09-06. All nine current reference-plus-candidate capacity controls passed on the RTX PRO 6000 Blackwell Server Edition. The 113 specified candidate files passed structural, complete-file hash and short operation checks; their separate sparse numerical screen is running. Production collection must use the resulting candidate roster, frozen references and the family runtime below.

This document collects metrics on existing reference trajectories. Reference weights, generation and private publication are covered by the [RunPod reference handover](reference-runpod-handover.md) and [InternVL/GLM/Muse handover](reference-runpod-final-handover.md). [WikiText-2/full-PG PPL and LLM KLD](perplexity-llm-kld-aws-handover.md) use a separate 512-token protocol.

## Scope and order

A family means quantizations of one base checkpoint. Different sizes can use different runtimes. Kimi Instruct and Kimi Thinking-2506 are separate checkpoints. Use only the model implementations supported by upstream base `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`; no new model, kernel or vision implementation is supplied here.

For each cohort, finish the SNR-decision group first: Gemma 31B, Gemma E4B, the two Kimi checkpoints, Qwen3.5-4B and Qwen3.6-35B-A3B. Then process InternVL3.5-30B-A3B, GLM-4.6V-Flash and Muse Glimmer. Gemma 26B is outside this experiment. Muse supports thinking only in the current profiles.

Use one scorer at a time per GPU, including across separate studies. Output/study locks do not provide a global GPU reservation. Reference generation can use validated parallel generation, while the VLM scorer remains single-sequence. Gemma 31B uses MTP to generate references; KLD uses ordinary main-model teacher forcing with MTP off.

## Frozen inputs

Use the reference LLM and mmproj provider/files from the generation profile. Verify all reference shards, the projector and candidate against their complete SHA256 receipts; sampled collector fingerprints do not replace that initial integrity check. For LLM quantization comparisons, keep the same reference projector on both scorer sides.

Materialize the published reference config at its exact Hub commit into a local NVMe `save_to_disk` directory. Preserve its source/audit manifest and publication receipt beside it. The generated-reference commit is different from the prepared-input dataset revision `6cb6a4d...`; do not substitute the latter. The collector's generic Hub loading does not accept a revision flag, so use a local frozen dataset for formal collection.

Before launching, compare the saved generation model/projector SHA256, vocabulary, image policy, eligible IDs and source config with the selected scoring profile. These full generation-to-scorer identity checks remain an operator responsibility. Then freeze binaries, loaded backend libraries, GPU/driver, relevant environment and runtime across every candidate of that checkpoint.

## Vocabulary compatibility

The CPU-only vocabulary audit checked all 113 candidates. Token text/ID mapping, size and vocabulary type match their reference. Sixteen Unsloth Gemma candidates differ only in token ID1 `<eos>` attributes (reference8, candidate12); the declared EOS ID is106 instead of1, while the complete end-of-generation token set and BOS policy match. Strict scorer preflight refuses these candidates before loading weights. This is not a measured KLD/PPL failure.

The native scorers already provide `--allow-vocab-attr-mismatch`. It relaxes candidate attributes only; reference identity and token-ID mapping remain strict. Its use requires a documented vocabulary audit and separate numerical confirmation. For the high-level VLM launcher, set the boolean `allow_vocab_attr_mismatch` on the checkpoint's model profile before creating a new study. The frozen profile applies it to every candidate and mode of that checkpoint and records it in `study.json`; the default remains false. Direct collectors must use the same explicit policy for every candidate being compared. Do not edit existing archives or silently retry with another policy.

The Gemma compatibility confirmation is separate from the primary strict screening records and is still pending. Qwen3.6 Unsloth IQ4 numerical outliers are a different issue: they pass vocabulary checks and require additional fixed-window confirmation before a final roster decision.

## Runtime and current capacity

| Setting | VLM scoring protocol |
| --- | --- |
| Context / logical batch | 32768 / 2048 |
| Microbatch | Gemma 31B: 2048; all other checkpoints: 512 |
| Teacher-forcing chunk | 2048 |
| Inference / batch / metric threads | 8 / 8 / 8 |
| GPU offload | All supported layers (`--n-gpu-layers -2`) |
| Flash attention | Enabled (`--flash-attn`) |
| KV types | K and V are F16 |
| Sequences | 1 |
| SWA cache | Omit `--swa-full`; this does not disable architectural sliding-window attention |
| Fit / MTP | No auto-fit reduction; MTP off |
| Image-token arguments | Omitted; current reference datasets record mmproj defaults |

The chosen projector and actual image layout determine capacity. Image count differs from tile count, and native image token counts can differ from position extents. Complete non-causal chunks must fit the model's batch/microbatch requirements. All native layers were offloaded in the tests; expected CPU-mapped input embeddings are not a missing-layer failure.

Current controls used the largest candidate file within each checkpoint's audited inventory, with the reference and both projectors resident. Each row used three source images selected from previous high-demand probes. These are capacity tests with synthetic target continuations, not quality/SNR measurements or exhaustive image maxima.

| Checkpoint | Candidate used | Targets | Peak MiB | Free MiB at sampled peak |
| --- | --- | ---: | ---: | ---: |
| Gemma 31B | Q4_1 | 16384 | 96411 | 1476 |
| Gemma E4B | Q4_1 | 8192 | 18383 | 79504 |
| Kimi Instruct | mradermacher i1-Q4_K_M | 8192 | 45593 | 52294 |
| Kimi Thinking-2506 | mradermacher i1-Q4_K_M | 8192 | 45593 | 52294 |
| Qwen3.5-4B | Q4_K_M | 8192 | 17449 | 80438 |
| Qwen3.6-35B-A3B | Q4_1 | 16384 | 92317 | 5570 |
| InternVL3.5-30B-A3B | Q4_1 | 8192 | 84931 | 12956 |
| GLM-4.6V-Flash | Unsloth UD-Q4_K_XL | 8192 | 32345 | 65542 |
| Muse Glimmer | Q4_1 | 8192 | 81361 | 16526 |

All 207 checks and 90,112 target records passed, including exact targets, finite fields, strict prefixes, actual runtime and complete GPU offload. The eight older selected rows retain revision `c3f0e320...`; Muse uses `6cb6a4d...`. Their three source images have identical hashes across those revisions, but the row revisions remain separately recorded. Current controls re-ran the current backend; old 8-thread results used different libraries and are not substitutes.

Gemma 31B's 1476 MiB margin is small. A largest-by-file-size candidate is a useful capacity control, not a proof that every kernel/workspace combination or every source image fits. Do not change context, microbatch, offload or image budget for a single candidate after OOM. Record the failed row; a protocol change applies to the whole checkpoint and a new collection identity. Accepted failures do not become replacement questions.

InternVL's old adjacent-tile strict mismatch was a Skymizer checker error. It was fixed without changing upstream model/mtmd code, passed the original three-row real-target replay, and passed the current 8k capacity control. Details and source hashes are in the final reference handover.

## Cohort horizons

| Cohort | Instruct cap | Thinking cap |
| --- | ---: | ---: |
| Pilot, 100 requested rows per source | 2048 | 8192 |
| Full collection, 500 requested rows per source | 1024 | 4096 |

Use the matching cohort profile and reference config; `generation_caps` and `kld_eval_tokens` are aligned. A short answer contributes only its actual targets. Prompt/image tokens consume capacity but are not scored. Preserve the largest intended horizon once, then derive shorter prefix reports offline. Runtime flags alone do not guarantee the same floating-point results when the final scoring batch changes with horizon.

Pilot overlap with the 500-question cohort is accepted for SNR planning. Repetition exclusions, truncation and failed rows remain explicit. Keep common item IDs and source/image clusters in the analysis; overlapping or related question sources are not independent replications. The earlier vLLM SNR plateau does not establish saturation for the new native-reference protocol.

## Prepare the AWS environment

Run from the repository root. Model files remain under `~/models`; builds, caches, temporary files, datasets, logs and outputs use NVMe.

```bash
export SKYMIZER_KLD_ROOT=/opt/dlami/nvme/skymizer-kld
export TMPDIR="$SKYMIZER_KLD_ROOT/tmp"
export HF_DATASETS_CACHE="$SKYMIZER_KLD_ROOT/cache/datasets"
export HF_HUB_CACHE="$SKYMIZER_KLD_ROOT/cache/hub"
export HF_XET_CACHE="$SKYMIZER_KLD_ROOT/cache/xet"
export HF_ASSETS_CACHE="$SKYMIZER_KLD_ROOT/cache/assets"
export CUDA_CACHE_PATH="$SKYMIZER_KLD_ROOT/cache/cuda"
export UV_CACHE_DIR="$SKYMIZER_KLD_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$SKYMIZER_KLD_ROOT/venv"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"
cmake -S . -B "$SKYMIZER_KLD_ROOT/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$SKYMIZER_KLD_ROOT/build" --target llama-vlm-kld -j8
uv sync --project tools/skymizer --frozen --no-dev
```

Cache relocation does not log in to a private Hub dataset. Reuse the existing authenticated environment or provide the intended account's token on the new host.

## Collect one candidate

The high-level launcher archives the selected cohort profile and derives the table's runtime and scoring cap. Set `COHORT_PROFILE` to the delivered pilot100 profile, `REFERENCE_DATASET` to the verified local dataset, and `CANDIDATE` to the exact audited quantized GGUF. This example selects Qwen3.5 instruct on MMMU Vision:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" tools/skymizer/cli/collect_model_kld.py \
  --study "$SKYMIZER_KLD_ROOT/pilot-qwen35" --size 100 --profiles "$COHORT_PROFILE" \
  --model qwen3.5-4b --source mmmu-pro-vision --mode instruct \
  --candidate Q4_K_M --cand-model "$CANDIDATE" --models-dir "$HOME/models" \
  --dataset "$REFERENCE_DATASET" --gpu 0 \
  --llama-vlm-kld "$SKYMIZER_KLD_ROOT/build/bin/llama-vlm-kld" --dry-run
```

Inspect the rendered command, then remove `--dry-run`. The dry run creates the frozen study metadata; changing profiles afterward requires a new study. Use each candidate's own label, keep the same reference dataset and study protocol, and schedule candidates serially. For collect500, use its profile, `--size 500`, the 500 reference dataset and a new study. The [direct collector contract](collect.md) covers lower-level commands and disjoint completed appends.

Preserve the whole output, including logs, attempt state, `collect_meta.json`, `manifest.csv`, metrics and publication/source receipts. Completed work must reconcile every planned row with a terminal status; partial artifacts alone are not completion. Use a fresh study/output after interrupted work and retain its evidence. The comparator enforces completed paired inputs, target/reference alignment and runtime/backend identity.

## Evidence and remaining scope

Current capacity evidence: `/opt/dlami/nvme/skymizer-candidate-audit-20260906/vlm-capacity-plan/FINAL_CAPACITY_REPORT.md`, its plan, per-row receipts and cleanup record. Candidate integrity/operation evidence: the adjacent `runtime/HEALTH_REPORT.md`; numerical screening and any confirmation results belong to `phase2/runtime/`. Capacity and structural acceptance do not by themselves approve numerical quality.

Reference generation versus teacher-forced replay can differ numerically even at identical target IDs because execution shapes differ. Earlier selected-token diagnostics observed differences up to 0.06698 nats; that is not a full-distribution KLD estimate. The 4k/16k Qwen3.6 test also showed a final-batch prefix difference. These observations motivate frozen scoring horizons and shared-reference checks, not per-candidate runtime adjustments.

Full pilot/500 metric collection and paper SNR conclusions are not produced by these acceptance smokes. Short decision-relevant anomalies and final candidate outcomes will be recorded with the final audit summary.
