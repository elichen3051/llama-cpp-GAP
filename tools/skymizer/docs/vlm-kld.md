# VLM KLD collection

Collect both candidates against the same frozen reference trajectories, reference GGUF, projector and runtime. A checkpoint and its quantizations form one comparison family. Models are supplied by this checkout's upstream implementation; Skymizer adds evaluation tools, not model architectures or vision preprocessing.

## Frozen inputs and runtime

Use the [shared setup](workflows.md). A generated local `save_to_disk` directory can be passed directly. For published data, download its exact generated-reference Hub revision into a local `save_to_disk` directory first and preserve its publication receipt. This revision is distinct from the prepared-input revision used during generation. Generic collectors do not accept a Hub revision flag, so a moving Hub name is unsuitable for a formal frozen run.

Verify complete file hashes for all reference shards, the projector and each candidate. Check native generation vocabulary, image policy and eligible IDs against the scoring inputs. The collector's sampled model fingerprints do not replace complete-file integrity checks. Keep the same reference projector on both sides for LLM quantization comparisons.

| Setting | Accepted profile starting point |
| --- | --- |
| Context / logical batch | 32768 / 2048 |
| Microbatch | Gemma 31B: 2048; other included checkpoints: 512 |
| Teacher-forcing chunk | 2048 |
| Inference / batch / metric threads | 8 / 8 / 8 |
| Offload / flash attention | All supported layers / on |
| KV types / sequences | F16 K and V / 1 |
| MTP / auto fit | Off / no automatic runtime reduction |
| Image limits | Omitted, preserving the native dataset's recorded policy |
| SWA full-cache allocation | Omitted; architectural sliding-window attention remains active |

Both models and both projectors remain resident. Capacity depends on the candidate's buffers and actual image layout, not only its file size. Image count differs from tile count; complete non-causal chunks must fit batch and microbatch. A single-model load does not prove pair capacity. If a protocol must change after an OOM, apply it to every candidate of that checkpoint and create a new collection identity.

## Collect a pair

This example selects an instruct pilot scoring horizon. Replace all model paths and use the actual frozen dataset. For collect500 use its corresponding reference and 1024 instruct or 4096 thinking targets; pilot thinking uses 8192. Short answers retain their available targets.

```bash
export OUT_ROOT="$SKYMIZER_WORK/qwen35-pilot-vlm"
export VLM_DATASET=/path/to/frozen-reference/dataset
export VLM_REF_MODEL=/path/to/reference-bf16.gguf
export VLM_REF_MMPROJ=/path/to/reference-mmproj.gguf
export VLM_CAND_A_MODEL=/path/to/candidate-a.gguf
export VLM_CAND_B_MODEL=/path/to/candidate-b.gguf
export VLM_LABEL_A=Q4_K_M
export VLM_LABEL_B=Q4_1
export N_EVAL_TOKENS=2048

tools/skymizer/scripts/03_collect_vlm_kld.sh
tools/skymizer/scripts/05_compare.sh vlm
```

The direct wrapper uses the runtime overrides in `00_env.sh`. Set `N_UBATCH=2048` for the accepted Gemma 31B runtime. It does not derive model-specific settings from a profile. For a profile-controlled single-candidate study, use the high-level launcher instead:

```bash
"$SKYMIZER_PYTHON" tools/skymizer/cli/collect_model_kld.py \
  --profiles "$COHORT_PROFILE" --study "$SKYMIZER_WORK/profile-study" --size 100 \
  --model qwen3.5-4b --source mmmu-pro-vision --mode instruct \
  --candidate Q4_K_M --cand-model "$VLM_CAND_A_MODEL" --models-dir "$MODELS_DIR" \
  --dataset "$VLM_DATASET" --gpu 0 \
  --llama-vlm-kld "$SKYMIZER_WORK/build/bin/llama-vlm-kld" --dry-run
```

Remove `--dry-run` to collect. The dry run creates frozen study metadata; changing its profile afterward requires a new study. Each candidate needs its own label, while the reference dataset and runtime remain shared. Use one scorer per GPU, including across unrelated studies. Reference generation may use validated parallelism; KLD remains single-sequence with MTP off.

## Vocabulary compatibility and completion

The scorer checks vocabulary type, token IDs/text and attributes. `--allow-vocab-attr-mismatch` relaxes only candidate attribute equality. Use it only after a complete vocabulary audit and numerical confirmation, uniformly across the comparison. The high-level launcher accepts the profile boolean `allow_vocab_attr_mismatch` and freezes it in the study; the default is false. Structural load success does not establish numerical quality or justify bypassing a strict check.

Keep `collect_meta.json`, `manifest.csv`, all metric files, native logs and attempt records. Reconcile every planned row before comparison. Failed or interrupted work stays as evidence; retry into a new output root. Common budget exclusions can be paired, but one-sided missing rows cannot disappear silently. Freeze the largest intended scoring horizon and derive shorter prefixes from the saved metrics; a separately scored shorter horizon can change the final batch and reference logits.

The report measures answer-token distribution fidelity conditioned on images. It does not measure vision-embedding distributions or task accuracy. Paired inference requires an appropriate sampling unit; source/image clusters and overlap between pilot100 and collect500 should be reported. See the [collection contract](collect.md), [statistics](compare.md) and [validation](../verify_and_validation_scripts/README.md).
