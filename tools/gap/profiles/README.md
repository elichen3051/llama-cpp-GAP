# Reference and KLD profiles

Select a JSON explicitly with `--profiles`. The old implicit six-model profile used an older dataset revision and generation caps; it is no longer a production default.

| Group | Pilot | 500 questions | Models |
| --- | --- | --- | --- |
| Small SNR models | [small-pilot100.json](small-pilot100.json) | [small-collect500.json](small-collect500.json) | Qwen3.5-4B, Gemma 4 E4B |
| Remaining SNR models | [snr-pilot100.json](snr-pilot100.json) | [snr-collect500.json](snr-collect500.json) | Gemma 4 31B, Qwen3.6 35B A3B, Kimi Instruct, Kimi Thinking 2506 |
| Final evaluation models | [final-pilot100.json](final-pilot100.json) | [final-collect500.json](final-collect500.json) | InternVL3.5, GLM-4.6V-Flash, Muse Glimmer |

For each cohort, finish the SNR groups before the final evaluation group. Use separate study directories for each group and cohort. Kimi checkpoints have separate supported modes. Muse has only the accepted thinking/high profile with fixed reasoning strength and date; the later four-strength diagnostic does not automatically replace it.

| Setting | pilot100 | collect500 |
| --- | ---: | ---: |
| Instruct generation cap | 2048 | 1024 |
| Thinking generation cap | 8192 | 4096 |
| Source rows requested per subset | 100 | 500 |

All six profiles freeze `user-company/vlm-prepared-dataset` at revision `6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b`, split `train`, with the seven listed source subsets. Cohort size is validated by the launchers and uploader. KLD scoring horizons are explicit profile fields; they describe a chosen collection protocol, not a demonstrated SNR plateau.

The profiles originate from the accepted 2026-09-06 RunPod supplements. Small profiles now encode the handover's explicit `--parallel 4` override; all other source model, tokenizer, projector, dataset, sampling and cap identities are retained. Per-sequence context is 32768, so small-model native generation receives total context 131072 and four slots. Other profiles use one sequence; Gemma31 uses its sidecar MTP head with draft length 8. All KLD scorers use one sequence and no MTP.

Paths for model shards, projectors and MTP heads are relative to `--models-dir`. Source/card hashes identify the intended files; free-form historical provenance is not a required filesystem dependency. Restore missing files with `cli/restore_reference_models.py` and the same profile. Do not substitute a different provider's projector or change a frozen study in place.

Bounded acceptance is not a guarantee that every image or candidate fits memory. Keep the model-specific context, batch and ubatch settings, eight inference/metric threads, flash attention, full layer offload and default mmproj image limits. Existing generated reference datasets retain their original settings and tokens; do not regenerate them merely to adopt this directory layout.
