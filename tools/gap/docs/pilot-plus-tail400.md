# Reuse pilot KLD and collect the remaining400 items

The `*-collect-400` reference repositories contain configs named
`<source>-tail-400-ins` or `<source>-tail-400-think`. Each selects the requested
positions101..500 of the prepared500 pool by stable item ID. Generation
exclusions remain exclusions; the logical dataset can contain fewer than400
rows. Pilot exclusions remain authoritative for positions1..100.

The current nine repositories contain 105 materialized configs. Each Parquet holds only the eligible tail rows, with original encoded images, generated tokens and parent500 row metadata preserved. The requested tail size is 400; exclusions can reduce the eligible count. `composition.json` binds each config to its exact parent and pilot cohorts, audit copies and full Parquet SHA256. The deleted parent repositories are not needed to freeze or load these materialized snapshots. Older filtered-view revisions are deliberately rejected by the new freeze path.

Use a new KLD study with the existing collect500 profile, `--tail-400`, and the exact collect400 Hub commit from the handover dataset lock. `--size 500` selects the parent scoring protocol; it does not collect 500 rows. First run the complete data validation without starting the scorer:

```bash
.venv/bin/python cli/collect_model_kld.py \
  --study /path/to/new-tail400-study --size 500 --tail-400 \
  --reference-revision 00700fec8cbdfbcad8406e8911c54f6b6284426c \
  --profiles profiles/snr-collect500.json \
  --model qwen3.6-35b-a3b --source mmstar --mode instruct \
  --candidate Q4_K_M --cand-model /models/Q4_K_M.gguf \
  --models-dir /models --metric-threads 12 \
  --llama-vlm-kld /path/to/accepted/llama-vlm-kld --gpu 0 \
  --freeze-only
```

After model-file and execution-environment preflight, repeat the same command without `--freeze-only` to collect. The example revision applies only to Qwen3.6; use the corresponding model's locked revision for other repositories. `--reference-cache-dir` optionally selects the Hub cache. The model/scorer arguments remain required with `--freeze-only`, but their files are not loaded by that mode. `--dry-run` only displays the command and does not validate or freeze data.

The launcher downloads composition/audits at the exact commit, verifies the full Parquet SHA256, validates native row/image/token contracts and the exact eligible-ID order, and saves a local dataset in `references/<model>/<config>/dataset`. Its `freeze-receipt.json` includes the revision, profile hash, source identities and hashes of all frozen files. Each collection also gets a copy as `reference-freeze.json`. Repeat invocations recheck the complete frozen files and rows before reuse; they reject a changed revision/profile or incomplete freeze. Preserve a failed freeze for diagnosis and use a fresh study path. Do not edit frozen files or bypass validation with a local `--dataset` argument in tail mode.

`--metric-threads 12` changes only CPU metric workers; decoder threads remain 8. The override is stored in the study plan and overview, so repeat it on every command for that study. Omitting it keeps the profile default of 8. Changing it later requires a new study. The native CPU test compares all record bytes for 1, 4, 8 and 12 workers; this establishes worker-count invariance for the same input logits and execution environment, not a throughput gain or equivalence between different machines.

Keep the pilot's model/projector files, scorer binary, loaded libraries and decoder settings identical. By default the comparison also requires identical GPU UUIDs. An archived old study executes its original scripts; create a new study to use the new tail option.

To collect the tail on another machine, preserve the original absolute reference model/projector paths, `CUDA_CACHE_PATH`, `LD_LIBRARY_PATH`, device selectors and every other recorded execution environment value. The full loaded-library list and SHA256 values must match, including the system loader, libc, libcuda, CUDA, cuBLAS and NCCL; copying only the scorer executable is insufficient. The new machine must record exactly one GPU with the same model name and driver version. Multiple recorded GPUs, unknown GPU identities and driver or library changes are rejected.

For an approved transfer meeting these conditions, add `--cross-part-execution-policy same-gpu-model-v1` to the comparison command below. This permits only a different physical GPU UUID between the pilot and tail scorer identities. Every A/B pair within each part still requires the complete original execution identity and bit-identical reference columns. Reference paths, all four model/projector fingerprints, decoder settings and environment values are not relaxed. The report records the selected policy, a warning and all four original metadata records; the source collections and generator provenance are never rewritten.

The same opt-in permits different positive `metric_threads` values across parts, for example pilot 8 and tail 12. This worker count only distributes independent token metric records; each token keeps its original vocabulary reduction order. The decoder's `n_threads`, batches and Flash Attention setting still must match. Within-part A/B metric worker counts and collection resume identity remain strict. Automatic or nonpositive metric worker counts are rejected by this policy.

This opt-in does not establish numerical or bitwise equivalence across physical GPUs, and the legacy identity does not record all host hardware details. GPU model/driver equality is a bounded compatibility check, not a hardware equivalence proof. Any broader hardware, path or runtime change needs a separately reviewed policy before collecting data for this composition.

Compare a pair by supplying its two pilot collections and two tail collections:

```bash
.venv/bin/python stats/cli/saved_metrics_paired_compare.py \
  --pilot-candidate-a /pilot/A --pilot-candidate-b /pilot/B \
  --candidate-a /tail400/A --candidate-b /tail400/B \
  --num-eval-tokens 1024 \
  --out /reports/A-vs-B.txt --output-json /reports/A-vs-B.json
```

Use a common prefix1024 for the current instruct references, or4096 for thinking.
Pilot generation limits are2048/8192; parent500 limits are1024/4096. The tool
reaggregates stored per-token values at the explicit comparison prefix, checking
that both collections actually retained that many positions when available.
Naturally shorter answers stay shorter. Existing KLD files and reference
trajectories are read without modification. Report paths must be outside all
four collection directories.

The tool verifies completed collection manifests, native generator-sidecar
hashes, per-row settings, source revisions, exact nested requested IDs, and
complete eligible coverage. Common budget skips are reported separately.
Missing eligible metrics, inconsistent candidate identities, overlapping source
IDs and reference drift fail before inference. The combined item count is at
most500; it is not padded to500 and no failed KLD row is silently excluded.

Item scores are concatenated and sent through the existing inference engine
**once**. Each retained item has equal weight; part means, CIs and p-values are
not averaged. Token-weighted results remain descriptive with no paired test.
JSON retains both datasets' metadata, generator hashes, caps and cohort counts.

The single-pair CLI assumes independent items. Disjoint IDs alone do not prove
image independence, and this feature does not add image-cluster inference or
campaign-wide multiplicity correction to that CLI. The production source audit
below found no exact repeated images within a single500 config. Broader source
dependence and selection based on observed SNR still need the campaign protocol,
appropriate clusters and held-out evaluation.

## Historical prepared-source audit, 2026-09-08

Source revision:
[`6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b`](https://huggingface.co/datasets/user-company/vlm-prepared-dataset/tree/6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b).
All14 target Parquet shards passed their full LFS SHA256 checks. Two independent
agents decoded full-resolution images, checked EXIF orientation and RGBA pixels,
and verified all seven100-row subsets equal the first100 complete500-row records.
Each individual config has unique item IDs and no exact shared-image components.
This supersedes the repeated-MMStar finding from a different, older cache revision.

Across the seven500 configs, two image pairs have identical pixels but different
encodings: MMMU-standard `test_History_8` with OCRBench-v2 `ocrv2-07473`, and
MMMU-standard `test_Electronics_93` with OCRBench-v2 `ocrv2-07294`. MMMU standard
and vision also share all500 underlying item IDs and answers despite different
image presentation. Do not treat the two representations as independent items
in a pooled campaign. Encoded-byte clustering alone misses these relationships.

The earlier inventory below the source audit covered 85 matching collect500 configs. The current materialized delivery supersedes that inventory: nine repositories, 105 tail configs, 42,000 requested and 41,704 eligible tail rows across model/source/mode configs. These counts do not mean distinct source images or completed KLD collections. Readiness of an actual KLD comparison still requires the scorer and artifact checks above.

## Historical filtered-view publisher

The existing publisher below creates the older filtered-view format, not the materialized format required by the new freeze path. It cannot recreate or update the current nine repositories after their parents were deleted. These commands are retained only to explain historical publications, and are not part of the current collection workflow:

```bash
.venv/bin/python cli/publish_collect400.py \
  --parent-repo user-company/MODEL-collect-500 \
  --revision IMMUTABLE_PARENT_COMMIT --stage /tmp/MODEL-tail400
.venv/bin/python cli/publish_collect400.py \
  --stage /tmp/MODEL-tail400 --publish
```

`HF_TOKEN` must be set in the process environment. Staging validates actual
parent Parquet IDs using projected reads, and publication uses server-side
copies. Before publication, the exact file allowlist, native audits, source
hashes and filters are rechecked. After publication, copied hashes and every
config's logical IDs are verified through `load_dataset`. The sibling receipt
records the destination commit and verification status. The publisher refuses
an existing destination; adding newly completed configs later requires a
separately reviewed update rather than overwriting this snapshot.
