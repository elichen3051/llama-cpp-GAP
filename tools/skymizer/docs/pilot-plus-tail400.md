# Reuse pilot KLD and collect the remaining400 items

The `*-collect-400` reference repositories contain configs named
`<source>-tail-400-ins` or `<source>-tail-400-think`. Each selects the requested
positions101..500 of the prepared500 pool by stable item ID. Generation
exclusions remain exclusions; the logical dataset can contain fewer than400
rows. Pilot exclusions remain authoritative for positions1..100.

These are standard Hugging Face **filtered views**. Load them using
`datasets.load_dataset(repo, config, split="train")`. Their physical Parquet
files are unchanged copies of the parent500 files; reading those files directly
bypasses the filter. Viewer is disabled because its native-Parquet shortcut
can bypass builder filters. The composition manifest pins the parent and pilot
commits and records exact requested, eligible and excluded IDs. The original
pilot and collect500 repositories remain unchanged.

Use a new KLD study with the existing collect500 profile and add `--tail-400`:

```bash
.venv/bin/python cli/collect_model_kld.py \
  --study /path/to/new-tail400-study --size 500 --tail-400 \
  --profiles profiles/small-collect500.json \
  --model qwen3.5-4b --source mmstar --mode instruct \
  --candidate Q4_K_M --cand-model /models/Q4_K_M.gguf \
  --llama-vlm-kld /path/to/llama-vlm-kld --gpu 0
```

Keep the pilot's scoring machine, model/projector files, scorer build, loaded
libraries and numerical runtime settings compatible. The comparison verifies
the recorded execution identity, including GPU identity. An archived old study
executes its original scripts; create a new study to use the new tail option.

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

## Production source and reference audit,2026-09-08

Source revision:
[`6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b`](https://huggingface.co/datasets/elichen-skymizer/vlm-prepared-dataset/tree/6cb6a4d65fcb2c8d788f68aaa0aa92c5ceca408b).
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

The pinned reference inventory contains105 pilot configs and85 matching
collect500 configs across nine model repositories. All85 satisfy the exact
requested-ID prefix rule. They provide8,440 eligible pilot items and33,712
eligible tail items in total across configs. These are reference availability
counts, not evidence that every production KLD collection has finished or passes
the scorer compatibility checks.

## Publishing another completed parent snapshot

Stage and inspect locally, then publish a new private sibling repository:

```bash
.venv/bin/python cli/publish_collect400.py \
  --parent-repo elichen-skymizer/MODEL-collect-500 \
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
