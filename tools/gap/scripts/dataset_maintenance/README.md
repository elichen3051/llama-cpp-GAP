# Prepared-dataset image clusters

CPU-only audit of embedded images in `user-company/vlm-prepared-dataset` or another repository with the same prepared schema. It records exact RGB image reuse, full ordered/unordered image lists, repeated image-and-question inputs, connected row clusters, categories, and nonexact equal-pHash candidates. The scanner loads no model and modifies no dataset. The separately reviewed replacement workflow below can update sampled subsets.

This optional audit/preparation workflow is separate from the numbered evaluation stages. Use the [shared setup](../../docs/workflows.md), then run from the llama.cpp repository root:

```bash
uv venv "$COMPANY_WORK/dataset-maintenance-venv" --python 3.12.3
export MAINTENANCE_PYTHON="$COMPANY_WORK/dataset-maintenance-venv/bin/python"
uv pip install --python "$MAINTENANCE_PYTHON" -r tools/gap/scripts/dataset_maintenance/requirements.txt
"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/scan.py \
  --dataset user-company/vlm-prepared-dataset \
  --revision c3f0e320b7f0360a3a10ee4c74aca0b93d697dc5 \
  --out ~/image-cluster-audit
```

The optional audit environment is separate from the locked collection environment. Its requirements pin the tested direct libraries, including ImageHash and PyWavelets; each audit also records its actual library versions.

The default selects `*-subsample-100` and `*-subsample-500`, split `train`, with two CPU workers. Use `--workers 1` to reduce CPU and storage pressure. Without `--revision`, the current Hub revision is resolved to a commit before any download. Existing Hugging Face cache files are reused and their complete SHA256 hashes are checked. For Hub files with an official LFS digest, the full SHA256 is compared with it. For non-LFS Hub files, the tool records a digest of the pinned download and explicitly marks that weaker integrity basis in metadata; later offline scans must match the recorded digest. Inputs must use the layout `<config>/<split>-*.parquet`; shards within a config are read in filename order.

Select particular subsets by repeating a glob:

```bash
"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/scan.py \
  --config 'ocrbench-v1-subsample-*' \
  --config 'ocrbench-v2-subsample-*' \
  --out ~/ocr-image-clusters
```

To rerun using local Parquet files already recorded in a scan, without Hub access:

```bash
"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/scan.py \
  --input-manifest ~/image-cluster-audit/metadata.json \
  --out ~/image-cluster-audit-rerun
```

`--input-manifest` supplies the dataset identity, revision, and local file paths; `--dataset` and `--revision` do not override it. Offline manifests must contain a 40-hex commit, a nonnegative integer size, and at least one valid expected SHA256 for every selected file. Every selected file is checked again against these values. This also accepts the `metadata.json` produced by the original prepared-dataset image audit. The output directory must be new or empty to prevent stale results from mixing with a new run.

The prepared Parquet columns are `source`, `item_id`, `origin_id`, `question`, `images`, `num_images`, `category`, and `ref_answer`. Each element of `images` must contain embedded `bytes`. Path-only images are rejected. Empty question text is supported and does not establish question identity. Row IDs are `<config>/<split>/<zero-based-index>` across all shards of the config.

Outputs:

- `REPORT.md`: concise automatic findings and interpretation limits.
- `metadata.json`: dataset commit, selected files, sizes, SHA256, and schemas.
- `rows.jsonl`: row identifiers, questions, source metadata, image positions, encoded SHA256, RGB/RGBA pixel hashes, pHash, dimensions, and complete-image-list fingerprints.
- `row_cluster_mapping.csv`: one row per input UID, with within-config exact cluster IDs and content-deduplicated global cluster IDs. Keep the dataset revision with this file when joining KLD rows.
- `summary.json`, `per_config_summary.csv`, `category_concentration.csv`: exact image, image-list, question-input and row-group counts. `duplicate_groups.jsonl` lists group members.
- `phash_candidates.jsonl`: matching perceptual hashes with different RGB pixels; these are not automatically merged into clusters.
- `overlap_100_500.json`: content overlap, ordered-prefix verification, and later rows reusing prefix images. Expected nested sampling is separate from within-subset image reuse.
- `status.json`, `error.log` on failure, `scripts/`, `analysis_provenance.json`, and `files.json`: run state, errors, copied implementation, library versions, and output checksums. Redirect or tee stdout if a separate progress log is wanted.

Exact comparisons use dimensions and RGB pixels after EXIF correction, without resizing. Alpha is excluded because this compares RGB inputs; a separate RGBA hash preserves evidence of transparency differences. Images use the first frame and the frame count is recorded. Rows sharing several images count once when reporting involved rows; connected components handle partial and transitive overlap. No-image rows remain separate image clusters.

A pHash match can be a collision. Conversely, changed point markers, crops, rendered text, and compression can hide shared base images from exact hashing. Manually inspect candidates and validate source relationships before adding those links. In particular, MMMU-Pro Standard and Vision can share underlying content despite distinct rendered-image hashes; this tool does not silently infer that relationship. Same text with different point markers can also be a different question. The original audit's visually confirmed and source-linked cluster mapping remains a separate reviewed artifact.

Category counts use the prepared `category` field. A missing category is reported as missing, not guessed from the source name. OCRBench v1 in the audited prepared revision has an empty category; recovering its original FUNSD/DocVQA/etc. labels requires a separately validated upstream metadata join.

Cluster counts are not measured effective sample sizes. `hypothetical_icc_scenarios.csv` is only an unequal-cluster-size sensitivity calculation under stated homogeneous-variance/common-correlation assumptions. Actual KLD/SNR uncertainty needs per-question metric data; see [Miller (2024), sections 2.2 and 4.2](https://arxiv.org/pdf/2411.00640).

A decode, schema, integrity, or I/O error stops the audit with nonzero exit status and `status=failed`; incomplete outputs must not be interpreted as a completed scan. Ctrl-C records `status=interrupted`. A forced process kill or machine failure can leave `status=running`; only `complete` means the scan finished. Rerun into a new directory after resolving the cause.

Tests (no GPU or network):

```bash
"$MAINTENANCE_PYTHON" -m unittest discover \
  -s tools/gap/scripts/dataset_maintenance -p test_scan.py -v
```

## Replace sampled shared-image questions

`replace_subsamples.py` prepares changes locally, then publishes only after an independent review. It operates on existing 500-row samples and their 100-row prefixes. The source pools are unchanged. It reserves every surviving original row at its original position, then fills excluded positions with unused source-pool items. Every 100-row result is the complete first 100 rows of its 500-row result, including image bytes, order, and sample metadata.

MMMU-Pro Standard and Vision are one selection unit: a conflict in either view replaces both rows at the same position with the same source item ID. Both final 500-row views and both 100-row views must align. Expected overlap between these two views, or between a 100-row sample and its parent 500-row sample, is not removed. Deduplication is within each subset, not across unrelated datasets.

The default exclusion policy rejects cross-row exact RGB matches or 64-bit pHash Hamming distances at most 4. This is deliberately conservative: a near hash is not proof that two images are the same photograph, and distant hashes do not prove independence. Any shared image in a multi-image question can trigger exclusion; repeated images inside a single question do not. Animated and image-free inputs are rejected by this replacement workflow.

The input manifest uses the scanner's verified `selected_files` format, with `repo`, a pinned `revision`, and an explicit `source_pools` map. Each family requires its source pool plus its existing `-subsample-500` and `-subsample-100` configs. For this dataset the seven families map to their own names except `ocrbench-v2`, whose source pool is `ocrbench-v2-1000`. Do not use `mathvision-testmini` or `mathvision-test-lmms-eval` as pools.

To download and verify just these inputs, from the repository root:

```bash
"$MAINTENANCE_PYTHON" - <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, 'tools/gap/scripts/dataset_maintenance')
from scan import hub_manifest, validate_manifest, save_json
families = ['blink-val', 'mathvision-test', 'mmmu-pro-standard-10',
            'mmmu-pro-vision', 'mmstar', 'ocrbench-v1', 'ocrbench-v2']
pools = {f: ('ocrbench-v2-1000' if f == 'ocrbench-v2' else f) for f in families}
configs = sorted(set(pools.values()) | {f'{f}-subsample-{n}' for f in families for n in (100, 500)})
args = SimpleNamespace(dataset='user-company/vlm-prepared-dataset', revision='main',
                       split='train', config=configs)
metadata = validate_manifest(hub_manifest(args), configs, 'train')
metadata['source_pools'] = pools
save_json(Path('replacement-inputs.json'), metadata)
PY

"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/replace_subsamples.py prepare \
  --inputs replacement-inputs.json \
  --out replacement-draft \
  --seed 1234 --phash-distance 4 --workers 2
```

`main` is resolved once to an immutable commit before downloads. To reproduce a particular run, use its recorded parent commit and the same optional strata file. `prepare` checks full input hashes, schema metadata, original source membership, and existing prefixes before selection; its output directory must be new or empty. Candidates prefer the same category, task, upstream dataset, and number of images, with SHA256 ranking from seed, family, and item ID to break ties. For unchanged slots, all original fields and image bytes are preserved. New rows retain source content and use the requested `sample_seed`.

Use `--strata strata.json` to restore subtask labels that the prepared schema does not contain. This optional JSON must have `dataset`, the same pinned `revision`, and `rows` containing `family`, `item_id`, `origin_id`, complete `question`, `task`, and `dataset_name`. Labels must come from an independently validated join; missing labels remain unknown. Stratum fallback tiers are 0 (same labels), 1 (same task but different upstream dataset), 2 (different task within the category), and 3 (different category). The row ledger records every fallback. If the pool cannot fill a position under the image policy, preparation fails without publishing a smaller sample.

Review these outputs before publication:

- `draft-manifest.json`: final local Parquet paths, SHA256, unchanged-file references, and per-family replacement counts. This is a local draft, not an existing Hub revision.
- `replacement-ledger.json` and `*-selection.json`: old/new item IDs and origins, exact positions, conflict evidence, source hashes, strata, and rejected candidates.
- `upload/`: only changed target Parquets, updated dataset-card metadata, and a public `sampling/` recipe with the row ledger. Unchanged subsets are not rewritten.
- `publication-plan.json`: exact upload paths, sizes, SHA256, deletion paths, dataset, and required parent revision.
- `inputs.json`, optional `strata.json`, `scripts/`, and `status.json`: reproducible inputs, implementation snapshot, and run state. Keep the stdout/stderr log alongside the draft.

An independent reviewer must decode the final images again, check exact and near reuse within every subset, validate source membership and surviving positions, confirm schemas and exact 100-row prefixes, and check both MMMU pairings. Review known source-linked or manually confirmed clusters as well; hashes alone can miss modified base images. Only after passing should that reviewer produce a JSON containing `status: "approved"` and `publication_plan_sha256` equal to the SHA256 of the reviewed plan. Keep detailed evidence with it; the command validates this binding, not the reviewer's reasoning.

After the code and draft review pass and the implementation is committed:

```bash
"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/replace_subsamples.py publish \
  --out replacement-draft --review independent-review.json
```

Publication uses a single Hub commit with `parent_commit` so a concurrent upstream change aborts the write. It verifies all changed file hashes and checks that every unplanned repository file is unchanged. `publication-receipt.json` records the dataset commit, local code commit, and verification counts. `published_verified` means the remote content matches the reviewed draft. If publication succeeds but verification fails, retrying `publish` verifies the recorded commit without creating another commit.

If the request was interrupted before its result was received, the receipt keeps `publishing_outcome_not_yet_known` and prevents blind retries. Inspect the Hub commit history to find whether the reviewed update landed, then verify that exact commit:

```bash
"$MAINTENANCE_PYTHON" tools/gap/scripts/dataset_maintenance/replace_subsamples.py verify \
  --out replacement-draft --review independent-review.json --revision FULL_HUB_COMMIT
```

Keep the receipt when investigating failures. A killed prepare process can leave `preparing`; only `ready_for_independent_review` is a complete local draft. Run both scanner and replacement tests without a GPU or network:

```bash
"$MAINTENANCE_PYTHON" -m unittest discover \
  -s tools/gap/scripts/dataset_maintenance -p 'test_*.py' -v
```
