# Prepared-dataset image clusters

CPU-only audit of embedded images in `elichen-skymizer/vlm-prepared-dataset` or another repository with the same prepared schema. It records exact RGB image reuse, full ordered/unordered image lists, repeated image-and-question inputs, connected row clusters, categories, and nonexact equal-pHash candidates. No model is loaded and no dataset is modified.

From the llama.cpp repository root:

```bash
uv pip install --python .venv/bin/python -r tools/skymizer/scripts/image_clusters/requirements.txt
.venv/bin/python tools/skymizer/scripts/image_clusters/scan.py \
  --dataset elichen-skymizer/vlm-prepared-dataset \
  --revision c3f0e320b7f0360a3a10ee4c74aca0b93d697dc5 \
  --out ~/image-cluster-audit
```

The default selects `*-subsample-100` and `*-subsample-500`, split `train`, with two CPU workers. Use `--workers 1` to reduce CPU and storage pressure. Without `--revision`, the current Hub revision is resolved to a commit before any download. Existing Hugging Face cache files are reused and their complete SHA256 hashes are checked. For Hub files with an official LFS digest, the full SHA256 is compared with it. For non-LFS Hub files, the tool records a digest of the pinned download and explicitly marks that weaker integrity basis in metadata; later offline scans must match the recorded digest. Inputs must use the layout `<config>/<split>-*.parquet`; shards within a config are read in filename order.

Select particular subsets by repeating a glob:

```bash
.venv/bin/python tools/skymizer/scripts/image_clusters/scan.py \
  --config 'ocrbench-v1-subsample-*' \
  --config 'ocrbench-v2-subsample-*' \
  --out ~/ocr-image-clusters
```

To rerun using local Parquet files already recorded in a scan, without Hub access:

```bash
.venv/bin/python tools/skymizer/scripts/image_clusters/scan.py \
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
.venv/bin/python -m unittest discover \
  -s tools/skymizer/scripts/image_clusters -p test_scan.py -v
```
