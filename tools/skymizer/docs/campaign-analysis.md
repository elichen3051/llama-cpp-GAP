# Quantization campaigns and benchmark selection

`stats/cli/campaign_compare.py` operates on a declared JSON plan. It reuses completed metric collections, reference/target alignment and reader locks. Nine quantization variants produce 36 unordered paired comparisons per cell. The correction family includes every pair across all cells in the plan, including all declared datasets, instruct/thinking modes and model families. The existing two-candidate report only adjusts its own exploratory metrics; that does not provide campaign-wide control.

```sh
uv run --python 3.12 python stats/cli/campaign_compare.py analyze --manifest campaign.json --out campaign.md
uv run --python 3.12 python stats/cli/campaign_compare.py select --manifest training.json --out selection.md
```

Both actions write a companion JSON file `<out>.json` by default. `--output-json` overrides its path. The selection JSON contains the selected IDs and receipt needed for validation. The analysis JSON contains all intervals, p-values, declared inputs and observed collection provenance.

## Declared plan

This minimal schema example has two variants and three items; a real plan lists every intended variant and every collection row. It illustrates fields, not an adequate sample-size recommendation.

```json
{
  "schema_version": "skymizer-campaign-plan-v1",
  "family_id": "quantization-study-01",
  "metric": "kld",
  "weighting": "item",
  "alpha": 0.05,
  "correction": "holm",
  "sampling_plan": "fixed_n",
  "prespecification_status": "user_declared",
  "cells": [{
    "cell_id": "dataset-model-instruct",
    "dataset_id": "prepared-subset",
    "dataset_content_hash": "copy the collected dataset_content_hash",
    "mode": "instruct",
    "model_family": "family-a",
    "model_id": "model-a",
    "fixed_analysis_n": 3,
    "roster": [
      {"item_id": "item1", "item_key": "000_item1", "cluster_id": "image-1", "source_id": "source-dataset", "image_hashes": ["image-sha256-1"]},
      {"item_id": "item2", "item_key": "001_item2", "cluster_id": "image-2", "source_id": "source-dataset", "image_hashes": ["image-sha256-2"]},
      {"item_id": "item3", "item_key": "002_item3", "cluster_id": "image-3", "source_id": "source-dataset", "image_hashes": ["image-sha256-3"]}
    ],
    "variants": [
      {"quant_id": "Q3-variant", "collection_dir": "collections/q3"},
      {"quant_id": "Q4-variant", "collection_dir": "collections/q4"}
    ]
  }]
}
```

Collection paths are relative to the plan file. `item_key` identifies the stored metric artifact; `item_id` is the stable collected source identity across model families and must match the ID encoded in `item_key`; completed collection records bind that ID to the artifact. `source_id` identifies a source stratum for selection quotas. Within a cell, the roster must exactly match completed collection artifacts. Missing items, reference drift, mismatched targets, absent model/projector fingerprints and nonfinite data fail the campaign. Observed reference and candidate fingerprints are persisted separately from declared labels. Reference-equals-candidate informational warnings are allowed, but unresolved paired variance still fails inference.

An optional `analysis_item_ids` list selects a fixed, declared cohort from the complete roster. `fixed_analysis_n` must equal its length. For example, a 40-ID list can analyze a frozen subset of an existing 500-item collection. The code cannot establish that this list, correction method or sample size was chosen before looking at outcomes. A plan hash records integrity, not temporal preregistration. Repeated outcome-dependent looks at 20/30/40 items require a sequential procedure or a separately justified error budget, which this fixed-N tool does not provide.

## Verified reference rosters

Use the producer's local native `Dataset.save_to_disk` directory to avoid manually assigning 500 artifact keys and image clusters. The first command works while KLD collection is still running; it exports an unbound identity audit without artifact keys. The second requires completed successful collection records and emits `cell_fields` containing the verified `roster`, `dataset_content_hash` and reference `mode` for a campaign cell.

```sh
uv run --python 3.12 python stats/cli/reference_roster.py --reference /study/reference/dataset --out identities.json
uv run --python 3.12 python stats/cli/reference_roster.py --reference /study/reference/dataset --collection /study/collections/q3 --out bound-roster.json
```

The output must be a new file. `--split` selects a saved DatasetDict split. This command reads local native v2 references; it does not download Hub datasets. `--source-field` defaults to the native `source` quota stratum. The tool validates the native row contract, binds the mode to request/default fields covered by the dataset hash, and recomputes SHA256 from each encoded image, then uses SciPy connected components for transitive multi-image links. Repeated image slots within one item do not create extra samples. Image-free rows remain separate components. Counts and cluster membership are included in the audit.

Binding reproduces the collector's recorded sort, compares the full ordered dataset hash, acquires a reader lock, and joins completed manifest rows by both index and stable ID. Partial collections require `--expected-item-ids expected.json`, a JSON list declaring their exact expected cohort. The full reference must still be supplied for its content hash. Missing, failed, skipped or actively collected rows cannot silently reduce that cohort. Copy `cell_fields` into the intended campaign cell, retain the audit, and declare the analysis IDs, fixed sample size, quantization variants and model-family labels separately. The campaign loader still verifies metric contents and all candidate alignments.

Encoded-byte matching does not detect recompressed, cropped or related source images. Additional known dependencies require broader cluster declarations. Cluster IDs are stable across row permutations of the same full pool, but component membership can change when the pool changes; do not treat the nested 100 and 500 pools as separate independent cohorts. The current training selector requires one item per cluster and rejects repeated-image pools; whole-cluster selection with exact size and source quotas requires a separate algorithm.

## Inference and interpretation

The estimand is the equal-item mean of `KLD_B - KLD_A`. Token-weighted inference is rejected. Items sharing an image or other dependent source belong to the same cluster. For multi-image items, use connected components of shared image hashes. The validator rejects splitting a declared shared image across clusters. Hashes and cluster assignments in the plan remain user declarations; stored metric alignment alone cannot establish image independence. Absence of exact image duplicates also does not establish independent source content.

An intercept-only statsmodels OLS fit supplies CRV1 clustered covariance with both small-sample covariance correction and `G-1` inference degrees of freedom. This preserves equal item weights even when images have different question counts. For item differences `d_i`, mean `theta`, `N` items and `G` clusters:

```text
U_g = sum(d_i - theta for items in cluster g)
SE^2 = G/(G-1) * sum(U_g^2) / N^2
```

With one independent item per cluster this reduces to the ordinary paired-t SE. With repeated-image items, the t reference is a cluster-robust approximation; few or highly unequal clusters can be poorly calibrated. Reports expose both `N` and `G`. No p-value correction repairs invalid marginal inference.

Choose the correction before inspecting outcomes:

| Method | Target | Dependence requirement |
|---|---|---|
| `holm` (default) | Strong family-wise error rate: probability of any false equality rejection in the declared family | Arbitrary dependence between valid p-values |
| `fdr_bh` | Expected false-discovery proportion among rejections | Independence or PRDS; declare `dependence_assumption: independent_or_prds` |
| `fdr_by` | Expected false-discovery proportion among rejections | Arbitrary dependence between valid p-values |

All three call `statsmodels.stats.multitest.multipletests` with an explicit method. BH's PRDS assumption is not established by observing that quantization comparisons are correlated. FDR does not bound the false-positive count conditional on a particular realized discovery list. The application uses strict `p_adjusted < alpha` to retain its conservative boundary convention.

Pointwise intervals and global Bonferroni simultaneous intervals are separate fields. The descriptive minimum-KLD variant is an observed ranking. A possible-best set removes variants shown worse by the simultaneous intervals; a singleton establishes a unique best only under that interval family's assumptions. Those intervals use the total campaign pair count even when the chosen p-value correction is BH/BY. An empty set can occur if incompatible directions are observed; it is not a best-candidate claim.

`observed_least_resolved_pair` identifies the largest adjusted p-value in the observed cell. It does not identify the population's hardest pair. `all_pairs_rejected_under_selected_correction` records a result at the declared fixed N, under the selected error-rate target. Neither field estimates prospective power, proves a minimum N, or establishes equivalence. Practical equivalence requires a prespecified margin and its own inferential accounting.

## Training-only subset selection

Add `selection_rule` to a complete training plan:

```json
{
  "algorithm": "greedy_backward_maximin_snr_v1",
  "size": 250,
  "training_families": ["family-a", "family-b"],
  "heldout_families": ["family-c", "family-d"],
  "source_min_counts": {"source-dataset": 100}
}
```

Select instruct and thinking benchmarks separately. Training cells must have the same complete source/image pool identities. Generated-reference dataset hashes can differ across model families, so the selector aligns stable source IDs and image hashes instead. This selector currently requires one item per independent image/source cluster; it rejects a repeated-cluster pool. Supporting a clustered selector with an exact item budget requires an explicit whole-cluster design and is not silently approximated by treating questions independently.

The deterministic backward search removes one item at a time to maximize the minimum `abs(mean(delta))/sample_SD(delta)` across all training pair tasks. The absolute value is taken after averaging. Source minima constrain deletion. SNR is a property of a subset's difference distribution, not a score for one item. A subset's own numerical scale determines whether its variance is resolved. The procedure is a local greedy search, not a global optimum. Its training objective is optimistic after optimization and carries no p-value or detection guarantee.

The receipt records the complete rule, candidate count, source/image pool hash, selected IDs and identity hashes, observed training metadata/reference identities, training score hashes, and training/heldout families. To validate it, attach the receipt as `selection_receipt` to a new analysis plan. Include all declared heldout families, use the same mode and frozen selected IDs, and provide distinct observed collection provenance. The tool rejects reused paths, copied metadata or shared reference-model identities. It accepts either the full original pool with the fixed selected-ID list or collections containing exactly the selected benchmark items. Heldout models therefore need not collect all 500 items after a 250-item benchmark is frozen.

Family/model/mode labels remain user declared; a receipt cannot prove a taxonomy or that heldout outcomes were never inspected. Validation on new families using the same selected image pool assesses family transfer conditional on that pool. Generalization beyond it needs fresh image/source clusters. Nested 100/500 pools are not independent validation cohorts.

## Methodological sources and package correspondence

The implementation uses original methods as its statistical basis:

- Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*: [original paper](https://www.jstor.org/stable/4615733). Strong FWER and simultaneous error accounting.
- Benjamini and Hochberg (1995), *Controlling the False Discovery Rate*: [original paper](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
- Benjamini and Yekutieli (2001), *The control of the false discovery rate in multiple testing under dependency*: [original paper](https://doi.org/10.1214/aos/1013699998). PRDS and arbitrary-dependence correction.
- Cameron and Miller (2015), *A Practitioner's Guide to Cluster-Robust Inference*: [paper](https://doi.org/10.3368/jhr.50.2.317), especially the covariance correction and small-cluster limitations.
- Cawley and Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*: [original paper](https://www.jmlr.org/papers/v11/cawley10a.html).
- Johari et al. (2022), *Always Valid Inference: Continuous Monitoring of A/B Tests*: [original paper](https://doi.org/10.1287/opre.2021.2135). This tool implements fixed-N inference, not their sequential method.

Runtime APIs and method references are also inside function docstrings. [statsmodels multipletests](https://www.statsmodels.org/v0.14.6/generated/statsmodels.stats.multitest.multipletests.html) implements Holm/BH/BY. [SciPy false_discovery_control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html) supplies an independent BH/BY numerical cross-check. [MATLAB mafdr](https://www.mathworks.com/help/bioinfo/ref/mafdr.html) corresponds to BH only with `BHFDR=true`; its default is a different procedure. [MultipleTesting.jl](https://juliangehring.github.io/MultipleTesting.jl/stable/adjustment/) provides the three corresponding adjustments. No MATLAB or Julia runtime was executed and no MATLAB source was ported.
