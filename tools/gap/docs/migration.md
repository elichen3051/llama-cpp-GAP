# Layout migration

This cleanup changes source and command paths. Native target names, metric records, numerical algorithms and report schemas retain their existing contracts. Keep prior campaign archives unchanged; their filenames and hashes are part of the recorded evidence.

| Previous path | Current location |
| --- | --- |
| Top-level GAP C++ sources and headers | `core/` |
| `compare/` | `stats/` |
| Statistical `cli/*.py` | `stats/cli/`; shared guarded collection loading is `stats/collection_io.py` |
| `env_setup.sh` | `scripts/setup.sh` |
| `scripts/01_save_vlm_kld.sh` | `scripts/03_collect_vlm_kld.sh` |
| `scripts/02_save_llm_kld.sh` | `scripts/collect_llm_trajectory.sh`; classic corpus work uses `scripts/04_collect_text_bridge.sh` |
| `scripts/03_paired_test_kld.sh` | `scripts/05_compare.sh` |
| `scripts/04_power_analysis.sh` | `scripts/06_power_analysis.sh` |
| `scripts/image_clusters/` | `scripts/dataset_maintenance/`, optional audit and source-data preparation |
| `review-functionality/` | `verify_and_validation_scripts/` |
| Implicit six-model reference profile | Explicit cohort/group profiles in `profiles/` |

The [workflow index](workflows.md) describes the new numbered reference, VLM and text branches. Reference publication is optional and explicitly private. The old sequential numbering did not mean that VLM and LLM collection must both run.

The dated RunPod/AWS handovers and initial upstream-port narrative were replaced by [portable reference generation](runpod-reference.md), [VLM collection](vlm-kld.md), [text corpus bridge](text-bridge.md) and [validation guidance](../verify_and_validation_scripts/README.md). Model-specific protocols and frozen corpus identities remain documented. Host-local patch routes, session chronology, external audit links and old test counts are no longer setup requirements. Historical acceptance records remain in their original archived runs and repository history; a prior short acceptance result is not a new validation of this layout.

The old port smoke asserted VLMK v2 and was incompatible with current v5 producers. Its maintained successor uses the shared current format version, explicit binary directory and original native image policy. Two fixture-specific legacy shell smokes remain labeled historical. They should not be mistaken for current native-reference acceptance procedures.
