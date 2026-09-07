# Functional and numerical validation

These scripts check implementation behavior. They are separate from the numbered data-collection workflow. Use the [shared setup](../docs/workflows.md); keep test outputs on the selected work volume and preserve evidence from failed runs.

| Script | Requirement and scope |
| --- | --- |
| `smoke_kld.py` | GPU and real frozen rows/models. Collects candidates A/B and an independent A repeat, checks current VLMK records, exact repeat metrics, strict pairing and zero self-pair deltas. `--bin-dir` selects the build; `--verify-only` checks an existing smoke without rescoring. |
| `check_manifest_mode.py` | GPU, models and existing preparation directories. Compares single-row and manifest execution, row order/KV isolation and partial-failure behavior at one sequence. |
| `measure_ci_coverage.py` | CPU Monte Carlo study for CI methods under its stated skewed null generator; this does not establish coverage for arbitrary real datasets. |
| `prepare_text_smoke.py` | Legacy HF reference fixture helper. Extracts answer continuations for ordinary LLM-lane functionality, not the classic corpus bridge. |
| `smoke_vlm_gemma4.sh` | Historical legacy HF/GAP double-collection smoke with explicit model/dataset overrides. Requires the `hf-tokenizer` extra. |
| `smoke_max_total_tokens.sh` | Historical Qwen3-VL/GAP budget/collision check tied to its specific eight-row fixture and one expected skipped item. It is not a generic native-reference acceptance test. |

Current CPU regression checks use the existing test suite:

```bash
CUDA_VISIBLE_DEVICES='' SKYMIZER_TEST_BIN="$SKYMIZER_WORK/build/bin" \
  "$SKYMIZER_PYTHON" -m pytest tools/skymizer/tests \
  -q -p no:cacheprovider -m 'not external_model' --basetemp "$SKYMIZER_WORK/pytest-validation"
```

Pytest clears `--basetemp`; use a dedicated directory. Native integration checks need the corresponding binaries and otherwise skip. The metric scorer self-tests do not require models. Preserve the [numerical contract](../NUMERICAL_CONTRACT.md) during source moves.

A current native VLM smoke uses explicit inputs and a fresh output:

```bash
"$SKYMIZER_PYTHON" tools/skymizer/verify_and_validation_scripts/smoke_kld.py \
  --bin-dir "$SKYMIZER_WORK/build/bin" --lane vlm \
  --ref-model "$VLM_REF_MODEL" --cand-a-model "$VLM_CAND_A_MODEL" \
  --cand-b-model "$VLM_CAND_B_MODEL" --mmproj "$VLM_REF_MMPROJ" \
  --dataset "$VLM_DATASET" --subset '' --rows 3 --num-eval-tokens 64 \
  --out "$SKYMIZER_WORK/native-vlm-smoke"
```

The smoke defaults to context32768, batch2048, microbatch512 and teacher-forcing chunk2048. Its `--n-ctx`, `--n-batch`, `--n-ubatch` and `--tf-chunk` options select another reviewed runtime; use microbatch2048 for the accepted Gemma31B profile. Check capacity against the chosen model/image layout first. Omit image-token overrides for native references so the collector preserves their recorded policy. For a classic text bridge smoke use `TEXT_CHUNKS=2` with `scripts/04_collect_text_bridge.sh`, then inspect its verification JSON. Do not use article/block aggregation on a partial corpus.

The two historical shell smokes remove their own candidate-output subdirectories by default; `KEEP_OUT=1` preserves them. Give them a dedicated disposable output root. Their old dataset names, image policy and fixture assumptions must be verified against a pinned input before use. Prefer `smoke_kld.py` for a new native-reference repeat check. The old port acceptance used VLMK v2; current producers write v5, and its old test count is not a current acceptance result.
