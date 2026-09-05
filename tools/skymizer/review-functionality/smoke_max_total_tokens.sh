#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# smoke_max_total_tokens.sh — functional validation of --max-total-tokens
# (shared helper collect_common.max_total_tokens_skip_info, semantics:
# needed = n_prefill_tokens + min(generated_tokens_len, num_eval_tokens);
# skip when needed > cap), driven through the KLD collector.
#
# Setup (Qwen3-VL-4B, num_images-sorted dataset, K=1024, cap=2048, END=8):
#   rows 0-5, 7: needed 1149..1831  -> scored (n_eval = min(n_answer, 1024))
#   row 6 (test_Finance_304): needed = 4150 + 1024 = 5174 -> SKIP_OVER_BUDGET
#
# Checks:
#   [1] both KLD collections record exactly one SKIP_OVER_BUDGET row (row 6)
#       with the needed/cap numbers in the manifest, and never prep/score it
#   [2] scored rows' n_eval == min(n_answer, 1024) in the manifest
#   [3] collision refusal: an identical re-run over the same window is
#       REFUSED before prep/scoring (every row already has output), exits
#       non-zero, and leaves manifest.csv + artifacts byte-identical
#   [4] shard identity: a DISJOINT window with --max-total-tokens 4096 on the
#       same dir is REFUSED (identity fields), non-zero exit, named message —
#       tested on a disjoint range so a row collision cannot mask it
#   [5] compare guard: saved_metrics_paired_compare against a dir collected
#       WITHOUT the cap (UNCAPPED_KLD, if present) hard-fails meta alignment
#   [6] positive path: the two capped dirs (same reference) compare
#       successfully over exactly the 7 non-skipped items
# =============================================================================

SKYMIZER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SKYMIZER_DIR"

PYTHON=${PYTHON:-../../.venv/bin/python}
VLM_KLD_BIN=${VLM_KLD_BIN:-../../build/bin/llama-vlm-kld}

MODEL_DIR=${MODEL_DIR:-/home/ubuntu/projects/models/qwen3-vl-4b-instruct}
REF_MODEL=${REF_MODEL:-$MODEL_DIR/Qwen3VL-4B-Instruct-F16.gguf}
CAND_MODEL=${CAND_MODEL:-$MODEL_DIR/Qwen3VL-4B-Instruct-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-$MODEL_DIR/mmproj-Qwen3VL-4B-Instruct-F16.gguf}

DATASET=${DATASET:-skymizer/ground-truth-mmmu-pro-vision-sampling-500}
SUBSET=${SUBSET:-Qwen3-VL-4B-Instruct}
SPLIT=${SPLIT:-train}

END=${END:-8}
K=${K:-1024}
MTT=${MTT:-2048}
TF_CHUNK=${TF_CHUNK:-16}
N_CTX=${N_CTX:-16384}

OUT_ROOT=${OUT_ROOT:-outputs/smoke-qwen3vl4b-mtt}
OUT_A=$OUT_ROOT/kld-a
OUT_B=$OUT_ROOT/kld-b
# This smoke owns $OUT_ROOT: start from scratch so a second invocation is not
# refused by the collectors' collision scan on its OWN previous output. (Set
# KEEP_OUT=1 to inspect a previous run instead.)
if [ "${KEEP_OUT:-0}" != "1" ]; then
    rm -rf "$OUT_A" "$OUT_B"
fi
UNCAPPED_KLD=${UNCAPPED_KLD:-}

COMMON=(--ref-model "$REF_MODEL" --ref-mmproj "$MMPROJ"
        --cand-model "$CAND_MODEL" --cand-mmproj "$MMPROJ"
        --dataset "$DATASET" --subset "$SUBSET" --split "$SPLIT"
        --end "$END" --num-eval-tokens "$K" --max-total-tokens "$MTT"
        --tf-chunk "$TF_CHUNK" --image-min-tokens -1 --image-max-tokens -1
        --n-ctx "$N_CTX"
        --llama-vlm-kld "$VLM_KLD_BIN")

echo "=== [1a] KLD collection A (K=$K, max-total-tokens=$MTT, end=$END)"
"$PYTHON" cli/collect_kld.py --out "$OUT_A" "${COMMON[@]}"

echo "=== [1b] KLD collection B (same pair, second dir for the paired test)"
"$PYTHON" cli/collect_kld.py --out "$OUT_B" "${COMMON[@]}"

echo "=== [1c] manifest skip/score assertions"
for d in "$OUT_A" "$OUT_B"; do
    n_skip=$(grep -c "SKIP_OVER_BUDGET" "$d/manifest.csv" || true)
    [ "$n_skip" -eq 1 ] || { echo "FAIL: $d expected 1 SKIP row, got $n_skip"; exit 1; }
    grep "SKIP_OVER_BUDGET" "$d/manifest.csv" | grep -q "test_Finance_304" \
        || { echo "FAIL: $d skip row is not test_Finance_304"; exit 1; }
    n_ok=$(tr -d '\r' < "$d/manifest.csv" | grep -c ",OK$" || true)  # csv rows end \r\n
    [ "$n_ok" -eq 7 ] || { echo "FAIL: $d expected 7 OK rows, got $n_ok"; exit 1; }
    ls "$d"/metrics/ | grep -q Finance && { echo "FAIL: skipped row has artifacts"; exit 1; }
    echo "  $d: 7 OK + 1 SKIP_OVER_BUDGET(test_Finance_304), no artifacts for the skip — pass"
done
echo "--- skip row (A):"; grep SKIP "$OUT_A/manifest.csv"
echo "--- n_eval column (A):"; cut -d, -f2,5,6 "$OUT_A/manifest.csv"

echo "=== [3] collision refusal: identical re-run must be refused, nothing changed"
snapshot() {   # path, md5, size, ns-mtime of every file except the lock
    (cd "$1" && find . -type f ! -name '.collect.lock' -print0 \
        | sort -z | xargs -0 -I{} sh -c 'printf "%s %s %s\n" "{}" "$(md5sum "{}" | cut -d" " -f1)" "$(stat -c "%s %.Y" "{}")"')
}
sums_before=$(snapshot "$OUT_B")
set +e
out=$("$PYTHON" cli/collect_kld.py --out "$OUT_B" "${COMMON[@]}" 2>&1)
rc=$?
set -e
[ $rc -ne 0 ] || { echo "FAIL: same-window re-run was accepted"; exit 1; }
echo "$out" | grep -q "collision for requested rows" \
    || { echo "FAIL: refusal is not a collision report"; echo "$out" | tail -5; exit 1; }
sums_after=$(snapshot "$OUT_B")
[ "$sums_before" = "$sums_after" ] \
    || { echo "FAIL: files changed (bytes/size/mtime) on a refused re-run"; exit 1; }
echo "  refused (exit $rc), manifest + artifacts byte- and mtime-identical — pass"

echo "=== [4] shard identity: changed --max-total-tokens on a DISJOINT window must be refused"
set +e
out=$("$PYTHON" cli/collect_kld.py --out "$OUT_B" \
    --ref-model "$REF_MODEL" --ref-mmproj "$MMPROJ" \
    --cand-model "$CAND_MODEL" --cand-mmproj "$MMPROJ" \
    --dataset "$DATASET" --subset "$SUBSET" --split "$SPLIT" \
    --start "$END" --end $((END + 1)) --num-eval-tokens "$K" --max-total-tokens 4096 \
    --tf-chunk "$TF_CHUNK" --image-min-tokens -1 --image-max-tokens -1 \
    --n-ctx "$N_CTX" \
    --llama-vlm-kld "$VLM_KLD_BIN" 2>&1)
rc=$?
set -e
[ $rc -ne 0 ] || { echo "FAIL: changed cap was accepted on a new shard"; exit 1; }
echo "$out" | grep -qi "max_total_tokens" \
    || { echo "FAIL: refusal does not name max_total_tokens"; echo "$out" | tail -5; exit 1; }
echo "  refused (exit $rc), message names max_total_tokens — pass:"
echo "$out" | grep -i "max_total_tokens" | head -2

echo "=== [5] compare guard: capped dir vs uncapped dir must hard-fail"
if [ -n "$UNCAPPED_KLD" ] && [ -d "$UNCAPPED_KLD" ]; then
    set +e
    out=$("$PYTHON" cli/saved_metrics_paired_compare.py \
        --candidate-a "$UNCAPPED_KLD" --candidate-b "$OUT_B" \
        --out "$OUT_ROOT/report-should-not-exist.md" 2>&1)
    rc=$?
    set -e
    [ $rc -ne 0 ] || { echo "FAIL: cross-cap compare was accepted"; exit 1; }
    echo "  refused (exit $rc) — pass:"
    echo "$out" | tail -3
else
    echo "  (UNCAPPED_KLD not set/missing — point it at a no-cap collect_kld dir to run this check)"
fi

echo "=== [6] positive path: capped pair compares exactly the 7 kept items"
"$PYTHON" cli/saved_metrics_paired_compare.py \
    --candidate-a "$OUT_A" --candidate-b "$OUT_B" \
    --label-a Q4KM --label-b Q4KM-dup \
    --out "$OUT_ROOT/report-selfpair.md" \
    --output-json "$OUT_ROOT/report-selfpair.json"
grep -q "Items compared: 7" "$OUT_ROOT/report-selfpair.md" \
    || { echo "FAIL: expected 'Items compared: 7'"; grep "Items compared" "$OUT_ROOT/report-selfpair.md"; exit 1; }
echo "  Items compared: 7 — pass"

echo
echo "=== all --max-total-tokens checks PASS; artifacts under $OUT_ROOT"
du -sh "$OUT_A" "$OUT_B" 2>/dev/null || true
