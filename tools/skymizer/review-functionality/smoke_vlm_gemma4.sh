#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# smoke_vlm_gemma4.sh — end-to-end VLM-lane GPU smoke test on the gemma-4
# model family (registered in prep_vlm_score_from_hf.MODEL_FAMILIES; see
# gemma4-family-support-spec.md for the verified wrapper facts).
#
# Real dataset rows only: the default dataset is the GAP collection's
# gemma-4 ground-truth config — no locally generated look-alike rows.
#
# Pipeline exercised (small scale: END rows, N_EVAL_TOKENS cap):
#   0. llama-vlm-kld --self-test        (metric kernel, no models)
#   1. collect_kld.py  ref-vs-cand  ->  OUT_A (prep + dual-model metrics)
#   2. collect_kld.py  ref-vs-cand  ->  OUT_B (independent re-collection)
#   3. determinism: the two collections' metric columns must be identical
#      (same build, same GPU, n_seq_max=1 — the same-build double-run gate)
#   4. saved_metrics_paired_compare.py OUT_A vs OUT_B:
#        reference-column bit-identity check passes on real re-collected
#        data; every A-B delta must be EXACTLY 0. Exercises loaders,
#        alignment guards, and the statistics engine.
#
# Any other registered family runs the same smoke by overriding the model
# and dataset variables, e.g. Qwen3-VL:
#
#   MODEL_DIR=/home/ubuntu/projects/models/qwen3-vl-4b-instruct \
#   REF_MODEL=$MODEL_DIR/Qwen3VL-4B-Instruct-F16.gguf \
#   CAND_MODEL=$MODEL_DIR/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
#   MMPROJ=$MODEL_DIR/mmproj-Qwen3VL-4B-Instruct-F16.gguf \
#   DATASET=skymizer/ground-truth-mmmu-pro-vision-sampling-500 \
#   SUBSET=Qwen3-VL-4B-Instruct \
#   OUT_ROOT=outputs/smoke-qwen3vl4b \
#     ./review-functionality/smoke_vlm_gemma4.sh
# =============================================================================

SKYMIZER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SKYMIZER_DIR"

PYTHON=${PYTHON:-../../.venv/bin/python}
VLM_KLD_BIN=${VLM_KLD_BIN:-../../build/bin/llama-vlm-kld}

MODEL_DIR=${MODEL_DIR:-/home/ubuntu/models/gemma-4-e4b-it}
REF_MODEL=${REF_MODEL:-$MODEL_DIR/gemma-4-E4B-it-BF16.gguf}
CAND_MODEL=${CAND_MODEL:-$MODEL_DIR/gemma-4-E4B-it-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}       # shared by ref and cand
LABEL=${LABEL:-Q4KM}

DATASET=${DATASET:-elichen-skymizer/GAP-mmmu-pro-standard-10}
SUBSET=${SUBSET:-gemma-4-e4b-it-ins-gen-2048-vt280}
SPLIT=${SPLIT:-train}

END=${END:-3}                        # rows 0..END-1 of the num_images-sorted view
N_EVAL_TOKENS=${N_EVAL_TOKENS:-128}
TF_CHUNK=${TF_CHUNK:-16}
N_CTX=${N_CTX:-16384}

OUT_ROOT=${OUT_ROOT:-outputs/smoke-gemma4}
OUT_A=$OUT_ROOT/kld-a
OUT_B=$OUT_ROOT/kld-b
# This smoke owns $OUT_ROOT: start from scratch so a second invocation is not
# refused by the collector's collision scan on its OWN previous output. (Set
# KEEP_OUT=1 to inspect a previous run instead.)
if [ "${KEEP_OUT:-0}" != "1" ]; then
    rm -rf "$OUT_A" "$OUT_B"
fi

COMMON=(--ref-model "$REF_MODEL" --ref-mmproj "$MMPROJ"
        --cand-model "$CAND_MODEL" --cand-mmproj "$MMPROJ"
        --dataset "$DATASET" --subset "$SUBSET" --split "$SPLIT"
        --end "$END" --num-eval-tokens "$N_EVAL_TOKENS" --tf-chunk "$TF_CHUNK"
        --image-min-tokens -1 --image-max-tokens -1 --n-ctx "$N_CTX"
        --llama-vlm-kld "$VLM_KLD_BIN")

echo "=== [0/4] vlm-kld --self-test"
"$VLM_KLD_BIN" --self-test

echo "=== [1/4] on-the-fly KLD (collection A): ref vs cand"
"$PYTHON" cli/collect_kld.py --out "$OUT_A" "${COMMON[@]}"

echo "=== [2/4] on-the-fly KLD (collection B): independent re-collection"
"$PYTHON" cli/collect_kld.py --out "$OUT_B" "${COMMON[@]}"

echo "=== [3/4] determinism: metric columns identical across the two runs"
for a in "$OUT_A"/metrics/*.npz; do
    b=$OUT_B/metrics/$(basename "$a")
    "$PYTHON" - "$a" "$b" <<'EOF'
import sys
import numpy as np
a, b = sys.argv[1], sys.argv[2]
za, zb = np.load(a), np.load(b)
assert sorted(za.files) == sorted(zb.files), (a, b, "member sets differ")
for k in za.files:
    if not np.array_equal(za[k], zb[k]):
        sys.exit(f"FAIL: {a} vs {b}: member {k} differs")
print(f"  {a.split('/')[-1]}: identical")
EOF
done

echo "=== [4/4] saved_metrics_paired_compare A vs B (deltas must be exactly 0)"
"$PYTHON" cli/saved_metrics_paired_compare.py \
    --candidate-a "$OUT_A" --candidate-b "$OUT_B" \
    --label-a "$LABEL" --label-b "${LABEL}-dup" \
    --out "$OUT_ROOT/report-kld-selfpair.md" \
    --output-json "$OUT_ROOT/report-kld-selfpair.json"

echo
echo "=== smoke test complete; artifacts under $OUT_ROOT"
du -sh "$OUT_A" "$OUT_B" 2>/dev/null || true
