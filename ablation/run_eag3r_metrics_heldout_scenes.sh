#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-7}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/ablation/results/eag3r_heldout_scenes}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/ablation/eag3r_eval_manifest.json}"

# Training ablations usually use initial_scene_idx=0, active_scene_count=3.
# The held-out protocol starts from scene index 3 by default.
HELDOUT_INITIAL_SCENE_IDX="${HELDOUT_INITIAL_SCENE_IDX:-3}"
HELDOUT_ACTIVE_SCENE_COUNT="${HELDOUT_ACTIVE_SCENE_COUNT:-3}"
SPLIT="${SPLIT:-all}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
MAX_BATCHES="${MAX_BATCHES:-}"

cd "${ROOT_DIR}"

ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

echo "[heldout-eval] gpu=${GPU}"
echo "[heldout-eval] split=${SPLIT}, initial_scene_idx=${HELDOUT_INITIAL_SCENE_IDX}, active_scene_count=${HELDOUT_ACTIVE_SCENE_COUNT}"
echo "[heldout-eval] manifest=${MANIFEST}"
echo "[heldout-eval] out=${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python ablation/eag3r_metrics_eval.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --device cuda:0 \
  --split "${SPLIT}" \
  --initial-scene-idx "${HELDOUT_INITIAL_SCENE_IDX}" \
  --active-scene-count "${HELDOUT_ACTIVE_SCENE_COUNT}" \
  --num-views "${NUM_VIEWS}" \
  --ldr-event-id "${LDR_ID}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --num-workers 0 \
  --skip-missing \
  "${ARGS[@]}" \
  "$@"

echo "[done] ${OUT_DIR}/summary.csv"
