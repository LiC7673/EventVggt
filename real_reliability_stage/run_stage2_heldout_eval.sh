#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
EXP_NAME="${EXP_NAME:-stage2_reliability_residual_train12_test4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/abl_event_exp/${EXP_NAME}/checkpoint-last.pth}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/abl_event_exp/${EXP_NAME}/heldout_eval_scene12_15}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-12}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-4}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi

ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] held-out scenes=${INITIAL_SCENE_IDX}..$((INITIAL_SCENE_IDX + ACTIVE_SCENE_COUNT - 1))"
echo "[eval] output=${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python real_reliability_stage/evaluate_stage2_heldout.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --initial-scene-idx "${INITIAL_SCENE_IDX}" \
  --active-scene-count "${ACTIVE_SCENE_COUNT}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --window-stride "${WINDOW_STRIDE}" \
  --num-views "${NUM_VIEWS}" \
  --ldr-event-id "${LDR_ID}" \
  --event-resize-bins 10 \
  --num-workers "${NUM_WORKERS}" \
  --device cuda:0 \
  --amp bf16 \
  "${ARGS[@]}" \
  "$@"

echo "[done] ${OUT_DIR}/summary.json"
echo "[done] ${OUT_DIR}/condition_metrics.csv"
