#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the normal-oriented checkpoint path}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for evaluation results}"
GPU="${GPU:-0}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m paired_token_reliability.evaluate_normal_oriented \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --initial-scene-idx "${TEST_INITIAL_SCENE_IDX:-12}" \
  --active-scene-count "${TEST_SCENE_COUNT:-4}" \
  --test-frame-count "${HELDOUT_TEST_FRAME_COUNT:-120}" \
  --window-stride "${WINDOW_STRIDE:-4}" --num-views "${NUM_VIEWS:-4}" \
  --event-resize-bins "${EVENT_BINS:-2}" \
  --num-workers "${NUM_WORKERS:-2}" --visualize-every "${TEST_VIS_EVERY:-20}" \
  "$@"
