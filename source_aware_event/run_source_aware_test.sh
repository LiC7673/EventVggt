#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/source_aware_60_12_12}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_split.json}"
SOURCE_MODE="${SOURCE_MODE:-learned}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_ROOT}/source_${SOURCE_MODE}_full_train60_val12/checkpoint-best.pth}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_ROOT}/final_test_12_scenes}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m source_aware_event.evaluate_heldout \
  --checkpoint "${CHECKPOINT}" \
  --scene-manifest "${SCENE_MANIFEST}" \
  --output-dir "${RESULT_DIR}" \
  --root "${DATA_ROOT}" \
  --num-views "${NUM_VIEWS:-2}" \
  --ldr-event-id "${LDR_ID:-ev_5}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device cuda:0

echo "[done] ${RESULT_DIR}/summary.csv"
