#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-2}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUT_ROOT="${OUT_ROOT:-abl_event_exp/real_reliability_stage}"
LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels}"
NET_DIR="${NET_DIR:-${OUT_ROOT}/reliability_net}"

INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-0}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-5}"
EVENT_BINS="${EVENT_BINS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[stage1] render weak real-event reliability labels"
python -m real_reliability_stage.render_reliability_labels \
  --root "${DATA_ROOT}" \
  --output-dir "${LABEL_DIR}" \
  --splits both \
  --ldr-event-id "${LDR_ID}" \
  --initial-scene-idx "${INITIAL_SCENE_IDX}" \
  --active-scene-count "${ACTIVE_SCENE_COUNT}" \
  --test-frame-count "${TEST_FRAME_COUNT}" \
  --num-views "${NUM_VIEWS}" \
  --event-bins "${EVENT_BINS}" \
  --num-workers "${NUM_WORKERS}" \
  --preview-count "${PREVIEW_COUNT:-32}" \
  "$@"

echo "[stage2] train standalone ReliabilityNet"
python -m real_reliability_stage.train_reliability_net \
  --data-dir "${LABEL_DIR}" \
  --out-dir "${NET_DIR}" \
  --num-bins "${EVENT_BINS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR:-1e-4}" \
  --base-channels "${BASE_CHANNELS:-32}" \
  ${AMP:+--amp}

echo "[done] labels: ${LABEL_DIR}"
echo "[done] net: ${NET_DIR}/checkpoint-best.pth"
echo "[done] label previews: ${LABEL_DIR}/preview/"
echo "[done] prediction previews: ${NET_DIR}/preview/"
