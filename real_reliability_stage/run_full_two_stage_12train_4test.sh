#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
STAGE1_GPU="${STAGE1_GPU:-${GPU_ARRAY[0]}}"
STAGE1_OUT="${STAGE1_OUT:-${ROOT_DIR}/abl_event_exp/real_reliability_stage}"
RELIABILITY_CKPT="${RELIABILITY_CKPT:-${STAGE1_OUT}/reliability_net/checkpoint-best.pth}"
RUN_STAGE1="${RUN_STAGE1:-auto}"

if [[ "${RUN_STAGE1}" == "true" || ( "${RUN_STAGE1}" == "auto" && ! -f "${RELIABILITY_CKPT}" ) ]]; then
  echo "[pipeline] Stage 1: render labels and train ReliabilityNet on 12 scenes"
  GPU="${STAGE1_GPU}" \
  OUT_ROOT="${STAGE1_OUT}" \
  INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}" \
  ACTIVE_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}" \
  NUM_VIEWS="${NUM_VIEWS:-4}" \
  EPOCHS="${STAGE1_EPOCHS:-20}" \
  bash real_reliability_stage/run_two_stage_real_reliability.sh
fi

if [[ ! -f "${RELIABILITY_CKPT}" ]]; then
  echo "[error] Stage 1 did not produce ${RELIABILITY_CKPT}" >&2
  exit 1
fi

echo "[pipeline] Stage 2: frozen ReliabilityNet + VGGT finetune; 12 train / 4 unseen test scenes"
GPUS="${GPUS}" \
RELIABILITY_CKPT="${RELIABILITY_CKPT}" \
bash real_reliability_stage/run_stage2_vggt_12train_4test.sh "$@"

