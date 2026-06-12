#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-29936}"
EPOCHS="${EPOCHS:-10}"
NUM_VIEWS="${NUM_VIEWS:-4}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-0}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
EXP_NAME="${EXP_NAME:-multildr_detail_then_reliability_scene12}"
TEACHER="${TEACHER:-${ROOT_DIR}/checkpoints/ablation_multildr_detail_gt_scene12/checkpoint-last.pth}"

cd "${ROOT_DIR}"

if [[ ! -f "${TEACHER}" ]]; then
  echo "[error] missing Multi-LDR detail teacher: ${TEACHER}" >&2
  exit 1
fi

echo "[train] frozen Multi-LDR detail teacher + local event reliability refinement"
echo "[train] teacher=${TEACHER}"
echo "[train] GPUs=${GPUS}, epochs=${EPOCHS}, exp=${EXP_NAME}"
echo "[train] scenes=[${INITIAL_SCENE_IDX}, $((INITIAL_SCENE_IDX + ACTIVE_SCENE_COUNT - 1))], count=${ACTIVE_SCENE_COUNT}"

CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_multildr_detail_then_reliability.py \
  pretrained="${TEACHER}" \
  exp_name="${EXP_NAME}" \
  epochs="${EPOCHS}" \
  data.num_views="${NUM_VIEWS}" \
  data.initial_scene_idx="${INITIAL_SCENE_IDX}" \
  data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  data.test_frame_count="${TEST_FRAME_COUNT}" \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  "$@"

echo "[done] checkpoints/${EXP_NAME}/checkpoint-last.pth"
