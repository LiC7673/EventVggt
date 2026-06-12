#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-29936}"
EPOCHS="${EPOCHS:-10}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
EXP_NAME="${EXP_NAME:-multildr_detail_then_reliability}"
TEACHER="${TEACHER:-${ROOT_DIR}/checkpoints/ablation_multildr_detail_gt/checkpoint-last.pth}"

cd "${ROOT_DIR}"

if [[ ! -f "${TEACHER}" ]]; then
  echo "[error] missing Multi-LDR detail teacher: ${TEACHER}" >&2
  exit 1
fi

echo "[train] frozen Multi-LDR detail teacher + local event reliability refinement"
echo "[train] teacher=${TEACHER}"
echo "[train] GPUs=${GPUS}, epochs=${EPOCHS}, exp=${EXP_NAME}"

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
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  "$@"

echo "[done] checkpoints/${EXP_NAME}/checkpoint-last.pth"
