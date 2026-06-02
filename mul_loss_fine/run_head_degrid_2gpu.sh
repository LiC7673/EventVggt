#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
DIAG_GPU="${DIAG_GPU:-${GPUS##*,}}"
PORT="${PORT:-29918}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
EXP_NAME="${EXP_NAME:-mul_loss_detail_gt_head_degrid}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/${EXP_NAME}/checkpoint-last.pth}"
DIAGNOSE_AFTER_TRAIN="${DIAGNOSE_AFTER_TRAIN:-true}"
DIAG_OUT="${DIAG_OUT:-${ROOT_DIR}/exp_test/grid_source_diagnostics/${EXP_NAME}}"

PRETRAINED_ARGS=()
if [[ -n "${PRETRAINED:-}" ]]; then
  PRETRAINED_ARGS=("pretrained=${PRETRAINED}")
fi

cd "${ROOT_DIR}"
echo "[train] head-degrid diagnostic on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  mul_loss_fine/finetune_mul_loss_detail_gt_head_degrid.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  exp_name="${EXP_NAME}" \
  data.num_views="${NUM_VIEWS}" \
  data.ldr_event_id="${LDR_ID}" \
  "${PRETRAINED_ARGS[@]}" \
  "$@"

if [[ "${DIAGNOSE_AFTER_TRAIN}" == "true" ]]; then
  echo "[diagnose] grid source on GPU ${DIAG_GPU}"
  CUDA_VISIBLE_DEVICES="${DIAG_GPU}" python exp_test/diagnose_grid_source.py \
    --checkpoint "${CKPT}" \
    --model-variant base \
    --event-hidden-dim 32 \
    --num-views "${NUM_VIEWS}" \
    --ldr-event-id "${LDR_ID}" \
    --samples-per-scene 1 \
    --max-samples 4 \
    --visual-samples 4 \
    --output-dir "${DIAG_OUT}"
  echo "[done] grid diagnostic summary: ${DIAG_OUT}/grid_source_summary.json"
fi
