#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-29880}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" HYDRA_FULL_ERROR=1 \
"$ACCELERATE_BIN" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "$PORT" \
  mul_loss_fine/finetune_mul_loss_detail_gt_temporal_gated.py \
  num_workers="$NUM_WORKERS" \
  pin_mem="$PIN_MEM" \
  "$@"
