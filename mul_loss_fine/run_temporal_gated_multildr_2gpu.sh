#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-29890}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
LDR_TRAIN_IDS="${LDR_TRAIN_IDS:-ev_2,ev_5,ev_10}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
NUM_VIEWS="${NUM_VIEWS:-4}"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" HYDRA_FULL_ERROR=1 \
"$ACCELERATE_BIN" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "$PORT" \
  mul_loss_fine/finetune_mul_loss_detail_gt_temporal_gated_multildr.py \
  num_workers="$NUM_WORKERS" \
  pin_mem="$PIN_MEM" \
  +data.eval_ldr_event_id="$EVAL_LDR_ID" \
  +data.mul_ldr_train_ids="[$LDR_TRAIN_IDS]" \
  +data.mul_ldr_exposures_per_sample=2 \
  +data.mul_ldr_scenes_per_batch=1 \
  +data.mul_ldr_num_views="$NUM_VIEWS" \
  "$@"
