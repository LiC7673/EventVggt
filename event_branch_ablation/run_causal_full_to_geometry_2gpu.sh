#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-2,3}"
PORT="${PORT:-30332}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
NUM_VIEWS="${NUM_VIEWS:-4}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LDR_ID="${LDR_ID:-ev_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/abl_event_exp/causal_full_to_geometry_scene12}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

env CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 2 --num_machines 1 \
  --main_process_port "${PORT}" --mixed_precision bf16 --dynamo_backend no \
  event_branch_ablation/finetune_causal_full_to_geometry.py \
  exp_name=causal_full_to_geometry_scene12 epochs="${EPOCHS}" \
  hydra.run.dir="${OUTPUT_DIR}" \
  data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" data.ldr_event_id="${LDR_ID}" \
  data.initial_scene_idx=0 data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  num_workers="${NUM_WORKERS}" pin_mem=false
