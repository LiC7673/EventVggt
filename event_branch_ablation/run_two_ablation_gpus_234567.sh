#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GEOMETRY_GPUS="${GEOMETRY_GPUS:-2,3,4}"
DECOMPOSE_GPUS="${DECOMPOSE_GPUS:-5,6,7}"
GEOMETRY_PROCESSES="${GEOMETRY_PROCESSES:-3}"
DECOMPOSE_PROCESSES="${DECOMPOSE_PROCESSES:-3}"
PORT_GEOMETRY="${PORT_GEOMETRY:-30320}"
PORT_DECOMPOSE="${PORT_DECOMPOSE:-30321}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
NUM_VIEWS="${NUM_VIEWS:-4}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LDR_ID="${LDR_ID:-ev_5}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/abl_event_exp/launch_logs_$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "[launch] geometry_motion full reliability on GPUs ${GEOMETRY_GPUS}"
env CUDA_VISIBLE_DEVICES="${GEOMETRY_GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch --multi_gpu \
  --num_processes "${GEOMETRY_PROCESSES}" --num_machines 1 \
  --main_process_port "${PORT_GEOMETRY}" --mixed_precision bf16 --dynamo_backend no \
  event_branch_ablation/finetune_geometry_motion_full_reliability.py \
  exp_name=geometry_motion_full_img_reliability_scene12 epochs="${EPOCHS}" \
  hydra.run.dir="${ROOT_DIR}/abl_event_exp/geometry_motion_full_img_reliability_scene12" \
  data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" data.ldr_event_id="${LDR_ID}" \
  data.initial_scene_idx=0 data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  num_workers="${NUM_WORKERS}" pin_mem=false \
  > "${LOG_DIR}/geometry_motion.log" 2>&1 &
PID_GEOMETRY=$!

echo "[launch] full-to-additive token decomposition on GPUs ${DECOMPOSE_GPUS}"
env CUDA_VISIBLE_DEVICES="${DECOMPOSE_GPUS}" HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${ACCELERATE_BIN}" launch --multi_gpu \
  --num_processes "${DECOMPOSE_PROCESSES}" --num_machines 1 \
  --main_process_port "${PORT_DECOMPOSE}" --mixed_precision bf16 --dynamo_backend no \
  event_branch_ablation/finetune_full_to_additive_tokens.py \
  exp_name=full_to_additive_tokens_img_reliability_scene12 epochs="${EPOCHS}" \
  hydra.run.dir="${ROOT_DIR}/abl_event_exp/full_to_additive_tokens_img_reliability_scene12" \
  data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" data.ldr_event_id="${LDR_ID}" \
  data.initial_scene_idx=0 data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  num_workers="${NUM_WORKERS}" pin_mem=false \
  > "${LOG_DIR}/full_to_additive_tokens.log" 2>&1 &
PID_DECOMPOSE=$!

STATUS=0
if ! wait "${PID_GEOMETRY}"; then
  echo "[fail] geometry_motion experiment; see ${LOG_DIR}/geometry_motion.log" >&2
  STATUS=1
else
  echo "[done] geometry_motion experiment"
fi
if ! wait "${PID_DECOMPOSE}"; then
  echo "[fail] full-to-additive experiment; see ${LOG_DIR}/full_to_additive_tokens.log" >&2
  STATUS=1
else
  echo "[done] full-to-additive experiment"
fi

echo "[outputs] ${ROOT_DIR}/abl_event_exp"
exit "${STATUS}"
