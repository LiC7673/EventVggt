#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-30070}"
EXP_NAME="${EXP_NAME:-geometry_event_temporal_detail_img_reliability_scene12}"
EPOCHS="${EPOCHS:-20}"
NUM_VIEWS="${NUM_VIEWS:-4}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-0}"
ACTIVE_SCENE_COUNT="${ACTIVE_SCENE_COUNT:-12}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-10}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEM="${PIN_MEM:-false}"
GEOMETRY_EVENT_DILATE="${GEOMETRY_EVENT_DILATE:-5}"

cd "${ROOT_DIR}"
echo "[train] geometry/diffuse event oracle reliability on GPUs ${GPUS}"
echo "[train] exp=${EXP_NAME}, scenes=${INITIAL_SCENE_IDX}..$((INITIAL_SCENE_IDX + ACTIVE_SCENE_COUNT - 1)), dilation=${GEOMETRY_EVENT_DILATE}"

CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  diffuse_event_ablation/finetune_geometry_event_reliability.py \
  exp_name="${EXP_NAME}" \
  epochs="${EPOCHS}" \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  data.num_views="${NUM_VIEWS}" \
  data.initial_scene_idx="${INITIAL_SCENE_IDX}" \
  data.active_scene_count="${ACTIVE_SCENE_COUNT}" \
  data.test_frame_count="${TEST_FRAME_COUNT}" \
  data.ldr_event_id="${LDR_ID}" \
  +data.additive_event_branch=geometry_motion \
  +data.additive_event_root=events_additive \
  +data.geometry_event_mask_dilate_kernel="${GEOMETRY_EVENT_DILATE}" \
  "$@"

echo "[done] checkpoints/${EXP_NAME}/checkpoint-last.pth"
