#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6,7}"
PORT="${PORT:-29942}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEM="${PIN_MEM:-true}"
NUM_VIEWS="${NUM_VIEWS:-4}"
EPOCHS="${EPOCHS:-20}"
LDR_ID="${LDR_ID:-ev_5}"
PRINT_FREQ="${PRINT_FREQ:-50}"
LOG_FREQ="${LOG_FREQ:-100}"
VIS_EVERY_STEPS="${VIS_EVERY_STEPS:-2000}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-2000}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
TEST_VIS_MAX_BATCHES="${TEST_VIS_MAX_BATCHES:-1}"
RESIDUAL_POSTFILTER_KERNEL="${RESIDUAL_POSTFILTER_KERNEL:-3}"
RESIDUAL_POSTFILTER_STRENGTH="${RESIDUAL_POSTFILTER_STRENGTH:-0.75}"
RELIABILITY_GATE_FLOOR="${RELIABILITY_GATE_FLOOR:-0.10}"

TRAIN_INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}"
TRAIN_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}"
TEST_INITIAL_SCENE_IDX="${TEST_INITIAL_SCENE_IDX:-12}"
TEST_SCENE_COUNT="${TEST_SCENE_COUNT:-4}"
TRAIN_HOLDOUT_FRAME_COUNT="${TRAIN_HOLDOUT_FRAME_COUNT:-0}"
HELDOUT_TEST_FRAME_COUNT="${HELDOUT_TEST_FRAME_COUNT:-120}"

EXP_NAME="${EXP_NAME:-stage2_frozen_real_reliability_train12_test4}"
RELIABILITY_CKPT="${RELIABILITY_CKPT:-${ROOT_DIR}/abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth}"

if [[ ! -f "${RELIABILITY_CKPT}" ]]; then
  echo "[error] ReliabilityNet checkpoint missing: ${RELIABILITY_CKPT}" >&2
  echo "Run: bash real_reliability_stage/run_two_stage_real_reliability.sh" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_PROCESSES="${#GPU_ARRAY[@]}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[stage2] GPUs=${GPUS}, processes=${NUM_PROCESSES}"
echo "[stage2] train scenes: ${TRAIN_INITIAL_SCENE_IDX}..$((TRAIN_INITIAL_SCENE_IDX + TRAIN_SCENE_COUNT - 1))"
echo "[stage2] held-out scenes: ${TEST_INITIAL_SCENE_IDX}..$((TEST_INITIAL_SCENE_IDX + TEST_SCENE_COUNT - 1))"
echo "[stage2] reliability: ${RELIABILITY_CKPT}"
echo "[stage2] output: ${ROOT_DIR}/abl_event_exp/${EXP_NAME}"

CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${PORT}" \
  real_reliability_stage/finetune_stage2_vggt.py \
  num_workers="${NUM_WORKERS}" \
  pin_mem="${PIN_MEM}" \
  epochs="${EPOCHS}" \
  exp_name="${EXP_NAME}" \
  data.num_views="${NUM_VIEWS}" \
  data.ldr_event_id="${LDR_ID}" \
  +data.train_initial_scene_idx="${TRAIN_INITIAL_SCENE_IDX}" \
  +data.train_scene_count="${TRAIN_SCENE_COUNT}" \
  +data.train_holdout_frame_count="${TRAIN_HOLDOUT_FRAME_COUNT}" \
  +data.test_initial_scene_idx="${TEST_INITIAL_SCENE_IDX}" \
  +data.test_scene_count="${TEST_SCENE_COUNT}" \
  +data.heldout_test_frame_count="${HELDOUT_TEST_FRAME_COUNT}" \
  +model.reliability_checkpoint="${RELIABILITY_CKPT}" \
  +model.reliability_gate_floor="${RELIABILITY_GATE_FLOOR}" \
  +model.residual_postfilter_kernel="${RESIDUAL_POSTFILTER_KERNEL}" \
  +model.residual_postfilter_strength="${RESIDUAL_POSTFILTER_STRENGTH}" \
  +stage2_print_freq="${PRINT_FREQ}" \
  +stage2_log_freq="${LOG_FREQ}" \
  +stage2_eval_every_steps="${EVAL_EVERY_STEPS}" \
  +stage2_checkpoint_every_steps="${CHECKPOINT_EVERY_STEPS}" \
  +vis.stage2_save_every_steps="${VIS_EVERY_STEPS}" \
  +vis.stage2_test_max_batches="${TEST_VIS_MAX_BATCHES}" \
  "$@"

echo "[done] ${ROOT_DIR}/abl_event_exp/${EXP_NAME}/checkpoint-last.pth"
echo "[done] held-out visuals: ${ROOT_DIR}/abl_event_exp/${EXP_NAME}/test_vis/"
