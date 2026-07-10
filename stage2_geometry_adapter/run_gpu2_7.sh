#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-6}"
PORT="${PORT:-29627}"
STAGE1_CKPT="${STAGE1_CKPT:-${ROOT}/abl_event_exp/event_contribution_stage1/checkpoint-best.pth}"
BASE_VGGT_CKPT="${BASE_VGGT_CKPT:-${ROOT}/ckpt/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/abl_event_exp/stage2_geometry_adapter}"
EXP_A="${EXP_A:-geometry_adapter_stage2_a}"
EXP_B="${EXP_B:-geometry_adapter_stage2_b}"
EPOCHS_A="${EPOCHS_A:-12}"
EPOCHS_B="${EPOCHS_B:-8}"
LR_A="${LR_A:-1e-4}"
LR_B="${LR_B:-2e-5}"
RUN_PHASE_B="${RUN_PHASE_B:-1}"
TRAIN_CONTRIBUTION_B="${TRAIN_CONTRIBUTION_B:-false}"
UNFREEZE_LAST_BLOCKS="${UNFREEZE_LAST_BLOCKS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_10}"
TRAIN_INITIAL_SCENE_IDX="${TRAIN_INITIAL_SCENE_IDX:-0}"
TRAIN_SCENE_COUNT="${TRAIN_SCENE_COUNT:-12}"
TEST_INITIAL_SCENE_IDX="${TEST_INITIAL_SCENE_IDX:-12}"
TEST_SCENE_COUNT="${TEST_SCENE_COUNT:-4}"
HELDOUT_TEST_FRAME_COUNT="${HELDOUT_TEST_FRAME_COUNT:-120}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage-1 contribution checkpoint missing: ${STAGE1_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${BASE_VGGT_CKPT}" ]]; then
  echo "[error] base VGGT checkpoint missing: ${BASE_VGGT_CKPT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_PROCESSES="${#GPU_ARRAY[@]}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[test] Stage-2 adapter invariants"
python -m stage2_geometry_adapter.test_model

COMMON_OVERRIDES=(
  "num_workers=${NUM_WORKERS}"
  "data.num_views=${NUM_VIEWS}"
  "data.ldr_event_id=${LDR_ID}"
  "+data.train_initial_scene_idx=${TRAIN_INITIAL_SCENE_IDX}"
  "+data.train_scene_count=${TRAIN_SCENE_COUNT}"
  "+data.train_holdout_frame_count=0"
  "+data.test_initial_scene_idx=${TEST_INITIAL_SCENE_IDX}"
  "+data.test_scene_count=${TEST_SCENE_COUNT}"
  "+data.heldout_test_frame_count=${HELDOUT_TEST_FRAME_COUNT}"
  "+model.stage1_contribution_checkpoint=${STAGE1_CKPT}"
  "+adapter_output_root=${OUTPUT_ROOT}"
  "+skip_final_eval=true"
)

echo "[Phase A] GPUs=${GPUS} processes=${NUM_PROCESSES}; adapters only"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${PORT}" \
  stage2_geometry_adapter/finetune.py \
  "pretrained=${BASE_VGGT_CKPT}" \
  "epochs=${EPOCHS_A}" \
  "lr=${LR_A}" \
  "exp_name=${EXP_A}" \
  "+train.adapter_phase=A" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"

PHASE_A_CKPT="${OUTPUT_ROOT}/${EXP_A}/checkpoint-last.pth"
if [[ ! -f "${PHASE_A_CKPT}" ]]; then
  echo "[error] Phase-A checkpoint missing after training: ${PHASE_A_CKPT}" >&2
  exit 1
fi

if [[ "${RUN_PHASE_B}" == "1" ]]; then
  echo "[Phase B] low-LR DPT/late-VGGT finetuning from ${PHASE_A_CKPT}"
  CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "$((PORT + 1))" \
    stage2_geometry_adapter/finetune.py \
    "pretrained=${PHASE_A_CKPT}" \
    "epochs=${EPOCHS_B}" \
    "lr=${LR_B}" \
    "exp_name=${EXP_B}" \
    "+train.adapter_phase=B" \
    "+train.adapter_unfreeze_last_blocks=${UNFREEZE_LAST_BLOCKS}" \
    "+train.adapter_train_contribution=${TRAIN_CONTRIBUTION_B}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
fi

echo "[done] Phase A: ${PHASE_A_CKPT}"
if [[ "${RUN_PHASE_B}" == "1" ]]; then
  echo "[done] Phase B: ${OUTPUT_ROOT}/${EXP_B}/checkpoint-last.pth"
fi

