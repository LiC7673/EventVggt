#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
GPUS="${GPUS:-2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
PORT_STAGE2="${PORT_STAGE2:-30220}"
PORT_STAGE3="${PORT_STAGE3:-30221}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
SCENE_COUNT="${SCENE_COUNT:-12}"
NUM_VIEWS="${NUM_VIEWS:-4}"
LDR_ID="${LDR_ID:-ev_5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-10}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/checkpoints/reliability_net_stage1_scene12}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_DIR}/checkpoint-best.pth}"
STAGE2_EXP="${STAGE2_EXP:-staged_reliability_stage2_frozen_scene12}"
STAGE3_EXP="${STAGE3_EXP:-staged_reliability_stage3_joint_scene12}"
STAGE2_CKPT="${ROOT_DIR}/checkpoints/${STAGE2_EXP}/checkpoint-last.pth"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/ablation_logs/staged_reliability_$(date +%Y%m%d_%H%M%S)}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[stage1] ReliabilityNet checkpoint missing; training on physical GPU 2"
  GPU=2 DATA_ROOT="${DATA_ROOT}" OUT_DIR="${STAGE1_DIR}" \
    ACTIVE_SCENE_COUNT="${SCENE_COUNT}" EPOCHS="${STAGE1_EPOCHS}" \
    bash reliability_pretrain/run_stage1_reliability_net.sh \
    > "${LOG_DIR}/stage1_reliability.log" 2>&1
else
  echo "[stage1] reuse ${STAGE1_CKPT}"
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 did not produce ${STAGE1_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_EXISTING}" == "true" && -f "${STAGE2_CKPT}" ]]; then
  echo "[stage2] reuse ${STAGE2_CKPT}"
else
  echo "[stage2] frozen ReliabilityNet, GPUs ${GPUS}"
  CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 \
    --main_process_port "${PORT_STAGE2}" --mixed_precision bf16 --dynamo_backend no \
    reliability_staged_finetune/finetune_stage2_frozen_reliability.py \
    exp_name="${STAGE2_EXP}" epochs="${STAGE2_EPOCHS}" \
    pretrained="${ROOT_DIR}/ckpt/model.pt" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" data.initial_scene_idx=0 \
    data.active_scene_count="${SCENE_COUNT}" num_workers="${NUM_WORKERS}" pin_mem=false \
    +data.additive_event_root=events_additive \
    +model.stage1_reliability_checkpoint="${STAGE1_CKPT}" \
    > "${LOG_DIR}/stage2_frozen.log" 2>&1
fi

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[error] Stage 2 did not produce ${STAGE2_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_EXISTING}" == "true" && -f "${ROOT_DIR}/checkpoints/${STAGE3_EXP}/checkpoint-last.pth" ]]; then
  echo "[stage3] already complete"
else
  echo "[stage3] joint low-LR finetuning, GPUs ${GPUS}"
  CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
  "${ACCELERATE_BIN}" launch \
    --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 \
    --main_process_port "${PORT_STAGE3}" --mixed_precision bf16 --dynamo_backend no \
    reliability_staged_finetune/finetune_stage3_joint_reliability.py \
    exp_name="${STAGE3_EXP}" epochs="${STAGE3_EPOCHS}" lr=1e-5 \
    pretrained="${STAGE2_CKPT}" \
    data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
    data.ldr_event_id="${LDR_ID}" data.initial_scene_idx=0 \
    data.active_scene_count="${SCENE_COUNT}" num_workers="${NUM_WORKERS}" pin_mem=false \
    +data.additive_event_root=events_additive \
    +model.stage1_reliability_checkpoint="${STAGE1_CKPT}" \
    +train.reliability_lr_scale=0.25 \
    +loss.joint_reliability_weight=0.30 \
    > "${LOG_DIR}/stage3_joint.log" 2>&1
fi

echo "[done] stage2=${STAGE2_CKPT}"
echo "[done] stage3=${ROOT_DIR}/checkpoints/${STAGE3_EXP}/checkpoint-last.pth"
echo "[logs] ${LOG_DIR}"

