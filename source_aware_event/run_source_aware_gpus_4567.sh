#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-4,5,6,7}"
STAGE1_GPU="${STAGE1_GPU:-4}"
PORT="${PORT:-30560}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/source_aware_60_12_12}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_ROOT}/scene_split.json}"
SOURCE_DIR="${SOURCE_DIR:-${OUTPUT_ROOT}/source_net}"
SOURCE_CKPT="${SOURCE_CKPT:-${SOURCE_DIR}/checkpoint-best.pth}"
NUM_VIEWS="${NUM_VIEWS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
LDR_IDS="${LDR_IDS:-[ev_2,ev_5,ev_10]}"
SOURCE_LDR_IDS="${SOURCE_LDR_IDS:-ev_2 ev_5 ev_10}"
EVAL_LDR_ID="${EVAL_LDR_ID:-ev_5}"
SOURCE_MODE="${SOURCE_MODE:-learned}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  python -m paper_scale_training.make_scene_split \
    --root "${DATA_ROOT}" \
    --output "${SCENE_MANIFEST}" \
    --train-count 60 --val-count 12 --test-count 12 \
    --seed "${SPLIT_SEED:-2026}"
fi

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "[source stage1] geometry/material/noise decomposition on GPU ${STAGE1_GPU}"
  read -r -a SOURCE_LDR_ARRAY <<< "${SOURCE_LDR_IDS}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m source_aware_event.train_source_net \
    --root "${DATA_ROOT}" \
    --scene-manifest "${SCENE_MANIFEST}" \
    --output-dir "${SOURCE_DIR}" \
    --train-ldr-ids "${SOURCE_LDR_ARRAY[@]}" \
    --val-ldr-id "${EVAL_LDR_ID}" \
    --event-bins 10 \
    --batch-size "${SOURCE_BATCH_SIZE:-1}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${SOURCE_EPOCHS:-20}" \
    --pin-memory --amp \
    > "${OUTPUT_ROOT}/source_stage1.log" 2>&1
fi

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "[error] source checkpoint missing: ${SOURCE_CKPT}" >&2
  exit 1
fi

echo "[source stage2] frozen source guidance on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" HYDRA_FULL_ERROR=1 \
accelerate launch \
  --multi_gpu --num_processes 4 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  --main_process_port "${PORT}" \
  source_aware_event/finetune_source_aware_60_12_12.py \
  +source_aware_output_root="${OUTPUT_ROOT}" \
  +source_aware_epochs="${EPOCHS}" \
  +source_aware_patience="${PATIENCE}" \
  +data.scene_split_manifest="${SCENE_MANIFEST}" \
  +data.mul_ldr_train_ids="${LDR_IDS}" \
  +data.eval_ldr_event_id="${EVAL_LDR_ID}" \
  +model.source_checkpoint="${SOURCE_CKPT}" \
  +model.source_ablation_mode="${SOURCE_MODE}" \
  data.root="${DATA_ROOT}" data.num_views="${NUM_VIEWS}" \
  num_workers="${NUM_WORKERS}" pin_mem=true \
  print_freq=100 log_freq=100 save_every_steps=2000 \
  vis.save_every_steps=5000 \
  > "${OUTPUT_ROOT}/source_stage2.log" 2>&1

echo "[done] ${OUTPUT_ROOT}/source_${SOURCE_MODE}_full_train60_val12/checkpoint-best.pth"
echo "[test] GPU=7 bash source_aware_event/run_source_aware_test.sh"
