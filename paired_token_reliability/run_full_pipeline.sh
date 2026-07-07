#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
TEACHER="${TEACHER:-abl_event_exp/multildr_token_strategy/paired_token_full/checkpoint-last.pth}"
OUT_ROOT="${OUT_ROOT:-abl_event_exp/paired_token_reliability}"
LABEL_DIR="${LABEL_DIR:-${OUT_ROOT}/labels}"
RELIABILITY_DIR="${RELIABILITY_DIR:-${OUT_ROOT}/reliability_net}"
STAGE2_DIR="${STAGE2_DIR:-${OUT_ROOT}/paired_token_reliability_stage2}"
STAGE1_GPU="${STAGE1_GPU:-6}"
STAGE2_GPUS="${STAGE2_GPUS:-6,7}"
STAGE2_PROCESSES="${STAGE2_PROCESSES:-2}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUT_ROOT}/logs"

if [[ ! -f "${TEACHER}" ]]; then
  echo "[error] paired-token teacher missing: ${TEACHER}" >&2
  exit 2
fi

if [[ ! -f "${LABEL_DIR}/manifest.json" ]]; then
  echo "[1/4] export paired-token geometry targets on GPU ${STAGE1_GPU}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.export_targets \
    --teacher "${TEACHER}" \
    --output "${LABEL_DIR}" \
    --ldr-ids ev_1 ev_2 ev_5 ev_10 \
    --val-scenes 2 \
    --token-cosine-floor 0.80 \
    --dilate-kernel 3 \
    2>&1 | tee "${OUT_ROOT}/logs/export_targets.log"
else
  echo "[1/4] reuse target manifest: ${LABEL_DIR}/manifest.json"
fi

if [[ ! -f "${RELIABILITY_DIR}/checkpoint-best.pth" ]]; then
  echo "[2/4] train standalone ReliabilityUNet on GPU ${STAGE1_GPU}"
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.train_reliability \
    --manifest "${LABEL_DIR}/manifest.json" \
    --output "${RELIABILITY_DIR}" \
    --epochs "${EPOCHS_STAGE1}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --amp \
    2>&1 | tee "${OUT_ROOT}/logs/train_reliability.log"
else
  echo "[2/4] reuse ReliabilityUNet: ${RELIABILITY_DIR}/checkpoint-best.pth"
fi

echo "[3/4] Stage 2 VGGT finetune on visible GPUs ${STAGE2_GPUS}"
CUDA_VISIBLE_DEVICES="${STAGE2_GPUS}" accelerate launch \
  --multi_gpu \
  --num_processes "${STAGE2_PROCESSES}" \
  --gpu_ids all \
  --mixed_precision bf16 \
  -m paired_token_reliability.finetune_stage2 \
  epochs="${EPOCHS_STAGE2}" \
  num_workers="${NUM_WORKERS}" \
  data.root="${DATA_ROOT}" \
  data.num_views=4 \
  ++data.train_initial_scene_idx=0 \
  ++data.train_scene_count=12 \
  ++data.train_holdout_frame_count=0 \
  ++data.test_initial_scene_idx=12 \
  ++data.test_scene_count=4 \
  ++data.heldout_test_frame_count=120 \
  ++model.reliability_checkpoint="${RELIABILITY_DIR}/checkpoint-best.pth" \
  ++model.reliability_gate_floor=0.15 \
  ++model.reliability_dilate_kernel=3 \
  ++vis.stage2_save_every_steps=2000 \
  2>&1 | tee "${OUT_ROOT}/logs/finetune_stage2.log"

CHECKPOINT="${STAGE2_DIR}/checkpoint-last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage 2 checkpoint missing after training: ${CHECKPOINT}" >&2
  exit 3
fi

echo "[4/4] held-out causal evaluation on GPU ${STAGE1_GPU}"
CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m paired_token_reliability.evaluate_stage2 \
  --checkpoint "${CHECKPOINT}" \
  --reliability-checkpoint "${RELIABILITY_DIR}/checkpoint-best.pth" \
  --output-dir "${STAGE2_DIR}/heldout_eval" \
  --root "${DATA_ROOT}" \
  --initial-scene-idx 12 \
  --active-scene-count 4 \
  --test-frame-count 120 \
  --window-stride 4 \
  --num-views 4 \
  --event-resize-bins 10 \
  --amp bf16 \
  2>&1 | tee "${OUT_ROOT}/logs/evaluate_stage2.log"

echo "done: ${OUT_ROOT}"
