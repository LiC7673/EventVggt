#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
TEACHER="${TEACHER:-abl_event_exp/multildr_token_strategy/paired_token_full/checkpoint-last.pth}"
OUT_ROOT="${OUT_ROOT:-abl_event_exp/paired_token_reliability_repair}"
BASE_STAGE1_ROOT="${BASE_STAGE1_ROOT:-abl_event_exp/paired_token_reliability}"
LABEL_DIR="${LABEL_DIR:-${BASE_STAGE1_ROOT}/labels}"
RELIABILITY_DIR="${RELIABILITY_DIR:-${BASE_STAGE1_ROOT}/reliability_net}"
EXP_NAME="${EXP_NAME:-paired_token_reliability_repair}"
STAGE1_GPU="${STAGE1_GPU:-6}"
STAGE2_GPUS="${STAGE2_GPUS:-6,7}"
STAGE2_PROCESSES="${STAGE2_PROCESSES:-2}"
PORT="${PORT:-29683}"
EPOCHS_STAGE1="${EPOCHS_STAGE1:-20}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${OUT_ROOT}/logs"

if [[ ! -f "${LABEL_DIR}/manifest.json" ]]; then
  if [[ ! -f "${TEACHER}" ]]; then
    echo "[error] paired-token teacher missing: ${TEACHER}" >&2
    exit 2
  fi
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
  echo "[2/4] train paired-token ReliabilityUNet on GPU ${STAGE1_GPU}"
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

echo "[3/4] repair Stage 2 VGGT finetune on visible GPUs ${STAGE2_GPUS}"
CUDA_VISIBLE_DEVICES="${STAGE2_GPUS}" accelerate launch \
  --multi_gpu \
  --num_processes "${STAGE2_PROCESSES}" \
  --main_process_port "${PORT}" \
  --gpu_ids all \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m repair_reliability.finetune_stage2_repair \
  exp_name="${EXP_NAME}" \
  ++repair_save_dir="${OUT_ROOT}" \
  epochs="${EPOCHS_STAGE2}" \
  num_workers="${NUM_WORKERS}" \
  data.root="${DATA_ROOT}" \
  data.num_views=4 \
  ++data.train_initial_scene_idx=0 \
  ++data.train_scene_count=12 \
  ++data.train_holdout_frame_count=0 \
  ++data.train_min_start_id=2 \
  ++data.test_initial_scene_idx=12 \
  ++data.test_scene_count=4 \
  ++data.heldout_test_frame_count=120 \
  ++model.reliability_checkpoint="${RELIABILITY_DIR}/checkpoint-best.pth" \
  ++model.reliability_gate_floor=0.05 \
  ++model.repair_reliability_threshold=0.58 \
  ++model.repair_reliability_temperature=0.12 \
  ++model.repair_reliability_top_fraction=0.35 \
  ++model.repair_event_support_dilate_kernel=5 \
  ++model.repair_event_support_floor=0.05 \
  ++model.repair_residual_gain=1.6 \
  ++model.repair_output_abs_limit=0.06 \
  ++model.repair_refiner_residual_scale=0.05 \
  ++model.repair_event_delta_highpass_kernel=0 \
  ++model.repair_event_delta_patch_zero_mean=false \
  ++model.repair_event_delta_abs_limit=0.05 \
  ++loss.stage2_residual_target_weight=2.0 \
  ++loss.stage2_residual_gradient_weight=3.0 \
  ++loss.stage2_target_reliability_floor=0.10 \
  ++loss.stage2_target_abs_limit=0.06 \
  ++loss.stage2_target_highpass_kernel=0 \
  ++loss.stage2_event_top_fraction=0.50 \
  ++loss.stage2_flat_normal_weight=0.25 \
  ++loss.stage2_no_event_residual_weight=0.20 \
  ++vis.save_every_steps=3000 \
  2>&1 | tee "${OUT_ROOT}/logs/finetune_stage2_repair.log"

STAGE2_DIR="${OUT_ROOT}/${EXP_NAME}"
CHECKPOINT="${STAGE2_DIR}/checkpoint-last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Stage 2 checkpoint missing after training: ${CHECKPOINT}" >&2
  exit 3
fi

echo "[4/4] held-out causal evaluation on GPU ${STAGE1_GPU}"
CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" python -m repair_reliability.evaluate_stage2_repair \
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
  2>&1 | tee "${OUT_ROOT}/logs/evaluate_stage2_repair.log"

echo "done: ${STAGE2_DIR}"
