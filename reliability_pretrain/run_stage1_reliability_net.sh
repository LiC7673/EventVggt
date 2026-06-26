#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-2}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/checkpoints/reliability_net_stage1_scene12}"

cd "${ROOT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python reliability_pretrain/train_reliability_net.py \
  --root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --initial-scene-idx "${INITIAL_SCENE_IDX:-0}" \
  --active-scene-count "${ACTIVE_SCENE_COUNT:-12}" \
  --test-scene-count "${TEST_SCENE_COUNT:-6}" \
  --num-bins "${NUM_BINS:-5}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-1e-4}" \
  --amp \
  "$@"

echo "[done] ${OUT_DIR}/checkpoint-best.pth"
