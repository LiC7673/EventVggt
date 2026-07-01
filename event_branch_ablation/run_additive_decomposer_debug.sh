#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-2}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUT_DIR="${OUT_DIR:-abl_event_exp/additive_decomposer_debug}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[debug] train standalone additive decomposer on physical GPU ${GPU}"
python -m event_branch_ablation.train_additive_decomposer_debug \
  --root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --initial-scene-idx "${INITIAL_SCENE_IDX:-12}" \
  --active-scene-count "${ACTIVE_SCENE_COUNT:-3}" \
  --num-views "${NUM_VIEWS:-4}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-2e-4}" \
  --event-bins "${EVENT_BINS:-10}" \
  --hidden-dim "${HIDDEN_DIM:-32}" \
  --geometry-weight "${GEOMETRY_WEIGHT:-4.0}" \
  --material-weight "${MATERIAL_WEIGHT:-1.0}" \
  --noise-weight "${NOISE_WEIGHT:-0.5}" \
  --geometry-dilate-kernel "${GEOMETRY_DILATE_KERNEL:-9}" \
  --presence-dilate-kernel "${PRESENCE_DILATE_KERNEL:-9}" \
  --vis-panel-width "${VIS_PANEL_WIDTH:-180}"

echo "[debug] done. Check:"
echo "  ${OUT_DIR}/metrics.json"
echo "  ${OUT_DIR}/debug_vis/event_bins/"
echo "  ${OUT_DIR}/decomposer-best.pth"
