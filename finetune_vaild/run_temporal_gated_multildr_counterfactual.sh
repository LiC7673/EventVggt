#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/mul_loss_detail_gt_temporal_gated_multildr/checkpoint-last.pth}"
LDR_ID="${LDR_ID:-ev_5}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/finetune_vaild/results/temporal_gated_multildr_counterfactual}"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" finetune_vaild/verify_event_input_counterfactual.py \
  --root "$DATA_ROOT" \
  --checkpoint "$CKPT" \
  --model-variant temporal_exposure_invariant \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --ldr-event-id "$LDR_ID" \
  --num-views 4 \
  --active-scene-count 4 \
  --samples-per-scene 1 \
  --event-support-mode temporal_polarity \
  --event-hidden-dim 16 \
  --refiner-residual-scale 0.01 \
  --event-gate-downsample 4 \
  --num-workers 0 \
  "$@"
