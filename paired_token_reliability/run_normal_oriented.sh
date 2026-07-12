#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-abl_event_exp/normal_oriented_geometry}"

CUDA_VISIBLE_DEVICES="$GPU" python -m pytest -q paired_token_reliability/test_normal_oriented_model.py
CUDA_VISIBLE_DEVICES="$GPU" python -m paired_token_reliability.train_normal_oriented \
  --pretrained "$PRETRAINED" --output "$OUTPUT" \
  --require-full-event-phase-b --no-budget \
  model.event_adapter_uses_rgb=false \
  model.event_adapter_levels='[0,1]' \
  model.normal_update_scale=0.15 \
  model.enable_event_depth_residual=false \
  model.support_dilation_kernel=5 \
  data.decomposition_supervision=true \
  data.event_source_mode=decomposition_full \
  "$@"
