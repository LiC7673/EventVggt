#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU:-4}"

python -m paired_token_reliability.visualize_cur_event_hf_v2_ev5_geo_full \
  --checkpoint "${CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}" \
  --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --scene "${SCENE:-Bearded Man_Ceramic_Glazed_White}" \
  --frames "${FRAMES:-10}" \
  --view-index "${VIEW_INDEX:-0}" \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --output "${OUTPUT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/ev5_geo_full_visualization}" \
  --device cuda
