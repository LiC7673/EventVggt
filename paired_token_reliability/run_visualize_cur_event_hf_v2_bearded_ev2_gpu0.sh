#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/bearded_ev2_first10}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

python -m paired_token_reliability.visualize_cur_event_hf_v2_bearded_ev2 \
 --checkpoint "${CHECKPOINT}" \
 --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
 --output "${OUTPUT}" --frames "${FRAMES:-10}" \
 --depth-scale "${DEPTH_SCALE:-2.0}" "$@"
