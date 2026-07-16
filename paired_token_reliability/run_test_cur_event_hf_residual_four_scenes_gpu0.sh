#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/test_four_scenes_all_ev}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
DEPTH_SCALE="${DEPTH_SCALE:-2.0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

python -m paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --root "${DATA_ROOT}" \
  --event-source-mode cur_event \
  --exposures 0,1,2,5,10 \
  --num-views 4 \
  --test-frame-count 120 \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-0}" \
  --depth-scale "${DEPTH_SCALE}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition 0 \
  "$@"
