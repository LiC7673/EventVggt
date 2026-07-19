#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_cross_view_dn_gpu4/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-exp_f/cur_event_refiner_first_cross_view_dn_gpu4/test_four_scenes_cross_view_patches}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"

python -m paired_token_reliability.evaluate_cur_event_cross_view_dn_four_scenes \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" --root "${DATA_ROOT}" \
  --event-source-mode cur_event --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --num-views 4 --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" "$@"

