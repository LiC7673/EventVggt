#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -m fine_rgb.evaluate_rgb_results_pretrained \
  --rgb-results-root "${RGB_RESULTS_ROOT:-/data1/lzh/method/event/HDRev-Diff/results}" \
  --data-root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --pretrained "${PRETRAINED:-ckpt/model.pt}" \
  --output-dir "${OUTPUT_DIR:-exp_f/rgb_pretrained_hdrev_results_7scenes_all_ev_scale20}" \
  --ldr-event-ids "${EXPOSURES:-0,1,2,5,10}" \
  --num-views "${NUM_VIEWS:-4}" \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --depth-scale 2.0 \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-4}" \
  --pin-memory \
  --amp "${AMP:-bf16}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}"
