#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

CALIBRATION_ARGS=()
if [[ "${CALIBRATE_SCALE:-1}" == "1" ]]; then
  CALIBRATION_ARGS+=(--calibrate-scale)
  if [[ -n "${CALIBRATION_SCENE:-}" ]]; then
    CALIBRATION_ARGS+=(--calibration-scene "${CALIBRATION_SCENE}")
  fi
  CALIBRATION_ARGS+=(--max-calibration-batches "${MAX_CALIBRATION_BATCHES:-20}")
fi

python -m paired_token_reliability.evaluate_rgb_pretrained_dsec \
  --checkpoint "${CHECKPOINT:-ckpt/model.pt}" \
  --root "${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}" \
  --output "${OUTPUT:-exp_f/dsec_rgb_pretrained_no_finetune_gpu5}" \
  --num-views "${NUM_VIEWS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --depth-scale "${DEPTH_SCALE:-1.0}" \
  --max-test-batches "${MAX_TEST_BATCHES:-0}" \
  --visualize-every "${VISUALIZE_EVERY:-10}" \
  --max-visualizations "${MAX_VISUALIZATIONS:-40}" \
  "${CALIBRATION_ARGS[@]}" "$@"
