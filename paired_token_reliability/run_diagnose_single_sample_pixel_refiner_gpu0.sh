#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-exp_f/refiner_single_sample_diagnostic}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"

python -m paired_token_reliability.diagnose_single_sample_pixel_refiner \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" --root "${DATA_ROOT}" \
  --scene "${SCENE:-Bearded Man_Ceramic_Glazed_White}" \
  --exposure "${EXPOSURE:-ev_2}" --steps "${STEPS:-500}" \
  --lr "${LR:-0.001}" --num-views "${NUM_VIEWS:-4}" "$@"
