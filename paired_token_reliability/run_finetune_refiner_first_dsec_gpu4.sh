#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_ev0_base_gpu4/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-exp_f/dsec_refiner_first_light_finetune_gpu4}"

python -m paired_token_reliability.finetune_refiner_first_dsec \
 --checkpoint "${CHECKPOINT}" \
 --root "${DSEC_ROOT:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}" \
 --output "${OUTPUT}" --epochs "${EPOCHS:-2}" \
 --max-train-steps "${MAX_TRAIN_STEPS:-2000}" \
 --max-test-batches "${MAX_TEST_BATCHES:-0}" \
 --lr "${LR:-0.00002}" --num-workers "${NUM_WORKERS:-2}" \
 --num-views "${NUM_VIEWS:-4}" "$@"
