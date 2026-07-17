#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch_v2_rgb_routes}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/noisy_event_only/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-${ROOT}/noisy_event_only/test_four_scenes_all_ev}"
python -m paired_token_reliability.evaluate_latest_strategy_ablation \
 --variant noisy_event_only --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
 --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
 --event-source-mode cur_event --exposures 0,1,2,5,10 --test-frame-count 120 \
 --num-views 4 --window-stride 1 --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
 --depth-scale "${DEPTH_SCALE:-2.0}" --visualize-every "${VISUALIZE_EVERY:-1}" \
 --max-visuals-per-condition "${MAX_VISUALS:-0}" "$@"
