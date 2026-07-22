#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch_v2_rgb_routes}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/without_refiner_normal/checkpoint-adapter-last.pth}"
OUTPUT="${OUTPUT:-${ROOT}/without_refiner_normal/test_four_scenes_all_ev}"
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
python -m paired_token_reliability.evaluate_latest_strategy_ablation \
 --variant without_refiner_normal --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
 --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
 --scene-names "Centaur_Anodized_Red" "Child_with_goose_Industrial_Plastic_Grey" \
   "Colchester Sphinx_Old_Copper" "Cupid as Shepherd_100MB_Old_Copper" \
 --event-source-mode cur_event --exposures 0,1,2,5,10 --test-frame-count 120 \
 --num-views 4 --window-stride 1 --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
 --depth-scale "${DEPTH_SCALE:-2.0}" --visualize-every "${VISUALIZE_EVERY:-1}" \
 --max-visuals-per-condition "${MAX_VISUALS:-0}" "$@"
