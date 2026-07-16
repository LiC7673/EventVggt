#!/usr/bin/env bash
# Strict cur_event four-scene/all-exposure evaluation.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
CHECKPOINT="${CHECKPOINT:-exp_f/alternating_geo_detail_first_dual_c_fixed_cur_event_gpu4/checkpoint-adapter-last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-exp_f/alternating_geo_detail_first_dual_c_fixed_cur_event_gpu4/test_cur_event_four_scenes_all_ev}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU:-0}" python -m \
  paired_token_reliability.evaluate_alternating_detail_first_fixed_four_scenes \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" \
  --event-source-mode cur_event \
  --scene-names \
    "Centaur_Anodized_Red" \
    "Child_with_goose_Industrial_Plastic_Grey" \
    "Colchester Sphinx_Old_Copper" \
    "Cupid as Shepherd_100MB_Old_Copper" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --num-views "${NUM_VIEWS:-4}" --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" "$@"
