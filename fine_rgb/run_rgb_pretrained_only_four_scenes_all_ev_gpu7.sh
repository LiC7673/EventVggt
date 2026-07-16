#!/usr/bin/env bash
# Untouched pretrained RGB-only baseline: four views, four scenes, every EV level.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUTPUT="${OUTPUT:-exp_f/rgb_pretrained_only_4views_four_scenes_all_ev_gpu7}"
mkdir -p "${OUTPUT}/logs"

python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
  --pretrained "${PRETRAINED:-ckpt/model.pt}" \
  --skip-finetuned \
  --data-root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --scenes \
    "Centaur_Anodized_Red" \
    "Child_with_goose_Industrial_Plastic_Grey" \
    "Colchester Sphinx_Old_Copper" \
    "Cupid as Shepherd_100MB_Old_Copper" \
  --ldr-event-ids "${LDR_EVENT_IDS:-0,1,2,5,10}" \
  --num-views "${NUM_VIEWS:-4}" \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
  --amp "${AMP:-none}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" \
  --output-dir "${OUTPUT}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/evaluate.log"
