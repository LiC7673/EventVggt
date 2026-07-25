#!/usr/bin/env bash
# Synthetic four-scene, five-exposure depth/normal/pose evaluation.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_pose_refine_gpu4/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-exp_f/cur_event_refiner_first_pose_refine_gpu4/test_four_scenes_pose}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  fallback="${CHECKPOINT%checkpoint-adapter-best.pth}checkpoint-adapter-last.pth"
  if [[ -f "${fallback}" ]]; then
    CHECKPOINT="${fallback}"
  else
    echo "Missing pose-refine checkpoint: ${CHECKPOINT}" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m paired_token_reliability.evaluate_cur_event_pose_refine_four_scenes \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --root "${DATA_ROOT}" \
  --event-source-mode cur_event \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --num-views 4 \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/test_depth_normal_pose.log"

echo "Metrics CSV/JSON and ATE/RPE results are under ${OUTPUT}"
