#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

GPU="${GPU:-0}"
EXP_DIR="${EXP_DIR:-exp_f/cur_event_spatial_pose_refine_v2_gpu4}"
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-${EXP_DIR}/test_four_scenes_depth_pose}"
[[ -f "${CHECKPOINT}" ]] || CHECKPOINT="${EXP_DIR}/checkpoint-adapter-last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing V2 checkpoint in ${EXP_DIR}" >&2; exit 2
fi
mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m paired_token_reliability.evaluate_cur_event_pose_refine_v2_four_scenes \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
  --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --event-source-mode cur_event --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --num-views 4 --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 --num-workers "${NUM_WORKERS:-0}" \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/test.log"

python -m paired_token_reliability.summarize_four_scene_depth_pose \
  --input "${OUTPUT}/metrics.csv" --output-dir "${OUTPUT}"
