#!/usr/bin/env bash
# Inference-only evaluation of cur_event_refiner_first_pose_refine_gpu4.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
EXP_DIR="${EXP_DIR:-exp_f/cur_event_refiner_first_pose_refine_gpu4}"
CHECKPOINT="${CHECKPOINT:-${EXP_DIR}/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-${EXP_DIR}/test_four_scenes_depth_pose}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  last="${EXP_DIR}/checkpoint-adapter-last.pth"
  if [[ -f "${last}" ]]; then
    echo "[checkpoint] best checkpoint absent; using ${last}"
    CHECKPOINT="${last}"
  else
    echo "No pose-refine checkpoint found. Checked:" >&2
    echo "  ${CHECKPOINT}" >&2
    echo "  ${last}" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[test] checkpoint=${CHECKPOINT}"
echo "[test] output=${OUTPUT}"
echo "[test] scenes=4 exposures=${EXPOSURES:-0,1,2,5,10} views=4 GPU=${GPU}"

python -m paired_token_reliability.evaluate_cur_event_pose_refine_four_scenes \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --root "${DATA_ROOT}" \
  --event-source-mode cur_event \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --num-views 4 \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 \
  --num-workers "${NUM_WORKERS:-0}" \
  --depth-scale "${DEPTH_SCALE:-2.0}" \
  --visualize-every "${VISUALIZE_EVERY:-1}" \
  --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/test.log"

python -m paired_token_reliability.summarize_four_scene_depth_pose \
  --input "${OUTPUT}/metrics.csv" \
  --output-dir "${OUTPUT}"

echo "[done] Full metrics: ${OUTPUT}/metrics.csv and summary.json"
echo "[done] Compact report: ${OUTPUT}/depth_pose_report.csv and depth_pose_report.txt"
echo "[done] Visualizations: ${OUTPUT}/visualizations"
