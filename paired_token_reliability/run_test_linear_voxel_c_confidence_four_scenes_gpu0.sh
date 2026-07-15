#!/usr/bin/env bash
# Stream the four paper scenes and ev_0/1/2/5/10 through GPU0.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
EXPERIMENT="exp/linear_voxel_c_confidence_normal_refine_gpu4"
CHECKPOINT="${CHECKPOINT:-${EXPERIMENT}/checkpoint-best.pth}"
OUTPUT="${OUTPUT:-${EXPERIMENT}/test_four_scenes_streaming_gpu0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  for CANDIDATE in \
    "${EXPERIMENT}/checkpoint-adapter-best.pth" \
    "${EXPERIMENT}/checkpoint-adapter-last.pth" \
    "${EXPERIMENT}/checkpoint-last.pth"; do
    if [[ -f "${CANDIDATE}" ]]; then CHECKPOINT="${CANDIDATE}"; break; fi
  done
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
echo "[test] gpu=${CUDA_VISIBLE_DEVICES} checkpoint=${CHECKPOINT} output=${OUTPUT}"
python -m paired_token_reliability.evaluate_linear_voxel_c_confidence_four_scenes_streaming \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --exposures "${EXPOSURES:-0,1,2,5,10}" \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --window-stride "${WINDOW_STRIDE:-1}" \
  --num-views "${NUM_VIEWS:-1}" \
  --batch-size 1 --num-workers 0 --amp "${AMP:-none}" \
  --event-resize-method voxel_linear_time --event-resize-bins 5 \
  --print-freq "${PRINT_FREQ:-20}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/evaluate.log"
