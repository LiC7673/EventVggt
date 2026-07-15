#!/usr/bin/env bash
# Same four-scene/EV metric protocol as the event model, using pure RGB on GPU7.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OUTPUT="${OUTPUT:-exp/rgb_four_scenes_streaming_gpu7}"
mkdir -p "${OUTPUT}/logs"

echo "[test] pure RGB gpu=${CUDA_VISIBLE_DEVICES} output=${OUTPUT}"
python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
  --pretrained "${PRETRAINED:-ckpt/model.pt}" \
  --finetuned-template "${FINETUNED_TEMPLATE:-checkpoints/fine_rgb_{ldr_event_id}/checkpoint-last.pth}" \
  --data-root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --ldr-event-ids "${LDR_EVENT_IDS:-0,1,2,5,10}" \
  --num-views "${NUM_VIEWS:-1}" \
  --test-frame-count "${TEST_FRAME_COUNT:-120}" \
  --batch-size 1 --num-workers 0 --amp "${AMP:-none}" \
  --output-dir "${OUTPUT}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/evaluate.log"
