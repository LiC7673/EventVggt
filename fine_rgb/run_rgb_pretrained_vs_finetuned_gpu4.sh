#!/usr/bin/env bash
# Pure-RGB comparison at every common ev_* RGB level:
# untouched pretrained model vs each matching fine_rgb_<ev_*> checkpoint.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -m fine_rgb.evaluate_rgb_pretrained_vs_finetuned \
  --pretrained "${PRETRAINED:-ckpt/model.pt}" \
  --finetuned-template "${FINETUNED_TEMPLATE:-checkpoints/fine_rgb_{ldr_event_id}/checkpoint-last.pth}" \
  --data-root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --ldr-event-ids "${LDR_EVENT_IDS:-auto}" \
  --num-views "${NUM_VIEWS:-1}" \
  --test-frame-count "${TEST_FRAME_COUNT:-10}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --output-dir "${OUTPUT:-exp/rgb_all_ev_pretrained_vs_finetuned}" \
  "$@" 2>&1 | tee "${LOG_FILE:-rgb_pretrained_vs_finetuned_gpu4.log}"
