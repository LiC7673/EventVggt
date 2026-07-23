#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python -m paired_token_reliability.benchmark_cur_event_hardware \
  --checkpoint "${CHECKPOINT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth}" \
  --output "${OUTPUT:-exp_f/cur_event_clean_hf_residual_v2_gpu4/hardware_benchmark.json}" \
  --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  --scene "${SCENE:-DH2_Socrates and Seneca_Car_Paint_Midnight}" \
  --exposure "${EXPOSURE:-ev_2}" \
  --num-views "${NUM_VIEWS:-4}" \
  --resolution "${WIDTH:-518}" "${HEIGHT:-392}" \
  --event-resize-bins 5 \
  --event-resize-method voxel_linear_time \
  --warmup "${WARMUP:-10}" \
  --repeats "${REPEATS:-50}" \
  --amp "${AMP:-none}" \
  "$@"
