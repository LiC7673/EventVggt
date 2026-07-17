#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
CHECKPOINT="${CHECKPOINT:-exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth}"
OUTPUT="${OUTPUT:-exp_f/contribution_causal_deletion_cur_event_refiner_first}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT to checkpoint-adapter-best.pth or checkpoint-adapter-last.pth" >&2
  exit 2
fi
mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU:-5}" PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
extra=()
if [[ -n "${MAX_BATCHES:-}" ]]; then extra+=(--max-batches "${MAX_BATCHES}"); fi
if [[ -n "${DATA_ROOT:-}" ]]; then extra+=(--root "${DATA_ROOT}"); fi
python -m paired_token_reliability.evaluate_contribution_causal_deletion \
  --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT}" \
  --exposures "${EXPOSURES:-2}" \
  --ratios "${RATIOS:-0.10,0.20,0.30,0.50}" \
  --test-frame-count "${TEST_FRAMES:-30}" \
  --random-repeats "${RANDOM_REPEATS:-3}" \
  --num-workers "${NUM_WORKERS:-0}" --depth-scale "${DEPTH_SCALE:-2.0}" \
  "${extra[@]}" "$@" \
  2>&1 | tee "${OUTPUT}/logs/evaluate.log"
