#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
OUTPUT="${OUTPUT:-exp_f/direct_full_event_no_strategy_12train_3epoch}"
TRAIN_EV="${TRAIN_EV:-5}"
DEPTH_SCALE="${DEPTH_SCALE:-2.0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT}/logs"

# One GPU, 12 scene-disjoint training scenes, exactly three epochs.
# E_full is read only from events_additive/full/events.h5.
python -m paired_token_reliability.train_direct_full_event_baseline \
  pretrained="${PRETRAINED}" \
  data.root="${DATA_ROOT}" \
  data.ldr_event_id="${TRAIN_EV}" \
  data.num_views=4 \
  +data.train_initial_scene_idx=0 \
  +data.train_scene_count=12 \
  +data.train_holdout_frame_count=0 \
  +data.test_initial_scene_idx=12 \
  +data.test_scene_count=4 \
  +data.heldout_test_frame_count="${TEST_FRAME_COUNT}" \
  epochs=3 \
  output_dir="${OUTPUT}" \
  logdir="${OUTPUT}/logs" \
  save_dir="${OUTPUT}" \
  exp_name="direct_full_event_no_strategy" \
  num_workers="${NUM_WORKERS}" \
  eval_every_steps=0 \
  +skip_final_eval=true \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

CHECKPOINT="${OUTPUT}/checkpoint-last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

# Immediate held-out evaluation on four fixed scenes at every exposure.
for EV in 0 1 2 5 10; do
  EVAL_OUT="${OUTPUT}/test_four_scenes/ev_${EV}"
  python -m paired_token_reliability.evaluate_direct_full_event_four_scenes \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${EVAL_OUT}" \
    --root "${DATA_ROOT}" \
    --exposure "ev_${EV}" \
    --test-frame-count "${TEST_FRAME_COUNT}" \
    --num-views 4 \
    --num-workers "${NUM_WORKERS}" \
    --depth-scale "${DEPTH_SCALE}" \
    2>&1 | tee "${OUTPUT}/logs/test_ev_${EV}.log"
done

echo "Training checkpoint: ${CHECKPOINT}"
echo "Four-scene test results: ${OUTPUT}/test_four_scenes"
