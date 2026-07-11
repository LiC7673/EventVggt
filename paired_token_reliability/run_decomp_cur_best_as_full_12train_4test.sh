#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT="${OUTPUT:-abl_event_exp/decomp_cur_best_as_full_12train_4test}"
EPOCHS_A="${EPOCHS_A:-5}"
EPOCHS_B="${EPOCHS_B:-10}"
EPOCHS_C="${EPOCHS_C:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DECOMP_WEIGHT="${DECOMP_WEIGHT:-0.2}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUTPUT}/logs"

# Input/denominator: cur_best_event.  Only geometry_motion is additionally
# opened; events_additive/full is never probed in this mode.
python -m paired_token_reliability.train_unified_geometry_contribution \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a "${EPOCHS_A}" --epochs-b "${EPOCHS_B}" --epochs-c "${EPOCHS_C}" \
  --num-workers "${NUM_WORKERS}" --decomposition-weight "${DECOMP_WEIGHT}" \
  "data.num_views=${NUM_VIEWS:-4}" \
  "+data.train_initial_scene_idx=0" \
  "+data.train_scene_count=12" \
  "+data.train_holdout_frame_count=0" \
  "+data.test_initial_scene_idx=12" \
  "+data.test_scene_count=4" \
  "+data.heldout_test_frame_count=120" \
  "+data.event_source_mode=cur_best" \
  "+data.decomposition_supervision=true" \
  "+data.decomposition_event_root=events_additive" \
  "+data.decomposition_geo_branch=geometry_motion" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

