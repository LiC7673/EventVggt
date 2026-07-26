#!/usr/bin/env bash
# Cur-event geometry pipeline with a spatial event-conditioned final pose head.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

GPU="${GPU:-4}"
OUTPUT="${OUTPUT:-exp_f/cur_event_spatial_pose_refine_v2_gpu4}"
DEFAULT_PRETRAINED="exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-best.pth"
if [[ ! -f "${DEFAULT_PRETRAINED}" ]]; then
  DEFAULT_PRETRAINED="exp_f/cur_event_refiner_first_1k_then_joint_gpu4/checkpoint-adapter-last.pth"
fi
PRETRAINED="${PRETRAINED:-${DEFAULT_PRETRAINED}}"
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing initialization checkpoint: ${PRETRAINED}" >&2
  echo "Set PRETRAINED explicitly if the geometry checkpoint is elsewhere." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export POSE_REFINE_WEIGHT="${POSE_REFINE_WEIGHT:-1.0}"
export POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-0.25}"
export POSE_RESIDUAL_REG_WEIGHT="${POSE_RESIDUAL_REG_WEIGHT:-0.0001}"
export CROSS_VIEW_DN_WEIGHT="${CROSS_VIEW_DN_WEIGHT:-0.20}"
mkdir -p "${OUTPUT}/logs"

python -m torch.distributed.run --nproc_per_node 1 \
  --master_port "${MASTER_PORT:-29547}" \
  -m paired_token_reliability.train_linear_voxel_cur_event_pose_refine_v2 \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --epochs-a 1 --epochs-b "${EPOCHS_B:-6}" --epochs-c 0 \
  --first-adapter-max-batches "${REFINER_WARMUP_STEPS:-1000}" \
  --lr "${LR:-0.0001}" --num-workers "${NUM_WORKERS:-2}" \
  --point-weight 0 --pair-weight 0 --decomposition-weight 0 \
  --event-normal-weight 1 --depth-event-normal-weight .5 \
  --update-weight 0 --no-budget \
  --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
  --visualize-val-every-batches "${VAL_VIS_EVERY:-5}" \
  "data.num_views=4" "data.event_resize_bins=5" \
  "data.event_resize_method=voxel_linear_time" \
  "data.train_initial_scene_idx=0" "data.train_scene_count=12" \
  "data.train_holdout_frame_count=0" \
  "data.test_initial_scene_idx=12" "data.test_scene_count=4" \
  "data.heldout_test_frame_count=120" \
  "data.decomposition_event_root=events_additive" \
  "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
  "model.pixel_refiner_delay=0" \
  "+model.pose_feature_dim=${POSE_FEATURE_DIM:-96}" \
  "+model.pose_refiner_hidden=${POSE_HIDDEN:-192}" \
  "+model.pose_refiner_delay=${POSE_DELAY:-0}" \
  "+model.pose_refiner_transition=${POSE_TRANSITION:-500}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_TEST_AFTER_TRAIN:-1}" == 1 ]]; then
  GPU="${GPU}" EXP_DIR="${OUTPUT}" \
    bash paired_token_reliability/run_test_cur_event_spatial_pose_refine_v2_gpu0.sh
fi

echo "[done] ${OUTPUT}"
