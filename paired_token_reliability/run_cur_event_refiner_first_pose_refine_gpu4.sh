#!/usr/bin/env bash
# Latest cur-event geometry pipeline plus delayed supervised pose refinement.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

GPU="${GPU:-4}"
OUTPUT="${OUTPUT:-exp_f/cur_event_refiner_first_pose_refine_gpu4}"
GPUS="${GPUS:-${GPU}}"; IFS=',' read -r -a gpu_array <<< "${GPUS}"
NPROC="${#gpu_array[@]}"
PORT="${MASTER_PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export POSE_REFINE_WEIGHT="${POSE_REFINE_WEIGHT:-0.10}"
export POSE_ROTATION_WEIGHT="${POSE_ROTATION_WEIGHT:-0.50}"
export POSE_RESIDUAL_REG_WEIGHT="${POSE_RESIDUAL_REG_WEIGHT:-0.001}"
export CROSS_VIEW_DN_WEIGHT="${CROSS_VIEW_DN_WEIGHT:-0.20}"
export CROSS_VIEW_PATCH_SIZE="${CROSS_VIEW_PATCH_SIZE:-14}"
export CROSS_VIEW_MIN_OVERLAP="${CROSS_VIEW_MIN_OVERLAP:-8}"
mkdir -p "${OUTPUT}/logs"

python -m torch.distributed.run --nproc_per_node "${NPROC}" --master_port "${PORT}" \
  -m paired_token_reliability.train_linear_voxel_cur_event_pose_refine \
  --pretrained "${PRETRAINED:-ckpt/model.pt}" \
  --output "${OUTPUT}" \
  --epochs-a 1 --epochs-b "${EPOCHS_B:-6}" --epochs-c 0 \
  --first-adapter-max-batches "${REFINER_WARMUP_STEPS:-1000}" \
  --lr "${LR:-0.0001}" --num-workers "${NUM_WORKERS:-2}" \
  --point-weight 0 --pair-weight 0 --decomposition-weight 0 \
  --event-normal-weight 1 --depth-event-normal-weight .5 \
  --update-weight 0 --no-budget \
  --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
  --visualize-val-every-batches "${VAL_VIS_EVERY:-5}" \
  "data.num_views=${NUM_VIEWS:-4}" \
  "data.event_resize_bins=5" \
  "data.event_resize_method=voxel_linear_time" \
  "data.train_initial_scene_idx=0" "data.train_scene_count=12" \
  "data.train_holdout_frame_count=0" \
  "data.test_initial_scene_idx=12" "data.test_scene_count=4" \
  "data.heldout_test_frame_count=120" \
  "data.decomposition_event_root=events_additive" \
  "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
  "model.pixel_refiner_delay=0" \
  "model.pixel_refine_log_limit=0.30" \
  "model.event_decay_tau=0.0015" \
  "+model.pose_refiner_delay=${POSE_DELAY:-1000}" \
  "+model.pose_refiner_transition=${POSE_TRANSITION:-1000}" \
  "$@" 2>&1 | tee "${OUTPUT}/logs/train.log"

if [[ "${RUN_TEST_AFTER_TRAIN:-1}" == 1 ]]; then
  checkpoint="${OUTPUT}/checkpoint-adapter-best.pth"
  [[ -f "${checkpoint}" ]] || checkpoint="${OUTPUT}/checkpoint-adapter-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Training finished but no adapter checkpoint was found in ${OUTPUT}" >&2
    exit 3
  fi
  echo "[POSE-REFINE] starting synthetic 4-scene x 5-exposure evaluation"
  CHECKPOINT="${checkpoint}" \
  OUTPUT="${OUTPUT}/test_four_scenes_pose" \
  GPU="${GPU}" \
  DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
  bash paired_token_reliability/run_test_cur_event_pose_refine_four_scenes_gpu0.sh
fi
