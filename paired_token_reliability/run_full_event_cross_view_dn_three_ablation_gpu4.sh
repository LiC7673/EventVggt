#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
if [[ "${#GPU_LIST[@]}" -ne 4 ]]; then
  echo "GPUS must contain exactly four devices, e.g. GPUS=0,1,2,3" >&2
  exit 2
fi
BASE_OUTPUT="${BASE_OUTPUT:-exp_f/full_event_cross_view_dn_ablation_gpu0123}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CROSS_VIEW_DN_WEIGHT="${CROSS_VIEW_DN_WEIGHT:-0.20}"
export CROSS_VIEW_PATCH_SIZE="${CROSS_VIEW_PATCH_SIZE:-14}"
export CROSS_VIEW_MIN_OVERLAP="${CROSS_VIEW_MIN_OVERLAP:-8}"

run_variant() {
  local variant="$1"
  local gpu="$2"
  local branch="$3"
  local port="$4"
  shift 4
  local output="${BASE_OUTPUT}/${variant}"
  mkdir -p "${output}/logs"

  echo "[RUN] variant=${variant} GPU=${gpu} event_branch=${branch} output=${output}"
  FULL_DN_ABLATION="${variant}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m torch.distributed.run --nproc_per_node 1 --master_port "${port}" \
    -m paired_token_reliability.train_full_event_cross_view_dn_ablation \
    --pretrained "${PRETRAINED:-ckpt/model.pt}" \
    --output "${output}" \
    --epochs-a 3 --epochs-b 0 --epochs-c 0 \
    --first-adapter-max-batches "${MAX_BATCHES_PER_EPOCH:--1}" \
    --lr "${LR:-0.0001}" \
    --num-workers "${NUM_WORKERS:-2}" \
    --point-weight 0 --pair-weight 0 --decomposition-weight 0 \
    --event-normal-weight 1 --depth-event-normal-weight .5 \
    --update-weight 0 --no-budget \
    --visualize-every-batches "${TRAIN_VIS_EVERY:-40}" \
    --visualize-val-every-batches "${VAL_VIS_EVERY:-5}" \
    "data.num_views=4" \
    "data.event_resize_bins=5" \
    "data.event_resize_method=voxel_linear_time" \
    "data.train_initial_scene_idx=0" \
    "data.train_scene_count=12" \
    "data.train_holdout_frame_count=0" \
    "data.test_initial_scene_idx=12" \
    "data.test_scene_count=4" \
    "data.heldout_test_frame_count=120" \
    "data.decomposition_event_root=events_additive" \
    "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
    "model.pixel_refiner_delay=0" \
    "model.pixel_refine_log_limit=0.30" \
    "model.event_decay_tau=0.0015" \
    "$@" 2>&1 | tee "${output}/logs/train.log"

  local checkpoint="${output}/checkpoint-adapter-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] missing checkpoint after ${variant}: ${checkpoint}" >&2
    return 3
  fi

  echo "[TEST] variant=${variant} GPU=${gpu} checkpoint=${checkpoint}"
  FULL_DN_EVENT_BRANCH="${branch}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m paired_token_reliability.evaluate_full_event_cross_view_dn_ablation \
    --checkpoint "${checkpoint}" \
    --output-dir "${output}/test_four_scenes" \
    --root "${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}" \
    --event-source-mode decomposition_full \
    --exposures "${EXPOSURES:-0,1,2,5,10}" \
    --num-views 4 \
    --test-frame-count "${TEST_FRAME_COUNT:-120}" \
    --batch-size 1 \
    --num-workers "${TEST_NUM_WORKERS:-0}" \
    --depth-scale "${DEPTH_SCALE:-2.0}" \
    --visualize-every "${VISUALIZE_EVERY:-1}" \
    --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" \
    2>&1 | tee "${output}/logs/test_four_scenes.log"
}

# Four independent jobs: training and the subsequent matched evaluation stay
# on the same assigned GPU.  wait propagates a failure from any child.
variants=(full no_hdr_align no_refiner_loss geo_only)
branches=(full full full geometry_motion)
pids=()
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29640}"
for index in 0 1 2 3; do
  run_variant \
    "${variants[$index]}" "${GPU_LIST[$index]}" "${branches[$index]}" \
    "$((MASTER_PORT_BASE + index))" "$@" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "[FAILED] at least one training/testing job failed; inspect ${BASE_OUTPUT}/*/logs" >&2
  exit "${status}"
fi

echo "[DONE] ${BASE_OUTPUT}/{full,no_hdr_align,no_refiner_loss,geo_only}"
