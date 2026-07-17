#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
ABLATION_ROOT="${ABLATION_ROOT:-exp_f/latest_three_strategy_ablation_3epoch}"
mkdir -p "${ABLATION_ROOT}/launcher_logs"
# Total budget is 1 + 2 * EPOCHS_B.  The paper ablations default to exactly
# three epochs: A->B->A for the full-minus component variant, and three
# adapter epochs for the single-strategy variants.
EPOCHS_B="${EPOCHS_B:-1}"
if [[ "${EPOCHS_B}" -ne 1 ]]; then
  echo "[warning] EPOCHS_B=${EPOCHS_B}; total epochs will be $((1 + 2 * EPOCHS_B)), not 3" >&2
else
  echo "[ablation budget] exactly 3 epochs per experiment"
fi

run_one() {
  local variant="$1" gpu="$2" port="$3"
  CUDA_VISIBLE_DEVICES="$gpu" ABLATION_VARIANT="$variant" \
  python -m torch.distributed.run --nproc_per_node 1 --master_port "$port" \
   -m paired_token_reliability.train_latest_strategy_ablation \
   --pretrained "${PRETRAINED:-ckpt/model.pt}" \
   --output "${ABLATION_ROOT}/${variant}" \
   --epochs-a 1 --epochs-b "${EPOCHS_B}" --epochs-c 0 \
   --lr "${LR:-0.0001}" --num-workers "${NUM_WORKERS:-2}" \
   --point-weight 0 --pair-weight 0 --decomposition-weight 0 --no-budget \
   --event-normal-weight 1 --depth-event-normal-weight .5 --update-weight 0 \
   --visualize-every-batches "${TRAIN_VIS_EVERY:-100}" \
   --visualize-val-every-batches "${VAL_VIS_EVERY:-20}" \
   "data.num_views=${NUM_VIEWS:-4}" "data.event_resize_bins=5" \
   "data.event_resize_method=voxel_linear_time" \
   "data.train_initial_scene_idx=0" "data.train_scene_count=12" \
   "data.train_holdout_frame_count=0" "data.test_initial_scene_idx=12" \
   "data.test_scene_count=4" "data.heldout_test_frame_count=120" \
   "data.decomposition_event_root=events_additive" \
   "model.head_frames_chunk_size=${HEAD_CHUNK:-1}" \
   "model.pixel_refiner_delay=0" "model.pixel_refine_log_limit=0.30" \
   "model.c_delay_steps=1000" "model.c_transition_steps=1000" \
   > "${ABLATION_ROOT}/launcher_logs/${variant}.log" 2>&1 &
  echo "$! $variant GPU=$gpu"
}

base="${MASTER_PORT_BASE:-29670}"
run_one noisy_event_only 0 "$base"
run_one multi_ldr_only 1 "$((base+1))"
run_one without_refiner_normal 2 "$((base+2))"
wait

if [[ "${AUTO_TEST:-1}" == "1" ]]; then
  echo "[auto-test] training completed; launching variant-correct four-scene/all-EV tests"
  ABLATION_ROOT="${ABLATION_ROOT}" \
    bash paired_token_reliability/run_test_latest_three_ablations_gpu012.sh
else
  echo "[auto-test] disabled (AUTO_TEST=${AUTO_TEST:-0})"
fi
