#!/usr/bin/env bash
# Five independent RGB-only models: four scenes, three epochs, pose supervised.
# Exposure/GPU mapping: ev_0->1, ev_1->2, ev_2->3, ev_5->4, ev_10->5.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/rgb_finetune_ev012510_4scenes_3epochs_relative_pose_fixed}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEPTH_SCALE="${DEPTH_SCALE:-2.0}"
AMP="${AMP:-bf16}"
VISUALIZE_EVERY="${VISUALIZE_EVERY:-1}"
MAX_VISUALS_PER_CONDITION="${MAX_VISUALS_PER_CONDITION:-0}"

EXPOSURES=(0 1 2 5 10)
GPUS=(1 2 3 4 5)
SCENES=(
  "Centaur_Anodized_Red"
  "Child_with_goose_Industrial_Plastic_Grey"
  "Colchester Sphinx_Old_Copper"
  "Cupid as Shepherd_100MB_Old_Copper"
)

mkdir -p "${OUTPUT_ROOT}/logs"

run_one() {
  local exposure="$1"
  local gpu="$2"
  local ldr="ev_${exposure}"
  local name="rgb_finetune_${ldr}_4scenes_3epochs_relative_pose_fixed"
  local experiment="${OUTPUT_ROOT}/${name}"

  echo "[GPU ${gpu}] ${ldr}: four-scene RGB-only training for 3 epochs"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  HYDRA_FULL_ERROR=1 \
  python fine_rgb/finetune_rgb_four_scenes_three_epochs.py \
    pretrained="${PRETRAINED}" \
    save_dir="${OUTPUT_ROOT}" exp_name="${name}" \
    data.root="${DATA_ROOT}" data.ldr_event_id="${ldr}" \
    data.num_views="${NUM_VIEWS}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    epochs=3 start_epoch=0 batch_size=1 accum_iter=1 \
    lr="${LR:-1.0e-5}" min_lr="${MIN_LR:-1.0e-6}" \
    warmup_epochs="${WARMUP_EPOCHS:-0.1}" \
    loss.pose_weight="${POSE_WEIGHT:-1.0}" \
    loss.align_depth_scale=true \
    train.unfreeze_heads=true \
    train.unfreeze_aggregator_blocks=false \
    mixed_precision="${AMP}" num_workers="${NUM_WORKERS}" pin_mem=true \
    eval_every_steps=0 +skip_final_eval=true \
    save_every_steps=100000000 vis.save_every_steps=0 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${ldr}_gpu${gpu}.log"

  local checkpoint="${experiment}/checkpoint-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    return 3
  fi

  echo "[GPU ${gpu}] ${ldr}: testing four scenes at ${ldr}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
    --skip-pretrained \
    --finetuned-template "${checkpoint}" \
    --data-root "${DATA_ROOT}" \
    --scenes "${SCENES[@]}" \
    --ldr-event-ids "${exposure}" \
    --num-views "${NUM_VIEWS}" \
    --test-frame-count "${TEST_FRAME_COUNT}" \
    --batch-size 1 --num-workers "${NUM_WORKERS}" \
    --amp "${AMP}" --depth-scale "${DEPTH_SCALE}" \
    --visualize-every "${VISUALIZE_EVERY}" \
    --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION}" \
    --output-dir="${experiment}/test_${ldr}_four_scenes" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/test_${ldr}_gpu${gpu}.log"
}

pids=()
for index in "${!EXPOSURES[@]}"; do
  run_one "${EXPOSURES[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "At least one job failed. Inspect ${OUTPUT_ROOT}/logs." >&2
  exit "${status}"
fi

echo "All five four-scene/three-epoch RGB experiments finished: ${OUTPUT_ROOT}"
