#!/usr/bin/env bash
# Train five independent RGB-only models and immediately evaluate each model.
# Exposure/GPU mapping: ev_0->1, ev_1->2, ev_2->3, ev_5->4, ev_10->5.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/rgb_finetune_ev012510_7scenes_1epoch_safe}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEPTH_SCALE="${DEPTH_SCALE:-2.05}"
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
  "DH2_Socrates and Seneca_Car_Paint_Midnight"
  "Dragon_1_Car_Paint_Midnight"
  "NAPOLEON_fix_Anodized_Red"
)

mkdir -p "${OUTPUT_ROOT}/logs"

run_one() {
  local exposure="$1"
  local gpu="$2"
  local ldr="ev_${exposure}"
  local name="rgb_finetune_${ldr}_7scenes_1epoch"
  local experiment="${OUTPUT_ROOT}/${name}"
  local train_log="${OUTPUT_ROOT}/logs/train_${ldr}_gpu${gpu}.log"
  local test_log="${OUTPUT_ROOT}/logs/test_${ldr}_gpu${gpu}.log"

  echo "[GPU ${gpu}] ${ldr}: RGB-only train, 7 scenes, 1 epoch, disjoint clips"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  HYDRA_FULL_ERROR=1 \
  python fine_rgb/finetune_rgb_seven_scenes_once.py \
    pretrained="${PRETRAINED}" \
    save_dir="${OUTPUT_ROOT}" \
    exp_name="${name}" \
    data.root="${DATA_ROOT}" \
    data.ldr_event_id="${ldr}" \
    data.num_views="${NUM_VIEWS}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    epochs=1 start_epoch=0 \
    batch_size=1 accum_iter=1 \
    mixed_precision="${AMP}" \
    num_workers="${NUM_WORKERS}" pin_mem=true \
    eval_every_steps=0 +skip_final_eval=true \
    save_every_steps=100000000 vis.save_every_steps=0 \
    2>&1 | tee "${train_log}"

  local checkpoint="${experiment}/checkpoint-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[GPU ${gpu}] missing checkpoint: ${checkpoint}" >&2
    return 3
  fi

  echo "[GPU ${gpu}] ${ldr}: per-scene depth/normal/pose test + visualization"
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
    --output-dir="${experiment}/test_${ldr}_seven_scenes" \
    2>&1 | tee "${test_log}"
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

echo "All five RGB-only models and tests finished: ${OUTPUT_ROOT}"
