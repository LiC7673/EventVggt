#!/usr/bin/env bash
# One-click pure-RGB LDR finetuning ablation.
#
# Each exposure gets an independent copy of the original RGB checkpoint.  The
# event stream is never loaded: ev_* selects only the matching LDR RGB folder.
# Five jobs are distributed across GPU 0/1, one job per GPU at a time.  Every
# model sees the 12 training scenes for exactly one epoch and is then evaluated
# on the same four held-out scenes at its own exposure level.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
PRETRAINED="${PRETRAINED:-ckpt/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_f/rgb_finetune_all_ev_12train_4test_gpu01}"
EXPOSURES_CSV="${EXPOSURES:-0,1,2,5,10}"
GPU_LIST_CSV="${GPU_LIST:-0,1}"
NUM_VIEWS="${NUM_VIEWS:-4}"
TEST_FRAME_COUNT="${TEST_FRAME_COUNT:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEPTH_SCALE="${DEPTH_SCALE:-2.0}"
AMP="${AMP:-bf16}"

IFS=',' read -r -a EXPOSURES_ARRAY <<< "${EXPOSURES_CSV}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST_CSV}"
if (( ${#GPU_ARRAY[@]} != 2 )); then
  echo "GPU_LIST must contain exactly two GPUs, for example GPU_LIST=0,1" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

SCENES=(
  "Centaur_Anodized_Red"
  "Child_with_goose_Industrial_Plastic_Grey"
  "Colchester Sphinx_Old_Copper"
  "Cupid as Shepherd_100MB_Old_Copper"
)

run_one() {
  local exposure="$1"
  local gpu="$2"
  local ldr="ev_${exposure#ev_}"
  local experiment="${OUTPUT_ROOT}/rgb_finetune_${ldr}_1epoch_12scenes"
  local train_log="${OUTPUT_ROOT}/logs/train_${ldr}_gpu${gpu}.log"
  local eval_log="${OUTPUT_ROOT}/logs/eval_${ldr}_gpu${gpu}.log"

  mkdir -p "${experiment}"
  echo "[GPU ${gpu}] train ${ldr}: 12 scenes, one epoch"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  HYDRA_FULL_ERROR=1 \
  python fine_rgb/finetune_rgb_ldr.py \
    pretrained="${PRETRAINED}" \
    save_dir="${OUTPUT_ROOT}" \
    exp_name="rgb_finetune_${ldr}_1epoch_12scenes" \
    data.root="${DATA_ROOT}" \
    data.ldr_event_id="${ldr}" \
    data.initial_scene_idx=0 \
    data.active_scene_count=12 \
    data.num_views="${NUM_VIEWS}" \
    data.test_frame_count="${TEST_FRAME_COUNT}" \
    epochs=1 start_epoch=0 \
    batch_size=1 accum_iter=1 \
    mixed_precision="${AMP}" \
    num_workers="${NUM_WORKERS}" pin_mem=true \
    eval_every_steps=0 skip_final_eval=true \
    save_every_steps=100000000 vis.save_every_steps=0 \
    2>&1 | tee "${train_log}"

  local checkpoint="${experiment}/checkpoint-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint after training: ${checkpoint}" >&2
    return 3
  fi

  echo "[GPU ${gpu}] test ${ldr}: fixed four scenes"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  python -m fine_rgb.evaluate_rgb_four_scenes_streaming \
    --skip-pretrained \
    --finetuned-template "${checkpoint}" \
    --data-root "${DATA_ROOT}" \
    --scenes "${SCENES[@]}" \
    --ldr-event-ids "${exposure#ev_}" \
    --num-views "${NUM_VIEWS}" \
    --test-frame-count "${TEST_FRAME_COUNT}" \
    --batch-size 1 --num-workers "${NUM_WORKERS}" \
    --amp "${AMP}" --depth-scale "${DEPTH_SCALE}" \
    --visualize-every "${VISUALIZE_EVERY:-0}" \
    --max-visuals-per-condition "${MAX_VISUALS_PER_CONDITION:-0}" \
    --output-dir "${experiment}/test_four_scenes" \
    2>&1 | tee "${eval_log}"
}

worker() {
  local worker_index="$1"
  local gpu="${GPU_ARRAY[$worker_index]}"
  local index
  for ((index=worker_index; index<${#EXPOSURES_ARRAY[@]}; index+=2)); do
    run_one "${EXPOSURES_ARRAY[$index]}" "${gpu}"
  done
}

echo "Output: ${OUTPUT_ROOT}"
echo "Exposure jobs: ${EXPOSURES_ARRAY[*]}"
echo "GPU workers: ${GPU_ARRAY[*]}"
worker 0 & pid0=$!
worker 1 & pid1=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
if (( status != 0 )); then
  echo "At least one exposure job failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

echo "All five RGB finetune/evaluation jobs finished: ${OUTPUT_ROOT}"
