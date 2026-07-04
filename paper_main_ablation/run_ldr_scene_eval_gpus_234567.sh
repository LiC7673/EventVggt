#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_POOL="${GPU_POOL:-2 3 4 5 6 7}"
EVAL_VARIANTS="${EVAL_VARIANTS:-a0_rgb_only a1_direct_event a2_wo_reliability a3_wo_multildr a4_wo_detail a5_full}"
LDR_LEVELS="${LDR_LEVELS:-1 2 5 10}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT:-${ROOT_DIR}/abl_event_exp/paper_module_ablation_extrel_normal}}"
RESULT_ROOT="${RESULT_ROOT:-${CHECKPOINT_ROOT}/test_4scenes_ldr_1_2_5_10}"
DATA_ROOT="${DATA_ROOT:-/data1/lzh/dataset/reflective_raw}"
INITIAL_SCENE_IDX="${INITIAL_SCENE_IDX:-12}"
SCENE_COUNT="${SCENE_COUNT:-4}"
NUM_VIEWS="${NUM_VIEWS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VISUAL_BATCHES="${VISUAL_BATCHES:-1}"
VISUAL_VIEWS="${VISUAL_VIEWS:-4}"
MAX_BATCHES="${MAX_BATCHES:-}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${CHECKPOINT_ROOT}/scene_manifest.json}"

read -r -a GPU_ARRAY <<< "${GPU_POOL}"
read -r -a VARIANT_ARRAY <<< "${EVAL_VARIANTS}"
read -r -a LDR_ARRAY <<< "${LDR_LEVELS}"
mkdir -p "${RESULT_ROOT}/logs"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${SCENE_MANIFEST}" ]]; then
  python -m paper_main_ablation.make_eval_scene_manifest \
    --root "${DATA_ROOT}" \
    --output "${SCENE_MANIFEST}" \
    --train-scene-count "${INITIAL_SCENE_IDX}" \
    --test-scene-count "${SCENE_COUNT}" \
    --ldr-levels "${LDR_ARRAY[@]}" \
    --num-views "${NUM_VIEWS}"
fi

display_name() {
  case "$1" in
    a0_rgb_only) echo "A0_RGB_only" ;;
    a1_direct_event) echo "A1_Direct_event" ;;
    a2_wo_reliability) echo "A2_w_o_Reliability" ;;
    a3_wo_multildr) echo "A3_w_o_MultiLDR" ;;
    a4_wo_detail) echo "A4_w_o_Detail" ;;
    a5_full) echo "A5_Full" ;;
    *) echo "$1" ;;
  esac
}

JOBS=()
for variant in "${VARIANT_ARRAY[@]}"; do
  checkpoint="${CHECKPOINT_ROOT}/${variant}/checkpoint-last.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  for ldr in "${LDR_ARRAY[@]}"; do
    JOBS+=("${variant}|${ldr}")
  done
done

launch_job() {
  local job="$1"
  local gpu="$2"
  local variant="${job%%|*}"
  local ldr="${job##*|}"
  local name
  name="$(display_name "${variant}")"
  local output="${RESULT_ROOT}/${variant}/ev_${ldr}"
  local log="${RESULT_ROOT}/logs/${variant}_ev_${ldr}_gpu_${gpu}.log"
  local extra=()
  if [[ -n "${MAX_BATCHES}" ]]; then
    extra+=(--max-batches "${MAX_BATCHES}")
  fi
  mkdir -p "${output}"
  echo "[eval launch] ${name} LDR=ev_${ldr} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m paper_main_ablation.evaluate_per_scene \
    --checkpoint "${CHECKPOINT_ROOT}/${variant}/checkpoint-last.pth" \
    --name "${name}" \
    --output-dir "${output}" \
    --root "${DATA_ROOT}" \
    --ldr-event-id "ev_${ldr}" \
    --initial-scene-idx "${INITIAL_SCENE_IDX}" \
    --scene-count "${SCENE_COUNT}" \
    --scene-manifest "${SCENE_MANIFEST}" \
    --num-views "${NUM_VIEWS}" \
    --event-resize-bins 10 \
    --num-workers "${NUM_WORKERS}" \
    --visual-batches "${VISUAL_BATCHES}" \
    --visual-views "${VISUAL_VIEWS}" \
    --device cuda:0 \
    "${extra[@]}" \
    > "${log}" 2>&1 &
  LAUNCHED_PID="$!"
}

job_index=0
while [[ "${job_index}" -lt "${#JOBS[@]}" ]]; do
  pids=()
  labels=()
  for gpu in "${GPU_ARRAY[@]}"; do
    [[ "${job_index}" -ge "${#JOBS[@]}" ]] && break
    job="${JOBS[job_index]}"
    job_index=$((job_index + 1))
    launch_job "${job}" "${gpu}"
    pids+=("${LAUNCHED_PID}")
    labels+=("${job}")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[index]}"; then
      echo "[eval done] ${labels[index]}"
    else
      echo "[eval fail] ${labels[index]}; inspect ${RESULT_ROOT}/logs" >&2
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || exit 1
done

python -m paper_main_ablation.collect_ldr_scene_results \
  --results-root "${RESULT_ROOT}" \
  --expected-models "${#VARIANT_ARRAY[@]}" \
  --expected-ldrs "${#LDR_ARRAY[@]}" \
  --expected-scenes "${SCENE_COUNT}"

echo "[done] per-scene table: ${RESULT_ROOT}/all_scene_metrics.csv"
echo "[done] per-LDR table:   ${RESULT_ROOT}/mean_metrics_by_model_ldr.csv"
echo "[done] visualizations:  ${RESULT_ROOT}/<variant>/ev_<LDR>/visuals/"
