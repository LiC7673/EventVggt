#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-6,7}"
EXP_NAME="${EXP_NAME:-stage2_causal_event_reliability_train12_test4}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
LAST_GPU_INDEX=$((${#GPU_ARRAY[@]} - 1))
EVAL_GPU="${EVAL_GPU:-${GPU_ARRAY[${LAST_GPU_INDEX}]}}"

echo "[pipeline] train causal event model on GPUs ${GPUS}"
GPUS="${GPUS}" EXP_NAME="${EXP_NAME}" \
bash real_reliability_stage/run_stage2_causal_event_train.sh "$@"

echo "[pipeline] evaluate four unseen scenes on GPU ${EVAL_GPU}"
GPU="${EVAL_GPU}" EXP_NAME="${EXP_NAME}" \
bash real_reliability_stage/run_stage2_causal_event_eval.sh
