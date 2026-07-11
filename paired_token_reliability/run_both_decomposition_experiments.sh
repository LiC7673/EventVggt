#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# The two jobs are completely independent. Never overlap these GPU lists.
CUR_BEST_GPUS="${CUR_BEST_GPUS:-0}"
FULL_GPUS="${FULL_GPUS:-6,7}"
if [[ -z "${CUR_BEST_PORT:-}" ]]; then
  CUR_BEST_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi
if [[ -z "${FULL_PORT:-}" ]]; then
  FULL_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi

EPOCHS_A="${EPOCHS_A:-10}"
EPOCHS_B="${EPOCHS_B:-20}"
EPOCHS_C="${EPOCHS_C:-3}"
NUM_VIEWS="${NUM_VIEWS:-6}"
HEAD_CHUNK="${HEAD_CHUNK:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_EVAL="${RUN_EVAL:-1}"

CUR_BEST_EXP_NAME="${CUR_BEST_EXP_NAME:-decomp_cur_best_as_full_12train_4test}"
FULL_EXP_NAME="${FULL_EXP_NAME:-decomp_full_as_event_12train_4test}"

IFS=',' read -r -a CUR_GPU_ARRAY <<< "${CUR_BEST_GPUS}"
IFS=',' read -r -a FULL_GPU_ARRAY <<< "${FULL_GPUS}"
for left in "${CUR_GPU_ARRAY[@]}"; do
  for right in "${FULL_GPU_ARRAY[@]}"; do
    if [[ "${left}" == "${right}" ]]; then
      echo "[error] GPU ${left} appears in both experiments." >&2
      exit 2
    fi
  done
done

mkdir -p "exp/${CUR_BEST_EXP_NAME}/logs" "exp/${FULL_EXP_NAME}/logs"

echo "[launch] cur_best-as-full on GPUs ${CUR_BEST_GPUS}, port ${CUR_BEST_PORT}"
GPUS="${CUR_BEST_GPUS}" MASTER_PORT="${CUR_BEST_PORT}" \
EXP_NAME="${CUR_BEST_EXP_NAME}" \
EPOCHS_A="${EPOCHS_A}" EPOCHS_B="${EPOCHS_B}" EPOCHS_C="${EPOCHS_C}" \
NUM_VIEWS="${NUM_VIEWS}" HEAD_CHUNK="${HEAD_CHUNK}" \
NUM_WORKERS="${NUM_WORKERS}" RUN_EVAL="${RUN_EVAL}" \
bash paired_token_reliability/run_decomp_cur_best_as_full_12train_4test.sh \
  > "exp/${CUR_BEST_EXP_NAME}/logs/launcher.log" 2>&1 &
CUR_PID=$!

echo "[launch] decomposition-full on GPUs ${FULL_GPUS}, port ${FULL_PORT}"
GPUS="${FULL_GPUS}" MASTER_PORT="${FULL_PORT}" \
EXP_NAME="${FULL_EXP_NAME}" \
EPOCHS_A="${EPOCHS_A}" EPOCHS_B="${EPOCHS_B}" EPOCHS_C="${EPOCHS_C}" \
NUM_VIEWS="${NUM_VIEWS}" HEAD_CHUNK="${HEAD_CHUNK}" \
NUM_WORKERS="${NUM_WORKERS}" RUN_EVAL="${RUN_EVAL}" \
bash paired_token_reliability/run_decomp_full_as_event_12train_4test.sh \
  > "exp/${FULL_EXP_NAME}/logs/launcher.log" 2>&1 &
FULL_PID=$!

echo "[running] cur_best PID=${CUR_PID}; full PID=${FULL_PID}"
echo "[logs] tail -f exp/${CUR_BEST_EXP_NAME}/logs/launcher.log"
echo "[logs] tail -f exp/${FULL_EXP_NAME}/logs/launcher.log"

status=0
wait "${CUR_PID}" || status=$?
wait "${FULL_PID}" || status=$?
echo "[done] both decomposition experiments finished (status=${status})"
exit "${status}"
