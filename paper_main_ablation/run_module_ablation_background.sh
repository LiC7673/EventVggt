#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MASTER_DIR="${MASTER_DIR:-${ROOT_DIR}/ablation_logs/paper_module_ablation_master}"
MASTER_LOG="${MASTER_LOG:-${MASTER_DIR}/${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${MASTER_DIR}/${RUN_ID}.pid}"
mkdir -p "${MASTER_DIR}"

nohup bash paper_main_ablation/run_train_and_eval.sh "$@" \
  > "${MASTER_LOG}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "[background] PID=${pid}"
echo "[background] log=${MASTER_LOG}"
echo "[background] pid_file=${PID_FILE}"
echo "[monitor] tail -f ${MASTER_LOG}"
