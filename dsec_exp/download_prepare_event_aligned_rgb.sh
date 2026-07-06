#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
WORK="${WORK:-${ROOT}/_dsec_detection_remapped}"
URL="${URL:-https://download.ifi.uzh.ch/rpg/DSEC/train_object_detection_coarse/train_left_images_distorted.zip}"
ARCHIVE="${WORK}/train_left_images_distorted.zip"
EXTRACTED="${WORK}/extracted"
WORKERS="${WORKERS:-8}"

mkdir -p "${WORK}" "${EXTRACTED}"
if [[ ! -s "${ARCHIVE}" ]] || ! unzip -tqq "${ARCHIVE}" >/dev/null 2>&1; then
  rm -f "${ARCHIVE}"
  echo "[download] DSEC-Detection event-view RGB"
  wget --continue --tries=20 --timeout=30 --waitretry=10 --show-progress -O "${ARCHIVE}.part" "${URL}"
  unzip -tqq "${ARCHIVE}.part"
  mv -f "${ARCHIVE}.part" "${ARCHIVE}"
fi

echo "[extract] ${ARCHIVE}"
unzip -oq "${ARCHIVE}" -d "${EXTRACTED}"

python -m dsec_exp.prepare_event_aligned_rgb \
  --root "${ROOT}" \
  --remapped-root "${EXTRACTED}" \
  --workers "${WORKERS}"

echo "Event-aligned RGB preparation complete."
