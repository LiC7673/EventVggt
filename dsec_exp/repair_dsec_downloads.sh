#!/usr/bin/env bash
set -Eeuo pipefail

# Audit and repair an existing DSEC download tree. The current val/test scene
# directories are preserved; this script does not reassign sequence splits.

ROOT="${1:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
BASE_URL="${BASE_URL:-https://download.ifi.uzh.ch/rpg/DSEC/train}"
KEEP_BAD="${KEEP_BAD:-0}"
EXTRACT="${EXTRACT:-0}"
TRIES="${TRIES:-20}"

WGET_OPTS=(
  --continue
  --tries="${TRIES}"
  --timeout=30
  --read-timeout=30
  --waitretry=10
  --retry-connrefused
  --show-progress
)

ARCHIVE_COMPONENTS=(
  calibration
  events_left
  images_rectified_left
  disparity_image
  disparity_event
)

TEXT_COMPONENTS=(
  image_timestamps
  image_exposure_timestamps_left
  disparity_timestamps
)

if [[ ! -d "${ROOT}" ]]; then
  echo "[error] dataset root does not exist: ${ROOT}" >&2
  exit 1
fi
if ! command -v wget >/dev/null 2>&1; then
  echo "[error] wget is required" >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "[error] unzip is required" >&2
  exit 1
fi

valid_zip() {
  local path="$1"
  [[ -s "${path}" ]] && unzip -tqq "${path}" >/dev/null 2>&1
}

valid_text() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  if head -c 512 "${path}" | grep -Eiq '<!doctype|<html|access denied|not found'; then
    return 1
  fi
  awk '
    NF {
      found=1
      for (i=1; i<=NF; i++) {
        if ($i !~ /^[-+]?[0-9]+([.][0-9]+)?$/) exit 1
      }
    }
    END { if (!found) exit 1 }
  ' "${path}" >/dev/null
}

valid_file() {
  local path="$1"
  case "${path,,}" in
    *.zip) valid_zip "${path}" ;;
    *.txt|*.csv) valid_text "${path}" ;;
    *) [[ -s "${path}" ]] ;;
  esac
}

discard_bad() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  if [[ "${KEEP_BAD}" == "1" ]]; then
    local backup="${path}.corrupt.$(date +%Y%m%d_%H%M%S)"
    mv -f "${path}" "${backup}"
    echo "[quarantine] ${path} -> ${backup}"
  else
    rm -f "${path}"
    echo "[remove bad] ${path}"
  fi
}

show_bad_type() {
  local path="$1"
  if command -v file >/dev/null 2>&1 && [[ -e "${path}" ]]; then
    file "${path}" >&2 || true
  fi
}

download_atomic() {
  local url="$1"
  local output="$2"
  local part="${output}.part"

  mkdir -p "$(dirname "${output}")"
  if valid_file "${output}"; then
    echo "[ok] ${output}"
    return 0
  fi

  if [[ -e "${output}" ]]; then
    echo "[corrupt] ${output}" >&2
    show_bad_type "${output}"
    discard_bad "${output}"
  fi
  if [[ -e "${part}" ]] && ! valid_file "${part}"; then
    echo "[resume] partial file exists: ${part}"
  elif [[ -e "${part}" ]]; then
    mv -f "${part}" "${output}"
    echo "[recovered complete part] ${output}"
    return 0
  fi

  echo "[download] ${url}"
  if ! wget "${WGET_OPTS[@]}" -O "${part}" "${url}"; then
    echo "[download failed] ${url}" >&2
    return 1
  fi
  if ! valid_file "${part}"; then
    echo "[invalid response] ${url}" >&2
    show_bad_type "${part}"
    echo "The server response is not the expected ZIP/text file. Check URL, connectivity, or access." >&2
    rm -f "${part}"
    return 1
  fi

  mv -f "${part}" "${output}"
  echo "[repaired] ${output}"
}

component_output() {
  local scene_root="$1"
  local sequence="$2"
  local component="$3"
  case "${component}" in
    calibration|events_left|images_rectified_left|disparity_image|disparity_event)
      printf '%s/%s/%s_%s.zip' "${scene_root}" "${component}" "${sequence}" "${component}"
      ;;
    image_timestamps|image_exposure_timestamps_left|disparity_timestamps)
      printf '%s/timestamps/%s_%s.txt' "${scene_root}" "${sequence}" "${component}"
      ;;
    *) return 1 ;;
  esac
}

repair_scene() {
  local scene_root="$1"
  local sequence
  sequence="$(basename "${scene_root}")"
  local sequence_url="${BASE_URL}/${sequence}"

  echo
  echo "============================================================"
  echo "Audit scene: ${scene_root}"
  echo "============================================================"

  local component output filename
  for component in "${ARCHIVE_COMPONENTS[@]}"; do
    output="$(component_output "${scene_root}" "${sequence}" "${component}")"
    filename="$(basename "${output}")"
    if ! download_atomic "${sequence_url}/${filename}" "${output}"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  for component in "${TEXT_COMPONENTS[@]}"; do
    output="$(component_output "${scene_root}" "${sequence}" "${component}")"
    filename="$(basename "${output}")"
    if ! download_atomic "${sequence_url}/${filename}" "${output}"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
}

FAILURES=0
SCENES=0

for split in val test; do
  split_root="${ROOT}/${split}"
  [[ -d "${split_root}" ]] || continue
  while IFS= read -r -d '' scene_root; do
    repair_scene "${scene_root}"
    SCENES=$((SCENES + 1))
  done < <(find "${split_root}" -mindepth 1 -maxdepth 1 -type d -print0)
done

if (( SCENES == 0 )); then
  echo "[error] no scene directories found below ${ROOT}/{val,test}" >&2
  exit 1
fi

# Repair additional ZIP files already present in nonstandard subdirectories.
# Their URL is inferred from the archive filename and containing scene.
while IFS= read -r -d '' archive; do
  if valid_zip "${archive}"; then
    continue
  fi
  filename="$(basename "${archive}")"
  sequence="${filename}"
  for suffix in _calibration.zip _events_left.zip _images_rectified_left.zip _disparity_image.zip _disparity_event.zip; do
    sequence="${sequence%${suffix}}"
  done
  if [[ "${sequence}" == "${filename}" || -z "${sequence}" ]]; then
    echo "[unmapped corrupt zip] ${archive}" >&2
    FAILURES=$((FAILURES + 1))
    continue
  fi
  if ! download_atomic "${BASE_URL}/${sequence}/${filename}" "${archive}"; then
    FAILURES=$((FAILURES + 1))
  fi
done < <(find "${ROOT}" -type f -iname '*.zip' -print0)

echo
echo "[summary] scenes=${SCENES} failures=${FAILURES}"
if (( FAILURES > 0 )); then
  echo "Some files could not be repaired. Search the log for [download failed] or [invalid response]." >&2
  exit 2
fi

if [[ "${EXTRACT}" == "1" ]]; then
  bash "$(dirname "${BASH_SOURCE[0]}")/extract_archives_recursive.sh" "${ROOT}"
fi

echo "DSEC download audit and repair completed successfully."
