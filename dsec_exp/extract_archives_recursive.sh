#!/usr/bin/env bash
set -uo pipefail

# Recursively extract every supported archive into a sibling directory named
# after the archive. Archives are kept. Successful extractions are marked so
# rerunning this script is safe and cheap.

ROOT="${1:-/data1/lzh/dataset/DESC/DSEC_EV_VGGT}"
MARKER=".eventvggt_extract_complete"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
declare -A FAILED_THIS_RUN=()

if [[ ! -d "${ROOT}" ]]; then
  echo "[error] directory does not exist: ${ROOT}" >&2
  exit 1
fi

archive_stem() {
  local name="$1"
  local lower="${name,,}"
  case "${lower}" in
    *.tar.gz)  printf '%s' "${name:0:${#name}-7}" ;;
    *.tar.bz2) printf '%s' "${name:0:${#name}-8}" ;;
    *.tar.xz)  printf '%s' "${name:0:${#name}-7}" ;;
    *.tgz)     printf '%s' "${name:0:${#name}-4}" ;;
    *.tbz2)    printf '%s' "${name:0:${#name}-5}" ;;
    *.txz)     printf '%s' "${name:0:${#name}-4}" ;;
    *.zip|*.tar|*.7z|*.rar)
      printf '%s' "${name%.*}"
      ;;
    *) return 1 ;;
  esac
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] required command not found: $1" >&2
    return 1
  fi
}

extract_one() {
  local archive="$1"
  local parent name lower stem output
  parent="$(dirname "${archive}")"
  name="$(basename "${archive}")"
  lower="${name,,}"
  stem="$(archive_stem "${name}")" || return 2
  output="${parent}/${stem}"

  if [[ -f "${output}/${MARKER}" ]]; then
    return 3
  fi

  mkdir -p "${output}"
  echo "[extract] ${archive}"
  echo "       -> ${output}"

  case "${lower}" in
    *.zip)
      if ! require_command unzip; then
        return 1
      fi
      if ! unzip -tqq "${archive}" >/dev/null 2>&1; then
        echo "[invalid zip] ${archive}" >&2
        if command -v file >/dev/null 2>&1; then
          file "${archive}" >&2 || true
        fi
        echo "The download is incomplete, is an HTML response, or is one part of a split archive." >&2
        return 4
      fi
      unzip -oq "${archive}" -d "${output}"
      ;;
    *.tar)
      require_command tar && tar -xf "${archive}" -C "${output}"
      ;;
    *.tar.gz|*.tgz)
      require_command tar && tar -xzf "${archive}" -C "${output}"
      ;;
    *.tar.bz2|*.tbz2)
      require_command tar && tar -xjf "${archive}" -C "${output}"
      ;;
    *.tar.xz|*.txz)
      require_command tar && tar -xJf "${archive}" -C "${output}"
      ;;
    *.7z)
      require_command 7z && 7z x -y -o"${output}" "${archive}" >/dev/null
      ;;
    *.rar)
      if command -v unrar >/dev/null 2>&1; then
        unrar x -o+ "${archive}" "${output}/" >/dev/null
      else
        require_command 7z && 7z x -y -o"${output}" "${archive}" >/dev/null
      fi
      ;;
    *) return 2 ;;
  esac

  local status=$?
  if [[ ${status} -eq 0 ]]; then
    printf 'archive=%s\ncompleted_utc=%s\n' \
      "${archive}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output}/${MARKER}"
    echo "[done] ${archive}"
  else
    echo "[failed:${status}] ${archive}" >&2
  fi
  return "${status}"
}

echo "Recursive extraction root: ${ROOT}"
echo "Archives are retained; outputs are sibling subdirectories."

total_extracted=0
total_failed=0
round=0

while (( round < MAX_ROUNDS )); do
  round=$((round + 1))
  extracted_this_round=0
  found_unfinished=0

  while IFS= read -r -d '' archive; do
    if [[ -n "${FAILED_THIS_RUN[${archive}]+x}" ]]; then
      continue
    fi
    name="$(basename "${archive}")"
    stem="$(archive_stem "${name}")" || continue
    output="$(dirname "${archive}")/${stem}"
    if [[ -f "${output}/${MARKER}" ]]; then
      continue
    fi

    found_unfinished=$((found_unfinished + 1))
    if extract_one "${archive}"; then
      extracted_this_round=$((extracted_this_round + 1))
      total_extracted=$((total_extracted + 1))
    else
      status=$?
      if [[ ${status} -ne 3 ]]; then
        FAILED_THIS_RUN["${archive}"]=1
        total_failed=$((total_failed + 1))
      fi
    fi
  done < <(find "${ROOT}" -type f \( -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' -o -iname '*.tar.bz2' -o -iname '*.tbz2' -o -iname '*.tar.xz' -o -iname '*.txz' -o -iname '*.7z' -o -iname '*.rar' \) -print0)

  if (( found_unfinished == 0 )); then
    break
  fi
  if (( extracted_this_round == 0 )); then
    echo "[stop] unfinished archives remain, but none could be extracted in round ${round}." >&2
    break
  fi
done

echo "[summary] extracted=${total_extracted} failed=${total_failed} rounds=${round}"
if (( total_failed > 0 )); then
  exit 2
fi
