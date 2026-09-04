#!/bin/bash
# Merge each SUN prompt-scale combination's 15 batch output folders into one
# resumable destination while preserving every original batch folder.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SOURCE_ROOT=${1:-"${REPO_ROOT}/CMIG_npz_data/sunseg"}
VERIFY_ONLY=${VERIFY_ONLY:-0}

SCALES=(10 12 14 18)
BATCH_COUNT=15
EXPECTED_LOGS=375
EXPECTED_INFO_JSON=1500
EXPECTED_ROUND_DIRS=1500
EXPECTED_MASK_PNG=150000

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required but is not available." >&2
    exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo "ERROR: SUN output root does not exist: ${SOURCE_ROOT}" >&2
    exit 1
fi

count_paths() {
    local root=$1
    local kind=$2
    local pattern=$3
    find "${root}" -type "${kind}" -name "${pattern}" -print0 | awk -v RS='\0' 'END {print NR}'
}

verify_destination() {
    local destination=$1
    local logs info round_dirs mask_png partials

    logs=$(count_paths "${destination}/logs" f '*.json')
    info=$(count_paths "${destination}/info_dict" f '*.json')
    round_dirs=$(count_paths "${destination}/sam2_masks" d '*_round_*')
    mask_png=$(count_paths "${destination}/sam2_masks" f '*.png')
    partials=$(find "${destination}" -type d -name '.rsync-partial' -print0 | awk -v RS='\0' 'END {print NR}')

    printf 'VERIFY %s: logs=%s info=%s round_dirs=%s masks=%s partial_dirs=%s\n' \
        "$(basename "${destination}")" "${logs}" "${info}" "${round_dirs}" "${mask_png}" "${partials}"

    [[ "${logs}" -eq "${EXPECTED_LOGS}" ]] || return 1
    [[ "${info}" -eq "${EXPECTED_INFO_JSON}" ]] || return 1
    [[ "${round_dirs}" -eq "${EXPECTED_ROUND_DIRS}" ]] || return 1
    [[ "${mask_png}" -eq "${EXPECTED_MASK_PNG}" ]] || return 1
    [[ "${partials}" -eq 0 ]] || return 1
}

check_unique_clip_names() {
    local initial=$1
    local correction=$2
    local section=$3
    local duplicate

    if [[ "${section}" == sam2_masks ]]; then
        duplicate=$(
            find "${SOURCE_ROOT}" -mindepth 3 -maxdepth 3 -type d \
                -path "${SOURCE_ROOT}/sunseg_${initial}_${correction}_batch_*/${section}/*" \
                -printf '%f\n' | sort | uniq -d
        )
    else
        duplicate=$(
            find "${SOURCE_ROOT}" -mindepth 3 -maxdepth 3 -type f \
                -path "${SOURCE_ROOT}/sunseg_${initial}_${correction}_batch_*/${section}/*" \
                -printf '%f\n' | sort | uniq -d
        )
    fi
    if [[ -n "${duplicate}" ]]; then
        echo "ERROR: duplicate ${section} entry across batches for sunseg_${initial}_${correction}: ${duplicate}" >&2
        exit 1
    fi
}

for correction in "${SCALES[@]}"; do
    for initial in "${SCALES[@]}"; do
        combination="sunseg_${initial}_${correction}"
        destination="${SOURCE_ROOT}/${combination}"
        state_dir="${destination}/.merge_state"

        echo
        echo "=== ${combination} ==="

        for ((batch=0; batch<BATCH_COUNT; batch++)); do
            source_batch="${SOURCE_ROOT}/${combination}_batch_${batch}"
            for section in info_dict sam2_masks logs; do
                if [[ ! -d "${source_batch}/${section}" ]]; then
                    echo "ERROR: missing source directory: ${source_batch}/${section}" >&2
                    exit 1
                fi
            done
        done

        if [[ "${VERIFY_ONLY}" != 1 ]]; then
            check_unique_clip_names "${initial}" "${correction}" logs
            check_unique_clip_names "${initial}" "${correction}" info_dict
            check_unique_clip_names "${initial}" "${correction}" sam2_masks

            mkdir -p "${destination}/info_dict" "${destination}/sam2_masks" \
                "${destination}/logs" "${state_dir}"

            for ((batch=0; batch<BATCH_COUNT; batch++)); do
                marker="${state_dir}/batch_${batch}.complete"
                if [[ -f "${marker}" ]]; then
                    echo "Skipping previously completed ${combination} batch ${batch}"
                    continue
                fi

                source_batch="${SOURCE_ROOT}/${combination}_batch_${batch}"
                echo "Copying ${combination} batch ${batch}/${BATCH_COUNT}"
                for section in info_dict sam2_masks logs; do
                    rsync -a --partial --partial-dir=.rsync-partial --info=stats1 \
                        "${source_batch}/${section}/" "${destination}/${section}/"
                done
                touch "${marker}"
            done
        fi

        if verify_destination "${destination}"; then
            if [[ "${VERIFY_ONLY}" != 1 ]]; then
                touch "${state_dir}/merge.complete"
            fi
            echo "COMPLETE: ${combination}"
        else
            echo "ERROR: verification failed for ${combination}; rerun this script to resume." >&2
            exit 1
        fi
    done
done

echo
echo "All 16 SUN prompt-scale combinations merged and verified successfully."
