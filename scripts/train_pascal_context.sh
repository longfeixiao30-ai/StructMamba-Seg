#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

GPUS="${GPUS:-2}"
CONFIG="${CONFIG:-segmentation/configs/structmamba_seg/pascal_context/upernet_structmamba_seg_512x512_80k_pascal_context.py}"
WORK_DIR="${WORK_DIR:-work_dirs/structmamba_seg/pascal_context/upernet_structmamba_seg_512x512_80k_pascal_context}"

bash segmentation/tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}" "$@"