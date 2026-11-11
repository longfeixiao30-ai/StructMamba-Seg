#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

GPUS="${GPUS:-2}"
CONFIG="${CONFIG:-segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_tta_2048x1024_cityscapes.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/structmamba_seg_cityscapes_120k.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/structmamba_seg/cityscapes/test_tta}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

bash segmentation/tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" --work-dir "${WORK_DIR}" "$@"