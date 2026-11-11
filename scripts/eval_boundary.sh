#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_512x1024_120k_cityscapes.py}"
PRED="${PRED:-}"
GT="${GT:-data/cityscapes/gtFine/val}"

if [ -z "${PRED}" ]; then
  echo "Please set PRED to your prediction file or directory."
  echo "Example:"
  echo "  PRED=work_dirs/structmamba_seg/cityscapes/preds.pkl bash scripts/eval_boundary.sh"
  exit 1
fi

python segmentation/evaluation/boundary_eval.py \
  --config "${CONFIG}" \
  --pred "${PRED}" \
  --gt "${GT}" \
  "$@"