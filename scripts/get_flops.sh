#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

DATASET="${1:-cityscapes}"

case "${DATASET}" in
  cityscapes)
    CONFIG="segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_512x1024_120k_cityscapes.py"
    H=1024
    W=2048
    ;;
  ade20k)
    CONFIG="segmentation/configs/structmamba_seg/ade20k/upernet_structmamba_seg_512x512_160k_ade20k.py"
    H=512
    W=512
    ;;
  pascal_context)
    CONFIG="segmentation/configs/structmamba_seg/pascal_context/upernet_structmamba_seg_512x512_80k_pascal_context.py"
    H=512
    W=512
    ;;
  *)
    echo "Usage: bash scripts/get_flops.sh [cityscapes|ade20k|pascal_context]"
    exit 1
    ;;
esac

python segmentation/tools/analysis_tools/get_flops.py "${CONFIG}" --shape "${H}" "${W}"