# Model Zoo

This file lists the released checkpoints for **StructMamba-Seg**.

---

## 1. Available Models

| Model | Dataset | Config | Checkpoint | Main Metric | Download |
|---|---|---|---|---|---|
| StructMamba-Seg | Cityscapes | `cityscapes/upernet_structmamba_seg_512x1024_120k_cityscapes.py` | `structmamba_seg_cityscapes_120k.pth` | mIoU | TBA |
| StructMamba-Seg | ADE20K | `ade20k/upernet_structmamba_seg_512x512_160k_ade20k.py` | `structmamba_seg_ade20k_160k.pth` | mIoU (SS/MS) | TBA |
| StructMamba-Seg | Pascal Context | `pascal_context/upernet_structmamba_seg_512x512_80k_pascal_context.py` | `structmamba_seg_pascal_context_80k.pth` | mIoU | TBA |

---

## 2. Recommended Release Assets

The main public release includes:
- `structmamba_seg_cityscapes_120k.pth`
- `structmamba_seg_ade20k_160k.pth`
- `structmamba_seg_pascal_context_80k.pth`
- `defm_small.pth` (backbone initialization)

---

## 3. Download Policy

Large checkpoint files are **not** stored in Git. Released models should be placed under:
`checkpoints/`

---

## 4. How to Evaluate Released Models

```bash
python segmentation/tools/test.py \
    segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_512x1024_120k_cityscapes.py \
    checkpoints/structmamba_seg_cityscapes_120k.pth