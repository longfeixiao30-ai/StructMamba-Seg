# Checkpoints

This directory is used to store pretrained weights and released checkpoints for StructMamba-Seg.

## Notes
We do not directly include large checkpoint files in this repository. Users should download the required weights separately and place them in this folder before training or evaluation.

## Recommended files
Please place the checkpoints in this directory with the following names:

- `structmamba_seg_cityscapes_120k.pth`
- `structmamba_seg_ade20k_160k.pth`
- `structmamba_seg_pascal_context_80k.pth`
- `defm_small.pth`  (backbone initialization, if required)

## Suggested directory layout
```text
checkpoints/
├── structmamba_seg_cityscapes_120k.pth
├── structmamba_seg_ade20k_160k.pth
├── structmamba_seg_pascal_context_80k.pth
└── defm_small.pth