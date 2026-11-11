---

### 2. 文件路径：`docs/RESULTS.md`

```markdown
# Results

This document explains how the reported results in the paper correspond to configs, checkpoints, and evaluation scripts in this repository.

---

## 1. Main Results in the Paper

The paper reports results on the following datasets:
- **ADE20K**
- **Cityscapes**
- **Pascal Context**

The main result sections include comparison with representative methods, ablation studies, boundary and structure-sensitive evaluation, and qualitative comparison.

---

## 2. Paper-to-Code Mapping

### 2.1 ADE20K Main Results
- **Config**: `segmentation/configs/structmamba_seg/ade20k/upernet_structmamba_seg_512x512_160k_ade20k.py`
- **Multi-scale testing config**: `segmentation/configs/structmamba_seg/ade20k/upernet_structmamba_seg_ms_test_512x512_ade20k.py`
- **Typical checkpoint**: `checkpoints/structmamba_seg_ade20k_160k.pth`

### 2.2 Cityscapes Main Results
- **Config**: `segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_512x1024_120k_cityscapes.py`
- **TTA / full-resolution testing config**: `segmentation/configs/structmamba_seg/cityscapes/upernet_structmamba_seg_tta_2048x1024_cityscapes.py`
- **Typical checkpoint**: `checkpoints/structmamba_seg_cityscapes_120k.pth`

### 2.3 Pascal Context Main Results
- **Config**: `segmentation/configs/structmamba_seg/pascal_context/upernet_structmamba_seg_512x512_80k_pascal_context.py`
- **Multi-scale testing config**: `segmentation/configs/structmamba_seg/pascal_context/upernet_structmamba_seg_ms_test_512x512_pascal_context.py`
- **Typical checkpoint**: `checkpoints/structmamba_seg_pascal_context_80k.pth`

---

## 3. Mapping to Paper Tables

### Table 1: Comparison on ADE20K
This table corresponds to the ADE20K configs (SS and MS). Reported metrics: mIoU (SS), mIoU (MS), Params, FLOPs.

### Table 2: Comparison on Cityscapes
This table corresponds to the Cityscapes configs. Reported metrics: mIoU, Params, FLOPs.

### Table 3–5: Ablation Studies
These tables correspond to internal variants of the main model. Themes include overall effectiveness, module-level contribution, and loss-level contribution.

### Table 6: Boundary and Structure-Sensitive Evaluation
This table corresponds to the Cityscapes evaluation scripts for Boundary IoU, Boundary F1, Hard-class mIoU, and Small-class mIoU.

---

## 4. Metric Definitions

- **mIoU (SS)**: Standard single-scale mean Intersection over Union.
- **mIoU (MS)**: Multi-scale or test-time augmented mean Intersection over Union.
- **Boundary IoU**: A boundary-focused overlap metric.
- **Boundary F1**: A boundary-sensitive F1 score assessing contour alignment.
- **Hard-class mIoU**: Average mIoU over predefined hard categories.
- **Small-class mIoU**: Average mIoU over predefined small-object categories.

---

## 5. External Baselines

External comparison methods in the paper are taken from original published results. StructMamba-Seg results are produced or reproduced within this repository.

---

## 6. Result Reproduction Workflow

1. Prepare the dataset.
2. Select the matching config.
3. Train the model or download the released checkpoint.
4. Run single-scale or multi-scale evaluation.
5. Compute auxiliary metrics if needed.
6. Compare the output against the paper table.