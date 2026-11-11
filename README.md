<div align="center">
<h1>StructMamba-Seg</h1>
<h3>StructMamba-Seg: Boundary-Context Guided Deformable Scanning for Semantic Segmentation</h3>
Paper: <a href="<ARXIV_LINK>">ArXiv</a> &nbsp;|&nbsp; Code: <a href="<GITHUB_REPO_LINK>">GitHub</a>
</div>

## Abstract

Scan-based visual state space models, especially Mamba-based methods, have shown strong potential for semantic segmentation by modeling long-range dependencies with high computational efficiency. These methods typically transform 2D feature maps into 1D scan sequences, where the sampling positions and scan orders directly affect the preservation of boundary and local structural information during recursive propagation. However, existing scan and offset modeling strategies are mostly designed for general visual representation learning, making them less effective at performing structure-sensitive decisions in boundary neighborhoods, locally deformed regions, and multi-scale semantic ambiguity areas. To address this issue, we propose **StructMamba-Seg** (**Structure-Sensitive Mamba for Semantic Segmentation**). Specifically, we introduce a **Boundary-Context Guided Deformable Scan (BCG-DefScan)** into the Mamba scanning backbone, and construct an offset prediction network jointly driven by boundary cues and multi-scale contextual information. This design enables structure-sensitive scan decisions for segmentation scenarios, thereby adaptively regulating feature sampling positions, local organization, and offset effect strength. During training, we further introduce **IoU-Boundary Joint Optimization** to complement the main scanning mechanism with additional supervision on region quality and boundary details. Experiments show that StructMamba-Seg achieves consistent improvements on benchmarks such as **Cityscapes** and **ADE20K**, while exhibiting stronger structure preservation and category separation ability in small objects, thin structures, and semantically ambiguous regions.

## Overview

- Overall Framework

<p align="center">
  <img src="./assets/framework.svg" width="900" />
</p>

- BCG State Space Model

<p align="center">
  <img src="./assets/bcg_ssm.svg" width="900" />
</p>

- Qualitative Comparison

<p align="center">
  <img src="./assets/qualitative_cityscapes.svg" width="900" />
</p>

## Main Results

### Semantic Segmentation on ADE20K

StructMamba-Seg achieves strong performance on the ADE20K validation set under both single-scale (SS) and multi-scale (MS) evaluation.

| Model | Input | Segmentor | Schedule | mIoU (SS) | mIoU (MS) |
|:------|:-----:|:---------:|:--------:|:---------:|:---------:|
| **StructMamba-Seg** | 512×512 | UPerNet | 160k | **50.2** | **51.1** |

### Semantic Segmentation on Cityscapes

On Cityscapes, StructMamba-Seg achieves **82.12% mIoU**, demonstrating strong segmentation performance in complex urban scenes with thin structures, small objects, and ambiguous boundaries.

| Model | Train Crop | Evaluation | Segmentor | Schedule | mIoU |
|:------|:----------:|:----------:|:---------:|:--------:|:----:|
| **StructMamba-Seg** | 512×1024 | Single-Scale | UPerNet | 120k | **82.12** |

### Ablation Study

All ablation experiments are conducted on Cityscapes.

| BCG | IBJO | mIoU |
|:---:|:----:|:----:|
| ✗ | ✗ | 79.88 |
| ✗ | ✓ | 80.96 |
| ✓ | ✗ | 81.38 |
| ✓ | ✓ | **82.12** |

Individually, BCG and IBJO improve the baseline by **1.50** and **1.08** mIoU, respectively. Combining BCG and IBJO achieves **82.12% mIoU**, yielding an overall improvement of **2.24 mIoU** over the baseline.

### Boundary and Structure-Sensitive Evaluation

To further evaluate performance on boundaries, fine structures, and challenging categories, we report four structure-sensitive metrics on Cityscapes: Boundary IoU, Boundary F1, Hard-class mIoU, and Small-class mIoU.

| IBJO | BCG | Boundary IoU | Boundary F1 | Hard-class mIoU | Small-class mIoU |
|:----:|:---:|:------------:|:-----------:|:---------------:|:----------------:|
| ✗ | ✗ | 60.32 | 74.38 | 72.12 | 73.86 |
| ✓ | ✗ | 61.68 | 75.47 | 73.76 | 75.31 |
| ✗ | ✓ | 62.25 | 75.97 | 73.88 | 75.36 |
| ✓ | ✓ | **62.67** | **76.27** | **75.31** | **76.67** |

Compared with the baseline, the full StructMamba-Seg model improves Boundary IoU by **2.35**, Boundary F1 by **1.89**, Hard-class mIoU by **3.19**, and Small-class mIoU by **2.81**.

Boundary metrics are evaluated at a resolution of **2048×1024** with a boundary width ratio of **0.005**.

## Supported Scope

The main experiments reported in the StructMamba-Seg paper are conducted on:

- **Cityscapes**
- **ADE20K**

At the **repository level**, the public codebase currently includes dataset configs, base dataset definitions, or tool support for the following segmentation datasets:

- **ADE20K**
- **BDD100K**
- **CHASE DB1**
- **Cityscapes**
- **COCO-Stuff10K**
- **COCO-Stuff164K**
- **DRIVE**
- **HRF**
- **iSAID**
- **LEVIR-CD**
- **LoveDA**
- **Mapillary Vistas v1.0**
- **Mapillary Vistas v1.65**
- **Mapillary Vistas v2.0**
- **NYU**
- **Pascal Context**
- **Pascal VOC 2012**
- **Potsdam**
- **REFUGE**
- **STARE**
- **Synapse**
- **Vaihingen**

In addition, the repository also includes multiple dataset-specific variants for some benchmarks, such as different input-resolution settings for **Cityscapes**, **ADE20K**, **NYU**, and **LEVIR-CD**.

Please note that **repository-level support does not mean that all supported datasets are included in the paper’s main experimental tables**. The official results reported in the paper are restricted to **Cityscapes and ADE20K**, while the remaining datasets are included as extended configuration and tooling support in the public release.

## Getting Started

### Installation

**Step 1: Clone the StructMamba-Seg repository**

```bash
git clone <GITHUB_REPO_LINK>
cd StructMamba-Seg
```

**Step 2: Environment Setup**

***Create and activate a new conda environment***

```bash
conda env create -f environment.yaml
conda activate structmamba_seg
```

Or manually:

```bash
conda create -n structmamba_seg python=3.10
conda activate structmamba_seg
```

***Install PyTorch***

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
cd ../..
```

***Dependencies for Semantic Segmentation***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmsegmentation==1.2.2
```

## Quick Start

### Training

To train StructMamba-Seg on the supported benchmarks:

**Cityscapes**
```bash
bash scripts/train_cityscapes.sh
```

**ADE20K**
```bash
bash scripts/train_ade20k.sh
```

**Pascal Context**
```bash
bash scripts/train_pascal_context.sh
```

### Evaluation

**Cityscapes (single-scale)**
```bash
bash scripts/test_cityscapes.sh
```

**Cityscapes (TTA / full-resolution evaluation)**
```bash
bash scripts/test_cityscapes_tta.sh
```

**Boundary Evaluation**
```bash
bash scripts/eval_boundary.sh
```

**Hard-class / Small-class Evaluation**
```bash
bash scripts/eval_hard_small.sh
```

### FLOPs and Params

**Cityscapes**
```bash
bash scripts/get_flops.sh cityscapes
```

**ADE20K**
```bash
bash scripts/get_flops.sh ade20k
```

**Pascal Context**
```bash
bash scripts/get_flops.sh pascal_context
```

## Dataset Preparation

Please prepare the datasets before training or testing.

Detailed instructions are provided in:

- `docs/DATA_PREPARATION.md`

The officially documented paper benchmarks are:

- Cityscapes
- ADE20K

Additional repository-level configuration and training support is provided for Pascal Context and several other segmentation datasets.

## Project Structure

```text
StructMamba-Seg/
├── assets/
├── checkpoints/
├── core/
├── docs/
├── kernels/
├── scripts/
├── segmentation/
├── tools/
├── .gitignore
├── environment.yaml
├── LICENSE
├── NOTICE
├── README.md
└── requirements.txt
```

## Checkpoints

Released checkpoints should be placed under:

```text
checkpoints/
```

Recommended filenames:

- `structmamba_seg_cityscapes_120k.pth`
- `structmamba_seg_ade20k_160k.pth`
- `structmamba_seg_pascal_context_80k.pth`
- `defm_small.pth`

Please see `checkpoints/README.md` and `docs/MODEL_ZOO.md` for more details.

## Citation

If StructMamba-Seg is helpful for your research, please cite the following paper:

```bibtex
@article{<CITATION_KEY>,
  title   = {StructMamba-Seg: Boundary-Context Guided Deformable Scanning for Semantic Segmentation},
  author  = {<AUTHORS>},
  journal = {arXiv preprint arXiv:<ARXIV_ID>},
  year    = {<YEAR>}
}
```

## Acknowledgment

This project is built upon the broader open-source ecosystem of visual state space models and semantic segmentation toolchains. In particular, our implementation is developed on top of the **DefMamba / DeformableMamba** codebase and further extended for semantic segmentation. We sincerely acknowledge the following excellent works and repositories:

- DefMamba / DeformableMamba
- VMamba
- Mamba
- OpenMMLab / MMSegmentation
- Swin Transformer
- ConvNeXt
- related visual Mamba / SSM baselines

We thank the authors and contributors of these projects for making their code and research publicly available.