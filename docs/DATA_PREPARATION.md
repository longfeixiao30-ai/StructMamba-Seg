# Data Preparation

This repository follows an MMSegmentation-style project layout for semantic segmentation and is designed to support a broad set of dataset preparation workflows. In practice, the codebase organization, conversion utilities, and dataset-facing interfaces are compatible with **15+ segmentation datasets**, while the **paper experiments reported in StructMamba-Seg are conducted on three official benchmarks only**:

- Cityscapes
- ADE20K
- Pascal Context

To avoid ambiguity, please distinguish between:

- **repository-level dataset support**: broader tooling and extensibility,
- **paper-level experimental coverage**: Cityscapes, ADE20K, and Pascal Context only.

---

## 1. Supported Datasets

### 1.1 Paper Benchmarks

The official experiments in the paper are conducted on:

- **Cityscapes**
- **ADE20K**
- **Pascal Context**

### 1.2 Additional Dataset Compatibility

The repository is organized in a way that is readily extensible to additional segmentation datasets commonly used in MMSegmentation-style workflows, including but not limited to:

- COCO-Stuff10K
- COCO-Stuff164K
- LoveDA
- Potsdam
- Vaihingen
- iSAID
- REFUGE
- DRIVE
- HRF
- STARE
- Synapse
- NYU
- VOC Aug
- LEVIR-CD
- CHASE DB1

These datasets are **not part of the paper’s main comparison tables**, but the repository structure is intentionally aligned with the broader OpenMMLab-style dataset preparation conventions.

---

## 2. Recommended Dataset Root Layout

We recommend placing datasets under a unified root such as:

```text
data/
├── cityscapes/
├── ADEChallengeData2016/
└── VOCdevkit/


data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/

data/ADEChallengeData2016/







