---

### 4. 文件路径：`docs/FAQ.md`

```markdown
# FAQ

This document collects common issues when using **StructMamba-Seg**.

---

## 1. Installation

### Q1. Why does `MM_VSSM` fail to register?
Check if `PYTHONPATH` is set to the repository root and if `custom_imports` in the config is correct.

### Q2. Why does `get_flops.py` report `ModuleNotFoundError`?
Ensure the repository root is added to your `PYTHONPATH`.

### Q3. Why does selective scan compilation fail?
Check for CUDA/PyTorch version mismatches or missing build tools.

---

## 2. Dataset

### Q4. Why does Cityscapes path not work?
Verify that `leftImg8bit/` and `gtFine/` exist and the config root path matches your local directory.

### Q5. What is the expected dataset root layout?
```text
data/
├── cityscapes/
├── ADEChallengeData2016/
└── VOCdevkit/