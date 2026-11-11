import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    gts = gt_sorted.sum()
    p = gt_sorted.numel()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probs(probs: torch.Tensor, labels: torch.Tensor, ignore_index: int):
    """
    probs:  [B, C, H, W]
    labels: [B, H, W] or [B, 1, H, W]
    """
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)

    B, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)

    valid = labels != ignore_index
    if valid.sum() == 0:
        return probs.new_zeros((0, C)), labels.new_zeros((0,), dtype=torch.long)

    return probs[valid], labels[valid]


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, classes='present'):
    """
    probs:  [P, C], class probabilities at valid pixels
    labels: [P], ground-truth labels at valid pixels
    """
    if probs.numel() == 0:
        return probs.sum() * 0.0

    C = probs.size(1)
    losses = []

    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue

        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if len(losses) == 0:
        return probs.sum() * 0.0

    return torch.stack(losses).mean()


def labels_to_onehot(labels: torch.Tensor, num_classes: int, ignore_index: int):
    """
    labels: [B, H, W] or [B, 1, H, W]
    return:
        onehot:    [B, C, H, W]
        valid_mask:[B, 1, H, W]
    """
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)

    valid = (labels != ignore_index)
    safe_labels = labels.clone()
    safe_labels[~valid] = 0

    onehot = F.one_hot(safe_labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    onehot = onehot * valid.unsqueeze(1).float()
    return onehot, valid.unsqueeze(1).float()


def soft_boundary_map(x: torch.Tensor, radius: int = 1):
    """
    x: [B, C, H, W], can be probabilities or one-hot maps
    returns a single-channel soft boundary map [B, 1, H, W]
    """
    k = 2 * radius + 1
    max_v = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    min_v = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=radius)
    bd = (max_v - min_v).amax(dim=1, keepdim=True)
    return bd.clamp(0.0, 1.0)


@MODELS.register_module()
class IoUBoundaryLoss(nn.Module):
    """
    CE 不在这里做，外面仍保留标准 CrossEntropyLoss。
    这个 loss 负责两部分：
    1) Lovasz-Softmax: 直接拉 IoU
    2) Boundary loss: 轻量边界监督（BCE + Dice）
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        loss_name='loss_iou_boundary',
        ignore_index=255,
        lovasz_weight=1.0,
        boundary_weight=0.25,
        boundary_radius=1,
        boundary_bce_weight=1.0,
        boundary_dice_weight=1.0,
    ):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self._loss_name = loss_name
        self.ignore_index = int(ignore_index)

        self.lovasz_weight = float(lovasz_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_radius = int(boundary_radius)
        self.boundary_bce_weight = float(boundary_bce_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=None,
        **kwargs,
    ):
        if label.dim() == 4 and label.size(1) == 1:
            label = label.squeeze(1)

        ignore_index = self.ignore_index if ignore_index is None else int(ignore_index)

        probs = F.softmax(cls_score, dim=1)

        # ---------- Lovasz ----------
        probs_flat, labels_flat = flatten_probs(probs, label, ignore_index)
        lovasz = lovasz_softmax_flat(probs_flat, labels_flat, classes='present')

        # ---------- Boundary ----------
        num_classes = cls_score.size(1)
        gt_onehot, valid_mask = labels_to_onehot(label, num_classes, ignore_index)
        pred_bd = soft_boundary_map(probs, radius=self.boundary_radius)
        gt_bd = soft_boundary_map(gt_onehot, radius=self.boundary_radius)

        # BCE on soft boundary map
# BCE on soft boundary map
        eps = 1e-6
        pred_bd_clamped = pred_bd.clamp(min=eps, max=1.0 - eps)

        # BCE 在 autocast 下不安全，这里单独关闭 autocast 并转 fp32
        with torch.cuda.amp.autocast(enabled=False):
            bce = F.binary_cross_entropy(
                pred_bd_clamped.float(),
                gt_bd.float(),
                reduction='none'
            )
            bce = (bce * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

        # Dice on boundary map
        pred_valid = pred_bd * valid_mask
        gt_valid = gt_bd * valid_mask
        inter = (pred_valid * gt_valid).sum(dim=(1, 2, 3))
        denom = pred_valid.sum(dim=(1, 2, 3)) + gt_valid.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        dice = dice.mean()

        boundary = self.boundary_bce_weight * bce + self.boundary_dice_weight * dice

        loss = self.lovasz_weight * lovasz + self.boundary_weight * boundary
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name