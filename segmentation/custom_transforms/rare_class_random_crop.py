import os
import json
import random
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RareClassRandomCrop(BaseTransform):
    """
    50% 正常随机裁剪
    50% 围绕尾类中心裁剪
    """

    def __init__(
        self,
        crop_size=(512, 1024),
        cat_max_ratio=0.75,
        ignore_index=255,
        rare_centers_file=None,
        rare_classes=None,
        rare_prob=0.5,
        center_jitter=0.25
    ):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.rare_prob = rare_prob
        self.center_jitter = center_jitter
        self.rare_classes = rare_classes if rare_classes is not None else []

        assert rare_centers_file is not None, "rare_centers_file must be provided"
        with open(rare_centers_file, "r", encoding="utf-8") as f:
            self.rare_centers = json.load(f)

    def _get_crop_bbox_random(self, img):
        h, w = img.shape[:2]
        ch, cw = self.crop_size

        if h <= ch:
            y1 = 0
        else:
            y1 = random.randint(0, h - ch)

        if w <= cw:
            x1 = 0
        else:
            x1 = random.randint(0, w - cw)

        return y1, y1 + ch, x1, x1 + cw

    def _get_crop_bbox_center(self, img, cx, cy):
        h, w = img.shape[:2]
        ch, cw = self.crop_size

        max_jx = int(cw * self.center_jitter)
        max_jy = int(ch * self.center_jitter)

        cx = int(cx + random.randint(-max_jx, max_jx))
        cy = int(cy + random.randint(-max_jy, max_jy))

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)

        if x1 + cw > w:
            x1 = max(0, w - cw)
        if y1 + ch > h:
            y1 = max(0, h - ch)

        return y1, y1 + ch, x1, x1 + cw

    def _crop(self, data, bbox):
        y1, y2, x1, x2 = bbox
        return data[y1:y2, x1:x2, ...]

    def _is_valid_crop(self, seg):
        labels, counts = np.unique(seg, return_counts=True)

        valid_mask = labels != self.ignore_index
        labels = labels[valid_mask]
        counts = counts[valid_mask]

        if len(counts) == 0:
            return False

        if np.max(counts) / np.sum(counts) >= self.cat_max_ratio:
            return False

        return True

    def transform(self, results):
        img = results["img"]
        seg = results["gt_seg_map"]

        basename = os.path.basename(results["seg_map_path"])
        use_rare = False

        if basename in self.rare_centers and len(self.rare_centers[basename]) > 0:
            if random.random() < self.rare_prob:
                use_rare = True

        # 优先尾类中心裁剪
        if use_rare:
            candidates = [
                x for x in self.rare_centers[basename]
                if x["cls_id"] in self.rare_classes
            ]
            if len(candidates) > 0:
                random.shuffle(candidates)
                for cand in candidates:
                    bbox = self._get_crop_bbox_center(img, cand["cx"], cand["cy"])
                    crop_seg = self._crop(seg, bbox)
                    if self._is_valid_crop(crop_seg):
                        results["img"] = self._crop(img, bbox)
                        results["gt_seg_map"] = crop_seg
                        results["img_shape"] = results["img"].shape[:2]
                        return results

        # 回退到普通随机裁剪
        for _ in range(10):
            bbox = self._get_crop_bbox_random(img)
            crop_seg = self._crop(seg, bbox)
            if self._is_valid_crop(crop_seg):
                results["img"] = self._crop(img, bbox)
                results["gt_seg_map"] = crop_seg
                results["img_shape"] = results["img"].shape[:2]
                return results

        # 最后兜底
        bbox = self._get_crop_bbox_random(img)
        results["img"] = self._crop(img, bbox)
        results["gt_seg_map"] = self._crop(seg, bbox)
        results["img_shape"] = results["img"].shape[:2]
        return results