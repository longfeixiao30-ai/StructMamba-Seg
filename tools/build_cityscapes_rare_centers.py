import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# 你自己的 Cityscapes 路径
# =========================
GT_DIR = "/root/autodl-tmp/datasets/cityscapes/gtFine/train"

# 输出索引文件
OUT_JSON = "/root/autodl-tmp/datasets/cityscapes/rare_centers_train.json"

# 重点尾类（trainId）
RARE_CLASSES = [3, 4, 5, 6, 12, 14, 15, 16, 17]

# 最小连通域面积，小于这个就不当作有效目标
MIN_AREA = 20


def collect_files(gt_dir):
    files = []
    for city in os.listdir(gt_dir):
        city_dir = os.path.join(gt_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for name in os.listdir(city_dir):
            if name.endswith("_gtFine_labelTrainIds.png"):
                files.append(os.path.join(city_dir, name))
    return sorted(files)


def extract_centers(mask, cls_id, min_area=20):
    binary = (mask == cls_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    centers = []

    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        centers.append({
            "cls_id": int(cls_id),
            "cx": float(cx),
            "cy": float(cy),
            "area": int(area)
        })
    return centers


def main():
    files = collect_files(GT_DIR)
    result = {}
    class_count = {c: 0 for c in RARE_CLASSES}

    for path in tqdm(files, desc="Scanning Cityscapes rare centers"):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        basename = os.path.basename(path)
        items = []

        for cls_id in RARE_CLASSES:
            centers = extract_centers(mask, cls_id, MIN_AREA)
            if len(centers) > 0:
                items.extend(centers)
                class_count[cls_id] += len(centers)

        result[basename] = items

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved rare-center index to: {OUT_JSON}")
    print("Connected components per rare class:")
    for k, v in class_count.items():
        print(f"  class {k}: {v}")


if __name__ == "__main__":
    main()