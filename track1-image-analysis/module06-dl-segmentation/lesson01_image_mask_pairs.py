# Module 6, Lesson 1: a segmentation dataset = pairs of (photo, mask)
# Run from the project root (plant-cv-tutorial).
# The masks come from our colour rule (Modules 2-3), so they are "pseudo-labels": good, not perfect.
from pathlib import Path

import cv2
import numpy as np

src = Path("data/root-vigor-30-04")
img_dir = Path("data/seg_dataset/images")
mask_dir = Path("data/seg_dataset/masks")
img_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)


def rule_mask(img):
    """Plant = G - B > 40 and G > R, cleaned with opening/closing (Module 3)."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    mask = (((g - b) > 40) & (g > r)).astype(np.uint8) * 255
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)


previews = []
for path in sorted(src.glob("IMG_*.jpg")):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_AREA)   # 4032x3024 -> 512x384
    mask = rule_mask(img)

    cv2.imwrite(str(img_dir / f"{path.stem}.jpg"), img)
    cv2.imwrite(str(mask_dir / f"{path.stem}.png"), mask)   # PNG: no blurry JPG edges in a mask

    if len(previews) < 8:                                  # side-by-side check: photo | mask
        previews.append(np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

print("pairs:", len(list(img_dir.glob("*.jpg"))), "images,", len(list(mask_dir.glob("*.png"))), "masks")
print("mask values:", np.unique(cv2.imread(str(mask_dir / "IMG_0275.png"), 0)))
cv2.imwrite("outputs/m06_l01_pairs.jpg", np.vstack(previews[::2]))
print("saved outputs/m06_l01_pairs.jpg")
