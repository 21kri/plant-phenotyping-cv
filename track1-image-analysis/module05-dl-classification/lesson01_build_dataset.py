# Module 5, Lesson 1: build a labelled image dataset - one folder per class
# Run from the project root (plant-cv-tutorial).
from pathlib import Path

import cv2
import pandas as pd

src = Path("data/root-vigor-30-04")
dst = Path("data/oplp_dataset")

labels = pd.read_csv(src / "labels.csv")      # columns: photo, part, label
print(labels.head())
print("\nimages per class:\n", labels["label"].value_counts())

for row in labels.itertuples():
    img = cv2.imread(str(src / row.photo))
    if img is None:
        raise SystemExit(f"Could not load {row.photo}")

    w = img.shape[1]
    if row.part == "left":        # two boxes side by side: cut the photo in half
        img = img[:, : w // 2]
    elif row.part == "right":
        img = img[:, w // 2 :]
    # part == "full": one box, keep the whole photo

    # Shrink so the longest side is 512 px - plenty for the model, much faster to load
    scale = 512 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    folder = dst / row.label                   # data/oplp_dataset/OP or .../LP
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{Path(row.photo).stem}_{row.part}.jpg"
    cv2.imwrite(str(folder / name), img)

for folder in sorted(dst.iterdir()):
    print(f"{folder}: {len(list(folder.glob('*.jpg')))} images")
