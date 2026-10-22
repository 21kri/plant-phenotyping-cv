# Module 7, Lesson 1: an automatic pipeline - every photo in a folder -> one row of traits in a CSV
# Run from the project root (plant-cv-tutorial).
#   Change FOLDER to a new day's photos and run again: no other edits needed.
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FOLDER = Path("data/root-vigor-30-04")
SHEET = FOLDER / "photo_sheet.csv"          # optional: photo, date, box, view, px_per_cm
OUT = Path("outputs/m07_l01_traits.csv")


def measure(img):
    """All traits for ONE photo. Returns a dict (one row of the table)."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    gb = np.clip(g - b, 0, 255).astype(np.uint8)
    t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Module 3
    mask = (mask > 0) & (g > r)                                             # no yellow tape
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, brush)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)             # drop specks
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= 2000)[0]
    plant = np.isin(labels, big[big != 0])

    px = img[plant].astype(float)                                            # B, G, R of plant pixels
    total = px.sum(axis=1) + 1e-6
    exg = (2 * px[:, 1] - px[:, 2] - px[:, 0]) / total                      # Module 4, Lesson 5
    return {
        "otsu_threshold": t,
        "plant_px": int(plant.sum()),
        "plant_%_of_photo": round(100 * plant.mean(), 2),
        "ExG": round(float(exg.mean()), 3) if len(exg) else np.nan,
    }


# Optional photo sheet: extra facts the pixels can't tell us (date, which box, scale)
sheet = pd.read_csv(SHEET) if SHEET.exists() else pd.DataFrame(columns=["photo"])

rows = []
start = time.time()
photos = sorted(FOLDER.glob("*.jpg"))
for i, path in enumerate(photos, start=1):
    img = cv2.imread(str(path))
    if img is None:                                     # a broken file must not stop the whole run
        print(f"  skipped {path.name}: could not read")
        continue
    row = {"photo": path.name, **measure(img)}
    rows.append(row)
    print(f"  {i:3d}/{len(photos)}  {path.name}  plant {row['plant_%_of_photo']:5.2f} %")

df = pd.DataFrame(rows).merge(sheet, on="photo", how="left")               # add date, box, scale
if "px_per_cm" in df:
    df["plant_cm2"] = (df["plant_px"] / df["px_per_cm"] ** 2).round(1)      # only where a scale is known

df.to_csv(OUT, index=False)
print(f"\n{len(df)} photos in {time.time() - start:.0f} s -> saved {OUT}")
print(df.dropna(subset=["plant_cm2"]).to_string(index=False) if "plant_cm2" in df else "")
