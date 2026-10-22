"""Measure plant traits for every photo in a folder and write one CSV.

Usage (from the project root, in the plantvision environment):
    python pipeline/measure_folder.py data/2026-05-07
    python pipeline/measure_folder.py data/2026-05-07 --sheet data/2026-05-07/photo_sheet.csv --out results/2026-05-07.csv

The optional photo sheet adds facts the pixels can't tell (date, box, view, scale);
see pipeline/photo_sheet_template.csv. With a px_per_cm value, plant area is also given in cm2.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MIN_BLOB_PX = 2000   # smaller green blobs are specks (reflections), not leaves


def plant_mask(img):
    """True where a pixel is green plant tissue (Otsu on G - B, G > R, cleaned)."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    gb = np.clip(g - b, 0, 255).astype(np.uint8)
    threshold, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0) & (g > r)                      # G > R rejects yellowish tape labels
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, brush)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= MIN_BLOB_PX)[0]
    return np.isin(labels, big[big != 0]), threshold


def measure(img):
    """All traits for one photo, as one table row."""
    plant, threshold = plant_mask(img)
    px = img[plant].astype(float)
    total = px.sum(axis=1) + 1e-6
    exg = (2 * px[:, 1] - px[:, 2] - px[:, 0]) / total   # Excess Green on chromatic coordinates
    return {
        "otsu_threshold": threshold,
        "plant_px": int(plant.sum()),
        "plant_pct_of_photo": round(100 * plant.mean(), 2),
        "ExG": round(float(exg.mean()), 3) if len(exg) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with .jpg photos")
    parser.add_argument("--sheet", type=Path, help="photo sheet CSV (default: <folder>/photo_sheet.csv if present)")
    parser.add_argument("--out", type=Path, help="output CSV (default: <folder>/traits.csv)")
    args = parser.parse_args()

    sheet_path = args.sheet or args.folder / "photo_sheet.csv"
    out = args.out or args.folder / "traits.csv"
    sheet = pd.read_csv(sheet_path) if sheet_path.exists() else pd.DataFrame(columns=["photo"])

    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        raise SystemExit(f"No .jpg/.png photos in {args.folder}")

    rows, start = [], time.time()
    for i, path in enumerate(photos, start=1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skipped {path.name}: could not read")
            continue
        rows.append({"photo": path.name, **measure(img)})
        print(f"  {i:3d}/{len(photos)}  {path.name}")

    df = pd.DataFrame(rows).merge(sheet, on="photo", how="left")
    if "px_per_cm" in df:
        df["plant_cm2"] = (df["plant_px"] / df["px_per_cm"] ** 2).round(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} photos in {time.time() - start:.0f} s -> {out}")


if __name__ == "__main__":
    main()
