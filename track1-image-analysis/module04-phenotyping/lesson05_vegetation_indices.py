# Module 4, Lesson 5: RGB vegetation indices (VIs) per plant
# Run from the project root (plant-cv-tutorial), after Lesson 3 has made its CSV.
import cv2
import numpy as np
import pandas as pd

from plants import label_plants

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

names, rows, cols, nearest = label_plants(img)

# B, G, R of every plant pixel, as decimals (float) so division works
B = img[rows, cols, 0].astype(float)
G = img[rows, cols, 1].astype(float)
R = img[rows, cols, 2].astype(float)
eps = 1e-6                                   # tiny number: avoids dividing by 0

# Chromatic coordinates: each channel's share of the total -> less sensitive to brightness
total = R + G + B + eps
r, g, b = R / total, G / total, B / total

vis = {
    "ExG":   2 * g - r - b,                          # Excess Green
    "NGRDI": (G - R) / (G + R + eps),                # Normalized Green-Red Difference
    "GLI":   (2 * G - R - B) / (2 * G + R + B + eps),  # Green Leaf Index
}

count = np.bincount(nearest, minlength=len(names))
df = pd.read_csv("outputs/m04_l03_plant_traits.csv")      # area + hue from Lesson 3
for name, values in vis.items():
    df[name] = (np.bincount(nearest, weights=values, minlength=len(names)) / count).round(3)

print(df[["plant", "tray", "area_cm2", "mean_hue", "ExG", "NGRDI", "GLI"]])
print("\nAverage per tray:")
print(df.groupby("tray")[["mean_hue", "ExG", "NGRDI", "GLI"]].mean().round(3))

df.to_csv("outputs/m04_l05_plant_traits.csv", index=False)
print("\nsaved outputs/m04_l05_plant_traits.csv")
