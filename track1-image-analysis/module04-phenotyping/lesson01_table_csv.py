# Module 4, Lesson 1: put per-plant measurements in a table and save it as CSV
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np
import pandas as pd

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

# --- Same steps as Module 3, Lesson 7: mask -> nearest pot -> pixels per plant ---
b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
gb = np.clip(g - b, 0, 255).astype(np.uint8)
t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)
roi = np.zeros(mask.shape, bool)
roi[800:2600, 150:3900] = mask[800:2600, 150:3900] > 0

pots = {
    "OP1": (1060, 1290), "OP2": (1490, 1320),
    "OP3": (800, 1516),  "OP4": (1245, 1520), "OP5": (1690, 1530),
    "OP6": (1000, 1690), "OP7": (1460, 1710),
    "OP8": (755, 1945),                       "OP9": (1650, 1960),
    "LP1": (2420, 1290), "LP2": (2860, 1340), "LP3": (3360, 1330),
                         "LP4": (3125, 1560),
    "LP5": (2435, 1755), "LP6": (2895, 1757), "LP7": (3350, 1770),
    "LP8": (2660, 2025), "LP9": (3115, 2035),
}
names = list(pots)
px = np.array([pots[n][0] for n in names])
py = np.array([pots[n][1] for n in names])
rows, cols = np.nonzero(roi)
dist2 = (rows[:, None] - py[None, :]) ** 2 + (cols[:, None] - px[None, :]) ** 2
nearest = dist2.argmin(axis=1)
areas = np.bincount(nearest, minlength=len(names))

# --- New: a pandas DataFrame = a table with named columns ---
df = pd.DataFrame({
    "plant":   names,
    "tray":    [n[:2] for n in names],   # first 2 letters: "OP" or "LP"
    "area_px": areas,
})
print(df)

print("\nAverage plant area per tray:")
print(df.groupby("tray")["area_px"].mean().round())

df.to_csv("outputs/m04_l01_plant_areas.csv", index=False)
print("\nsaved outputs/m04_l01_plant_areas.csv")
