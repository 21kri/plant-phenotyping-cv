# Module 4, Lesson 8: how much of each plant is yellow / senescent? (a stress trait)
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np
import pandas as pd

from plants import POTS, PX_PER_CM

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))

# 1. ALL plant tissue: green leaves (G - B rule) OR yellow-brown tissue (hue 12-30, coloured, bright)
plant = (((g - b) > 40) & (g > r)) | ((H >= 12) & (H <= 30) & (S > 70) & (V > 150))
roi = np.zeros(H.shape, bool)
roi[800:2600, 150:3900] = True
plugs = np.zeros(H.shape, np.uint8)                   # brown growing-medium plugs sit in the pots
for x, y in list(POTS.values()) + [(1194, 1950), (2660, 1550)]:
    cv2.circle(plugs, (x, y), 60, 1, -1)
plant &= roi & (plugs == 0)
brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
plant = cv2.morphologyEx(plant.astype(np.uint8), cv2.MORPH_OPEN, brush) > 0

# 2. How sensitive is "yellow" to where we put the hue cutoff?
op_side = np.zeros(H.shape, bool)
op_side[:, :2130] = True
print("hue cutoff   OP yellow %   LP yellow %")
for cut in (30, 33, 35, 37):
    yel = plant & (H < cut) & (V > 150)
    print(f"  < {cut}      {100 * (yel & op_side).sum() / (plant & op_side).sum():6.1f}       "
          f"{100 * (yel & ~op_side).sum() / (plant & ~op_side).sum():6.1f}")

# 3. Per plant, with the strict cutoff: only clearly yellow-brown (senescent) tissue
senescent = plant & (H < 30) & (V > 150)
names = list(POTS)
px = np.array([POTS[n][0] for n in names])
py = np.array([POTS[n][1] for n in names])
rows, cols = np.nonzero(plant)
nearest = ((rows[:, None] - py[None, :]) ** 2 + (cols[:, None] - px[None, :]) ** 2).argmin(axis=1)
total = np.bincount(nearest, minlength=len(names))
sen = np.bincount(nearest, weights=senescent[rows, cols], minlength=len(names))

df = pd.DataFrame({
    "plant": names,
    "tray": [n[:2] for n in names],
    "senescent_cm2": (sen / PX_PER_CM**2).round(2),
    "senescent_%": (100 * sen / total).round(1),
})
print("\nper plant, hue < 30 (most senescent first):")
print(df.sort_values("senescent_%", ascending=False).head(6).to_string(index=False))
print("\naverage per tray:")
print(df.groupby("tray")[["senescent_cm2", "senescent_%"]].mean().round(2))
df.to_csv("outputs/m04_l08_senescence.csv", index=False)

overlay = img.copy()
overlay[senescent] = (255, 0, 255)                    # senescent tissue in magenta
cv2.imwrite("outputs/m04_l08_senescence.jpg", cv2.resize(overlay[800:2600, 150:3900], None, fx=0.3, fy=0.3))
print("\nsaved outputs/m04_l08_senescence.csv and outputs/m04_l08_senescence.jpg")
