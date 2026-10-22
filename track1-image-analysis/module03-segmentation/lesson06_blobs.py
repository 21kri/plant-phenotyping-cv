# Module 3, Lesson 6: label each blob, give it its own colour, measure its area
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

# Cleanup pipeline from Lessons 3-5 (+ ROI from Module 2, Lesson 5)
b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
gb = np.clip(g - b, 0, 255).astype(np.uint8)
t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)
top, bottom, left, right = 800, 2600, 150, 3900
roi = np.zeros_like(mask)
roi[top:bottom, left:right] = mask[top:bottom, left:right]

# labels: same shape as the image; 0 = background, 1, 2, 3 ... = blob number
n, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
print("blobs:", n - 1)

# stats has one row per label: x, y, width, height, area
areas = stats[1:, cv2.CC_STAT_AREA]            # skip row 0 (background)
order = np.argsort(areas)[::-1]                # biggest first
print("\n  label    area   tray")
for i in order[:8]:
    label = i + 1
    cx = centroids[label][0]                   # centre column of the blob
    tray = "OP" if cx < 2130 else "LP"
    print(f"  {label:5d} {areas[i]:8d}   {tray}")

# Give every label a random colour (label 0 stays black)
rng = np.random.default_rng(0)
colours = rng.integers(60, 256, size=(n, 3), dtype=np.uint8)
colours[0] = 0
coloured = colours[labels]                     # look up a colour for every pixel
crop = coloured[top:bottom, left:right]
cv2.imwrite("outputs/m03_l06_blobs.png", cv2.resize(crop, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST))
print("\nsaved outputs/m03_l06_blobs.png")
