# Module 3, Lesson 7: one plant = one label, by giving each plant pixel to its nearest pot
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

# Cleanup pipeline from Lessons 3-5 (+ ROI)
b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
gb = np.clip(g - b, 0, 255).astype(np.uint8)
t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, brush)
top, bottom, left, right = 800, 2600, 150, 3900
roi = np.zeros(mask.shape, bool)
roi[top:bottom, left:right] = mask[top:bottom, left:right] > 0

# Pot centres (x = column, y = row), read off the photo by hand. Empty pots are left out.
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

# Row and column of every plant pixel
rows, cols = np.nonzero(roi)

# Squared distance from every plant pixel to every pot: shape (pixels, pots)
dist2 = (rows[:, None] - py[None, :]) ** 2 + (cols[:, None] - px[None, :]) ** 2
nearest = dist2.argmin(axis=1)          # index of the closest pot, for each pixel

areas = np.bincount(nearest, minlength=len(names))   # count pixels per pot
print("plant    area")
for name, area in zip(names, areas):
    print(f"{name:5s} {area:8d}")

# Colour each plant, mark the pots
hues = (np.arange(len(names)) * 7 % len(names)) * 180 // len(names)   # spread hues, neighbours differ
hsv = np.stack([hues, np.full_like(hues, 200), np.full_like(hues, 255)], axis=1).astype(np.uint8)
colours = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]                 # 1 row of HSV -> BGR
coloured = np.zeros_like(img)
coloured[rows, cols] = colours[nearest]
for name in names:
    x, y = pots[name]
    cv2.circle(coloured, (x, y), 25, (255, 255, 255), -1)
    cv2.putText(coloured, name, (x - 60, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
crop = coloured[top:bottom, left:right]
cv2.imwrite("outputs/m03_l07_plants.png", cv2.resize(crop, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
print("saved outputs/m03_l07_plants.png")
