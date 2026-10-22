# Module 2, Lesson 5: keep only the region of interest (ROI) to drop specks outside the trays
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
mask = (g - b) > 50

# ROI: a rectangle around both trays, with room for leaves hanging over the edges
top, bottom, left, right = 800, 2600, 150, 3900

clean = np.zeros_like(mask)                                   # all False
clean[top:bottom, left:right] = mask[top:bottom, left:right]  # copy only the ROI

print("plant pixels before:", mask.sum())
print("plant pixels after: ", clean.sum())
print("removed:            ", mask.sum() - clean.sum())
print("OP tray:", clean[:, :2130].sum(), "  LP tray:", clean[:, 2130:].sum())

overlay = img.copy()
overlay[clean] = (255, 0, 255)
cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), 12)  # yellow ROI box
cv2.imwrite("outputs/m02_l05_roi.jpg", cv2.resize(overlay, None, fx=0.25, fy=0.25))
print("saved outputs/m02_l05_roi.jpg")
