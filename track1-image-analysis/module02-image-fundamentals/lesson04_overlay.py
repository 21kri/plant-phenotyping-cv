# Module 2, Lesson 4: paint the mask onto the photo to check it by eye
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
mask = (g - b) > 50

overlay = img.copy()                 # copy, so the original photo stays untouched
overlay[mask] = (255, 0, 255)        # every plant pixel -> magenta (B, G, R)

small = cv2.resize(overlay, None, fx=0.25, fy=0.25)
cv2.imwrite("outputs/m02_l04_overlay.jpg", small)
print("saved outputs/m02_l04_overlay.jpg, shape:", small.shape)
