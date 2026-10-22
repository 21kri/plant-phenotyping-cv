# Module 3, Lesson 3: Otsu's method picks the threshold for you
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)

# Otsu needs a single-channel uint8 image, so clip G - B into 0..255 (Module 1, Lesson 4)
gb = np.clip(g - b, 0, 255).astype(np.uint8)

# The 0 is ignored: with THRESH_OTSU, OpenCV chooses the threshold itself and returns it as t
t, otsu = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu = otsu > 0

fixed = gb > 50
print("Otsu chose t =", t)
print("plant pixels, our  G - B > 50:", fixed.sum())
print("plant pixels, Otsu G - B >", int(t), ":", otsu.sum())

# Paint red the pixels the two rules disagree on (kept by 50, dropped by Otsu)
diff = fixed & ~otsu                  # ~ flips True <-> False
print("pixels only in the 50 mask:", diff.sum())
overlay = img.copy()
overlay[diff] = (0, 0, 255)
crop = overlay[800:2600, 150:3900]
cv2.imwrite("outputs/m03_l03_otsu_diff.jpg", cv2.resize(crop, None, fx=0.3, fy=0.3))
print("saved outputs/m03_l03_otsu_diff.jpg")
