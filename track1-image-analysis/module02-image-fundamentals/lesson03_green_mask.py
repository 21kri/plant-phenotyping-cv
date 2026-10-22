# Module 2, Lesson 3: apply a colour rule to every pixel to find the plants
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

# int16 so that G - B can go negative instead of wrapping (Module 1, Lesson 4)
b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)

mask = (g - b) > 50                  # True = plant, for all 12 million pixels at once

plant = mask.sum()
print("plant pixels:", plant)
print(f"plant share of photo: {100 * plant / mask.size:.1f} %")

# True/False -> 255/0 so it can be saved as a black-and-white image
cv2.imwrite("outputs/m02_l03_mask.png", mask.astype(np.uint8) * 255)
print("saved outputs/m02_l03_mask.png")
op = mask[:, :2130].sum()    # left tray = OP
lp = mask[:, 2130:].sum()    # right tray = LP
print("OP tray plant pixels:", op)
print("LP tray plant pixels:", lp)