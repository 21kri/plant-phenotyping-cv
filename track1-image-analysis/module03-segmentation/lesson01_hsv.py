# Module 3, Lesson 1: HSV describes colour as Hue, Saturation, Value
# Run from the project root (plant-cv-tutorial).
import cv2

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # same shape, 3 new channels: H, S, V

spots = {
    "leaf":        (1512, 2016),
    "stem base":   (1314, 2878),
    "dark ground": (1497, 3723),
    "reflection":  (627,  728),
}

print("               B   G   R  |   H   S   V")
for name, (r, c) in spots.items():
    b, g, r_ = img[r, c]
    h, s, v = hsv[r, c]
    print(f"{name:12s} {b:3d} {g:3d} {r_:3d}  | {h:3d} {s:3d} {v:3d}")
