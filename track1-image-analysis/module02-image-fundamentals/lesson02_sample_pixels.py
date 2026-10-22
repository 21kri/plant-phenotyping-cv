# Module 2, Lesson 2: look at real pixel values before choosing a threshold
# Run from the project root (plant-cv-tutorial).
import cv2

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

# (row, col) of three hand-picked spots in the photo
spots = {
    "leaf":        (1512, 2016),
    "dark ground": (1497, 3723),
    "reflection":  (627,  728),
}

for name, (r, c) in spots.items():
    b, g, r_ = img[r, c]
    brightness = (int(b) + int(g) + int(r_)) / 3
    print(f"{name:12s} B={b:3d} G={g:3d} R={r_:3d}   brightness={brightness:5.1f}")
