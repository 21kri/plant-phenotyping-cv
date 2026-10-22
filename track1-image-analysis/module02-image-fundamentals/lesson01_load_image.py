# Module 2, Lesson 1: load a real plant photo with OpenCV
# Run from the project root (plant-cv-tutorial) so the relative path works.
import cv2

img = cv2.imread("data/IMG_0275.jpg")

# imread does NOT raise an error for a wrong path - it quietly returns None
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

print("shape:", img.shape)            # (height, width, channels)
print("dtype:", img.dtype)
print("centre pixel [B, G, R]:", img[img.shape[0] // 2, img.shape[1] // 2])

# Save a small copy you can open quickly
small = cv2.resize(img, None, fx=0.2, fy=0.2)
cv2.imwrite("outputs/m02_l01_small.jpg", small)
print("saved outputs/m02_l01_small.jpg, shape:", small.shape)

spots= {
    "leaf":        (1512, 2016),
    "dark ground": (1497, 3723),
    "reflection":  (627,  728),
}

for name, (r, c) in spots.items():
    b, g, r_ = img[r, c]
    brightness = (int(b) + int(g) + int(r_)) / 3
    print(f"{name:12s} B={b:3d} G={g:3d} R={r_:3d}   brightness={brightness:5.1f}")
