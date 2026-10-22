# Module 3, Lesson 4: "opening" removes specks smaller than a brush
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

# --- Part 1: tiny example you can read by eye -----------------------------
toy = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1],   # <- lone speck at row 2, col 6
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

brush = np.ones((3, 3), np.uint8)              # the "brush" (kernel): 3x3 square
opened = cv2.morphologyEx(toy, cv2.MORPH_OPEN, brush)
print("toy after opening:\n", opened)

# --- Part 2: the real photo -----------------------------------------------
img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
gb = np.clip(g - b, 0, 255).astype(np.uint8)
t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def count_blobs(m):
    return cv2.connectedComponents(m)[0] - 1   # -1: don't count the background

print("before opening:  blobs =", count_blobs(mask), " pixels =", (mask > 0).sum())
for size in (5, 15):
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))   # round brush
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)
    print(f"brush {size:2d}x{size:<2d}:   blobs = {count_blobs(opened)}  pixels = {(opened > 0).sum()}")

# --- Part 3: paint red what the 15x15 brush erased --------------------------
lost = (mask > 0) & ~(opened > 0)     # 'opened' is the 15x15 result (last in the loop)
overlay = img.copy()
overlay[lost] = (0, 0, 255)
crop = overlay[800:2600, 150:3900]
cv2.imwrite("outputs/m03_l04_lost15.jpg", cv2.resize(crop, None, fx=0.3, fy=0.3))
print("saved outputs/m03_l04_lost15.jpg")
