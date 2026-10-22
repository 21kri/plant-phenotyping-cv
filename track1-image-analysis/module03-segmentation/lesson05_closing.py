# Module 3, Lesson 5: "closing" fills holes and gaps smaller than a brush
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

# --- Part 1: tiny example you can read by eye -----------------------------
toy = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0],   # <- one-pixel hole in the middle
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

brush = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(toy, cv2.MORPH_CLOSE, brush)
print("toy after closing:\n", closed)

# --- Part 2: the real photo (Otsu mask, cleaned with opening from Lesson 4) --
img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

b = img[:, :, 0].astype(np.int16)
g = img[:, :, 1].astype(np.int16)
gb = np.clip(g - b, 0, 255).astype(np.uint8)
t, mask = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, brush)     # Lesson 4: remove specks
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, brush)  # Lesson 5: fill holes

def count_blobs(m):
    return cv2.connectedComponents(m)[0] - 1

def count_holes(m):
    # a hole = a black area that does not touch the image border
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8))
    h, w = m.shape
    holes = 0
    for x, y, bw, bh, area in stats[1:]:
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            holes += 1
    return holes

print("after opening:   blobs =", count_blobs(opened), " holes =", count_holes(opened), " pixels =", (opened > 0).sum())
print("after closing:   blobs =", count_blobs(closed), " holes =", count_holes(closed), " pixels =", (closed > 0).sum())

# Zoom on one OP plant: photo on the left, mask on the right, closing's additions in red
top, bottom, left, right = 1150, 1450, 1430, 1730
added = (closed > 0) & ~(opened > 0)

mask_view = cv2.cvtColor(opened[top:bottom, left:right], cv2.COLOR_GRAY2BGR)  # 1 channel -> 3
mask_view[added[top:bottom, left:right]] = (0, 0, 255)
side_by_side = np.hstack([img[top:bottom, left:right], mask_view])           # glue left|right
big = cv2.resize(side_by_side, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("outputs/m03_l05_closing_zoom.png", big)
print("saved outputs/m03_l05_closing_zoom.png")
