# Module 8, Lesson 4: the first step of 3D - find the same physical points in two photos
# Run from the project root (plant-cv-tutorial). Uses two oblique views of the LP-C box.
import cv2
import numpy as np


def load(photo):
    img = cv2.imread(f"data/root-vigor-30-04/{photo}.jpg")
    return cv2.resize(img, (1344, 1008), interpolation=cv2.INTER_AREA)   # 1/3 size: faster


def is_plant(img, pts):
    """True for points that sit on a green plant pixel (colour rule, Module 2)."""
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    plant = ((g - b) > 40) & (g > r)
    return np.array([plant[int(y), int(x)] for x, y in pts])


a, b = load("IMG_0310"), load("IMG_0315")

# 1. Keypoints: distinctive little patches (corners, spots, edges) + a description of each
sift = cv2.SIFT_create(4000)
kp_a, desc_a = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
kp_b, desc_b = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
print("keypoints:", len(kp_a), "in photo A,", len(kp_b), "in photo B")

# 2. Match descriptions; keep a match only if it is clearly better than the 2nd-best (ratio test)
pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
print("matches after ratio test:", len(good))

# 3. Geometry check: real matches must agree with ONE camera movement (RANSAC throws out the rest)
pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
F, inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.99)
keep = inliers.ravel() == 1
pts_a, pts_b = pts_a[keep], pts_b[keep]
print("matches that fit one camera movement:", len(pts_a))

on_plant = is_plant(a, pts_a)
print(f"of those, on plants: {on_plant.sum()}   on box / labels / cloth: {(~on_plant).sum()}")

# Picture: both photos side by side, matches joined by lines (green = on a plant, grey = elsewhere)
canvas = np.hstack([a, b])
for (xa, ya), (xb, yb), plant in zip(pts_a, pts_b, on_plant):
    colour = (0, 220, 0) if plant else (170, 170, 170)
    cv2.line(canvas, (int(xa), int(ya)), (int(xb) + a.shape[1], int(yb)), colour, 1)
    cv2.circle(canvas, (int(xa), int(ya)), 3, colour, -1)
    cv2.circle(canvas, (int(xb) + a.shape[1], int(yb)), 3, colour, -1)
cv2.imwrite("outputs/m08_l04_matches.jpg", canvas)
print("saved outputs/m08_l04_matches.jpg")
