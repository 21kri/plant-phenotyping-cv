# Lesson 3: a comparison turns an image into a True/False mask
import numpy as np

# More realistic: soil and leaf brightness vary a little
img = np.array([
    [ 40,  35,  50,  45,  30],
    [ 55, 180, 210,  60,  40],
    [ 35, 190, 200, 170,  50],
    [ 45,  60, 220,  55,  35],
], dtype=np.uint8)

mask = img > 100          # True where the pixel is bright (leaf)
print("mask:\n", mask)
print("mask shape:", mask.shape, " dtype:", mask.dtype)

print("leaf pixels:", mask.sum())           # True counts as 1
print("leaf values:", img[mask])            # pick out only the leaf pixels
print("mean leaf brightness:", img[mask].mean())
