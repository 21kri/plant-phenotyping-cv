# Lesson 1b: one index gives a row, two indices give a single pixel
import numpy as np

img = np.array([
    [0,   0,   0,   0,   0],
    [0, 200, 200,   0,   0],
    [0, 200, 200, 200,   0],
    [0,   0, 200,   0,   0],
], dtype=np.uint8)

print("img[1]    ->", img[1])      # ONE number: the whole row 1
print("img[1, 4] ->", img[1, 4])   # TWO numbers: row 1, then column 4 in that row
print("img[0, 0] ->", img[0, 0])   # top-left corner pixel

print("soil pixels:", (img == 0).sum(), "of", img.size)
