# Lesson 2: slicing = cropping a rectangle out of an image
import numpy as np

img = np.array([
    [0,   0,   0,   0,   0],
    [0, 200, 200,   0,   0],
    [0, 200, 200, 200,   0],
    [0,   0, 200,   0,   0],
], dtype=np.uint8)

# img[row_start:row_stop, col_start:col_stop]  (stop is NOT included)
crop = img[1:3, 1:4]   # rows 1-2, columns 1-3
print("crop:\n", crop)
print("crop shape:", crop.shape)

print("first column:", img[:, 0])   # ':' alone means "all rows"
