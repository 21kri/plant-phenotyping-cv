# Lesson 1: an image is just a grid of numbers (a NumPy array)
import numpy as np

# A tiny 4x5 grayscale "image": 0 = black soil, 200 = bright leaf
img = np.array([
    [0,   0,   0,   0,   0],
    [0, 200, 200,   0,   0],
    [0, 200, 200, 200,   0],
    [0,   0, 200,   0,   0],
], dtype=np.uint8)

print("shape:", img.shape)   # (rows, columns) = (height, width)
print("dtype:", img.dtype)   # uint8 = whole numbers from 0 to 255
print("pixel at row 1, col 2:", img[1, 2])
print("min / max:", img.min(), img.max())

# Bonus: count leaf pixels. (img == 200) is a True/False grid; sum() counts the Trues.
print("leaf pixels:", (img == 200).sum())
print("img[3, 2] =", img[3, 2], " img[2, 3] =", img[2, 3])
print ("img[0, 0] , img[0, 0])
       
       
       
       

