# Lesson 5: a colour image has 3 numbers per pixel (OpenCV order: Blue, Green, Red)
import numpy as np

soil = [40, 70, 110]    # B, G, R -> brownish
leaf = [50, 160, 60]    # B, G, R -> green

# 2 rows x 3 columns, each pixel is [B, G, R]
img = np.array([
    [soil, leaf, soil],
    [leaf, leaf, soil],
], dtype=np.uint8)

print("shape:", img.shape)           # (rows, columns, channels)
print("pixel [0, 1]:", img[0, 1])    # all 3 numbers of one pixel

green = img[:, :, 1]                 # channel 1 = green, for every pixel
print("green channel:\n", green)
print("green mask:\n", green > 100)
