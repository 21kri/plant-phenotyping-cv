# Lesson 4: uint8 wraps around past 255 - brighten images safely
import numpy as np

row = np.array([30, 100, 200, 220, 250], dtype=np.uint8)

wrong = row + 50                      # stays uint8, so values past 255 wrap around
print("wrong:", wrong)

# Fix: do the math in a bigger type, clip to 0..255, convert back
right = np.clip(row.astype(np.int16) + 50, 0, 255).astype(np.uint8)
print("right:", right)
