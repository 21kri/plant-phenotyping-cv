# Module 4, Lesson 3: a greenness score per plant (average hue of its pixels)
# Run from the project root (plant-cv-tutorial), after Lesson 2 has made its CSV.
import cv2
import numpy as np
import pandas as pd

from plants import label_plants     # the mask + nearest-pot steps, in plants.py

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

names, rows, cols, nearest = label_plants(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[rows, cols, 0].astype(float)     # hue of every plant pixel (~30 = yellow, ~60 = green)

# Average per plant: sum of hues per plant / number of pixels per plant
hue_sum = np.bincount(nearest, weights=hue, minlength=len(names))
count = np.bincount(nearest, minlength=len(names))
mean_hue = hue_sum / count

df = pd.read_csv("outputs/m04_l02_plant_areas_cm2.csv")   # table from Lesson 2
df["mean_hue"] = mean_hue.round(1)
print(df.sort_values("mean_hue"))                          # yellowest plants first

print("\nAverage hue per tray:")
print(df.groupby("tray")["mean_hue"].mean().round(1))

df.to_csv("outputs/m04_l03_plant_traits.csv", index=False)
print("\nsaved outputs/m04_l03_plant_traits.csv")
