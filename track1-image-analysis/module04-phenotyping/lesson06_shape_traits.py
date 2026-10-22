# Module 4, Lesson 6: shape traits per plant - perimeter, convex hull, solidity, width
# Run from the project root (plant-cv-tutorial), after Lesson 5 has made its CSV.
import cv2
import numpy as np
import pandas as pd

from plants import label_plants, PX_PER_CM

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

names, rows, cols, nearest = label_plants(img)
overlay = img.copy()
results = []

for i, name in enumerate(names):
    # Black image with only this plant's pixels in white
    plant = np.zeros(img.shape[:2], np.uint8)
    plant[rows[nearest == i], cols[nearest == i]] = 255

    # Outlines of every piece of the plant (a plant can be split into several pieces)
    contours, _ = cv2.findContours(plant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # Rubber band around ALL pieces together
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    hull_area = cv2.contourArea(hull)

    # Width = diameter of the smallest circle that fits around the plant
    (cx, cy), radius = cv2.minEnclosingCircle(all_points)

    area = (nearest == i).sum()
    results.append({
        "plant": name,
        "perimeter_cm": round(perimeter / PX_PER_CM, 1),
        "hull_cm2": round(hull_area / PX_PER_CM**2, 1),
        "solidity": round(area / hull_area, 2),
        "width_cm": round(2 * radius / PX_PER_CM, 1),
    })

    cv2.drawContours(overlay, contours, -1, (255, 0, 255), 4)        # magenta outline
    cv2.drawContours(overlay, [hull], -1, (0, 255, 255), 6)          # yellow rubber band

shape = pd.DataFrame(results)
df = pd.read_csv("outputs/m04_l05_plant_traits.csv").merge(shape, on="plant")
print(df[["plant", "area_cm2", "perimeter_cm", "hull_cm2", "solidity", "width_cm"]])

df.to_csv("outputs/m04_l06_plant_traits.csv", index=False)
crop = overlay[800:2600, 150:3900]
cv2.imwrite("outputs/m04_l06_shapes.jpg", cv2.resize(crop, None, fx=0.3, fy=0.3))
print("\nsaved outputs/m04_l06_plant_traits.csv and outputs/m04_l06_shapes.jpg")
