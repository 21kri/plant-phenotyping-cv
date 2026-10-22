# Module 4, Lesson 2: convert pixel area to real area (cm²) with a known reference size
# Run from the project root (plant-cv-tutorial), after Lesson 1 has made the CSV.
import pandas as pd

# Reference: the box rim is 35 cm x 25 cm (IKEA 9 L box).
# Measured on the LP tray in the photo: 1653 px wide, 1168 px tall.
px_per_cm_w = 1653 / 35
px_per_cm_h = 1168 / 25
px_per_cm = (px_per_cm_w + px_per_cm_h) / 2
print(f"pixels per cm: width {px_per_cm_w:.1f}, height {px_per_cm_h:.1f}, average {px_per_cm:.1f}")

# 1 cm x 1 cm square = px_per_cm x px_per_cm pixels
px_per_cm2 = px_per_cm ** 2
print(f"pixels per cm2: {px_per_cm2:.0f}")

df = pd.read_csv("outputs/m04_l01_plant_areas.csv")      # table from Lesson 1
df["area_cm2"] = (df["area_px"] / px_per_cm2).round(1)   # new column
print(df)

print("\nAverage leaf area per tray (cm2):")
print(df.groupby("tray")["area_cm2"].mean().round(1))

df.to_csv("outputs/m04_l02_plant_areas_cm2.csv", index=False)
print("\nsaved outputs/m04_l02_plant_areas_cm2.csv")
