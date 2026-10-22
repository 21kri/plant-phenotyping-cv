# Module 8, Lesson 2: combine top-view leaf area and side-view height into a canopy volume
# Run from the project root (plant-cv-tutorial). Uses the Module 4 table for leaf area.
import pandas as pd

from side import canopy_height

# The SAME two boxes from both directions:
#   top view  IMG_0275: left box = OP, right box = LP-C   (measured in Modules 2-4)
#   side view IMG_0288 = OP,  IMG_0314 = LP-C            (rim / bottom y read off a pixel grid)
boxes = {
    "OP": {"side": "IMG_0288", "rim_y": 1690, "bottom_y": 2855},
    "LP": {"side": "IMG_0314", "rim_y": 1735, "bottom_y": 2935},
}

traits = pd.read_csv("outputs/m04_l06_plant_traits.csv")     # per-plant leaf area from the top view

rows = []
for tray, info in boxes.items():
    px_per_cm, height = canopy_height(info["side"], info["rim_y"], info["bottom_y"])
    area = traits.loc[traits["tray"] == tray, "area_cm2"]     # 9 plants per box
    rows.append({
        "box": tray,
        "plants": len(area),
        "leaf_area_cm2": round(area.sum(), 1),               # whole box, from above
        "canopy_height_cm": round(height, 1),                # from the side
        "volume_cm3": round(area.sum() * height),            # area x height
        "volume_per_plant_cm3": round(area.mean() * height),
    })

df = pd.DataFrame(rows).set_index("box")
print(df.T)                                                 # .T flips rows and columns: easier to read

print("\nOP / LP ratio:")
for col in ("leaf_area_cm2", "canopy_height_cm", "volume_cm3"):
    print(f"  {col:18s} {df.loc['OP', col] / df.loc['LP', col]:.2f}x")
