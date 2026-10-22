# Module 8, Lesson 1: plant height from a side view, using the box (14 cm tall) as a ruler
# Run from the project root (plant-cv-tutorial).
import cv2
import numpy as np

BOX_HEIGHT_CM = 14.0

# Read by hand from a pixel grid: y of the top of the box rim, y of the bottom of the box
side_views = {
    "IMG_0288": {"tray": "OP", "rim_y": 1690, "bottom_y": 2855},
    "IMG_0294": {"tray": "LP", "rim_y": 1735, "bottom_y": 2905},
}

for name, info in side_views.items():
    img = cv2.imread(f"data/root-vigor-30-04/{name}.jpg")
    px_per_cm = (info["bottom_y"] - info["rim_y"]) / BOX_HEIGHT_CM

    # Plant pixels (colour rule), only ABOVE the rim - ignores leaves hanging over the edge
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    plant = ((g - b) > 40) & (g > r)
    plant[info["rim_y"]:, :] = False

    # Drop specks: shiny cloth reflections pass the colour rule as tiny blobs (Module 3, Lesson 6)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(plant.astype(np.uint8))
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= 2000)[0]
    big = big[big != 0]                                 # label 0 is the background
    plant = np.isin(labels, big)

    rows = np.nonzero(plant)[0]                         # row number of every plant pixel

    # Heights above the rim (in cm) of every plant pixel
    heights = (info["rim_y"] - rows) / px_per_cm
    canopy = np.percentile(heights, 95)                 # 95% of the leaf pixels are below this
    tallest = heights.max()

    print(f"{name} ({info['tray']}):  {px_per_cm:.1f} px/cm   "
          f"canopy height (95%) {canopy:.1f} cm   tallest leaf tip {tallest:.1f} cm")

    # Picture: rim line (yellow) and canopy line (magenta)
    canopy_y = int(info["rim_y"] - canopy * px_per_cm)
    cv2.line(img, (0, info["rim_y"]), (4031, info["rim_y"]), (0, 255, 255), 8)
    cv2.line(img, (0, canopy_y), (4031, canopy_y), (255, 0, 255), 8)
    cv2.putText(img, f"{info['tray']}  canopy {canopy:.1f} cm", (80, canopy_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 8)
    cv2.imwrite(f"outputs/m08_l01_{name}_height.jpg", cv2.resize(img, None, fx=0.25, fy=0.25))

print("saved outputs/m08_l01_IMG_0288_height.jpg and outputs/m08_l01_IMG_0294_height.jpg")
