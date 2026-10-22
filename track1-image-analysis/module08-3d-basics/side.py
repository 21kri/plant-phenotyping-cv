# Shared helper for Module 8: canopy height from a side view (Lesson 1 shows the steps).
import cv2
import numpy as np

BOX_HEIGHT_CM = 14.0


def canopy_height(photo, rim_y, bottom_y, percentile=95):
    """Return (px_per_cm, canopy height in cm) for a side-view photo of one box."""
    img = cv2.imread(f"data/root-vigor-30-04/{photo}.jpg")
    px_per_cm = (bottom_y - rim_y) / BOX_HEIGHT_CM

    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    plant = ((g - b) > 40) & (g > r)
    plant[rim_y:, :] = False                                   # only above the rim

    n, labels, stats, _ = cv2.connectedComponentsWithStats(plant.astype(np.uint8))
    big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= 2000)[0]    # drop specks
    plant = np.isin(labels, big[big != 0])

    heights = (rim_y - np.nonzero(plant)[0]) / px_per_cm
    return px_per_cm, float(np.percentile(heights, percentile))
