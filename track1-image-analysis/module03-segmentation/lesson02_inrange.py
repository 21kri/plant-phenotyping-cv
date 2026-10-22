# Module 3, Lesson 2: cv2.inRange picks pixels whose H, S and V all fall in a range
# Run from the project root (plant-cv-tutorial).
import cv2

img = cv2.imread("data/IMG_0275.jpg")
if img is None:
    raise SystemExit("Could not load the image - check the path and the folder you run from.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#              H    S    V
lower = (8,   80,  80)    # brown/orange hue, some colour, not too dark
upper = (25, 255, 255)
brown = cv2.inRange(hsv, lower, upper) > 0   # inRange gives 0/255; > 0 makes True/False

# Same ROI as Module 2, Lesson 5: ignore everything outside the trays
top, bottom, left, right = 800, 2600, 150, 3900
brown[:top] = False
brown[bottom:] = False
brown[:, :left] = False
brown[:, right:] = False

print("brown pixels:", brown.sum())

overlay = img.copy()
overlay[brown] = (255, 255, 0)               # cyan (B, G, R) - stands out on green
crop = overlay[top:bottom, left:right]       # only show the trays
cv2.imwrite("outputs/m03_l02_brown.jpg", cv2.resize(crop, None, fx=0.3, fy=0.3))
print("saved outputs/m03_l02_brown.jpg")
