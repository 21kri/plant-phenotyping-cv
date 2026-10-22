# Module 6, Lesson 3: fixed rule vs learned network when the lighting changes
# Run from the project root (plant-cv-tutorial), after Lesson 2 has saved outputs/m06_segmenter.pt.
# Takes about 2 minutes: it trains one new network.
import cv2
import numpy as np
import torch

from seg import iou, load, load_pair, predict, rule, split, train

train_names, val_names = split()

# Simulated lighting conditions (multiply R, G, B by these factors)
conditions = {
    "original":         (1.0, 1.0, 1.0),
    "dark (50%)":       (0.5, 0.5, 0.5),
    "very dark (30%)":  (0.3, 0.3, 0.3),
    "warm light":       (1.15, 1.0, 0.8),
    "cool light":       (0.85, 1.0, 1.2),
}

plain = load("outputs/m06_segmenter.pt")               # Lesson 2's network
print("training a second network with random lighting changes...")
jittered = train(train_names, lighting_jitter=True)
torch.save(jittered.state_dict(), "outputs/m06_segmenter_jitter.pt")

print(f"\n{'condition':17s} {'rule':>6s} {'network':>8s} {'network+jitter':>15s}   (mean IoU, 8 validation photos)")
for cond, gains in conditions.items():
    scores = {"rule": [], "plain": [], "jitter": []}
    for name in val_names:
        img, truth = load_pair(name)                    # truth = rule mask on the ORIGINAL photo
        shifted = np.clip(img * np.array(gains, np.float32), 0, 1)
        scores["rule"].append(iou(rule(shifted), truth))
        scores["plain"].append(iou(predict(plain, shifted), truth))
        scores["jitter"].append(iou(predict(jittered, shifted), truth))
    print(f"{cond:17s} {np.mean(scores['rule']):6.2f} {np.mean(scores['plain']):8.2f} {np.mean(scores['jitter']):15.2f}")

# Picture for one photo in very dark light: photo | rule | network | network+jitter
img, _ = load_pair(val_names[0])
dark = np.clip(img * 0.3, 0, 1)
tiles = [(dark[:, :, ::-1] * 255).astype(np.uint8)]
for m in (rule(dark), predict(plain, dark), predict(jittered, dark)):
    tiles.append(cv2.cvtColor(m.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
cv2.imwrite("outputs/m06_l03_very_dark.jpg", np.hstack(tiles))
print("\nsaved outputs/m06_l03_very_dark.jpg  (dark photo | rule | network | network+jitter)")
