# Module 6, Lesson 4: the network as a measuring tool - plant area per photo, network vs rule
# Run from the project root (plant-cv-tutorial), after Lesson 3 has saved outputs/m06_segmenter_jitter.pt.
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seg import load, predict, rule, split

model = load("outputs/m06_segmenter_jitter.pt")
train_names, val_names = split()

rows = []
photos = [(f"data/seg_dataset/images/{n}.jpg", "training photo" if n in train_names else "validation photo")
          for n in sorted(train_names + val_names)]
photos += [(str(p), "white room") for p in sorted(Path("data/oplp_outside_test").glob("*/*.jpg"))]

for path, group in photos:
    img = cv2.imread(path)[:, :, ::-1].astype(np.float32) / 255     # RGB, 0..1
    rule_mask, net_mask = rule(img), predict(model, img)
    rows.append({
        "photo": Path(path).stem,
        "group": group,
        "rule_%": round(100 * rule_mask.mean(), 1),      # mean of True/False = share of plant pixels
        "network_%": round(100 * net_mask.mean(), 1),
    })

df = pd.DataFrame(rows)
df["difference"] = df["network_%"] - df["rule_%"]
print(df.groupby("group")[["rule_%", "network_%", "difference"]].mean().round(1))
print("\nwhite room:\n", df[df["group"] == "white room"].to_string(index=False))
print("\nbiggest disagreements:\n", df.reindex(df["difference"].abs().sort_values(ascending=False).index).head(4).to_string(index=False))
print(f"\ncorrelation rule vs network (40 box photos): {df.loc[df['group'] != 'white room', ['rule_%', 'network_%']].corr().iloc[0, 1]:.2f}")
df.to_csv("outputs/m06_l04_area_rule_vs_network.csv", index=False)

# Scatter: each dot is one photo; dots on the dashed line = rule and network agree
colours = {"training photo": "#2a78d6", "validation photo": "#eb6834", "white room": "#1baf7a"}
fig, ax = plt.subplots(figsize=(6, 6))
for group, colour in colours.items():
    d = df[df["group"] == group]
    ax.scatter(d["rule_%"], d["network_%"], s=40, color=colour, label=group, zorder=3)
top = max(df["rule_%"].max(), df["network_%"].max()) * 1.1
ax.plot([0, top], [0, top], "--", color="#898781", linewidth=1, label="perfect agreement")
ax.set_xlim(0, top)
ax.set_ylim(0, top)
ax.set_xlabel("Plant pixels, colour rule (%)")
ax.set_ylabel("Plant pixels, network (%)")
ax.set_title("Plant area per photo: rule vs network", loc="left")
ax.grid(color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/m06_l04_rule_vs_network.png", dpi=150)
print("saved outputs/m06_l04_rule_vs_network.png and outputs/m06_l04_area_rule_vs_network.csv")
