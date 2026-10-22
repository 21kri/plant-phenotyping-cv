# Module 4, Lesson 4: a bar chart of leaf area per plant, coloured by tray
# Run from the project root (plant-cv-tutorial), after Lesson 3 has made its CSV.
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/m04_l03_plant_traits.csv")

# Order: OP first, then LP; biggest plant first inside each tray
df = df.sort_values(["tray", "area_cm2"], ascending=[False, False])

tray_colour = {"OP": "#2a78d6", "LP": "#eb6834"}       # blue, orange
colours = df["tray"].map(tray_colour)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(df["plant"], df["area_cm2"], color=colours, width=0.7)

# Dashed line at each tray's average, only across that tray's bars
for tray, colour in tray_colour.items():
    positions = [i for i, t in enumerate(df["tray"]) if t == tray]
    mean = df.loc[df["tray"] == tray, "area_cm2"].mean()
    ax.hlines(mean, positions[0] - 0.4, positions[-1] + 0.4, colors="#52514e", linestyles="--", linewidth=1)
    ax.text(positions[-1] + 0.45, mean, f"{tray} mean {mean:.1f}", va="center", fontsize=9, color="#52514e")

# Legend: one coloured square per tray
for tray, colour in tray_colour.items():
    ax.bar(0, 0, color=colour, label=tray)
ax.legend(frameon=False)

ax.set_title("Visible leaf area per seedling", loc="left", fontsize=13)
ax.set_ylabel("Leaf area (cm²)")
ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)                      # grid behind the bars
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors="#52514e", length=0)

fig.tight_layout()
fig.savefig("outputs/m04_l04_leaf_area_bars.png", dpi=150)
print("saved outputs/m04_l04_leaf_area_bars.png")
