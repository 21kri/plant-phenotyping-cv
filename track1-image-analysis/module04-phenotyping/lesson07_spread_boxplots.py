# Module 4, Lesson 7: how spread out is each trait? describe() + box plots
# Run from the project root (plant-cv-tutorial), after Lesson 6 has made its CSV.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/m04_l06_plant_traits.csv")

# One line per tray: count, mean, std, min, 25%, median (50%), 75%, max
print(df.groupby("tray")["area_cm2"].describe().round(1))

traits = {
    "area_cm2": "Leaf area (cm²)",
    "solidity": "Solidity",
    "ExG":      "ExG",
    "width_cm": "Width (cm)",
}
trays = ["OP", "LP"]
tray_colour = {"OP": "#2a78d6", "LP": "#eb6834"}

# One small chart per trait, side by side (each with its own y-axis)
fig, axes = plt.subplots(1, len(traits), figsize=(12, 4))
for ax, (col, label) in zip(axes, traits.items()):
    groups = [df.loc[df["tray"] == t, col] for t in trays]
    ax.boxplot(groups, widths=0.5, showfliers=False,
               medianprops={"color": "#0b0b0b"}, boxprops={"color": "#52514e"},
               whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})

    # Every plant as a dot, spread a little sideways so dots don't hide each other
    rng = np.random.default_rng(0)
    for i, (t, values) in enumerate(zip(trays, groups), start=1):
        x = i + rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(x, values, s=28, color=tray_colour[t], zorder=3)

    ax.set_xticks([1, 2], trays)
    ax.set_title(label, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)

fig.suptitle("OP vs LP: 9 seedlings per tray", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig("outputs/m04_l07_boxplots.png", dpi=150)
print("\nsaved outputs/m04_l07_boxplots.png")
