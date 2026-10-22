# Module 4, Lesson 9: is OP really bigger than LP? Testing it properly (and the pseudo-replication trap)
# Run from the project root (plant-cv-tutorial).
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("outputs/m04_l06_plant_traits.csv")
op = df.loc[df["tray"] == "OP", "area_cm2"]
lp = df.loc[df["tray"] == "LP", "area_cm2"]

# 1. The tempting test: treat the 9 plants per box as 9 independent replicates
t, p = stats.ttest_ind(op, lp, equal_var=False)            # Welch's t-test
print(f"naive t-test, 9 plants vs 9 plants:  t = {t:.2f}, p = {p:.4f}")

# 2. Why that p-value can't be trusted: a simulation where phosphate has NO effect at all.
#    Every box gets its own random "box effect" (its tank, position, light), and every
#    plant varies around its box. Sizes are made-up but realistic (cm2).
rng = np.random.default_rng(0)
mean_area, box_sd, plant_sd = 38, 8, 12


def experiment(boxes_per_treatment, plants_per_box=9):
    """One fake experiment with NO real treatment effect. Returns (naive p, correct p)."""
    groups = []
    for _ in range(2):                                         # two 'treatments', identical in truth
        boxes = [mean_area + rng.normal(0, box_sd) + rng.normal(0, plant_sd, plants_per_box)
                 for _ in range(boxes_per_treatment)]
        groups.append(boxes)
    plants_a, plants_b = np.concatenate(groups[0]), np.concatenate(groups[1])
    naive = stats.ttest_ind(plants_a, plants_b, equal_var=False).pvalue
    if boxes_per_treatment < 2:
        return naive, np.nan                                   # 1 box each: no correct test exists
    means_a = [b.mean() for b in groups[0]]                    # correct: one number per box
    means_b = [b.mean() for b in groups[1]]
    correct = stats.ttest_ind(means_a, means_b, equal_var=False).pvalue
    return naive, correct


print("\nSimulated experiments with NO real difference (should be 'significant' only 5% of the time):")
for n_boxes in (1, 3, 5):
    res = np.array([experiment(n_boxes) for _ in range(2000)])
    naive_rate = (res[:, 0] < 0.05).mean()
    correct_rate = np.nanmean(res[:, 1] < 0.05) if n_boxes > 1 else float("nan")
    correct_txt = "no test possible" if n_boxes == 1 else f"{correct_rate:.0%}"
    print(f"  {n_boxes} box(es) per treatment:  plants as replicates -> {naive_rate:.0%} false 'discoveries'   "
          f"boxes as replicates -> {correct_txt}")
