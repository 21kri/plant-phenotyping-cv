# Module 8, Lesson 3: a 3D plant as a point cloud - an (N, 3) array of x, y, z in cm
# Run from the project root (plant-cv-tutorial). No photos needed: we build a simple seedling.
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

rng = np.random.default_rng(0)


def leaf(angle_deg, length, width, tilt_deg, base_height):
    """Points filling a flat oval leaf blade on a stalk, pointing out at angle_deg and tilted up."""
    n = 1500
    u = rng.uniform(0, 1, n)                        # 0 = stalk end, 1 = leaf tip
    v = rng.uniform(-1, 1, n) * np.sin(np.pi * u)   # oval: widest in the middle
    keep = rng.uniform(0, 1, n) < np.sin(np.pi * u) + 0.05
    u, v = u[keep], v[keep]

    stalk = 2.0                                     # 2 cm stalk before the blade starts
    along = stalk + u * length                      # distance from the stem, along the leaf
    across = v * width / 2
    tilt = np.radians(tilt_deg)
    # leaf in its own frame, then rotate around the stem (z axis)
    x0, y0, z0 = along * np.cos(tilt), across, base_height + along * np.sin(tilt)
    a = np.radians(angle_deg)
    return np.column_stack([x0 * np.cos(a) - y0 * np.sin(a), x0 * np.sin(a) + y0 * np.cos(a), z0])


# A seedling: 4 leaves at different angles, lengths and tilts
plant = np.vstack([
    leaf(0,   6.0, 2.6, 45, 0.5),
    leaf(95,  5.0, 2.2, 55, 0.5),
    leaf(185, 6.5, 2.8, 35, 0.5),
    leaf(270, 4.0, 1.8, 65, 0.5),
])
print("point cloud shape:", plant.shape, "  (points, x/y/z)")
x, y, z = plant[:, 0], plant[:, 1], plant[:, 2]

# --- Measurements straight from the points ---
height = z.max() - 0                                            # stem base is at z = 0
top_area = ConvexHull(plant[:, :2]).volume                      # 2D hull: "volume" = area
hull_volume = ConvexHull(plant).volume                          # 3D rubber-band volume
block_volume = top_area * height                                # Lesson 2's area x height
print(f"height              {height:.1f} cm")
print(f"top-view hull area  {top_area:.1f} cm2")
print(f"area x height       {block_volume:.0f} cm3   (Lesson 2 method)")
print(f"3D convex hull      {hull_volume:.0f} cm3   (rubber band around the points)")

# --- Pictures: 3D view, view from above, view from the side ---
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(x, y, z, s=1, c=z, cmap="Greens", vmin=-2)
ax.set_title("3D point cloud", loc="left")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(x, y, s=1, color="#1baf7a")
ax2.set_aspect("equal"); ax2.set_title("From above (x, y)", loc="left")
ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)")

ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(x, z, s=1, color="#1baf7a")
ax3.set_aspect("equal"); ax3.set_title("From the side (x, z)", loc="left")
ax3.set_xlabel("x (cm)"); ax3.set_ylabel("z = height (cm)")

fig.subplots_adjust(left=0.02, right=0.98, wspace=0.35, bottom=0.15)
fig.savefig("outputs/m08_l03_point_cloud.png", dpi=130)
print("saved outputs/m08_l03_point_cloud.png")
