"""Detailed debug of apply_pose_anchor for color 0."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.kernel import (
    pack_grid_to_planes,
    order_colors,
    pose_plane,
    shift_plane,
)
from arcbit.kernel.frames import apply_pose_anchor


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
with open(arc_data_path, "r") as f:
    tasks = json.load(f)

task = tasks["00576224"]
grid = task["train"][0]["input"]

print("Grid:")
for row in grid:
    print(f"  {row}")

H = len(grid)
W = len(grid[0]) if grid else 0

colors = order_colors({c for row in grid for c in row} | {0})
planes_orig = pack_grid_to_planes(grid, H, W, colors)

print(f"\nColor 0 original plane:")
for i, mask in enumerate(planes_orig[0]):
    print(f"  Row {i}: {mask:08b} (decimal {mask})")

# Make a copy for manual test
planes_for_apply = {c: list(p) for c, p in planes_orig.items()}
planes_for_manual = {c: list(p) for c, p in planes_orig.items()}

pid = "R90"
anchor = (1, 1)

# Apply via apply_pose_anchor
print(f"\n=== apply_pose_anchor ===")
planes_transformed, H_out, W_out = apply_pose_anchor(planes_for_apply, pid, anchor, H, W, colors)

print(f"Color 0 after apply_pose_anchor:")
for i, mask in enumerate(planes_transformed[0]):
    print(f"  Row {i}: {mask:08b} (decimal {mask})")

# Manual
print(f"\n=== Manual composition ===")
plane_0 = planes_for_manual[0]
print(f"Color 0 initial plane: {plane_0}")

plane_0_posed, H_p, W_p = pose_plane(plane_0, pid, H, W)
print(f"Color 0 after pose {pid}: {plane_0_posed} (H={H_p}, W={W_p})")
for i, mask in enumerate(plane_0_posed):
    print(f"  Row {i}: {mask:08b} (decimal {mask})")

plane_0_shifted = shift_plane(plane_0_posed, -anchor[0], -anchor[1], H_p, W_p)
print(f"Color 0 after shift ({-anchor[0]}, {-anchor[1]}): {plane_0_shifted}")
for i, mask in enumerate(plane_0_shifted):
    print(f"  Row {i}: {mask:08b} (decimal {mask})")
