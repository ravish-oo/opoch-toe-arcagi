"""Check if apply_pose_anchor mutates input or has cross-contamination."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.kernel import pack_grid_to_planes, order_colors
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

print(f"\nOriginal planes (before apply_pose_anchor):")
for c in colors:
    print(f"  Color {c}: {planes_orig[c]}")

# Make deep copy
planes_copy = {c: list(p) for c, p in planes_orig.items()}

print(f"\nCopy (before apply_pose_anchor):")
for c in colors:
    print(f"  Color {c}: {planes_copy[c]}")

pid = "R90"
anchor = (1, 1)

print(f"\nCalling apply_pose_anchor(planes_copy, '{pid}', {anchor}, {H}, {W}, {colors})")
planes_transformed, H_out, W_out = apply_pose_anchor(planes_copy, pid, anchor, H, W, colors)

print(f"\nCopy (after apply_pose_anchor - should be unchanged):")
for c in colors:
    print(f"  Color {c}: {planes_copy[c]}")

print(f"\nTransformed planes:")
for c in colors:
    print(f"  Color {c}: {planes_transformed[c]}")

print(f"\nChecking if copy was mutated:")
for c in colors:
    if planes_copy[c] != planes_orig[c]:
        print(f"  ❌ Color {c} WAS MUTATED: {planes_orig[c]} -> {planes_copy[c]}")
    else:
        print(f"  ✅ Color {c} unchanged")
