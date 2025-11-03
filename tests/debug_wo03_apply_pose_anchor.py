"""Debug apply_pose_anchor test failure."""

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
H = len(grid)
W = len(grid[0]) if grid else 0

colors = order_colors({c for row in grid for c in row} | {0})
planes = pack_grid_to_planes(grid, H, W, colors)

pid = "R90"
anchor = (1, 1)

print(f"Input: H={H}, W={W}, colors={colors}")
print(f"Pose: {pid}, Anchor: {anchor}")

# Apply via apply_pose_anchor
planes_transformed, H_out, W_out = apply_pose_anchor(planes, pid, anchor, H, W, colors)

# Manually apply via WO-01 ops
planes_manual = {}
H_manual, W_manual = H, W

for color in colors:
    plane = planes[color]
    print(f"\nColor {color}:")
    print(f"  Original plane length: {len(plane)}")

    # Step 1: pose
    plane_posed, H_p, W_p = pose_plane(plane, pid, H, W)
    print(f"  After pose {pid}: H={H_p}, W={W_p}, plane length={len(plane_posed)}")
    H_manual, W_manual = H_p, W_p

    # Step 2: shift by (-anchor[0], -anchor[1])
    plane_shifted = shift_plane(plane_posed, -anchor[0], -anchor[1], H_p, W_p)
    print(f"  After shift ({-anchor[0]}, {-anchor[1]}): plane length={len(plane_shifted)}")

    planes_manual[color] = plane_shifted

print(f"\nOutput shapes:")
print(f"  apply_pose_anchor: H={H_out}, W={W_out}")
print(f"  Manual: H={H_manual}, W={W_manual}")

# Compare results
print(f"\nPlane comparison:")
for color in colors:
    t = planes_transformed[color]
    m = planes_manual[color]
    match = t == m
    print(f"  Color {color}: match={match}, len_t={len(t)}, len_m={len(m)}")
    if not match:
        print(f"    Transformed: {t[:3] if len(t) >= 3 else t}")
        print(f"    Manual:      {m[:3] if len(m) >= 3 else m}")
