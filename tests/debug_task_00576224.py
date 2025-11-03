"""Debug task 00576224 to understand its period structure."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.kernel import pack_grid_to_planes, order_colors
from arcbit.kernel.period import period_2d_planes


arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
with open(arc_data_path, 'r') as f:
    tasks = json.load(f)

task = tasks["00576224"]

print("=" * 70)
print("TASK 00576224 ANALYSIS")
print("=" * 70)

for idx, example in enumerate(task["train"]):
    print(f"\n[TRAINING EXAMPLE {idx}]")
    print("-" * 70)

    input_grid = example["input"]
    output_grid = example["output"]

    print(f"Input ({len(input_grid)}×{len(input_grid[0]) if input_grid else 0}):")
    for row in input_grid:
        print(f"  {row}")

    print(f"\nOutput ({len(output_grid)}×{len(output_grid[0]) if output_grid else 0}):")
    for row in output_grid:
        print(f"  {row}")

    # Analyze output period
    H, W = len(output_grid), len(output_grid[0]) if output_grid else 0
    if H > 0 and W > 0:
        colors_set = {0} | {c for row in output_grid for c in row}
        colors_order = order_colors(colors_set)

        planes = pack_grid_to_planes(output_grid, H, W, colors_order)
        p_r, p_c, residues = period_2d_planes(planes, H, W, colors_order)

        print(f"\nPeriod Analysis:")
        print(f"  p_r (row period): {p_r}")
        print(f"  p_c (column period): {p_c}")
        print(f"  Residues: {len(residues)}")

        # Check if rows are identical
        all_rows_same = all(output_grid[0] == output_grid[r] for r in range(H))
        print(f"  All rows identical: {all_rows_same}")

        # Check column pattern
        if p_c:
            print(f"  Column pattern (period {p_c}):")
            for c in range(min(W, p_c * 2)):
                col_vals = [output_grid[r][c] for r in range(H)]
                print(f"    col {c}: {col_vals}")
