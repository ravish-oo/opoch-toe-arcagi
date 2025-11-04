#!/usr/bin/env python3
"""
Test script to verify canvas.py H8/H9 fixes for SHAPE_MISMATCH issues.

Tests two specific failing tasks:
- Task 00576224: H8 predicted 3×3, should be 6×6
- Task 017c7c7b: H9 predicted 1×1, should be 9×3
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.arcbit.canvas import choose_working_canvas

print("=" * 70)
print("CANVAS H8/H9 BUGFIX VERIFICATION")
print("=" * 70)

# =============================================================================
# Test 1: Task 00576224 (H8 issue: predicted 3×3, should be 6×6)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: Task 00576224 (H8 Issue)")
print("=" * 70)

# Training: 2×2 → 6×6 (both trainings)
train_pairs_1 = [
    {
        "X": [[0, 4], [4, 0]],
        "Y": [[0, 0, 0, 4, 4, 4], [0, 0, 0, 4, 4, 4], [0, 0, 0, 4, 4, 4],
              [4, 4, 4, 0, 0, 0], [4, 4, 4, 0, 0, 0], [4, 4, 4, 0, 0, 0]]
    },
    {
        "X": [[4, 0], [0, 4]],
        "Y": [[4, 4, 4, 0, 0, 0], [4, 4, 4, 0, 0, 0], [4, 4, 4, 0, 0, 0],
              [0, 0, 0, 4, 4, 4], [0, 0, 0, 4, 4, 4], [0, 0, 0, 4, 4, 4]]
    }
]

# Test input: 2×2
xstar_grid_1 = [[4, 4], [0, 0]]
xstar_shape_1 = (2, 2)
colors_order_1 = [0, 4, 6]

# Dummy frames (not used for canvas selection, just for receipts)
frames_in_1 = [{"H": 2, "W": 2}, {"H": 2, "W": 2}]
frames_out_1 = [{"H": 6, "W": 6}, {"H": 6, "W": 6}]

try:
    R_out_1, C_out_1, receipts_1 = choose_working_canvas(
        train_pairs_1, frames_in_1, frames_out_1,
        xstar_shape_1, colors_order_1, xstar_grid_1,
        families=("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")
    )

    print(f"\n✓ Canvas selection completed")
    print(f"  Predicted: {R_out_1}×{C_out_1}")
    print(f"  Expected:  6×6")

    winner = receipts_1["payload"]["winner"]
    print(f"  Winner family: {winner['family']}")

    if R_out_1 == 6 and C_out_1 == 6:
        print("\n✅ TEST 1 PASSED: Correct canvas size!")
    else:
        print(f"\n❌ TEST 1 FAILED: Got {R_out_1}×{C_out_1}, expected 6×6")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ TEST 1 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 2: Task 017c7c7b (H9 issue: predicted 1×1, should be 9×3)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Task 017c7c7b (H9 Issue)")
print("=" * 70)

# Training: 6×3 → 9×3 (all 3 trainings)
# Simplified grids (actual task has more complex patterns)
train_pairs_2 = [
    {
        "X": [[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1]],
        "Y": [[0, 2, 0], [2, 2, 2], [0, 2, 0], [2, 2, 2], [0, 2, 0], [2, 2, 2], [0, 2, 0], [2, 2, 2], [0, 2, 0]]
    },
    {
        "X": [[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
        "Y": [[2, 0, 2], [0, 2, 0], [2, 0, 2], [0, 2, 0], [2, 0, 2], [0, 2, 0], [2, 0, 2], [0, 2, 0], [2, 0, 2]]
    },
    {
        "X": [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]],
        "Y": [[0, 2, 2], [2, 2, 0], [2, 0, 2], [0, 2, 2], [2, 2, 0], [2, 0, 0], [0, 2, 2], [2, 2, 0], [2, 0, 0]]
    }
]

# Test input: 6×3
xstar_grid_2 = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 1]]
xstar_shape_2 = (6, 3)
colors_order_2 = [0, 1, 2]

frames_in_2 = [{"H": 6, "W": 3}, {"H": 6, "W": 3}, {"H": 6, "W": 3}]
frames_out_2 = [{"H": 9, "W": 3}, {"H": 9, "W": 3}, {"H": 9, "W": 3}]

try:
    R_out_2, C_out_2, receipts_2 = choose_working_canvas(
        train_pairs_2, frames_in_2, frames_out_2,
        xstar_shape_2, colors_order_2, xstar_grid_2,
        families=("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")
    )

    print(f"\n✓ Canvas selection completed")
    print(f"  Predicted: {R_out_2}×{C_out_2}")
    print(f"  Expected:  9×3")

    winner = receipts_2["payload"]["winner"]
    print(f"  Winner family: {winner['family']}")

    if R_out_2 == 9 and C_out_2 == 3:
        print("\n✅ TEST 2 PASSED: Correct canvas size!")
    else:
        print(f"\n❌ TEST 2 FAILED: Got {R_out_2}×{C_out_2}, expected 9×3")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ TEST 2 FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("✅ ALL CANVAS BUGFIX TESTS PASSED")
print("=" * 70)
print("\nFixes verified:")
print("  1. H8: Filters out inconsistent feature-dependent predictions")
print("  2. H9: Skips candidates with inconsistent guard evaluation")
print("=" * 70)

sys.exit(0)
