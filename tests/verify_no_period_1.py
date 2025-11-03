#!/usr/bin/env python3
"""
Verification: Ensure period 1 never appears in receipts.

Test cases:
1. All zeros (constant)
2. All ones (constant)
3. Mixed constant rows
4. Large grids with various patterns
"""

import sys
sys.path.insert(0, '/Users/ravishq/code/opoch-toe-arcagi')

from src.arcbit.kernel import pack_grid_to_planes, period_2d_planes


def verify_no_period_1_in_receipts():
    """Exhaustively verify that period 1 never appears in receipts."""
    print("Verifying period 1 exclusion...")

    test_cases = [
        # All zeros (constant)
        {
            "label": "all_zeros",
            "G": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "H": 3, "W": 3, "colors": [0]
        },
        # All ones (constant)
        {
            "label": "all_ones",
            "G": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "H": 2, "W": 4, "colors": [0, 1]
        },
        # Mixed constant rows (period 2 vertical)
        {
            "label": "vert_stripes_period_2",
            "G": [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]],
            "H": 3, "W": 4, "colors": [0, 1]
        },
        # Large grid with period 3
        {
            "label": "period_3_pattern",
            "G": [
                [0, 1, 2, 0, 1, 2],
                [1, 2, 0, 1, 2, 0],
                [2, 0, 1, 2, 0, 1],
                [0, 1, 2, 0, 1, 2]
            ],
            "H": 4, "W": 6, "colors": [0, 1, 2]
        },
        # Checkerboard (period 2x2)
        {
            "label": "checkerboard_2x2",
            "G": [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ],
            "H": 4, "W": 4, "colors": [0, 1]
        },
        # Single row (no period possible)
        {
            "label": "single_row",
            "G": [[0, 1, 0, 1, 0, 1]],
            "H": 1, "W": 6, "colors": [0, 1]
        },
        # Single column (no period possible)
        {
            "label": "single_col",
            "G": [[0], [1], [0], [1]],
            "H": 4, "W": 1, "colors": [0, 1]
        },
    ]

    all_passed = True

    for tc in test_cases:
        G = tc["G"]
        H, W = tc["H"], tc["W"]
        colors = tc["colors"]
        label = tc["label"]

        planes = pack_grid_to_planes(G, H, W, colors)
        p_r, p_c, residues, receipts = period_2d_planes(
            planes, H, W, colors, return_receipts=True
        )

        # Check row_periods_nontrivial
        row_periods = receipts["period.candidates"]["row_periods_nontrivial"]
        if 1 in row_periods:
            print(f"❌ FAIL [{label}]: Period 1 in row_periods: {row_periods}")
            all_passed = False
        else:
            print(f"✓ [{label}] row_periods: {row_periods} (no 1)")

        # Check col_periods_nontrivial
        col_periods = receipts["period.candidates"]["col_periods_nontrivial"]
        if 1 in col_periods:
            print(f"❌ FAIL [{label}]: Period 1 in col_periods: {col_periods}")
            all_passed = False
        else:
            print(f"✓ [{label}] col_periods: {col_periods} (no 1)")

        # Check final p_r and p_c
        if p_r is not None and p_r < 2:
            print(f"❌ FAIL [{label}]: p_r={p_r} < 2")
            all_passed = False

        if p_c is not None and p_c < 2:
            print(f"❌ FAIL [{label}]: p_c={p_c} < 2")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Sub-WO-02a: Period 1 Exclusion Verification")
    print("=" * 60)
    print()

    passed = verify_no_period_1_in_receipts()

    print()
    print("=" * 60)
    if passed:
        print("✅ VERIFIED: Period 1 never appears in receipts!")
    else:
        print("❌ FAILED: Period 1 found in receipts!")
    print("=" * 60)
