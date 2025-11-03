#!/usr/bin/env python3
"""
Verification test for WO-02 implementation.

Tests:
1. minimal_period_1d (alias)
2. period_2d_planes with 2D periodic patterns
3. period_2d_planes with vertical stripes (only p_c)
4. period_2d_planes with horizontal stripes (only p_r)
5. period_2d_planes with no period
6. Residue masks correctness
7. Edge cases (small grids)
"""

from src.arcbit.kernel import (
    pack_grid_to_planes,
    minimal_period_1d,
    minimal_period_row,
    period_2d_planes
)


def test_minimal_period_1d_alias():
    """Test minimal_period_1d is an alias for minimal_period_row."""
    print("Testing minimal_period_1d (alias)...")

    # Test several cases
    test_cases = [
        (0b101010, 6, 2),
        (0b110110, 6, 3),
        (0b111111, 6, None),  # constant
        (0b000000, 6, None),  # constant
        (0b101011, 6, None),  # no period
    ]

    for mask, W, expected in test_cases:
        p1 = minimal_period_1d(mask, W)
        p2 = minimal_period_row(mask, W)
        assert p1 == p2, f"Alias broken: {p1} != {p2} for mask={mask:b}, W={W}"
        assert p1 == expected, f"Wrong result: got {p1}, expected {expected}"

    print("✓ minimal_period_1d is correct alias")


def test_period_2d_checkerboard():
    """Test 2D period on checkerboard pattern tiled 2x2."""
    print("Testing period_2d_planes on 2x2 checkerboard...")

    # 4x4 grid: checkerboard with period (2, 2)
    # Pattern:
    #   0 1 0 1
    #   1 0 1 0
    #   0 1 0 1
    #   1 0 1 0
    G = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    H, W = 4, 4
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    # Expected: p_r=2, p_c=2
    assert p_r == 2, f"Expected p_r=2, got {p_r}"
    assert p_c == 2, f"Expected p_c=2, got {p_c}"

    # Expected: 2*2 = 4 residues
    assert len(residues) == 4, f"Expected 4 residues, got {len(residues)}"

    # Verify residue (0,0): rows 0,2; cols 0,2
    # Should have bits at (0,0), (0,2), (2,0), (2,2)
    res_00 = residues[0]  # i=0, j=0
    assert len(res_00) == H, "Residue height mismatch"

    # Check specific bits for residue (0,0)
    # Row 0: bits at cols 0, 2 → mask = 0b0101
    assert res_00[0] == 0b0101, f"Residue (0,0) row 0 wrong: {res_00[0]:b}"
    # Row 1: no bits (row 1 % 2 == 1, not 0) → mask = 0b0000
    assert res_00[1] == 0b0000, f"Residue (0,0) row 1 wrong: {res_00[1]:b}"
    # Row 2: bits at cols 0, 2 → mask = 0b0101
    assert res_00[2] == 0b0101, f"Residue (0,0) row 2 wrong: {res_00[2]:b}"
    # Row 3: no bits
    assert res_00[3] == 0b0000, f"Residue (0,0) row 3 wrong: {res_00[3]:b}"

    print(f"✓ 2D checkerboard: p_r={p_r}, p_c={p_c}, {len(residues)} residues")


def test_period_2d_vertical_stripes():
    """Test 2D period on vertical stripes (only p_c)."""
    print("Testing period_2d_planes on vertical stripes...")

    # 3x6 grid: vertical stripes with period 2
    # Pattern:
    #   0 1 0 1 0 1
    #   0 1 0 1 0 1
    #   0 1 0 1 0 1
    G = [
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1]
    ]
    H, W = 3, 6
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    # Expected: p_r=None (no row period), p_c=2
    assert p_r is None, f"Expected p_r=None for vertical stripes, got {p_r}"
    assert p_c == 2, f"Expected p_c=2, got {p_c}"

    # Expected: 1*2 = 2 residues (internally p_r=1 for construction)
    assert len(residues) == 2, f"Expected 2 residues, got {len(residues)}"

    print(f"✓ Vertical stripes: p_r={p_r}, p_c={p_c}, {len(residues)} residues")


def test_period_2d_horizontal_stripes():
    """Test 2D period on horizontal stripes (only p_r)."""
    print("Testing period_2d_planes on horizontal stripes...")

    # 6x3 grid: horizontal stripes with period 2
    # Pattern:
    #   0 0 0
    #   1 1 1
    #   0 0 0
    #   1 1 1
    #   0 0 0
    #   1 1 1
    G = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    H, W = 6, 3
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    # Expected: p_r=2, p_c=None (no column period)
    assert p_r == 2, f"Expected p_r=2, got {p_r}"
    assert p_c is None, f"Expected p_c=None for horizontal stripes, got {p_c}"

    # Expected: 2*1 = 2 residues (internally p_c=1 for construction)
    assert len(residues) == 2, f"Expected 2 residues, got {len(residues)}"

    print(f"✓ Horizontal stripes: p_r={p_r}, p_c={p_c}, {len(residues)} residues")


def test_period_2d_no_period():
    """Test 2D period on non-periodic pattern."""
    print("Testing period_2d_planes on non-periodic pattern...")

    # 3x3 grid: no period
    G = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1]
    ]
    H, W = 3, 3
    colors = [0, 1, 2]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    # Expected: both None
    assert p_r is None, f"Expected p_r=None, got {p_r}"
    assert p_c is None, f"Expected p_c=None, got {p_c}"

    # Expected: no residues
    assert len(residues) == 0, f"Expected 0 residues, got {len(residues)}"

    print(f"✓ No period: p_r={p_r}, p_c={p_c}, {len(residues)} residues")


def test_period_2d_small_grid():
    """Test 2D period on small grids (H<2 or W<2)."""
    print("Testing period_2d_planes on small grids...")

    # 1x4 grid (H<2): no row period possible
    G = [[0, 1, 0, 1]]
    H, W = 1, 4
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    assert p_r is None, f"Expected p_r=None for H<2, got {p_r}"
    assert p_c is None, f"Expected p_c=None for H<2, got {p_c}"
    assert len(residues) == 0, f"Expected 0 residues for small grid, got {len(residues)}"

    # 4x1 grid (W<2): no column period possible
    G = [[0], [1], [0], [1]]
    H, W = 4, 1
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    assert p_r is None, f"Expected p_r=None for W<2, got {p_r}"
    assert p_c is None, f"Expected p_c=None for W<2, got {p_c}"
    assert len(residues) == 0, f"Expected 0 residues for small grid, got {len(residues)}"

    print("✓ Small grids handled correctly")


def test_period_2d_residue_reconstruction():
    """Test that residues correctly tile to reconstruct period."""
    print("Testing residue reconstruction...")

    # 4x4 checkerboard with period (2, 2)
    G = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    H, W = 4, 4
    colors = [0, 1]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    assert p_r == 2 and p_c == 2, "Periods wrong"
    assert len(residues) == 4, "Residue count wrong"

    # Verify that combining all residues covers the entire grid exactly once
    # (This is a sanity check on residue construction)
    combined_mask = [0] * H
    for residue in residues:
        for r in range(H):
            # Check no overlap
            assert (combined_mask[r] & residue[r]) == 0, f"Residues overlap at row {r}"
            combined_mask[r] |= residue[r]

    # Check all cells covered
    expected_full_mask = (1 << W) - 1  # all W bits set
    for r in range(H):
        assert combined_mask[r] == expected_full_mask, f"Not all cells covered at row {r}"

    print("✓ Residues tile correctly (no gaps, no overlaps)")


def test_period_2d_multicolor():
    """Test 2D period on multi-color (K=3) pattern."""
    print("Testing period_2d_planes on 3-color pattern...")

    # 4x6 grid with period (2, 3): each 2x3 tile is identical
    # Tile:
    #   0 1 2
    #   1 2 0
    # Tiled 2x2:
    #   0 1 2 | 0 1 2
    #   1 2 0 | 1 2 0
    #   ------+------
    #   0 1 2 | 0 1 2
    #   1 2 0 | 1 2 0
    G = [
        [0, 1, 2, 0, 1, 2],
        [1, 2, 0, 1, 2, 0],
        [0, 1, 2, 0, 1, 2],
        [1, 2, 0, 1, 2, 0]
    ]
    H, W = 4, 6
    colors = [0, 1, 2]

    planes = pack_grid_to_planes(G, H, W, colors)
    p_r, p_c, residues = period_2d_planes(planes, H, W, colors)

    # Expected: p_r=2, p_c=3
    assert p_r == 2, f"Expected p_r=2, got {p_r}"
    assert p_c == 3, f"Expected p_c=3, got {p_c}"

    # Expected: 2*3 = 6 residues
    assert len(residues) == 6, f"Expected 6 residues, got {len(residues)}"

    print(f"✓ Multi-color (K=3): p_r={p_r}, p_c={p_c}, {len(residues)} residues")


if __name__ == "__main__":
    print("=" * 60)
    print("WO-02 Verification Tests")
    print("=" * 60)
    print()

    test_minimal_period_1d_alias()
    test_period_2d_checkerboard()
    test_period_2d_vertical_stripes()
    test_period_2d_horizontal_stripes()
    test_period_2d_no_period()
    test_period_2d_small_grid()
    test_period_2d_residue_reconstruction()
    test_period_2d_multicolor()

    print()
    print("=" * 60)
    print("✅ All WO-02 tests passed!")
    print("=" * 60)
    print()
    print("WO-02 Implementation Complete:")
    print("  - minimal_period_1d (alias for minimal_period_row)")
    print("  - period_2d_planes with K-tuple KMP")
    print("  - LCM aggregation with global validation")
    print("  - Residue masks with phase (0,0)")
    print("  - Pure tuple equality (no hashing)")
    print("  - Deterministic, no floats, exact")
