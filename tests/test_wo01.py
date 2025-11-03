#!/usr/bin/env python3
"""
Verification test for WO-01 implementation.

Tests:
1. PACK/UNPACK round-trip identity
2. SHIFT zero-fill behavior
3. POSE coordinate transformations + inverses
4. BITWISE operations
5. PERIOD minimal period detection (KMP)
6. kernel_receipts() generates valid receipts
"""

from src.arcbit.kernel import (
    order_colors,
    pack_grid_to_planes,
    unpack_planes_to_grid,
    shift_plane,
    pose_plane,
    pose_inverse,
    plane_and,
    plane_or,
    plane_andn,
    minimal_period_row,
    kernel_receipts
)
from src.arcbit.core import blake3_hash
from src.arcbit.core.bytesio import serialize_grid_be_row_major, serialize_planes_be_row_major


def test_pack_unpack_roundtrip():
    """Test PACK then UNPACK returns original grid."""
    print("Testing PACK/UNPACK round-trip...")

    # Simple 3x3 grid
    G = [
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ]
    H, W = 3, 3
    colors = [0, 1, 2]

    # Pack
    planes = pack_grid_to_planes(G, H, W, colors)

    # Unpack
    G_reconstructed = unpack_planes_to_grid(planes, H, W, colors)

    assert G == G_reconstructed, "Round-trip failed"
    print("✓ PACK/UNPACK round-trip identity")


def test_pack_consistency_with_serialization():
    """Test that PACK/UNPACK preserves grid content."""
    print("Testing PACK consistency...")

    G = [
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ]
    H, W = 3, 3
    colors = [0, 1, 2]

    # Pack to planes
    planes = pack_grid_to_planes(G, H, W, colors)

    # Verify plane structure
    assert len(planes) == len(colors), "Wrong number of planes"
    for color in colors:
        assert color in planes, f"Missing plane for color {color}"
        assert len(planes[color]) == H, f"Plane for color {color} has wrong height"

    # Unpack back to grid
    G_back = unpack_planes_to_grid(planes, H, W, colors)
    assert G == G_back, "Pack/unpack round-trip failed"

    # Verify planes can be serialized
    planes_bytes = serialize_planes_be_row_major(planes, H, W, colors)
    planes_hash = blake3_hash(planes_bytes)

    print(f"✓ PACK consistency verified (planes_hash: {planes_hash[:16]}...)")


def test_shift_zero_fill():
    """Test SHIFT zero-fill behavior."""
    print("Testing SHIFT zero-fill...")

    # 3x3 plane with single bit at (1, 1)
    plane = [
        0b000,
        0b010,  # bit 1 = col 1
        0b000
    ]
    H, W = 3, 3

    # Shift right by 1 (positive dx)
    plane_r = shift_plane(plane, 0, 1, H, W)
    expected_r = [
        0b000,
        0b100,  # bit moved from col 1 to col 2
        0b000
    ]
    assert plane_r == expected_r, f"Shift right failed: {plane_r} != {expected_r}"

    # Shift left by 1 (negative dx)
    plane_l = shift_plane(plane, 0, -1, H, W)
    expected_l = [
        0b000,
        0b001,  # bit moved from col 1 to col 0
        0b000
    ]
    assert plane_l == expected_l, f"Shift left failed: {plane_l} != {expected_l}"

    # Shift down by 1 (positive dy)
    plane_d = shift_plane(plane, 1, 0, H, W)
    expected_d = [
        0b000,
        0b000,
        0b010  # row moved down
    ]
    assert plane_d == expected_d, f"Shift down failed: {plane_d} != {expected_d}"

    # Shift up by 1 (negative dy)
    plane_u = shift_plane(plane, -1, 0, H, W)
    expected_u = [
        0b010,  # row moved up
        0b000,
        0b000
    ]
    assert plane_u == expected_u, f"Shift up failed: {plane_u} != {expected_u}"

    print("✓ SHIFT zero-fill correct (all 4 directions)")


def test_pose_inverses():
    """Test POSE inverses: pose(pose(plane, pid), inv(pid)) == plane."""
    print("Testing POSE inverses...")

    # 3x2 plane (non-square to test dimension swaps)
    plane = [
        0b10,  # row 0: cols [1]
        0b01,  # row 1: cols [0]
        0b11   # row 2: cols [0, 1]
    ]
    H, W = 3, 2

    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    for pid in pose_ids:
        # Apply pose
        plane_fwd, H_fwd, W_fwd = pose_plane(plane, pid, H, W)

        # Apply inverse
        inv_pid = pose_inverse(pid)
        plane_back, H_back, W_back = pose_plane(plane_fwd, inv_pid, H_fwd, W_fwd)

        # Check round-trip
        assert H_back == H, f"{pid}: Height mismatch after round-trip"
        assert W_back == W, f"{pid}: Width mismatch after round-trip"
        assert plane_back == plane, f"{pid}: Plane mismatch after round-trip"

    print(f"✓ POSE inverses correct (all 8 transforms)")


def test_pose_r90_coordinates():
    """Test POSE R90 exact coordinate mapping."""
    print("Testing POSE R90 coordinates...")

    # 3x2 plane with single bit at (0, 1)
    plane = [
        0b10,  # row 0, col 1 (bit 1 set)
        0b00,
        0b00
    ]
    H, W = 3, 2

    # R90: r=H-1-c', c=r' (pull mapping)
    # Output shape: (W, H) = (2, 3)
    # For output (r', c'), we pull from input (H-1-c', r')
    # Input bit at (0, 1) appears in output where:
    #   H-1-c' = 0 and r' = 1
    #   => c' = H-1 = 2, r' = 1
    # So output bit should be at (1, 2)
    plane_r90, H_out, W_out = pose_plane(plane, "R90", H, W)

    assert H_out == 2 and W_out == 3, f"R90 shape wrong: got ({H_out}, {W_out}), expected (2, 3)"
    expected = [
        0b000,  # row 0: no bits
        0b100   # row 1: bit 2 set (col 2)
    ]
    assert plane_r90 == expected, f"R90 failed: {plane_r90} ({[bin(m) for m in plane_r90]}) != {expected} ({[bin(m) for m in expected]})"

    print("✓ POSE R90 coordinates exact")


def test_bitwise():
    """Test BITWISE operations."""
    print("Testing BITWISE operations...")

    a = [0b101, 0b110]
    b = [0b011, 0b101]
    H, W = 2, 3

    # AND
    result_and = plane_and(a, b, H, W)
    expected_and = [0b001, 0b100]
    assert result_and == expected_and, f"AND failed: {result_and} != {expected_and}"

    # OR
    result_or = plane_or(a, b, H, W)
    expected_or = [0b111, 0b111]
    assert result_or == expected_or, f"OR failed: {result_or} != {expected_or}"

    # ANDN (a & ~b)
    result_andn = plane_andn(a, b, H, W)
    expected_andn = [0b100, 0b010]
    assert result_andn == expected_andn, f"ANDN failed: {result_andn} != {expected_andn}"

    print("✓ BITWISE operations correct")


def test_period_kmp():
    """Test PERIOD minimal period detection."""
    print("Testing PERIOD (KMP)...")

    # Period 2: "101010"
    p = minimal_period_row(0b101010, 6)
    assert p == 2, f"Period 2 failed: got {p}"

    # Period 3: "110110"
    p = minimal_period_row(0b110110, 6)
    assert p == 3, f"Period 3 failed: got {p}"

    # Constant row (all 1s): "111111" → None (period 1 excluded per spec)
    p = minimal_period_row(0b111111, 6)
    assert p is None, f"Constant row (all 1s) should return None: got {p}"

    # Constant row (all 0s): "000000" → None (period 1 excluded per spec)
    p = minimal_period_row(0b000000, 6)
    assert p is None, f"Constant row (all 0s) should return None: got {p}"

    # No period: "101011"
    p = minimal_period_row(0b101011, 6)
    assert p is None, f"No period failed: got {p}"

    # Period 2 with different width: "1010"
    p = minimal_period_row(0b1010, 4)
    assert p == 2, f"Period 2 (W=4) failed: got {p}"

    print("✓ PERIOD (KMP) correct (non-trivial periods only, p >= 2)")


def test_kernel_receipts():
    """Test kernel_receipts() generates valid receipts."""
    print("Testing kernel_receipts()...")

    # Define fixtures
    fixtures = [
        {
            "grid": [[0, 1], [1, 0]],
            "H": 2,
            "W": 2,
            "colors": [0, 1],
            "label": "checkerboard_2x2"
        },
        {
            "grid": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            "H": 3,
            "W": 3,
            "colors": [0, 1, 2],
            "label": "latin_square_3x3"
        }
    ]

    # Generate receipts
    digest = kernel_receipts("WO-01-test", fixtures)

    # Verify structure
    assert digest["section"] == "WO-01-test"
    assert digest["spec_version"] == "1.5"
    assert "kernel.params_hash" in digest["payload"]
    assert "pack_consistency" in digest["payload"]
    assert "pose_inverse_ok" in digest["payload"]
    assert "shift_boundary_counts" in digest["payload"]
    assert "period_kmp_examples" in digest["payload"]

    # Verify pose inverse tests all passed
    assert digest["payload"]["pose_inverse_ok"] == True

    # Verify pack consistency
    for item in digest["payload"]["pack_consistency"]:
        assert item["roundtrip_ok"] == True, f"Pack consistency failed for {item['label']}"

    print(f"✓ kernel_receipts() valid (section_hash: {digest['section_hash'][:16]}...)")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    # Empty grid (H=0)
    G_empty = []
    planes_empty = pack_grid_to_planes(G_empty, 0, 0, [0])
    assert planes_empty[0] == [], "Empty grid failed"

    # Single pixel
    G_single = [[5]]
    planes_single = pack_grid_to_planes(G_single, 1, 1, [0, 5])
    G_back = unpack_planes_to_grid(planes_single, 1, 1, [0, 5])
    assert G_back == G_single, "Single pixel round-trip failed"

    # SHIFT on empty plane
    plane_empty = []
    shifted_empty = shift_plane(plane_empty, 1, 1, 0, 0)
    assert shifted_empty == [], "SHIFT on empty failed"

    print("✓ Edge cases handled")


if __name__ == "__main__":
    print("=" * 60)
    print("WO-01 Verification Tests")
    print("=" * 60)
    print()

    test_pack_unpack_roundtrip()
    test_pack_consistency_with_serialization()
    test_shift_zero_fill()
    test_pose_inverses()
    test_pose_r90_coordinates()
    test_bitwise()
    test_period_kmp()
    test_kernel_receipts()
    test_edge_cases()

    print()
    print("=" * 60)
    print("✅ All WO-01 tests passed!")
    print("=" * 60)
    print()
    print("WO-01 Implementation Complete:")
    print("  - PACK/UNPACK with mutual exclusivity")
    print("  - SHIFT with zero-fill (4 directions)")
    print("  - POSE with 8 D4 transforms + inverses")
    print("  - BITWISE (AND/OR/ANDN)")
    print("  - PERIOD with KMP (minimal period detection)")
    print("  - kernel_receipts() with algebraic debugging")
    print()
    print("All operations pure, deterministic, exact.")
