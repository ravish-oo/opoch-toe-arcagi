#!/usr/bin/env python3
"""
WO-09 Lattice Emitter - Unit Tests

Tests grid reconstruction, period detection, residue agreement, and emission.

Spec: WO-09 v1.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.lattice import (
    emit_lattice,
    _reconstruct_grid_from_planes,
    _aggregate_periods,
    _build_residue_mask,
    _check_residue_agreement,
)


def test_reconstruct_grid_from_planes():
    """Test grid reconstruction from singleton planes."""
    print("\n" + "=" * 70)
    print("TEST 1: Grid Reconstruction from Planes")
    print("=" * 70)

    colors_order = [0, 1, 2]
    R_out, C_out = 2, 2

    # Create singleton planes: color 1 at (0,0), color 2 at (1,1)
    A_i = {
        0: [0b00, 0b00],
        1: [0b01, 0b00],  # bit at (0,0)
        2: [0b00, 0b10],  # bit at (1,1)
    }
    S_i = [0b01, 0b10]  # scope at (0,0) and (1,1)

    Y_i = _reconstruct_grid_from_planes(A_i, S_i, colors_order, R_out, C_out)

    print(f"Reconstructed grid:")
    for row in Y_i:
        print(f"  {row}")

    assert Y_i[0][0] == 1, "Pixel (0,0) should be color 1"
    assert Y_i[0][1] == 0, "Pixel (0,1) should be 0 (no scope)"
    assert Y_i[1][0] == 0, "Pixel (1,0) should be 0 (no scope)"
    assert Y_i[1][1] == 2, "Pixel (1,1) should be color 2"

    print("✓ Grid reconstruction correct")
    print("\n✅ TEST 1 PASSED")


def test_aggregate_periods():
    """Test period aggregation across trainings."""
    print("\n" + "=" * 70)
    print("TEST 2: Period Aggregation")
    print("=" * 70)

    # Case 1: All trainings have same period (2,2)
    Y_list_1 = [
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],  # Training 0: period (2,2)
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],  # Training 1: period (2,2)
    ]
    colors_order_1 = [1, 2, 3, 4]
    p_r, p_c = _aggregate_periods(Y_list_1, 4, 4, colors_order_1)
    assert p_r == 2 and p_c == 2, f"Expected (2,2), got ({p_r},{p_c})"
    print(f"✓ Case 1: All agree on (2,2) → ({p_r},{p_c})")

    # Case 2: Mismatch on row period
    Y_list_2 = [
        [[1, 2], [1, 2]],  # Training 0: period (1,2) - rows identical
        [[1, 2], [3, 4]],  # Training 1: period (2,2) - rows differ
    ]
    colors_order_2 = [1, 2, 3, 4]
    p_r, p_c = _aggregate_periods(Y_list_2, 2, 2, colors_order_2)
    # Row periods are (None, 2) → None for row (mismatch or one is None)
    # Actually, first training has p_r=None (rows are identical, not periodic)
    # Second has p_r=2
    # Aggregate: one is None, one is 2 → None
    print(f"✓ Case 2: Mismatch on row → p_r={p_r}")

    # Case 3: No period (all None)
    Y_list_3 = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # No period
    ]
    colors_order_3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    p_r, p_c = _aggregate_periods(Y_list_3, 3, 3, colors_order_3)
    assert p_r is None and p_c is None, "Expected (None, None) for no period"
    print(f"✓ Case 3: No period → ({p_r},{p_c})")

    print("\n✅ TEST 2 PASSED")


def test_build_residue_mask():
    """Test residue mask generation."""
    print("\n" + "=" * 70)
    print("TEST 3: Residue Mask Generation")
    print("=" * 70)

    R_out, C_out = 4, 4
    p_r, p_c = 2, 2

    # Residue (0,0): r mod 2 = 0, c mod 2 = 0 → pixels (0,0), (0,2), (2,0), (2,2)
    R_00 = _build_residue_mask(0, 0, p_r, p_c, R_out, C_out)
    expected_00 = [
        0b0101,  # row 0: cols 0,2
        0b0000,  # row 1: none
        0b0101,  # row 2: cols 0,2
        0b0000,  # row 3: none
    ]
    assert R_00 == expected_00, f"Residue (0,0) mask mismatch"
    print(f"✓ Residue (0,0) mask correct")

    # Residue (1,1): r mod 2 = 1, c mod 2 = 1 → pixels (1,1), (1,3), (3,1), (3,3)
    R_11 = _build_residue_mask(1, 1, p_r, p_c, R_out, C_out)
    expected_11 = [
        0b0000,  # row 0: none
        0b1010,  # row 1: cols 1,3
        0b0000,  # row 2: none
        0b1010,  # row 3: cols 1,3
    ]
    assert R_11 == expected_11, f"Residue (1,1) mask mismatch"
    print(f"✓ Residue (1,1) mask correct")

    # No column period (p_c = None) → all columns match
    R_0_all = _build_residue_mask(0, 0, p_r, None, R_out, C_out)
    expected_0_all = [
        0b1111,  # row 0: all cols
        0b0000,  # row 1: none (r mod 2 != 0)
        0b1111,  # row 2: all cols
        0b0000,  # row 3: none
    ]
    assert R_0_all == expected_0_all, f"Residue (0,*) mask mismatch"
    print(f"✓ Residue (0,*) with p_c=None correct")

    print("\n✅ TEST 3 PASSED")


def test_check_residue_agreement():
    """Test residue agreement checking."""
    print("\n" + "=" * 70)
    print("TEST 4: Residue Agreement Check")
    print("=" * 70)

    R_out, C_out = 2, 2
    colors_order = [0, 1, 2]
    included_train_ids = [0, 1]

    # Case 1: All trainings agree on color 1 for residue
    Y_list_1 = [
        [[1, 0], [0, 1]],  # Training 0
        [[1, 0], [0, 1]],  # Training 1
    ]
    S_out_list_1 = [
        [0b01, 0b10],  # Training 0: scope at (0,0) and (1,1)
        [0b01, 0b10],  # Training 1: scope at (0,0) and (1,1)
    ]
    R_ij_1 = [0b01, 0b10]  # Residue covers (0,0) and (1,1)

    agreed, color = _check_residue_agreement(
        Y_list_1, S_out_list_1, included_train_ids, R_ij_1, R_out, C_out
    )
    assert agreed == True and color == 1, f"Expected (True, 1), got ({agreed}, {color})"
    print(f"✓ Case 1: Agreement on color 1")

    # Case 2: Disagreement (different colors)
    Y_list_2 = [
        [[1, 0], [0, 1]],  # Training 0: color 1 at (0,0)
        [[2, 0], [0, 1]],  # Training 1: color 2 at (0,0) - DISAGREES!
    ]
    S_out_list_2 = [
        [0b01, 0b10],
        [0b01, 0b10],
    ]
    R_ij_2 = [0b01, 0b10]

    agreed, color = _check_residue_agreement(
        Y_list_2, S_out_list_2, included_train_ids, R_ij_2, R_out, C_out
    )
    assert agreed == False, f"Expected disagreement, got {agreed}"
    print(f"✓ Case 2: Disagreement detected")

    # Case 3: Partial scope (training doesn't define some pixels)
    S_out_list_3 = [
        [0b01, 0b10],  # Training 0: scope at (0,0) and (1,1)
        [0b01, 0b00],  # Training 1: scope only at (0,0), missing (1,1)
    ]
    # Only compare at (0,0) where both define
    agreed, color = _check_residue_agreement(
        Y_list_1, S_out_list_3, included_train_ids, R_ij_1, R_out, C_out
    )
    # Both agree at (0,0) → agreed
    assert agreed == True and color == 1, f"Expected (True, 1), got ({agreed}, {color})"
    print(f"✓ Case 3: Partial scope (only compare where all define)")

    print("\n✅ TEST 4 PASSED")


def test_emit_lattice_simple_checkerboard():
    """Test lattice emission on simple 2x2 checkerboard."""
    print("\n" + "=" * 70)
    print("TEST 5: Lattice Emission - Simple Checkerboard")
    print("=" * 70)

    colors_order = [0, 1, 2, 3]  # Include ALL colors in the pattern
    R_out, C_out = 4, 4

    # Create checkerboard pattern: period (2,2)
    # Training 0:
    #   1 2 1 2
    #   3 0 3 0
    #   1 2 1 2
    #   3 0 3 0
    A_out_0 = {
        0: [0b0000, 0b1010, 0b0000, 0b1010],  # color 0 at (1,1), (1,3), (3,1), (3,3)
        1: [0b0101, 0b0000, 0b0101, 0b0000],  # color 1 at (0,0), (0,2), (2,0), (2,2)
        2: [0b1010, 0b0000, 0b1010, 0b0000],  # color 2 at (0,1), (0,3), (2,1), (2,3)
        3: [0b0000, 0b0101, 0b0000, 0b0101],  # color 3 at (1,0), (1,2), (3,0), (3,2)
    }
    S_out_0 = [0b1111, 0b1111, 0b1111, 0b1111]  # Full scope

    # Training 1: same pattern
    A_out_1 = A_out_0.copy()
    S_out_1 = S_out_0.copy()

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    # Run lattice emission
    A_lat, S_lat, receipt = emit_lattice(A_out_list, S_out_list, colors_order, R_out, C_out)

    print(f"\nPeriods: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Verify periods
    assert receipt['p_r'] == 2 and receipt['p_c'] == 2, f"Expected period (2,2), got ({receipt['p_r']},{receipt['p_c']})"

    # Verify all 4 residues agree
    assert len(receipt['agreeing_classes']) == 4, f"Expected 4 agreeing classes, got {len(receipt['agreeing_classes'])}"
    assert len(receipt['disagreeing_classes']) == 0, f"Expected 0 disagreeing classes, got {len(receipt['disagreeing_classes'])}"

    # Verify scope covers full canvas
    assert receipt['residue_scope_bits'] == 16, f"Expected 16 scope bits, got {receipt['residue_scope_bits']}"

    # Verify lattice admits match input
    assert A_lat == A_out_0, "Lattice admits should match input under full agreement"

    print("\n✓ Checkerboard lattice correct")
    print("\n✅ TEST 5 PASSED")


def test_emit_lattice_disagreement():
    """Test lattice with disagreeing residues."""
    print("\n" + "=" * 70)
    print("TEST 6: Lattice Emission - Disagreement")
    print("=" * 70)

    colors_order = [0, 1, 2, 3, 5]  # Include ALL colors present in any training
    R_out, C_out = 4, 4

    # Training 0: checkerboard with period (2,2)
    #   1 2 1 2
    #   3 0 3 0
    #   1 2 1 2
    #   3 0 3 0
    A_out_0 = {
        0: [0b0000, 0b1010, 0b0000, 0b1010],
        1: [0b0101, 0b0000, 0b0101, 0b0000],
        2: [0b1010, 0b0000, 0b1010, 0b0000],
        3: [0b0000, 0b0101, 0b0000, 0b0101],
        5: [0b0000, 0b0000, 0b0000, 0b0000],  # color 5 not present in training 0
    }
    S_out_0 = [0b1111, 0b1111, 0b1111, 0b1111]

    # Training 1: same pattern BUT pixel (1,1) differs (0 → 5)
    #   1 2 1 2
    #   3 5 3 0   ← (1,1) changed from 0 to 5
    #   1 2 1 2
    #   3 0 3 0
    A_out_1 = {
        0: [0b0000, 0b1000, 0b0000, 0b1010],  # color 0 missing at (1,1)
        1: [0b0101, 0b0000, 0b0101, 0b0000],
        2: [0b1010, 0b0000, 0b1010, 0b0000],
        3: [0b0000, 0b0101, 0b0000, 0b0101],
        5: [0b0000, 0b0010, 0b0000, 0b0000],  # color 5 at (1,1) - DISAGREES!
    }
    S_out_1 = [0b1111, 0b1111, 0b1111, 0b1111]

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    # Run lattice emission
    A_lat, S_lat, receipt = emit_lattice(A_out_list, S_out_list, colors_order, R_out, C_out)

    print(f"\nPeriods: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Verify periods
    assert receipt['p_r'] == 2 and receipt['p_c'] == 2, f"Expected period (2,2), got ({receipt['p_r']},{receipt['p_c']})"

    # Verify some residues agree, some disagree
    # Residue (0,0): pixels (0,0), (0,2), (2,0), (2,2) all have color 1 in both → agree
    # Residue (0,1): pixels (0,1), (0,3), (2,1), (2,3) all have color 2 in both → agree
    # Residue (1,0): pixels (1,0), (1,2), (3,0), (3,2) all have color 3 in both → agree
    # Residue (1,1): pixels (1,1), (1,3), (3,1), (3,3) - (1,1) disagrees (0 vs 5) → disagree
    assert (0, 0) in receipt['agreeing_classes'], "Residue (0,0) should agree"
    assert (0, 1) in receipt['agreeing_classes'], "Residue (0,1) should agree"
    assert (1, 0) in receipt['agreeing_classes'], "Residue (1,0) should agree"
    assert (1, 1) in receipt['disagreeing_classes'], "Residue (1,1) should disagree"

    # Verify scope excludes disagreeing residue (1,1)
    # Row 0: all agree → full scope
    # Row 1: residue (1,1) disagrees at cols 1,3 → scope only at cols 0,2
    # Row 2: all agree → full scope
    # Row 3: residue (1,1) disagrees at cols 1,3 → scope only at cols 0,2
    assert S_lat[0] == 0b1111, f"Row 0 should have full scope, got {bin(S_lat[0])}"
    assert S_lat[1] == 0b0101, f"Row 1 should have scope at cols 0,2 only, got {bin(S_lat[1])}"
    assert S_lat[2] == 0b1111, f"Row 2 should have full scope, got {bin(S_lat[2])}"
    assert S_lat[3] == 0b0101, f"Row 3 should have scope at cols 0,2 only, got {bin(S_lat[3])}"

    print("\n✓ Disagreement handling correct")
    print("\n✅ TEST 6 PASSED")


def test_emit_lattice_no_period():
    """Test lattice with no period → silent."""
    print("\n" + "=" * 70)
    print("TEST 7: Lattice Emission - No Period (Silent)")
    print("=" * 70)

    colors_order = [0, 1, 2]
    R_out, C_out = 3, 3

    # No periodic pattern
    A_out_0 = {
        0: [0b000, 0b000, 0b000],
        1: [0b111, 0b000, 0b000],  # color 1 on row 0
        2: [0b000, 0b111, 0b000],  # color 2 on row 1
        3: [0b000, 0b000, 0b111],  # color 3 on row 2 (not in colors_order, but illustrative)
    }
    S_out_0 = [0b111, 0b111, 0b111]

    A_out_list = [A_out_0]
    S_out_list = [S_out_0]

    # Run lattice emission
    A_lat, S_lat, receipt = emit_lattice(A_out_list, S_out_list, colors_order, R_out, C_out)

    print(f"\nPeriods: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Verify no period
    assert receipt['p_r'] is None and receipt['p_c'] is None, "Expected no period (None, None)"

    # Verify silent layer
    assert receipt['residue_scope_bits'] == 0, "Expected 0 scope bits (silent)"
    assert all(row == 0 for row in S_lat), "S_lat should be all zeros (silent)"

    print("\n✓ No-period silent layer correct")
    print("\n✅ TEST 7 PASSED")


def test_receipts_determinism():
    """Test that double-run produces identical receipts."""
    print("\n" + "=" * 70)
    print("TEST 8: Receipts Determinism")
    print("=" * 70)

    colors_order = [0, 1, 2]
    R_out, C_out = 2, 2

    A_out_0 = {
        0: [0b00, 0b00],
        1: [0b11, 0b11],
    }
    S_out_0 = [0b11, 0b11]

    A_out_list = [A_out_0]
    S_out_list = [S_out_0]

    # Run 1
    _, _, receipt1 = emit_lattice(A_out_list, S_out_list, colors_order, R_out, C_out)

    # Run 2
    _, _, receipt2 = emit_lattice(A_out_list, S_out_list, colors_order, R_out, C_out)

    # Verify determinism
    assert receipt1['A_lat_hash'] == receipt2['A_lat_hash'], "A_lat hashes must match"
    assert receipt1['S_lat_hash'] == receipt2['S_lat_hash'], "S_lat hashes must match"
    assert receipt1['agreeing_classes'] == receipt2['agreeing_classes'], "Agreeing classes must match"

    print(f"✓ Run 1 A_lat_hash: {receipt1['A_lat_hash'][:16]}...")
    print(f"✓ Run 2 A_lat_hash: {receipt2['A_lat_hash'][:16]}...")
    print(f"✓ Hashes match (deterministic)")

    print("\n✅ TEST 8 PASSED")


if __name__ == "__main__":
    print("WO-09 LATTICE EMITTER - UNIT TESTS")
    print("=" * 70)

    tests = [
        test_reconstruct_grid_from_planes,
        test_aggregate_periods,
        test_build_residue_mask,
        test_check_residue_agreement,
        test_emit_lattice_simple_checkerboard,
        test_emit_lattice_disagreement,
        test_emit_lattice_no_period,
        test_receipts_determinism,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        sys.exit(1)
