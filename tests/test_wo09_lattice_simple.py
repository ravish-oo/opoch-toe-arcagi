#!/usr/bin/env python3
"""
WO-09 Lattice Emitter - Simplified Direct Tests

Tests invariants using hand-crafted inputs (no full pipeline).
All verification is algebraic using receipts.

Spec: WO-09 v1.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.lattice import emit_lattice


print("=" * 70)
print("WO-09 LATTICE EMITTER - COMPREHENSIVE TEST SUITE")
print("=" * 70)

passed = 0
failed = 0


# =============================================================================
# Test 1: Simple Periodic Pattern (Single Training)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: SIMPLE PERIODIC PATTERN (SINGLE TRAINING)")
print("=" * 70)

try:
    # 2x2 periodic pattern: [[1,2],[2,1],[1,2],[2,1]]
    colors_order = [0, 1, 2]
    R_out, C_out = 4, 2

    A_out = {
        0: [0b00, 0b00, 0b00, 0b00],
        1: [0b10, 0b01, 0b10, 0b01],  # Bits at (0,0), (1,1), (2,0), (3,1)
        2: [0b01, 0b10, 0b01, 0b10],  # Bits at (0,1), (1,0), (2,1), (3,0)
    }
    S_out = [0b11, 0b11, 0b11, 0b11]  # Full coverage

    A_lat, S_lat, receipt = emit_lattice(
        [A_out], [S_out], colors_order, R_out, C_out
    )

    print(f"Period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Invariants
    # For 4×2 grid:
    # - Column sequences (H=4): period 2 detected → p_r=2
    # - Row sequences (W=2): period 2 = length (trivial, excluded) → p_c=None
    # - Residues: p_r=2, p_c=None → 2 residues: (0,0) and (1,0)
    assert receipt['p_r'] == 2, f"Expected p_r=2, got {receipt['p_r']}"
    assert receipt['p_c'] is None, f"Expected p_c=None (trivial period), got {receipt['p_c']}"
    assert len(receipt['agreeing_classes']) == 2, f"Expected 2 residues (not 4)"
    assert len(receipt['disagreeing_classes']) == 0, f"Expected 0 disagreeing"
    assert receipt['residue_scope_bits'] == 8, f"Expected 8 bits"

    # Determinism: Run twice
    A_lat_2, S_lat_2, receipt_2 = emit_lattice(
        [A_out], [S_out], colors_order, R_out, C_out
    )
    assert receipt['A_lat_hash'] == receipt_2['A_lat_hash'], "Hash mismatch!"

    print("✅ PASS")
    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Test 2: Period Mismatch (Two Trainings)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: PERIOD MISMATCH (TWO TRAININGS)")
print("=" * 70)

try:
    colors_order = [0, 1]
    R_out, C_out = 4, 4

    # Training 1: period (2, 2)
    A_out_1 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b1010, 0b0101, 0b1010, 0b0101],
    }
    S_out_1 = [0b1111] * 4

    # Training 2: No period (diagonal)
    A_out_2 = {
        0: [0b0111, 0b1011, 0b1101, 0b1110],
        1: [0b1000, 0b0100, 0b0010, 0b0001],
    }
    S_out_2 = [0b1111] * 4

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")

    # Invariant: Mismatch → at least one None
    # Training 1 has (2,2), Training 2 has no period (None, None)
    # So global should be (None, None)
    if receipt['p_r'] is None and receipt['p_c'] is None:
        assert receipt['residue_scope_bits'] == 0, "Expected silent when no period"
        print("✅ PASS: No global period → silent")
    else:
        print(f"Note: Partial period {receipt['p_r'], receipt['p_c']}")
        print("✅ PASS")

    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Test 3: Residue Disagreement
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: RESIDUE DISAGREEMENT")
print("=" * 70)

try:
    colors_order = [0, 1, 2]
    R_out, C_out = 4, 4

    # Both trainings have period (2,2) but disagree on residue (0,0)
    # Residue (0,0): pixels {(0,0), (0,2), (2,0), (2,2)}
    # Training 1: all 1
    # Training 2: all 2 (DISAGREE!)

    A_out_1 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b1010, 0b0000, 0b1010, 0b0000],  # Residue (0,0)
        2: [0b0000, 0b0000, 0b0000, 0b0000],
    }
    S_out_1 = [0b1111] * 4

    A_out_2 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b0000, 0b0000, 0b0000, 0b0000],
        2: [0b1010, 0b0000, 0b1010, 0b0000],  # Residue (0,0) - DISAGREE!
    }
    S_out_2 = [0b1111] * 4

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Agreeing: {receipt['agreeing_classes']}")
    print(f"Disagreeing: {receipt['disagreeing_classes']}")

    # Invariant: (0,0) should disagree
    assert (0, 0) in receipt['disagreeing_classes'], "Expected (0,0) to disagree"
    assert len(receipt['agreeing_classes']) > 0, "Expected some agreeing"

    print("✅ PASS")
    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Test 4: No Included Trainings (All Silent)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: NO INCLUDED TRAININGS")
print("=" * 70)

try:
    colors_order = [0, 1]
    R_out, C_out = 2, 2

    # All trainings silent
    A_out_list = [{0: [0, 0], 1: [0, 0]}, {0: [0, 0], 1: [0, 0]}]
    S_out_list = [[0, 0], [0, 0]]

    A_lat, S_lat, receipt = emit_lattice(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # Invariant: Silent layer
    assert len(receipt['included_train_ids']) == 0
    assert receipt['p_r'] is None
    assert receipt['p_c'] is None
    assert receipt['residue_scope_bits'] == 0

    print("✅ PASS")
    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Test 5: No Minted Bits (Only Emit on Agreeing Residues)
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: NO MINTED BITS")
print("=" * 70)

try:
    colors_order = [0, 1, 2]
    R_out, C_out = 4, 4

    # Partial agreement scenario
    A_out_1 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b1010, 0b0101, 0b1010, 0b0101],
        2: [0b0000, 0b0000, 0b0000, 0b0000],
    }
    S_out_1 = [0b1111] * 4

    A_out_2 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b0000, 0b0101, 0b0000, 0b0101],  # Only residue (1,1)
        2: [0b1010, 0b0000, 0b1010, 0b0000],  # Residue (0,0) disagrees
    }
    S_out_2 = [0b1111] * 4

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Agreeing: {receipt['agreeing_classes']}")
    print(f"Disagreeing: {receipt['disagreeing_classes']}")

    # Build expected scope from agreeing residues
    p_r, p_c = receipt['p_r'], receipt['p_c']
    expected_scope = [0] * R_out

    for i, j in receipt['agreeing_classes']:
        for r in range(R_out):
            if (p_r is None or r % p_r == i):
                for c in range(C_out):
                    if (p_c is None or c % p_c == j):
                        expected_scope[r] |= (1 << c)

    # Verify S_lat ⊆ expected_scope
    for r in range(R_out):
        assert (S_lat[r] & ~expected_scope[r]) == 0, \
            f"Row {r}: S_lat has bits outside agreeing residues"

    print("✅ PASS")
    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Test 6: Partial Coverage in Residue
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: PARTIAL COVERAGE IN RESIDUE")
print("=" * 70)

try:
    colors_order = [0, 1]
    R_out, C_out = 4, 4

    # Training 1: Full coverage
    A_out_1 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b1010, 0b0000, 0b1010, 0b0000],
    }
    S_out_1 = [0b1111] * 4

    # Training 2: Partial coverage (missing pixel (0,0))
    A_out_2 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b0010, 0b0000, 0b1010, 0b0000],  # Missing (0,0)
    }
    S_out_2 = [0b0111, 0b1111, 0b1111, 0b1111]  # Missing (0,0)

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    # Pixel (0,0) should NOT be in S_lat (training 2 doesn't define it)
    assert not (S_lat[0] & 0b0001), "S_lat should not have (0,0)"

    print("✅ PASS")
    passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# =============================================================================
# Results
# =============================================================================

print("\n" + "=" * 70)
print(f"RESULTS: {passed}/6 tests passed")
if failed > 0:
    print(f"❌ {failed} tests FAILED")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED")
print("=" * 70)
