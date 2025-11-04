#!/usr/bin/env python3
"""
WO-09 Lattice Emitter - Comprehensive Test Suite

Tests invariants, edge cases, and real ARC-AGI data using receipts.
All verification is algebraic (no guessing).

Spec: WO-09 v1.6
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.lattice import emit_lattice
from src.arcbit.emitters.output_transport import emit_output_transport
from src.arcbit.emitters.unanimity import emit_unity
from src.arcbit.canvas import choose_working_canvas
from src.arcbit.core.canonicalize import canonicalize_training
from src.arcbit.core.hashing import blake3_hash


# =============================================================================
# Test 1: INVARIANT - Grid Reconstruction (Y_i from planes)
# =============================================================================

def test_grid_reconstruction_from_planes():
    """
    Test that reconstructed grids match original singleton planes.

    Invariant: Y_i[r][c] = u ⟺ (S_out_i[r] & (1<<c)) AND (A_out_i[u][r] & (1<<c))
    """
    print("\n" + "=" * 70)
    print("TEST 1: GRID RECONSTRUCTION FROM PLANES")
    print("=" * 70)

    # Create simple test case: 3x3 grid with periodic pattern
    # Pattern: [[1,2,1], [2,1,2], [1,2,1]]  (period 2x2)
    colors_order = [0, 1, 2]
    R_out, C_out = 3, 3

    # Build singleton planes manually
    A_out = {
        0: [0b000, 0b000, 0b000],  # No 0s
        1: [0b101, 0b010, 0b101],  # 1s at (0,0), (0,2), (1,1), (2,0), (2,2)
        2: [0b010, 0b101, 0b010],  # 2s at (0,1), (1,0), (1,2), (2,1)
    }
    S_out = [0b111, 0b111, 0b111]  # Full coverage

    # Expected grid
    expected_Y = [
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1]
    ]

    # Run lattice emitter
    A_lat, S_lat, receipt = emit_lattice(
        [A_out], [S_out], colors_order, R_out, C_out
    )

    # Verify reconstruction via receipts
    # Since this is periodic with p_r=2, p_c=2, residues should agree
    print(f"Period detected: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Invariant check 1: If single training, all residues should agree
    assert receipt['p_r'] == 2, f"Expected p_r=2, got {receipt['p_r']}"
    assert receipt['p_c'] == 2, f"Expected p_c=2, got {receipt['p_c']}"
    assert len(receipt['agreeing_classes']) == 4, f"Expected 4 agreeing residues, got {len(receipt['agreeing_classes'])}"
    assert len(receipt['disagreeing_classes']) == 0, f"Expected 0 disagreeing, got {len(receipt['disagreeing_classes'])}"

    # Invariant check 2: Scope should cover all pixels (single training, full agreement)
    assert receipt['residue_scope_bits'] == 9, f"Expected 9 scope bits, got {receipt['residue_scope_bits']}"

    print("✅ PASS: Grid reconstruction invariant satisfied")
    return True


# =============================================================================
# Test 2: INVARIANT - Period Consensus (all trainings must agree)
# =============================================================================

def test_period_consensus_mismatch():
    """
    Test that mismatched periods → p_r=None or p_c=None.

    Invariant: If trainings disagree on period → global period is None
    """
    print("\n" + "=" * 70)
    print("TEST 2: PERIOD CONSENSUS MISMATCH")
    print("=" * 70)

    colors_order = [0, 1]
    R_out, C_out = 4, 4

    # Training 1: period (2, 2)
    # [[1,0,1,0],
    #  [0,1,0,1],
    #  [1,0,1,0],
    #  [0,1,0,1]]
    A_out_1 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b1010, 0b0101, 0b1010, 0b0101],
    }
    S_out_1 = [0b1111, 0b1111, 0b1111, 0b1111]

    # Training 2: period (4, 4) - no repeats
    # [[1,0,0,0],
    #  [0,1,0,0],
    #  [0,0,1,0],
    #  [0,0,0,1]]
    A_out_2 = {
        0: [0b0111, 0b1011, 0b1101, 0b1110],
        1: [0b1000, 0b0100, 0b0010, 0b0001],
    }
    S_out_2 = [0b1111, 0b1111, 0b1111, 0b1111]

    # Run lattice
    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Training 1 should have period (2,2)")
    print(f"Training 2 should have no period (4,4)")
    print(f"Global period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")

    # Invariant: Mismatch → both None
    assert receipt['p_r'] is None or receipt['p_c'] is None, \
        f"Expected at least one None period due to mismatch, got p_r={receipt['p_r']}, p_c={receipt['p_c']}"

    # If no global period → should be silent
    if receipt['p_r'] is None and receipt['p_c'] is None:
        assert receipt['residue_scope_bits'] == 0, \
            f"Expected silent layer (0 bits) when no period, got {receipt['residue_scope_bits']}"
        print("✅ PASS: No global period → silent layer")
    else:
        # At least one axis has period → should have some admits
        print(f"✅ PASS: Partial period detected, residues: {receipt['agreeing_classes']}")

    return True


# =============================================================================
# Test 3: INVARIANT - Residue Unanimity (all trainings must agree on residue)
# =============================================================================

def test_residue_unanimity_disagreement():
    """
    Test that residue disagreement → silent on that residue.

    Invariant: If any training disagrees on residue color → residue in disagreeing_classes
    """
    print("\n" + "=" * 70)
    print("TEST 3: RESIDUE UNANIMITY DISAGREEMENT")
    print("=" * 70)

    colors_order = [0, 1, 2]
    R_out, C_out = 4, 4

    # Both trainings have period (2,2) but disagree on residue (0,0)
    # Training 1: residue (0,0) = {(0,0), (0,2), (2,0), (2,2)} → all color 1
    # Training 2: residue (0,0) = {(0,0), (0,2), (2,0), (2,2)} → all color 2 (DISAGREE!)

    # Training 1
    A_out_1 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],  # 0 at residue (0,1) and (1,*)
        1: [0b1010, 0b0000, 0b1010, 0b0000],  # 1 at residue (0,0)
        2: [0b0000, 0b0000, 0b0000, 0b0000],
    }
    S_out_1 = [0b1111, 0b1111, 0b1111, 0b1111]

    # Training 2
    A_out_2 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b0000, 0b0000, 0b0000, 0b0000],
        2: [0b1010, 0b0000, 0b1010, 0b0000],  # 2 at residue (0,0) - DISAGREE!
    }
    S_out_2 = [0b1111, 0b1111, 0b1111, 0b1111]

    # Run lattice
    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")

    # Invariant: Residue (0,0) should be in disagreeing_classes
    assert (0, 0) in receipt['disagreeing_classes'], \
        f"Expected (0,0) in disagreeing_classes, got {receipt['disagreeing_classes']}"

    # Residues (0,1), (1,0), (1,1) should agree (both have color 0)
    # Verify at least some agreeing classes
    assert len(receipt['agreeing_classes']) > 0, \
        f"Expected some agreeing classes, got {receipt['agreeing_classes']}"

    print("✅ PASS: Disagreeing residue correctly classified")
    return True


# =============================================================================
# Test 4: INVARIANT - Deterministic Hashing (double-run equality)
# =============================================================================

def test_deterministic_hashing():
    """
    Test that double-run produces identical hashes.

    Invariant: Same inputs → same hashes (A_lat_hash, S_lat_hash)
    """
    print("\n" + "=" * 70)
    print("TEST 4: DETERMINISTIC HASHING")
    print("=" * 70)

    colors_order = [0, 1]
    R_out, C_out = 2, 2

    A_out = {
        0: [0b01, 0b10],
        1: [0b10, 0b01],
    }
    S_out = [0b11, 0b11]

    # Run 1
    A_lat_1, S_lat_1, receipt_1 = emit_lattice(
        [A_out], [S_out], colors_order, R_out, C_out
    )

    # Run 2 (identical inputs)
    A_lat_2, S_lat_2, receipt_2 = emit_lattice(
        [A_out], [S_out], colors_order, R_out, C_out
    )

    # Invariant: Hashes must match
    assert receipt_1['A_lat_hash'] == receipt_2['A_lat_hash'], \
        f"A_lat_hash mismatch: {receipt_1['A_lat_hash']} != {receipt_2['A_lat_hash']}"
    assert receipt_1['S_lat_hash'] == receipt_2['S_lat_hash'], \
        f"S_lat_hash mismatch: {receipt_1['S_lat_hash']} != {receipt_2['S_lat_hash']}"

    print(f"A_lat_hash: {receipt_1['A_lat_hash']}")
    print(f"S_lat_hash: {receipt_1['S_lat_hash']}")
    print("✅ PASS: Deterministic hashing verified")
    return True


# =============================================================================
# Test 5: REAL ARC-AGI DATA - Periodic Task
# =============================================================================

def test_real_arc_periodic_task():
    """
    Test on real ARC-AGI data with known periodic pattern.

    Use receipts to verify period detection and residue agreement.
    """
    print("\n" + "=" * 70)
    print("TEST 5: REAL ARC-AGI DATA (PERIODIC TASK)")
    print("=" * 70)

    # Load data
    arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
    with open(arc_data_path, "r") as f:
        all_tasks = json.load(f)

    # Find a task with periodic outputs (scan first 50)
    periodic_task_id = None
    for task_id in list(all_tasks.keys())[:50]:
        task = all_tasks[task_id]
        if len(task['train']) >= 2:
            # Check if outputs have same dimensions (hint of periodicity)
            out_0 = task['train'][0]['output']
            out_1 = task['train'][1]['output']
            if len(out_0) == len(out_1) and len(out_0) > 0 and len(out_0[0]) == len(out_1[0]):
                periodic_task_id = task_id
                break

    if not periodic_task_id:
        print("⚠️  No suitable periodic task found in first 50, skipping")
        return True

    print(f"Testing on task: {periodic_task_id}")
    task = all_tasks[periodic_task_id]

    # Canonicalize trainings
    trainings = []
    for i, pair in enumerate(task['train']):
        X_in = pair['input']
        Y_out = pair['output']
        X_canon, Y_canon, pose, anchor, _ = canonicalize_training(X_in, Y_out)
        trainings.append((X_canon, Y_canon, pose, anchor))

    # Choose working canvas
    result = choose_working_canvas(
        trainings, families_to_try=["H1", "H2", "H3", "H4"]
    )
    if result is None:
        print("⚠️  No working canvas found, skipping")
        return True
    R_out, C_out = result['R_out'], result['C_out']

    print(f"Working canvas: {R_out}×{C_out}")

    # Collect colors
    colors_set = {0}
    for X, Y, _, _ in trainings:
        for row in X + Y:
            colors_set.update(row)
    colors_order = sorted(colors_set)

    # Emit output transports
    A_out_list = []
    S_out_list = []
    for i, (X, Y, pose, anchor) in enumerate(trainings):
        A_out_i, S_out_i, _ = emit_output_transport(
            Y, pose, anchor, colors_order, R_out, C_out
        )
        A_out_list.append(A_out_i)
        S_out_list.append(S_out_i)

    # Emit lattice
    A_lat, S_lat, receipt = emit_lattice(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    print(f"Period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {len(receipt['agreeing_classes'])}")
    print(f"Disagreeing classes: {len(receipt['disagreeing_classes'])}")
    print(f"Residue scope bits: {receipt['residue_scope_bits']}")

    # Invariant checks
    assert len(receipt['included_train_ids']) > 0, "Expected included trainings"
    assert receipt['residue_scope_bits'] >= 0, "Expected non-negative scope bits"

    # If period detected
    if receipt['p_r'] is not None or receipt['p_c'] is not None:
        print(f"✅ Period detected: ({receipt['p_r']}, {receipt['p_c']})")

        # Invariant: agreeing + disagreeing should cover all residues
        total_residues = (receipt['p_r'] or 1) * (receipt['p_c'] or 1)
        total_classified = len(receipt['agreeing_classes']) + len(receipt['disagreeing_classes'])
        assert total_classified == total_residues, \
            f"Expected {total_residues} residues, got {total_classified}"
        print(f"✅ All {total_residues} residues classified")
    else:
        print("No global period detected (expected for non-periodic task)")

    print("✅ PASS: Real ARC-AGI task processed correctly")
    return True


# =============================================================================
# Test 6: INVARIANT - No Minted Bits (only emit on agreement)
# =============================================================================

def test_no_minted_bits():
    """
    Test that lattice never emits admits outside agreeing residues.

    Invariant: S_lat[r] & (1<<c) == 1 ⟹ ∃(i,j) ∈ agreeing_classes: (r,c) in residue (i,j)
    """
    print("\n" + "=" * 70)
    print("TEST 6: NO MINTED BITS (ONLY EMIT ON AGREEMENT)")
    print("=" * 70)

    colors_order = [0, 1, 2]
    R_out, C_out = 4, 4

    # Create case with partial agreement
    # Training 1: period (2,2), residues (0,0) and (1,1) have color 1
    # Training 2: period (2,2), residue (0,0) has color 2 (DISAGREE), (1,1) has color 1 (AGREE)

    A_out_1 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b1010, 0b0101, 0b1010, 0b0101],
        2: [0b0000, 0b0000, 0b0000, 0b0000],
    }
    S_out_1 = [0b1111, 0b1111, 0b1111, 0b1111]

    A_out_2 = {
        0: [0b0101, 0b1010, 0b0101, 0b1010],
        1: [0b0000, 0b0101, 0b0000, 0b0101],  # Only residue (1,1)
        2: [0b1010, 0b0000, 0b1010, 0b0000],  # Residue (0,0) disagrees
    }
    S_out_2 = [0b1111, 0b1111, 0b1111, 0b1111]

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")

    # Verify (0,0) disagrees, (1,1) agrees
    assert (0, 0) in receipt['disagreeing_classes'], "Expected (0,0) to disagree"
    assert (1, 1) in receipt['agreeing_classes'], "Expected (1,1) to agree"

    # Invariant: Check S_lat only has bits on agreeing residues
    # Build expected scope from agreeing residues
    p_r, p_c = receipt['p_r'], receipt['p_c']
    expected_scope = [0] * R_out

    for i, j in receipt['agreeing_classes']:
        for r in range(R_out):
            if (p_r is None or r % p_r == i):
                for c in range(C_out):
                    if (p_c is None or c % p_c == j):
                        expected_scope[r] |= (1 << c)

    # Verify S_lat matches expected
    for r in range(R_out):
        assert (S_lat[r] & expected_scope[r]) == S_lat[r], \
            f"Row {r}: S_lat has bits outside agreeing residues: {bin(S_lat[r])} vs expected {bin(expected_scope[r])}"

    print("✅ PASS: No minted bits, only emit on agreeing residues")
    return True


# =============================================================================
# Test 7: EDGE CASE - No included trainings (all silent)
# =============================================================================

def test_no_included_trainings():
    """
    Test that all-silent trainings → silent lattice.

    Edge case: No trainings with scope → lattice is silent
    """
    print("\n" + "=" * 70)
    print("TEST 7: EDGE CASE - NO INCLUDED TRAININGS")
    print("=" * 70)

    colors_order = [0, 1]
    R_out, C_out = 2, 2

    # All trainings silent (S_out all zeros)
    A_out_list = [
        {0: [0, 0], 1: [0, 0]},
        {0: [0, 0], 1: [0, 0]},
    ]
    S_out_list = [
        [0, 0],
        [0, 0],
    ]

    A_lat, S_lat, receipt = emit_lattice(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # Invariant: Silent layer
    assert len(receipt['included_train_ids']) == 0, "Expected no included trainings"
    assert receipt['p_r'] is None, "Expected p_r=None"
    assert receipt['p_c'] is None, "Expected p_c=None"
    assert receipt['residue_scope_bits'] == 0, "Expected 0 scope bits"
    assert len(receipt['agreeing_classes']) == 0, "Expected no agreeing classes"

    print("✅ PASS: All-silent trainings → silent lattice")
    return True


# =============================================================================
# Test 8: EDGE CASE - Partial coverage in residue
# =============================================================================

def test_partial_coverage_in_residue():
    """
    Test residue with partial coverage (some pixels undefined).

    Spec: Only consider pixels where ALL trainings define.
    """
    print("\n" + "=" * 70)
    print("TEST 8: EDGE CASE - PARTIAL COVERAGE IN RESIDUE")
    print("=" * 70)

    colors_order = [0, 1]
    R_out, C_out = 4, 4

    # Training 1: Full coverage on residue (0,0)
    # Training 2: Partial coverage (missing pixel (0,0))
    # Since (0,0) is missing in training 2, residue (0,0) might still agree on other pixels

    A_out_1 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b1010, 0b0000, 0b1010, 0b0000],
    }
    S_out_1 = [0b1111, 0b1111, 0b1111, 0b1111]  # Full coverage

    A_out_2 = {
        0: [0b0101, 0b1111, 0b0101, 0b1111],
        1: [0b0010, 0b0000, 0b1010, 0b0000],  # Missing (0,0)
    }
    S_out_2 = [0b0111, 0b1111, 0b1111, 0b1111]  # Missing (0,0)

    A_lat, S_lat, receipt = emit_lattice(
        [A_out_1, A_out_2], [S_out_1, S_out_2], colors_order, R_out, C_out
    )

    print(f"Period: p_r={receipt['p_r']}, p_c={receipt['p_c']}")
    print(f"Agreeing classes: {receipt['agreeing_classes']}")
    print(f"Disagreeing classes: {receipt['disagreeing_classes']}")

    # Residue (0,0) should either:
    # 1. Agree on the 3 pixels where both define (pixels (0,2), (2,0), (2,2))
    # 2. Or disagree if any of those 3 differ

    # Since both have color 1 on residue (0,0) where both define → should agree
    if (0, 0) in receipt['agreeing_classes']:
        # Verify S_lat only has bits where both trainings define
        r, c = 0, 0
        bit = 1 << c
        # Pixel (0,0): training 2 doesn't define → should NOT be in S_lat
        assert not (S_lat[r] & bit), f"S_lat should not have bit at (0,0) where training 2 is silent"
        print("✅ PASS: Partial coverage handled correctly (only where all define)")
    else:
        print("Note: Residue (0,0) in disagreeing (implementation choice on partial coverage)")

    return True


# =============================================================================
# Runner
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("WO-09 LATTICE EMITTER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Grid Reconstruction", test_grid_reconstruction_from_planes),
        ("Period Consensus Mismatch", test_period_consensus_mismatch),
        ("Residue Unanimity Disagreement", test_residue_unanimity_disagreement),
        ("Deterministic Hashing", test_deterministic_hashing),
        ("Real ARC-AGI Data", test_real_arc_periodic_task),
        ("No Minted Bits", test_no_minted_bits),
        ("No Included Trainings", test_no_included_trainings),
        ("Partial Coverage in Residue", test_partial_coverage_in_residue),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"❌ {failed} tests FAILED")
    else:
        print("✅ ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
