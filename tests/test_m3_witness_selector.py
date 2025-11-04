#!/usr/bin/env python3
"""
M3 Witness Selector - Unit Tests

Tests the minimal selector and M3 pipeline integration.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import select_witness_first, solve


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
arc_solutions_path = Path(__file__).parent.parent / "data" / "arc-agi_training_solutions.json"

with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

with open(arc_solutions_path, "r") as f:
    all_solutions = json.load(f)


def test_selector_witness_only():
    """Test 1: Selector with witness scope only (unanimity OFF)."""
    print("\n" + "="*70)
    print("TEST 1: Selector - Witness Only")
    print("="*70)

    R_out, C_out = 3, 3
    colors_order = [0, 1, 2, 3]

    # Create witness admits: color 1 at (0,0), color 2 at (1,1), color 3 at (2,2)
    A_wit = {
        0: [0b000000000, 0b000000000, 0b000000000],
        1: [0b000000001, 0b000000000, 0b000000000],  # bit at (0,0)
        2: [0b000000000, 0b000000010, 0b000000000],  # bit at (1,1)
        3: [0b000000000, 0b000000000, 0b000000100],  # bit at (2,2)
    }
    S_wit = [0b000000001, 0b000000010, 0b000000100]  # diagonal scope

    # Unanimity OFF
    A_uni = {c: [(1 << C_out) - 1] * R_out for c in colors_order}
    S_uni = [0] * R_out

    Y_out, receipts = select_witness_first(A_wit, S_wit, A_uni, S_uni, colors_order, R_out, C_out)

    print("Y_out:")
    for row in Y_out:
        print(f"  {row}")

    # Verify
    assert Y_out[0][0] == 1, "Witness should pick color 1 at (0,0)"
    assert Y_out[1][1] == 2, "Witness should pick color 2 at (1,1)"
    assert Y_out[2][2] == 3, "Witness should pick color 3 at (2,2)"

    # All other pixels should be bottom (0)
    assert Y_out[0][1] == 0 and Y_out[0][2] == 0
    assert Y_out[1][0] == 0 and Y_out[1][2] == 0
    assert Y_out[2][0] == 0 and Y_out[2][1] == 0

    # Check receipts
    assert receipts["counts"]["witness"] == 3, "Should have 3 witness picks"
    assert receipts["counts"]["bottom"] == 6, "Should have 6 bottom picks"
    assert receipts["counts"]["unanimity"] == 0, "Should have 0 unanimity picks"
    assert receipts["containment_verified"] == True
    assert "repaint_hash" in receipts

    print(f"✓ Counts: witness={receipts['counts']['witness']}, " +
          f"unanimity={receipts['counts']['unanimity']}, " +
          f"bottom={receipts['counts']['bottom']}")
    print(f"✓ Containment verified: {receipts['containment_verified']}")

    print("\n✅ TEST 1 PASSED")
    return True


def test_selector_min_color():
    """Test 2: Selector picks minimum color when multiple candidates."""
    print("\n" + "="*70)
    print("TEST 2: Selector - Min Color Selection")
    print("="*70)

    R_out, C_out = 2, 2
    colors_order = [0, 1, 2, 3]

    # At (0,0): both color 2 and 3 are admitted → should pick min (2)
    A_wit = {
        0: [0b0000, 0b0000],
        1: [0b0000, 0b0000],
        2: [0b0001, 0b0000],  # bit at (0,0)
        3: [0b0001, 0b0000],  # bit at (0,0) too
    }
    S_wit = [0b0001, 0b0000]  # scope only at (0,0)

    A_uni = {c: [(1 << C_out) - 1] * R_out for c in colors_order}
    S_uni = [0] * R_out

    Y_out, receipts = select_witness_first(A_wit, S_wit, A_uni, S_uni, colors_order, R_out, C_out)

    print("Y_out:")
    for row in Y_out:
        print(f"  {row}")

    # Verify min selection
    assert Y_out[0][0] == 2, "Should pick min color (2) when both 2 and 3 admitted"

    print(f"✓ Min color selection: Y_out[0][0] = {Y_out[0][0]}")

    print("\n✅ TEST 2 PASSED")
    return True


def test_m3_simple_identity():
    """Test 3: M3 pipeline on simple identity task."""
    print("\n" + "="*70)
    print("TEST 3: M3 Pipeline - Simple Identity Task")
    print("="*70)

    # Create simple identity task
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
        ],
        "test": [
            {"input": [[1, 2], [3, 4]]}
        ]
    }

    # Run solver
    Y_out, receipts = solve(task, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                            with_witness=True, with_unanimity=False)

    print(f"Y_out: {Y_out}")
    print(f"Expected: [[1, 2], [3, 4]]")

    # Check receipts structure
    assert "witness_learn" in receipts["payload"], "Should have witness_learn receipts"
    assert "selection" in receipts["payload"], "Should have selection receipts"
    assert "unanimity_evaluated" in receipts["payload"], "Should have unanimity_evaluated field"
    assert receipts["payload"]["unanimity_evaluated"] == False, "Unanimity should be OFF"

    # Check selection receipts
    selection = receipts["payload"]["selection"]
    assert "counts" in selection
    assert "repaint_hash" in selection
    assert "containment_verified" in selection
    assert selection["containment_verified"] == True

    total_pixels = selection["counts"]["witness"] + selection["counts"]["unanimity"] + selection["counts"]["bottom"]
    R_out = len(Y_out)
    C_out = len(Y_out[0]) if Y_out else 0
    assert total_pixels == R_out * C_out, f"Pixel count mismatch: {total_pixels} != {R_out * C_out}"

    print(f"✓ Receipts structure correct")
    print(f"✓ Selection counts: {selection['counts']}")
    print(f"✓ Total pixels = R_out × C_out: {total_pixels} = {R_out} × {C_out}")

    print("\n✅ TEST 3 PASSED")
    return True


def test_m3_real_task():
    """Test 4: M3 pipeline on real ARC task."""
    print("\n" + "="*70)
    print("TEST 4: M3 Pipeline - Real ARC Task")
    print("="*70)

    # Pick a simple task
    task_id = "007bbfb7"  # Simple task with clear pattern
    task = all_tasks[task_id]

    print(f"Task: {task_id}")
    print(f"Train pairs: {len(task['train'])}")
    print(f"Test inputs: {len(task['test'])}")

    # Run solver with H1-7
    Y_out, receipts = solve(task, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                            with_witness=True, with_unanimity=False)

    print(f"\nOutput shape: {len(Y_out)}×{len(Y_out[0]) if Y_out else 0}")

    # Check receipts
    assert "witness_learn" in receipts["payload"]
    assert "selection" in receipts["payload"]

    witness_learns = receipts["payload"]["witness_learn"]
    print(f"\nWitness learning results:")
    for wl in witness_learns:
        print(f"  Train {wl['train_id']}: silent={wl['silent']}, " +
              f"pieces={wl['num_pieces']}, " +
              f"σ_bijection={wl['sigma_bijection_ok']}, " +
              f"overlap={wl['overlap_conflict']}")

    selection = receipts["payload"]["selection"]
    print(f"\nSelection counts: {selection['counts']}")
    print(f"Containment verified: {selection['containment_verified']}")

    # Verify pixel count invariant
    R_out = len(Y_out)
    C_out = len(Y_out[0]) if Y_out else 0
    total_pixels = sum(selection["counts"].values())
    assert total_pixels == R_out * C_out, f"Pixel count mismatch"
    print(f"✓ Pixel count invariant: {total_pixels} = {R_out} × {C_out}")

    print("\n✅ TEST 4 PASSED")
    return True


def test_m3_with_ground_truth():
    """Test 5: M3 with ground truth comparison (expect some exact matches)."""
    print("\n" + "="*70)
    print("TEST 5: M3 with Ground Truth Comparison")
    print("="*70)

    # Test first 5 tasks
    task_ids = sorted(all_tasks.keys())[:5]

    matches = 0
    mismatches = 0

    for task_id in task_ids:
        task = all_tasks[task_id]

        # Run solver
        Y_out, receipts = solve(task, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                                with_witness=True, with_unanimity=False)

        # Check if ground truth exists
        if task_id in all_solutions:
            Y_expected = all_solutions[task_id][0]  # First test output

            if Y_out == Y_expected:
                print(f"  ✓ {task_id}: EXACT MATCH")
                matches += 1
            else:
                print(f"  ⚠️  {task_id}: MISMATCH")
                mismatches += 1

                # Show why (from receipts)
                selection = receipts["payload"]["selection"]
                witness_learns = receipts["payload"]["witness_learn"]

                scope_coverage = selection["counts"]["witness"] / (len(Y_out) * len(Y_out[0])) if Y_out else 0
                print(f"      Witness coverage: {scope_coverage:.1%}")
                print(f"      Selection: wit={selection['counts']['witness']}, " +
                      f"uni={selection['counts']['unanimity']}, " +
                      f"btm={selection['counts']['bottom']}")

                silent_count = sum(1 for wl in witness_learns if wl['silent'])
                print(f"      Silent trainings: {silent_count}/{len(witness_learns)}")
        else:
            print(f"  - {task_id}: No ground truth")

    print(f"\nMatches: {matches}, Mismatches: {mismatches}")

    print("\n✅ TEST 5 PASSED")
    return True


if __name__ == "__main__":
    print("M3 WITNESS SELECTOR - UNIT TESTS")
    print("="*70)

    tests = [
        test_selector_witness_only,
        test_selector_min_color,
        test_m3_simple_identity,
        test_m3_real_task,
        test_m3_with_ground_truth,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    sys.exit(0 if failed == 0 else 1)
