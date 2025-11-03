"""
Sub-WO-02a Comprehensive Verification (PERIOD p>=2 Patch)

Tests proper period enforcement, filtering, and receipts on real ARC-AGI data.
Uses ONLY receipts for algebraic debugging (no internal state inspection).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import assert_double_run_equal, Receipts, blake3_hash
from arcbit.kernel import (
    pack_grid_to_planes,
    order_colors,
    minimal_period_1d,
)
from arcbit.kernel.period import period_2d_planes


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
with open(arc_data_path, "r") as f:
    tasks = json.load(f)


def test_minimal_period_1d_constants():
    """
    Verify constant rows return None (not 1).

    Spec: Sub-WO-02a Requirement 1 - Gate p>=2.
    """
    print("\n" + "=" * 70)
    print("TEST: minimal_period_1d - Constants Return None")
    print("=" * 70)

    test_cases = [
        {"label": "all_zeros", "mask": 0b000000, "W": 6, "expected": None},
        {"label": "all_ones", "mask": 0b111111, "W": 6, "expected": None},
        {"label": "all_zeros_W8", "mask": 0b00000000, "W": 8, "expected": None},
        {"label": "all_ones_W8", "mask": 0b11111111, "W": 8, "expected": None},
    ]

    all_pass = True
    for tc in test_cases:
        result = minimal_period_1d(tc["mask"], tc["W"])
        passed = result == tc["expected"]

        print(f"  {tc['label']}: W={tc['W']}, mask={tc['mask']:0{tc['W']}b}")
        print(f"    Expected: {tc['expected']}, Got: {result}, {'✅' if passed else '❌'}")

        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ✅ PASS: All constant rows return None")
        return True
    else:
        print("\n  ❌ FAIL: Some constants did not return None")
        return False


def test_minimal_period_1d_proper_periods():
    """
    Verify proper periods (p >= 2) are detected correctly.

    Spec: Sub-WO-02a Requirement 1 - Return p >= 2 for proper periods.
    """
    print("\n" + "=" * 70)
    print("TEST: minimal_period_1d - Proper Periods (p >= 2)")
    print("=" * 70)

    test_cases = [
        {"label": "period_2", "mask": 0b101010, "W": 6, "expected": 2},
        {"label": "period_3", "mask": 0b110110, "W": 6, "expected": 3},
        {"label": "period_2_W8", "mask": 0b10101010, "W": 8, "expected": 2},
        {"label": "period_4_W8", "mask": 0b11001100, "W": 8, "expected": 4},
        {"label": "no_period", "mask": 0b101011, "W": 6, "expected": None},
    ]

    all_pass = True
    for tc in test_cases:
        result = minimal_period_1d(tc["mask"], tc["W"])
        passed = result == tc["expected"]

        # Verify p >= 2 if not None
        if result is not None and result < 2:
            print(f"  ❌ CRITICAL: {tc['label']} returned p={result} < 2!")
            all_pass = False
            passed = False

        print(f"  {tc['label']}: W={tc['W']}, mask={tc['mask']:0{tc['W']}b}")
        print(f"    Expected: {tc['expected']}, Got: {result}, {'✅' if passed else '❌'}")

        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ✅ PASS: All proper periods >= 2")
        return True
    else:
        print("\n  ❌ FAIL: Some periods violate p >= 2")
        return False


def test_period_2d_receipts_structure():
    """
    Verify receipts structure conforms to Sub-WO-02a spec.

    Spec: Sub-WO-02a Requirement 4 - Receipts with all required fields.
    """
    print("\n" + "=" * 70)
    print("TEST: period_2d_planes - Receipts Structure")
    print("=" * 70)

    # Use real ARC task
    task = tasks["00576224"]
    grid = task["train"][0]["output"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    # Call with receipts
    p_r, p_c, residues, receipts = period_2d_planes(
        planes, H, W, colors, return_receipts=True
    )

    print(f"  Grid: {H}×{W}, colors: {colors}")
    print(f"  Results: p_r={p_r}, p_c={p_c}, residues={len(residues)}")

    # Check required fields
    required_sections = {
        "period.inputs": ["H", "W", "K", "colors_count"],
        "period.candidates": ["row_periods_nontrivial", "col_periods_nontrivial", "p_r_lcm_pre", "p_c_lcm_pre"],
        "period.validation": ["p_r_valid", "p_c_valid", "p_r", "p_c", "phase"],
        "period.residues": ["count", "first_two_popcounts"],
    }

    all_present = True
    for section, fields in required_sections.items():
        if section not in receipts:
            print(f"  ❌ Missing section: {section}")
            all_present = False
        else:
            for field in fields:
                if field not in receipts[section]:
                    print(f"  ❌ Missing field: {section}.{field}")
                    all_present = False

    if "section_hash" not in receipts:
        print(f"  ❌ Missing section_hash")
        all_present = False

    if all_present:
        print("  ✅ All required receipt fields present")
        print(f"  section_hash: {receipts['section_hash']}")
        return True
    else:
        print("  ❌ Some receipt fields missing")
        return False


def test_period_2d_no_ones_in_candidates():
    """
    Verify row_periods_nontrivial and col_periods_nontrivial contain no 1s.

    Spec: Sub-WO-02a Requirement 2 - Filter out 1 when collecting.
    """
    print("\n" + "=" * 70)
    print("TEST: period_2d_planes - No 1s in Candidates")
    print("=" * 70)

    # Test multiple real ARC grids
    test_grids = []
    for task_id in ["00576224", "009d5c81", "00dbd492"]:
        if task_id in tasks:
            task = tasks[task_id]
            for idx, example in enumerate(task["train"]):
                for grid_type in ["input", "output"]:
                    grid = example[grid_type]
                    test_grids.append({
                        "task_id": task_id,
                        "example": idx,
                        "type": grid_type,
                        "grid": grid
                    })

    all_pass = True
    for test_info in test_grids[:5]:  # Test first 5
        grid = test_info["grid"]
        H = len(grid)
        W = len(grid[0]) if grid else 0

        if H < 2 or W < 2:
            continue  # Skip edge cases

        colors = order_colors({c for row in grid for c in row} | {0})
        planes = pack_grid_to_planes(grid, H, W, colors)

        p_r, p_c, residues, receipts = period_2d_planes(
            planes, H, W, colors, return_receipts=True
        )

        row_periods = receipts["period.candidates"]["row_periods_nontrivial"]
        col_periods = receipts["period.candidates"]["col_periods_nontrivial"]

        # Check no 1s
        has_one_row = 1 in row_periods
        has_one_col = 1 in col_periods

        print(f"  {test_info['task_id']} ex{test_info['example']} {test_info['type']}: {H}×{W}")
        print(f"    row_periods: {row_periods}")
        print(f"    col_periods: {col_periods}")

        if has_one_row:
            print(f"    ❌ FAIL: row_periods contains 1")
            all_pass = False
        if has_one_col:
            print(f"    ❌ FAIL: col_periods contains 1")
            all_pass = False

        # Verify all periods >= 2
        if any(p < 2 for p in row_periods):
            print(f"    ❌ FAIL: row_periods has p < 2")
            all_pass = False
        if any(p < 2 for p in col_periods):
            print(f"    ❌ FAIL: col_periods has p < 2")
            all_pass = False

        if not has_one_row and not has_one_col:
            print(f"    ✅ No 1s, all periods >= 2")

    if all_pass:
        print("\n  ✅ PASS: No 1s in any candidate lists")
        return True
    else:
        print("\n  ❌ FAIL: Found 1s in candidate lists")
        return False


def test_period_2d_phase_fixed():
    """
    Verify phase is always [0, 0].

    Spec: Sub-WO-02a Invariant - Phase fixed at (0,0).
    """
    print("\n" + "=" * 70)
    print("TEST: period_2d_planes - Phase Fixed at (0,0)")
    print("=" * 70)

    # Test on real ARC task
    task = tasks["00576224"]
    grid = task["train"][0]["output"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    p_r, p_c, residues, receipts = period_2d_planes(
        planes, H, W, colors, return_receipts=True
    )

    phase = receipts["period.validation"]["phase"]

    print(f"  Phase: {phase}")

    if phase == [0, 0]:
        print("  ✅ PASS: Phase fixed at [0, 0]")
        return True
    else:
        print(f"  ❌ FAIL: Phase is {phase}, expected [0, 0]")
        return False


def test_double_run_determinism_receipts():
    """
    Verify double-run produces identical section_hash.

    Spec: Sub-WO-02a Invariant - Determinism.
    """
    print("\n" + "=" * 70)
    print("TEST: period_2d_planes - Double-Run Determinism")
    print("=" * 70)

    task = tasks["00576224"]
    grids = [task["train"][0]["output"], task["train"][1]["output"]]

    def build_receipts():
        r = Receipts("test-sub-wo02a-determinism")

        for idx, grid in enumerate(grids):
            H = len(grid)
            W = len(grid[0]) if grid else 0
            colors = order_colors({c for row in grid for c in row} | {0})
            planes = pack_grid_to_planes(grid, H, W, colors)

            p_r, p_c, residues, receipts = period_2d_planes(
                planes, H, W, colors, return_receipts=True
            )

            r.put(f"grid_{idx}_p_r", p_r)
            r.put(f"grid_{idx}_p_c", p_c)
            r.put(f"grid_{idx}_section_hash", receipts["section_hash"])
            r.put(f"grid_{idx}_row_periods", receipts["period.candidates"]["row_periods_nontrivial"])
            r.put(f"grid_{idx}_col_periods", receipts["period.candidates"]["col_periods_nontrivial"])

        return r

    try:
        assert_double_run_equal(build_receipts)
        print("  ✅ PASS: Double-run section hashes identical")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: Double-run mismatch: {e}")
        return False


def test_real_arc_solid_grids():
    """
    Verify solid (constant) grids on real ARC data return (None, None, []).

    Spec: Sub-WO-02a Quick Check - Solid rows return None.
    """
    print("\n" + "=" * 70)
    print("TEST: Real ARC Solid Grids")
    print("=" * 70)

    # Create a solid grid (all one color)
    solid_grid = [[7, 7, 7], [7, 7, 7], [7, 7, 7]]
    H, W = 3, 3

    colors = order_colors({c for row in solid_grid for c in row} | {0})
    planes = pack_grid_to_planes(solid_grid, H, W, colors)

    p_r, p_c, residues, receipts = period_2d_planes(
        planes, H, W, colors, return_receipts=True
    )

    print(f"  Solid 3×3 grid (all 7s)")
    print(f"  p_r: {p_r}, p_c: {p_c}, residues: {len(residues)}")
    print(f"  row_periods: {receipts['period.candidates']['row_periods_nontrivial']}")
    print(f"  col_periods: {receipts['period.candidates']['col_periods_nontrivial']}")

    # Solid grid should have no proper period
    correct = (p_r is None and p_c is None and len(residues) == 0)

    # Check candidates lists are empty (all rows/cols are constant → no proper periods)
    row_periods = receipts["period.candidates"]["row_periods_nontrivial"]
    col_periods = receipts["period.candidates"]["col_periods_nontrivial"]
    no_ones = (1 not in row_periods and 1 not in col_periods)

    if correct and no_ones:
        print("  ✅ PASS: Solid grid returns (None, None, []) with no 1s")
        return True
    else:
        print(f"  ❌ FAIL: Expected (None, None, []), got ({p_r}, {p_c}, {len(residues)} residues)")
        return False


def test_real_arc_striped_grid():
    """
    Verify striped (periodic) grids detect proper periods >= 2.

    Spec: Sub-WO-02a Quick Check - Striped output reflects true divisors >= 2.
    """
    print("\n" + "=" * 70)
    print("TEST: Real ARC Striped Grid")
    print("=" * 70)

    # Task 00576224 has column period 2
    task = tasks["00576224"]
    grid = task["train"][0]["output"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    print(f"  Task 00576224, training example 0, output:")
    for row in grid:
        print(f"    {row}")

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    p_r, p_c, residues, receipts = period_2d_planes(
        planes, H, W, colors, return_receipts=True
    )

    print(f"\n  Results:")
    print(f"  p_r: {p_r}, p_c: {p_c}, residues: {len(residues)}")
    print(f"  row_periods: {receipts['period.candidates']['row_periods_nontrivial']}")
    print(f"  col_periods: {receipts['period.candidates']['col_periods_nontrivial']}")

    # This task has column period 2
    has_proper_period = (p_c is not None and p_c >= 2)

    # Check no 1s in candidates
    row_periods = receipts["period.candidates"]["row_periods_nontrivial"]
    col_periods = receipts["period.candidates"]["col_periods_nontrivial"]
    no_ones = (1 not in row_periods and 1 not in col_periods)

    if has_proper_period and no_ones:
        print(f"  ✅ PASS: Detected proper period p_c={p_c} >= 2, no 1s in candidates")
        return True
    else:
        print(f"  ❌ FAIL: Expected proper period >= 2, got p_c={p_c}")
        return False


def run_all_tests():
    """Run all Sub-WO-02a tests."""
    print("\n" + "=" * 70)
    print("SUB-WO-02a COMPREHENSIVE TEST SUITE (PERIOD p>=2 PATCH)")
    print("=" * 70)

    tests = [
        test_minimal_period_1d_constants,
        test_minimal_period_1d_proper_periods,
        test_period_2d_receipts_structure,
        test_period_2d_no_ones_in_candidates,
        test_period_2d_phase_fixed,
        test_double_run_determinism_receipts,
        test_real_arc_solid_grids,
        test_real_arc_striped_grid,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"PASSED: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
