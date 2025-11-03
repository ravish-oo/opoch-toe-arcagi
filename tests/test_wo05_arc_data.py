#!/usr/bin/env python3
"""
WO-05 Component Tests on Real ARC-AGI Data (Receipts-Only Debugging)

Tests WO-05 4-CC component extraction on real ARC tasks using algebraic debugging.
NO internal state inspection - all verification via receipts.

Focus on 3 key invariants (from spec):
  1. Union invariant: OR(component_masks) == input_plane for each color
  2. Overlap invariant: pairwise AND(component_i, component_j) == 0
  3. Perimeter sanity checks: area=1→4, 1×k→2+2k, k×1→2+2k
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.kernel import (
    pack_grid_to_planes,
    order_colors,
    components
)
from arcbit.core import assert_double_run_equal, Receipts


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def test_single_task_receipts_only(task_id):
    """
    Test WO-05 on a single task using ONLY receipts for verification.

    Args:
        task_id: ARC task ID (e.g., "00576224")

    Returns:
        bool: True if all receipt-based checks pass
    """
    if task_id not in all_tasks:
        print(f"  ⚠️  Task {task_id} not found")
        return False

    task_json = all_tasks[task_id]

    # Test on first training input
    grid = task_json["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    # Extract colors and pack to planes
    color_set = {0}
    for row in grid:
        for val in row:
            color_set.add(val)

    colors = order_colors(color_set)
    planes = pack_grid_to_planes(grid, H, W, colors)

    try:
        comps, receipts = components(planes, H, W, colors)

        # Extract payload
        if "payload" not in receipts:
            print(f"  ❌ Missing payload wrapper")
            return False

        payload = receipts["payload"]

        # CHECK 1: Union invariant (from receipts, not code inspection)
        if not payload.get("union_equal_input", False):
            print(f"  ❌ union_equal_input = False (components don't cover input)")
            return False

        # CHECK 2: Overlap invariant (from receipts)
        if not payload.get("overlap_zero", False):
            print(f"  ❌ overlap_zero = False (components overlap)")
            return False

        # CHECK 3: Per-color summary consistency (algebraic cross-check)
        per_color_summary = payload.get("per_color_summary", [])
        components_receipt = payload.get("components", [])

        for color_summary in per_color_summary:
            color = color_summary["color"]
            n_cc = color_summary["n_cc"]
            area_sum = color_summary["area_sum"]
            area_min = color_summary["area_min"]
            area_max = color_summary["area_max"]
            perim4_sum = color_summary["perim4_sum"]

            # Get components for this color
            color_comps = [c for c in components_receipt if c["color"] == color]

            # Cross-check: n_cc
            if len(color_comps) != n_cc:
                print(f"  ❌ Color {color}: n_cc mismatch ({len(color_comps)} != {n_cc})")
                return False

            if n_cc > 0:
                # Cross-check: area_sum
                actual_area_sum = sum(c["area"] for c in color_comps)
                if actual_area_sum != area_sum:
                    print(f"  ❌ Color {color}: area_sum mismatch ({actual_area_sum} != {area_sum})")
                    return False

                # Cross-check: area_min/max
                areas = [c["area"] for c in color_comps]
                if min(areas) != area_min or max(areas) != area_max:
                    print(f"  ❌ Color {color}: area_min/max mismatch")
                    return False

                # Cross-check: perim4_sum
                actual_perim4_sum = sum(c["perim4"] for c in color_comps)
                if actual_perim4_sum != perim4_sum:
                    print(f"  ❌ Color {color}: perim4_sum mismatch ({actual_perim4_sum} != {perim4_sum})")
                    return False

        # CHECK 4: Perimeter sanity checks (algebraic invariants)
        for comp_r in components_receipt:
            area = comp_r["area"]
            perim4 = comp_r["perim4"]

            # Single pixel: perim4 must be 4
            if area == 1 and perim4 != 4:
                print(f"  ❌ Single pixel component has perim4={perim4}, expected 4")
                return False

            # Perimeter must be positive and even
            if perim4 <= 0 or perim4 % 2 != 0:
                print(f"  ❌ Invalid perim4={perim4} for area={area}")
                return False

        # CHECK 5: Section hash present (determinism check anchor)
        if "section_hash" not in receipts:
            print(f"  ❌ Missing section_hash")
            return False

        return True

    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arc_task_00576224():
    """Test WO-05 on task 00576224 (receipts-only)."""
    print("\n" + "=" * 70)
    print("TEST: ARC Task 00576224 (WO-05 Components)")
    print("=" * 70)

    task_id = "00576224"
    if test_single_task_receipts_only(task_id):
        print(f"  ✅ Task {task_id} PASS (all invariants hold)")
        return True
    else:
        print(f"  ❌ Task {task_id} FAIL")
        return False


def test_arc_task_009d5c81():
    """Test WO-05 on task 009d5c81 (idempotence case from M0)."""
    print("\n" + "=" * 70)
    print("TEST: ARC Task 009d5c81 (WO-05 Components)")
    print("=" * 70)

    task_id = "009d5c81"
    if test_single_task_receipts_only(task_id):
        print(f"  ✅ Task {task_id} PASS (all invariants hold)")
        return True
    else:
        print(f"  ❌ Task {task_id} FAIL")
        return False


def test_determinism_double_run():
    """Verify WO-05 determinism via double-run receipt equality."""
    print("\n" + "=" * 70)
    print("TEST: Determinism (Double-Run Receipt Equality)")
    print("=" * 70)

    task_id = "00576224"
    task_json = all_tasks[task_id]

    grid = task_json["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    color_set = {0}
    for row in grid:
        for val in row:
            color_set.add(val)

    colors = order_colors(color_set)
    planes = pack_grid_to_planes(grid, H, W, colors)

    def run_components():
        r = Receipts("test-wo05-determinism")
        comps, receipts = components(planes, H, W, colors)

        # Store key receipt fields
        payload = receipts["payload"]
        r.put("n_components", len(comps))
        r.put("union_equal_input", payload["union_equal_input"])
        r.put("overlap_zero", payload["overlap_zero"])
        r.put("per_color_summary", payload["per_color_summary"])
        r.put("components", payload["components"])

        return r

    try:
        assert_double_run_equal(run_components)
        print("  ✅ PASS: Double-run receipts identical (deterministic)")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: Determinism violation: {e}")
        return False


def test_10_task_sweep():
    """
    Test WO-05 on 10 diverse tasks (receipts-only).

    All checks via receipts (algebraic debugging):
      - union_equal_input
      - overlap_zero
      - per_color_summary cross-checks
      - perimeter sanity
      - section_hash presence
    """
    print("\n" + "=" * 70)
    print("TEST: 10-Task Sweep (Receipts-Only Debugging)")
    print("=" * 70)

    task_ids = sorted(all_tasks.keys())[:10]

    passed = 0
    failed = 0
    failed_tasks = []

    for task_id in task_ids:
        if test_single_task_receipts_only(task_id):
            passed += 1
            print(f"  ✅ {task_id}")
        else:
            failed += 1
            failed_tasks.append(task_id)
            print(f"  ❌ {task_id}")

    print("\n" + "-" * 70)
    print(f"  PASSED: {passed}/{len(task_ids)}")
    print(f"  FAILED: {failed}/{len(task_ids)}")

    if failed > 0:
        print(f"  Failed tasks: {failed_tasks}")
        return False
    else:
        print("  ✅ ALL PASS")
        return True


def test_perimeter_sanity_on_arc_data():
    """
    Test perimeter formula sanity checks on diverse ARC grids.

    Sanity checks (algebraic invariants):
      - area=1 → perim4=4
      - perim4 > 0 and even
      - perim4 ≤ 4*area (perimeter can't exceed if each pixel standalone)
    """
    print("\n" + "=" * 70)
    print("TEST: Perimeter Sanity Checks on ARC Data")
    print("=" * 70)

    task_ids = sorted(all_tasks.keys())[:20]

    violations = []

    for task_id in task_ids:
        task_json = all_tasks[task_id]
        grid = task_json["train"][0]["input"]
        H = len(grid)
        W = len(grid[0]) if grid else 0

        color_set = {0}
        for row in grid:
            for val in row:
                color_set.add(val)

        colors = order_colors(color_set)
        planes = pack_grid_to_planes(grid, H, W, colors)

        try:
            comps, receipts = components(planes, H, W, colors)
            payload = receipts["payload"]
            components_receipt = payload["components"]

            for comp_r in components_receipt:
                area = comp_r["area"]
                perim4 = comp_r["perim4"]

                # Sanity 1: area=1 → perim4=4
                if area == 1 and perim4 != 4:
                    violations.append((task_id, f"area=1 but perim4={perim4}"))

                # Sanity 2: perim4 > 0 and even
                if perim4 <= 0 or perim4 % 2 != 0:
                    violations.append((task_id, f"invalid perim4={perim4}"))

                # Sanity 3: perim4 ≤ 4*area (loose upper bound)
                if perim4 > 4 * area:
                    violations.append((task_id, f"perim4={perim4} > 4*area={4*area}"))

        except Exception as e:
            violations.append((task_id, f"Exception: {e}"))

    if violations:
        print(f"  ❌ FAIL: {len(violations)} violations")
        for task_id, msg in violations[:5]:
            print(f"    {task_id}: {msg}")
        return False
    else:
        print(f"  ✅ PASS: All {len(task_ids)} tasks satisfy perimeter sanity checks")
        return True


def run_all_tests():
    """Run all WO-05 tests on real ARC data."""
    print("\n" + "=" * 70)
    print("WO-05 REAL ARC DATA TESTS (RECEIPTS-ONLY DEBUGGING)")
    print("=" * 70)

    tests = [
        ("Task 00576224", test_arc_task_00576224),
        ("Task 009d5c81", test_arc_task_009d5c81),
        ("Determinism Check", test_determinism_double_run),
        ("10-Task Sweep", test_10_task_sweep),
        ("Perimeter Sanity", test_perimeter_sanity_on_arc_data),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Final summary
    print("\n" + "=" * 70)
    print("WO-05 ARC DATA TEST SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n  Total: {passed}/{total} test suites passed")

    if passed == total:
        print("\n✅ WO-05 VALIDATED ON REAL ARC DATA")
        print("   All invariants hold (receipts-only verification)")
        return 0
    else:
        print("\n❌ WO-05 VALIDATION INCOMPLETE")
        print("   Fix failures before locking")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
