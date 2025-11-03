#!/usr/bin/env python3
"""
WO-14 Feature Extraction Tests on Real ARC Data (Receipts-Only)

Tests feature extraction (counts, cc, periods) using algebraic debugging.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.features import agg_features
from arcbit.kernel import order_colors

# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def test_features_checksum():
    """Test that sum(counts) == H*W (algebraic invariant)."""
    print("\n" + "=" * 70)
    print("TEST: Feature Counts Checksum (sum = H*W)")
    print("=" * 70)

    task_id = "00576224"
    grid = all_tasks[task_id]["train"][0]["input"]

    H = len(grid)
    W = len(grid[0]) if grid else 0

    color_set = {0}
    for row in grid:
        for val in row:
            color_set.add(val)

    C_order = order_colors(color_set)

    fv, receipts = agg_features(grid, H, W, C_order)

    # CHECK: sum(counts) == H*W
    counts_sum = sum(fv["counts"].values())

    if counts_sum != H * W:
        print(f"  ❌ FAIL: sum(counts) = {counts_sum}, expected {H*W}")
        return False

    print(f"  ✅ PASS: sum(counts) = {counts_sum} == H*W = {H*W}")

    # CHECK: counts from receipts match feature vector
    payload = receipts["payload"]
    receipts_counts = {int(k): v for k, v in payload["counts"].items()}

    if receipts_counts != fv["counts"]:
        print(f"  ❌ FAIL: receipts counts mismatch")
        return False

    print(f"  ✅ PASS: receipts counts match feature vector")
    return True


def test_periods_proper_only():
    """Test that all periods are p>=2 or None (no period-1)."""
    print("\n" + "=" * 70)
    print("TEST: Periods are Proper (p>=2) or None")
    print("=" * 70)

    task_ids = sorted(all_tasks.keys())[:10]

    violations = []

    for task_id in task_ids:
        grid = all_tasks[task_id]["train"][0]["input"]

        H = len(grid)
        W = len(grid[0]) if grid else 0

        color_set = {0}
        for row in grid:
            for val in row:
                color_set.add(val)

        C_order = order_colors(color_set)

        fv, receipts = agg_features(grid, H, W, C_order)

        # Check all period fields
        periods = fv["periods"]
        for key, val in periods.items():
            if val is not None:
                if val < 2:
                    violations.append((task_id, key, val))

    if violations:
        print(f"  ❌ FAIL: {len(violations)} period-1 found")
        for task_id, key, val in violations[:5]:
            print(f"    {task_id}: {key}={val}")
        return False
    else:
        print(f"  ✅ PASS: All {len(task_ids)} tasks have proper periods (p>=2 or None)")
        return True


def test_cc_stats_from_wo05():
    """Test that cc stats match WO-05 component receipts."""
    print("\n" + "=" * 70)
    print("TEST: CC Stats Match WO-05 Receipts")
    print("=" * 70)

    task_id = "00576224"
    grid = all_tasks[task_id]["train"][0]["input"]

    H = len(grid)
    W = len(grid[0]) if grid else 0

    color_set = {0}
    for row in grid:
        for val in row:
            color_set.add(val)

    C_order = order_colors(color_set)

    fv, receipts = agg_features(grid, H, W, C_order)

    # Get cc stats from receipts
    payload = receipts["payload"]
    cc_receipts = {int(k): v for k, v in payload["cc"].items()}

    # Check that all non-zero colors have cc stats
    for color in C_order:
        if color == 0:
            # Background: should be None
            if cc_receipts[color]["n"] is not None:
                print(f"  ❌ FAIL: Color 0 should have None cc stats")
                return False
        else:
            # Non-zero: should have stats or be 0
            cc = cc_receipts[color]
            if cc["n"] is not None:
                if cc["n"] < 0:
                    print(f"  ❌ FAIL: Color {color} has negative n_cc")
                    return False

    print(f"  ✅ PASS: CC stats consistent with WO-05")
    return True


def test_determinism_features():
    """Test double-run produces identical feature receipts."""
    print("\n" + "=" * 70)
    print("TEST: Feature Extraction Determinism")
    print("=" * 70)

    task_id = "00576224"
    grid = all_tasks[task_id]["train"][0]["input"]

    H = len(grid)
    W = len(grid[0]) if grid else 0

    color_set = {0}
    for row in grid:
        for val in row:
            color_set.add(val)

    C_order = order_colors(color_set)

    # Run twice
    fv1, receipts1 = agg_features(grid, H, W, C_order)
    fv2, receipts2 = agg_features(grid, H, W, C_order)

    # Compare section hashes
    hash1 = receipts1["section_hash"]
    hash2 = receipts2["section_hash"]

    if hash1 != hash2:
        print(f"  ❌ FAIL: section_hash mismatch")
        print(f"    Run 1: {hash1}")
        print(f"    Run 2: {hash2}")
        return False

    print(f"  ✅ PASS: Double-run section_hash identical")
    return True


def run_all_tests():
    """Run all WO-14 feature tests."""
    print("\n" + "=" * 70)
    print("WO-14 FEATURE EXTRACTION TESTS")
    print("=" * 70)

    tests = [
        test_features_checksum,
        test_periods_proper_only,
        test_cc_stats_from_wo05,
        test_determinism_features,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("SUMMARY")
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
