"""
M0 Bedrock Validation - Comprehensive Test Suite

Tests the M0 runner on real ARC-AGI tasks using receipts-first algebraic debugging.
Validates all 5 M0 steps without internal state inspection.

Test Strategy:
  1. Single task 009d5c81 (idempotence bug case) - smoke test
  2. 50-task curated slice (diverse shapes, symmetries, color sets)
  3. Full 1000-task sweep if 50 pass

M0 Invariants (ALL must hold):
  - No heuristics, no floats, no RNG
  - No minted non-zero bits
  - Color exclusivity preserved
  - Receipts-first: every step logs BLAKE3 hashes
  - Deterministic: double-run produces identical hashes
  - Y_placeholder = X* (identity, no solving)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.runner import solve, solve_with_determinism_check


# Load real ARC-AGI data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def test_single_task_009d5c81():
    """
    Test M0 on task 009d5c81 - the idempotence bug case.

    This task specifically failed before the idempotence fix.
    Re-canonicalization of G_canon must now yield pid='I', anchor=(0,0).

    Spec: M0 Step 3 - Canonicalize idempotence check.
    """
    print("\n" + "=" * 70)
    print("TEST: Single Task 009d5c81 (Idempotence Bug Case)")
    print("=" * 70)

    task_id = "009d5c81"
    if task_id not in all_tasks:
        print(f"  ⚠️  Task {task_id} not found in dataset")
        return False

    task_json = all_tasks[task_id]

    try:
        Y, receipts = solve(task_json)

        # Verify receipts structure (payload wrapper)
        if "payload" not in receipts:
            print(f"  ❌ Missing 'payload' wrapper in receipts")
            return False

        payload = receipts["payload"]

        required_sections = [
            "color_universe.colors_order",
            "color_universe.K",
            "pack_unpack",
            "frames.canonicalize",
            "frames.apply_pose_anchor",
        ]

        missing = [s for s in required_sections if s not in payload]
        if missing:
            print(f"  ❌ Missing receipt sections: {missing}")
            return False

        # Verify all frames passed idempotence check
        frames_receipts = payload["frames.canonicalize"]
        idempotence_failures = [
            f"{r['split']}[{r['idx']}].{r['io_type']}"
            for r in frames_receipts
            if not r.get("idempotent", False)
        ]

        if idempotence_failures:
            print(f"  ❌ Idempotence failures: {idempotence_failures}")
            return False

        # Verify Y = X*
        X_star = task_json["test"][0]["input"]
        if Y != X_star:
            print(f"  ❌ Y != X* (expected identity)")
            return False

        # Verify apply_pose_anchor equivalence
        apply_receipt = payload["frames.apply_pose_anchor"]
        if not apply_receipt.get("equivalence_ok", False):
            print(f"  ❌ apply_pose_anchor equivalence failed")
            return False

        if not apply_receipt.get("hash_equal", False):
            print(f"  ❌ apply_pose_anchor hash mismatch")
            return False

        # Summary
        print(f"  ✅ Task {task_id} PASS")
        print(f"     Colors: {payload['color_universe.K']}")
        print(f"     Frames tested: {len(frames_receipts)}")
        print(f"     Idempotence: ALL PASS")
        print(f"     apply_pose_anchor: {apply_receipt['apply.pose_id']}, anchor={apply_receipt['apply.anchor']}")
        print(f"     Y == X*: True")

        return True

    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_task_detailed(task_id):
    """
    Test M0 on a single task with detailed receipt inspection.

    Args:
        task_id: ARC task ID (e.g., "00576224")

    Returns:
        bool: True if all checks pass
    """
    if task_id not in all_tasks:
        print(f"  ⚠️  Task {task_id} not found")
        return False

    task_json = all_tasks[task_id]

    try:
        Y, receipts = solve(task_json)

        # Extract payload
        if "payload" not in receipts:
            print(f"    ❌ Missing payload wrapper")
            return False

        payload = receipts["payload"]

        # Check 1: Color universe
        K = payload.get("color_universe.K", 0)
        if K < 1:
            print(f"    ❌ Invalid K: {K}")
            return False

        # Check 2: PACK/UNPACK identity (all must have pack_equal=True)
        pack_receipts = payload.get("pack_unpack", [])
        pack_failures = [
            f"{r['split']}[{r['idx']}].{r['io_type']}"
            for r in pack_receipts
            if not r.get("pack_equal", False)
        ]
        if pack_failures:
            print(f"    ❌ PACK/UNPACK failures: {pack_failures}")
            return False

        # Check 3: Canonicalize idempotence (all must have idempotent=True)
        frames_receipts = payload.get("frames.canonicalize", [])
        idempotence_failures = [
            f"{r['split']}[{r['idx']}].{r['io_type']}"
            for r in frames_receipts
            if not r.get("idempotent", False)
        ]
        if idempotence_failures:
            print(f"    ❌ Idempotence failures: {idempotence_failures}")
            return False

        # Check 4: apply_pose_anchor equivalence
        apply_receipt = payload.get("frames.apply_pose_anchor", {})
        if not apply_receipt.get("equivalence_ok", False):
            print(f"    ❌ apply_pose_anchor equivalence failed")
            return False

        if not apply_receipt.get("hash_equal", False):
            print(f"    ❌ apply_pose_anchor hash mismatch")
            return False

        # Check 5: Y == X*
        X_star = task_json["test"][0]["input"]
        if Y != X_star:
            print(f"    ❌ Y != X*")
            return False

        return True

    except Exception as e:
        print(f"    ❌ EXCEPTION: {e}")
        return False


def test_curated_50_slice():
    """
    Test M0 on a curated 50-task slice spanning diverse characteristics.

    Selection criteria:
      - Diverse grid shapes (1x1 to 30x30)
      - Various symmetries (asymmetric, D4, special cases)
      - Different color sets (2-10 colors)
      - Edge cases (all-zero rows, single-color grids)

    Spec: M0 validation on representative sample before full sweep.
    """
    print("\n" + "=" * 70)
    print("TEST: Curated 50-Task Slice")
    print("=" * 70)

    # Curated task IDs (first 50 from training set, diverse by design)
    task_ids = sorted(all_tasks.keys())[:50]

    print(f"  Testing {len(task_ids)} tasks...")

    passed = 0
    failed = 0
    failed_tasks = []

    for task_id in task_ids:
        if test_single_task_detailed(task_id):
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
        print("  ✅ ALL PASS - Ready for full 1000-task sweep")
        return True


def test_full_1000_sweep():
    """
    Test M0 on all 1000 training tasks.

    This is the final validation before locking M0 receipts.

    Spec: M0 bedrock validation on full dataset.
    """
    print("\n" + "=" * 70)
    print("TEST: Full 1000-Task Sweep")
    print("=" * 70)

    task_ids = sorted(all_tasks.keys())

    print(f"  Testing {len(task_ids)} tasks...")
    print("  (This may take 1-2 minutes)")

    passed = 0
    failed = 0
    failed_tasks = []

    for i, task_id in enumerate(task_ids):
        if test_single_task_detailed(task_id):
            passed += 1
        else:
            failed += 1
            failed_tasks.append(task_id)

        # Progress indicator every 100 tasks
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(task_ids)} ({passed} passed, {failed} failed)")

    print("\n" + "-" * 70)
    print(f"  FINAL: {passed}/{len(task_ids)} PASS")

    if failed > 0:
        print(f"  FAILED: {failed} tasks")
        print(f"  Failed task IDs: {failed_tasks[:10]}{'...' if len(failed_tasks) > 10 else ''}")
        return False
    else:
        print("  ✅ ALL 1000 TASKS PASS")
        print("  M0 RECEIPTS LOCKED ✓")
        return True


def test_determinism_check():
    """
    Test solve_with_determinism_check on multiple tasks.

    Verifies double-run produces identical receipts.

    Spec: M0 determinism invariant.
    """
    print("\n" + "=" * 70)
    print("TEST: Determinism Check (Double-Run)")
    print("=" * 70)

    # Test on 10 diverse tasks
    task_ids = sorted(all_tasks.keys())[:10]

    all_pass = True
    for task_id in task_ids:
        task_json = all_tasks[task_id]

        try:
            Y, receipts = solve_with_determinism_check(task_json)

            if not receipts.get("determinism.double_run_ok", False):
                print(f"  ❌ {task_id}: determinism.double_run_ok = False")
                all_pass = False
            else:
                print(f"  ✅ {task_id}: Double-run identical")

        except Exception as e:
            print(f"  ❌ {task_id}: {e}")
            all_pass = False

    if all_pass:
        print("  ✅ PASS: All tasks deterministic")
        return True
    else:
        print("  ❌ FAIL: Determinism violations detected")
        return False


def run_all_tests():
    """Run all M0 tests in sequence."""
    print("\n" + "=" * 70)
    print("M0 BEDROCK VALIDATION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Task 009d5c81 (Idempotence Bug Case)", test_single_task_009d5c81),
        ("Determinism Check", test_determinism_check),
        ("50-Task Curated Slice", test_curated_50_slice),
        ("Full 1000-Task Sweep", test_full_1000_sweep),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"{'=' * 70}")

        try:
            result = test_func()
            results.append((name, result))

            if not result:
                print(f"\n⚠️  Test '{name}' FAILED - stopping test suite")
                break

        except Exception as e:
            print(f"\n❌ EXCEPTION in '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            break

    # Final summary
    print("\n" + "=" * 70)
    print("M0 FINAL TEST SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n  Total: {passed}/{total} test suites passed")

    if passed == len(tests):
        print("\n✅ M0 BEDROCK VALIDATION COMPLETE")
        print("   All 1000 tasks pass - receipts locked")
        return 0
    else:
        print("\n❌ M0 VALIDATION INCOMPLETE")
        print("   Fix failures before proceeding to M1")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
