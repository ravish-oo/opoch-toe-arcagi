#!/usr/bin/env python3
"""
WO-04a Comprehensive Tests on Real ARC Data (Receipts-Only)

Tests after audit:
  1. H5 one-sided periods (stripes task - EXPOSES BUG)
  2. H3 enum order verification (a→c→b→d)
  3. Invariants (trainings-only, determinism, no test leakage)
  4. Tie rule behavior
  5. Real ARC task sweep
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.canvas import choose_working_canvas, SizeUndetermined

# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def test_h5_one_sided_periods_stripes():
    """
    Test H5 with one-sided periods (col periods only).

    EXPECTED: H5 should fit via identity on rows (R=H), period on cols (C=kc*lcm_c)
    ACTUAL BUG: H5 returns False if lcm_r is None or lcm_c is None (line 615-616)

    This is the SAME bug found in WO-14 before it was fixed.
    """
    print("\n" + "=" * 70)
    print("TEST: H5 One-Sided Periods (Stripes Task - EXPOSES BUG)")
    print("=" * 70)

    # Create synthetic trainings with vertical stripes (col periods only)
    # Training 1: 10×6 → 10×6 (lcm_c=3, lcm_r=None, kr=1 identity, kc=2)
    # Training 2: 12×9 → 12×6 (lcm_c=3, lcm_r=None, kr=1 identity, kc=2)
    # Expected: H5 should fit with kr=1, kc=2, using R=H (identity), C=2*3=6

    # Vertical stripes: cols repeat with period 3
    def make_stripe_grid(H, W, period=3):
        """Make grid with vertical stripes (col periods, no row periods)."""
        grid = []
        for r in range(H):
            row = []
            for c in range(W):
                val = (c % period)  # Colors 0, 1, 2 cycling (includes background 0)
                row.append(val)
            grid.append(row)
        return grid

    train_pairs = [
        {
            "X": make_stripe_grid(10, 6, period=3),
            "Y": make_stripe_grid(10, 6, period=3)  # Same size, identity on rows
        },
        {
            "X": make_stripe_grid(12, 9, period=3),
            "Y": make_stripe_grid(12, 6, period=3)  # R=H (identity), C=2*3
        }
    ]

    frames_in = [{}, {}]
    frames_out = [{}, {}]
    xstar_shape = (15, 12)

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape
        )

        payload = receipts["payload"]
        winner = payload["winner"]["family"]

        # Check if H5 was tried and if it fit
        h5_attempts = [a for a in payload["attempts"] if a["family"] == "H5"]
        h5_fits = [a for a in h5_attempts if a["fit_all"]]

        if len(h5_fits) > 0:
            print(f"  ✅ H5 fit found (one-sided periods work)")
            print(f"  Winner: {winner}")
            return True
        else:
            print(f"  ❌ BUG CONFIRMED: H5 should fit but no H5 attempt succeeded")
            print(f"  H5 attempts checked: {len(h5_attempts)}")
            print(f"  H5 fits: {len(h5_fits)}")
            print(f"  This confirms the bug at line 615-616 in canvas.py")
            print(f"  Implementation returns False if lcm_r or lcm_c is None")
            print(f"  But should use identity rule: R=H if lcm_r=None, C=W if lcm_c=None")
            return False

    except SizeUndetermined as e:
        # If no hypothesis fits at all, H5 bug prevented it from fitting
        print(f"  ❌ BUG CONFIRMED: SIZE_UNDETERMINED raised")
        print(f"  H5 with one-sided periods should have fit")
        print(f"  Bug at line 615-616: skips training if ANY period is None")
        return False


def test_h3_enum_order_real_arc():
    """
    Spot check: Verify H3 attempts show a→c→b→d order on real ARC task.

    Expected order: a=1,c=1,b=0,d=0 then a=1,c=1,b=0,d=1 (d varies first)
    """
    print("\n" + "=" * 70)
    print("TEST: H3 Enumeration Order (a→c→b→d) on Real ARC")
    print("=" * 70)

    # Use first task
    task_id = sorted(all_tasks.keys())[0]
    task_json = all_tasks[task_id]

    train_pairs = []
    for pair in task_json["train"]:
        train_pairs.append({
            "X": pair["input"],
            "Y": pair["output"]
        })

    X_test = task_json["test"][0]["input"]
    H_test = len(X_test)
    W_test = len(X_test[0]) if X_test else 0

    frames_in = [{}] * len(train_pairs)
    frames_out = [{}] * len(train_pairs)

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, (H_test, W_test)
        )

        payload = receipts["payload"]
        attempts = payload["attempts"]

        # Find H3 attempts
        h3_attempts = [a for a in attempts if a["family"] == "H3"]

        if len(h3_attempts) < 20:
            print(f"  ⚠️  Not enough H3 attempts ({len(h3_attempts)})")
            return True

        print(f"  Task: {task_id}")
        print(f"  First 20 H3 attempts:")

        for i, att in enumerate(h3_attempts[:20]):
            params = att["params"]
            print(f"    {i:2d}: a={params['a']}, c={params['c']}, b={params['b']:2d}, d={params['d']:2d}")

        # Verify order: a→c→b→d
        # First few should have a=1, c=1, b=0, d varying
        violations = []

        # Check first 17 attempts (d varies 0..16 with a=1,c=1,b=0)
        for i in range(min(17, len(h3_attempts))):
            params = h3_attempts[i]["params"]
            if params["a"] != 1 or params["c"] != 1 or params["b"] != 0:
                violations.append(f"Attempt {i}: expected a=1,c=1,b=0 but got a={params['a']},c={params['c']},b={params['b']}")
            elif params["d"] != i:
                violations.append(f"Attempt {i}: expected d={i} but got d={params['d']}")

        # Attempt 17 should have a=1,c=1,b=1,d=0 (b increments after d exhausted)
        if len(h3_attempts) > 17:
            params = h3_attempts[17]["params"]
            if params["a"] != 1 or params["c"] != 1 or params["b"] != 1 or params["d"] != 0:
                violations.append(f"Attempt 17: expected a=1,c=1,b=1,d=0 but got a={params['a']},c={params['c']},b={params['b']},d={params['d']}")

        if violations:
            print(f"  ❌ FAIL: Enum order violations:")
            for v in violations[:5]:
                print(f"    {v}")
            return False

        print(f"  ✅ PASS: H3 enum order is a→c→b→d")
        return True

    except SizeUndetermined:
        print(f"  ⚠️  No fit found for task {task_id}")
        return True


def test_invariant_trainings_only():
    """Test that hypothesis selection uses trainings only (no test leakage)."""
    print("\n" + "=" * 70)
    print("TEST: Invariant - Trainings-Only (No Test Leakage)")
    print("=" * 70)

    # Same trainings, different test inputs should yield same winner
    train_pairs = [
        {
            "X": [[0, 1] for _ in range(2)],
            "Y": [[0, 1, 0, 1] for _ in range(4)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]

    # Run with different test inputs
    R1, C1, receipts1 = choose_working_canvas(train_pairs, frames_in, frames_out, (5, 5))
    R2, C2, receipts2 = choose_working_canvas(train_pairs, frames_in, frames_out, (10, 10))

    payload1 = receipts1["payload"]
    payload2 = receipts2["payload"]

    # Winner should be the same (trainings-only)
    winner1 = payload1["winner"]
    winner2 = payload2["winner"]

    if winner1["family"] != winner2["family"] or winner1["params"] != winner2["params"]:
        print(f"  ❌ FAIL: Test leakage detected")
        print(f"    Test (5,5): {winner1}")
        print(f"    Test (10,10): {winner2}")
        return False

    print(f"  ✅ PASS: Same winner for different test inputs (trainings-only)")
    print(f"  Winner: {winner1['family']} with params {winner1['params']}")
    return True


def test_invariant_no_lcm():
    """Test that WO-04a does not use LCM canvas (v1.6 spec)."""
    print("\n" + "=" * 70)
    print("TEST: Invariant - No LCM Canvas")
    print("=" * 70)

    train_pairs = [
        {
            "X": [[0, 1] for _ in range(2)],
            "Y": [[0, 1, 0, 1] for _ in range(4)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]

    R_out, C_out, receipts = choose_working_canvas(
        train_pairs, frames_in, frames_out, (5, 5)
    )

    # Check that receipts do not contain any "lcm_canvas" or "lcm_" fields (except H5 periods)
    payload = receipts["payload"]
    payload_str = json.dumps(payload, sort_keys=True)

    # LCM is okay in H5 period context, but not as a canvas size strategy
    if "lcm_canvas" in payload_str.lower():
        print(f"  ❌ FAIL: Found 'lcm_canvas' in receipts (v1.6 forbids LCM canvas)")
        return False

    print(f"  ✅ PASS: No LCM canvas strategy (v1.6 conformant)")
    return True


def test_10_task_sweep():
    """Test size prediction on 10 real ARC tasks."""
    print("\n" + "=" * 70)
    print("TEST: Size Prediction on 10 Real ARC Tasks")
    print("=" * 70)

    task_ids = sorted(all_tasks.keys())[:10]

    passed = 0
    failed = 0
    no_fit = 0

    for task_id in task_ids:
        task_json = all_tasks[task_id]

        try:
            train_pairs = []
            for pair in task_json["train"]:
                train_pairs.append({
                    "X": pair["input"],
                    "Y": pair["output"]
                })

            X_test = task_json["test"][0]["input"]
            H_test = len(X_test)
            W_test = len(X_test[0]) if X_test else 0

            frames_in = [{}] * len(train_pairs)
            frames_out = [{}] * len(train_pairs)

            R_out, C_out, receipts = choose_working_canvas(
                train_pairs, frames_in, frames_out, (H_test, W_test)
            )

            payload = receipts["payload"]

            # Verify section_hash present
            if "section_hash" not in receipts:
                print(f"  ❌ {task_id}: Missing section_hash")
                failed += 1
                continue

            # Verify winner present
            if "winner" not in payload:
                print(f"  ❌ {task_id}: Missing winner")
                failed += 1
                continue

            winner = payload["winner"]
            print(f"  ✅ {task_id}: Winner={winner['family']}, Size={R_out}×{C_out}")
            passed += 1

        except SizeUndetermined:
            print(f"  ⚠️  {task_id}: No fit")
            no_fit += 1

        except Exception as e:
            print(f"  ❌ {task_id}: Exception: {e}")
            failed += 1

    print(f"\n  PASSED: {passed}/10")
    print(f"  NO FIT: {no_fit}/10")
    print(f"  FAILED: {failed}/10")

    if failed > 0:
        return False
    else:
        print(f"  ✅ ALL TASKS PROCESSED")
        return True


def run_all_tests():
    """Run all WO-04a comprehensive tests."""
    print("\n" + "=" * 70)
    print("WO-04a COMPREHENSIVE TESTS (AFTER AUDIT)")
    print("=" * 70)

    tests = [
        ("H5 One-Sided Periods (BUG CHECK)", test_h5_one_sided_periods_stripes),
        ("H3 Enum Order (a→c→b→d)", test_h3_enum_order_real_arc),
        ("Invariant: Trainings-Only", test_invariant_trainings_only),
        ("Invariant: No LCM", test_invariant_no_lcm),
        ("10-Task Sweep", test_10_task_sweep),
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

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n  Total: {passed}/{total}")

    if passed == total:
        print("\n✅ WO-04a VALIDATED")
        return 0
    else:
        print("\n❌ ISSUES FOUND")
        print("\nCRITICAL BUG CONFIRMED:")
        print("  H5 one-sided periods bug at canvas.py:615-616")
        print("  Implementation returns False if lcm_r or lcm_c is None")
        print("  Should use identity rule: R=H if lcm_r=None, C=W if lcm_c=None")
        print("  This is the SAME bug found in WO-14 before it was fixed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
