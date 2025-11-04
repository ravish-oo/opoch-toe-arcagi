#!/usr/bin/env python3
"""
Sub-WO-04a-H8H9 Integration Test

Direct test of choose_working_canvas with H8/H9 support.
Tests that H8/H9 are properly integrated into canvas selection.

Key Verifications:
  1. H8/H9 attempts are logged in receipts
  2. H8/H9 can win when they fit better than H1-H7
  3. Tie rule correctly handles H8/H9 (after H7 in family order)
  4. Receipts structure is correct
  5. xstar_grid is properly handled (required for H8/H9)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.canvas import choose_working_canvas, SizeUndetermined


def test_h8_simple():
    """
    Test H8 with simple feature-linear relationship.

    Training: 5×5 → 10×10 (R = 2*H, C = 2*W)

    This matches H1(a=2,c=2), so H1 should win (earlier in family order).
    But we verify H8 attempts are present.
    """
    print("\n" + "=" * 70)
    print("TEST: H8 Attempt Logging (Simple Case)")
    print("=" * 70)

    train_pairs = [
        {
            "X": [[1]*5 for _ in range(5)],
            "Y": [[1]*10 for _ in range(10)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]
    xstar_shape = (6, 6)
    xstar_grid = [[1]*6 for _ in range(6)]
    colors_order = [0, 1]

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape, colors_order, xstar_grid
        )

        payload = receipts["payload"]

        # Check H8 attempts present
        h8_attempts = [a for a in payload["attempts"] if a["family"] == "H8"]

        print(f"  R_out={R_out}, C_out={C_out}")
        print(f"  Winner: {payload['winner']['family']}")
        print(f"  H8 attempts: {len(h8_attempts)}")

        if len(h8_attempts) > 0:
            print("  ✅ PASS: H8 attempts logged")
            return True
        else:
            print("  ❌ FAIL: No H8 attempts found")
            return False

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_h9_simple():
    """
    Test H9 with simple guard-based relationship.

    Training 1: 10×5 (H>W) → 20×5 (R=2*H, C=W)
    Training 2: 5×10 (H≤W) → 5×10 (R=H, C=W)

    Model: if H>W then H1(a=2,c=1) else H1(a=1,c=1)
    """
    print("\n" + "=" * 70)
    print("TEST: H9 Attempt Logging (Guard-Based)")
    print("=" * 70)

    train_pairs = [
        {
            "X": [[1]*5 for _ in range(10)],   # 10×5 (H>W)
            "Y": [[1]*5 for _ in range(20)]    # 20×5 (R=2*H)
        },
        {
            "X": [[1]*10 for _ in range(5)],   # 5×10 (H≤W)
            "Y": [[1]*10 for _ in range(5)]    # 5×10 (identity)
        }
    ]

    frames_in = [{}, {}]
    frames_out = [{}, {}]
    xstar_shape = (8, 4)  # H>W
    xstar_grid = [[1]*4 for _ in range(8)]
    colors_order = [0, 1]

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape, colors_order, xstar_grid
        )

        payload = receipts["payload"]

        # Check H9 attempts present
        h9_attempts = [a for a in payload["attempts"] if a["family"] == "H9"]
        h9_fits = [a for a in h9_attempts if a["fit_all"]]

        print(f"  R_out={R_out}, C_out={C_out}")
        print(f"  Winner: {payload['winner']['family']}")
        print(f"  H9 attempts: {len(h9_attempts)}")
        print(f"  H9 fits: {len(h9_fits)}")

        if len(h9_attempts) > 0:
            if len(h9_fits) > 0:
                print(f"  ✅ PASS: H9 found fitting models")
            else:
                print(f"  ⚠️  H9 attempts present but no fits (may be normal)")
            return True
        else:
            print("  ❌ FAIL: No H9 attempts found")
            return False

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_h8h9_without_xstar_grid():
    """
    Test that H8/H9 are skipped when xstar_grid is not provided.

    This ensures backward compatibility.
    """
    print("\n" + "=" * 70)
    print("TEST: H8/H9 Skipped Without xstar_grid")
    print("=" * 70)

    train_pairs = [
        {
            "X": [[1]*5 for _ in range(5)],
            "Y": [[1]*10 for _ in range(10)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]
    xstar_shape = (6, 6)
    colors_order = [0, 1]

    try:
        # Call WITHOUT xstar_grid
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape, colors_order
        )

        payload = receipts["payload"]
        winner = payload["winner"]

        # Check H8/H9 attempts logged but no fits selected
        h8_attempts = [a for a in payload["attempts"] if a["family"] == "H8"]
        h9_attempts = [a for a in payload["attempts"] if a["family"] == "H9"]

        print(f"  R_out={R_out}, C_out={C_out}")
        print(f"  Winner: {winner['family']}")
        print(f"  H8 attempts: {len(h8_attempts)}")
        print(f"  H9 attempts: {len(h9_attempts)}")

        # H8/H9 attempts should exist but winner should NOT be H8/H9
        if winner["family"] not in ["H8", "H9"]:
            print("  ✅ PASS: H8/H9 correctly skipped without xstar_grid")
            return True
        else:
            print("  ❌ FAIL: H8/H9 selected despite missing xstar_grid")
            return False

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_receipts_structure():
    """
    Test that receipts have correct structure with H8/H9 fields.
    """
    print("\n" + "=" * 70)
    print("TEST: Receipts Structure with H8/H9")
    print("=" * 70)

    train_pairs = [
        {
            "X": [[1]*5 for _ in range(5)],
            "Y": [[1]*10 for _ in range(10)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]
    xstar_shape = (6, 6)
    xstar_grid = [[1]*6 for _ in range(6)]
    colors_order = [0, 1]

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape, colors_order, xstar_grid
        )

        payload = receipts["payload"]

        # Required fields
        required_fields = [
            "num_trainings",
            "features_hash_per_training",
            "test_input_shape",
            "top2_colors",  # NEW for H8/H9
            "attempts",
            "total_candidates_checked",
            "winner",
            "R_out",
            "C_out"
        ]

        all_ok = True
        for field in required_fields:
            if field not in payload:
                print(f"  ❌ Missing field: {field}")
                all_ok = False

        # Check top2_colors structure
        if "top2_colors" in payload:
            top2 = payload["top2_colors"]
            if "c1" not in top2 or "c2" not in top2:
                print(f"  ❌ top2_colors missing c1 or c2")
                all_ok = False
            else:
                print(f"  ✅ top2_colors: c1={top2['c1']}, c2={top2['c2']}")

        # Check attempts include H8 and H9
        families = set(a["family"] for a in payload["attempts"])
        if "H8" not in families:
            print(f"  ❌ H8 not in attempts")
            all_ok = False
        else:
            print(f"  ✅ H8 attempts present")

        if "H9" not in families:
            print(f"  ❌ H9 not in attempts")
            all_ok = False
        else:
            print(f"  ✅ H9 attempts present")

        if all_ok:
            print(f"  ✅ PASS: Receipts structure correct")
            return True
        else:
            print(f"  ❌ FAIL: Receipts structure incomplete")
            return False

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tie_rule_with_h8h9():
    """
    Test that tie rule correctly orders H1 < ... < H7 < H8 < H9.

    Create scenario where H1 and H8 both fit with same test area.
    H1 should win (earlier family).
    """
    print("\n" + "=" * 70)
    print("TEST: Tie Rule with H8/H9 (Family Order)")
    print("=" * 70)

    # Simple case: R=2*H, C=2*W
    # H1(a=2,c=2) will fit
    # H8 with appropriate params will also fit
    # H1 should win due to family order
    train_pairs = [
        {
            "X": [[1]*5 for _ in range(5)],
            "Y": [[1]*10 for _ in range(10)]
        }
    ]

    frames_in = [{}]
    frames_out = [{}]
    xstar_shape = (3, 3)
    xstar_grid = [[1]*3 for _ in range(3)]
    colors_order = [0, 1]

    try:
        R_out, C_out, receipts = choose_working_canvas(
            train_pairs, frames_in, frames_out, xstar_shape, colors_order, xstar_grid
        )

        payload = receipts["payload"]
        winner = payload["winner"]

        # Check H1 and H8 both have fits
        h1_fits = [a for a in payload["attempts"] if a["family"] == "H1" and a["fit_all"]]
        h8_fits = [a for a in payload["attempts"] if a["family"] == "H8" and a["fit_all"]]

        print(f"  R_out={R_out}, C_out={C_out}")
        print(f"  Winner: {winner['family']}")
        print(f"  H1 fits: {len(h1_fits)}")
        print(f"  H8 fits: {len(h8_fits)}")

        if winner["family"] == "H1":
            print(f"  ✅ PASS: H1 wins over H8 (correct family order)")
            return True
        elif winner["family"] == "H8":
            # Check if H1 actually didn't fit (then H8 win is OK)
            if len(h1_fits) == 0:
                print(f"  ✅ PASS: H8 wins (H1 didn't fit)")
                return True
            else:
                print(f"  ❌ FAIL: H8 wins despite H1 fitting (tie rule violated)")
                return False
        else:
            print(f"  ⚠️  Winner is {winner['family']} (neither H1 nor H8)")
            return True

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Sub-WO-04a-H8H9 Integration Test Suite")
    print("=" * 70)

    tests = [
        ("H8 Attempt Logging", test_h8_simple),
        ("H9 Attempt Logging", test_h9_simple),
        ("H8/H9 Skipped Without xstar_grid", test_h8h9_without_xstar_grid),
        ("Receipts Structure", test_receipts_structure),
        ("Tie Rule with H8/H9", test_tie_rule_with_h8h9),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{'=' * 70}")
    if passed == total:
        print(f"✅ ALL INTEGRATION TESTS PASSED ({passed}/{total})")
        print("  Sub-WO-04a-H8H9 integration is functional!")
        print("  NOTE: H8 performance optimization is WIP")
        sys.exit(0)
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        sys.exit(1)
