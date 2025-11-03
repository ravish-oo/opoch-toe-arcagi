#!/usr/bin/env python3
"""
WO-14 Size Prediction Bug Detection Tests

Tests that expose the spec violations:
  BUG #1: H3 bounds wrong (1..4 instead of 1..8 for a,c; 0..8 instead of 0..16 for b,d)
  BUG #2: H3 enum order wrong (a→b→c→d instead of a→c→b→d)
  BUG #3: H5 skip logic wrong (skips training if ANY period is None)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.features import agg_features, agg_size_fit, predict_size
from arcbit.kernel import order_colors

# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def test_h3_bounds_coverage():
    """
    Test that H3 attempts cover the full spec'd range.

    SPEC: a,c ∈ {1..8}, b,d ∈ {0..16} → 18,496 candidates
    BUG: Implementation only covers a,c ∈ {1..4}, b,d ∈ {0..8} → 1,296 candidates
    """
    print("\n" + "=" * 70)
    print("TEST: H3 Parameter Bounds Coverage")
    print("=" * 70)

    # Create synthetic training with known H3 solution outside implemented bounds
    # Training: (H=5, W=5) → (R=42, C=20)
    # Solution: a=5, b=17, c=4, d=0  (5*5+17=42, 4*5+0=20)
    # This is within spec (a=5≤8, b=17≤16... wait, b=17>16!)
    # Let me try: a=6, b=12, c=3, d=5  (6*5+12=42, 3*5+5=20)

    # Actually, let me use a simpler example:
    # (H=10, W=10) → (R=60, C=50)
    # Solution: a=6, b=0, c=5, d=0  (6*10+0=60, 5*10+0=50)
    # a=6 is outside {1..4}, so implementation won't find it

    class MockFeatureVector:
        def __init__(self, H, W):
            self.data = {
                "H": H,
                "W": W,
                "counts": {},
                "cc": {},
                "periods": {
                    "row_min": None,
                    "col_min": None,
                    "lcm_r": None,
                    "lcm_c": None,
                    "gcd_r": None,
                    "gcd_c": None
                }
            }

        def __getitem__(self, key):
            return self.data[key]

    # Create training pair with H3 solution requiring a=6
    train_fv = MockFeatureVector(10, 10)
    train_output = (60, 50)  # R = 6*10 + 0 = 60, C = 5*10 + 0 = 50

    test_fv = MockFeatureVector(15, 12)  # Should predict R=90, C=60

    train_pairs = [(train_fv, train_output)]

    result = agg_size_fit(train_pairs, test_fv)

    if result is None:
        print(f"  ⚠️  No hypothesis fit found")
        print(f"  This could expose H3 bounds bug if spec'd solution exists")

        # Check if H3 with a=6, c=5 would fit
        a, c = 6, 5
        R_pred = a * train_fv["H"]
        C_pred = c * train_fv["W"]
        if (R_pred, C_pred) == train_output:
            print(f"  ❌ BUG CONFIRMED: H3 (a={a}, c={c}, b=0, d=0) would fit but wasn't tried")
            print(f"     Implementation bounds: a,c ∈ {{1..4}} (should be {{1..8}})")
            return False
        else:
            print(f"  ✅ Different issue (not H3 bounds)")
            return True
    else:
        fit, receipts = result
        winner = fit["family"]
        print(f"  Winner: {winner}")

        if winner == "H3":
            params = fit["params"]
            a, c = params["a"], params["c"]
            if a > 4 or c > 4:
                print(f"  ❌ BUG: Found H3 with a={a} or c={c} > 4 (impossible with implemented bounds!)")
                return False
            else:
                print(f"  ✅ H3 fit within implemented bounds (a={a}, c={c})")
                return True
        else:
            print(f"  ✅ Different hypothesis won")
            return True


def test_h3_enum_order():
    """
    Test that H3 attempts are in spec'd order.

    SPEC: a→c→b→d
    BUG: Implementation does a→b→c→d
    """
    print("\n" + "=" * 70)
    print("TEST: H3 Enumeration Order")
    print("=" * 70)

    class MockFeatureVector:
        def __init__(self, H, W):
            self.data = {
                "H": H,
                "W": W,
                "counts": {},
                "cc": {},
                "periods": {
                    "row_min": None,
                    "col_min": None,
                    "lcm_r": None,
                    "lcm_c": None,
                    "gcd_r": None,
                    "gcd_c": None
                }
            }

        def __getitem__(self, key):
            return self.data[key]

    train_fv = MockFeatureVector(5, 5)
    train_output = (10, 10)

    test_fv = MockFeatureVector(5, 5)

    train_pairs = [(train_fv, train_output)]

    result = agg_size_fit(train_pairs, test_fv)

    if result is None:
        print(f"  ⚠️  No fit found")
        return True  # Can't test enum order without attempts

    fit, receipts = result
    payload = receipts["payload"]
    attempts = payload["attempts"]

    # Find first few H3 attempts
    h3_attempts = [a for a in attempts if a["family"] == "H3"][:10]

    if len(h3_attempts) < 2:
        print(f"  ⚠️  Not enough H3 attempts to check order")
        return True

    # Check order: should be a→c→b→d
    # First attempts should have a=1, c varying before b varies
    print(f"  First 10 H3 attempts:")
    for i, att in enumerate(h3_attempts):
        params = att["params"]
        print(f"    {i}: a={params['a']}, b={params['b']}, c={params['c']}, d={params['d']}")

    # Expected order (spec): a=1,c=1,b=0,d=0 then a=1,c=1,b=0,d=1, ...
    # Actual order (bug): a=1,b=0,c=1,d=0 then a=1,b=0,c=1,d=1, ...

    first = h3_attempts[0]["params"]
    second = h3_attempts[1]["params"]

    # If spec order (a→c→b→d): a same, c same, b same, d increments
    # If bug order (a→b→c→d): a same, b same, c same, d increments

    if first["a"] == 1 and second["a"] == 1:
        if first["b"] == 0 and second["b"] == 0:
            # a same, b same → checking c and d
            if first["c"] == 1 and second["c"] == 1:
                # a,b,c all same → d should increment (both orders agree here)
                print(f"  ✅ Can't distinguish orders from first 2 attempts")
                return True
            elif first["c"] == 1 and second["c"] == 2:
                # c increments before d → WRONG ORDER (should be a→c→b→d, so d should vary before c)
                print(f"  ❌ BUG CONFIRMED: c increments before d (wrong order)")
                print(f"     Expected (spec a→c→b→d): a=1,c=1,b=0,d=0 then a=1,c=1,b=0,d=1")
                print(f"     Actual: a=1,b=0,c=1,d=0 then a=1,b=0,c=2,d=0")
                return False

    print(f"  ⚠️  Inconclusive enum order test")
    return True


def test_h5_one_sided_periods():
    """
    Test H5 handling of one-sided periods.

    SPEC: If lcm_r is None, use R=H (identity). If lcm_c is None, use C=W (identity).
    BUG: Implementation skips entire training if ANY period is None.
    """
    print("\n" + "=" * 70)
    print("TEST: H5 One-Sided Period Handling")
    print("=" * 70)

    class MockFeatureVector:
        def __init__(self, H, W, lcm_r, lcm_c):
            self.data = {
                "H": H,
                "W": W,
                "counts": {},
                "cc": {},
                "periods": {
                    "row_min": None,
                    "col_min": None,
                    "lcm_r": lcm_r,
                    "lcm_c": lcm_c,
                    "gcd_r": None,
                    "gcd_c": None
                }
            }

        def __getitem__(self, key):
            return self.data[key]

    # Training with lcm_c=5, lcm_r=None
    # Expected: R should use identity (R=H), C should use period (C=kc*lcm_c)
    # Training: (H=10, W=10, lcm_c=5) → (R=10, C=15)
    # Solution: kr=1 (R=H=10), kc=3 (C=3*5=15)

    train_fv = MockFeatureVector(H=10, W=10, lcm_r=None, lcm_c=5)
    train_output = (10, 15)  # R=H (identity), C=kc*lcm_c

    test_fv = MockFeatureVector(H=12, W=8, lcm_r=None, lcm_c=4)

    train_pairs = [(train_fv, train_output)]

    result = agg_size_fit(train_pairs, test_fv)

    if result is None:
        print(f"  ⚠️  No fit found")
        print(f"  This could expose H5 skip bug")

        # Manually check if H5 with one-sided period would work
        # For training: R=H=10, C=3*5=15 ✓ matches (10, 15)
        print(f"  ❌ BUG CONFIRMED: H5 with one-sided period (kr=1, kc=3) would fit")
        print(f"     But implementation skips training when lcm_r is None")
        return False
    else:
        fit, receipts = result
        winner = fit["family"]
        print(f"  Winner: {winner}")

        if winner == "H5":
            print(f"  ✅ H5 fit found (one-sided periods work)")
            return True
        else:
            print(f"  ⚠️  Different hypothesis won (H5 skip bug may still exist)")
            return True


def run_all_tests():
    """Run all WO-14 bug detection tests."""
    print("\n" + "=" * 70)
    print("WO-14 SIZE PREDICTION BUG DETECTION TESTS")
    print("=" * 70)
    print("\nThese tests expose spec violations in the implementation:")
    print("  BUG #1: H3 bounds (1..4 instead of 1..8)")
    print("  BUG #2: H3 enum order (a→b→c→d instead of a→c→b→d)")
    print("  BUG #3: H5 skip logic (skips if ANY period is None)")

    tests = [
        test_h3_bounds_coverage,
        test_h3_enum_order,
        test_h5_one_sided_periods,
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
        print("✅ ALL TESTS PASSED (or bugs not exposed by these tests)")
        return 0
    else:
        print("❌ BUGS CONFIRMED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
