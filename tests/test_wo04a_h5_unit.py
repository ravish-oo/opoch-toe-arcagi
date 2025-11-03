#!/usr/bin/env python3
"""
WO-04a H5 Identity Rule Unit Test

Direct unit test of _check_fit_H5 function to verify identity rule.
Tests the fix at canvas.py:639-640 without relying on period extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.canvas import _check_fit_H5


class MockFeature:
    """Mock feature vector for testing."""
    def __init__(self, lcm_r, lcm_c):
        self.data = {
            "periods": {
                "lcm_r": lcm_r,
                "lcm_c": lcm_c
            }
        }

    def __getitem__(self, key):
        return self.data[key]


def test_h5_identity_row_none():
    """
    Test H5 with lcm_r=None (identity on rows).

    Training 1: 10×6 → 10×6 (lcm_r=None, lcm_c=3, kr=1, kc=2)
    Training 2: 12×9 → 12×6 (lcm_r=None, lcm_c=3, kr=1, kc=2)

    Expected: R_pred = H_in (identity), C_pred = kc * lcm_c
    """
    print("\n" + "=" * 70)
    print("TEST: H5 Identity Rule - lcm_r=None (Row Identity)")
    print("=" * 70)

    features_out = [
        MockFeature(lcm_r=None, lcm_c=3),
        MockFeature(lcm_r=None, lcm_c=3)
    ]

    sizes_out = [
        (10, 6),  # R=10 (identity from input), C=2*3
        (12, 6)   # R=12 (identity from input), C=2*3
    ]

    sizes_in = [
        (10, 6),  # Input sizes
        (12, 9)
    ]

    params = {"kr": 1, "kc": 2}

    fit_all, ok_train_ids, common_lcm_r, common_lcm_c = _check_fit_H5(
        features_out, sizes_out, sizes_in, params
    )

    print(f"  fit_all: {fit_all}")
    print(f"  ok_train_ids: {ok_train_ids}")
    print(f"  common_lcm_r: {common_lcm_r}")
    print(f"  common_lcm_c: {common_lcm_c}")

    if fit_all and ok_train_ids == [0, 1]:
        print(f"  ✅ PASS: H5 fits with identity on rows (lcm_r=None)")
        print(f"    Training 1: R=10 (identity from H_in=10), C=2*3=6")
        print(f"    Training 2: R=12 (identity from H_in=12), C=2*3=6")
        return True
    else:
        print(f"  ❌ FAIL: H5 should fit with identity rule")
        print(f"    Expected: fit_all=True, ok_train_ids=[0,1]")
        print(f"    Got: fit_all={fit_all}, ok_train_ids={ok_train_ids}")
        return False


def test_h5_identity_col_none():
    """
    Test H5 with lcm_c=None (identity on cols).

    Training: 4×8 → 4×8 (lcm_r=2, lcm_c=None, kr=2, kc=1)

    Expected: R_pred = kr * lcm_r, C_pred = W_in (identity)
    """
    print("\n" + "=" * 70)
    print("TEST: H5 Identity Rule - lcm_c=None (Col Identity)")
    print("=" * 70)

    features_out = [
        MockFeature(lcm_r=2, lcm_c=None)
    ]

    sizes_out = [
        (4, 8)  # R=2*2, C=8 (identity from input)
    ]

    sizes_in = [
        (6, 8)  # Input size
    ]

    params = {"kr": 2, "kc": 1}

    fit_all, ok_train_ids, common_lcm_r, common_lcm_c = _check_fit_H5(
        features_out, sizes_out, sizes_in, params
    )

    print(f"  fit_all: {fit_all}")
    print(f"  ok_train_ids: {ok_train_ids}")
    print(f"  common_lcm_r: {common_lcm_r}")
    print(f"  common_lcm_c: {common_lcm_c}")

    if fit_all and ok_train_ids == [0]:
        print(f"  ✅ PASS: H5 fits with identity on cols (lcm_c=None)")
        print(f"    Training: R=2*2=4, C=8 (identity from W_in=8)")
        return True
    else:
        print(f"  ❌ FAIL: H5 should fit with identity rule")
        return False


def test_h5_both_none():
    """
    Test H5 with both lcm_r=None and lcm_c=None (identity on both).

    Training: 5×7 → 5×7 (lcm_r=None, lcm_c=None, kr=1, kc=1)

    Expected: R_pred = H_in, C_pred = W_in (full identity)
    """
    print("\n" + "=" * 70)
    print("TEST: H5 Identity Rule - Both None (Full Identity)")
    print("=" * 70)

    features_out = [
        MockFeature(lcm_r=None, lcm_c=None)
    ]

    sizes_out = [
        (5, 7)  # Same as input (full identity)
    ]

    sizes_in = [
        (5, 7)
    ]

    params = {"kr": 1, "kc": 1}

    fit_all, ok_train_ids, common_lcm_r, common_lcm_c = _check_fit_H5(
        features_out, sizes_out, sizes_in, params
    )

    print(f"  fit_all: {fit_all}")
    print(f"  ok_train_ids: {ok_train_ids}")

    if fit_all and ok_train_ids == [0]:
        print(f"  ✅ PASS: H5 fits with full identity (both periods None)")
        return True
    else:
        print(f"  ❌ FAIL: H5 should fit with full identity")
        return False


def test_h5_both_present():
    """
    Test H5 with both periods present (no identity, baseline).

    Training: 4×6 → 4×6 (lcm_r=2, lcm_c=3, kr=2, kc=2)

    Expected: R_pred = 2*2=4, C_pred = 2*3=6
    """
    print("\n" + "=" * 70)
    print("TEST: H5 Both Periods Present (Baseline)")
    print("=" * 70)

    features_out = [
        MockFeature(lcm_r=2, lcm_c=3)
    ]

    sizes_out = [
        (4, 6)  # R=2*2, C=2*3
    ]

    sizes_in = [
        (10, 12)  # Input size (not used when periods are present)
    ]

    params = {"kr": 2, "kc": 2}

    fit_all, ok_train_ids, common_lcm_r, common_lcm_c = _check_fit_H5(
        features_out, sizes_out, sizes_in, params
    )

    print(f"  fit_all: {fit_all}")
    print(f"  common_lcm_r: {common_lcm_r}, common_lcm_c: {common_lcm_c}")

    if fit_all and common_lcm_r == 2 and common_lcm_c == 3:
        print(f"  ✅ PASS: H5 works with both periods present")
        return True
    else:
        print(f"  ❌ FAIL: H5 baseline case failed")
        return False


def test_h5_inconsistent_periods():
    """
    Test H5 rejects inconsistent periods across trainings.

    Training 1: lcm_r=2, lcm_c=3
    Training 2: lcm_r=3, lcm_c=3  (inconsistent row period)

    Expected: fit_all=False
    """
    print("\n" + "=" * 70)
    print("TEST: H5 Rejects Inconsistent Periods")
    print("=" * 70)

    features_out = [
        MockFeature(lcm_r=2, lcm_c=3),
        MockFeature(lcm_r=3, lcm_c=3)  # Different row period
    ]

    sizes_out = [(4, 6), (6, 6)]
    sizes_in = [(10, 12), (10, 12)]
    params = {"kr": 2, "kc": 2}

    fit_all, ok_train_ids, _, _ = _check_fit_H5(
        features_out, sizes_out, sizes_in, params
    )

    if not fit_all:
        print(f"  ✅ PASS: H5 correctly rejects inconsistent periods")
        return True
    else:
        print(f"  ❌ FAIL: H5 should reject inconsistent periods")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("WO-04a H5 Identity Rule Unit Tests")
    print("=" * 70)

    tests = [
        ("H5 Identity: lcm_r=None (Row Identity)", test_h5_identity_row_none),
        ("H5 Identity: lcm_c=None (Col Identity)", test_h5_identity_col_none),
        ("H5 Identity: Both None (Full Identity)", test_h5_both_none),
        ("H5 Both Periods Present (Baseline)", test_h5_both_present),
        ("H5 Rejects Inconsistent Periods", test_h5_inconsistent_periods),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    if passed == total:
        print(f"\n✅ ALL H5 UNIT TESTS PASSED ({passed}/{total})")
        print("  H5 identity rule fix at canvas.py:639-640 is working correctly!")
        sys.exit(0)
    else:
        print(f"\n❌ SOME H5 TESTS FAILED ({passed}/{total})")
        sys.exit(1)
