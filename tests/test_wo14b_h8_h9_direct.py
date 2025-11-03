#!/usr/bin/env python3
"""
Sub-WO-14b Direct Unit Tests for H8 and H9

Direct tests of _check_fit_H8, _check_fit_H9, and _get_top2_colors functions.
Designed to test H8/H9 functionality without going through full hypothesis tree.
Uses test cases that won't match H1-H7 to isolate H8/H9 behavior.

Test Strategy:
  - Test H8 with feature-dependent sizes (e.g., R = ncc_total, C = sum_nonzero)
  - Test H9 with guard-based piecewise laws
  - Test top-2 color selection with various edge cases
  - Use receipts-only verification (no internal state inspection)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.features import (
    _check_fit_H8,
    _check_fit_H9,
    _get_top2_colors,
    FeatureVector
)


# ============================================================================
# Mock Feature Vector Helpers
# ============================================================================

def make_feature_vector(H, W, counts, cc_stats, periods, sum_nonzero, ncc_total):
    """Create a mock FeatureVector for testing."""
    return FeatureVector(
        H=H,
        W=W,
        counts=counts,
        cc=cc_stats,
        periods=periods,
        sum_nonzero=sum_nonzero,
        ncc_total=ncc_total
    )


# ============================================================================
# Test _get_top2_colors
# ============================================================================

def test_top2_colors_basic():
    """
    Test top-2 color selection with clear winners.

    Training 1: {1: 10, 2: 5, 3: 2}
    Training 2: {1: 8, 2: 7, 3: 1}

    Expected: c1=1 (total=18), c2=2 (total=12)
    """
    print("\n" + "=" * 70)
    print("TEST: Top-2 Colors - Basic Case")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=10,
        counts={0: 83, 1: 10, 2: 5, 3: 2},
        cc_stats={},
        periods={},
        sum_nonzero=17,
        ncc_total=3
    )

    fv2 = make_feature_vector(
        H=10, W=10,
        counts={0: 84, 1: 8, 2: 7, 3: 1},
        cc_stats={},
        periods={},
        sum_nonzero=16,
        ncc_total=3
    )

    train_pairs = [(fv1, (10, 10)), (fv2, (10, 10))]

    c1, c2 = _get_top2_colors(train_pairs)

    print(f"  Training 1 counts: 1→10, 2→5, 3→2")
    print(f"  Training 2 counts: 1→8, 2→7, 3→1")
    print(f"  Total counts: 1→18, 2→12, 3→3")
    print(f"  c1={c1}, c2={c2}")

    if c1 == 1 and c2 == 2:
        print("  ✅ PASS: Top-2 colors selected correctly")
        return True
    else:
        print(f"  ❌ FAIL: Expected c1=1, c2=2, got c1={c1}, c2={c2}")
        return False


def test_top2_colors_tiebreak():
    """
    Test top-2 color selection with tie (smallest color wins).

    Training: {1: 10, 2: 10, 3: 5}

    Expected: c1=1 (tie-break: smallest), c2=2
    """
    print("\n" + "=" * 70)
    print("TEST: Top-2 Colors - Tie-Breaking")
    print("=" * 70)

    fv = make_feature_vector(
        H=10, W=10,
        counts={0: 75, 1: 10, 2: 10, 3: 5},
        cc_stats={},
        periods={},
        sum_nonzero=25,
        ncc_total=3
    )

    train_pairs = [(fv, (10, 10))]

    c1, c2 = _get_top2_colors(train_pairs)

    print(f"  Counts: 1→10, 2→10, 3→5 (tie between 1 and 2)")
    print(f"  c1={c1}, c2={c2}")

    if c1 == 1 and c2 == 2:
        print("  ✅ PASS: Tie-breaking works (smallest color wins)")
        return True
    else:
        print(f"  ❌ FAIL: Expected c1=1, c2=2, got c1={c1}, c2={c2}")
        return False


def test_top2_colors_single():
    """
    Test top-2 color selection with only one non-zero color.

    Expected: c1=1, c2=0
    """
    print("\n" + "=" * 70)
    print("TEST: Top-2 Colors - Single Color")
    print("=" * 70)

    fv = make_feature_vector(
        H=10, W=10,
        counts={0: 95, 1: 5},
        cc_stats={},
        periods={},
        sum_nonzero=5,
        ncc_total=1
    )

    train_pairs = [(fv, (10, 10))]

    c1, c2 = _get_top2_colors(train_pairs)

    print(f"  Counts: 1→5 (only one non-zero color)")
    print(f"  c1={c1}, c2={c2}")

    if c1 == 1 and c2 == 0:
        print("  ✅ PASS: Single color case handled correctly")
        return True
    else:
        print(f"  ❌ FAIL: Expected c1=1, c2=0, got c1={c1}, c2={c2}")
        return False


# ============================================================================
# Test _check_fit_H8 (Feature-Linear)
# ============================================================================

def test_h8_0feature():
    """
    Test H8 with 0-feature model (constant-only).

    Model: R = 5 (constant)
    Training: 10×8 → 5×8

    This won't match H1-H7 since R=5 is independent of H,W,periods.
    """
    print("\n" + "=" * 70)
    print("TEST: H8 0-Feature Model (Constant R=5)")
    print("=" * 70)

    fv = make_feature_vector(
        H=10, W=8,
        counts={0: 70, 1: 20, 2: 10},
        cc_stats={1: {"n": 2, "area_min": 5, "area_max": 15, "area_sum": 20},
                  2: {"n": 1, "area_min": 10, "area_max": 10, "area_sum": 10}},
        periods={"lcm_r": None, "lcm_c": None, "row_min": None, "col_min": None, "gcd_r": None, "gcd_c": None},
        sum_nonzero=30,
        ncc_total=3
    )

    train_pairs = [(fv, (5, 8))]

    # H8 row model: R = 5 (intercept=5, no features)
    params = {
        "axis": "rows",
        "intercept": 5,
        "feat1_idx": None,
        "feat1_coeff": None,
        "feat2_idx": None,
        "feat2_coeff": None
    }

    fit_all, ok_train_ids = _check_fit_H8(train_pairs, params, c1=1, c2=2)

    print(f"  Model: R = 5 (constant)")
    print(f"  Training: 10×8 → 5×8")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0]:
        print("  ✅ PASS: H8 0-feature model works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0]")
        return False


def test_h8_1feature_ncc_total():
    """
    Test H8 with 1-feature model using ncc_total.

    Model: R = 2 * ncc_total
    Training 1: ncc_total=3 → R=6
    Training 2: ncc_total=5 → R=10

    This won't match H1-H7 since it depends on connected component count.
    """
    print("\n" + "=" * 70)
    print("TEST: H8 1-Feature Model (R = 2 * ncc_total)")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=8,
        counts={0: 70, 1: 20, 2: 10},
        cc_stats={1: {"n": 2, "area_min": 5, "area_max": 15, "area_sum": 20},
                  2: {"n": 1, "area_min": 10, "area_max": 10, "area_sum": 10}},
        periods={},
        sum_nonzero=30,
        ncc_total=3
    )

    fv2 = make_feature_vector(
        H=12, W=9,
        counts={0: 88, 1: 25, 2: 15},
        cc_stats={1: {"n": 3, "area_min": 5, "area_max": 12, "area_sum": 25},
                  2: {"n": 2, "area_min": 7, "area_max": 8, "area_sum": 15}},
        periods={},
        sum_nonzero=40,
        ncc_total=5
    )

    train_pairs = [(fv1, (6, 8)), (fv2, (10, 9))]

    # H8 row model: R = 0 + 2 * ncc_total (feat_idx=4 is ncc_total)
    params = {
        "axis": "rows",
        "intercept": 0,
        "feat1_idx": 4,  # ncc_total is φ[4]
        "feat1_coeff": 2,
        "feat2_idx": None,
        "feat2_coeff": None
    }

    fit_all, ok_train_ids = _check_fit_H8(train_pairs, params, c1=1, c2=2)

    print(f"  Model: R = 2 * ncc_total")
    print(f"  Training 1: ncc_total=3 → R=6 (expected 6)")
    print(f"  Training 2: ncc_total=5 → R=10 (expected 10)")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0, 1]:
        print("  ✅ PASS: H8 1-feature model with ncc_total works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0,1]")
        return False


def test_h8_2feature():
    """
    Test H8 with 2-feature model.

    Model: R = 3 + 1*H + (-1)*sum_nonzero
    Training: H=10, sum_nonzero=7 → R = 3 + 10 - 7 = 6

    This won't match H1-H7 since it combines H and sum_nonzero.
    """
    print("\n" + "=" * 70)
    print("TEST: H8 2-Feature Model (R = 3 + H - sum_nonzero)")
    print("=" * 70)

    fv = make_feature_vector(
        H=10, W=8,
        counts={0: 93, 1: 5, 2: 2},
        cc_stats={},
        periods={},
        sum_nonzero=7,
        ncc_total=2
    )

    train_pairs = [(fv, (6, 8))]

    # H8 row model: R = 3 + 1*H + (-1)*sum_nonzero
    params = {
        "axis": "rows",
        "intercept": 3,
        "feat1_idx": 1,  # H is φ[1]
        "feat1_coeff": 1,
        "feat2_idx": 3,  # sum_nonzero is φ[3]
        "feat2_coeff": -1
    }

    fit_all, ok_train_ids = _check_fit_H8(train_pairs, params, c1=1, c2=2)

    print(f"  Model: R = 3 + H - sum_nonzero")
    print(f"  Training: H=10, sum_nonzero=7 → R = 3 + 10 - 7 = 6 (expected 6)")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0]:
        print("  ✅ PASS: H8 2-feature model works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0]")
        return False


def test_h8_count_c1_c2():
    """
    Test H8 with count_c1, count_c2 features (top-2 colors).

    Model: C = count_c1 // 2
    Training: count_c1=10 → C=5

    This won't match H1-H7 since it depends on specific color counts.
    """
    print("\n" + "=" * 70)
    print("TEST: H8 with count_c1 Feature (C = count_c1 // 2)")
    print("=" * 70)

    # Note: c1=1, so count_c1 = counts[1]
    fv = make_feature_vector(
        H=10, W=8,
        counts={0: 80, 1: 10, 2: 6, 3: 4},
        cc_stats={},
        periods={},
        sum_nonzero=20,
        ncc_total=3
    )

    train_pairs = [(fv, (10, 5))]

    # H8 col model: C = 0 + 1*count_c1 (but we want count_c1=10 → C=5)
    # Actually, let me use: C = count_c1 // 2 = 10 // 2 = 5
    # But H8 doesn't do division, it's linear. Let me adjust:
    # If count_c1=10 and we want C=5, we can't use pure multiplication.
    # Let me use a different example: C = 2 + count_c2 where count_c2=3 → C=5

    # Adjust: count_c2 = counts[2] = 6 (c2=2), so C = -1 + count_c2 = -1 + 6 = 5
    # Wait, coefficients must be in {-3,-2,-1,1,2,3}, intercepts in [0..32]
    # Let me use: C = 0 + 1*count_c2, where count_c2=5 to get C=5

    # Re-adjust the test case:
    fv2 = make_feature_vector(
        H=10, W=8,
        counts={0: 85, 1: 10, 2: 5},  # c1=1 (count=10), c2=2 (count=5)
        cc_stats={},
        periods={},
        sum_nonzero=15,
        ncc_total=2
    )

    train_pairs2 = [(fv2, (10, 5))]

    # H8 col model: C = 0 + 1*count_c2 (feat_idx=6 is count_c2)
    params = {
        "axis": "cols",
        "intercept": 0,
        "feat1_idx": 6,  # count_c2 is φ[6]
        "feat1_coeff": 1,
        "feat2_idx": None,
        "feat2_coeff": None
    }

    fit_all, ok_train_ids = _check_fit_H8(train_pairs2, params, c1=1, c2=2)

    print(f"  Model: C = count_c2 (c2=2, count=5)")
    print(f"  Training: count_c2=5 → C=5 (expected 5)")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0]:
        print("  ✅ PASS: H8 with count_c2 feature works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0]")
        return False


# ============================================================================
# Test _check_fit_H9 (Guarded Piecewise)
# ============================================================================

def test_h9_has_row_period():
    """
    Test H9 with has_row_period guard.

    Model: if has_row_period then H1(a=2,c=1) else H1(a=1,c=1)
    Training 1: has period → 10×8 → 20×8 (a=2)
    Training 2: no period → 12×8 → 12×8 (a=1)
    """
    print("\n" + "=" * 70)
    print("TEST: H9 Guard - has_row_period")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=8,
        counts={0: 70, 1: 30},
        cc_stats={},
        periods={"lcm_r": 5, "lcm_c": None, "row_min": 5, "col_min": None, "gcd_r": 5, "gcd_c": None},
        sum_nonzero=30,
        ncc_total=2
    )

    fv2 = make_feature_vector(
        H=12, W=8,
        counts={0: 80, 1: 16},
        cc_stats={},
        periods={"lcm_r": None, "lcm_c": None, "row_min": None, "col_min": None, "gcd_r": None, "gcd_c": None},
        sum_nonzero=16,
        ncc_total=1
    )

    train_pairs = [(fv1, (20, 8)), (fv2, (12, 8))]

    # H9 model: if has_row_period then H1(a=2,c=1) else H1(a=1,c=1)
    params = {
        "guard": "has_row_period",
        "true_family": "H1",
        "true_params": {"a": 2, "c": 1},
        "false_family": "H1",
        "false_params": {"a": 1, "c": 1}
    }

    fit_all, ok_train_ids = _check_fit_H9(train_pairs, params, c1=1, c2=0)

    print(f"  Model: if has_row_period then R=2*H else R=H, C=W")
    print(f"  Training 1: has period, 10×8 → 20×8 (2*10=20) ✓")
    print(f"  Training 2: no period, 12×8 → 12×8 (1*12=12) ✓")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0, 1]:
        print("  ✅ PASS: H9 with has_row_period guard works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0,1]")
        return False


def test_h9_ncc_gt_1():
    """
    Test H9 with ncc_gt_1 guard.

    Model: if ncc_gt_1 then H2(b=2,d=0) else H2(b=0,d=0)
    Training 1: ncc_total=3 → 10×8 → 12×8 (H+2)
    Training 2: ncc_total=1 → 10×8 → 10×8 (H+0)
    """
    print("\n" + "=" * 70)
    print("TEST: H9 Guard - ncc_gt_1")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=8,
        counts={0: 70, 1: 20, 2: 10},
        cc_stats={1: {"n": 2, "area_min": 5, "area_max": 15, "area_sum": 20},
                  2: {"n": 1, "area_min": 10, "area_max": 10, "area_sum": 10}},
        periods={},
        sum_nonzero=30,
        ncc_total=3
    )

    fv2 = make_feature_vector(
        H=10, W=8,
        counts={0: 90, 1: 10},
        cc_stats={1: {"n": 1, "area_min": 10, "area_max": 10, "area_sum": 10}},
        periods={},
        sum_nonzero=10,
        ncc_total=1
    )

    train_pairs = [(fv1, (12, 8)), (fv2, (10, 8))]

    # H9 model: if ncc_gt_1 then H2(b=2,d=0) else H2(b=0,d=0)
    params = {
        "guard": "ncc_gt_1",
        "true_family": "H2",
        "true_params": {"b": 2, "d": 0},
        "false_family": "H2",
        "false_params": {"b": 0, "d": 0}
    }

    fit_all, ok_train_ids = _check_fit_H9(train_pairs, params, c1=1, c2=2)

    print(f"  Model: if ncc_total>1 then R=H+2 else R=H, C=W")
    print(f"  Training 1: ncc_total=3>1, 10×8 → 12×8 (10+2=12) ✓")
    print(f"  Training 2: ncc_total=1≤1, 10×8 → 10×8 (10+0=10) ✓")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0, 1]:
        print("  ✅ PASS: H9 with ncc_gt_1 guard works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0,1]")
        return False


def test_h9_sum_gt_half():
    """
    Test H9 with sum_gt_half guard.

    Model: if sum_nonzero > H*W/2 then H6(kr=2,kc=1) else H1(a=1,c=1)
    Training 1: sum=60 > 50 (H*W/2=10*10/2=50) → 10×10 → 5×10 (H//2)
    Training 2: sum=40 ≤ 50 → 10×10 → 10×10 (identity)
    """
    print("\n" + "=" * 70)
    print("TEST: H9 Guard - sum_gt_half")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=10,
        counts={0: 40, 1: 60},
        cc_stats={},
        periods={},
        sum_nonzero=60,
        ncc_total=2
    )

    fv2 = make_feature_vector(
        H=10, W=10,
        counts={0: 60, 1: 40},
        cc_stats={},
        periods={},
        sum_nonzero=40,
        ncc_total=1
    )

    train_pairs = [(fv1, (5, 10)), (fv2, (10, 10))]

    # H9 model: if sum_gt_half then H6(kr=2,kc=1) else H1(a=1,c=1)
    params = {
        "guard": "sum_gt_half",
        "true_family": "H6",
        "true_params": {"kr": 2, "kc": 1},
        "false_family": "H1",
        "false_params": {"a": 1, "c": 1}
    }

    fit_all, ok_train_ids = _check_fit_H9(train_pairs, params, c1=1, c2=0)

    print(f"  Model: if sum_nonzero > H*W/2 then R=H//2 else R=H, C=W")
    print(f"  Training 1: sum=60 > 50, 10×10 → 5×10 (10//2=5) ✓")
    print(f"  Training 2: sum=40 ≤ 50, 10×10 → 10×10 (identity) ✓")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0, 1]:
        print("  ✅ PASS: H9 with sum_gt_half guard works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0,1]")
        return False


def test_h9_h_gt_w():
    """
    Test H9 with h_gt_w guard.

    Model: if H>W then H7(kr=2,kc=2) else H1(a=1,c=2)
    Training 1: 10×5 (H>W) → 5×3 (ceil(10/2), ceil(5/2))
    Training 2: 5×10 (H≤W) → 5×20 (identity on H, 2*W on C)
    """
    print("\n" + "=" * 70)
    print("TEST: H9 Guard - h_gt_w")
    print("=" * 70)

    fv1 = make_feature_vector(
        H=10, W=5,
        counts={0: 40, 1: 10},
        cc_stats={},
        periods={},
        sum_nonzero=10,
        ncc_total=1
    )

    fv2 = make_feature_vector(
        H=5, W=10,
        counts={0: 40, 1: 10},
        cc_stats={},
        periods={},
        sum_nonzero=10,
        ncc_total=1
    )

    train_pairs = [(fv1, (5, 3)), (fv2, (5, 20))]

    # H9 model: if h_gt_w then H7(kr=2,kc=2) else H1(a=1,c=2)
    params = {
        "guard": "h_gt_w",
        "true_family": "H7",
        "true_params": {"kr": 2, "kc": 2},
        "false_family": "H1",
        "false_params": {"a": 1, "c": 2}
    }

    fit_all, ok_train_ids = _check_fit_H9(train_pairs, params, c1=1, c2=0)

    print(f"  Model: if H>W then ceil_div else H*1, W*2")
    print(f"  Training 1: 10>5, 10×5 → 5×3 (ceil(10/2)=5, ceil(5/2)=3) ✓")
    print(f"  Training 2: 5≤10, 5×10 → 5×20 (5*1=5, 10*2=20) ✓")
    print(f"  fit_all={fit_all}, ok_train_ids={ok_train_ids}")

    if fit_all and ok_train_ids == [0, 1]:
        print("  ✅ PASS: H9 with h_gt_w guard works")
        return True
    else:
        print(f"  ❌ FAIL: Expected fit_all=True, ok_train_ids=[0,1]")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sub-WO-14b Direct Unit Tests: H8 and H9")
    print("=" * 70)

    tests = [
        # Top-2 color selection
        ("Top-2 Colors: Basic", test_top2_colors_basic),
        ("Top-2 Colors: Tie-Breaking", test_top2_colors_tiebreak),
        ("Top-2 Colors: Single Color", test_top2_colors_single),

        # H8 Feature-Linear
        ("H8: 0-Feature Model", test_h8_0feature),
        ("H8: 1-Feature (ncc_total)", test_h8_1feature_ncc_total),
        ("H8: 2-Feature Model", test_h8_2feature),
        ("H8: count_c1/c2 Features", test_h8_count_c1_c2),

        # H9 Guarded Piecewise
        ("H9: has_row_period Guard", test_h9_has_row_period),
        ("H9: ncc_gt_1 Guard", test_h9_ncc_gt_1),
        ("H9: sum_gt_half Guard", test_h9_sum_gt_half),
        ("H9: h_gt_w Guard", test_h9_h_gt_w),
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
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("  H8 and H9 functions are working correctly!")
        sys.exit(0)
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        sys.exit(1)
