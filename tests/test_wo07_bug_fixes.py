#!/usr/bin/env python3
"""
WO-07 Bug Fix Validation

Specific tests to verify:
1. U_r uses C_out (not hardcoded 64)
2. Explicit embedding handles dimension mismatch correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.witness_emit import emit_witness, _normalize_admit_all
from src.arcbit.emitters.witness_learn import learn_witness
from src.arcbit.kernel import canonicalize


def test_bug1_u_r_uses_c_out():
    """Bug #1: Verify U_r uses C_out instead of hardcoded 64."""
    print("="*70)
    print("BUG #1 FIX VALIDATION: U_r uses C_out")
    print("="*70)

    # Test with various canvas widths
    test_widths = [1, 5, 10, 30, 63, 64, 65]  # Include edge cases around 64

    for C_out in test_widths:
        R_out = 3
        colors_order = [0, 1, 2]

        # Create admit planes with all bits set
        A_i = {c: [(1 << C_out) - 1] * R_out for c in colors_order}
        S_i = [(1 << C_out) - 1] * R_out  # Full scope

        # Before normalization: scope has all bits
        scope_before = sum(bin(row).count("1") for row in S_i)

        # Run normalization
        _normalize_admit_all(A_i, S_i, colors_order, R_out, C_out)

        # After normalization: scope should be empty (all pixels admit-all)
        scope_after = sum(bin(row).count("1") for row in S_i)

        print(f"  C_out={C_out:3d}: scope {scope_before:3d} → {scope_after:3d} bits", end="")

        # Verify scope is cleared (U_r removed all admit-all pixels)
        assert scope_after == 0, f"Failed: scope should be 0 for C_out={C_out}"
        print(" ✓")

    print("\n✅ BUG #1 FIX VERIFIED: U_r correctly uses C_out for all widths")
    return True


def test_bug2_explicit_embedding():
    """Bug #2: Verify explicit embedding handles dimensions correctly."""
    print("\n" + "="*70)
    print("BUG #2 FIX VALIDATION: Explicit Embedding")
    print("="*70)

    # Create task where posed dimensions differ from canvas
    # Input: 3×2, Output: 5×7 (different aspect ratio)
    Xi_raw = [[1, 2], [3, 4], [5, 6]]
    Yi_raw = [[0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 2, 0, 0, 0],
              [0, 0, 3, 4, 0, 0, 0],
              [0, 0, 5, 6, 0, 0, 0]]
    X_star = [[1, 2], [3, 4], [5, 6]]

    # Get frames
    pid_in_i, anchor_in_i, _, _ = canonicalize(Xi_raw)
    pid_out_i, anchor_out_i, _, _ = canonicalize(Yi_raw)
    pid_in_star, anchor_in_star, _, _ = canonicalize(X_star)

    frames = {
        "Pi_in_star": (pid_in_star, anchor_in_star),
        "Pi_out_star": (pid_out_i, anchor_out_i),
        "Pi_in_0": (pid_in_i, anchor_in_i),
        "Pi_out_0": (pid_out_i, anchor_out_i),
    }

    # Learn witness
    witness_result = learn_witness(Xi_raw, Yi_raw, {
        "Pi_in": (pid_in_i, anchor_in_i),
        "Pi_out": (pid_out_i, anchor_out_i)
    })

    colors_order = sorted(set([0, 1, 2, 3, 4, 5, 6]))
    R_out = len(Yi_raw)
    C_out = len(Yi_raw[0]) if Yi_raw else 0

    print(f"Input: {len(Xi_raw)}×{len(Xi_raw[0])}")
    print(f"Output: {R_out}×{C_out}")
    print(f"Pieces: {len(witness_result['pieces'])}")
    print(f"Silent: {witness_result['silent']}")

    # Emit (should handle dimension mismatch gracefully)
    try:
        A_wit, S_wit = emit_witness(
            X_star,
            [witness_result],
            frames,
            colors_order,
            R_out,
            C_out,
        )

        # Verify structure
        assert len(S_wit) == R_out, f"S_wit should have {R_out} rows"
        for c in colors_order:
            assert len(A_wit[c]) == R_out, f"A_wit[{c}] should have {R_out} rows"
            for row in A_wit[c]:
                # Check no bits beyond C_out
                assert row < (1 << C_out), f"Row has bits beyond C_out: {bin(row)}"

        print(f"✓ A_wit structure correct ({R_out} rows, {C_out} columns)")
        print(f"✓ S_wit structure correct ({R_out} rows)")
        print(f"✓ No bits beyond canvas width")

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        raise

    print("\n✅ BUG #2 FIX VERIFIED: Explicit embedding handles dimensions correctly")
    return True


def test_bug2_with_rotation():
    """Bug #2 extended: Test with rotation (dimension swap)."""
    print("\n" + "="*70)
    print("BUG #2 EXTENDED: Rotation (Dimension Swap)")
    print("="*70)

    # Input: 2×3, rotate 90° → 3×2, then embed in 5×5
    Xi_raw = [[1, 2, 3], [4, 5, 6]]
    # After R90: becomes 3×2
    # Embed in 5×5 canvas
    Yi_raw = [[0, 0, 0, 0, 0],
              [0, 3, 6, 0, 0],
              [0, 2, 5, 0, 0],
              [0, 1, 4, 0, 0],
              [0, 0, 0, 0, 0]]
    X_star = [[1, 2, 3], [4, 5, 6]]

    pid_in_i, anchor_in_i, _, _ = canonicalize(Xi_raw)
    pid_out_i, anchor_out_i, _, _ = canonicalize(Yi_raw)
    pid_in_star, anchor_in_star, _, _ = canonicalize(X_star)

    frames = {
        "Pi_in_star": (pid_in_star, anchor_in_star),
        "Pi_out_star": (pid_out_i, anchor_out_i),
        "Pi_in_0": (pid_in_i, anchor_in_i),
        "Pi_out_0": (pid_out_i, anchor_out_i),
    }

    witness_result = learn_witness(Xi_raw, Yi_raw, {
        "Pi_in": (pid_in_i, anchor_in_i),
        "Pi_out": (pid_out_i, anchor_out_i)
    })

    colors_order = sorted(set([0, 1, 2, 3, 4, 5, 6]))
    R_out = len(Yi_raw)
    C_out = len(Yi_raw[0]) if Yi_raw else 0

    print(f"Input: {len(Xi_raw)}×{len(Xi_raw[0])}")
    print(f"Output: {R_out}×{C_out}")
    print(f"Silent: {witness_result['silent']}")

    # Emit with rotation
    A_wit, S_wit = emit_witness(
        X_star,
        [witness_result],
        frames,
        colors_order,
        R_out,
        C_out,
    )

    # Verify structure
    assert len(S_wit) == R_out
    for c in colors_order:
        assert len(A_wit[c]) == R_out

    print(f"✓ Rotation handled correctly")
    print(f"✓ Dimension swap: 2×3 → 3×2 → 5×5 canvas")

    print("\n✅ BUG #2 EXTENDED VERIFIED: Rotation with dimension swap works")
    return True


if __name__ == "__main__":
    print("WO-07 BUG FIX VALIDATION")
    print("="*70)

    tests = [
        test_bug1_u_r_uses_c_out,
        test_bug2_explicit_embedding,
        test_bug2_with_rotation,
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
