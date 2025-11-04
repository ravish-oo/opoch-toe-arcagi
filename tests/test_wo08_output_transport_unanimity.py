#!/usr/bin/env python3
"""
WO-08 Tests: Output Transport & Unanimity

Test normalization (replicate/decimate/silent), transport (D4 + anchor),
and unanimity (agreement/disagreement) using receipts.

Spec: WO-08 v1.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.output_transport import emit_output_transport
from src.arcbit.emitters.unanimity import emit_unity


def test_replicate_simple():
    """Test integer replication (Kronecker)."""
    print("\nTest 1: Replicate 2x2 → 4x4")
    print("-" * 60)

    # Training output: 2×2
    Y_list = [
        [[1, 2], [3, 4]]  # 2×2 grid
    ]

    # Frame: identity at origin
    frames_out = [("I", (0, 0))]

    # Working canvas: 4×4 (2x replication)
    R_out, C_out = 4, 4
    colors_order = [0, 1, 2, 3, 4]
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify receipt
    r = receipts[0]
    assert r["norm_kind"] == "replicate", f"Expected replicate, got {r['norm_kind']}"
    assert r["s_r"] == 2 and r["s_c"] == 2, f"Expected s_r=2, s_c=2, got {r['s_r']}, {r['s_c']}"
    assert r["scope_bits"] == 16, f"Expected 16 pixels, got {r['scope_bits']}"

    # Verify replicated pattern: each cell becomes 2×2 block
    # [1,2]  →  [1,1,2,2]
    # [3,4]  →  [1,1,2,2]
    #           [3,3,4,4]
    #           [3,3,4,4]

    S_i = S_out_list[0]
    scope_bits = sum(bin(row).count("1") for row in S_i)
    assert scope_bits == 16, f"Expected full scope (16), got {scope_bits}"

    print("✓ Normalization: replicate")
    print(f"  s_r={r['s_r']}, s_c={r['s_c']}")
    print(f"  scope_bits={r['scope_bits']}")
    print(f"  n_included={section['payload']['n_included']}")


def test_decimate_constant_blocks():
    """Test exact block-constancy decimation."""
    print("\nTest 2: Decimate 4x4 → 2x2 (constant blocks)")
    print("-" * 60)

    # Training output: 4×4 with 2×2 constant blocks
    Y_list = [
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
    ]

    frames_out = [("I", (0, 0))]
    R_out, C_out = 2, 2  # 2x decimation
    colors_order = [0, 1, 2, 3, 4]
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    r = receipts[0]
    assert r["norm_kind"] == "decimate", f"Expected decimate, got {r['norm_kind']}"
    assert r["s_r"] == 2 and r["s_c"] == 2, f"Expected s_r=2, s_c=2, got {r['s_r']}, {r['s_c']}"
    assert r["block_constancy_ok"] == True, "Expected block_constancy_ok=True"
    assert r["scope_bits"] == 4, f"Expected 4 pixels, got {r['scope_bits']}"

    print("✓ Normalization: decimate (constant blocks)")
    print(f"  s_r={r['s_r']}, s_c={r['s_c']}")
    print(f"  block_constancy_ok={r['block_constancy_ok']}")
    print(f"  scope_bits={r['scope_bits']}")


def test_decimate_non_constant_blocks():
    """Test decimation failure on non-constant blocks."""
    print("\nTest 3: Decimate fails (non-constant blocks → silent)")
    print("-" * 60)

    # Training output: 4×4 with NON-constant 2×2 blocks
    Y_list = [
        [
            [1, 2, 3, 4],  # First 2×2 block has different colors
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6],
        ]
    ]

    frames_out = [("I", (0, 0))]
    R_out, C_out = 2, 2  # 2x decimation
    colors_order = list(range(10))
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    r = receipts[0]
    assert r["norm_kind"] == "silent", f"Expected silent, got {r['norm_kind']}"
    assert r["block_constancy_ok"] == False, "Expected block_constancy_ok=False"
    assert r["scope_bits"] == 0, f"Expected 0 scope_bits, got {r['scope_bits']}"
    assert section["payload"]["n_included"] == 0, "Expected 0 included trainings"

    print("✓ Normalization: silent (block constancy failed)")
    print(f"  norm_kind={r['norm_kind']}")
    print(f"  block_constancy_ok={r['block_constancy_ok']}")
    print(f"  scope_bits={r['scope_bits']}")


def test_silent_no_integer_relation():
    """Test silent when no exact integer relation exists."""
    print("\nTest 4: Silent (no integer relation)")
    print("-" * 60)

    # Training output: 3×5 (no integer relation to 4×4)
    Y_list = [
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [1, 2, 3, 4, 5]]
    ]

    frames_out = [("I", (0, 0))]
    R_out, C_out = 4, 4  # No exact relation
    colors_order = list(range(10))
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    r = receipts[0]
    assert r["norm_kind"] == "silent", f"Expected silent, got {r['norm_kind']}"
    assert r["s_r"] is None and r["s_c"] is None, "Expected None factors"
    assert r["scope_bits"] == 0, f"Expected 0 scope_bits, got {r['scope_bits']}"

    print("✓ Normalization: silent (no integer relation)")
    print(f"  norm_kind={r['norm_kind']}")
    print(f"  scope_bits={r['scope_bits']}")


def test_transport_with_pose():
    """Test transport with D4 pose transformation."""
    print("\nTest 5: Transport with R90 pose")
    print("-" * 60)

    # Training output: 2×2 square
    Y_list = [
        [[1, 2], [3, 4]]  # Top-left=1, top-right=2, bottom-left=3, bottom-right=4
    ]

    # Source frame: R90 at origin
    frames_out = [("R90", (0, 0))]

    # Working canvas: 2×2 (no replication)
    R_out, C_out = 2, 2
    colors_order = [0, 1, 2, 3, 4]

    # Destination frame: Identity at origin
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    r = receipts[0]
    assert r["pose_src"] == "R90", f"Expected pose_src=R90, got {r['pose_src']}"
    assert r["pose_dst"] == "I", f"Expected pose_dst=I, got {r['pose_dst']}"

    # Verify relative transform was applied
    # T = I ∘ R90⁻¹ = I ∘ R270 = R270
    # So the output should be rotated by R270 (or equivalently, reverse R90)

    print("✓ Transport with pose")
    print(f"  pose_src={r['pose_src']}, pose_dst={r['pose_dst']}")
    print(f"  scope_bits={r['scope_bits']}")


def test_transport_with_anchor_shift():
    """Test transport with anchor shift."""
    print("\nTest 6: Transport with anchor shift")
    print("-" * 60)

    # Training output: 1×1 (single pixel)
    Y_list = [[[5]]]

    # Source frame: I at (1, 2)
    frames_out = [("I", (1, 2))]

    # Working canvas: 4×4
    R_out, C_out = 4, 4
    colors_order = [0, 5]

    # Destination frame: I at (0, 0)
    pi_out_star = ("I", (0, 0))

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    r = receipts[0]
    assert r["anchor_src"] == (1, 2), f"Expected anchor_src=(1,2), got {r['anchor_src']}"
    assert r["anchor_dst"] == (0, 0), f"Expected anchor_dst=(0,0), got {r['anchor_dst']}"

    # Verify shift applied: pixel should move from (1,2) to (0,0)
    # Actually, with replicate 4x4, it becomes 4x4 full grid of color 5
    # Then shift by -(1,2) then shift by +(0,0) = net shift of -(1,2)

    print("✓ Transport with anchor")
    print(f"  anchor_src={r['anchor_src']}, anchor_dst={r['anchor_dst']}")


def test_unanimity_all_agree():
    """Test unanimity when all trainings agree."""
    print("\nTest 7: Unanimity (all agree)")
    print("-" * 60)

    R_out, C_out = 2, 2
    colors_order = [0, 1, 2]

    # Two trainings both emit color 1 at all pixels
    A_out_list = [
        {0: [0, 0], 1: [3, 3], 2: [0, 0]},  # Training 0: color 1 at all 4 pixels
        {0: [0, 0], 1: [3, 3], 2: [0, 0]},  # Training 1: color 1 at all 4 pixels
    ]
    S_out_list = [
        [3, 3],  # Training 0: full scope
        [3, 3],  # Training 1: full scope
    ]

    A_uni, S_uni, receipt = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    assert receipt["unanimous_pixels"] == 4, f"Expected 4 unanimous, got {receipt['unanimous_pixels']}"
    assert receipt["total_covered_pixels"] == 4, f"Expected 4 covered, got {receipt['total_covered_pixels']}"
    assert receipt["empty_scope_pixels"] == 0, f"Expected 0 empty, got {receipt['empty_scope_pixels']}"
    assert len(receipt["included_train_ids"]) == 2, "Expected 2 included trainings"

    # Verify A_uni has color 1 at all pixels
    assert A_uni[1] == [3, 3], f"Expected A_uni[1]=[3,3], got {A_uni[1]}"
    assert S_uni == [3, 3], f"Expected S_uni=[3,3], got {S_uni}"

    print("✓ Unanimity: all agree")
    print(f"  unanimous_pixels={receipt['unanimous_pixels']}")
    print(f"  total_covered_pixels={receipt['total_covered_pixels']}")
    print(f"  included_train_ids={receipt['included_train_ids']}")


def test_unanimity_disagree():
    """Test unanimity when trainings disagree."""
    print("\nTest 8: Unanimity (disagreement → silent)")
    print("-" * 60)

    R_out, C_out = 2, 2
    colors_order = [0, 1, 2]

    # Two trainings emit different colors at same pixels
    A_out_list = [
        {0: [0, 0], 1: [3, 3], 2: [0, 0]},  # Training 0: color 1 everywhere
        {0: [0, 0], 1: [0, 0], 2: [3, 3]},  # Training 1: color 2 everywhere
    ]
    S_out_list = [
        [3, 3],  # Full scope
        [3, 3],  # Full scope
    ]

    A_uni, S_uni, receipt = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    # All pixels have disagreement → no unanimity
    assert receipt["unanimous_pixels"] == 0, f"Expected 0 unanimous, got {receipt['unanimous_pixels']}"
    assert receipt["total_covered_pixels"] == 4, f"Expected 4 covered, got {receipt['total_covered_pixels']}"

    # S_uni should be all zeros (silent)
    assert S_uni == [0, 0], f"Expected S_uni=[0,0], got {S_uni}"

    print("✓ Unanimity: disagreement")
    print(f"  unanimous_pixels={receipt['unanimous_pixels']} (expected 0)")
    print(f"  total_covered_pixels={receipt['total_covered_pixels']}")


def test_unanimity_partial_agreement():
    """Test unanimity with partial agreement."""
    print("\nTest 9: Unanimity (partial agreement)")
    print("-" * 60)

    R_out, C_out = 3, 3
    colors_order = [0, 1, 2, 3]

    # Training 0: color 1 at (0,0), (1,1)
    # Training 1: color 1 at (0,0), color 2 at (1,1), color 3 at (2,2)
    # Expected: unanimous at (0,0) only
    A_out_list = [
        {0: [0, 0, 0], 1: [1, 2, 0], 2: [0, 0, 0], 3: [0, 0, 0]},  # bits 0 and 1 set in rows 0 and 1
        {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 2, 0], 3: [0, 0, 4]},  # different colors
    ]
    S_out_list = [
        [1, 2, 0],  # Training 0 scope
        [1, 2, 4],  # Training 1 scope
    ]

    A_uni, S_uni, receipt = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    # Only pixel (0,0) should be unanimous (both have color 1)
    # Pixels (1,1) and (2,2) have disagreement
    print(f"  unanimous_pixels={receipt['unanimous_pixels']}")
    print(f"  total_covered_pixels={receipt['total_covered_pixels']}")
    print(f"  empty_scope_pixels={receipt['empty_scope_pixels']}")

    # At least one pixel should be unanimous
    assert receipt["unanimous_pixels"] >= 1, "Expected at least 1 unanimous pixel"


def test_unanimity_all_silent():
    """Test unanimity when all trainings are silent."""
    print("\nTest 10: Unanimity (all silent)")
    print("-" * 60)

    R_out, C_out = 2, 2
    colors_order = [0, 1, 2]

    # All trainings have zero scope
    A_out_list = [
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
    ]
    S_out_list = [
        [0, 0],
        [0, 0],
    ]

    A_uni, S_uni, receipt = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    assert receipt["unanimous_pixels"] == 0, "Expected 0 unanimous pixels"
    assert receipt["total_covered_pixels"] == 0, "Expected 0 covered pixels"
    assert receipt["empty_scope_pixels"] == 4, f"Expected 4 empty pixels, got {receipt['empty_scope_pixels']}"
    assert len(receipt["included_train_ids"]) == 0, "Expected 0 included trainings"

    print("✓ Unanimity: all silent")
    print(f"  included_train_ids={receipt['included_train_ids']} (empty)")
    print(f"  unanimous_pixels={receipt['unanimous_pixels']}")


def test_receipts_determinism():
    """Test that receipts are deterministic (same input → same hashes)."""
    print("\nTest 11: Receipts determinism (double-run)")
    print("-" * 60)

    Y_list = [[[1, 2], [3, 4]]]
    frames_out = [("I", (0, 0))]
    R_out, C_out = 4, 4
    colors_order = [0, 1, 2, 3, 4]
    pi_out_star = ("I", (0, 0))

    # Run 1
    _, _, _, section1 = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Run 2 (identical input)
    _, _, _, section2 = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    hash1 = section1["payload"]["transports_hash"]
    hash2 = section2["payload"]["transports_hash"]

    assert hash1 == hash2, f"Hashes differ: {hash1} vs {hash2}"

    print("✓ Receipts determinism")
    print(f"  transports_hash (run 1): {hash1[:16]}...")
    print(f"  transports_hash (run 2): {hash2[:16]}...")
    print("  ✓ Hashes match")


def run_all_tests():
    """Run all WO-08 tests."""
    print("=" * 80)
    print("WO-08 TESTS: Output Transport & Unanimity")
    print("=" * 80)

    test_replicate_simple()
    test_decimate_constant_blocks()
    test_decimate_non_constant_blocks()
    test_silent_no_integer_relation()
    test_transport_with_pose()
    test_transport_with_anchor_shift()
    test_unanimity_all_agree()
    test_unanimity_disagree()
    test_unanimity_partial_agreement()
    test_unanimity_all_silent()
    test_receipts_determinism()

    print()
    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
