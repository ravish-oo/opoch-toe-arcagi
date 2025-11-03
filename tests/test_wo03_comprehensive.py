"""
WO-03 Frame Canonicalizer - Comprehensive Verification (Post-Update)

Tests mathematical equivalence, exclusivity, receipts, and spec conformance.
Uses receipts-first algebraic debugging (no internal state inspection).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import assert_double_run_equal, Receipts, blake3_hash
from arcbit.core.bytesio import serialize_grid_be_row_major, serialize_planes_be_row_major
from arcbit.kernel import (
    pack_grid_to_planes,
    unpack_planes_to_grid,
    order_colors,
    pose_plane,
    shift_plane,
)
from arcbit.kernel.frames import canonicalize, apply_pose_anchor


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
with open(arc_data_path, "r") as f:
    tasks = json.load(f)


def test_receipts_present_and_complete():
    """
    Verify canonicalize returns receipts with all required fields.

    Spec: WO-03 Receipts - frame.inputs, frame.pose, frame.anchor, frame.bytes.
    """
    print("\n" + "=" * 70)
    print("TEST: Receipts Present and Complete")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]

    pid, anchor, G_canon, receipts = canonicalize(grid)

    # Check required fields
    required_fields = {
        "frame.inputs": ["H", "W", "colors_order", "nonzero_count"],
        "frame.pose": ["pose_id", "pose_tie_count"],
        "frame.anchor": ["r", "c", "all_zero"],
        "frame.bytes": ["hash_before", "hash_after"]
    }

    all_present = True
    for section, fields in required_fields.items():
        if section not in receipts:
            print(f"  ❌ Missing section: {section}")
            all_present = False
        else:
            for field in fields:
                if field not in receipts[section]:
                    print(f"  ❌ Missing field: {section}.{field}")
                    all_present = False

    if all_present:
        print("  ✅ All required receipt fields present")
        print(f"  pose_id: {receipts['frame.pose']['pose_id']}")
        print(f"  pose_tie_count: {receipts['frame.pose']['pose_tie_count']}")
        print(f"  anchor: ({receipts['frame.anchor']['r']}, {receipts['frame.anchor']['c']})")
        print(f"  all_zero: {receipts['frame.anchor']['all_zero']}")
        print(f"  nonzero_count: {receipts['frame.inputs']['nonzero_count']}")
        return True
    else:
        print("  ❌ Some receipt fields missing")
        return False


def test_pose_tie_count_tracking():
    """
    Verify pose_tie_count correctly tracks number of poses with lex-min bytes.

    Spec: WO-03 Receipts - pose_tie_count >= 1, counts ties.
    """
    print("\n" + "=" * 70)
    print("TEST: pose_tie_count Tracking")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]

    pid, anchor, G_canon, receipts = canonicalize(grid)
    tie_count = receipts["frame.pose"]["pose_tie_count"]

    print(f"  pose_id: {pid}")
    print(f"  pose_tie_count: {tie_count}")

    # Manually verify by computing all pose byte streams
    H = len(grid)
    W = len(grid[0]) if grid else 0
    colors = order_colors({c for row in grid for c in row} | {0})
    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    pose_bytes = {}
    for p in pose_ids:
        planes = pack_grid_to_planes(grid, H, W, colors)
        planes_posed = {}
        for color in colors:
            plane = planes[color]
            plane_p, H_p, W_p = pose_plane(plane, p, H, W)
            planes_posed[color] = plane_p

        G_posed = unpack_planes_to_grid(planes_posed, H_p, W_p, colors)
        grid_bytes = serialize_grid_be_row_major(G_posed, H_p, W_p, colors)
        pose_bytes[p] = grid_bytes

    min_bytes = min(pose_bytes.values())
    ties = [p for p, b in pose_bytes.items() if b == min_bytes]
    expected_tie_count = len(ties)

    print(f"  Poses with lex-min: {ties}")
    print(f"  Expected tie_count: {expected_tie_count}")
    print(f"  Actual tie_count: {tie_count}")

    if tie_count == expected_tie_count and tie_count >= 1:
        print("  ✅ PASS: pose_tie_count correct")
        return True
    else:
        print("  ❌ FAIL: pose_tie_count mismatch")
        return False


def test_mathematical_equivalence():
    """
    Verify apply_pose_anchor(planes) is equivalent to grid-transform + repack.

    Spec: WO-03 apply_pose_anchor invariant - equivalent to transforming grid with zero-fill.
    """
    print("\n" + "=" * 70)
    print("TEST: Mathematical Equivalence (Grid vs Planes)")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    # Test multiple transformations
    test_cases = [
        ("R90", (0, 0)),
        ("R90", (1, 1)),
        ("FX", (0, 0)),
        ("R180", (1, 0)),
    ]

    all_match = True
    for pid, anchor in test_cases:
        print(f"\n  Testing: pid={pid}, anchor={anchor}")

        # Method 1: apply_pose_anchor on planes
        planes_copy = {c: list(p) for c, p in planes.items()}
        planes_transformed, H_out, W_out = apply_pose_anchor(
            planes_copy, pid, anchor, H, W, colors
        )

        # Method 2: Transform grid directly, then repack
        # Step 1: Pose grid
        planes_grid = pack_grid_to_planes(grid, H, W, colors)
        planes_grid_posed = {}
        for color in colors:
            plane = planes_grid[color]
            plane_posed, H_p, W_p = pose_plane(plane, pid, H, W)
            planes_grid_posed[color] = plane_posed

        G_posed = unpack_planes_to_grid(planes_grid_posed, H_p, W_p, colors)

        # Step 2: Translate grid with zero-fill
        from arcbit.kernel.frames import _translate_grid
        G_transformed = _translate_grid(G_posed, -anchor[0], -anchor[1])

        # Step 3: Repack to planes
        H_trans = len(G_transformed)
        W_trans = len(G_transformed[0]) if G_transformed else 0
        planes_expected = pack_grid_to_planes(G_transformed, H_trans, W_trans, colors)

        # Compare results
        shapes_match = (H_out == H_trans and W_out == W_trans)
        planes_match = all(
            planes_transformed[c] == planes_expected[c] for c in colors
        )

        if shapes_match and planes_match:
            print(f"    ✅ Shapes and planes match")
        else:
            print(f"    ❌ Mismatch detected")
            print(f"       Shapes: ({H_out}, {W_out}) vs ({H_trans}, {W_trans})")
            for c in colors:
                if planes_transformed[c] != planes_expected[c]:
                    print(f"       Color {c}: {planes_transformed[c]} != {planes_expected[c]}")
            all_match = False

    if all_match:
        print("\n  ✅ PASS: apply_pose_anchor mathematically equivalent to grid transform")
        return True
    else:
        print("\n  ❌ FAIL: Mathematical equivalence broken")
        return False


def test_exclusivity_invariant():
    """
    Verify exclusivity: union == full_mask, pairwise overlaps == 0.

    Spec: WO-03 apply_pose_anchor invariant - Exclusivity.
    """
    print("\n" + "=" * 70)
    print("TEST: Exclusivity Invariant")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    # Test multiple transformations
    test_cases = [
        ("I", (0, 0)),
        ("R90", (1, 1)),
        ("FX", (0, 0)),
    ]

    all_valid = True
    for pid, anchor in test_cases:
        print(f"\n  Testing: pid={pid}, anchor={anchor}")

        try:
            planes_copy = {c: list(p) for c, p in planes.items()}
            planes_transformed, H_out, W_out = apply_pose_anchor(
                planes_copy, pid, anchor, H, W, colors
            )

            # Exclusivity should pass (no exception raised)
            print(f"    ✅ Exclusivity validated (no exception)")

            # Double-check manually
            full_mask = (1 << W_out) - 1 if W_out > 0 else 0
            for r in range(H_out):
                union = 0
                for color in colors:
                    union |= planes_transformed[color][r]

                if union != full_mask:
                    print(f"    ❌ Union mismatch at row {r}: {union:b} != {full_mask:b}")
                    all_valid = False

        except ValueError as e:
            print(f"    ❌ Exclusivity violation: {e}")
            all_valid = False

    if all_valid:
        print("\n  ✅ PASS: Exclusivity invariant holds")
        return True
    else:
        print("\n  ❌ FAIL: Exclusivity violations detected")
        return False


def test_color0_complement_step():
    """
    Verify Step 3: color 0 plane is complement of non-zero union after shift.

    Spec: WO-03 apply_pose_anchor Step 3 - rebuild 0-plane as complement.
    """
    print("\n" + "=" * 70)
    print("TEST: Color 0 Complement Rebuild (Step 3)")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if grid else 0

    colors = order_colors({c for row in grid for c in row} | {0})
    planes = pack_grid_to_planes(grid, H, W, colors)

    pid = "R90"
    anchor = (1, 1)

    print(f"  Testing: pid={pid}, anchor={anchor}")
    print(f"  Original grid:")
    for row in grid:
        print(f"    {row}")

    planes_copy = {c: list(p) for c, p in planes.items()}
    planes_transformed, H_out, W_out = apply_pose_anchor(
        planes_copy, pid, anchor, H, W, colors
    )

    # Check color 0 is complement of non-zero union
    full_mask = (1 << W_out) - 1 if W_out > 0 else 0
    complement_correct = True

    for r in range(H_out):
        # Union of non-zero planes
        nonzero_union = 0
        for color in colors:
            if color != 0:
                nonzero_union |= planes_transformed[color][r]

        # Expected color 0: complement of nonzero_union
        expected_color0 = full_mask & ~nonzero_union
        actual_color0 = planes_transformed[0][r]

        if actual_color0 != expected_color0:
            print(f"    ❌ Row {r}: color0={actual_color0:b}, expected={expected_color0:b}")
            complement_correct = False

    if complement_correct:
        print(f"  ✅ PASS: Color 0 is complement of non-zero union")
        return True
    else:
        print(f"  ❌ FAIL: Color 0 complement incorrect")
        return False


def test_idempotence_via_receipts():
    """
    Verify idempotence using receipts (algebraic debugging).

    Spec: WO-03 Invariants - Idempotence.
    """
    print("\n" + "=" * 70)
    print("TEST: Idempotence via Receipts")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]

    # First canonicalization
    pid1, anchor1, G_canon1, receipts1 = canonicalize(grid)
    hash1 = receipts1["frame.bytes"]["hash_after"]

    # Second canonicalization (on canonical grid)
    pid2, anchor2, G_canon2, receipts2 = canonicalize(G_canon1)
    hash2 = receipts2["frame.bytes"]["hash_after"]

    print(f"  First:  pose={pid1}, anchor={anchor1}, hash={hash1}")
    print(f"  Second: pose={pid2}, anchor={anchor2}, hash={hash2}")

    # Idempotence: second should be identity
    idempotent = (pid2 == "I" and anchor2 == (0, 0) and hash1 == hash2)

    if idempotent:
        print("  ✅ PASS: Idempotence verified via receipts")
        return True
    else:
        print("  ❌ FAIL: Not idempotent")
        return False


def test_double_run_determinism_receipts():
    """
    Verify double-run produces identical receipts.

    Spec: WO-03 Invariants - Determinism.
    """
    print("\n" + "=" * 70)
    print("TEST: Double-Run Determinism (Receipts)")
    print("=" * 70)

    task = tasks["00576224"]
    grids = [ex["input"] for ex in task["train"]]

    def build_receipts():
        r = Receipts("test-wo03-determinism")

        for idx, grid in enumerate(grids):
            pid, anchor, G_canon, receipts = canonicalize(grid)

            r.put(f"grid_{idx}_pose_id", receipts["frame.pose"]["pose_id"])
            r.put(f"grid_{idx}_pose_tie_count", receipts["frame.pose"]["pose_tie_count"])
            r.put(f"grid_{idx}_anchor", (receipts["frame.anchor"]["r"], receipts["frame.anchor"]["c"]))
            r.put(f"grid_{idx}_all_zero", receipts["frame.anchor"]["all_zero"])
            r.put(f"grid_{idx}_hash_before", receipts["frame.bytes"]["hash_before"])
            r.put(f"grid_{idx}_hash_after", receipts["frame.bytes"]["hash_after"])

        return r

    try:
        assert_double_run_equal(build_receipts)
        print("  ✅ PASS: Double-run receipts identical")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: Double-run mismatch: {e}")
        return False


def test_no_minted_bits_nonzero():
    """
    Verify no non-zero colors are minted (only background can increase).

    Spec: WO-03 Invariants - No minted bits (non-zero colors).
    """
    print("\n" + "=" * 70)
    print("TEST: No Minted Bits (Non-Zero Colors)")
    print("=" * 70)

    task = tasks["00576224"]
    grid = task["train"][0]["input"]

    # Count non-zero cells
    nonzero_cells_orig = [(r, c, grid[r][c]) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != 0]
    nonzero_count_orig = len(nonzero_cells_orig)

    pid, anchor, G_canon, receipts = canonicalize(grid)

    # Count non-zero cells in canonical
    nonzero_count_canon = sum(1 for row in G_canon for val in row if val != 0)

    print(f"  Original non-zero cells: {nonzero_count_orig}")
    print(f"  Canonical non-zero cells: {nonzero_count_canon}")
    print(f"  Receipt nonzero_count: {receipts['frame.inputs']['nonzero_count']}")

    # Non-zero cells may decrease (crop) or stay same, but NEVER increase
    no_minted = nonzero_count_canon <= nonzero_count_orig

    if no_minted:
        print("  ✅ PASS: No non-zero colors minted")
        return True
    else:
        print("  ❌ FAIL: Non-zero colors minted!")
        return False


def run_all_tests():
    """Run all comprehensive WO-03 tests."""
    print("\n" + "=" * 70)
    print("WO-03 COMPREHENSIVE TEST SUITE (POST-UPDATE)")
    print("=" * 70)

    tests = [
        test_receipts_present_and_complete,
        test_pose_tie_count_tracking,
        test_mathematical_equivalence,
        test_exclusivity_invariant,
        test_color0_complement_step,
        test_idempotence_via_receipts,
        test_double_run_determinism_receipts,
        test_no_minted_bits_nonzero,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
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
