"""
WO-01 VERIFICATION SUITE - Real ARC Data, Receipts-First

Complete test coverage for WO-01 kernel operations:
  ✓ PACK/UNPACK roundtrip on 3 real ARC tasks
  ✓ POSE inverse mapping (all 8 D4 transforms)
  ✓ SHIFT boundary behavior (zero-fill, no wrap)
  ✓ PERIOD on canonical patterns (KMP correctness)
  ✓ Bitwise ops (AND/OR/ANDN)
  ✓ Edge cases (H=0, W=0, non-square)
  ✓ kernel_receipts() completeness

Verification: ALGEBRAIC (receipts and hashes only).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import (
    Receipts,
    blake3_hash,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
    assert_double_run_equal,
)
from arcbit.kernel import (
    order_colors,
    pack_grid_to_planes,
    unpack_planes_to_grid,
    shift_plane,
    pose_plane,
    pose_inverse,
    plane_and,
    plane_or,
    plane_andn,
    minimal_period_row,
    kernel_receipts,
)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: PACK/UNPACK Roundtrip on Real ARC Tasks
# ═══════════════════════════════════════════════════════════════════════

def test_pack_unpack_roundtrip_real_arc():
    """
    Load 3 real ARC tasks, pack to planes, unpack back, verify exact match.

    Spec: WO-01 section 2 invariant: unpack(pack(G)) == G
    """

    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    # Pick first 3 tasks
    task_ids = list(tasks.keys())[:3]

    receipts = Receipts("test-pack-unpack-roundtrip")
    roundtrips = []

    for task_id in task_ids:
        # Use first training input
        grid_orig = tasks[task_id]["train"][0]["input"]
        H = len(grid_orig)
        W = len(grid_orig[0]) if H > 0 else 0

        # Build color universe
        colors_set = {0}
        for row in grid_orig:
            colors_set.update(row)
        colors_order = order_colors(colors_set)

        # PACK
        planes = pack_grid_to_planes(grid_orig, H, W, colors_order)

        # UNPACK
        grid_recon = unpack_planes_to_grid(planes, H, W, colors_order)

        # Algebraic check: serialize both and compare hashes
        grid_orig_bytes = serialize_grid_be_row_major(grid_orig, H, W, colors_order)
        grid_recon_bytes = serialize_grid_be_row_major(grid_recon, H, W, colors_order)

        hash_orig = blake3_hash(grid_orig_bytes)
        hash_recon = blake3_hash(grid_recon_bytes)

        roundtrips.append({
            "task_id": task_id,
            "H": H,
            "W": W,
            "colors_count": len(colors_order),
            "hash_original": hash_orig,
            "hash_reconstructed": hash_recon,
            "roundtrip_ok": hash_orig == hash_recon
        })

    receipts.put("roundtrips", roundtrips)
    receipts.put("all_ok", all(rt["roundtrip_ok"] for rt in roundtrips))

    digest = receipts.digest()

    assert digest["payload"]["all_ok"], (
        f"PACK/UNPACK roundtrip failed on some tasks:\n"
        f"{json.dumps(roundtrips, indent=2)}"
    )

    print(f"✅ PASS: PACK/UNPACK roundtrip on {len(task_ids)} tasks")
    for rt in roundtrips:
        print(f"  {rt['task_id']}: {rt['H']}×{rt['W']}, {rt['colors_count']} colors")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: POSE Inverse Mapping (All 8 D4 Transforms)
# ═══════════════════════════════════════════════════════════════════════

def test_pose_inverse_all_transforms():
    """
    Verify pose(pose(plane, pid), inv(pid)) == plane for all 8 D4 transforms.

    Uses real ARC task grid.

    Spec: WO-01 section 4 invariant.
    """

    # Load real ARC task
    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]["input"]
    H, W = len(grid), len(grid[0]) if len(grid) > 0 else 0

    colors_set = {0} | {c for row in grid for c in row}
    colors_order = order_colors(colors_set)

    planes = pack_grid_to_planes(grid, H, W, colors_order)

    # Test all 8 pose IDs
    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    receipts = Receipts("test-pose-inverse")
    tests = []

    for pid in pose_ids:
        # Pick first color's plane
        plane_orig = planes[colors_order[0]]

        # Apply pose
        plane_fwd, H_fwd, W_fwd = pose_plane(plane_orig, pid, H, W)

        # Apply inverse
        inv_pid = pose_inverse(pid)
        plane_back, H_back, W_back = pose_plane(plane_fwd, inv_pid, H_fwd, W_fwd)

        # Algebraic checks
        dims_ok = (H_back == H and W_back == W)
        plane_ok = (plane_back == plane_orig)

        tests.append({
            "pid": pid,
            "inv_pid": inv_pid,
            "H_orig": H,
            "W_orig": W,
            "H_fwd": H_fwd,
            "W_fwd": W_fwd,
            "H_back": H_back,
            "W_back": W_back,
            "dims_ok": dims_ok,
            "plane_ok": plane_ok,
            "ok": dims_ok and plane_ok
        })

    receipts.put("pose_inverse_tests", tests)
    receipts.put("all_ok", all(t["ok"] for t in tests))

    digest = receipts.digest()

    assert digest["payload"]["all_ok"], (
        f"POSE inverse failed:\n{json.dumps(tests, indent=2)}"
    )

    print(f"✅ PASS: POSE inverse for all 8 transforms")

    # Specifically verify FXR90 and FXR270 are self-inverse
    fxr90_test = next(t for t in tests if t["pid"] == "FXR90")
    fxr270_test = next(t for t in tests if t["pid"] == "FXR270")

    assert fxr90_test["inv_pid"] == "FXR90", "FXR90 must be self-inverse"
    assert fxr270_test["inv_pid"] == "FXR270", "FXR270 must be self-inverse"

    print(f"  ✓ FXR90 self-inverse: {fxr90_test['ok']}")
    print(f"  ✓ FXR270 self-inverse: {fxr270_test['ok']}")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: SHIFT Boundary Behavior (Zero-Fill)
# ═══════════════════════════════════════════════════════════════════════

def test_shift_boundary_zero_fill():
    """
    Verify SHIFT drops bits at edges and zero-fills.

    Spec: WO-01 section 3 - zero-fill only, no wrap.
    """

    # Create a known pattern: all 1s
    H, W = 4, 6
    plane_all_ones = [(1 << W) - 1 for _ in range(H)]  # All bits set

    initial_bits = H * W

    receipts = Receipts("test-shift-boundary")

    # Shift right by 1 (should drop rightmost column = H bits)
    plane_r1 = shift_plane(plane_all_ones, 0, 1, H, W)
    bits_r1 = sum(bin(mask).count('1') for mask in plane_r1)
    dropped_r1 = initial_bits - bits_r1

    # Shift down by 1 (should drop bottom row = W bits)
    plane_d1 = shift_plane(plane_all_ones, 1, 0, H, W)
    bits_d1 = sum(bin(mask).count('1') for mask in plane_d1)
    dropped_d1 = initial_bits - bits_d1

    # Shift left by 1 (should drop leftmost column = H bits)
    plane_l1 = shift_plane(plane_all_ones, 0, -1, H, W)
    bits_l1 = sum(bin(mask).count('1') for mask in plane_l1)
    dropped_l1 = initial_bits - bits_l1

    # Shift up by 1 (should drop top row = W bits)
    plane_u1 = shift_plane(plane_all_ones, -1, 0, H, W)
    bits_u1 = sum(bin(mask).count('1') for mask in plane_u1)
    dropped_u1 = initial_bits - bits_u1

    receipts.put("initial_bits", initial_bits)
    receipts.put("shift_right_1_dropped", dropped_r1)
    receipts.put("shift_down_1_dropped", dropped_d1)
    receipts.put("shift_left_1_dropped", dropped_l1)
    receipts.put("shift_up_1_dropped", dropped_u1)

    # Expected: right/left drop H bits, down/up drop W bits
    receipts.put("right_expected", H)
    receipts.put("down_expected", W)
    receipts.put("left_expected", H)
    receipts.put("up_expected", W)

    receipts.put("all_ok",
        dropped_r1 == H and dropped_d1 == W and
        dropped_l1 == H and dropped_u1 == W
    )

    digest = receipts.digest()

    assert digest["payload"]["all_ok"], (
        f"SHIFT boundary behavior incorrect:\n"
        f"  Right dropped {dropped_r1}, expected {H}\n"
        f"  Down dropped {dropped_d1}, expected {W}\n"
        f"  Left dropped {dropped_l1}, expected {H}\n"
        f"  Up dropped {dropped_u1}, expected {W}"
    )

    print(f"✅ PASS: SHIFT boundary (zero-fill, no wrap)")
    print(f"  Right/Left drop {H} bits, Down/Up drop {W} bits ✓")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: PERIOD on Canonical Patterns (KMP Correctness)
# ═══════════════════════════════════════════════════════════════════════

def test_period_canonical_patterns():
    """
    Test minimal_period_row on canonical bitstrings.

    Spec: WO-01 section 6 - exact KMP algorithm.
    """

    receipts = Receipts("test-period-kmp")

    test_cases = [
        {"label": "solid_0", "mask": 0b000000, "W": 6, "expected": None},
        {"label": "solid_1", "mask": 0b111111, "W": 6, "expected": None},  # Fixed: period=1 excluded
        {"label": "stripe_2", "mask": 0b101010, "W": 6, "expected": 2},
        {"label": "stripe_3", "mask": 0b110110, "W": 6, "expected": 3},
        {"label": "no_period", "mask": 0b101011, "W": 6, "expected": None},
        {"label": "stripe_2_offset", "mask": 0b010101, "W": 6, "expected": 2},
        {"label": "period_4", "mask": 0b10111011, "W": 8, "expected": 4},
    ]

    results = []
    for tc in test_cases:
        p = minimal_period_row(tc["mask"], tc["W"])
        match = (p == tc["expected"])

        results.append({
            "label": tc["label"],
            "mask": bin(tc["mask"]),
            "W": tc["W"],
            "expected": tc["expected"],
            "actual": p,
            "ok": match
        })

    receipts.put("period_tests", results)
    receipts.put("all_ok", all(r["ok"] for r in results))

    digest = receipts.digest()

    assert digest["payload"]["all_ok"], (
        f"PERIOD KMP incorrect:\n{json.dumps(results, indent=2)}"
    )

    print(f"✅ PASS: PERIOD on canonical patterns")
    for r in results:
        print(f"  {r['label']}: period={r['actual']} ✓")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Bitwise Ops (AND/OR/ANDN)
# ═══════════════════════════════════════════════════════════════════════

def test_bitwise_ops():
    """
    Test plane_and, plane_or, plane_andn.

    Spec: WO-01 section 5.
    """

    H, W = 2, 4

    # Planes: a = 0b1010, 0b0101
    #         b = 0b1100, 0b0011
    plane_a = [0b1010, 0b0101]
    plane_b = [0b1100, 0b0011]

    # AND: 0b1000, 0b0001
    result_and = plane_and(plane_a, plane_b, H, W)
    expected_and = [0b1000, 0b0001]

    # OR: 0b1110, 0b0111
    result_or = plane_or(plane_a, plane_b, H, W)
    expected_or = [0b1110, 0b0111]

    # ANDN (a & ~b): 0b0010, 0b0100
    result_andn = plane_andn(plane_a, plane_b, H, W)
    expected_andn = [0b0010, 0b0100]

    receipts = Receipts("test-bitwise")
    receipts.put("and_ok", result_and == expected_and)
    receipts.put("or_ok", result_or == expected_or)
    receipts.put("andn_ok", result_andn == expected_andn)
    receipts.put("all_ok",
        result_and == expected_and and
        result_or == expected_or and
        result_andn == expected_andn
    )

    digest = receipts.digest()

    assert digest["payload"]["all_ok"], "Bitwise ops failed"

    print(f"✅ PASS: Bitwise ops (AND/OR/ANDN)")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Edge Cases (H=0, W=0, Non-Square)
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    """
    Test edge cases: empty grids, non-square, single pixel.

    Spec: WO-01 edge cases section.
    """

    # Empty grid (H=0)
    grid_empty = []
    planes_empty = pack_grid_to_planes(grid_empty, 0, 0, [0])
    grid_empty_recon = unpack_planes_to_grid(planes_empty, 0, 0, [0])
    assert grid_empty == grid_empty_recon

    # Single pixel
    grid_single = [[5]]
    planes_single = pack_grid_to_planes(grid_single, 1, 1, [0, 5])
    grid_single_recon = unpack_planes_to_grid(planes_single, 1, 1, [0, 5])
    assert grid_single == grid_single_recon

    # Non-square (tall)
    grid_tall = [[1], [2], [3], [4]]
    planes_tall = pack_grid_to_planes(grid_tall, 4, 1, [0, 1, 2, 3, 4])
    grid_tall_recon = unpack_planes_to_grid(planes_tall, 4, 1, [0, 1, 2, 3, 4])
    assert grid_tall == grid_tall_recon

    # Non-square (wide)
    grid_wide = [[1, 2, 3, 4]]
    planes_wide = pack_grid_to_planes(grid_wide, 1, 4, [0, 1, 2, 3, 4])
    grid_wide_recon = unpack_planes_to_grid(planes_wide, 1, 4, [0, 1, 2, 3, 4])
    assert grid_wide == grid_wide_recon

    # SHIFT on empty
    plane_empty = shift_plane([], 1, 1, 0, 0)
    assert plane_empty == []

    # POSE on empty
    plane_empty_posed, H_posed, W_posed = pose_plane([], "R90", 0, 0)
    assert plane_empty_posed == []

    print(f"✅ PASS: Edge cases (H=0, W=0, non-square, single pixel)")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: kernel_receipts() Completeness
# ═══════════════════════════════════════════════════════════════════════

def test_kernel_receipts_completeness():
    """
    Verify kernel_receipts() generates all required receipt keys.

    Spec: WO-01 receipts section.
    """

    # Use real ARC fixture
    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    colors_set = {0} | {c for row in grid for c in row}
    colors_order = order_colors(colors_set)

    fixtures = [{
        "grid": grid,
        "H": H,
        "W": W,
        "colors": colors_order,
        "label": f"task_{task_id}"
    }]

    receipt_digest = kernel_receipts("WO-01-test", fixtures)

    # Required keys
    required_keys = {
        "kernel.params_hash",
        "pack_consistency",
        "pose_inverse_ok",
        "pose_inverse_tests",
        "shift_boundary_counts",
        "period_kmp_examples"
    }

    actual_keys = set(receipt_digest["payload"].keys())

    receipts = Receipts("test-kernel-receipts-completeness")
    receipts.put("required_keys", sorted(required_keys))
    receipts.put("actual_keys", sorted(actual_keys))
    receipts.put("missing", sorted(required_keys - actual_keys))
    receipts.put("extra", sorted(actual_keys - required_keys))
    receipts.put("keys_match", actual_keys >= required_keys)

    digest = receipts.digest()

    assert digest["payload"]["keys_match"], (
        f"kernel_receipts() missing keys: {digest['payload']['missing']}"
    )

    print(f"✅ PASS: kernel_receipts() completeness")
    print(f"  kernel.params_hash: {receipt_digest['payload']['kernel.params_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Double-Run Determinism
# ═══════════════════════════════════════════════════════════════════════

def test_double_run_determinism():
    """
    Verify kernel_receipts() produces identical hashes across double-run.

    Spec: WO-01 determinism requirement.
    """

    # Use real ARC fixture
    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]["input"]
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    colors_set = {0} | {c for row in grid for c in row}
    colors_order = order_colors(colors_set)

    fixtures = [{
        "grid": grid,
        "H": H,
        "W": W,
        "colors": colors_order,
        "label": f"task_{task_id}"
    }]

    def build_kernel_receipts():
        # Wrap kernel_receipts in a Receipts object for double-run check
        digest = kernel_receipts("WO-01-double-run", fixtures)

        r = Receipts("test-double-run-wrapper")
        r.put("kernel_section_hash", digest["section_hash"])
        return r

    assert_double_run_equal(build_kernel_receipts)

    print(f"✅ PASS: Double-run determinism on kernel_receipts()")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("WO-01 VERIFICATION SUITE - Real ARC Data, Receipts-First")
    print("=" * 70)

    tests = [
        ("PACK/UNPACK Roundtrip (3 Real ARC Tasks)", test_pack_unpack_roundtrip_real_arc),
        ("POSE Inverse (All 8 D4 Transforms)", test_pose_inverse_all_transforms),
        ("SHIFT Boundary (Zero-Fill)", test_shift_boundary_zero_fill),
        ("PERIOD (Canonical Patterns)", test_period_canonical_patterns),
        ("Bitwise Ops (AND/OR/ANDN)", test_bitwise_ops),
        ("Edge Cases (H=0, W=0, Non-Square)", test_edge_cases),
        ("kernel_receipts() Completeness", test_kernel_receipts_completeness),
        ("Double-Run Determinism", test_double_run_determinism),
    ]

    passed, failed = 0, 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 70)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
