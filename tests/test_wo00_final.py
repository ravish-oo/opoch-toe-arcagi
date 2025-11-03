"""
WO-00 FINAL VERIFICATION SUITE - Real ARC Data, Receipts-First

Complete test coverage for WO-00 components:
  ✓ param_registry() completeness and frozen values
  ✓ Double-run determinism on real ARC tasks
  ✓ Byte-level sensitivity (flip pixel → hash changes)
  ✓ Grid and Planes both deterministic (different iteration orders OK)
  ✓ Planes roundtrip reconstruction
  ✓ param_registry_hash binding
  ✓ Edge cases (empty, all-zero, color 0, unaligned width)
  ✓ Receipts type validation (floats forbidden)

Verification: ALGEBRAIC (receipts and hashes only, no internal state checks).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import (
    param_registry,
    blake3_hash,
    order_colors,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
    Receipts,
    assert_double_run_equal,
    RegistryError,
    SerializationError,
    ReceiptError,
    DeterminismError,
)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: param_registry() Completeness
# ═══════════════════════════════════════════════════════════════════════

def test_param_registry_completeness():
    reg = param_registry()
    required = {
        "spec_version", "endianness", "pose_order", "ac3_neighbor",
        "ac3_queue_order", "engine_priority", "bottom_color",
        "period_phase_origin", "strict_downscale", "hash_algo",
        "color_ordering", "byte_frame_tags"
    }
    actual = set(reg.keys())

    receipts = Receipts("test-param-registry-completeness")
    receipts.put("required_keys", sorted(required))
    receipts.put("actual_keys", sorted(actual))
    receipts.put("missing", sorted(required - actual))
    receipts.put("extra", sorted(actual - required))
    receipts.put("keys_match", actual == required)
    digest = receipts.digest()

    assert digest["payload"]["keys_match"], f"Key mismatch: {digest['payload']}"
    assert reg["spec_version"] == "1.5"
    assert reg["endianness"] == "BE"
    assert reg["bottom_color"] == 0
    assert reg["strict_downscale"] is True
    assert len(reg["pose_order"]) == 8
    assert reg["pose_order"] == ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    print(f"✅ PASS: param_registry() - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Double-Run on Real ARC Data
# ═══════════════════════════════════════════════════════════════════════

def test_double_run_real_arc():
    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]["input"]
    H, W = len(grid), len(grid[0]) if grid else 0
    colors_order = order_colors({0} | {c for row in grid for c in row})

    def build_receipts():
        r = Receipts("test-double-run-arc")
        grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)
        r.put("task_id", task_id)
        r.put("grid_hash", blake3_hash(grid_bytes))
        return r

    assert_double_run_equal(build_receipts)
    print(f"✅ PASS: Double-run on task {task_id}")


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Byte Sensitivity (Flip Pixel)
# ═══════════════════════════════════════════════════════════════════════

def test_byte_sensitivity():
    grid_orig = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    grid_flip = [row[:] for row in grid_orig]
    grid_flip[1][1] = 0

    H, W = 3, 3
    colors_order = order_colors({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

    hash_orig = blake3_hash(serialize_grid_be_row_major(grid_orig, H, W, colors_order))
    hash_flip = blake3_hash(serialize_grid_be_row_major(grid_flip, H, W, colors_order))

    receipts = Receipts("test-byte-sensitivity")
    receipts.put("hash_original", hash_orig)
    receipts.put("hash_flipped", hash_flip)
    receipts.put("hashes_differ", hash_orig != hash_flip)
    digest = receipts.digest()

    assert digest["payload"]["hashes_differ"], "Flip pixel must change hash"
    print(f"✅ PASS: Byte sensitivity - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Grid & Planes Both Deterministic
# ═══════════════════════════════════════════════════════════════════════

def test_grid_planes_deterministic():
    grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    H, W = 3, 3
    colors_order = [0, 1]

    def build_grid_receipt():
        r = Receipts("test-grid")
        r.put("hash", blake3_hash(serialize_grid_be_row_major(grid, H, W, colors_order)))
        return r

    assert_double_run_equal(build_grid_receipt)

    # Build planes
    planes = {}
    for c in colors_order:
        planes[c] = [sum((1 << col) for col in range(W) if grid[r][col] == c) for r in range(H)]

    def build_planes_receipt():
        r = Receipts("test-planes")
        r.put("hash", blake3_hash(serialize_planes_be_row_major(planes, H, W, colors_order)))
        return r

    assert_double_run_equal(build_planes_receipt)
    print(f"✅ PASS: Grid & Planes both deterministic")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Planes Roundtrip
# ═══════════════════════════════════════════════════════════════════════

def test_planes_roundtrip():
    grid_orig = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    H, W = 3, 3
    colors_order = order_colors({0, 1, 2})

    planes = {c: [sum((1 << col) for col in range(W) if grid_orig[r][col] == c) for r in range(H)] for c in colors_order}
    grid_recon = [[next(c for c in colors_order if planes[c][r] & (1 << col)) for col in range(W)] for r in range(H)]

    assert grid_orig == grid_recon
    print(f"✅ PASS: Planes roundtrip")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: param_registry_hash Binding
# ═══════════════════════════════════════════════════════════════════════

def test_registry_hash_binding():
    import copy
    reg_orig = param_registry()
    reg_mod = copy.deepcopy(reg_orig)
    reg_mod["pose_order"] = reg_mod["pose_order"][::-1]  # reverse order

    def stable_json_bytes(obj):
        return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')

    hash_orig = blake3_hash(stable_json_bytes(reg_orig))
    hash_mod = blake3_hash(stable_json_bytes(reg_mod))

    assert hash_orig != hash_mod, "Registry modification must change hash"
    print(f"✅ PASS: param_registry_hash binding")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    # Empty grid
    serialize_grid_be_row_major([], 0, 0, [0])

    # All-zero grid
    serialize_grid_be_row_major([[0, 0], [0, 0]], 2, 2, [0])

    # Color 0 missing → must fail
    try:
        order_colors({1, 2, 3})
        assert False, "Should have raised SerializationError"
    except SerializationError:
        pass

    # Unaligned width
    serialize_grid_be_row_major([[1, 2, 3, 4, 5]], 1, 5, order_colors({0, 1, 2, 3, 4, 5}))

    print(f"✅ PASS: Edge cases")


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Receipts Forbid Floats
# ═══════════════════════════════════════════════════════════════════════

def test_receipts_forbid_floats():
    r = Receipts("test")
    try:
        r.put("bad", 3.14)
        assert False, "Floats must be rejected"
    except ReceiptError:
        pass

    try:
        r.put("nested", [1, 2.5])
        assert False, "Nested floats must be rejected"
    except ReceiptError:
        pass

    print(f"✅ PASS: Receipts forbid floats")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("WO-00 FINAL VERIFICATION SUITE")
    print("=" * 70)

    tests = [
        test_param_registry_completeness,
        test_double_run_real_arc,
        test_byte_sensitivity,
        test_grid_planes_deterministic,
        test_planes_roundtrip,
        test_registry_hash_binding,
        test_edge_cases,
        test_receipts_forbid_floats,
    ]

    passed, failed = 0, 0
    for test in tests:
        print(f"\n[TEST] {test.__name__}")
        print("-" * 70)
        try:
            test()
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
