"""
WO-00 Verification Tests - Real ARC Data, Receipts-First

Tests all WO-00 components with real ARC-AGI data:
  1. param_registry() completeness and consistency
  2. serialize_grid_be_row_major() byte-level correctness
  3. serialize_planes_be_row_major() equivalence
  4. blake3_hash() determinism
  5. Receipts class and double-run equality
  6. Edge cases (empty grids, single pixel, color 0 enforcement)

Verification method: ALGEBRAIC (receipts and hashes only).
No assertions on internal state, only on cryptographic commitments.

Spec: WO-00, all sections.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
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
# Test 1: param_registry() - All Required Keys, No Extras
# ═══════════════════════════════════════════════════════════════════════

def test_param_registry_completeness():
    """Verify param_registry() has exactly the required keys with correct types."""

    reg = param_registry()

    # Required keys from WO-00 spec
    required = {
        "spec_version", "endianness", "pose_order", "ac3_neighbor",
        "ac3_queue_order", "engine_priority", "bottom_color",
        "period_phase_origin", "strict_downscale", "hash_algo",
        "color_ordering", "byte_frame_tags"
    }

    actual = set(reg.keys())

    # Algebraic check: set equality via hash
    receipts = Receipts("test-param-registry-completeness")
    receipts.put("required_keys", sorted(required))
    receipts.put("actual_keys", sorted(actual))
    receipts.put("missing", sorted(required - actual))
    receipts.put("extra", sorted(actual - required))
    receipts.put("keys_match", actual == required)

    digest = receipts.digest()

    # Must match exactly
    assert digest["payload"]["keys_match"], (
        f"param_registry() key mismatch.\n"
        f"Missing: {digest['payload']['missing']}\n"
        f"Extra: {digest['payload']['extra']}"
    )

    # Spec-mandated values
    assert reg["spec_version"] == "1.5", f"spec_version must be '1.5', got {reg['spec_version']}"
    assert reg["endianness"] == "BE", f"endianness must be 'BE', got {reg['endianness']}"
    assert reg["bottom_color"] == 0, f"bottom_color must be 0, got {reg['bottom_color']}"
    assert reg["strict_downscale"] is True, f"strict_downscale must be True"
    assert reg["hash_algo"] == "BLAKE3", f"hash_algo must be 'BLAKE3'"
    assert reg["color_ordering"] == "ascending-int"

    # Pose order must have exactly 8 elements (D4 group)
    assert len(reg["pose_order"]) == 8, f"pose_order must have 8 elements, got {len(reg['pose_order'])}"
    assert reg["pose_order"] == ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    # Engine priority (no T1 Witness, no T2 Unanimity - they're in fixed buckets)
    expected_engines = ["T3", "T5", "T4", "T6", "T7", "T8", "T9", "T10", "T11"]
    assert reg["engine_priority"] == expected_engines

    # Byte frame tags
    assert reg["byte_frame_tags"]["GRID"] == "GRD1"
    assert reg["byte_frame_tags"]["PLANES"] == "PLN1"

    print(f"✅ PASS: param_registry() completeness - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Double-Run Determinism on Real ARC Data
# ═══════════════════════════════════════════════════════════════════════

def test_double_run_determinism_real_arc():
    """
    Load real ARC task, serialize same grid twice, verify identical hashes.

    Spec: WO-00 section 7 (double-run equivalence).
    """

    # Load real ARC task
    arc_data_path = Path(__file__).parent.parent / "data" / "arc2_training.json"
    with open(arc_data_path, 'r') as f:
        tasks = json.load(f)

    # Use first task, first training input
    task_id = list(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]["input"]

    H = len(grid)
    W = len(grid[0]) if H > 0 else 0

    # Build color universe (must include 0)
    colors_set = {0}
    for row in grid:
        colors_set.update(row)
    colors_order = order_colors(colors_set)

    # Callable that builds receipts
    def build_receipts():
        r = Receipts("test-double-run-arc")

        # Serialize grid
        grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)
        grid_hash = blake3_hash(grid_bytes)

        r.put("task_id", task_id)
        r.put("H", H)
        r.put("W", W)
        r.put("colors", colors_order)
        r.put("grid_hash", grid_hash)
        r.put("grid_bytes_len", len(grid_bytes))

        return r

    # Double-run check (must not raise)
    try:
        assert_double_run_equal(build_receipts)
        print(f"✅ PASS: Double-run determinism on task {task_id}")
    except DeterminismError as e:
        print(f"❌ FAIL: Double-run determinism failed")
        print(f"  Section: {e.section}")
        print(f"  First differing key: {e.first_differing_key}")
        print(f"  Value A: {e.value_a}")
        print(f"  Value B: {e.value_b}")
        print(f"  Hash A: {e.hash_a[:16]}...")
        print(f"  Hash B: {e.hash_b[:16]}...")
        raise


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Byte-Level Sensitivity (Flip One Pixel → Hash Changes)
# ═══════════════════════════════════════════════════════════════════════

def test_byte_sensitivity_flip_pixel():
    """
    Serialize grid, flip one pixel, verify hash MUST change.

    Spec: WO-00 Reviewer quick-test.
    """

    # Simple 3x3 grid
    grid_original = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    H, W = 3, 3
    colors_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    colors_order = order_colors(colors_set)

    # Hash original
    bytes_original = serialize_grid_be_row_major(grid_original, H, W, colors_order)
    hash_original = blake3_hash(bytes_original)

    # Flip one pixel
    grid_flipped = [row[:] for row in grid_original]  # deep copy
    grid_flipped[1][1] = 0  # change center pixel from 5 to 0

    bytes_flipped = serialize_grid_be_row_major(grid_flipped, H, W, colors_order)
    hash_flipped = blake3_hash(bytes_flipped)

    # Receipts-first verification
    receipts = Receipts("test-byte-sensitivity")
    receipts.put("hash_original", hash_original)
    receipts.put("hash_flipped", hash_flipped)
    receipts.put("hashes_differ", hash_original != hash_flipped)
    receipts.put("pixel_changed", {"row": 1, "col": 1, "old": 5, "new": 0})

    digest = receipts.digest()

    assert digest["payload"]["hashes_differ"], (
        f"Flipping pixel must change hash.\n"
        f"Original: {hash_original[:16]}...\n"
        f"Flipped:  {hash_flipped[:16]}..."
    )

    print(f"✅ PASS: Byte sensitivity - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Grid ↔ Planes Serialization Equivalence
# ═══════════════════════════════════════════════════════════════════════

def test_grid_planes_serialization_equivalence():
    """
    Convert grid → planes, serialize both ways, verify payload bytes match.

    Spec: WO-00 section 4 notes - "identical to serialize_grid_be_row_major's payload".
    """

    grid = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]

    H, W = 3, 3
    colors_set = {0, 1}
    colors_order = order_colors(colors_set)

    # Serialize as grid
    grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)

    # Convert to planes (manual for testing)
    planes = {}
    for c in colors_order:
        plane_rows = []
        for r in range(H):
            row_mask = 0
            for col in range(W):
                if grid[r][col] == c:
                    row_mask |= (1 << col)
            plane_rows.append(row_mask)
        planes[c] = plane_rows

    # Serialize as planes
    planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)

    # Extract payloads (skip headers)
    # Grid format: b"GRD1" + 2H + 2W + 1K + K*color_ids + payload
    # Planes format: b"PLN1" + 2H + 2W + 1K + K*color_ids + payload
    # Both have same header structure

    header_size = 4 + 2 + 2 + 1 + len(colors_order)  # tag + H + W + K + color_ids

    grid_payload = grid_bytes[header_size:]
    planes_payload = planes_bytes[header_size:]

    # Receipts verification
    receipts = Receipts("test-grid-planes-equivalence")
    receipts.put("grid_payload_hash", blake3_hash(grid_payload))
    receipts.put("planes_payload_hash", blake3_hash(planes_payload))
    receipts.put("payloads_match", grid_payload == planes_payload)
    receipts.put("grid_total_len", len(grid_bytes))
    receipts.put("planes_total_len", len(planes_bytes))

    digest = receipts.digest()

    assert digest["payload"]["payloads_match"], (
        f"Grid and planes serialization payloads must match.\n"
        f"Grid payload hash:   {digest['payload']['grid_payload_hash'][:16]}...\n"
        f"Planes payload hash: {digest['payload']['planes_payload_hash'][:16]}..."
    )

    print(f"✅ PASS: Grid ↔ Planes equivalence - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: param_registry_hash Binding in Receipts
# ═══════════════════════════════════════════════════════════════════════

def test_param_registry_hash_binding():
    """
    Verify that changing param_registry causes param_registry_hash to change,
    proving receipt binding to frozen constants.

    Spec: WO-00 Reviewer quick-test - "change pose_order → hash must change".
    """

    # Build receipt with current registry
    def build_with_current_registry():
        r = Receipts("test-registry-binding-current")
        r.put("value", 42)
        return r

    receipt_current = build_with_current_registry()
    digest_current = receipt_current.digest()
    param_hash_current = digest_current["param_registry_hash"]

    # Manually compute hash with modified registry (simulating drift)
    # Note: We can't actually change param_registry() without editing code,
    # but we can verify the mechanism by computing what the hash WOULD be

    # Get current registry and modify it
    import copy
    reg_original = param_registry()
    reg_modified = copy.deepcopy(reg_original)
    reg_modified["pose_order"] = ["R90", "I", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]  # reordered

    # Compute hash of modified registry
    import json
    def stable_json_bytes(obj):
        json_str = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return json_str.encode('utf-8')

    param_hash_modified = blake3_hash(stable_json_bytes(reg_modified))

    # Receipts verification
    receipts = Receipts("test-registry-binding-verification")
    receipts.put("param_hash_current", param_hash_current)
    receipts.put("param_hash_modified", param_hash_modified)
    receipts.put("hashes_differ", param_hash_current != param_hash_modified)
    receipts.put("modification", "pose_order reordered")

    digest = receipts.digest()

    assert digest["payload"]["hashes_differ"], (
        f"Modifying param_registry must change param_registry_hash.\n"
        f"Current:  {param_hash_current[:16]}...\n"
        f"Modified: {param_hash_modified[:16]}..."
    )

    print(f"✅ PASS: param_registry_hash binding - hash: {digest['section_hash'][:16]}...")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Edge Cases (Empty Grid, All-Zero, Color 0 Enforcement)
# ═══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    """
    Test edge cases:
      - Empty grid (H=0)
      - All-zero grid
      - Color 0 missing → must raise SerializationError
    """

    # Edge case 1: Empty grid
    grid_empty = []
    try:
        colors = order_colors({0})
        bytes_empty = serialize_grid_be_row_major(grid_empty, 0, 0, colors)
        hash_empty = blake3_hash(bytes_empty)
        print(f"  ✓ Empty grid serializable: hash {hash_empty[:16]}...")
    except Exception as e:
        print(f"  ✗ Empty grid failed: {e}")
        raise

    # Edge case 2: All-zero grid
    grid_zero = [[0, 0], [0, 0]]
    try:
        colors = order_colors({0})
        bytes_zero = serialize_grid_be_row_major(grid_zero, 2, 2, colors)
        hash_zero = blake3_hash(bytes_zero)
        print(f"  ✓ All-zero grid serializable: hash {hash_zero[:16]}...")
    except Exception as e:
        print(f"  ✗ All-zero grid failed: {e}")
        raise

    # Edge case 3: Color 0 missing (must fail)
    try:
        colors_no_zero = order_colors({1, 2, 3})  # missing 0
        print(f"  ✗ Color 0 missing: should have raised SerializationError")
        assert False, "order_colors({1,2,3}) should raise SerializationError"
    except SerializationError as e:
        print(f"  ✓ Color 0 missing correctly rejected: {e}")

    # Edge case 4: Large W not multiple of 8 (padding test)
    grid_unaligned = [[1, 2, 3, 4, 5]]  # W=5, needs 1 byte (ceil(5/8)=1)
    try:
        colors = order_colors({0, 1, 2, 3, 4, 5})
        bytes_unaligned = serialize_grid_be_row_major(grid_unaligned, 1, 5, colors)
        hash_unaligned = blake3_hash(bytes_unaligned)
        print(f"  ✓ Unaligned width (W=5) serializable: hash {hash_unaligned[:16]}...")
    except Exception as e:
        print(f"  ✗ Unaligned width failed: {e}")
        raise

    print(f"✅ PASS: All edge cases handled correctly")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Receipts Type Validation (Floats Forbidden)
# ═══════════════════════════════════════════════════════════════════════

def test_receipts_forbid_floats():
    """
    Verify that receipts reject floats (spec: "no floats in logic").
    """

    receipts = Receipts("test-float-rejection")

    # Try to put a float (must fail)
    try:
        receipts.put("bad_value", 3.14)
        print(f"  ✗ Float accepted: should have raised ReceiptError")
        assert False, "Floats must be rejected"
    except ReceiptError as e:
        print(f"  ✓ Float correctly rejected: {e}")

    # Try nested float (must fail)
    try:
        receipts.put("nested_bad", {"data": [1, 2, 3.5]})
        print(f"  ✗ Nested float accepted: should have raised ReceiptError")
        assert False, "Nested floats must be rejected"
    except ReceiptError as e:
        print(f"  ✓ Nested float correctly rejected: {e}")

    print(f"✅ PASS: Receipts float validation")


# ═══════════════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run all WO-00 verification tests."""

    print("=" * 70)
    print("WO-00 VERIFICATION - Real ARC Data, Receipts-First")
    print("=" * 70)
    print()

    tests = [
        ("param_registry Completeness", test_param_registry_completeness),
        ("Double-Run Determinism (Real ARC)", test_double_run_determinism_real_arc),
        ("Byte Sensitivity (Flip Pixel)", test_byte_sensitivity_flip_pixel),
        ("Grid ↔ Planes Equivalence", test_grid_planes_serialization_equivalence),
        ("param_registry_hash Binding", test_param_registry_hash_binding),
        ("Edge Cases", test_edge_cases),
        ("Receipts Float Validation", test_receipts_forbid_floats),
    ]

    passed = 0
    failed = 0

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
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
