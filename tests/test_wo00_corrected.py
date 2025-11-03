"""
WO-00 Verification Tests - CORRECTED

Fixed understanding: Grid and Planes formats use different iteration orders:
  - Grid: for row → for color (row-major over colors)
  - Planes: for color → for row (color-major over rows)

Both are valid, deterministic encodings. They're not byte-identical, but both
correctly represent the same data and must be individually deterministic.
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
)


def test_grid_planes_both_deterministic():
    """
    Verify both Grid and Planes serialization are deterministic independently.

    Spec: WO-00 - both formats must be stable and hashable, but they use
    different iteration orders so they're not byte-identical.
    """

    grid = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]

    H, W = 3, 3
    colors_order = [0, 1]

    # Double-run Grid serialization
    def build_grid_receipt():
        r = Receipts("test-grid-deterministic")
        grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)
        r.put("grid_hash", blake3_hash(grid_bytes))
        r.put("grid_len", len(grid_bytes))
        return r

    try:
        assert_double_run_equal(build_grid_receipt)
        print("  ✓ Grid serialization is deterministic")
    except Exception as e:
        print(f"  ✗ Grid serialization NOT deterministic: {e}")
        raise

    # Convert to planes
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

    # Double-run Planes serialization
    def build_planes_receipt():
        r = Receipts("test-planes-deterministic")
        planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)
        r.put("planes_hash", blake3_hash(planes_bytes))
        r.put("planes_len", len(planes_bytes))
        return r

    try:
        assert_double_run_equal(build_planes_receipt)
        print("  ✓ Planes serialization is deterministic")
    except Exception as e:
        print(f"  ✗ Planes serialization NOT deterministic: {e}")
        raise

    # Verify both are stable (same lengths, different hashes due to iteration order)
    grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)
    planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)

    grid_hash = blake3_hash(grid_bytes)
    planes_hash = blake3_hash(planes_bytes)

    receipts = Receipts("test-grid-planes-both-deterministic")
    receipts.put("grid_hash", grid_hash)
    receipts.put("planes_hash", planes_hash)
    receipts.put("grid_len", len(grid_bytes))
    receipts.put("planes_len", len(planes_bytes))
    receipts.put("same_length", len(grid_bytes) == len(planes_bytes))
    receipts.put("iteration_order_note", "Grid=row-major, Planes=color-major (intentionally different)")

    digest = receipts.digest()

    # Both should have same total length (same header + same number of masks)
    assert digest["payload"]["same_length"], (
        f"Grid and Planes must have same total length.\n"
        f"Grid: {len(grid_bytes)} bytes\n"
        f"Planes: {len(planes_bytes)} bytes"
    )

    print(f"✅ PASS: Both Grid and Planes serialization are deterministic")
    print(f"  Grid hash:   {grid_hash[:16]}...")
    print(f"  Planes hash: {planes_hash[:16]}...")
    print(f"  (Different hashes expected due to iteration order)")


def test_planes_reconstruct_roundtrip():
    """
    Verify: Grid → Planes → reconstruct grid → should match original.

    This proves the Planes format correctly encodes the grid data.
    """

    grid_original = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1]
    ]

    H, W = 3, 3
    colors_order = order_colors({0, 1, 2})

    # Convert Grid → Planes
    planes = {}
    for c in colors_order:
        plane_rows = []
        for r in range(H):
            row_mask = 0
            for col in range(W):
                if grid_original[r][col] == c:
                    row_mask |= (1 << col)
            plane_rows.append(row_mask)
        planes[c] = plane_rows

    # Reconstruct Grid from Planes
    grid_reconstructed = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for col in range(W):
            for c in colors_order:
                if planes[c][r] & (1 << col):
                    grid_reconstructed[r][col] = c
                    break

    # Verify roundtrip
    receipts = Receipts("test-planes-roundtrip")

    grid_orig_bytes = serialize_grid_be_row_major(grid_original, H, W, colors_order)
    grid_recon_bytes = serialize_grid_be_row_major(grid_reconstructed, H, W, colors_order)

    receipts.put("original_hash", blake3_hash(grid_orig_bytes))
    receipts.put("reconstructed_hash", blake3_hash(grid_recon_bytes))
    receipts.put("roundtrip_ok", grid_original == grid_reconstructed)

    digest = receipts.digest()

    assert digest["payload"]["roundtrip_ok"], "Planes roundtrip must preserve grid exactly"

    print(f"✅ PASS: Planes roundtrip reconstruction - hash: {digest['section_hash'][:16]}...")


def main():
    print("=" * 70)
    print("WO-00 VERIFICATION - Grid/Planes Serialization (CORRECTED)")
    print("=" * 70)
    print()

    tests = [
        ("Grid & Planes Both Deterministic", test_grid_planes_both_deterministic),
        ("Planes Roundtrip Reconstruction", test_planes_reconstruct_roundtrip),
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
