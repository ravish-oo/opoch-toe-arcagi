#!/usr/bin/env python3
"""
Quick verification test for WO-00 implementation.

Tests:
1. param_registry() has all required keys
2. blake3_hash() is deterministic
3. serialize_grid_be_row_major() produces stable bytes
4. Receipts.digest() is deterministic
5. assert_double_run_equal() catches non-determinism
"""

from src.arcbit.core.registry import param_registry, RegistryError
from src.arcbit.core.hashing import blake3_hash
from src.arcbit.core.bytesio import (
    order_colors,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
    SerializationError
)
from src.arcbit.core.receipts import (
    Receipts,
    assert_double_run_equal,
    DeterminismError,
    ReceiptError
)


def test_param_registry():
    """Verify param_registry has all required keys and correct values."""
    print("Testing param_registry()...")

    registry = param_registry()

    # Check required keys
    required_keys = {
        "spec_version", "endianness", "pose_order", "ac3_neighbor",
        "ac3_queue_order", "engine_priority", "bottom_color",
        "period_phase_origin", "strict_downscale", "hash_algo",
        "color_ordering", "byte_frame_tags"
    }
    assert set(registry.keys()) == required_keys, "Registry key mismatch"

    # Check values
    assert registry["spec_version"] == "1.5"
    assert registry["endianness"] == "BE"
    assert registry["pose_order"] == ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]
    assert registry["bottom_color"] == 0
    assert registry["hash_algo"] == "BLAKE3"

    print("✓ param_registry() correct")


def test_blake3_deterministic():
    """Verify BLAKE3 is deterministic."""
    print("Testing blake3_hash() determinism...")

    data = b"test data for hashing"
    hash1 = blake3_hash(data)
    hash2 = blake3_hash(data)

    assert hash1 == hash2, "BLAKE3 not deterministic"
    assert len(hash1) == 64, f"Hash length wrong: {len(hash1)}"

    # Different data should produce different hash
    hash3 = blake3_hash(b"different data")
    assert hash3 != hash1, "Different data produced same hash"

    print(f"✓ blake3_hash() deterministic (sample: {hash1[:16]}...)")


def test_grid_serialization():
    """Verify grid serialization is stable and correct."""
    print("Testing serialize_grid_be_row_major()...")

    # Simple 2x3 grid with colors 0, 1, 2
    G = [
        [0, 1, 2],
        [1, 0, 1]
    ]
    H, W = 2, 3
    colors = order_colors({0, 1, 2})

    # Serialize twice
    bytes1 = serialize_grid_be_row_major(G, H, W, colors)
    bytes2 = serialize_grid_be_row_major(G, H, W, colors)

    assert bytes1 == bytes2, "Grid serialization not deterministic"

    # Check hash stability
    hash1 = blake3_hash(bytes1)
    hash2 = blake3_hash(bytes2)
    assert hash1 == hash2, "Grid hash not stable"

    # Flip one pixel and verify hash changes
    G_modified = [
        [0, 1, 2],
        [1, 2, 1]  # changed (1,1) from 0 to 2
    ]
    bytes3 = serialize_grid_be_row_major(G_modified, H, W, colors)
    hash3 = blake3_hash(bytes3)
    assert hash3 != hash1, "Modified grid produced same hash"

    print(f"✓ Grid serialization stable (hash: {hash1[:16]}...)")


def test_planes_serialization():
    """Verify plane serialization."""
    print("Testing serialize_planes_be_row_major()...")

    # Simple planes for 2x3 grid
    H, W = 2, 3
    colors = [0, 1, 2]

    # Row masks (bit encoding: bit 0 = col 0, bit 1 = col 1, etc.)
    planes = {
        0: [0b001, 0b010],  # color 0 at (0,0) and (1,1)
        1: [0b010, 0b101],  # color 1 at (0,1), (1,0), (1,2)
        2: [0b100, 0b000],  # color 2 at (0,2)
    }

    bytes1 = serialize_planes_be_row_major(planes, H, W, colors)
    bytes2 = serialize_planes_be_row_major(planes, H, W, colors)

    assert bytes1 == bytes2, "Plane serialization not deterministic"

    hash1 = blake3_hash(bytes1)
    print(f"✓ Plane serialization stable (hash: {hash1[:16]}...)")


def test_receipts():
    """Verify Receipts class works correctly."""
    print("Testing Receipts class...")

    receipts = Receipts("test-section")
    receipts.put("key1", 42)
    receipts.put("key2", "value")
    receipts.put("key3", [1, 2, 3])
    receipts.put("key4", {"nested": True})

    digest = receipts.digest()

    # Check structure
    assert digest["section"] == "test-section"
    assert digest["spec_version"] == "1.5"
    assert "param_registry_hash" in digest
    assert "section_hash" in digest
    assert digest["payload"]["key1"] == 42

    print(f"✓ Receipts digest created (hash: {digest['section_hash'][:16]}...)")


def test_receipts_forbids_floats():
    """Verify Receipts rejects floats."""
    print("Testing Receipts float rejection...")

    receipts = Receipts("test")
    try:
        receipts.put("bad", 3.14)
        assert False, "Should have raised ReceiptError for float"
    except ReceiptError as e:
        assert "float" in str(e).lower()

    print("✓ Receipts correctly rejects floats")


def test_double_run():
    """Verify double-run checker works."""
    print("Testing assert_double_run_equal()...")

    # Deterministic builder
    def build_deterministic():
        r = Receipts("deterministic")
        r.put("value", 42)
        r.put("list", [1, 2, 3])
        return r

    # Should pass
    assert_double_run_equal(build_deterministic)
    print("✓ Double-run check passes for deterministic code")

    # Non-deterministic builder (using counter)
    counter = [0]
    def build_nondeterministic():
        r = Receipts("nondeterministic")
        counter[0] += 1
        r.put("value", counter[0])  # different each time
        return r

    try:
        assert_double_run_equal(build_nondeterministic)
        assert False, "Should have raised DeterminismError"
    except DeterminismError as e:
        assert e.first_differing_key == "value"
        assert e.value_a == 1
        assert e.value_b == 2

    print("✓ Double-run check catches non-determinism")


def test_order_colors():
    """Verify color ordering."""
    print("Testing order_colors()...")

    colors = order_colors({5, 0, 3, 1})
    assert colors == [0, 1, 3, 5], f"Wrong order: {colors}"

    # Must include 0
    try:
        order_colors({1, 2, 3})
        assert False, "Should have raised SerializationError"
    except SerializationError as e:
        assert "0" in str(e)

    print("✓ Color ordering correct")


if __name__ == "__main__":
    print("=" * 60)
    print("WO-00 Verification Tests")
    print("=" * 60)
    print()

    test_param_registry()
    test_blake3_deterministic()
    test_order_colors()
    test_grid_serialization()
    test_planes_serialization()
    test_receipts()
    test_receipts_forbids_floats()
    test_double_run()

    print()
    print("=" * 60)
    print("✅ All WO-00 tests passed!")
    print("=" * 60)
    print()
    print("WO-00 Implementation Complete:")
    print("  - param_registry() with 12 frozen keys")
    print("  - blake3_hash() deterministic")
    print("  - serialize_grid_be_row_major() with bit 7→col 0 mapping")
    print("  - serialize_planes_be_row_major() identical format")
    print("  - Receipts class with section_hash binding")
    print("  - assert_double_run_equal() catches non-determinism")
    print()
    print("Foundation is solid. Ready for WO-01+ (Bit-Kernel).")
