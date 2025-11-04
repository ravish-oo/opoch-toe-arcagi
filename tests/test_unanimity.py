"""
Tests for WO-08: Unanimity Emitter

Verifies:
  - Unanimous pixels (all trainings agree)
  - Disagreement handling (silent at pixel)
  - Empty scope pixels
  - Singleton admits only
"""

import pytest
from src.arcbit.emitters import emit_unity


def test_unanimous_single_training():
    """Test unanimity with single training."""
    # Single training: all pixels should be unanimous
    colors_order = [0, 1, 2]
    R_out, C_out = 2, 2

    # Training 0: color 1 at all 4 pixels
    A_out_0 = {
        0: [0b00, 0b00],
        1: [0b11, 0b11],  # All pixels are color 1
        2: [0b00, 0b00],
    }
    S_out_0 = [0b11, 0b11]  # All pixels in scope

    A_out_list = [A_out_0]
    S_out_list = [S_out_0]

    A_uni, S_uni, receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # All 4 pixels should be unanimous
    assert receipt["unanimous_pixels"] == 4
    assert receipt["total_covered_pixels"] == 4
    assert receipt["empty_scope_pixels"] == 0

    # Verify unanimous color is 1 at all pixels
    assert S_uni == [0b11, 0b11]
    assert A_uni[1] == [0b11, 0b11]
    assert A_uni[0] == [0b00, 0b00]
    assert A_uni[2] == [0b00, 0b00]

    print("✓ Unanimous single training works")


def test_unanimous_agreement_across_trainings():
    """Test unanimity when multiple trainings agree."""
    colors_order = [0, 1, 2]
    R_out, C_out = 1, 2

    # Training 0: [1, 2]
    A_out_0 = {
        0: [0b00],
        1: [0b01],  # Pixel 0 is color 1
        2: [0b10],  # Pixel 1 is color 2
    }
    S_out_0 = [0b11]

    # Training 1: [1, 2] (same as training 0)
    A_out_1 = {
        0: [0b00],
        1: [0b01],
        2: [0b10],
    }
    S_out_1 = [0b11]

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    A_uni, S_uni, receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # Both pixels unanimous
    assert receipt["unanimous_pixels"] == 2
    assert receipt["total_covered_pixels"] == 2
    assert receipt["included_train_ids"] == [0, 1]

    # Verify colors
    assert S_uni == [0b11]
    assert A_uni[1] == [0b01]
    assert A_uni[2] == [0b10]

    print("✓ Unanimous agreement across trainings")


def test_disagreement_silent():
    """Test that disagreeing trainings result in silent pixel."""
    colors_order = [0, 1, 2]
    R_out, C_out = 1, 1

    # Training 0: color 1
    A_out_0 = {0: [0b0], 1: [0b1], 2: [0b0]}
    S_out_0 = [0b1]

    # Training 1: color 2 (disagrees!)
    A_out_1 = {0: [0b0], 1: [0b0], 2: [0b1]}
    S_out_1 = [0b1]

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    A_uni, S_uni, receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # No unanimous pixels (disagreement)
    assert receipt["unanimous_pixels"] == 0
    assert receipt["total_covered_pixels"] == 1
    assert receipt["empty_scope_pixels"] == 0

    # Scope should be silent at pixel
    assert S_uni == [0b0]

    print("✓ Disagreement results in silent pixel")


def test_partial_coverage():
    """Test pixels where only some trainings speak."""
    colors_order = [0, 1, 2]
    R_out, C_out = 1, 3

    # Training 0: [1, _, _] (only first pixel)
    A_out_0 = {0: [0b000], 1: [0b001], 2: [0b000]}
    S_out_0 = [0b001]

    # Training 1: [_, 2, _] (only second pixel)
    A_out_1 = {0: [0b000], 1: [0b000], 2: [0b010]}
    S_out_1 = [0b010]

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    A_uni, S_uni, receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # Two pixels unanimous (each has only one training speaking)
    assert receipt["unanimous_pixels"] == 2
    assert receipt["total_covered_pixels"] == 2
    assert receipt["empty_scope_pixels"] == 1  # Pixel 2 has no coverage

    # Verify unanimity at covered pixels
    assert S_uni == [0b011]
    assert A_uni[1] == [0b001]  # Pixel 0
    assert A_uni[2] == [0b010]  # Pixel 1

    print("✓ Partial coverage works correctly")


def test_all_silent_trainings():
    """Test when all trainings are silent."""
    colors_order = [0, 1, 2]
    R_out, C_out = 2, 2

    # All trainings have zero scope
    A_out_0 = {0: [0b00, 0b00], 1: [0b00, 0b00], 2: [0b00, 0b00]}
    S_out_0 = [0b00, 0b00]

    A_out_1 = {0: [0b00, 0b00], 1: [0b00, 0b00], 2: [0b00, 0b00]}
    S_out_1 = [0b00, 0b00]

    A_out_list = [A_out_0, A_out_1]
    S_out_list = [S_out_0, S_out_1]

    A_uni, S_uni, receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # No unanimous pixels
    assert receipt["unanimous_pixels"] == 0
    assert receipt["total_covered_pixels"] == 0
    assert receipt["empty_scope_pixels"] == 4
    assert receipt["included_train_ids"] == []

    # All silent
    assert S_uni == [0b00, 0b00]

    print("✓ All silent trainings handled correctly")


def test_receipts_determinism():
    """Test that double-run produces identical receipts."""
    colors_order = [0, 1]
    R_out, C_out = 2, 2

    A_out_0 = {0: [0b00, 0b00], 1: [0b11, 0b11]}
    S_out_0 = [0b11, 0b11]

    A_out_list = [A_out_0]
    S_out_list = [S_out_0]

    # Run 1
    _, _, receipt1 = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    # Run 2
    _, _, receipt2 = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)

    # Verify identical hashes
    assert receipt1["unanimity_hash"] == receipt2["unanimity_hash"]
    assert receipt1["scope_hash"] == receipt2["scope_hash"]

    print("✓ Unanimity receipts are deterministic")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
