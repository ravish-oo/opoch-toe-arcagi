"""
Tests for WO-08: Output Transport Emitter

Verifies:
  - Integer replication (Kronecker product)
  - Exact block-constancy decimation
  - Silent training handling
  - Transport to test output frame
"""

import pytest
from src.arcbit.emitters import emit_output_transport


def test_replicate_2x2():
    """Test integer replication (2x upsampling)."""
    # 1x1 grid → 2x2 canvas
    Y_list = [[[5]]]
    frames_out = [("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = [0, 5]
    R_out, C_out = 2, 2

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify normalization
    assert receipts[0]["norm_kind"] == "replicate"
    assert receipts[0]["s_r"] == 2
    assert receipts[0]["s_c"] == 2
    assert receipts[0]["block_constancy_ok"] is None

    # Verify scope covers all 4 pixels
    assert receipts[0]["scope_bits"] == 4

    # Verify included count
    assert section["payload"]["n_included"] == 1

    print("✓ Replicate 2x2 works")


def test_decimate_2x2_constant_blocks():
    """Test exact decimation with constant blocks."""
    # 4x4 grid with 2x2 constant blocks → 2x2 canvas
    Y_list = [
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
    ]
    frames_out = [("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = [0, 1, 2, 3, 4]
    R_out, C_out = 2, 2

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify normalization
    assert receipts[0]["norm_kind"] == "decimate"
    assert receipts[0]["s_r"] == 2
    assert receipts[0]["s_c"] == 2
    assert receipts[0]["block_constancy_ok"] is True

    # Verify scope
    assert receipts[0]["scope_bits"] == 4

    print("✓ Decimate 2x2 with constant blocks works")


def test_decimate_fails_non_constant_block():
    """Test decimation fails when block is not constant."""
    # 2x2 grid with non-constant block → 1x1 canvas
    Y_list = [
        [
            [1, 2],
            [3, 4],
        ]
    ]
    frames_out = [("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = [0, 1, 2, 3, 4]
    R_out, C_out = 1, 1

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify normalization failed
    assert receipts[0]["norm_kind"] == "silent"
    assert receipts[0]["block_constancy_ok"] is False

    # Verify no scope
    assert receipts[0]["scope_bits"] == 0

    # Verify not included
    assert section["payload"]["n_included"] == 0

    print("✓ Decimate fails on non-constant block")


def test_silent_no_integer_relation():
    """Test silent when no exact integer relation exists."""
    # 3x3 grid → 2x2 canvas (no integer factor)
    Y_list = [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    ]
    frames_out = [("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = list(range(10))
    R_out, C_out = 2, 2

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify silent
    assert receipts[0]["norm_kind"] == "silent"
    assert receipts[0]["s_r"] is None
    assert receipts[0]["s_c"] is None
    assert receipts[0]["block_constancy_ok"] is None

    # Verify no scope
    assert receipts[0]["scope_bits"] == 0

    print("✓ Silent when no integer relation")


def test_multiple_trainings_mixed():
    """Test multiple trainings with mixed normalization results."""
    # Training 0: 1x1 → 2x2 (replicate)
    # Training 1: 4x4 → 2x2 (decimate, constant blocks)
    # Training 2: 3x3 → 2x2 (silent)
    Y_list = [
        [[5]],
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    ]
    frames_out = [("I", (0, 0)), ("I", (0, 0)), ("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = list(range(10))
    R_out, C_out = 2, 2

    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify results
    assert receipts[0]["norm_kind"] == "replicate"
    assert receipts[1]["norm_kind"] == "decimate"
    assert receipts[2]["norm_kind"] == "silent"

    # Verify included count (2 out of 3)
    assert section["payload"]["n_included"] == 2

    print("✓ Multiple trainings with mixed normalization")


def test_receipts_determinism():
    """Test that double-run produces identical receipts."""
    Y_list = [[[1, 2], [3, 4]]]
    frames_out = [("I", (0, 0))]
    pi_out_star = ("I", (0, 0))
    colors_order = [0, 1, 2, 3, 4]
    R_out, C_out = 4, 4

    # Run 1
    _, _, receipts1, section1 = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Run 2
    _, _, receipts2, section2 = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    # Verify identical hashes
    assert receipts1[0]["transport_hash"] == receipts2[0]["transport_hash"]
    assert section1["payload"]["transports_hash"] == section2["payload"]["transports_hash"]
    assert section1["section_hash"] == section2["section_hash"]

    print("✓ Receipts are deterministic")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
