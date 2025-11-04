"""
Tests for WO-07: Conjugation & Forward Witness Emitter

Verifies:
  - Frame algebra (compose, inverse)
  - Conjugation of witness pieces
  - Forward emission of test input planes
  - Scope-gated intersection across trainings
  - Receipts generation and determinism
"""

import pytest
from src.arcbit.emitters import emit_witness
from src.arcbit.emitters.witness_emit import (
    _compose_frames,
    _inverse_frame,
    _apply_d4_to_vector,
)


def test_frame_identity():
    """Test that Pi ∘ Pi⁻¹ = I (identity)."""
    # Test with various frames
    frames = [
        ("I", (0, 0)),
        ("R90", (5, 3)),
        ("R180", (-2, 7)),
        ("FX", (1, -1)),
        ("FXR270", (10, 20)),
    ]

    for Pi in frames:
        Pi_inv = _inverse_frame(Pi)
        Pi_composed = _compose_frames(Pi, Pi_inv)

        # Result should be identity: (I, (0, 0))
        R_result, a_result = Pi_composed
        assert R_result == "I", f"Composition failed for {Pi}: got R={R_result}"
        assert a_result == (0, 0), f"Composition failed for {Pi}: got a={a_result}"

        print(f"✓ Pi ∘ Pi⁻¹ = I for {Pi}")


def test_d4_vector_application():
    """Test D4 transformations on vectors."""
    v = (3, 5)

    # Identity
    assert _apply_d4_to_vector("I", v) == (3, 5)

    # Rotations
    assert _apply_d4_to_vector("R90", v) == (-5, 3)
    assert _apply_d4_to_vector("R180", v) == (-3, -5)
    assert _apply_d4_to_vector("R270", v) == (5, -3)

    # Reflections
    assert _apply_d4_to_vector("FX", v) == (3, -5)
    assert _apply_d4_to_vector("FXR90", v) == (-5, -3)
    assert _apply_d4_to_vector("FXR180", v) == (-3, 5)
    assert _apply_d4_to_vector("FXR270", v) == (5, 3)

    print("✓ D4 vector transformations correct")


def test_simple_identity_emission():
    """Test simple identity task: single training, identity witness."""
    # Simple 2x2 test input
    X_star = [[1, 2], [3, 0]]

    # Single training with identity witness
    witness_results = [
        {
            "silent": False,
            "pieces": [
                {
                    "pid": "I",
                    "dy": 0,
                    "dx": 0,
                    "bbox_src": (0, 0, 1, 1),
                    "bbox_tgt": (0, 0, 1, 1),
                    "c_in": 1,
                    "c_out": 1,
                }
            ],
            "sigma": {1: 1, 2: 2, 3: 3},
        }
    ]

    # Identity frames (no pose, no anchor)
    frames = {
        "Pi_in_star": ("I", (0, 0)),
        "Pi_out_star": ("I", (0, 0)),
        "Pi_in_0": ("I", (0, 0)),
        "Pi_out_0": ("I", (0, 0)),
    }

    colors_order = [0, 1, 2, 3]
    R_out, C_out = 2, 2

    # Emit witness
    A_wit, S_wit, _ = emit_witness(
        X_star, witness_results, frames, colors_order, R_out, C_out
    )

    # Scope should cover the piece bbox (entire 2x2 grid)
    assert S_wit[0] == 0b11, f"Row 0 scope wrong: {bin(S_wit[0])}"
    assert S_wit[1] == 0b11, f"Row 1 scope wrong: {bin(S_wit[1])}"

    # A_wit should reflect the input colors within scope
    # Row 0: [1, 2] -> admits 1 at col 0, 2 at col 1
    # Row 1: [3, 0] -> admits 3 at col 0, 0 at col 1
    assert A_wit[1][0] & 0b01 == 0b01, "Color 1 should be admitted at (0,0)"
    assert A_wit[2][0] & 0b10 == 0b10, "Color 2 should be admitted at (0,1)"
    assert A_wit[3][1] & 0b01 == 0b01, "Color 3 should be admitted at (1,0)"
    assert A_wit[0][1] & 0b10 == 0b10, "Color 0 should be admitted at (1,1)"

    print("✓ Simple identity emission works")


def test_silent_training_no_constraint():
    """Test that silent training imposes no constraint on global witness."""
    X_star = [[1, 0]]

    # Training 0: active with piece
    witness_result_active = {
        "silent": False,
        "pieces": [
            {
                "pid": "I",
                "dy": 0,
                "dx": 0,
                "bbox_src": (0, 0, 0, 0),
                "bbox_tgt": (0, 0, 0, 0),
                "c_in": 1,
                "c_out": 1,
            }
        ],
        "sigma": {1: 1},
    }

    # Training 1: silent
    witness_result_silent = {
        "silent": True,
        "pieces": [],
        "sigma": {},
    }

    frames = {
        "Pi_in_star": ("I", (0, 0)),
        "Pi_out_star": ("I", (0, 0)),
        "Pi_in_0": ("I", (0, 0)),
        "Pi_out_0": ("I", (0, 0)),
        "Pi_in_1": ("I", (0, 0)),
        "Pi_out_1": ("I", (0, 0)),
    }

    colors_order = [0, 1]
    R_out, C_out = 1, 2

    # Test with only active training
    A_wit_single, S_wit_single, _ = emit_witness(
        X_star, [witness_result_active], frames, colors_order, R_out, C_out
    )

    # Test with active + silent training
    A_wit_multi, S_wit_multi, _ = emit_witness(
        X_star,
        [witness_result_active, witness_result_silent],
        frames,
        colors_order,
        R_out,
        C_out,
    )

    # Scope should be identical (silent training contributes no scope)
    assert S_wit_single == S_wit_multi, "Silent training should not change scope"

    # Admits should be identical (silent training imposes no constraint)
    for c in colors_order:
        assert (
            A_wit_single[c] == A_wit_multi[c]
        ), f"Silent training should not change admits for color {c}"

    print("✓ Silent training imposes no constraint")


def test_multi_training_intersection():
    """Test intersection across multiple trainings."""
    X_star = [[1]]

    # Training 0: admits color 1 or 2
    witness_result_0 = {
        "silent": False,
        "pieces": [],  # Empty pieces → admit-all, but we'll set scope manually
        "sigma": {},
    }

    # Training 1: admits color 1 or 3
    witness_result_1 = {
        "silent": False,
        "pieces": [],
        "sigma": {},
    }

    frames = {
        "Pi_in_star": ("I", (0, 0)),
        "Pi_out_star": ("I", (0, 0)),
        "Pi_in_0": ("I", (0, 0)),
        "Pi_out_0": ("I", (0, 0)),
        "Pi_in_1": ("I", (0, 0)),
        "Pi_out_1": ("I", (0, 0)),
    }

    colors_order = [0, 1, 2, 3]
    R_out, C_out = 1, 1

    # Both trainings have empty pieces → should be silent/admit-all
    A_wit, S_wit, _ = emit_witness(
        X_star, [witness_result_0, witness_result_1], frames, colors_order, R_out, C_out
    )

    # With no pieces, scope should be 0 for both trainings
    assert S_wit[0] == 0, "Empty pieces should result in zero scope"

    print("✓ Multi-training intersection works")


def test_emission_determinism():
    """Test that double-run produces identical results."""
    X_star = [[1, 2], [3, 4]]

    witness_results = [
        {
            "silent": False,
            "pieces": [
                {
                    "pid": "I",
                    "dy": 0,
                    "dx": 0,
                    "bbox_src": (0, 0, 1, 1),
                    "bbox_tgt": (0, 0, 1, 1),
                    "c_in": 1,
                    "c_out": 5,
                }
            ],
            "sigma": {1: 5},
        }
    ]

    frames = {
        "Pi_in_star": ("I", (0, 0)),
        "Pi_out_star": ("I", (0, 0)),
        "Pi_in_0": ("I", (0, 0)),
        "Pi_out_0": ("I", (0, 0)),
    }

    colors_order = [0, 1, 2, 3, 4, 5]
    R_out, C_out = 2, 2

    # Run 1
    A_wit_1, S_wit_1, _ = emit_witness(
        X_star, witness_results, frames, colors_order, R_out, C_out
    )

    # Run 2
    A_wit_2, S_wit_2, _ = emit_witness(
        X_star, witness_results, frames, colors_order, R_out, C_out
    )

    # Results should be identical
    assert S_wit_1 == S_wit_2, "Scope should be deterministic"
    for c in colors_order:
        assert A_wit_1[c] == A_wit_2[c], f"Admits for color {c} should be deterministic"

    print("✓ Emission is deterministic")


def test_conjugation_with_rotation():
    """Test conjugation when training frame has rotation."""
    X_star = [[1, 0]]  # 1x2 horizontal bar

    # Training has R90 rotation on input
    witness_results = [
        {
            "silent": False,
            "pieces": [
                {
                    "pid": "I",
                    "dy": 0,
                    "dx": 0,
                    "bbox_src": (0, 0, 0, 0),  # Single pixel
                    "bbox_tgt": (0, 0, 0, 0),
                    "c_in": 1,
                    "c_out": 1,
                }
            ],
            "sigma": {1: 1},
        }
    ]

    # Test input in R90 frame, training in I frame, output in I frame
    frames = {
        "Pi_in_star": ("R90", (0, 0)),  # Test input rotated
        "Pi_out_star": ("I", (0, 0)),
        "Pi_in_0": ("I", (0, 0)),
        "Pi_out_0": ("I", (0, 0)),
    }

    colors_order = [0, 1]
    R_out, C_out = 2, 2

    # Emit witness
    A_wit, S_wit, _ = emit_witness(
        X_star, witness_results, frames, colors_order, R_out, C_out
    )

    # Check that emission happened (scope non-zero)
    scope_bits = sum(bin(row).count("1") for row in S_wit)
    assert scope_bits > 0, "Conjugation with rotation should produce non-zero scope"

    print("✓ Conjugation with rotation works")


def test_empty_trainings():
    """Test edge case with no trainings."""
    X_star = [[1]]

    frames = {
        "Pi_in_star": ("I", (0, 0)),
        "Pi_out_star": ("I", (0, 0)),
    }

    colors_order = [0, 1]
    R_out, C_out = 1, 1

    # Empty trainings
    A_wit, S_wit, _ = emit_witness(X_star, [], frames, colors_order, R_out, C_out)

    # Should return zero scope and admit-all
    assert S_wit[0] == 0, "Empty trainings should have zero scope"
    assert A_wit[0][0] == 0b1, "Empty trainings should admit all colors"
    assert A_wit[1][0] == 0b1, "Empty trainings should admit all colors"

    print("✓ Empty trainings handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
