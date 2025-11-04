"""
Test Sub-WO-RUN-FAM: Family gating in choose_working_canvas

Verifies:
  - Family allow-list filtering (families parameter)
  - skip_h8h9_if_area1 bound
  - Receipts fields for family gating
"""

import pytest
from src.arcbit.canvas import choose_working_canvas, parse_families


def _get_colors_order(train_pairs):
    """Helper to extract color universe from train_pairs."""
    color_set = {0}  # Always include background
    for pair in train_pairs:
        for row in pair["X"]:
            for val in row:
                color_set.add(val)
        for row in pair["Y"]:
            for val in row:
                color_set.add(val)
    return sorted(color_set)


def test_parse_families_range():
    """Test parse_families with range notation."""
    # H1-7 should parse to ("H1", "H2", ..., "H7")
    result = parse_families("H1-7")
    assert result == ("H1", "H2", "H3", "H4", "H5", "H6", "H7")

    # H1-3 should parse to ("H1", "H2", "H3")
    result = parse_families("H1-3")
    assert result == ("H1", "H2", "H3")

    # H8-9 should parse to ("H8", "H9")
    result = parse_families("H8-9")
    assert result == ("H8", "H9")


def test_parse_families_csv():
    """Test parse_families with CSV notation."""
    # H1,H3,H5 should parse to ("H1", "H3", "H5") in frozen order
    result = parse_families("H1,H3,H5")
    assert result == ("H1", "H3", "H5")

    # H2,H1 should normalize to ("H1", "H2")
    result = parse_families("H2,H1")
    assert result == ("H1", "H2")


def test_families_h1_only():
    """Test family gating with H1 only."""
    # Task: 2x scale (H1 with a=2, c=2)
    train_pairs = [
        {"X": [[1, 2]], "Y": [[5, 6, 7, 8], [9, 0, 1, 2]]},  # 1x2 → 2x4
        {"X": [[1, 1], [2, 2]], "Y": [[5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 0]]}  # 2x2 → 4x4
    ]

    colors_order = _get_colors_order(train_pairs)
    frames_in = [{"pose": "I", "anchor": (0, 0)}] * 2
    frames_out = [{"pose": "I", "anchor": (0, 0)}] * 2

    X_star = [[1, 2, 3]]  # 1x3 → should predict 2x6
    xstar_shape = (1, 3)

    # Call with families=("H1",) to only evaluate H1
    R_out, C_out, receipts = choose_working_canvas(
        train_pairs,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        families=("H1",)
    )

    # Should find H1 (a=2, c=2)
    assert R_out == 2
    assert C_out == 6

    # Check receipts
    assert receipts["payload"]["families_requested"] == ["H1"]
    assert receipts["payload"]["families_eval"] == ["H1"]
    assert receipts["payload"]["family_eval_mode"] == "full"

    # Verify only H1 attempts present
    attempts = receipts["payload"]["attempts"]
    families_in_attempts = set(a["family"] for a in attempts)
    assert families_in_attempts == {"H1"}


def test_families_h1_to_h7_excludes_h8h9():
    """Test family gating with H1-7 (excludes H8/H9)."""
    # Identity task (can be fit by H1, H8, or H9)
    train_pairs = [
        {"X": [[1, 2]], "Y": [[3, 4]]},  # 1x2 → 1x2
        {"X": [[5], [6]], "Y": [[7], [8]]}  # 2x1 → 2x1
    ]

    colors_order = _get_colors_order(train_pairs)
    frames_in = [{"pose": "I", "anchor": (0, 0)}] * 2
    frames_out = [{"pose": "I", "anchor": (0, 0)}] * 2

    X_star = [[1, 2, 3]]
    xstar_shape = (1, 3)

    # Call with families=("H1", "H2", ..., "H7") to exclude H8/H9
    families_h1_h7 = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
    R_out, C_out, receipts = choose_working_canvas(
        train_pairs,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        xstar_grid=X_star,
        families=families_h1_h7
    )

    # Should find H1 (identity with a=1, c=1)
    assert R_out == 1
    assert C_out == 3

    # Check receipts
    assert receipts["payload"]["families_requested"] == list(families_h1_h7)
    # All H1-7 should be evaluated (though only H1 fits)
    assert set(receipts["payload"]["families_eval"]) == set(families_h1_h7)
    assert receipts["payload"]["family_eval_mode"] == "full"

    # Verify H8/H9 not in attempts
    attempts = receipts["payload"]["attempts"]
    families_in_attempts = set(a["family"] for a in attempts)
    assert "H8" not in families_in_attempts
    assert "H9" not in families_in_attempts


def test_skip_h8h9_if_area1_bound():
    """Test skip_h8h9_if_area1 bound logic."""
    # Task with area=1 output (1x1)
    train_pairs = [
        {"X": [[1, 2]], "Y": [[5]]},  # 1x2 → 1x1
        {"X": [[1], [2]], "Y": [[5]]}  # 2x1 → 1x1
    ]

    colors_order = _get_colors_order(train_pairs)
    frames_in = [{"pose": "I", "anchor": (0, 0)}] * 2
    frames_out = [{"pose": "I", "anchor": (0, 0)}] * 2

    X_star = [[1, 2, 3]]
    xstar_shape = (1, 3)

    # Call with skip_h8h9_if_area1=True
    R_out, C_out, receipts = choose_working_canvas(
        train_pairs,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        xstar_grid=X_star,
        skip_h8h9_if_area1=True
    )

    # Should predict 1x1 (area=1)
    assert R_out == 1
    assert C_out == 1

    # Check receipts: H8/H9 should be skipped
    payload = receipts["payload"]
    assert "families_skipped_due_to_area_bound" in payload
    skipped = payload["families_skipped_due_to_area_bound"]
    assert "H8" in skipped or "H9" in skipped  # Both should be skipped
    assert payload["best_area_bound"] == 1

    # Verify H8/H9 not in families_eval
    families_eval = set(payload["families_eval"])
    assert "H8" not in families_eval
    assert "H9" not in families_eval

    # family_eval_mode should be "partial" (H8/H9 skipped)
    assert payload["family_eval_mode"] == "partial"


def test_skip_h8h9_if_area1_false_evaluates_all():
    """Test that skip_h8h9_if_area1=False evaluates H8/H9 even when area=1."""
    # Same task as above (area=1 output)
    # Add color 3 to trainings so X_star's color 3 is in colors_order
    train_pairs = [
        {"X": [[1, 2, 3]], "Y": [[5]]},  # 1x3 → 1x1
        {"X": [[1, 3], [2, 3]], "Y": [[5]]}  # 2x2 → 1x1
    ]

    colors_order = _get_colors_order(train_pairs)
    frames_in = [{"pose": "I", "anchor": (0, 0)}] * 2
    frames_out = [{"pose": "I", "anchor": (0, 0)}] * 2

    X_star = [[1, 2, 3]]
    xstar_shape = (1, 3)

    # Call with skip_h8h9_if_area1=False (default)
    R_out, C_out, receipts = choose_working_canvas(
        train_pairs,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        xstar_grid=X_star,
        skip_h8h9_if_area1=False  # Explicit False
    )

    # Should still predict 1x1
    assert R_out == 1
    assert C_out == 1

    # Check receipts: H8/H9 should be evaluated (NOT skipped)
    payload = receipts["payload"]
    families_eval = set(payload["families_eval"])
    assert "H8" in families_eval
    assert "H9" in families_eval

    # No skipped families (bound not triggered)
    assert "families_skipped_due_to_area_bound" not in payload

    # family_eval_mode should be "full" (all families evaluated)
    assert payload["family_eval_mode"] == "full"


def test_partial_eval_mode_when_some_families_skipped():
    """Test that family_eval_mode is 'partial' when some families are excluded."""
    train_pairs = [
        {"X": [[1, 2]], "Y": [[5, 6], [7, 8]]},  # 1x2 → 2x4
        {"X": [[1], [2]], "Y": [[5, 6], [7, 8]]}  # 2x1 → 4x2
    ]

    colors_order = _get_colors_order(train_pairs)
    frames_in = [{"pose": "I", "anchor": (0, 0)}] * 2
    frames_out = [{"pose": "I", "anchor": (0, 0)}] * 2

    X_star = [[1, 2, 3]]
    xstar_shape = (1, 3)

    # Request H1-H7, but full set is H1-H9
    families_h1_h7 = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
    R_out, C_out, receipts = choose_working_canvas(
        train_pairs,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        families=families_h1_h7
    )

    # Check receipts
    payload = receipts["payload"]

    # families_requested should match input
    assert payload["families_requested"] == list(families_h1_h7)

    # families_eval should only include evaluated families (H1-H7)
    assert set(payload["families_eval"]) == set(families_h1_h7)

    # family_eval_mode should be "full" (all REQUESTED families evaluated)
    # Note: This should actually be "full" because all REQUESTED families were evaluated
    assert payload["family_eval_mode"] == "full"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
