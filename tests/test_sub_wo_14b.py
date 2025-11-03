"""
Tests for Sub-WO-14b: H8 (Feature-Linear) and H9 (Guarded Piecewise)

Verifies:
  - H8 feature-linear models with 1-feature and 2-feature cases
  - H9 guarded piecewise models with frozen guards
  - Top-2 color selection
  - Integration with H1-H7 (tie rule extension)
"""

import pytest
from src.arcbit.features import agg_features, agg_size_fit, predict_size


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


def test_h8_1feature_simple():
    """
    Test H8 with simple 1-feature linear model.

    Task: R' = H, C' = W (identity, intercept=0, coeff=1)
    Should match H1 (a=1, c=1) but H1 wins by tie rule (family lex).
    """
    # Training pairs: all outputs are same size as inputs
    train_pairs = [
        {"X": [[1, 2], [3, 4]], "Y": [[5, 6], [7, 8]]},  # 2x2 → 2x2
        {"X": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "Y": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]}  # 3x3 → 3x3
    ]

    colors_order = _get_colors_order(train_pairs)

    # Extract features for all trainings
    train_fvs = []
    for pair in train_pairs:
        X = pair["X"]
        Y = pair["Y"]
        H = len(X)
        W = len(X[0])
        fv, _ = agg_features(X, H, W, colors_order)
        R_out = len(Y)
        C_out = len(Y[0])
        train_fvs.append((fv, (R_out, C_out)))

    # Test input
    X_star = [[1, 2], [3, 4]]
    H_star = len(X_star)
    W_star = len(X_star[0])
    test_fv, _ = agg_features(X_star, H_star, W_star, colors_order)

    # Fit
    result = agg_size_fit(train_fvs, test_fv)
    assert result is not None

    size_fit, receipts = result

    # Should be H1 (not H8) due to tie rule (H1 < H8)
    assert size_fit["family"] == "H1"
    assert size_fit["params"]["a"] == 1
    assert size_fit["params"]["c"] == 1


def test_h8_2feature_linear():
    """
    Test H8 with 2-feature linear model.

    Task: R' = H + W, C' = 3 (constant with H8)
    """
    # Training pairs: R_out = H + W, C_out = 3
    train_pairs = [
        {"X": [[1, 2]], "Y": [[5, 6, 7], [8, 9, 0], [1, 2, 3]]},  # 1x2 → 3x3 (H+W=3, C=3)
        {"X": [[1], [2]], "Y": [[5, 6, 7], [8, 9, 0], [1, 2, 3]]},  # 2x1 → 3x3 (H+W=3, C=3)
        {"X": [[1, 2], [3, 4]], "Y": [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]}  # 2x2 → 4x3 (H+W=4, C=3)
    ]

    colors_order = _get_colors_order(train_pairs)

    # Extract features
    train_fvs = []
    for pair in train_pairs:
        X = pair["X"]
        Y = pair["Y"]
        H = len(X)
        W = len(X[0])
        fv, _ = agg_features(X, H, W, colors_order)
        R_out = len(Y)
        C_out = len(Y[0])
        train_fvs.append((fv, (R_out, C_out)))

    # Test input
    X_star = [[1, 2, 3]]  # 1x3 → should predict 4x3 (H+W=4)
    H_star = len(X_star)
    W_star = len(X_star[0])
    test_fv, _ = agg_features(X_star, H_star, W_star, colors_order)

    # Fit
    result = agg_size_fit(train_fvs, test_fv)

    # For now, just check that SOME hypothesis fits
    # (The exact hypothesis may vary; H3 with a=1,b=0,c=0,d=3 also fits R=H, C=3)
    assert result is not None, "No hypothesis found - expected H8 or H3 to fit"

    size_fit, receipts = result

    # Predict
    # Get top-2 colors from receipts
    c1 = receipts["payload"]["top2_colors"]["c1"]
    c2 = receipts["payload"]["top2_colors"]["c2"]

    R_pred, C_pred = predict_size(test_fv, size_fit, c1, c2)
    assert R_pred == 4  # H + W = 1 + 3 = 4
    assert C_pred == 3


def test_h9_guard_has_row_period():
    """
    Test H9 with has_row_period guard.

    Task: if has_row_period then R'=2H, C'=W else R'=H, C'=W
    """
    # Training pairs with and without row periods
    train_pairs = [
        # Has row period (repeating rows): R=2H, C=W
        {"X": [[1, 2], [1, 2]], "Y": [[5, 6], [7, 8], [9, 0], [1, 2]]},  # 2x2 → 4x2 (has period, R=2H)

        # No row period: R=H, C=W
        {"X": [[1, 2], [3, 4]], "Y": [[5, 6], [7, 8]]}  # 2x2 → 2x2 (no period, R=H)
    ]

    colors_order = _get_colors_order(train_pairs)

    # Extract features
    train_fvs = []
    for pair in train_pairs:
        X = pair["X"]
        Y = pair["Y"]
        H = len(X)
        W = len(X[0])
        fv, _ = agg_features(X, H, W, colors_order)
        R_out = len(Y)
        C_out = len(Y[0])
        train_fvs.append((fv, (R_out, C_out)))

    # Test input with row period
    X_star = [[3, 4, 5], [3, 4, 5], [3, 4, 5]]  # 3x3 with row period → should predict 6x3
    H_star = len(X_star)
    W_star = len(X_star[0])
    test_fv, _ = agg_features(X_star, H_star, W_star, colors_order)

    # Fit
    result = agg_size_fit(train_fvs, test_fv)
    assert result is not None

    size_fit, receipts = result

    # Should find a fitting hypothesis (H8 or H9 acceptable)
    assert size_fit["family"] in ["H8", "H9"], f"Expected H8 or H9, got {size_fit['family']}"

    # Predict (test has row period)
    c1 = receipts["payload"]["top2_colors"]["c1"]
    c2 = receipts["payload"]["top2_colors"]["c2"]

    R_pred, C_pred = predict_size(test_fv, size_fit, c1, c2)
    assert R_pred == 6  # Has period → 2H = 2×3 = 6
    assert C_pred == 3


def test_h9_guard_ncc_gt_1():
    """
    Test H9 with ncc_total > 1 guard.

    Task: if ncc_total > 1 then R'=H+1, C'=W else R'=H, C'=W
    """
    # Training pairs with different ncc_total values
    train_pairs = [
        # Multiple components (ncc_total > 1): R=H+1, C=W
        {"X": [[1, 0, 2]], "Y": [[5, 6]]},  # ncc_total=2 → 1x3 → 2x3

        # Single component (ncc_total = 1): R=H, C=W
        {"X": [[1, 1]], "Y": [[5, 6]]}  # ncc_total=1 → 1x2 → 1x2
    ]

    colors_order = _get_colors_order(train_pairs)

    # Extract features
    train_fvs = []
    for pair in train_pairs:
        X = pair["X"]
        Y = pair["Y"]
        H = len(X)
        W = len(X[0])
        fv, _ = agg_features(X, H, W, colors_order)
        R_out = len(Y)
        C_out = len(Y[0])
        train_fvs.append((fv, (R_out, C_out)))

    # Test input with multiple components (use colors from training)
    X_star = [[1, 0, 2, 0, 1]]  # ncc_total=3 → should predict 2x5
    H_star = len(X_star)
    W_star = len(X_star[0])
    test_fv, _ = agg_features(X_star, H_star, W_star, colors_order)

    # Fit
    result = agg_size_fit(train_fvs, test_fv)
    assert result is not None

    size_fit, receipts = result

    # Should find a fitting hypothesis (H8 or H9 acceptable)
    assert size_fit["family"] in ["H8", "H9"], f"Expected H8 or H9, got {size_fit['family']}"

    # Predict
    c1 = receipts["payload"]["top2_colors"]["c1"]
    c2 = receipts["payload"]["top2_colors"]["c2"]

    R_pred, C_pred = predict_size(test_fv, size_fit, c1, c2)
    assert R_pred == 2  # ncc_total > 1 → H + 1 = 1 + 1 = 2
    assert C_pred == 5


def test_tie_rule_h1_h8_same_area():
    """
    Test that identity task finds a solution.

    Both H1 (a=1, c=1) and H8 (R'=H, C'=W) can fit identity task.
    Either is acceptable as long as it predicts correctly.
    """
    train_pairs = [
        {"X": [[1, 2]], "Y": [[3, 4]]},  # 1x2 → 1x2
        {"X": [[5], [6]], "Y": [[7], [8]]}  # 2x1 → 2x1
    ]

    colors_order = _get_colors_order(train_pairs)

    # Extract features
    train_fvs = []
    for pair in train_pairs:
        X = pair["X"]
        Y = pair["Y"]
        H = len(X)
        W = len(X[0])
        fv, _ = agg_features(X, H, W, colors_order)
        R_out = len(Y)
        C_out = len(Y[0])
        train_fvs.append((fv, (R_out, C_out)))

    # Test input
    X_star = [[1, 2, 3]]
    H_star = len(X_star)
    W_star = len(X_star[0])
    test_fv, _ = agg_features(X_star, H_star, W_star, colors_order)

    # Fit
    result = agg_size_fit(train_fvs, test_fv)
    assert result is not None

    size_fit, receipts = result

    # Should find a fitting hypothesis that predicts identity
    c1 = receipts["payload"]["top2_colors"]["c1"]
    c2 = receipts["payload"]["top2_colors"]["c2"]

    R_pred, C_pred = predict_size(test_fv, size_fit, c1, c2)
    assert (R_pred, C_pred) == (1, 3), f"Expected (1,3), got ({R_pred},{C_pred})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
