"""
WO-04a: Working Canvas Provider (Extended with H8/H9)

Single working canvas size selection via frozen T9 hypothesis class (H1-H9).
Trainings-only evaluation, fail-closed, receipts-first.

Spec: WO-04a (v1.6) + Sub-WO-04a-H8H9
"""

from typing import Dict, List, Tuple, TypedDict, Optional, Any
from .core import Receipts
from .core.hashing import blake3_hash
from .kernel.planes import pack_grid_to_planes, order_colors
from .kernel.period import minimal_period_row, minimal_period_1d
from .kernel.components import components
import json


# ============================================================================
# Exception Classes
# ============================================================================

class SizeUndetermined(Exception):
    """Raised when no hypothesis fits all trainings."""
    def __init__(self, message: str, receipts: Dict):
        super().__init__(message)
        self.receipts = receipts


# ============================================================================
# Type Definitions
# ============================================================================

class SizeFeatures(TypedDict):
    """Features for size prediction (extended for H8/H9)."""
    H: int
    W: int
    periods: Dict[str, Optional[int]]  # {row_min, col_min, lcm_r, lcm_c, gcd_r, gcd_c}
    sum_nonzero: int  # Σ h(c) for c≠0 (Sub-WO-04a-H8H9)
    ncc_total: int  # Σ n_cc(c) for c≠0 (Sub-WO-04a-H8H9)
    counts: Dict[int, int]  # {color: count} (Sub-WO-04a-H8H9)


# ============================================================================
# Main Entry Point
# ============================================================================

def choose_working_canvas(
    train_pairs: List[Dict],
    frames_in: List[Dict],
    frames_out: List[Dict],
    xstar_shape: Tuple[int, int],
    colors_order: List[int],
    xstar_grid: Optional[List[List[int]]] = None
) -> Tuple[int, int, Dict]:
    """
    Choose single working canvas size (R_out, C_out) via frozen H1-H9 evaluation.

    Args:
        train_pairs: List of {"X": Grid, "Y": Grid} training pairs.
        frames_in: List of frame objects for inputs (receipts only).
        frames_out: List of frame objects for outputs (receipts only).
        xstar_shape: (H*, W*) test input dimensions.
        colors_order: Global color universe (sorted, includes 0).
        xstar_grid: Optional test input grid for H8/H9 feature extraction (Sub-WO-04a-H8H9).

    Returns:
        Tuple of (R_out, C_out, receipts_dict).

    Raises:
        SizeUndetermined: If no hypothesis fits all trainings.

    Spec:
        WO-04a + Sub-WO-04a-H8H9: Trainings-only size prediction, fail-closed.
        Uses OUTPUT periods for H5 (not input periods).
        H8: Feature-linear with frozen basis φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
        H9: Guarded piecewise over frozen guards and clause families {H1,H2,H6,H7}
    """
    receipts = Receipts("working_canvas")

    # A) Validate inputs
    if not train_pairs:
        receipts.put("error", "Zero trainings provided")
        receipts_bundle = receipts.digest()
        raise SizeUndetermined("Cannot determine canvas size with zero trainings", receipts_bundle)

    num_trainings = len(train_pairs)
    receipts.put("num_trainings", num_trainings)

    # B) Extract sizes and features for each training
    sizes_in = []
    sizes_out = []
    features_in = []   # For receipts/provenance only
    features_out = []  # For H5 period extraction

    for i, pair in enumerate(train_pairs):
        X = pair["X"]
        Y = pair["Y"]

        H_in = len(X)
        W_in = len(X[0]) if X else 0
        H_out = len(Y)
        W_out = len(Y[0]) if Y else 0

        sizes_in.append((H_in, W_in))
        sizes_out.append((H_out, W_out))

        # Extract features from INPUT (for receipts only)
        # Use global colors_order (not per-grid, as some grids may lack color 0)
        feat_in = _extract_size_features(X, H_in, W_in, colors_order)
        features_in.append(feat_in)

        # Extract features from OUTPUT (for H5 period detection)
        # Use global colors_order (not per-grid, as some grids may lack color 0)
        feat_out = _extract_size_features(Y, H_out, W_out, colors_order)
        features_out.append(feat_out)

    # Store features hashes (from OUTPUT, as H5 uses output periods)
    features_hash_per_training = []
    for feat in features_out:
        feat_json = json.dumps(feat, sort_keys=True)
        features_hash_per_training.append(blake3_hash(feat_json.encode('utf-8')))

    receipts.put("features_hash_per_training", features_hash_per_training)

    H_star, W_star = xstar_shape
    receipts.put("test_input_shape", {"H": H_star, "W": W_star})

    # C) Enumerate hypotheses H1-H7 in frozen order
    attempts = []
    candidates = []  # (family, params, test_area)

    # H1: R = a·H, C = c·W | a,c ∈ {1..8}
    for a in range(1, 9):
        for c in range(1, 9):
            params = {"a": a, "c": c}
            fit_all, ok_train_ids = _check_fit_H1(sizes_in, sizes_out, params)

            attempts.append({
                "family": "H1",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = a * H_star
                C_test = c * W_star
                test_area = R_test * C_test
                candidates.append(("H1", params, test_area))

    # H2: R = H + b, C = W + d | b,d ∈ {0..16}
    for b in range(0, 17):
        for d in range(0, 17):
            params = {"b": b, "d": d}
            fit_all, ok_train_ids = _check_fit_H2(sizes_in, sizes_out, params)

            attempts.append({
                "family": "H2",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = H_star + b
                C_test = W_star + d
                test_area = R_test * C_test
                candidates.append(("H2", params, test_area))

    # H3: R = a·H + b, C = c·W + d | a,c ∈ {1..8}, b,d ∈ {0..16}
    # Enumeration order: a → c → b → d (frozen)
    for a in range(1, 9):
        for c in range(1, 9):
            for b in range(0, 17):
                for d in range(0, 17):
                    params = {"a": a, "b": b, "c": c, "d": d}
                    fit_all, ok_train_ids = _check_fit_H3(sizes_in, sizes_out, params)

                    attempts.append({
                        "family": "H3",
                        "params": params,
                        "ok_train_ids": ok_train_ids,
                        "fit_all": fit_all
                    })

                    if fit_all:
                        R_test = a * H_star + b
                        C_test = c * W_star + d
                        test_area = R_test * C_test
                        candidates.append(("H3", params, test_area))

    # H4: R = R₀, C = C₀ | R₀,C₀ ∈ {1..30}
    for R0 in range(1, 31):
        for C0 in range(1, 31):
            params = {"R0": R0, "C0": C0}
            fit_all, ok_train_ids = _check_fit_H4(sizes_in, sizes_out, params)

            attempts.append({
                "family": "H4",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = R0
                C_test = C0
                test_area = R_test * C_test
                candidates.append(("H4", params, test_area))

    # H5: R = kr·lcm_r, C = kc·lcm_c | kr,kc ∈ {1..8}
    # Uses OUTPUT periods with identity rule (H_in/W_in when periods are None)
    # For test prediction: uses the COMMON output periods from trainings
    for kr in range(1, 9):
        for kc in range(1, 9):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids, common_lcm_r, common_lcm_c = _check_fit_H5(
                features_out, sizes_out, sizes_in, params
            )

            attempts.append({
                "family": "H5",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                # Test prediction uses common periods from training outputs
                # Identity rule: if common period is None, use test input size
                R_test = (kr * common_lcm_r) if common_lcm_r is not None else H_star
                C_test = (kc * common_lcm_c) if common_lcm_c is not None else W_star
                test_area = R_test * C_test
                candidates.append(("H5", params, test_area, common_lcm_r, common_lcm_c))

    # H6: R = ⌊H/kr⌋, C = ⌊W/kc⌋ | kr,kc ∈ {2..5}
    for kr in range(2, 6):
        for kc in range(2, 6):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids = _check_fit_H6(sizes_in, sizes_out, params)

            attempts.append({
                "family": "H6",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = H_star // kr
                C_test = W_star // kc
                test_area = R_test * C_test
                candidates.append(("H6", params, test_area))

    # H7: R = ⌈H/kr⌉, C = ⌈W/kc⌉ | kr,kc ∈ {2..5}
    for kr in range(2, 6):
        for kc in range(2, 6):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids = _check_fit_H7(sizes_in, sizes_out, params)

            attempts.append({
                "family": "H7",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = _ceil_div(H_star, kr)
                C_test = _ceil_div(W_star, kc)
                test_area = R_test * C_test
                candidates.append(("H7", params, test_area))

    # ========================================================================
    # Sub-WO-04a-H8H9: Compute top-2 colors for H8 features
    # ========================================================================

    c1, c2 = _get_top2_colors(features_in)
    receipts.put("top2_colors", {"c1": c1, "c2": c2})

    # ========================================================================
    # H8: Feature-Linear (Sub-WO-04a-H8H9) with early stopping
    # ========================================================================
    # Model: R' = α₀ + Σ αᵢ·fᵢ, C' = β₀ + Σ βᵢ·fᵢ
    # Features: φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
    # Bounds: α₀,β₀ ∈ [0..32], αᵢ,βᵢ ∈ {-3,-2,-1,1,2,3}
    # Enumeration: rows → cols, 0-feature → 1-feature → 2-feature
    # Early stopping: only enumerate col models if row model fits

    coeffs = [-3, -2, -1, 1, 2, 3]

    # Helper function to generate model parameter generator
    def generate_models():
        # 0-feature models first (constant-only)
        for intercept in range(0, 33):  # 0..32
            yield {
                "intercept": intercept,
                "feat1_idx": None,
                "feat1_coeff": None,
                "feat2_idx": None,
                "feat2_coeff": None
            }

        # 1-feature models second
        for intercept in range(0, 33):  # 0..32
            for feat_idx in range(1, 7):  # 1..6 (skip φ[0]=1 constant)
                for coeff in coeffs:
                    yield {
                        "intercept": intercept,
                        "feat1_idx": feat_idx,
                        "feat1_coeff": coeff,
                        "feat2_idx": None,
                        "feat2_coeff": None
                    }

        # 2-feature models third
        for intercept in range(0, 33):  # 0..32
            for feat1_idx in range(1, 7):  # 1..6
                for feat2_idx in range(feat1_idx + 1, 7):  # feat2 > feat1
                    for coeff1 in coeffs:
                        for coeff2 in coeffs:
                            yield {
                                "intercept": intercept,
                                "feat1_idx": feat1_idx,
                                "feat1_coeff": coeff1,
                                "feat2_idx": feat2_idx,
                                "feat2_coeff": coeff2
                            }

    # ========================================================================
    # H8 Performance Fix: 3-Phase Enumeration (exploits R'/C' independence)
    # ========================================================================
    # Phase 1: Filter row models independently (check R only)
    # Phase 2: Filter col models independently (check C only)
    # Phase 3: Combine Rows_OK × Cols_OK (guaranteed fit_all)

    # Phase 1: Enumerate and filter row models
    rows_ok = []
    rows_total = 0
    for row_params in generate_models():
        rows_total += 1
        row_params_with_axis = {**row_params, "axis": "rows"}
        fit_rows, ok_train_ids_rows = _check_fit_H8(features_in, sizes_out, row_params_with_axis, c1, c2)

        if fit_rows:
            rows_ok.append((row_params, ok_train_ids_rows))

    # Phase 2: Enumerate and filter col models
    cols_ok = []
    cols_total = 0
    for col_params in generate_models():
        cols_total += 1
        col_params_with_axis = {**col_params, "axis": "cols"}
        fit_cols, ok_train_ids_cols = _check_fit_H8(features_in, sizes_out, col_params_with_axis, c1, c2)

        if fit_cols:
            cols_ok.append((col_params, ok_train_ids_cols))

    # Phase 3: Combine Rows_OK × Cols_OK (frozen lex order: rows outer, cols inner)
    for row_params, ok_train_ids_rows in rows_ok:
        for col_params, ok_train_ids_cols in cols_ok:
            # Both sides fit, so combined model fits all
            params = {
                "row_model": row_params,
                "col_model": col_params
            }

            attempts.append({
                "family": "H8",
                "params": params,
                "ok_train_ids": ok_train_ids_rows,  # Same for both sides (all trainings)
                "fit_all": True
            })

            # Predict test output size (only if xstar_grid provided)
            if xstar_grid is None:
                continue  # Skip H8 candidates if test grid not provided

            fv_star = _extract_size_features(xstar_grid, H_star, W_star, colors_order)

            count_c1_star = fv_star["counts"].get(c1, 0)
            count_c2_star = fv_star["counts"].get(c2, 0)

            features_star = [1, H_star, W_star, fv_star["sum_nonzero"], fv_star["ncc_total"], count_c1_star, count_c2_star]

            # R prediction
            R_test = row_params["intercept"]
            if row_params["feat1_idx"] is not None:
                R_test += row_params["feat1_coeff"] * features_star[row_params["feat1_idx"]]
            if row_params["feat2_idx"] is not None:
                R_test += row_params["feat2_coeff"] * features_star[row_params["feat2_idx"]]

            # C prediction
            C_test = col_params["intercept"]
            if col_params["feat1_idx"] is not None:
                C_test += col_params["feat1_coeff"] * features_star[col_params["feat1_idx"]]
            if col_params["feat2_idx"] is not None:
                C_test += col_params["feat2_coeff"] * features_star[col_params["feat2_idx"]]

            # Validate: sizes must be positive
            if R_test > 0 and C_test > 0:
                test_area = R_test * C_test
                candidates.append(("H8", params, test_area))

    # ========================================================================
    # H9: Guarded Piecewise (Sub-WO-04a-H8H9)
    # ========================================================================
    # Model: if guard(X) then F_true(H,W) else F_false(H,W)
    # Guards: {has_row_period, has_col_period, ncc_total>1, sum_nonzero>⌊H·W/2⌋, H>W}
    # Clauses: {H1, H2, H6, H7}

    guards = ["has_row_period", "has_col_period", "ncc_gt_1", "sum_gt_half", "h_gt_w"]
    clause_families = ["H1", "H2", "H6", "H7"]

    # Build clause parameter spaces
    clause_param_spaces = {
        "H1": [{"a": a, "c": c} for a in range(1, 9) for c in range(1, 9)],
        "H2": [{"b": b, "d": d} for b in range(0, 17) for d in range(0, 17)],
        "H6": [{"kr": kr, "kc": kc} for kr in range(2, 6) for kc in range(2, 6)],
        "H7": [{"kr": kr, "kc": kc} for kr in range(2, 6) for kc in range(2, 6)]
    }

    # Enumerate guards → true_clause → false_clause
    for guard in guards:
        for true_family in clause_families:
            for true_params in clause_param_spaces[true_family]:
                for false_family in clause_families:
                    for false_params in clause_param_spaces[false_family]:
                        params = {
                            "guard": guard,
                            "true_family": true_family,
                            "true_params": true_params,
                            "false_family": false_family,
                            "false_params": false_params
                        }

                        fit_all, ok_train_ids = _check_fit_H9(features_in, sizes_out, params)

                        attempts.append({
                            "family": "H9",
                            "params": params,
                            "ok_train_ids": ok_train_ids,
                            "fit_all": fit_all
                        })

                        if fit_all:
                            # Predict test output size
                            # Extract test features (only if xstar_grid provided)
                            if xstar_grid is None:
                                continue  # Skip H9 candidates if test grid not provided

                            fv_star = _extract_size_features(xstar_grid, H_star, W_star, colors_order)

                            # Evaluate guard on test input
                            if guard == "has_row_period":
                                guard_result = (fv_star["periods"]["lcm_r"] is not None)
                            elif guard == "has_col_period":
                                guard_result = (fv_star["periods"]["lcm_c"] is not None)
                            elif guard == "ncc_gt_1":
                                guard_result = (fv_star["ncc_total"] > 1)
                            elif guard == "sum_gt_half":
                                half_area = (H_star * W_star) // 2
                                guard_result = (fv_star["sum_nonzero"] > half_area)
                            elif guard == "h_gt_w":
                                guard_result = (H_star > W_star)

                            # Select clause
                            if guard_result:
                                family = true_family
                                clause_params = true_params
                            else:
                                family = false_family
                                clause_params = false_params

                            # Predict
                            R_test, C_test = _predict_clause(fv_star, family, clause_params)

                            # Validate: sizes must be positive (skip models that predict non-positive sizes)
                            if R_test > 0 and C_test > 0:
                                test_area = R_test * C_test
                                candidates.append(("H9", params, test_area))

    receipts.put("attempts", attempts)
    receipts.put("total_candidates_checked", len(attempts))

    # Add H8 performance summary (audit coverage)
    receipts.put("attempts_summary", {
        "H8": {
            "rows_total": rows_total,
            "rows_ok": len(rows_ok),
            "cols_total": cols_total,
            "cols_ok": len(cols_ok),
            "pairs_evaluated": len(rows_ok) * len(cols_ok)
        }
    })

    # D) Apply tie rule to select winner
    if not candidates:
        # No hypothesis fits - find first counterexample
        first_counterexample = None
        for att in attempts:
            if not att["fit_all"] and att["ok_train_ids"] != list(range(num_trainings)):
                # Find first training that failed
                for i in range(num_trainings):
                    if i not in att["ok_train_ids"]:
                        # Compute what this hypothesis predicted
                        family = att["family"]
                        params = att["params"]
                        H_in, W_in = sizes_in[i]
                        H_exp, W_exp = sizes_out[i]

                        # Compute prediction based on family
                        if family == "H1":
                            R_pred = params["a"] * H_in
                            C_pred = params["c"] * W_in
                        elif family == "H2":
                            R_pred = H_in + params["b"]
                            C_pred = W_in + params["d"]
                        elif family == "H3":
                            R_pred = params["a"] * H_in + params["b"]
                            C_pred = params["c"] * W_in + params["d"]
                        elif family == "H4":
                            R_pred = params["R0"]
                            C_pred = params["C0"]
                        elif family == "H6":
                            R_pred = H_in // params["kr"]
                            C_pred = W_in // params["kc"]
                        elif family == "H7":
                            R_pred = _ceil_div(H_in, params["kr"])
                            C_pred = _ceil_div(W_in, params["kc"])
                        elif family == "H8":
                            # Feature-Linear model (use c1, c2 from enclosing scope)
                            fv_i = features_in[i]
                            count_c1_i = fv_i["counts"].get(c1, 0)
                            count_c2_i = fv_i["counts"].get(c2, 0)
                            features_i = [1, H_in, W_in, fv_i["sum_nonzero"], fv_i["ncc_total"], count_c1_i, count_c2_i]

                            # Predict R
                            row_params = params["row_model"]
                            R_pred = row_params["intercept"]
                            if row_params["feat1_idx"] is not None:
                                R_pred += row_params["feat1_coeff"] * features_i[row_params["feat1_idx"]]
                            if row_params["feat2_idx"] is not None:
                                R_pred += row_params["feat2_coeff"] * features_i[row_params["feat2_idx"]]

                            # Predict C
                            col_params = params["col_model"]
                            C_pred = col_params["intercept"]
                            if col_params["feat1_idx"] is not None:
                                C_pred += col_params["feat1_coeff"] * features_i[col_params["feat1_idx"]]
                            if col_params["feat2_idx"] is not None:
                                C_pred += col_params["feat2_coeff"] * features_i[col_params["feat2_idx"]]
                        elif family == "H9":
                            # Guarded Piecewise model
                            fv_i = features_in[i]

                            # Evaluate guard
                            guard_name = params["guard"]
                            if guard_name == "has_row_period":
                                guard_result = (fv_i["periods"]["lcm_r"] is not None)
                            elif guard_name == "has_col_period":
                                guard_result = (fv_i["periods"]["lcm_c"] is not None)
                            elif guard_name == "ncc_gt_1":
                                guard_result = (fv_i["ncc_total"] > 1)
                            elif guard_name == "sum_gt_half":
                                half_area = (H_in * W_in) // 2
                                guard_result = (fv_i["sum_nonzero"] > half_area)
                            elif guard_name == "h_gt_w":
                                guard_result = (H_in > W_in)
                            else:
                                guard_result = False

                            # Select clause
                            if guard_result:
                                clause_family = params["true_family"]
                                clause_params = params["true_params"]
                            else:
                                clause_family = params["false_family"]
                                clause_params = params["false_params"]

                            # Predict using clause
                            R_pred, C_pred = _predict_clause(fv_i, clause_family, clause_params)
                        else:
                            R_pred, C_pred = None, None

                        first_counterexample = {
                            "train_id": i,
                            "expected": [H_exp, W_exp],
                            "predicted": [R_pred, C_pred] if R_pred is not None else None,
                            "family": family,
                            "params": params
                        }
                        break
                if first_counterexample:
                    break

        receipts.put("winner", None)
        receipts.put("first_counterexample", first_counterexample)
        receipts_bundle = receipts.digest()

        raise SizeUndetermined(
            f"No hypothesis fits all {num_trainings} trainings",
            receipts_bundle
        )

    # Sort candidates by tie rule: (1) test_area, (2) family lex, (3) params lex
    # Note: H5 candidates have extra data (common_lcm_r, common_lcm_c), handle gracefully

    def params_to_sortable(params):
        """Convert params dict to sortable tuple, handling nested dicts and None values."""
        items = []
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                # Recursively convert nested dict
                items.append((k, params_to_sortable(v)))
            elif v is None:
                # None sorts before all other values (use (0, None))
                items.append((k, (0, None)))
            else:
                # Non-None values sort after None (use (1, value))
                items.append((k, (1, v)))
        return tuple(items)

    def candidate_sort_key(cand):
        if len(cand) == 3:
            # H1, H2, H3, H4, H6, H7, H8, H9
            family, params, test_area = cand
            return (test_area, family, params_to_sortable(params))
        else:
            # H5 with extra period data
            family, params, test_area, lcm_r, lcm_c = cand
            return (test_area, family, params_to_sortable(params))

    candidates.sort(key=candidate_sort_key)

    winner_candidate = candidates[0]
    winner_family = winner_candidate[0]
    winner_params = winner_candidate[1]
    winner_test_area = winner_candidate[2]

    # Extract H5 common periods if winner is H5
    h5_common_lcm_r = None
    h5_common_lcm_c = None
    if winner_family == "H5" and len(winner_candidate) == 5:
        h5_common_lcm_r = winner_candidate[3]
        h5_common_lcm_c = winner_candidate[4]

    # Compute final R_out, C_out
    if winner_family == "H1":
        R_out = winner_params["a"] * H_star
        C_out = winner_params["c"] * W_star
    elif winner_family == "H2":
        R_out = H_star + winner_params["b"]
        C_out = W_star + winner_params["d"]
    elif winner_family == "H3":
        R_out = winner_params["a"] * H_star + winner_params["b"]
        C_out = winner_params["c"] * W_star + winner_params["d"]
    elif winner_family == "H4":
        R_out = winner_params["R0"]
        C_out = winner_params["C0"]
    elif winner_family == "H5":
        # H5 uses common output periods from trainings with identity rule
        # If common period is None, use test input size (identity)
        kr = winner_params["kr"]
        kc = winner_params["kc"]
        R_out = (kr * h5_common_lcm_r) if h5_common_lcm_r is not None else H_star
        C_out = (kc * h5_common_lcm_c) if h5_common_lcm_c is not None else W_star
    elif winner_family == "H6":
        R_out = H_star // winner_params["kr"]
        C_out = W_star // winner_params["kc"]
    elif winner_family == "H7":
        R_out = _ceil_div(H_star, winner_params["kr"])
        C_out = _ceil_div(W_star, winner_params["kc"])
    elif winner_family == "H8":
        # Feature-Linear model
        if xstar_grid is None:
            raise ValueError("H8 winner requires xstar_grid for feature extraction")

        # Extract features from test input
        fv_star = _extract_size_features(xstar_grid, H_star, W_star, colors_order)

        # Use c1, c2 from enclosing scope (already computed)

        count_c1_star = fv_star["counts"].get(c1, 0)
        count_c2_star = fv_star["counts"].get(c2, 0)
        features_star = [1, H_star, W_star, fv_star["sum_nonzero"], fv_star["ncc_total"], count_c1_star, count_c2_star]

        # Predict R
        row_params = winner_params["row_model"]
        R_out = row_params["intercept"]
        if row_params["feat1_idx"] is not None:
            R_out += row_params["feat1_coeff"] * features_star[row_params["feat1_idx"]]
        if row_params["feat2_idx"] is not None:
            R_out += row_params["feat2_coeff"] * features_star[row_params["feat2_idx"]]

        # Predict C
        col_params = winner_params["col_model"]
        C_out = col_params["intercept"]
        if col_params["feat1_idx"] is not None:
            C_out += col_params["feat1_coeff"] * features_star[col_params["feat1_idx"]]
        if col_params["feat2_idx"] is not None:
            C_out += col_params["feat2_coeff"] * features_star[col_params["feat2_idx"]]
    elif winner_family == "H9":
        # Guarded Piecewise model
        if xstar_grid is None:
            raise ValueError("H9 winner requires xstar_grid for feature extraction")

        # Extract features from test input
        fv_star = _extract_size_features(xstar_grid, H_star, W_star, colors_order)

        # Evaluate guard
        guard_name = winner_params["guard"]
        if guard_name == "has_row_period":
            guard_result = (fv_star["periods"]["lcm_r"] is not None)
        elif guard_name == "has_col_period":
            guard_result = (fv_star["periods"]["lcm_c"] is not None)
        elif guard_name == "ncc_gt_1":
            guard_result = (fv_star["ncc_total"] > 1)
        elif guard_name == "sum_gt_half":
            half_area = (H_star * W_star) // 2
            guard_result = (fv_star["sum_nonzero"] > half_area)
        elif guard_name == "h_gt_w":
            guard_result = (H_star > W_star)
        else:
            raise ValueError(f"Unknown guard: {guard_name}")

        # Select clause
        if guard_result:
            family = winner_params["true_family"]
            clause_params = winner_params["true_params"]
        else:
            family = winner_params["false_family"]
            clause_params = winner_params["false_params"]

        # Predict using clause
        R_out, C_out = _predict_clause(fv_star, family, clause_params)
    else:
        raise ValueError(f"Unknown hypothesis family: {winner_family}")

    receipts.put("winner", {
        "family": winner_family,
        "params": winner_params,
        "test_area": winner_test_area
    })
    receipts.put("R_out", R_out)
    receipts.put("C_out", C_out)
    receipts.put("verified_train_ids", list(range(num_trainings)))

    # E) Seal and return
    receipts_bundle = receipts.digest()

    return R_out, C_out, receipts_bundle


# ============================================================================
# Feature Extraction (Minimal for Size Prediction)
# ============================================================================

def _extract_size_features(
    G: List[List[int]],
    H: int,
    W: int,
    C_order: List[int]
) -> SizeFeatures:
    """
    Extract full features for size prediction (extended for H8/H9).

    Extracts H, W, periods (for H5), counts, sum_nonzero, ncc_total (for H8).
    """
    # Pack to bit-planes
    planes = pack_grid_to_planes(G, H, W, C_order)

    # A) Counts: popcount per color
    counts = {}
    for color in C_order:
        count = 0
        for r in range(H):
            count += _popcount(planes[color][r])
        counts[color] = count

    # B) Compute sum_nonzero and ncc_total
    sum_nonzero = sum(counts[c] for c in C_order if c != 0)

    # C) 4-CC stats (for ncc_total only)
    comps, cc_receipts = components(planes, H, W, C_order)

    ncc_total = 0
    for color in C_order:
        if color == 0:
            continue  # Background excluded
        # Get summary from WO-05 receipts
        color_summary = None
        for summary in cc_receipts["payload"]["per_color_summary"]:
            if summary["color"] == color:
                color_summary = summary
                break

        if color_summary and color_summary["n_cc"] > 0:
            ncc_total += color_summary["n_cc"]

    # Extract periods (proper periods p≥2 only)
    row_periods = []
    col_periods = []

    for color in C_order:
        if color == 0:
            continue  # Skip background

        plane = planes[color]

        # Row periods
        for r in range(H):
            p = minimal_period_row(plane[r], W)
            if p is not None and p >= 2:
                row_periods.append(p)

        # Column periods
        for c in range(W):
            col_mask = 0
            for r in range(H):
                bit = (plane[r] >> c) & 1
                col_mask |= (bit << r)
            p = minimal_period_1d(col_mask, H)
            if p is not None and p >= 2:
                col_periods.append(p)

    # Compute aggregates
    if row_periods:
        row_min = min(row_periods)
        lcm_r = row_periods[0]
        for p in row_periods[1:]:
            lcm_r = _lcm(lcm_r, p)
        gcd_r = row_periods[0]
        for p in row_periods[1:]:
            gcd_r = _gcd(gcd_r, p)
    else:
        row_min = None
        lcm_r = None
        gcd_r = None

    if col_periods:
        col_min = min(col_periods)
        lcm_c = col_periods[0]
        for p in col_periods[1:]:
            lcm_c = _lcm(lcm_c, p)
        gcd_c = col_periods[0]
        for p in col_periods[1:]:
            gcd_c = _gcd(gcd_c, p)
    else:
        col_min = None
        lcm_c = None
        gcd_c = None

    return SizeFeatures(
        H=H,
        W=W,
        periods={
            "row_min": row_min,
            "col_min": col_min,
            "lcm_r": lcm_r,
            "lcm_c": lcm_c,
            "gcd_r": gcd_r,
            "gcd_c": gcd_c
        },
        sum_nonzero=sum_nonzero,
        ncc_total=ncc_total,
        counts=counts
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _popcount(mask: int) -> int:
    """Count number of 1-bits in mask (Brian Kernighan's algorithm)."""
    count = 0
    while mask:
        count += 1
        mask &= mask - 1
    return count


def _gcd(a: int, b: int) -> int:
    """Euclidean GCD."""
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    """LCM via GCD."""
    return (a * b) // _gcd(a, b)


def _ceil_div(n: int, d: int) -> int:
    """Ceiling division: ⌈n/d⌉."""
    return (n + d - 1) // d


# ============================================================================
# Hypothesis Fit Checkers
# ============================================================================

def _check_fit_H1(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H1 (R=a·H, C=c·W) fits all trainings."""
    a = params["a"]
    c = params["c"]

    ok_train_ids = []
    for i, ((H_in, W_in), (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        R_pred = a * H_in
        C_pred = c * W_in

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


def _check_fit_H2(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H2 (R=H+b, C=W+d) fits all trainings."""
    b = params["b"]
    d = params["d"]

    ok_train_ids = []
    for i, ((H_in, W_in), (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        R_pred = H_in + b
        C_pred = W_in + d

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


def _check_fit_H3(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H3 (R=a·H+b, C=c·W+d) fits all trainings."""
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]

    ok_train_ids = []
    for i, ((H_in, W_in), (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        R_pred = a * H_in + b
        C_pred = c * W_in + d

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


def _check_fit_H4(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H4 (R=R₀, C=C₀) fits all trainings."""
    R0 = params["R0"]
    C0 = params["C0"]

    ok_train_ids = []
    for i, (_, (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        if (R0, C0) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


def _check_fit_H5(
    features_out: List[SizeFeatures],
    sizes_out: List[Tuple[int, int]],
    sizes_in: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int], Optional[int], Optional[int]]:
    """
    Check if H5 (R=kr·lcm_r, C=kc·lcm_c) fits all trainings.

    Uses OUTPUT periods with identity rule:
    - If lcm_r is None → R_pred = H_in (row identity)
    - If lcm_c is None → C_pred = W_in (col identity)
    - Enforces consistency of present periods across all trainings

    Returns:
        Tuple of (fit_all, ok_train_ids, common_lcm_r, common_lcm_c)
    """
    kr = params["kr"]
    kc = params["kc"]

    # Determine common periods where present; allow None on either axis
    common_lcm_r = None
    common_lcm_c = None

    for feat in features_out:
        lcm_r = feat["periods"]["lcm_r"]
        lcm_c = feat["periods"]["lcm_c"]

        # Enforce consistency only for the axes that are present
        if lcm_r is not None:
            if common_lcm_r is None:
                common_lcm_r = lcm_r
            elif lcm_r != common_lcm_r:
                # Inconsistent row periods across outputs
                return False, [], None, None

        if lcm_c is not None:
            if common_lcm_c is None:
                common_lcm_c = lcm_c
            elif lcm_c != common_lcm_c:
                # Inconsistent col periods across outputs
                return False, [], None, None

    # Check if sizes match using identity rule
    ok_train_ids = []
    for i, (feat, (H_out, W_out), (H_in, W_in)) in enumerate(zip(features_out, sizes_out, sizes_in)):
        lcm_r = feat["periods"]["lcm_r"]
        lcm_c = feat["periods"]["lcm_c"]

        # Identity rule: use H_in/W_in when period is None
        R_pred = (kr * common_lcm_r) if lcm_r is not None else H_in
        C_pred = (kc * common_lcm_c) if lcm_c is not None else W_in

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(features_out))
    return fit_all, ok_train_ids, common_lcm_r, common_lcm_c


def _check_fit_H6(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H6 (R=⌊H/kr⌋, C=⌊W/kc⌋) fits all trainings."""
    kr = params["kr"]
    kc = params["kc"]

    ok_train_ids = []
    for i, ((H_in, W_in), (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        R_pred = H_in // kr
        C_pred = W_in // kc

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


def _check_fit_H7(
    sizes_in: List[Tuple[int, int]],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H7 (R=⌈H/kr⌉, C=⌈W/kc⌉) fits all trainings."""
    kr = params["kr"]
    kc = params["kc"]

    ok_train_ids = []
    for i, ((H_in, W_in), (H_out, W_out)) in enumerate(zip(sizes_in, sizes_out)):
        R_pred = _ceil_div(H_in, kr)
        C_pred = _ceil_div(W_in, kc)

        if (R_pred, C_pred) == (H_out, W_out):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(sizes_in))
    return fit_all, ok_train_ids


# ============================================================================
# Sub-WO-04a-H8H9: H8/H9 Helpers
# ============================================================================

def _get_top2_colors(features: List[SizeFeatures]) -> Tuple[int, int]:
    """
    Select top-2 most frequent non-zero colors across ALL training inputs.

    Returns:
        Tuple of (c1, c2) where c1 is most frequent, c2 is second most frequent.
        If fewer than 2 non-zero colors exist, returns (c1, 0) or (0, 0).

    Spec:
        Sub-WO-04a-H8H9: c₁ = argmax_{c≠0} Σᵢ h(c,Xᵢ)
                          c₂ = argmax_{c≠0,c≠c₁} Σᵢ h(c,Xᵢ)
        Tie: smallest color wins (lex order).
    """
    # Sum counts across all training inputs
    color_total_counts = {}

    for feat in features:
        for color, count in feat["counts"].items():
            if color == 0:
                continue  # Skip background
            if color not in color_total_counts:
                color_total_counts[color] = 0
            color_total_counts[color] += count

    # Sort by: (1) descending count, (2) ascending color (tie-breaker)
    sorted_colors = sorted(
        color_total_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Extract top-2
    c1 = sorted_colors[0][0] if len(sorted_colors) >= 1 else 0
    c2 = sorted_colors[1][0] if len(sorted_colors) >= 2 else 0

    return c1, c2


def _check_fit_H8(
    features: List[SizeFeatures],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, Any],
    c1: int,
    c2: int
) -> Tuple[bool, List[int]]:
    """
    Check if H8 (Feature-Linear) fits all trainings.

    H8: R' = α₀ + Σ αᵢ·fᵢ, C' = β₀ + Σ βᵢ·fᵢ
    Feature basis: φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]

    Args:
        features: List of SizeFeatures for trainings.
        sizes_out: List of (R_out, C_out) for trainings.
        params: Dict with keys:
            - axis: "rows" or "cols"
            - intercept: α₀ or β₀ (0..32)
            - feat1_idx: First feature index (1..6, or None for 0-feature)
            - feat1_coeff: Coefficient for first feature (-3,-2,-1,1,2,3)
            - feat2_idx: Second feature index (1..6, > feat1_idx)
            - feat2_coeff: Coefficient for second feature (-3,-2,-1,1,2,3)
        c1, c2: Top-2 colors for count_c1, count_c2 features.

    Returns:
        Tuple of (fit_all, ok_train_ids).

    Spec:
        Sub-WO-04a-H8H9: Linear model with ≤2 features, bounded coefficients.
    """
    axis = params["axis"]
    intercept = params["intercept"]
    feat1_idx = params.get("feat1_idx", None)
    feat1_coeff = params.get("feat1_coeff", None)
    feat2_idx = params.get("feat2_idx", None)
    feat2_coeff = params.get("feat2_coeff", None)

    ok_train_ids = []

    for i, (fv, (R_expected, C_expected)) in enumerate(zip(features, sizes_out)):
        # Build feature vector φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
        count_c1 = fv["counts"].get(c1, 0)
        count_c2 = fv["counts"].get(c2, 0)

        feature_vec = [1, fv["H"], fv["W"], fv["sum_nonzero"], fv["ncc_total"], count_c1, count_c2]

        # Compute prediction (handle 0-feature, 1-feature, 2-feature models)
        pred = intercept

        if feat1_idx is not None:
            pred += feat1_coeff * feature_vec[feat1_idx]

        if feat2_idx is not None:
            pred += feat2_coeff * feature_vec[feat2_idx]

        # Check against expected
        expected = R_expected if axis == "rows" else C_expected

        if pred == expected:
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(features))
    return fit_all, ok_train_ids


def _check_fit_H9(
    features: List[SizeFeatures],
    sizes_out: List[Tuple[int, int]],
    params: Dict[str, Any]
) -> Tuple[bool, List[int]]:
    """
    Check if H9 (Guarded Piecewise) fits all trainings.

    H9: if guard(X) then F_true(H,W) else F_false(H,W)
    Guards: {has_row_period, has_col_period, ncc_gt_1, sum_gt_half, h_gt_w}
    Clauses: {H1, H2, H6, H7}

    Args:
        features: List of SizeFeatures for trainings.
        sizes_out: List of (R_out, C_out) for trainings.
        params: Dict with keys:
            - guard: "has_row_period", "has_col_period", "ncc_gt_1", "sum_gt_half", "h_gt_w"
            - true_family: "H1", "H2", "H6", or "H7"
            - true_params: Dict with parameters for true clause
            - false_family: "H1", "H2", "H6", or "H7"
            - false_params: Dict with parameters for false clause

    Returns:
        Tuple of (fit_all, ok_train_ids).

    Spec:
        Sub-WO-04a-H8H9: Guarded piecewise with frozen guards and clause families.
    """
    guard_name = params["guard"]
    true_family = params["true_family"]
    true_params = params["true_params"]
    false_family = params["false_family"]
    false_params = params["false_params"]

    ok_train_ids = []

    for i, (fv, (R_expected, C_expected)) in enumerate(zip(features, sizes_out)):
        # Evaluate guard
        if guard_name == "has_row_period":
            guard_result = (fv["periods"]["lcm_r"] is not None)
        elif guard_name == "has_col_period":
            guard_result = (fv["periods"]["lcm_c"] is not None)
        elif guard_name == "ncc_gt_1":
            guard_result = (fv["ncc_total"] > 1)
        elif guard_name == "sum_gt_half":
            half_area = (fv["H"] * fv["W"]) // 2
            guard_result = (fv["sum_nonzero"] > half_area)
        elif guard_name == "h_gt_w":
            guard_result = (fv["H"] > fv["W"])
        else:
            raise ValueError(f"Unknown guard: {guard_name}")

        # Select clause based on guard
        if guard_result:
            family = true_family
            clause_params = true_params
        else:
            family = false_family
            clause_params = false_params

        # Predict using selected clause
        R_pred, C_pred = _predict_clause(fv, family, clause_params)

        # Check against expected
        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(features))
    return fit_all, ok_train_ids


def _predict_clause(fv: SizeFeatures, family: str, params: Dict[str, int]) -> Tuple[int, int]:
    """
    Predict size using a clause family (H1, H2, H6, or H7).

    Helper for H9 guarded piecewise.
    """
    H = fv["H"]
    W = fv["W"]

    if family == "H1":
        a = params["a"]
        c = params["c"]
        return (a * H, c * W)
    elif family == "H2":
        b = params["b"]
        d = params["d"]
        return (H + b, W + d)
    elif family == "H6":
        kr = params["kr"]
        kc = params["kc"]
        return (H // kr, W // kc)
    elif family == "H7":
        kr = params["kr"]
        kc = params["kc"]
        return (_ceil_div(H, kr), _ceil_div(W, kc))
    else:
        raise ValueError(f"Unknown clause family: {family}")
