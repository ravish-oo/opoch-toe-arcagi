"""
WO-14: Aggregate Mapping (T9) - Features & Size Predictor

Deterministic feature extraction and size prediction via bounded hypothesis class H1-H7.
Optional constant-color mapping for grid-aggregate transforms.

Spec: WO-14 (v1.5 + v1.6)
"""

from typing import Dict, List, Optional, Tuple, TypedDict, Any
import math
from .core import Receipts
from .core.hashing import blake3_hash
from .kernel.planes import pack_grid_to_planes, order_colors
from .kernel.components import components
from .kernel.period import period_2d_planes


# ============================================================================
# Type Definitions
# ============================================================================

class FeatureVector(TypedDict):
    """Frozen feature schema for aggregate mapping."""
    H: int
    W: int
    counts: Dict[int, int]  # {color: count}
    cc: Dict[int, Dict[str, Optional[int]]]  # {color: {n, area_min, area_max, area_sum}}
    periods: Dict[str, Optional[int]]  # {row_min, col_min, lcm_r, lcm_c, gcd_r, gcd_c}
    sum_nonzero: int  # Σ h(c) for c≠0 (Sub-WO-14b)
    ncc_total: int  # Σ n_cc(c) for c≠0 (Sub-WO-14b)


class SizeFit(TypedDict):
    """Size prediction hypothesis with fit status."""
    family: str  # H1, H2, ..., H9 (Sub-WO-14b)
    params: Dict[str, int]
    receipts: Dict  # Section receipts with attempts trail


class ColorMap(TypedDict):
    """Constant-color mapping result."""
    family: str  # CONST_ARGMAX or CONST_MAJORITY_OUT
    params: Dict[str, Any]
    mapping: Dict[str, int]  # e.g., {"ARGMAX_COLOR": 8}
    receipts: Dict


# ============================================================================
# Feature Extraction
# ============================================================================

def agg_features(
    G: List[List[int]],
    H: int,
    W: int,
    C_order: List[int]
) -> Tuple[FeatureVector, Dict]:
    """
    Extract aggregate features from grid G.

    Features (frozen schema):
      - H, W: Grid dimensions
      - counts: {color: count} for all colors in C_order
      - cc: {color: {n, area_min, area_max, area_sum}} from WO-05 components
      - periods: {row_min, col_min, lcm_r, lcm_c, gcd_r, gcd_c} from WO-02

    Args:
        G: Grid (H × W nested list).
        H: Grid height.
        W: Grid width.
        C_order: Ordered color palette (ascending integers).

    Returns:
        Tuple of (FeatureVector, receipts_dict).

    Spec:
        WO-14: Frozen feature schema, no heuristics.
        Uses WO-05 (components) and WO-02 (periods).
    """
    receipts = Receipts("features")

    # A) Validate inputs
    if len(G) != H:
        raise ValueError(f"WO-14: Grid has {len(G)} rows, expected {H}")
    for r, row in enumerate(G):
        if len(row) != W:
            raise ValueError(f"WO-14: Row {r} has {len(row)} columns, expected {W}")

    receipts.put("inputs", {
        "H": H,
        "W": W,
        "colors_order": C_order
    })

    # B) Pack to bit-planes
    planes = pack_grid_to_planes(G, H, W, C_order)

    # C) Counts: popcount per color
    counts = {}
    for color in C_order:
        count = 0
        for r in range(H):
            count += _popcount(planes[color][r])
        counts[color] = count

    # Store in receipts with string keys
    receipts.put("counts", {str(c): v for c, v in counts.items()})

    # D) 4-CC stats (WO-05)
    comps, cc_receipts = components(planes, H, W, C_order)

    cc_stats = {}
    for color in C_order:
        if color == 0:
            # Background excluded by WO-05
            cc_stats[color] = {
                "n": None,
                "area_min": None,
                "area_max": None,
                "area_sum": None
            }
        else:
            # Get summary from WO-05 receipts
            color_summary = None
            for summary in cc_receipts["payload"]["per_color_summary"]:
                if summary["color"] == color:
                    color_summary = summary
                    break

            if color_summary and color_summary["n_cc"] > 0:
                cc_stats[color] = {
                    "n": color_summary["n_cc"],
                    "area_min": color_summary["area_min"],
                    "area_max": color_summary["area_max"],
                    "area_sum": color_summary["area_sum"]
                }
            else:
                cc_stats[color] = {
                    "n": 0,
                    "area_min": None,
                    "area_max": None,
                    "area_sum": None
                }

    # Store in receipts with string keys
    receipts.put("cc", {str(c): v for c, v in cc_stats.items()})

    # D2) Compute sum_nonzero and ncc_total (Sub-WO-14b)
    sum_nonzero = sum(counts[c] for c in C_order if c != 0)
    ncc_total = sum(cc_stats[c]["n"] for c in C_order if c != 0 and cc_stats[c]["n"] is not None)

    receipts.put("sum_nonzero", sum_nonzero)
    receipts.put("ncc_total", ncc_total)

    # E) Periods (WO-02) - INPUT only, proper periods p≥2 only
    # Import period functions
    from .kernel.period import minimal_period_row, minimal_period_1d

    # Compute per-color periods: row and column minimums
    row_periods = []
    col_periods = []

    for color in C_order:
        if color == 0:
            continue  # Skip background

        plane = planes[color]

        # Row periods: check each row's period
        color_row_periods = []
        for r in range(H):
            p = minimal_period_row(plane[r], W)
            if p is not None and p >= 2:
                color_row_periods.append(p)

        # Column periods: check each column's period
        # Extract columns as bit masks
        color_col_periods = []
        for c in range(W):
            col_mask = 0
            for r in range(H):
                bit = (plane[r] >> c) & 1
                col_mask |= (bit << r)
            p = minimal_period_1d(col_mask, H)
            if p is not None and p >= 2:
                color_col_periods.append(p)

        # Take minimum period for this color (if any)
        if color_row_periods:
            row_periods.append(min(color_row_periods))
        if color_col_periods:
            col_periods.append(min(color_col_periods))

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

    periods = {
        "row_min": row_min,
        "col_min": col_min,
        "lcm_r": lcm_r,
        "lcm_c": lcm_c,
        "gcd_r": gcd_r,
        "gcd_c": gcd_c
    }

    receipts.put("periods", periods)

    # F) Build feature vector
    fv = FeatureVector(
        H=H,
        W=W,
        counts=counts,
        cc=cc_stats,
        periods=periods,
        sum_nonzero=sum_nonzero,
        ncc_total=ncc_total
    )

    # G) Seal receipts
    receipts_bundle = receipts.digest()

    return fv, receipts_bundle


# ============================================================================
# Size Prediction (H1-H7)
# ============================================================================

def agg_size_fit(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    test_features: FeatureVector
) -> Optional[Tuple[SizeFit, Dict]]:
    """
    Find best-fit size hypothesis H1-H9 using trainings only.

    Frozen enumeration order: H1 → H2 → H3 → H4 → H5 → H6 → H7 → H8 → H9.
    For each hypothesis, enumerate parameters in frozen order (outer→inner loops).
    Fit criterion: ∀i: (R'ᵢ, C'ᵢ) = (Rᵢ, Cᵢ) (exact match on ALL trainings).

    Tie rule (3 levels):
      1. Smallest predicted test output area (R_test × C_test)
      2. Hypothesis family lexicographic order (H1 < H2 < ... < H9)
      3. Parameters lexicographic order (tuple comparison)

    Args:
        train_pairs: List of (FeatureVector, (R_out, C_out)) for trainings.
        test_features: FeatureVector for test input X*.

    Returns:
        Tuple of (SizeFit, receipts) if fit found, else None.

    Spec:
        WO-14 + Sub-WO-14b: Bounded search with H1-H9.
        No test leakage: selection uses trainings only.
    """
    receipts = Receipts("size_fit")

    # A) Validate inputs
    if not train_pairs:
        receipts.put("error", "No training pairs provided")
        receipts_bundle = receipts.digest()
        return None

    receipts.put("num_trainings", len(train_pairs))
    receipts.put("test_features_H", test_features["H"])
    receipts.put("test_features_W", test_features["W"])

    # B) Enumerate hypotheses in frozen order
    attempts = []
    candidates = []  # (family, params, test_area)

    # H1: R = a·H, C = c·W | a,c ∈ {1..8}
    for a in range(1, 9):
        for c in range(1, 9):
            params = {"a": a, "c": c}
            fit_all, ok_train_ids = _check_fit_H1(train_pairs, params)

            attempts.append({
                "family": "H1",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = a * test_features["H"]
                C_test = c * test_features["W"]
                test_area = R_test * C_test
                candidates.append(("H1", params, test_area))

    # H2: R = H + b, C = W + d | b,d ∈ {0..16}
    for b in range(0, 17):
        for d in range(0, 17):
            params = {"b": b, "d": d}
            fit_all, ok_train_ids = _check_fit_H2(train_pairs, params)

            attempts.append({
                "family": "H2",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = test_features["H"] + b
                C_test = test_features["W"] + d
                test_area = R_test * C_test
                candidates.append(("H2", params, test_area))

    # H3: R = a·H + b, C = c·W + d | a,c ∈ {1..8}, b,d ∈ {0..16}
    # Enumeration order: a → c → b → d (frozen)
    for a in range(1, 9):       # 1..8
        for c in range(1, 9):   # 1..8
            for b in range(0, 17):  # 0..16
                for d in range(0, 17):  # 0..16
                    params = {"a": a, "b": b, "c": c, "d": d}
                    fit_all, ok_train_ids = _check_fit_H3(train_pairs, params)

                    attempts.append({
                        "family": "H3",
                        "params": params,
                        "ok_train_ids": ok_train_ids,
                        "fit_all": fit_all
                    })

                    if fit_all:
                        R_test = a * test_features["H"] + b
                        C_test = c * test_features["W"] + d
                        test_area = R_test * C_test
                        candidates.append(("H3", params, test_area))

    # H4: R = R₀, C = C₀ | R₀,C₀ ∈ {1..30}
    for R0 in range(1, 31):
        for C0 in range(1, 31):
            params = {"R0": R0, "C0": C0}
            fit_all, ok_train_ids = _check_fit_H4(train_pairs, params)

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
    # Uses identity rule: if period is None, use H or W
    for kr in range(1, 9):
        for kc in range(1, 9):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids = _check_fit_H5(train_pairs, params)

            attempts.append({
                "family": "H5",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                H_star = test_features["H"]
                W_star = test_features["W"]
                lcm_r_star = test_features["periods"]["lcm_r"]
                lcm_c_star = test_features["periods"]["lcm_c"]

                # Identity rule for test prediction
                R_test = (kr * lcm_r_star) if lcm_r_star is not None else H_star
                C_test = (kc * lcm_c_star) if lcm_c_star is not None else W_star
                test_area = R_test * C_test
                candidates.append(("H5", params, test_area))

    # H6: R = ⌊H/kr⌋, C = ⌊W/kc⌋ | kr,kc ∈ {2..5}
    for kr in range(2, 6):
        for kc in range(2, 6):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids = _check_fit_H6(train_pairs, params)

            attempts.append({
                "family": "H6",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = test_features["H"] // kr
                C_test = test_features["W"] // kc
                test_area = R_test * C_test
                candidates.append(("H6", params, test_area))

    # H7: R = ⌈H/kr⌉, C = ⌈W/kc⌉ | kr,kc ∈ {2..5}
    for kr in range(2, 6):
        for kc in range(2, 6):
            params = {"kr": kr, "kc": kc}
            fit_all, ok_train_ids = _check_fit_H7(train_pairs, params)

            attempts.append({
                "family": "H7",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                R_test = _ceil_div(test_features["H"], kr)
                C_test = _ceil_div(test_features["W"], kc)
                test_area = R_test * C_test
                candidates.append(("H7", params, test_area))

    # ========================================================================
    # Sub-WO-14b: Compute top-2 colors for H8 features
    # ========================================================================

    c1, c2 = _get_top2_colors(train_pairs)
    receipts.put("top2_colors", {"c1": c1, "c2": c2})

    # ========================================================================
    # H8: Feature-Linear (Sub-WO-14b) with early stopping
    # ========================================================================
    # Model: R' = α₀ + Σ αᵢ·fᵢ, C' = β₀ + Σ βᵢ·fᵢ
    # Features: φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
    # Bounds: α₀,β₀ ∈ [0..32], αᵢ,βᵢ ∈ {-3,-2,-1,1,2,3}
    # Enumeration: rows → cols, 1-feature → 2-feature, intercept → features → coeffs
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

    # Enumerate row models with early stopping
    for row_params in generate_models():
        # Check if row model fits
        row_params_with_axis = {**row_params, "axis": "rows"}
        fit_rows, ok_train_ids_rows = _check_fit_H8(train_pairs, row_params_with_axis, c1, c2)

        if not fit_rows:
            # Row model doesn't fit, skip col enumeration
            # Still log attempt for receipts
            params = {"row_model": row_params, "col_model": None}
            attempts.append({
                "family": "H8",
                "params": params,
                "ok_train_ids": [],
                "fit_all": False
            })
            continue

        # Row model fits, enumerate col models
        for col_params in generate_models():
            # Check if col model fits
            col_params_with_axis = {**col_params, "axis": "cols"}
            fit_cols, ok_train_ids_cols = _check_fit_H8(train_pairs, col_params_with_axis, c1, c2)

            # Both must fit all trainings
            fit_all = fit_cols  # Row already checked
            ok_train_ids = ok_train_ids_rows if fit_all else []

            # Combine params for receipts
            params = {
                "row_model": row_params,
                "col_model": col_params
            }

            attempts.append({
                "family": "H8",
                "params": params,
                "ok_train_ids": ok_train_ids,
                "fit_all": fit_all
            })

            if fit_all:
                # Predict test output size
                H_star = test_features["H"]
                W_star = test_features["W"]
                sum_nonzero_star = test_features["sum_nonzero"]
                ncc_total_star = test_features["ncc_total"]
                count_c1_star = test_features["counts"].get(c1, 0)
                count_c2_star = test_features["counts"].get(c2, 0)

                features_star = [1, H_star, W_star, sum_nonzero_star, ncc_total_star, count_c1_star, count_c2_star]

                # R prediction (handle 0-feature, 1-feature, 2-feature models)
                R_test = row_params["intercept"]
                if row_params["feat1_idx"] is not None:
                    R_test += row_params["feat1_coeff"] * features_star[row_params["feat1_idx"]]
                if row_params["feat2_idx"] is not None:
                    R_test += row_params["feat2_coeff"] * features_star[row_params["feat2_idx"]]

                # C prediction (handle 0-feature, 1-feature, 2-feature models)
                C_test = col_params["intercept"]
                if col_params["feat1_idx"] is not None:
                    C_test += col_params["feat1_coeff"] * features_star[col_params["feat1_idx"]]
                if col_params["feat2_idx"] is not None:
                    C_test += col_params["feat2_coeff"] * features_star[col_params["feat2_idx"]]

                test_area = R_test * C_test
                candidates.append(("H8", params, test_area))

    # ========================================================================
    # H9: Guarded Piecewise (Sub-WO-14b)
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

                        fit_all, ok_train_ids = _check_fit_H9(train_pairs, params, c1, c2)

                        attempts.append({
                            "family": "H9",
                            "params": params,
                            "ok_train_ids": ok_train_ids,
                            "fit_all": fit_all
                        })

                        if fit_all:
                            # Predict test output size
                            fv_star = test_features

                            # Evaluate guard on test input
                            if guard == "has_row_period":
                                guard_result = (fv_star["periods"]["lcm_r"] is not None)
                            elif guard == "has_col_period":
                                guard_result = (fv_star["periods"]["lcm_c"] is not None)
                            elif guard == "ncc_gt_1":
                                guard_result = (fv_star["ncc_total"] > 1)
                            elif guard == "sum_gt_half":
                                half_area = (fv_star["H"] * fv_star["W"]) // 2
                                guard_result = (fv_star["sum_nonzero"] > half_area)
                            elif guard == "h_gt_w":
                                guard_result = (fv_star["H"] > fv_star["W"])

                            # Select clause
                            if guard_result:
                                family = true_family
                                clause_params = true_params
                            else:
                                family = false_family
                                clause_params = false_params

                            # Predict
                            R_test, C_test = _predict_clause(fv_star, family, clause_params)
                            test_area = R_test * C_test
                            candidates.append(("H9", params, test_area))

    receipts.put("attempts", attempts)
    receipts.put("total_candidates_checked", len(attempts))

    # C) Apply tie rule to select winner
    if not candidates:
        receipts.put("winner", None)
        receipts.put("verified_train_ids", [])
        receipts_bundle = receipts.digest()
        return None

    # Helper to convert params to sortable tuple (handles nested dicts and None)
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

    # Sort by: (1) test_area, (2) family lex, (3) params lex
    candidates.sort(key=lambda x: (
        x[2],  # test_area (smallest first)
        x[0],  # family lex (H1 < H2 < ... < H9)
        params_to_sortable(x[1])  # params lex (handles nested dicts)
    ))

    winner_family, winner_params, winner_test_area = candidates[0]

    receipts.put("winner", {
        "family": winner_family,
        "params": winner_params,
        "test_area": winner_test_area
    })
    receipts.put("verified_train_ids", list(range(len(train_pairs))))

    # D) Seal and return
    receipts_bundle = receipts.digest()

    size_fit = SizeFit(
        family=winner_family,
        params=winner_params,
        receipts=receipts_bundle
    )

    return size_fit, receipts_bundle


def predict_size(
    fv: FeatureVector,
    fit: SizeFit,
    c1: int = 0,
    c2: int = 0
) -> Tuple[int, int]:
    """
    Predict output size (R, C) from feature vector using fitted hypothesis.

    Args:
        fv: FeatureVector for input grid.
        fit: SizeFit from agg_size_fit().
        c1, c2: Top-2 colors (required for H8, optional for others).

    Returns:
        Tuple of (R_out, C_out).

    Spec:
        WO-14 + Sub-WO-14b: Deterministic prediction from frozen hypothesis H1-H9.
    """
    family = fit["family"]
    params = fit["params"]

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

    elif family == "H3":
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        return (a * H + b, c * W + d)

    elif family == "H4":
        R0 = params["R0"]
        C0 = params["C0"]
        return (R0, C0)

    elif family == "H5":
        kr = params["kr"]
        kc = params["kc"]
        H = fv["H"]
        W = fv["W"]
        lcm_r = fv["periods"]["lcm_r"]
        lcm_c = fv["periods"]["lcm_c"]

        # Identity rule: use H/W when period is None
        R_out = (kr * lcm_r) if lcm_r is not None else H
        C_out = (kc * lcm_c) if lcm_c is not None else W

        return (R_out, C_out)

    elif family == "H6":
        kr = params["kr"]
        kc = params["kc"]
        return (H // kr, W // kc)

    elif family == "H7":
        kr = params["kr"]
        kc = params["kc"]
        return (_ceil_div(H, kr), _ceil_div(W, kc))

    elif family == "H8":
        # Feature-Linear model
        row_params = params["row_model"]
        col_params = params["col_model"]

        # Build feature vector φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
        count_c1 = fv["counts"].get(c1, 0)
        count_c2 = fv["counts"].get(c2, 0)
        features = [1, fv["H"], fv["W"], fv["sum_nonzero"], fv["ncc_total"], count_c1, count_c2]

        # R prediction (handle 0-feature, 1-feature, 2-feature models)
        R_out = row_params["intercept"]
        if row_params["feat1_idx"] is not None:
            R_out += row_params["feat1_coeff"] * features[row_params["feat1_idx"]]
        if row_params["feat2_idx"] is not None:
            R_out += row_params["feat2_coeff"] * features[row_params["feat2_idx"]]

        # C prediction (handle 0-feature, 1-feature, 2-feature models)
        C_out = col_params["intercept"]
        if col_params["feat1_idx"] is not None:
            C_out += col_params["feat1_coeff"] * features[col_params["feat1_idx"]]
        if col_params["feat2_idx"] is not None:
            C_out += col_params["feat2_coeff"] * features[col_params["feat2_idx"]]

        return (R_out, C_out)

    elif family == "H9":
        # Guarded Piecewise model
        guard_name = params["guard"]
        true_family = params["true_family"]
        true_params = params["true_params"]
        false_family = params["false_family"]
        false_params = params["false_params"]

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

        # Select clause
        if guard_result:
            family = true_family
            clause_params = true_params
        else:
            family = false_family
            clause_params = false_params

        # Predict
        return _predict_clause(fv, family, clause_params)

    else:
        raise ValueError(f"WO-14: Unknown hypothesis family: {family}")


# ============================================================================
# Constant-Color Mapping (Optional)
# ============================================================================

def agg_color_map(
    train_pairs: List[Tuple[FeatureVector, List[List[int]]]],
    C_order: List[int]
) -> Optional[Tuple[ColorMap, Dict]]:
    """
    Find constant-color mapping using trainings only.

    Two families:
      - CONST_ARGMAX: Map all cells to argmax(counts[Xi]) across trainings
      - CONST_MAJORITY_OUT: Map all cells to majority color in ⋃Yi

    Fit criterion: ∀i: Yi == constant_grid(Ri, Ci, mapped_color)

    Args:
        train_pairs: List of (FeatureVector, Y_output_grid) for trainings.
        C_order: Ordered color palette.

    Returns:
        Tuple of (ColorMap, receipts) if fit found, else None.

    Spec:
        WO-14: Optional constant-color mapping for grid-aggregate tasks.
    """
    receipts = Receipts("color_map")

    if not train_pairs:
        receipts.put("error", "No training pairs provided")
        receipts_bundle = receipts.digest()
        return None

    receipts.put("num_trainings", len(train_pairs))
    receipts.put("colors_order", C_order)

    attempts = []
    candidates = []

    # A) CONST_ARGMAX: argmax(counts[Xi]) across trainings
    # Find argmax color across all trainings
    color_counts_sum = {c: 0 for c in C_order}
    for fv, Y in train_pairs:
        for color, count in fv["counts"].items():
            color_counts_sum[color] += count

    # Argmax (tie: smallest color)
    argmax_color = max(C_order, key=lambda c: (color_counts_sum[c], -c))

    # Check if this fits all trainings
    fit_all = True
    ok_train_ids = []

    for i, (fv, Y) in enumerate(train_pairs):
        R_out = len(Y)
        C_out = len(Y[0]) if Y else 0

        # Check if Y is all argmax_color
        matches = True
        for r in range(R_out):
            for c in range(C_out):
                if Y[r][c] != argmax_color:
                    matches = False
                    break
            if not matches:
                break

        if matches:
            ok_train_ids.append(i)
        else:
            fit_all = False

    attempts.append({
        "family": "CONST_ARGMAX",
        "params": {},
        "mapping": {"ARGMAX_COLOR": argmax_color},
        "ok_train_ids": ok_train_ids,
        "fit_all": fit_all
    })

    if fit_all:
        candidates.append(("CONST_ARGMAX", {}, {"ARGMAX_COLOR": argmax_color}))

    # B) CONST_MAJORITY_OUT: majority color in ⋃Yi
    # Count all colors across all output grids
    output_color_counts = {c: 0 for c in C_order}

    for fv, Y in train_pairs:
        for row in Y:
            for cell in row:
                if cell in output_color_counts:
                    output_color_counts[cell] += 1

    # Majority (tie: smallest color)
    majority_color = max(C_order, key=lambda c: (output_color_counts[c], -c))

    # Check if this fits all trainings
    fit_all = True
    ok_train_ids = []

    for i, (fv, Y) in enumerate(train_pairs):
        R_out = len(Y)
        C_out = len(Y[0]) if Y else 0

        # Check if Y is all majority_color
        matches = True
        for r in range(R_out):
            for c in range(C_out):
                if Y[r][c] != majority_color:
                    matches = False
                    break
            if not matches:
                break

        if matches:
            ok_train_ids.append(i)
        else:
            fit_all = False

    attempts.append({
        "family": "CONST_MAJORITY_OUT",
        "params": {},
        "mapping": {"MAJORITY_COLOR": majority_color},
        "ok_train_ids": ok_train_ids,
        "fit_all": fit_all
    })

    if fit_all:
        candidates.append(("CONST_MAJORITY_OUT", {}, {"MAJORITY_COLOR": majority_color}))

    receipts.put("attempts", attempts)

    # C) Select winner (first fit in enumeration order)
    if not candidates:
        receipts.put("winner", None)
        receipts_bundle = receipts.digest()
        return None

    winner_family, winner_params, winner_mapping = candidates[0]

    receipts.put("winner", {
        "family": winner_family,
        "params": winner_params,
        "mapping": winner_mapping
    })
    receipts.put("verified_train_ids", list(range(len(train_pairs))))

    # D) Seal and return
    receipts_bundle = receipts.digest()

    color_map = ColorMap(
        family=winner_family,
        params=winner_params,
        mapping=winner_mapping,
        receipts=receipts_bundle
    )

    return color_map, receipts_bundle


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
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H1 (R=a·H, C=c·W) fits all trainings."""
    a = params["a"]
    c = params["c"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        R_pred = a * fv["H"]
        C_pred = c * fv["W"]

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H2(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H2 (R=H+b, C=W+d) fits all trainings."""
    b = params["b"]
    d = params["d"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        R_pred = fv["H"] + b
        C_pred = fv["W"] + d

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H3(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H3 (R=a·H+b, C=c·W+d) fits all trainings."""
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        R_pred = a * fv["H"] + b
        C_pred = c * fv["W"] + d

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H4(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H4 (R=R₀, C=C₀) fits all trainings."""
    R0 = params["R0"]
    C0 = params["C0"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        if (R0, C0) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H5(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """
    Check if H5 (R=kr·lcm_r, C=kc·lcm_c) fits all trainings.

    Uses identity rule: if lcm_r is None → R_pred = H_i (row identity)
                        if lcm_c is None → C_pred = W_i (col identity)
    """
    kr = params["kr"]
    kc = params["kc"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        H_i = fv["H"]
        W_i = fv["W"]
        lcm_r = fv["periods"]["lcm_r"]
        lcm_c = fv["periods"]["lcm_c"]

        # Identity rule: use H_i/W_i when period is None
        R_pred = (kr * lcm_r) if lcm_r is not None else H_i
        C_pred = (kc * lcm_c) if lcm_c is not None else W_i

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H6(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H6 (R=⌊H/kr⌋, C=⌊W/kc⌋) fits all trainings."""
    kr = params["kr"]
    kc = params["kc"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        R_pred = fv["H"] // kr
        C_pred = fv["W"] // kc

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H7(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, int]
) -> Tuple[bool, List[int]]:
    """Check if H7 (R=⌈H/kr⌉, C=⌈W/kc⌉) fits all trainings."""
    kr = params["kr"]
    kc = params["kc"]

    ok_train_ids = []
    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        R_pred = _ceil_div(fv["H"], kr)
        C_pred = _ceil_div(fv["W"], kc)

        if (R_pred, C_pred) == (R_expected, C_expected):
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _get_top2_colors(train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Select top-2 most frequent non-zero colors across ALL training inputs.

    Returns:
        Tuple of (c1, c2) where c1 is most frequent, c2 is second most frequent.
        If fewer than 2 non-zero colors exist, returns (c1, 0) or (0, 0).

    Spec:
        Sub-WO-14b: c₁ = argmax_{c≠0} Σᵢ h(c,Xᵢ)
                    c₂ = argmax_{c≠0,c≠c₁} Σᵢ h(c,Xᵢ)
        Tie: smallest color wins (lex order).
    """
    # Sum counts across all training inputs
    color_total_counts = {}

    for fv, _ in train_pairs:
        for color, count in fv["counts"].items():
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
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, Any],
    c1: int,
    c2: int
) -> Tuple[bool, List[int]]:
    """
    Check if H8 (Feature-Linear) fits all trainings.

    H8: R' = α₀ + Σ αᵢ·fᵢ, C' = β₀ + Σ βᵢ·fᵢ
    Feature basis: φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]

    Args:
        train_pairs: Training pairs with feature vectors.
        params: Dict with keys:
            - axis: "rows" or "cols"
            - intercept: α₀ or β₀ (0..32)
            - feat1_idx: First feature index (1..6)
            - feat1_coeff: Coefficient for first feature (-3,-2,-1,1,2,3)
            - feat2_idx: Second feature index (1..6, > feat1_idx)
            - feat2_coeff: Coefficient for second feature (-3,-2,-1,1,2,3)
        c1, c2: Top-2 colors for count_c1, count_c2 features.

    Returns:
        Tuple of (fit_all, ok_train_ids).

    Spec:
        Sub-WO-14b H8: Linear model with ≤2 features, bounded coefficients.
    """
    axis = params["axis"]
    intercept = params["intercept"]
    feat1_idx = params["feat1_idx"]
    feat1_coeff = params["feat1_coeff"]
    feat2_idx = params.get("feat2_idx", None)
    feat2_coeff = params.get("feat2_coeff", None)

    ok_train_ids = []

    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
        # Build feature vector φ = [1, H, W, sum_nonzero, ncc_total, count_c1, count_c2]
        count_c1 = fv["counts"].get(c1, 0)
        count_c2 = fv["counts"].get(c2, 0)

        features = [1, fv["H"], fv["W"], fv["sum_nonzero"], fv["ncc_total"], count_c1, count_c2]

        # Compute prediction (handle 0-feature, 1-feature, 2-feature models)
        pred = intercept

        if feat1_idx is not None:
            pred += feat1_coeff * features[feat1_idx]

        if feat2_idx is not None:
            pred += feat2_coeff * features[feat2_idx]

        # Check against expected
        expected = R_expected if axis == "rows" else C_expected

        if pred == expected:
            ok_train_ids.append(i)

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _check_fit_H9(
    train_pairs: List[Tuple[FeatureVector, Tuple[int, int]]],
    params: Dict[str, Any],
    c1: int,
    c2: int
) -> Tuple[bool, List[int]]:
    """
    Check if H9 (Guarded Piecewise) fits all trainings.

    H9: if guard(X) then F_true(H,W) else F_false(H,W)
    Guards: {has_row_period, has_col_period, ncc_total>1, sum_nonzero>⌊H·W/2⌋, H>W}
    Clauses: {H1, H2, H6, H7}

    Args:
        train_pairs: Training pairs with feature vectors.
        params: Dict with keys:
            - guard: "has_row_period", "has_col_period", "ncc_gt_1", "sum_gt_half", "h_gt_w"
            - true_family: "H1", "H2", "H6", or "H7"
            - true_params: Dict with parameters for true clause
            - false_family: "H1", "H2", "H6", or "H7"
            - false_params: Dict with parameters for false clause
        c1, c2: Top-2 colors (unused for H9, but kept for consistency).

    Returns:
        Tuple of (fit_all, ok_train_ids).

    Spec:
        Sub-WO-14b H9: Guarded piecewise with frozen guards and clause families.
    """
    guard_name = params["guard"]
    true_family = params["true_family"]
    true_params = params["true_params"]
    false_family = params["false_family"]
    false_params = params["false_params"]

    ok_train_ids = []

    for i, (fv, (R_expected, C_expected)) in enumerate(train_pairs):
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

    fit_all = (len(ok_train_ids) == len(train_pairs))
    return fit_all, ok_train_ids


def _predict_clause(fv: FeatureVector, family: str, params: Dict[str, int]) -> Tuple[int, int]:
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
