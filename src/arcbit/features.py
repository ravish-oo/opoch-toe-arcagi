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


class SizeFit(TypedDict):
    """Size prediction hypothesis with fit status."""
    family: str  # H1, H2, ..., H7
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
        periods=periods
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
    Find best-fit size hypothesis H1-H7 using trainings only.

    Frozen enumeration order: H1 → H2 → H3 → H4 → H5 → H6 → H7.
    For each hypothesis, enumerate parameters in frozen order (outer→inner loops).
    Fit criterion: ∀i: (R'ᵢ, C'ᵢ) = (Rᵢ, Cᵢ) (exact match on ALL trainings).

    Tie rule (3 levels):
      1. Smallest predicted test output area (R_test × C_test)
      2. Hypothesis family lexicographic order (H1 < H2 < ... < H7)
      3. Parameters lexicographic order (tuple comparison)

    Args:
        train_pairs: List of (FeatureVector, (R_out, C_out)) for trainings.
        test_features: FeatureVector for test input X*.

    Returns:
        Tuple of (SizeFit, receipts) if fit found, else None.

    Spec:
        WO-14: Bounded search ≤19,845 candidates total.
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

    receipts.put("attempts", attempts)
    receipts.put("total_candidates_checked", len(attempts))

    # C) Apply tie rule to select winner
    if not candidates:
        receipts.put("winner", None)
        receipts.put("verified_train_ids", [])
        receipts_bundle = receipts.digest()
        return None

    # Sort by: (1) test_area, (2) family lex, (3) params lex
    candidates.sort(key=lambda x: (
        x[2],  # test_area (smallest first)
        x[0],  # family lex (H1 < H2 < ... < H7)
        tuple(sorted(x[1].items()))  # params lex
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
    fit: SizeFit
) -> Tuple[int, int]:
    """
    Predict output size (R, C) from feature vector using fitted hypothesis.

    Args:
        fv: FeatureVector for input grid.
        fit: SizeFit from agg_size_fit().

    Returns:
        Tuple of (R_out, C_out).

    Spec:
        WO-14: Deterministic prediction from frozen hypothesis.
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
