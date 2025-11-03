"""
WO-04a: Working Canvas Provider

Single working canvas size selection via frozen T9 hypothesis class (H1-H7).
Trainings-only evaluation, fail-closed, receipts-first.

Spec: WO-04a (v1.6)
"""

from typing import Dict, List, Tuple, TypedDict, Optional
from .core import Receipts
from .core.hashing import blake3_hash
from .kernel.planes import pack_grid_to_planes, order_colors
from .kernel.period import minimal_period_row, minimal_period_1d
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
    """Minimal features for size prediction."""
    H: int
    W: int
    periods: Dict[str, Optional[int]]  # {row_min, col_min, lcm_r, lcm_c, gcd_r, gcd_c}


# ============================================================================
# Main Entry Point
# ============================================================================

def choose_working_canvas(
    train_pairs: List[Dict],
    frames_in: List[Dict],
    frames_out: List[Dict],
    xstar_shape: Tuple[int, int]
) -> Tuple[int, int, Dict]:
    """
    Choose single working canvas size (R_out, C_out) via frozen H1-H7 evaluation.

    Args:
        train_pairs: List of {"X": Grid, "Y": Grid} training pairs.
        frames_in: List of frame objects for inputs (receipts only).
        frames_out: List of frame objects for outputs (receipts only).
        xstar_shape: (H*, W*) test input dimensions.

    Returns:
        Tuple of (R_out, C_out, receipts_dict).

    Raises:
        SizeUndetermined: If no hypothesis fits all trainings.

    Spec:
        WO-04a: Trainings-only size prediction, fail-closed, no LCM.
        Uses OUTPUT periods for H5 (not input periods).
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
        C_order_in = order_colors(set(cell for row in X for cell in row))
        feat_in = _extract_size_features(X, H_in, W_in, C_order_in)
        features_in.append(feat_in)

        # Extract features from OUTPUT (for H5 period detection)
        C_order_out = order_colors(set(cell for row in Y for cell in row))
        feat_out = _extract_size_features(Y, H_out, W_out, C_order_out)
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

    receipts.put("attempts", attempts)
    receipts.put("total_candidates_checked", len(attempts))

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
    def candidate_sort_key(cand):
        if len(cand) == 3:
            # H1, H2, H3, H4, H6, H7
            family, params, test_area = cand
            return (test_area, family, tuple(sorted(params.items())))
        else:
            # H5 with extra period data
            family, params, test_area, lcm_r, lcm_c = cand
            return (test_area, family, tuple(sorted(params.items())))

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
    Extract minimal features for size prediction.

    Only extracts H, W, and periods (for H5).
    Does NOT extract counts or CC stats (not needed for size prediction).
    """
    # Pack to bit-planes
    planes = pack_grid_to_planes(G, H, W, C_order)

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
        }
    )


# ============================================================================
# Helper Functions
# ============================================================================

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
