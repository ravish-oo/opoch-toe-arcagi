"""
WO-07: Conjugation & Forward Witness Emitter

Conjugates per-training witness pieces to working canvas and test input frames,
forward-emits test input planes through conjugated pieces, and produces global
witness emitter (A_wit, S_wit) via scope-gated intersection.

Pure functional, deterministic, receipts-first.
"""

from typing import Dict, List, Tuple, Optional
import json
from src.arcbit.kernel.ops import (
    pose_plane,
    shift_plane,
    pose_compose,
    pose_inverse,
)
from src.arcbit.core import Receipts, blake3_hash


def emit_witness(
    X_star: List[List[int]],
    witness_results_all_trainings: List[dict],
    frames: dict,
    colors_order: List[int],
    R_out: int,
    C_out: int,
    included_train_ids: Optional[List[int]] = None,
    debug_arrays: bool = False
) -> Tuple[Dict[int, List[int]], List[int], Dict]:
    """
    Conjugate witness pieces and forward-emit test input to working canvas.

    Args:
        X_star: Test input grid (raw integers)
        witness_results_all_trainings: List of WitnessResult dicts from WO-06
        frames: Dict with Pi_in_star, Pi_out_star, and per-training frames
        colors_order: Ascending color list (includes 0 and all task colors)
        R_out, C_out: Working canvas shape from WO-04a
        included_train_ids: Optional list of training IDs to use (from transport)

    Returns:
        A_wit: Dict color -> plane (scope-gated intersection)
        S_wit: Scope mask (union of all training scopes)
        receipts_bundle: Dict with section_hash and payload per WO-07
    """
    receipts = Receipts("witness_emit")
    receipts.put("inputs", {
        "R_out": R_out,
        "C_out": C_out,
        "num_trainings": len(witness_results_all_trainings),
        "num_pieces_total": sum(
            len(wr["pieces"]) for wr in witness_results_all_trainings
        ),
    })

    # Step 0: Prepare test input planes in Pi_in_star frame
    Pi_in_star = frames["Pi_in_star"]
    planes_in_star, H_in_star, W_in_star = _prepare_test_planes(
        X_star, Pi_in_star, colors_order
    )

    # Step 1: Build per-training emitters
    per_training_emitters = []
    per_training_scope_bits = []
    per_training_piece_counts = []
    active_trainings = []

    for i, witness_result in enumerate(witness_results_all_trainings):
        # BUGFIX: Skip trainings not included in transport (failed normalization)
        if included_train_ids is not None and i not in included_train_ids:
            # Training was silent in transport, skip it in witness too
            per_training_emitters.append(({c: [0] * R_out for c in colors_order}, [0] * R_out))
            per_training_scope_bits.append(0)
            per_training_piece_counts.append(0)
            continue

        A_i, S_i = _build_per_training_emitter(
            witness_result,
            i,
            frames,
            planes_in_star,
            H_in_star,
            W_in_star,
            colors_order,
            R_out,
            C_out,
        )
        per_training_emitters.append((A_i, S_i))
        per_training_scope_bits.append(_popcount_scope(S_i))
        per_training_piece_counts.append(len(witness_result.get("pieces", [])))

        # Track active (non-silent) trainings
        if not witness_result.get("silent", True) and witness_result.get("pieces"):
            active_trainings.append(i)

    # Step 2: Global combine via scope-gated intersection
    A_wit, S_wit = _combine_trainings(
        per_training_emitters, colors_order, R_out, C_out
    )

    # Step 3: Normalize global scope - remove admit-none pixels (phantom scope)
    # After intersection, some pixels may have S_wit[p]==1 but ∀c: A_wit[c][p]==0
    # These "phantom scope" bits must be cleared (Sub-WO-M3-SelectorFix)
    for r in range(R_out):
        for c_bit in range(C_out):
            bit = 1 << c_bit
            # Check if ANY color admits this pixel
            has_any_color = any(A_wit[c][r] & bit for c in colors_order)
            if not has_any_color:
                # Clear phantom scope bit - no color actually admitted here
                S_wit[r] &= ~bit

    # Generate receipts
    receipts.put("scopes", {
        "per_training_scope_bits": per_training_scope_bits,
        "scope_bits": _popcount_scope(S_wit),
    })

    receipts.put("admits", {
        **{str(c): A_wit[c] for c in colors_order if c in A_wit},  # Actual color planes
        "A_wit_hash": _hash_planes(A_wit, R_out, C_out, colors_order),
    })

    receipts.put("families", {
        "combined_from_trainings": active_trainings,
    })

    # Optional nice-to-have audit fields
    receipts.put("per_training_piece_counts", per_training_piece_counts)

    # Add debug arrays if requested
    if debug_arrays:
        from ..core.bytesio import serialize_planes_be_row_major, serialize_scope_be_row_major
        receipts.put("debug_arrays", {
            "A_wit_planes_bytes": serialize_planes_be_row_major(A_wit, R_out, C_out, colors_order).hex(),
            "S_wit_bytes": serialize_scope_be_row_major(S_wit, R_out, C_out).hex(),
        })

    # Generate receipts bundle per WO-07 spec
    receipts_bundle = receipts.digest()

    return A_wit, S_wit, receipts_bundle


# ============================================================================
# Frame Algebra
# ============================================================================


def _compose_frames(
    Pi1: Tuple[str, Tuple[int, int]],
    Pi2: Tuple[str, Tuple[int, int]],
) -> Tuple[str, Tuple[int, int]]:
    """
    Compose two frames: Pi1 ∘ Pi2 = (R1·R2, R1·a2 + a1).

    Args:
        Pi1: Frame (R1, a1)
        Pi2: Frame (R2, a2)

    Returns:
        Composed frame (R1·R2, R1·a2 + a1)
    """
    R1, a1 = Pi1
    R2, a2 = Pi2

    # Compose rotations/reflections using D4 algebra
    R_composed = pose_compose(R1, R2)

    # Transform a2 by R1 then add a1
    # R1·a2 = apply R1 to vector a2
    a2_transformed = _apply_d4_to_vector(R1, a2)
    a_composed = (a2_transformed[0] + a1[0], a2_transformed[1] + a1[1])

    return R_composed, a_composed


def _inverse_frame(Pi: Tuple[str, Tuple[int, int]]) -> Tuple[str, Tuple[int, int]]:
    """
    Invert a frame: Pi⁻¹ = (R⁻¹, -R⁻¹·a).

    Args:
        Pi: Frame (R, a)

    Returns:
        Inverse frame (R⁻¹, -R⁻¹·a)
    """
    R, a = Pi

    # Invert rotation/reflection
    R_inv = pose_inverse(R)

    # Compute -R⁻¹·a
    a_neg = (-a[0], -a[1])
    a_inv = _apply_d4_to_vector(R_inv, a_neg)

    return R_inv, a_inv


def _apply_d4_to_vector(R: str, v: Tuple[int, int]) -> Tuple[int, int]:
    """
    Apply D4 element R to vector v = (row, col).

    Args:
        R: D4 element name (e.g., "R90", "FX", "I")
        v: Vector (r, c)

    Returns:
        Transformed vector R·v
    """
    r, c = v

    # D4 transformations on vectors (treating as row/col offsets)
    if R == "I":
        return (r, c)
    elif R == "R90":
        return (-c, r)  # 90° CCW rotation
    elif R == "R180":
        return (-r, -c)
    elif R == "R270":
        return (c, -r)  # 270° CCW = 90° CW
    elif R == "FX":
        return (r, -c)  # Horizontal flip
    elif R == "FXR90":
        return (-c, -r)  # FX then R90
    elif R == "FXR180":
        return (-r, c)  # FX then R180
    elif R == "FXR270":
        return (c, r)  # FX then R270
    else:
        raise ValueError(f"Unknown D4 element: {R}")


# ============================================================================
# Test Input Preparation
# ============================================================================


def _prepare_test_planes(
    X_star: List[List[int]],
    Pi_in_star: Tuple[str, Tuple[int, int]],
    colors_order: List[int],
) -> Tuple[Dict[int, List[int]], int, int]:
    """
    Canonicalize test input to Pi_in_star frame and extract bit-planes.

    Args:
        X_star: Test input grid
        Pi_in_star: Test input engine frame (pose, anchor)
        colors_order: Color list

    Returns:
        Tuple of (planes dict, H_posed, W_posed)
    """
    from src.arcbit.kernel.planes import pack_grid_to_planes
    from src.arcbit.kernel.frames import apply_pose_anchor

    pid_in_star, anchor_in_star = Pi_in_star

    # Get raw grid dimensions
    H_raw = len(X_star)
    W_raw = len(X_star[0]) if H_raw > 0 else 0

    # Convert grid to bit-planes
    planes_raw = pack_grid_to_planes(X_star, H_raw, W_raw, colors_order)

    # Apply pose and anchor
    planes_in_star, H_posed, W_posed = apply_pose_anchor(
        planes_raw, pid_in_star, anchor_in_star, H_raw, W_raw, colors_order
    )

    return planes_in_star, H_posed, W_posed


# ============================================================================
# Per-Training Emission
# ============================================================================


def _build_per_training_emitter(
    witness_result: dict,
    training_idx: int,
    frames: dict,
    planes_in_star: Dict[int, List[int]],
    H_in_star: int,
    W_in_star: int,
    colors_order: List[int],
    R_out: int,
    C_out: int,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Build per-training emitter (A_i, S_i) by conjugating pieces and forward-mapping.

    Args:
        witness_result: WitnessResult from WO-06
        training_idx: Training index i
        frames: All frames dict
        planes_in_star: Test input planes in Pi_in_star
        H_in_star, W_in_star: Test input dimensions in Pi_in_star
        colors_order: Color list
        R_out, C_out: Working canvas dimensions

    Returns:
        A_i: Per-training admit planes (color -> plane)
        S_i: Per-training scope mask
    """
    silent = witness_result.get("silent", True)
    pieces = witness_result.get("pieces", [])

    # If silent or no pieces: admit-all outside scope (scope = 0)
    if silent or not pieces:
        S_i = [0] * R_out
        A_i = {c: [(1 << C_out) - 1] * R_out for c in colors_order}
        return A_i, S_i

    # Initialize: admit-all for every color, scope = 0
    A_i = {c: [(1 << C_out) - 1] * R_out for c in colors_order}
    S_i = [0] * R_out

    # Extract frames for this training
    Pi_in_star = frames["Pi_in_star"]
    Pi_out_star = frames["Pi_out_star"]
    Pi_in_i = frames[f"Pi_in_{training_idx}"]
    Pi_out_i = frames[f"Pi_out_{training_idx}"]

    sigma_i = witness_result.get("sigma", {})

    # DEBUG: Print sigma mapping
    import os
    if os.environ.get("DEBUG_WITNESS"):
        print(f"\n=== WITNESS EMIT - Training {training_idx} ===")
        print(f"  σ mapping: {sigma_i}")
        print(f"  Processing {len(pieces)} pieces")

    # Process each piece
    for piece in pieces:
        _process_piece(
            piece,
            Pi_in_star,
            Pi_out_star,
            Pi_in_i,
            Pi_out_i,
            sigma_i,
            planes_in_star,
            H_in_star,
            W_in_star,
            A_i,
            S_i,
            colors_order,
            R_out,
            C_out,
            training_idx,  # Pass training index for debug
        )

    # Normalize: remove admit-all pixels from scope
    _normalize_admit_all(A_i, S_i, colors_order, R_out, C_out)

    return A_i, S_i


def _process_piece(
    piece: dict,
    Pi_in_star: Tuple[str, Tuple[int, int]],
    Pi_out_star: Tuple[str, Tuple[int, int]],
    Pi_in_i: Tuple[str, Tuple[int, int]],
    Pi_out_i: Tuple[str, Tuple[int, int]],
    sigma_i: Dict[int, int],
    planes_in_star: Dict[int, List[int]],
    H_in_star: int,
    W_in_star: int,
    A_i: Dict[int, List[int]],
    S_i: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int,
    training_idx: int = -1,  # For debug only
):
    """
    Process one piece: conjugate, forward-map, update A_i and S_i.

    Args:
        piece: Piece dict from WO-06
        Pi_in_star, Pi_out_star: Working frames
        Pi_in_i, Pi_out_i: Training frames
        sigma_i: Color permutation for this training
        planes_in_star: Test input planes
        H_in_star, W_in_star: Test input dimensions
        A_i: Per-training admits (modified in-place)
        S_i: Per-training scope (modified in-place)
        colors_order: Color list
        R_out, C_out: Working canvas dimensions
    """
    # Extract piece data
    pid = piece["pid"]
    dy = piece["dy"]
    dx = piece["dx"]
    bbox_src = piece["bbox_src"]

    # Construct φ_i = (R_i, t_i) in (Pi_in_i -> Pi_out_i) coordinates
    phi_i = (pid, (dy, dx))

    # Conjugate: φ_i* = Pi_out_* ∘ Pi_out_i⁻¹ ∘ φ_i ∘ Pi_in_i ∘ Pi_in_*⁻¹
    Pi_in_star_inv = _inverse_frame(Pi_in_star)
    Pi_out_i_inv = _inverse_frame(Pi_out_i)

    # Build 5-frame composition from right to left
    temp1 = _compose_frames(Pi_in_i, Pi_in_star_inv)
    temp2 = _compose_frames(phi_i, temp1)
    temp3 = _compose_frames(Pi_out_i_inv, temp2)
    phi_i_star = _compose_frames(Pi_out_star, temp3)

    pid_star, (dy_star, dx_star) = phi_i_star

    # DEBUG: Diagnostic for conflict pixels as per author's request
    import os
    if os.environ.get("DEBUG_CONJUGATION"):
        print(f"\n=== CONJUGATION DEBUG - Training {training_idx}, Piece ===")
        print(f"  Original φ_i: pose={pid}, t=({dy},{dx})")
        print(f"  Conjugated φ_i*: pose={pid_star}, t*=({dy_star},{dx_star})")
        print(f"  Frames:")
        print(f"    Pi_in_i={Pi_in_i}")
        print(f"    Pi_in_star={Pi_in_star}")
        print(f"    Pi_out_i={Pi_out_i}")
        print(f"    Pi_out_star={Pi_out_star}")

    # Compute target bbox in working canvas
    B_tgt_rows = _compute_target_bbox_mask(
        bbox_src, pid_star, dy_star, dx_star, H_in_star, W_in_star, R_out, C_out
    )

    # Forward-map each color plane
    for c in colors_order:
        c_prime = sigma_i.get(c, c)  # Recolor (identity for untouched)

        # Get test input plane for color c
        plane_c = planes_in_star.get(c, [0] * H_in_star)

        # DEBUG: Show test input at first few pixels
        if os.environ.get("DEBUG_CONJUGATION") and c in [0, 2]:
            import sys
            print(f"    Color {c}: plane_c[0] bits = {bin(plane_c[0]) if plane_c else 'empty'}", file=sys.stderr)

        # Pose the plane (get posed dimensions)
        P, H_posed, W_posed = pose_plane(plane_c, pid_star, H_in_star, W_in_star)

        # DEBUG: Show posed plane
        if os.environ.get("DEBUG_CONJUGATION") and c in [0, 2]:
            print(f"    After pose {pid_star}: P[0] = {bin(P[0]) if P else 'empty'}", file=sys.stderr)

        # Embed posed plane into target canvas with offset
        # (Explicit embedding avoids dimension mismatch with shift_plane)
        T = [0] * R_out
        for r_src in range(len(P)):
            r_tgt = r_src + dy_star
            if 0 <= r_tgt < R_out:
                # Horizontal shift and clip to canvas width
                if dx_star >= 0:
                    row_shifted = (P[r_src] << dx_star) & ((1 << C_out) - 1)
                else:
                    row_shifted = (P[r_src] >> (-dx_star))
                T[r_tgt] = row_shifted

        # Update A_i[c']: within bbox, require T; outside bbox, leave as-is
        # A_i[c'] &= (~B_tgt | T)
        for r in range(R_out):
            mask_bbox = B_tgt_rows[r]
            A_i[c_prime][r] &= (~mask_bbox | T[r])

    # Update scope: S_i |= B_tgt
    for r in range(R_out):
        S_i[r] |= B_tgt_rows[r]


def _compute_target_bbox_mask(
    bbox_src: Tuple[int, int, int, int],
    pid_star: str,
    dy_star: int,
    dx_star: int,
    H_in_star: int,
    W_in_star: int,
    R_out: int,
    C_out: int,
) -> List[int]:
    """
    Compute target bbox mask after pose and shift.

    Args:
        bbox_src: Source bbox (rmin, cmin, rmax, cmax) in Pi_in_i
        pid_star: Conjugated pose
        dy_star, dx_star: Conjugated shift
        H_in_star, W_in_star: Test input dimensions
        R_out, C_out: Working canvas dimensions

    Returns:
        List of row masks (length R_out) for target bbox
    """
    rmin, cmin, rmax, cmax = bbox_src
    H_src = rmax - rmin + 1
    W_src = cmax - cmin + 1

    # Pose the bbox
    if pid_star in ("R90", "R270", "FXR90", "FXR270"):
        H_posed = W_src
        W_posed = H_src
    else:
        H_posed = H_src
        W_posed = W_src

    # Shift by (dy_star, dx_star) and clip to canvas
    rt_min = dy_star
    ct_min = dx_star
    rt_max = rt_min + H_posed - 1
    ct_max = ct_min + W_posed - 1

    # Clip to [0, R_out) × [0, C_out)
    rt_min_clipped = max(0, rt_min)
    ct_min_clipped = max(0, ct_min)
    rt_max_clipped = min(R_out - 1, rt_max)
    ct_max_clipped = min(C_out - 1, ct_max)

    # Build row masks
    B_tgt_rows = [0] * R_out

    if rt_min_clipped <= rt_max_clipped and ct_min_clipped <= ct_max_clipped:
        W_clipped = ct_max_clipped - ct_min_clipped + 1
        row_mask = ((1 << W_clipped) - 1) << ct_min_clipped

        for r in range(rt_min_clipped, rt_max_clipped + 1):
            B_tgt_rows[r] = row_mask

    return B_tgt_rows


def _normalize_admit_all(
    A_i: Dict[int, List[int]],
    S_i: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int,
):
    """
    Remove admit-all pixels from scope (per-training normalization).

    Pixels where all colors are admitted (intersection = all 1s) are
    considered silent and removed from scope.

    Args:
        A_i: Per-training admits (modified in-place)
        S_i: Per-training scope (modified in-place)
        colors_order: Color list
        R_out: Canvas height
        C_out: Canvas width
    """
    for r in range(R_out):
        # Compute AND across all colors
        U_r = (1 << C_out) - 1  # Start with all 1s for canvas width
        for c in colors_order:
            U_r &= A_i[c][r]

        # Clear admit-all pixels from scope
        S_i[r] &= ~U_r


# ============================================================================
# Global Combination
# ============================================================================


def _combine_trainings(
    per_training_emitters: List[Tuple[Dict[int, List[int]], List[int]]],
    colors_order: List[int],
    R_out: int,
    C_out: int,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Combine per-training emitters via scope-gated intersection.

    Args:
        per_training_emitters: List of (A_i, S_i) tuples
        colors_order: Color list
        R_out, C_out: Working canvas dimensions

    Returns:
        A_wit: Global witness admits
        S_wit: Global witness scope
    """
    # Scope union
    S_wit = [0] * R_out
    for A_i, S_i in per_training_emitters:
        for r in range(R_out):
            S_wit[r] |= S_i[r]

    # Admits intersection with scope gating
    A_wit = {c: [(1 << C_out) - 1] * R_out for c in colors_order}

    for A_i, S_i in per_training_emitters:
        for c in colors_order:
            for r in range(R_out):
                # Outside scope: admit all (M_i_c = all 1s)
                # Inside scope: use A_i[c][r]
                M_i_c_r = A_i[c][r] | (~S_i[r])
                A_wit[c][r] &= M_i_c_r

    return A_wit, S_wit


# ============================================================================
# Receipts Helpers
# ============================================================================


def _popcount_scope(S: List[int]) -> int:
    """Count total bits set in scope mask."""
    total = 0
    for row_mask in S:
        total += bin(row_mask).count("1")
    return total


def _hash_planes(
    A: Dict[int, List[int]],
    R_out: int,
    C_out: int,
    colors_order: List[int],
) -> str:
    """
    Compute BLAKE3 hash of admit planes.

    Args:
        A: Admit planes (color -> plane)
        R_out, C_out: Canvas dimensions
        colors_order: Color list

    Returns:
        BLAKE3 hash (hex string)
    """
    # Serialize planes in canonical order
    serialized = {
        "R_out": R_out,
        "C_out": C_out,
        "colors_order": colors_order,
        "planes": {str(c): A.get(c, [0] * R_out) for c in colors_order},
    }

    canonical_json = json.dumps(serialized, sort_keys=True, separators=(",", ":"))

    # Use BLAKE3 hash from core
    return blake3_hash(canonical_json.encode("utf-8"))
