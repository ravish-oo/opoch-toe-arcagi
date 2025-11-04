"""
WO-06: Witness Matcher (per training)

Learns rigid pieces φ_k: p ↦ R_k·p + t_k (R_k ∈ D4, t_k ∈ ℤ²)
and color permutation σ_i: colors(X_i) → colors(Y_i)

Exact bitwise matching, no free translation scan, receipts-first.
"""

from typing import TypedDict, List, Dict, Tuple, Optional, Any
from ..core import Receipts, blake3_hash
from ..kernel import (
    pack_grid_to_planes,
    unpack_planes_to_grid,
    pose_plane,
    shift_plane,
    canonicalize,
    order_colors
)
from ..kernel.components import components
import json


# ============================================================================
# Type Definitions
# ============================================================================

class Piece(TypedDict):
    """Rigid piece: R·p + t mapping from source to target component."""
    pid: str  # Pose ID: one of D4
    dy: int   # Translation in rows (engine frame)
    dx: int   # Translation in cols (engine frame)
    bbox_src: Tuple[int, int, int, int]  # (rmin, cmin, rmax, cmax) in X
    bbox_tgt: Tuple[int, int, int, int]  # (rmin, cmin, rmax, cmax) in Y
    c_in: int   # Source color (for overlap checking)
    c_out: int  # Target color (for overlap checking)


class WitnessResult(TypedDict):
    """Result of witness learning for one training pair."""
    pieces: List[Piece]
    sigma: Dict[int, int]  # color_in → color_out (bijection on touched)
    silent: bool           # True if conflict prevents using witness
    receipts: Dict         # Sealed section receipts


# ============================================================================
# Constants
# ============================================================================

# Frozen D4 pose order
D4_ORDER = ("I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270")


# ============================================================================
# Main Entry Point
# ============================================================================

def learn_witness(Xi_raw: List[List[int]], Yi_raw: List[List[int]], frames: Dict) -> WitnessResult:
    """
    Learn witness (rigid pieces + color permutation) for one training pair.

    Args:
        Xi_raw: Training input grid (raw ARC ints).
        Yi_raw: Training output grid (raw ARC ints).
        frames: Dict with Pi_in=(pid_in, anchor_in) and Pi_out=(pid_out, anchor_out).

    Returns:
        WitnessResult with pieces, sigma, silent flag, and receipts.

    Spec:
        WO-06: Exact bitwise matching with D4-invariant outline hashes.
        No free translation scan: t computed from bbox alignment.
        Receipts-first: full audit trail of all trials.

    Invariants:
        - No minted bits: pieces accepted only if exact bitwise match
        - No free translation: t = tgt.bbox.min - src_posed.bbox.min
        - σ injective: each c_in maps to unique c_out
        - Overlap conflict → silent
        - Bijection on touched colors required
    """
    receipts = Receipts("witness_train")

    # Define allowed keys for Piece in receipts (no internal fields)
    ALLOWED_PIECE_KEYS = ("pid", "dy", "dx", "bbox_src", "bbox_tgt", "c_in", "c_out")

    # Extract frames
    Pi_in = frames.get("Pi_in")
    Pi_out = frames.get("Pi_out")

    if Pi_in is None or Pi_out is None:
        raise ValueError("WO-06: frames must contain Pi_in and Pi_out")

    pid_in, anchor_in = Pi_in
    pid_out, anchor_out = Pi_out

    # Get dimensions
    H_in = len(Xi_raw)
    W_in = len(Xi_raw[0]) if Xi_raw else 0
    H_out = len(Yi_raw)
    W_out = len(Yi_raw[0]) if Yi_raw else 0

    receipts.put("inputs", {
        "H_in": H_in,
        "W_in": W_in,
        "H_out": H_out,
        "W_out": W_out
    })

    # Build color universe (must include 0)
    color_set_in = {0}  # Always include background
    for row in Xi_raw:
        for val in row:
            color_set_in.add(val)
    colors_in = order_colors(color_set_in)

    color_set_out = {0}  # Always include background
    for row in Yi_raw:
        for val in row:
            color_set_out.add(val)
    colors_out = order_colors(color_set_out)

    # Derive engine views: X = Pose(Xi_raw, pid_in) then translate by -anchor_in
    # Y = Pose(Yi_raw, pid_out) then translate by -anchor_out
    X = _apply_frame_to_engine(Xi_raw, H_in, W_in, pid_in, anchor_in, colors_in)
    Y = _apply_frame_to_engine(Yi_raw, H_out, W_out, pid_out, anchor_out, colors_out)

    # Handle all-zero grids
    if not X or not Y:
        receipts.put("trials", [])
        receipts.put("pieces", [])
        receipts.put("sigma", {
            "map": {},
            "bijection_ok": True,
            "touched_in": [],
            "touched_out": []
        })
        receipts.put("overlap", {
            "conflict": False,
            "conflict_pixels": 0,
            "per_color_overlaps": {}
        })
        receipts.put("silent", False)
        receipts_bundle = receipts.digest()
        return WitnessResult(
            pieces=[],
            sigma={},
            silent=False,
            receipts=receipts_bundle
        )

    # Precompute components and outline hashes (WO-05)
    X_comps, X_receipts = components(
        pack_grid_to_planes(X, len(X), len(X[0]) if X else 0, colors_in),
        len(X),
        len(X[0]) if X else 0,
        colors_in
    )

    Y_comps, Y_receipts = components(
        pack_grid_to_planes(Y, len(Y), len(Y[0]) if Y else 0, colors_out),
        len(Y),
        len(Y[0]) if Y else 0,
        colors_out
    )

    # Build by_hash_anycolor index on Y components
    by_hash_anycolor = _build_hash_index(Y_comps, colors_out)

    # Enumerate and match components
    trials = []
    pieces = []
    sigma = {}  # c_in → c_out

    # Track which c_out values are taken (for injectivity)
    sigma_inverse = {}  # c_out → c_in

    # Enumerate colors in ascending order (exclude 0)
    for c_in in sorted(colors_in):
        if c_in == 0:
            continue  # Skip background

        # Get components for this color from X
        X_color_comps = [comp for comp in X_comps if comp["color"] == c_in]

        # Sort components by (rmin, cmin, -area) for determinism
        X_color_comps_sorted = sorted(
            X_color_comps,
            key=lambda c: (c["bbox"][0], c["bbox"][1], -c["area"])
        )

        # Enumerate components in row-major order
        for C_src in X_color_comps_sorted:
            bbox_src = C_src["bbox"]  # (rmin, cmin, rmax, cmax)
            mask_src = C_src["mask_plane"]  # Full grid mask
            h_src = C_src["outline_hash"]  # D4-minimal hash from WO-05

            # Get candidate targets from Y with same D4-minimal outline hash
            candidates = by_hash_anycolor.get(h_src, [])

            # Try each pose in frozen D4 order to find which one matches
            for pid in D4_ORDER:
                # Pose the source mask (full global mask)
                M_src_posed, H_posed, W_posed = pose_plane(mask_src, pid, len(X), len(X[0]) if X else 0)

                # Compute posed bbox
                bbox_src_posed = _compute_posed_bbox(bbox_src, len(X), len(X[0]) if X else 0, pid)
                rmin_p, cmin_p, rmax_p, cmax_p = bbox_src_posed

                # Sort candidates by (color_out asc, rmin, cmin, -area)
                candidates_sorted = sorted(
                    candidates,
                    key=lambda c: (c["color"], c["bbox"][0], c["bbox"][1], -c["area"])
                )

                # Try each candidate in order
                piece_accepted = False
                for C_tgt in candidates_sorted:
                    c_out = C_tgt["color"]
                    bbox_tgt = C_tgt["bbox"]  # (rt_min, ct_min, rt_max, ct_max)
                    rt_min, ct_min, rt_max, ct_max = bbox_tgt
                    mask_tgt = C_tgt["mask_plane"]

                    # Compute translation: align posed source top-left to target top-left
                    dy = rt_min - rmin_p
                    dx = ct_min - cmin_p

                    # Shift posed source mask by (dy, dx)
                    M_shifted = shift_plane(M_src_posed, dy, dx, len(Y), len(Y[0]) if Y else 0)

                    # Check exact bitwise match
                    match_ok = _check_exact_match(M_shifted, mask_tgt, len(Y), len(Y[0]) if Y else 0)

                    # Log trial
                    trial = {
                        "src_color": c_in,
                        "src_bbox": list(bbox_src),
                        "pose": pid,
                        "t": [dy, dx],
                        "outline_hash": h_src,
                        "tgt_color": c_out,
                        "tgt_bbox": list(bbox_tgt),
                        "ok": match_ok
                    }
                    trials.append(trial)

                    if match_ok:
                        # Check σ consistency
                        sigma_ok = True

                        # Check if c_in already mapped
                        if c_in in sigma:
                            if sigma[c_in] != c_out:
                                # Conflict: c_in maps to different c_out
                                sigma_ok = False
                        else:
                            # Check if c_out already taken by another c_in
                            if c_out in sigma_inverse:
                                # Injectivity violation
                                sigma_ok = False
                            else:
                                # Accept mapping
                                sigma[c_in] = c_out
                                sigma_inverse[c_out] = c_in

                        if sigma_ok:
                            # Accept piece
                            piece = Piece(
                                pid=pid,
                                dy=dy,
                                dx=dx,
                                bbox_src=tuple(bbox_src),
                                bbox_tgt=tuple(bbox_tgt),
                                c_in=c_in,
                                c_out=c_out
                            )
                            # Store bbox-cropped mask for overlap checking (not full global mask)
                            mask_local, h_local, w_local = _crop_mask_to_bbox(mask_src, bbox_src)
                            piece["_mask_local"] = mask_local  # Internal, bbox-local coordinates
                            piece["_h_src"] = h_local
                            piece["_w_src"] = w_local
                            pieces.append(piece)
                            piece_accepted = True
                            break  # First valid target wins

                if piece_accepted:
                    # First valid target wins for this source component (determinism)
                    break  # Stop trying other poses for this source component

    # Check overlap conflicts
    overlap_conflict, conflict_pixels, per_color_overlaps = _check_overlap_conflicts(
        pieces, sigma, len(Y), len(Y[0]) if Y else 0
    )

    # Check bijection on touched colors
    touched_in = list(sigma.keys())
    touched_out = list(set(sigma.values()))
    bijection_ok = len(touched_in) == len(touched_out) == len(sigma)

    # Determine if witness is silent
    silent = overlap_conflict or not bijection_ok

    if silent:
        # Witness is invalid: clear pieces and sigma
        final_pieces = []
        final_sigma = {}
    else:
        final_pieces = pieces
        final_sigma = sigma

    # Build receipts
    receipts.put("trials", trials)
    receipts.put("pieces", [
        {k: p[k] for k in ALLOWED_PIECE_KEYS} for p in final_pieces
    ])
    receipts.put("sigma", {
        "map": {str(k): v for k, v in final_sigma.items()},  # Convert int keys to strings
        "bijection_ok": bijection_ok,
        "touched_in": sorted(touched_in),
        "touched_out": sorted(touched_out)
    })
    receipts.put("overlap", {
        "conflict": overlap_conflict,
        "conflict_pixels": conflict_pixels,
        "per_color_overlaps": {str(k): v for k, v in per_color_overlaps.items()}  # Convert int keys to strings
    })
    receipts.put("silent", silent)

    receipts_bundle = receipts.digest()

    return WitnessResult(
        pieces=final_pieces,
        sigma=final_sigma,
        silent=silent,
        receipts=receipts_bundle
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _apply_frame_to_engine(
    G_raw: List[List[int]],
    H: int,
    W: int,
    pid: str,
    anchor: Tuple[int, int],
    colors_order: List[int]
) -> List[List[int]]:
    """
    Apply frame transformation to get engine view.

    Steps:
        1. Pose grid by pid
        2. Translate by -anchor (shift opposite direction)

    Returns:
        Engine-view grid (zero-filled if translation shifts content out).
    """
    if H == 0 or W == 0:
        return []

    # Pack to planes
    planes = pack_grid_to_planes(G_raw, H, W, colors_order)

    # Apply pose (this changes dimensions for R90/R270)
    planes_posed = {}
    for color in colors_order:
        planes_posed[color], _, _ = pose_plane(planes[color], pid, H, W)

    # Get new dimensions after pose
    if pid in ("R90", "R270", "FXR90", "FXR270"):
        H_posed, W_posed = W, H
    else:
        H_posed, W_posed = H, W

    # Translate by -anchor
    dy, dx = anchor
    planes_shifted = {}

    # First shift all non-zero colors
    for color in colors_order:
        if color != 0:
            planes_shifted[color] = shift_plane(planes_posed[color], -dy, -dx, H_posed, W_posed)

    # Recompute color 0 as complement of all other colors
    # Color 0 should be ON wherever no other color is ON
    mask_all_cols = (1 << W_posed) - 1 if W_posed > 0 else 0
    plane_0 = []
    for r in range(H_posed):
        # OR together all non-zero color masks for this row
        union_mask = 0
        for color in colors_order:
            if color != 0:
                union_mask |= planes_shifted[color][r]
        # Color 0 is the complement
        plane_0.append((~union_mask) & mask_all_cols)
    planes_shifted[0] = plane_0

    # Unpack back to grid
    G_engine = unpack_planes_to_grid(planes_shifted, H_posed, W_posed, colors_order)

    return G_engine


def _build_hash_index(Y_comps: List[Dict], colors_order: List[int]) -> Dict[str, List[Dict]]:
    """
    Build by_hash_anycolor index: outline_hash → list of (color, comp).

    For each Y component, compute its D4-minimal outline hash and index it.
    """
    index = {}

    for comp in Y_comps:
        color = comp["color"]
        if color == 0:
            continue  # Skip background

        # Compute outline hash (D4-minimal from component receipts)
        # WO-05 already provides outline_hash in component
        h = comp.get("outline_hash")
        if h is None:
            # Compute it manually if not provided
            h = _compute_outline_hash_from_comp(comp)

        if h not in index:
            index[h] = []
        index[h].append(comp)

    return index


def _crop_mask_to_bbox(
    mask_full: List[int],
    bbox: Tuple[int, int, int, int]
) -> Tuple[List[int], int, int]:
    """
    Extract local mask cropped to bbox with (0,0)-anchored coordinates.

    Args:
        mask_full: Full mask plane (H rows, global columns)
        bbox: (rmin, cmin, rmax, cmax) inclusive

    Returns:
        Tuple of (local_mask, H_src, W_src) where local_mask has bbox dims
    """
    rmin, cmin, rmax, cmax = bbox
    H_src = rmax - rmin + 1
    W_src = cmax - cmin + 1

    local = []
    full_mask = (1 << W_src) - 1 if W_src > 0 else 0

    for r in range(H_src):
        row = mask_full[rmin + r]
        # Shift right by cmin to bring bbox left edge to bit 0, then clamp to width
        local.append((row >> cmin) & full_mask)

    return local, H_src, W_src


def _compute_outline_hash_from_comp(comp: Dict) -> str:
    """Get D4-minimal outline hash from component (WO-05 provides this)."""
    h = comp.get("outline_hash")
    if not h:
        raise ValueError("WO-05 component missing outline_hash; cannot match witness")
    return h


def _compute_posed_bbox(
    bbox: Tuple[int, int, int, int],
    H: int,
    W: int,
    pid: str
) -> Tuple[int, int, int, int]:
    """
    Compute bounding box after applying pose.

    For a bbox (rmin, cmin, rmax, cmax), apply pose pid and return new bbox.
    """
    rmin, cmin, rmax, cmax = bbox

    # Create corners of bbox
    corners = [
        (rmin, cmin),
        (rmin, cmax),
        (rmax, cmin),
        (rmax, cmax)
    ]

    # Transform each corner
    posed_corners = []
    for r, c in corners:
        r_new, c_new = _apply_pose_to_point(r, c, H, W, pid)
        posed_corners.append((r_new, c_new))

    # Compute new bbox
    r_vals = [r for r, c in posed_corners]
    c_vals = [c for r, c in posed_corners]

    return (min(r_vals), min(c_vals), max(r_vals), max(c_vals))


def _apply_pose_to_point(r: int, c: int, H: int, W: int, pid: str) -> Tuple[int, int]:
    """Apply pose transformation to a point (r, c)."""
    if pid == "I":
        return (r, c)
    elif pid == "R90":
        return (c, H - 1 - r)
    elif pid == "R180":
        return (H - 1 - r, W - 1 - c)
    elif pid == "R270":
        return (W - 1 - c, r)
    elif pid == "FX":
        return (r, W - 1 - c)
    elif pid == "FXR90":
        return (c, W - 1 - r)
    elif pid == "FXR180":
        return (H - 1 - r, c)
    elif pid == "FXR270":
        return (W - 1 - c, H - 1 - r)
    else:
        raise ValueError(f"Unknown pose ID: {pid}")


def _check_exact_match(
    M1: List[int],
    M2: List[int],
    H: int,
    W: int
) -> bool:
    """Check if two mask planes are exactly equal (rowwise)."""
    if len(M1) != len(M2):
        return False

    for i in range(min(H, len(M1), len(M2))):
        if M1[i] != M2[i]:
            return False

    return True


def _check_overlap_conflicts(
    pieces: List[Piece],
    sigma: Dict[int, int],
    H: int,
    W: int
) -> Tuple[bool, int, Dict[int, int]]:
    """
    Check for overlap conflicts among accepted pieces.

    Returns:
        (conflict, conflict_pixels, per_color_overlaps)

    Conflict occurs if two pieces assign different c_out to same pixel.
    """
    # Track global ownership: (r, c) → c_out
    global_owner = {}
    per_color_overlaps = {}
    conflict = False
    conflict_pixels_set = set()

    for piece in pieces:
        c_out = piece["c_out"]

        # Reconstruct target mask for this piece
        # T_piece = shift(pose(mask_local, pid), (dy, dx))
        # These fields are ALWAYS set at piece acceptance (lines 280-282)
        mask_local = piece["_mask_local"]
        h_src = piece["_h_src"]
        w_src = piece["_w_src"]

        pid = piece["pid"]
        bbox_tgt = piece["bbox_tgt"]

        # Pose the bbox-local mask with correct dimensions
        M_posed, _, _ = pose_plane(mask_local, pid, h_src, w_src)

        # Get posed dimensions (from bbox-local dimensions)
        if pid in ("R90", "R270", "FXR90", "FXR270"):
            H_posed, W_posed = w_src, h_src
        else:
            H_posed, W_posed = h_src, w_src

        # Shift the posed local mask to target bbox position in global coordinates
        # bbox_tgt gives the target position where this piece should be placed
        rt_min, ct_min = bbox_tgt[0], bbox_tgt[1]
        M_shifted = shift_plane(M_posed, rt_min, ct_min, H, W)

        # Check each pixel in shifted mask
        for r in range(H):
            if r >= len(M_shifted):
                break
            row_mask = M_shifted[r]
            if row_mask == 0:
                continue

            for c in range(W):
                bit = (row_mask >> c) & 1
                if bit == 0:
                    continue

                # Pixel (r, c) is set by this piece
                if (r, c) in global_owner:
                    # Overlap detected
                    prev_c_out = global_owner[(r, c)]
                    if prev_c_out != c_out:
                        # Different c_out → conflict
                        conflict = True
                        conflict_pixels_set.add((r, c))
                    else:
                        # Same c_out → overlap but not conflict (per-color overlap)
                        if c_out not in per_color_overlaps:
                            per_color_overlaps[c_out] = 0
                        per_color_overlaps[c_out] += 1
                else:
                    global_owner[(r, c)] = c_out

    conflict_pixels = len(conflict_pixels_set)

    return (conflict, conflict_pixels, per_color_overlaps)
