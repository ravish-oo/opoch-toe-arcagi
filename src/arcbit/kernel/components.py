"""
WO-05: 4-CC Components & Shape Invariants (bit-ops only)

Pure bit-algebra extraction of 4-connected components from bit-planes.
Computes bbox, area, 4-neighbor perimeter, and D4-minimal outline hash.

Spec: WO-05 (v1.5 + v1.6)
"""

from typing import Dict, List, Tuple, TypedDict
from ..core.hashing import blake3_hash
from ..core.bytesio import serialize_grid_be_row_major
from .ops import pose_plane


class Component(TypedDict):
    """4-CC component with shape invariants."""
    color: int
    bbox: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max), inclusive
    mask_plane: List[int]            # H-length row masks (original coordinates)
    area: int
    perim4: int
    outline_hash: str                # BLAKE3 hex of D4-min cropped bytes


def components(
    planes: Dict[int, List[int]],
    H: int,
    W: int,
    colors_order: List[int]
) -> List[Component]:
    """
    Extract all 4-CC components for colors in colors_order EXCEPT color 0.

    Algorithm:
      1. Validate inputs (H rows, bits in [0..W-1])
      2. For each color (except 0):
         - Residual mask R = planes[color]
         - While R has bits:
           * Find row-major earliest bit (seed)
           * Grow 4-CC via bit-ops BFS
           * Compute invariants: bbox, area, perim4, outline_hash
           * Remove component from R
      3. Assert union/overlap invariants
      4. Generate receipts

    Args:
        planes: Dict mapping color → H row masks (WO-01 format).
        H: Grid height.
        W: Grid width.
        colors_order: Ascending integers, includes 0.

    Returns:
        List of Component dicts (excluding color 0 by default).

    Raises:
        ValueError: If row count != H or bits outside [0..W-1].

    Spec:
        WO-05: Pure bit-ops; deterministic seed selection; no heuristics.
    """
    # A) Validate inputs
    for color in colors_order:
        if color not in planes:
            raise ValueError(f"WO-05: Color {color} not in planes dict")

        plane = planes[color]
        if len(plane) != H:
            raise ValueError(
                f"WO-05: Color {color} plane has {len(plane)} rows, expected {H}"
            )

        # Check no bits outside [0..W-1]
        for r, row_mask in enumerate(plane):
            if (row_mask >> W) != 0:
                raise ValueError(
                    f"WO-05: Color {color} row {r} has bits outside [0..{W-1}]: "
                    f"mask={bin(row_mask)}"
                )

    # B) Extract components for each color (except 0)
    all_components = []
    per_color_summary = []

    for color in colors_order:
        if color == 0:
            continue  # Skip background by default

        # Copy residual mask
        R = list(planes[color])
        color_components = []

        # While residual has any bits
        while any(r_mask != 0 for r_mask in R):
            # Find first seed (row-major earliest bit)
            seed_r = None
            seed_c = None
            for r in range(H):
                if R[r] != 0:
                    seed_r = r
                    seed_c = _lsb_index(R[r])  # Least significant bit set
                    break

            assert seed_r is not None and seed_c is not None

            # Initialize component mask C and frontier F
            C = [0] * H
            F = [0] * H
            F[seed_r] = (1 << seed_c)

            # Grow 4-CC via bit-ops BFS
            while any(f_mask != 0 for f_mask in F):
                # Horizontal neighbors (same row)
                Hnb = [0] * H
                full_mask = (1 << W) - 1 if W > 0 else 0
                for r in range(H):
                    left_shift = (F[r] << 1) & full_mask
                    right_shift = (F[r] >> 1)
                    Hnb[r] = left_shift | right_shift

                # Vertical neighbors
                Vnb = [0] * H
                for r in range(H):
                    above = F[r - 1] if r > 0 else 0
                    below = F[r + 1] if r + 1 < H else 0
                    Vnb[r] = above | below

                # Update component with current frontier
                for r in range(H):
                    C[r] |= F[r]
                    R[r] &= ~F[r]  # Remove from residual IMMEDIATELY (prevents revisiting)

                # Compute next frontier (constrained to remaining residual)
                N = [0] * H
                for r in range(H):
                    N[r] = (Hnb[r] | Vnb[r]) & R[r]

                F = N

            # Component complete (residual already updated inside BFS loop)

            # Compute invariants for this component
            comp = _compute_component_invariants(C, H, W, color, colors_order)
            color_components.append(comp)
            all_components.append(comp)

        # Per-color summary
        if color_components:
            areas = [c['area'] for c in color_components]
            perim4s = [c['perim4'] for c in color_components]
            per_color_summary.append({
                "color": color,
                "n_cc": len(color_components),
                "area_sum": sum(areas),
                "area_min": min(areas),
                "area_max": max(areas),
                "perim4_sum": sum(perim4s)
            })
        else:
            per_color_summary.append({
                "color": color,
                "n_cc": 0,
                "area_sum": 0,
                "area_min": None,
                "area_max": None,
                "perim4_sum": 0
            })

    # C) Assert union/overlap invariants
    union_equal_input = True
    overlap_zero = True

    for color in colors_order:
        if color == 0:
            continue

        # Get all components for this color
        color_comps = [c for c in all_components if c['color'] == color]

        if not color_comps:
            # No components: plane should be all zeros
            if any(mask != 0 for mask in planes[color]):
                union_equal_input = False
            continue

        # Union should equal input
        union_mask = [0] * H
        for comp in color_comps:
            for r in range(H):
                union_mask[r] |= comp['mask_plane'][r]

        if union_mask != planes[color]:
            union_equal_input = False

        # Pairwise overlaps should be zero
        for i, comp_i in enumerate(color_comps):
            for comp_j in color_comps[i + 1:]:
                for r in range(H):
                    if (comp_i['mask_plane'][r] & comp_j['mask_plane'][r]) != 0:
                        overlap_zero = False
                        break

    if not union_equal_input:
        raise ValueError("WO-05: Union of component masks != input plane (content mismatch)")

    if not overlap_zero:
        raise ValueError("WO-05: Component masks overlap (non-disjoint components)")

    # D) Generate receipts (TODO: add to return via wrapper or global state)
    # For now, we rely on assertions above to catch errors

    return all_components


# ============================================================================
# Helper Functions
# ============================================================================

def _lsb_index(mask: int) -> int:
    """
    Find index of least significant bit set (rightmost 1-bit).

    Args:
        mask: Non-zero integer.

    Returns:
        Bit index (0-based from right).

    Raises:
        ValueError: If mask is zero.
    """
    if mask == 0:
        raise ValueError("Cannot find LSB of zero mask")

    idx = 0
    while (mask & 1) == 0:
        mask >>= 1
        idx += 1
    return idx


def _popcount(mask: int) -> int:
    """
    Count number of 1-bits in mask.

    Args:
        mask: Integer bit mask.

    Returns:
        Number of set bits.

    Spec:
        Brian Kernighan's algorithm (mask &= mask-1 clears lowest bit).
    """
    count = 0
    while mask:
        count += 1
        mask &= mask - 1  # Clear lowest bit
    return count


def _compute_component_invariants(
    C: List[int],
    H: int,
    W: int,
    color: int,
    colors_order: List[int]
) -> Component:
    """
    Compute all invariants for a component mask C.

    Args:
        C: Component mask (H row masks).
        H: Grid height.
        W: Grid width.
        color: Component color.
        colors_order: Color palette (for serialization).

    Returns:
        Component dict with bbox, area, perim4, outline_hash, mask_plane.

    Spec:
        WO-05: Exact algebraic formulas for all invariants.
    """
    # 1. Area
    area = sum(_popcount(C[r]) for r in range(H))

    # 2. BBox
    r_min = None
    r_max = None
    c_min = W
    c_max = -1

    for r in range(H):
        if C[r] != 0:
            if r_min is None:
                r_min = r
            r_max = r

            # Find min/max bit indices in this row
            for c in range(W):
                if C[r] & (1 << c):
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c

    assert r_min is not None, "Component must have at least one bit"

    # 3. Perimeter (4-neighbor exact formula)
    U = area

    # Shared vertical edges: Sv = Σ popcount(C[r] & C[r-1]) for r=1..H-1
    Sv = 0
    for r in range(1, H):
        Sv += _popcount(C[r] & C[r - 1])

    # Shared horizontal edges: Sh = Σ popcount(C[r] & (C[r] << 1)) for all rows
    Sh = 0
    for r in range(H):
        Sh += _popcount(C[r] & (C[r] << 1))

    # Perimeter: 4*U - 2*(Sv + Sh)
    perim4 = 4 * U - 2 * (Sv + Sh)

    # 4. D4-minimal outline hash
    outline_hash = _compute_outline_hash(
        C, H, W, r_min, r_max, c_min, c_max, colors_order
    )

    # 5. mask_plane storage (H-length with original coordinates)
    mask_plane = list(C)

    return Component(
        color=color,
        bbox=(r_min, c_min, r_max, c_max),
        mask_plane=mask_plane,
        area=area,
        perim4=perim4,
        outline_hash=outline_hash
    )


def _compute_outline_hash(
    C: List[int],
    H: int,
    W: int,
    r_min: int,
    r_max: int,
    c_min: int,
    c_max: int,
    colors_order: List[int]
) -> str:
    """
    Compute D4-minimal shape hash (translation-free outline hash).

    Algorithm:
      1. Crop to bbox
      2. For each D4 pose:
         - Apply pose (WO-01 pose_plane)
         - Re-crop to remove new outer zeros
         - Serialize to bytes (WO-00 format with H, W header)
      3. Take lex-min bytes
      4. BLAKE3 hash

    Args:
        C: Component mask.
        H, W: Grid dimensions.
        r_min, r_max, c_min, c_max: Bounding box (inclusive).
        colors_order: Color palette.

    Returns:
        BLAKE3 hex digest of lex-min serialized bytes.

    Spec:
        WO-05: D4-minimal outline hash for translation-free shape comparison.
    """
    # Crop to bbox
    Hc = r_max - r_min + 1
    Wc = c_max - c_min + 1

    C_crop = [0] * Hc
    for r in range(Hc):
        src_r = r + r_min
        # Extract bits [c_min..c_max] from source row
        src_mask = C[src_r]
        # Shift right to align c_min to bit 0
        cropped_mask = (src_mask >> c_min) & ((1 << Wc) - 1)
        C_crop[r] = cropped_mask

    # Enumerate D4 poses
    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    min_bytes = None

    for pid in pose_ids:
        # Apply pose
        C_posed, H_posed, W_posed = pose_plane(C_crop, pid, Hc, Wc)

        # Re-crop to remove any new outer zeros
        C_recrop, H_recrop, W_recrop = _recrop_to_bbox(C_posed, H_posed, W_posed)

        # Serialize with WO-00 format
        # Convert plane to grid (single color, use value 1)
        G_temp = [[0] * W_recrop for _ in range(H_recrop)]
        for r in range(H_recrop):
            for c in range(W_recrop):
                if C_recrop[r] & (1 << c):
                    G_temp[r][c] = 1  # Use dummy color 1

        # Serialize (use minimal color set [0, 1])
        dummy_colors = [0, 1]
        grid_bytes = serialize_grid_be_row_major(G_temp, H_recrop, W_recrop, dummy_colors)

        # Track lex-min
        if min_bytes is None or grid_bytes < min_bytes:
            min_bytes = grid_bytes

    # BLAKE3 hash
    return blake3_hash(min_bytes)


def _recrop_to_bbox(
    plane: List[int],
    H: int,
    W: int
) -> Tuple[List[int], int, int]:
    """
    Re-crop plane to remove all-zero border rows/cols.

    Args:
        plane: Row masks.
        H: Height.
        W: Width.

    Returns:
        Tuple of (cropped_plane, H_crop, W_crop).
        Returns ([], 0, 0) if all zeros.

    Spec:
        WO-05 helper: Trim all-zero top, bottom, left, right borders only.
        Interior zeros remain.
    """
    if H == 0 or W == 0:
        return [], 0, 0

    # Find bbox
    r_min = None
    r_max = None
    c_min = W
    c_max = -1

    for r in range(H):
        if plane[r] != 0:
            if r_min is None:
                r_min = r
            r_max = r

            for c in range(W):
                if plane[r] & (1 << c):
                    if c < c_min:
                        c_min = c
                    if c > c_max:
                        c_max = c

    # All zeros?
    if r_max is None:
        return [], 0, 0

    # Crop
    Hc = r_max - r_min + 1
    Wc = c_max - c_min + 1

    plane_crop = [0] * Hc
    for r in range(Hc):
        src_r = r + r_min
        src_mask = plane[src_r]
        cropped_mask = (src_mask >> c_min) & ((1 << Wc) - 1)
        plane_crop[r] = cropped_mask

    return plane_crop, Hc, Wc
