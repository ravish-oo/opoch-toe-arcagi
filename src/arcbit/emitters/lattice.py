"""
WO-09 Component: Lattice Emitter

Detect global 2D period on training outputs and emit residue-class admits
where all trainings agree.

Spec: WO-09 v1.6
"""

from typing import List, Dict, Tuple, Optional, TypedDict
from ..core.hashing import blake3_hash
from ..kernel.period import period_2d_planes
from ..kernel.planes import pack_grid_to_planes


class LatticeReceipt(TypedDict, total=False):
    # Required fields
    included_train_ids: List[int]
    p_r: Optional[int]  # proper minimal period ≥2 if present, else None (kept for backward compat)
    p_c: Optional[int]  # kept for backward compat
    per_training_periods: List[Dict]  # [{"train_id": i, "p_r": ..., "p_c": ...}, ...]
    global_p_r: Optional[int]  # same as p_r (clearer naming)
    global_p_c: Optional[int]  # same as p_c (clearer naming)
    global_validated: bool  # true if agreeing_classes non-empty
    agreeing_classes: List[Tuple[int, int]]  # residue pairs (i,j) where all agree
    disagreeing_classes: List[Tuple[int, int]]  # residue pairs with disagreement
    residue_scope_bits: int  # popcount of S_lat
    A_lat_hash: str
    S_lat_hash: str

    # Optional debug fields (only when debug_arrays=True and residue_scope_bits > 0)
    A_lat_planes_bytes: str  # hex-encoded bytes
    S_lat_bytes: str  # hex-encoded bytes


def emit_lattice(
    A_out_list: List[Dict[int, List[int]]],
    S_out_list: List[List[int]],
    colors_order: List[int],
    R_out: int,
    C_out: int,
    debug_arrays: bool = False
) -> Tuple[Dict[int, List[int]], List[int], LatticeReceipt]:
    """
    Emit lattice admits for residue classes where all trainings agree.

    Args:
        A_out_list: Per-training admits (color -> plane) from WO-08
        S_out_list: Per-training scope masks from WO-08
        colors_order: Ascending color list
        R_out, C_out: Working canvas dimensions

    Returns:
        A_lat: Lattice admits (color -> plane)
        S_lat: Lattice scope mask
        receipt: LatticeReceipt

    Spec: WO-09 v1.6
    """
    # Step 0: Determine included trainings (non-silent from WO-08)
    included_train_ids = []
    for i, S_i in enumerate(S_out_list):
        has_scope = any(row != 0 for row in S_i)
        if has_scope:
            included_train_ids.append(i)

    # If no included trainings → silent layer
    if not included_train_ids:
        A_lat = {c: [0] * R_out for c in colors_order}
        S_lat = [0] * R_out
        receipt = LatticeReceipt(
            included_train_ids=[],
            p_r=None,
            p_c=None,
            per_training_periods=[],
            global_p_r=None,
            global_p_c=None,
            global_validated=False,
            agreeing_classes=[],
            disagreeing_classes=[],
            residue_scope_bits=0,
            A_lat_hash=_hash_planes(A_lat, R_out, C_out, colors_order),
            S_lat_hash=_hash_scope(S_lat, C_out),
        )
        return A_lat, S_lat, receipt

    # Step 1: Reconstruct per-training grids Y_i from planes
    Y_list = []
    for i in included_train_ids:
        Y_i = _reconstruct_grid_from_planes(
            A_out_list[i], S_out_list[i], colors_order, R_out, C_out
        )
        Y_list.append(Y_i)

    # Step 2: Detect global 2D period across all trainings
    p_r, p_c, per_training_periods_list = _aggregate_periods(
        Y_list, included_train_ids, R_out, C_out, colors_order
    )

    # If no global period → silent layer
    if p_r is None and p_c is None:
        A_lat = {c: [0] * R_out for c in colors_order}
        S_lat = [0] * R_out
        receipt = LatticeReceipt(
            included_train_ids=included_train_ids,
            p_r=None,
            p_c=None,
            per_training_periods=per_training_periods_list,
            global_p_r=None,
            global_p_c=None,
            global_validated=False,
            agreeing_classes=[],
            disagreeing_classes=[],
            residue_scope_bits=0,
            A_lat_hash=_hash_planes(A_lat, R_out, C_out, colors_order),
            S_lat_hash=_hash_scope(S_lat, C_out),
        )
        return A_lat, S_lat, receipt

    # Step 3: Emit residue-class admits where all trainings agree
    A_lat, S_lat, agreeing_classes, disagreeing_classes = _emit_residue_admits(
        Y_list, S_out_list, included_train_ids, p_r, p_c,
        colors_order, R_out, C_out
    )

    # Step 4: Build receipt
    residue_scope_bits = _popcount_scope(S_lat)
    A_lat_hash = _hash_planes(A_lat, R_out, C_out, colors_order)
    S_lat_hash = _hash_scope(S_lat, C_out)
    global_validated = len(agreeing_classes) > 0

    receipt = LatticeReceipt(
        included_train_ids=included_train_ids,
        p_r=p_r,
        p_c=p_c,
        per_training_periods=per_training_periods_list,
        global_p_r=p_r,
        global_p_c=p_c,
        global_validated=global_validated,
        agreeing_classes=agreeing_classes,
        disagreeing_classes=disagreeing_classes,
        residue_scope_bits=residue_scope_bits,
        A_lat_hash=A_lat_hash,
        S_lat_hash=S_lat_hash,
    )

    # Add debug arrays if requested and residue scope is non-empty
    if debug_arrays and residue_scope_bits > 0:
        from ..core.bytesio import serialize_planes_be_row_major, serialize_scope_be_row_major
        receipt["A_lat_planes_bytes"] = serialize_planes_be_row_major(A_lat, R_out, C_out, colors_order).hex()
        receipt["S_lat_bytes"] = serialize_scope_be_row_major(S_lat, R_out, C_out).hex()

    return A_lat, S_lat, receipt


# ============================================================================
# Step 0: Grid Reconstruction
# ============================================================================


def _reconstruct_grid_from_planes(
    A_i: Dict[int, List[int]],
    S_i: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> List[List[int]]:
    """
    Reconstruct integer grid from singleton planes.

    For each pixel (r,c) where S_i[r] has bit c set,
    find the unique color u where A_i[u][r] has bit c set.

    Spec: WO-09 v1.6 section 0
    """
    Y_i = [[0] * C_out for _ in range(R_out)]

    for r in range(R_out):
        row_scope = S_i[r] if r < len(S_i) else 0
        if row_scope == 0:
            continue

        for c in range(C_out):
            bit = 1 << c
            if not (row_scope & bit):
                continue

            # Find the unique color (singleton from WO-08)
            u = None
            for color in colors_order:
                plane = A_i.get(color)
                if not plane:
                    continue
                row_mask = plane[r] if r < len(plane) else 0
                if row_mask & bit:
                    u = color
                    break  # WO-08 guarantees singleton

            # Defensive: if no color found despite scope, leave as 0 (silent)
            if u is not None:
                Y_i[r][c] = u

    return Y_i


# ============================================================================
# Step 1: Period Detection and Aggregation
# ============================================================================


def _aggregate_periods(
    Y_list: List[List[List[int]]],
    included_train_ids: List[int],
    R_out: int,
    C_out: int,
    colors_order: List[int]
) -> Tuple[Optional[int], Optional[int], List[Dict]]:
    """
    Aggregate global 2D period across all trainings.

    Run WO-02 period detection on each training grid using the GLOBAL colors_order.
    If all trainings have the same proper period (≥2) on an axis → global period.
    If any mismatch or all None → None.

    Args:
        Y_list: Per-training integer grids
        included_train_ids: Training IDs (for receipt)
        R_out, C_out: Canvas dimensions
        colors_order: GLOBAL color order (must be consistent across all trainings)

    Returns:
        (p_r, p_c, per_training_periods_list)

    Spec: WO-09 v1.6 section 1

    CRITICAL: Use the same colors_order for all trainings to ensure
    consistent K-tuple interpretation in period detection.
    """
    per_training_periods = []
    per_training_periods_receipt = []

    for idx, Y_i in enumerate(Y_list):
        # Pack grid to planes using GLOBAL colors_order
        # This ensures consistent K-tuple interpretation across trainings
        planes_i = pack_grid_to_planes(Y_i, R_out, C_out, colors_order)

        # Run WO-02 period detection (proper periods ≥2 or None)
        p_r_i, p_c_i, _ = period_2d_planes(planes_i, R_out, C_out, colors_order)
        per_training_periods.append((p_r_i, p_c_i))

        # Build receipt entry
        per_training_periods_receipt.append({
            "train_id": included_train_ids[idx],
            "p_r": p_r_i,
            "p_c": p_c_i
        })

    # Aggregate row periods
    row_periods = [p[0] for p in per_training_periods]
    p_r = _aggregate_axis_period(row_periods)

    # Aggregate col periods
    col_periods = [p[1] for p in per_training_periods]
    p_c = _aggregate_axis_period(col_periods)

    return p_r, p_c, per_training_periods_receipt


def _aggregate_axis_period(periods: List[Optional[int]]) -> Optional[int]:
    """
    Aggregate periods on a single axis.

    If all are None → None.
    If all non-None are equal → that period.
    If any mismatch → None.
    """
    non_none = [p for p in periods if p is not None]

    if not non_none:
        # All None
        return None

    if len(set(non_none)) > 1:
        # Mismatch
        return None

    # All agree (or only one non-None)
    return non_none[0]


# ============================================================================
# Step 2: Residue Admits Emission
# ============================================================================


def _emit_residue_admits(
    Y_list: List[List[List[int]]],
    S_out_list: List[List[int]],
    included_train_ids: List[int],
    p_r: Optional[int],
    p_c: Optional[int],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> Tuple[Dict[int, List[int]], List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Emit residue-class admits where all trainings agree.

    For each residue class (i,j):
      - Build mask R_{i,j} of pixels in that residue
      - Check if all trainings agree on color at ALL pixels in residue
      - If yes → emit singleton admit and set scope
      - If no → disagreeing class (silent)

    Spec: WO-09 v1.6 section 2
    """
    # Initialize outputs
    A_lat = {c: [0] * R_out for c in colors_order}
    S_lat = [0] * R_out
    agreeing_classes = []
    disagreeing_classes = []

    # Determine residue ranges
    # If p_r is None → treat as single residue i=0 covering all rows
    # If p_c is None → treat as single residue j=0 covering all cols
    residue_i_range = range(p_r) if p_r is not None else range(1)  # [0..p_r-1] or [0]
    residue_j_range = range(p_c) if p_c is not None else range(1)  # [0..p_c-1] or [0]

    # Iterate residues in row-major order
    for i in residue_i_range:
        for j in residue_j_range:
            # Build residue mask R_{i,j}
            R_ij = _build_residue_mask(i, j, p_r, p_c, R_out, C_out)

            # Check agreement across trainings
            agreed, agreed_color = _check_residue_agreement(
                Y_list, S_out_list, included_train_ids, R_ij, R_out, C_out
            )

            if agreed:
                # Agreeing class: emit singleton admit on pixels where all trainings define
                agreeing_classes.append((i, j))

                # Set bits on residue pixels where all trainings have scope
                for r in range(R_out):
                    residue_row = R_ij[r]
                    if residue_row == 0:
                        continue

                    # For each pixel in residue, check if ALL trainings have scope
                    for c in range(C_out):
                        bit = 1 << c
                        if not (residue_row & bit):
                            continue

                        # Check if all trainings define this pixel
                        all_define = True
                        for idx, train_id in enumerate(included_train_ids):
                            S_i = S_out_list[train_id]
                            scope_row = S_i[r] if r < len(S_i) else 0
                            if not (scope_row & bit):
                                all_define = False
                                break

                        if all_define:
                            # Set scope and admit for agreed color
                            S_lat[r] |= bit
                            A_lat[agreed_color][r] |= bit
            else:
                # Disagreeing class: silent (no scope, no admits)
                disagreeing_classes.append((i, j))

    return A_lat, S_lat, agreeing_classes, disagreeing_classes


def _build_residue_mask(
    i: int,
    j: int,
    p_r: Optional[int],
    p_c: Optional[int],
    R_out: int,
    C_out: int
) -> List[int]:
    """
    Build bit mask for residue class (i,j).

    R_{i,j}[r] has bit c set iff:
      - r mod p_r == i (or p_r is None → all rows match)
      - c mod p_c == j (or p_c is None → all cols match)

    Spec: WO-09 v1.6 section 2
    """
    R_ij = [0] * R_out

    for r in range(R_out):
        # Check row residue
        if p_r is not None and r % p_r != i:
            continue  # Row doesn't match residue class

        # Build column mask for this row
        col_mask = 0
        for c in range(C_out):
            # Check col residue
            if p_c is not None and c % p_c != j:
                continue  # Col doesn't match residue class

            # This pixel is in residue (i,j)
            col_mask |= (1 << c)

        R_ij[r] = col_mask

    return R_ij


def _check_residue_agreement(
    Y_list: List[List[List[int]]],
    S_out_list: List[List[int]],
    included_train_ids: List[int],
    R_ij: List[int],
    R_out: int,
    C_out: int
) -> Tuple[bool, Optional[int]]:
    """
    Check if all trainings agree on color for residue R_{i,j}.

    Returns:
        (agreed, color): True and color u if all trainings agree on u
                        at ALL pixels in residue where all define.
                        False, None if disagreement or no coverage.

    Spec: WO-09 v1.6 section 2
    """
    # Collect colors from all trainings at pixels in residue
    # Only consider pixels where ALL trainings define (have scope)
    agreed_color = None
    has_any_pixel = False

    for r in range(R_out):
        residue_row = R_ij[r]
        if residue_row == 0:
            continue

        for c in range(C_out):
            bit = 1 << c
            if not (residue_row & bit):
                continue

            # Check if ALL trainings define this pixel
            colors_at_pixel = []
            all_define = True

            for idx, train_id in enumerate(included_train_ids):
                S_i = S_out_list[train_id]
                scope_row = S_i[r] if r < len(S_i) else 0

                if not (scope_row & bit):
                    # Training doesn't define this pixel
                    all_define = False
                    break

                # Training defines this pixel → get its color
                Y_i = Y_list[idx]
                colors_at_pixel.append(Y_i[r][c])

            if not all_define:
                # Skip pixels where not all trainings define
                continue

            # All trainings define this pixel
            has_any_pixel = True

            # Check if all colors are equal
            if len(set(colors_at_pixel)) > 1:
                # Disagreement!
                return False, None

            # All agree on this pixel
            pixel_color = colors_at_pixel[0]

            # Check consistency across residue
            if agreed_color is None:
                agreed_color = pixel_color
            elif agreed_color != pixel_color:
                # Different color in residue → disagreement
                return False, None

    # If no pixel where all trainings define → disagreeing
    if not has_any_pixel:
        return False, None

    # All pixels agree on same color
    return True, agreed_color


# ============================================================================
# Helpers
# ============================================================================


def _popcount_scope(S: List[int]) -> int:
    """Count total bits set in scope mask."""
    return sum(bin(row).count("1") for row in S)


def _hash_planes(
    planes: Dict[int, List[int]], R: int, C: int, colors_order: List[int]
) -> str:
    """Hash color planes in BE row-major order."""
    result = b""
    for c in colors_order:
        plane = planes.get(c, [0] * R)
        for row_mask in plane:
            result += row_mask.to_bytes((C + 7) // 8, "big")
    return blake3_hash(result)


def _hash_scope(S: List[int], C: int) -> str:
    """Hash scope mask in BE row-major order."""
    result = b""
    for row_mask in S:
        result += row_mask.to_bytes((C + 7) // 8, "big")
    return blake3_hash(result)
