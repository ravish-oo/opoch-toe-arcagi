"""
WO-01 Component: Kernel Ops (SHIFT, POSE, BITWISE)

Pure bit-plane operations: translation, D4 transforms, Boolean algebra.

Spec: WO-01 sections 3-5.
"""


# ============================================================================
# SHIFT (zero-fill; no wrap)
# ============================================================================

def shift_plane(plane: list[int], dy: int, dx: int, H: int, W: int) -> list[int]:
    """
    Logical translate by (dy, dx). Positive dy moves bits DOWN (to larger r).
    Positive dx moves bits RIGHT (to larger c). Off-canvas bits are dropped.

    Args:
        plane: List of H row masks.
        dy: Vertical shift (positive = down, negative = up).
        dx: Horizontal shift (positive = right, negative = left).
        H: Height (number of rows).
        W: Width (number of columns).

    Returns:
        list[int]: Shifted plane (length H).

    Spec:
        WO-01 section 3: SHIFT.

        Vertical:
          if dy >= 0: [0]*dy + plane[0:H-dy]
          else: plane[-dy:H] + [0]*(-dy)

        Horizontal per row mask m:
          if dx >= 0: (m << dx) & ((1<<W)-1)
          else: (m >> (-dx))

    Invariant:
        Zero-fill only; no wrap.

    Edge case:
        If H=0, returns [] (empty plane).
    """
    # Handle empty plane
    if H == 0:
        return []

    # Vertical shift
    if dy >= 0:
        new_rows = [0] * min(dy, H) + plane[0:max(0, H-dy)]
    else:
        new_rows = plane[max(0, -dy):H] + [0] * min(-dy, H)

    # Horizontal shift per row
    if W == 0:
        # No columns, return list of zeros
        return [0] * len(new_rows)

    mask_all_cols = (1 << W) - 1
    result = []
    for m in new_rows:
        if dx >= 0:
            shifted = (m << dx) & mask_all_cols
        else:
            shifted = (m >> (-dx))
        result.append(shifted)

    return result


# ============================================================================
# POSE (8 exact D4 transforms; rectangular-aware)
# ============================================================================

# Frozen pose IDs (order from param_registry)
_POSE_IDS = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

# Inverse mapping (computed from coordinate formulas)
# NOTE: FXR90 and FXR270 are self-inverse (verified algebraically)
_POSE_INVERSE = {
    "I": "I",
    "R90": "R270",
    "R180": "R180",
    "R270": "R90",
    "FX": "FX",
    "FXR90": "FXR90",     # self-inverse (corrected)
    "FXR180": "FXR180",
    "FXR270": "FXR270"    # self-inverse (corrected)
}


def pose_plane(plane: list[int], pid: str, H: int, W: int) -> tuple[list[int], int, int]:
    """
    Apply one of 8 D4 transforms by coordinate remap.

    Args:
        plane: List of H row masks.
        pid: Pose ID ∈ {"I","R90","R180","R270","FX","FXR90","FXR180","FXR270"}.
        H: Source height.
        W: Source width.

    Returns:
        tuple[list[int], int, int]: (plane', H', W')

    Spec:
        WO-01 section 4: POSE.

        Output shape:
          If pid ∈ {R90, R270, FXR90, FXR270}: (H', W') = (W, H) (swap)
          Else: (H', W') = (H, W)

        Exact pull mapping (dest ← src):
          I:      r=r',           c=c'
          R90:    r=H-1-c',       c=r'
          R180:   r=H-1-r',       c=W-1-c'
          R270:   r=c',           c=W-1-r'
          FX:     r=r',           c=W-1-c'
          FXR90:  r=H-1-c',       c=W-1-r'
          FXR180: r=H-1-r',       c=c'
          FXR270: r=c',           c=r'

    Raises:
        ValueError: If pid is invalid.

    Invariant:
        pose(pose(plane, pid), inv(pid)) == plane (bit-for-bit).
    """
    if pid not in _POSE_IDS:
        raise ValueError(f"Invalid pose ID: '{pid}'. Must be one of {_POSE_IDS}")

    # Determine output shape
    if pid in {"R90", "R270", "FXR90", "FXR270"}:
        H_out, W_out = W, H
    else:
        H_out, W_out = H, W

    # Build output plane
    plane_out = []
    for r_out in range(H_out):
        mask_out = 0
        for c_out in range(W_out):
            # Compute source (r, c) using pull mapping
            if pid == "I":
                r, c = r_out, c_out
            elif pid == "R90":
                r, c = H - 1 - c_out, r_out
            elif pid == "R180":
                r, c = H - 1 - r_out, W - 1 - c_out
            elif pid == "R270":
                r, c = c_out, W - 1 - r_out
            elif pid == "FX":
                r, c = r_out, W - 1 - c_out
            elif pid == "FXR90":
                r, c = H - 1 - c_out, W - 1 - r_out
            elif pid == "FXR180":
                r, c = H - 1 - r_out, c_out
            elif pid == "FXR270":
                r, c = c_out, r_out
            else:
                raise ValueError(f"Unhandled pose ID: '{pid}'")

            # If source is in bounds and bit is set, set output bit
            if 0 <= r < H and 0 <= c < W:
                if (plane[r] >> c) & 1:
                    mask_out |= (1 << c_out)

        plane_out.append(mask_out)

    return (plane_out, H_out, W_out)


def pose_inverse(pid: str) -> str:
    """
    Return the inverse pose ID.

    Args:
        pid: Pose ID.

    Returns:
        str: Inverse pose ID.

    Raises:
        ValueError: If pid is invalid.

    Spec:
        Frozen mapping from WO-01 section 4.
    """
    if pid not in _POSE_INVERSE:
        raise ValueError(f"Invalid pose ID: '{pid}'")
    return _POSE_INVERSE[pid]


# ============================================================================
# BITWISE (AND/OR/ANDN)
# ============================================================================

def plane_and(a: list[int], b: list[int], H: int, W: int) -> list[int]:
    """
    Bitwise AND of two planes.

    Args:
        a: First plane (H row masks).
        b: Second plane (H row masks).
        H: Height.
        W: Width.

    Returns:
        list[int]: Result plane (a & b per row).

    Raises:
        ValueError: If row counts differ or bits outside [0..W-1].

    Spec:
        WO-01 section 5: BITWISE.
    """
    if len(a) != H or len(b) != H:
        raise ValueError(f"Row count mismatch: a={len(a)}, b={len(b)}, expected {H}")

    mask_all_cols = (1 << W) - 1
    result = []
    for i in range(H):
        # Check no bits outside [0..W-1]
        if (a[i] >> W) != 0 or (b[i] >> W) != 0:
            raise ValueError(
                f"Row {i} has bits outside [0..{W-1}]: a={a[i]:b}, b={b[i]:b}"
            )
        result.append(a[i] & b[i])

    return result


def plane_or(a: list[int], b: list[int], H: int, W: int) -> list[int]:
    """
    Bitwise OR of two planes.

    Args:
        a: First plane (H row masks).
        b: Second plane (H row masks).
        H: Height.
        W: Width.

    Returns:
        list[int]: Result plane (a | b per row).

    Raises:
        ValueError: If row counts differ or bits outside [0..W-1].

    Spec:
        WO-01 section 5: BITWISE.
    """
    if len(a) != H or len(b) != H:
        raise ValueError(f"Row count mismatch: a={len(a)}, b={len(b)}, expected {H}")

    mask_all_cols = (1 << W) - 1
    result = []
    for i in range(H):
        # Check no bits outside [0..W-1]
        if (a[i] >> W) != 0 or (b[i] >> W) != 0:
            raise ValueError(
                f"Row {i} has bits outside [0..{W-1}]: a={a[i]:b}, b={b[i]:b}"
            )
        result.append(a[i] | b[i])

    return result


def plane_andn(a: list[int], notmask: list[int], H: int, W: int) -> list[int]:
    """
    Bitwise AND-NOT: a & ~notmask.

    Args:
        a: First plane (H row masks).
        notmask: Plane to negate then AND (H row masks).
        H: Height.
        W: Width.

    Returns:
        list[int]: Result plane (a & ~notmask per row).

    Raises:
        ValueError: If row counts differ or bits outside [0..W-1].

    Spec:
        WO-01 section 5: BITWISE.
    """
    if len(a) != H or len(notmask) != H:
        raise ValueError(
            f"Row count mismatch: a={len(a)}, notmask={len(notmask)}, expected {H}"
        )

    mask_all_cols = (1 << W) - 1
    result = []
    for i in range(H):
        # Check no bits outside [0..W-1]
        if (a[i] >> W) != 0 or (notmask[i] >> W) != 0:
            raise ValueError(
                f"Row {i} has bits outside [0..{W-1}]: a={a[i]:b}, notmask={notmask[i]:b}"
            )
        result.append(a[i] & ~notmask[i] & mask_all_cols)

    return result
