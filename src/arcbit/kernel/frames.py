"""
WO-03: Frame Canonicalizer (D4 lex-min + anchor)

Canonical frame selection for grids using D4 symmetry breaking.

Components:
  - canonicalize: Select minimal D4 pose + anchor to origin
  - apply_pose_anchor: Apply frame transformation to bit-planes

Spec: Addendum v1.5 section B.1 (Frame Canonicalizer).
"""

from typing import Tuple, List, Optional, Dict
from .planes import pack_grid_to_planes, unpack_planes_to_grid, order_colors
from .ops import pose_plane, shift_plane
from ..core.bytesio import serialize_grid_be_row_major, serialize_planes_be_row_major
from ..core.hashing import blake3_hash


def canonicalize(
    G: List[List[int]]
) -> Tuple[str, Tuple[int, int], List[List[int]], Dict]:
    """
    Return canonical frame for grid G: (pose_id, anchor, G_canon, receipts).

    Algorithm:
      1. Enumerate all 8 D4 poses of G
      2. Serialize each pose using WO-00 BE row-major
      3. Choose pose with lexicographically minimal bytes
      4. Find first nonzero cell (row-major scan)
      5. Translate to origin (anchor at 0,0)
      6. Generate receipts (always)

    Args:
        G: Input grid (list[list[int]]).

    Returns:
        Tuple of (pose_id, anchor, G_canon, receipts):
          - pose_id: D4 transform ID (e.g., "R90")
          - anchor: Translation offset (r_offset, c_offset)
          - G_canon: Canonicalized grid (posed + anchored)
          - receipts: Receipts dict (always included)

    Spec:
        Addendum v1.5 section B.1: Frame Canonicalizer.

        Phase 1 (D4 lex-min):
          - Frozen pose order: ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]
          - Serialize each pose using serialize_grid_be_row_major
          - Tie-break: first in frozen order wins

        Phase 2 (Anchor to origin):
          - Row-major scan for first nonzero cell: (r_first, c_first)
          - Translate grid so (r_first, c_first) moves to (0, 0)
          - anchor = (r_first, c_first) in canonical frame coordinates

    Invariants:
        - Deterministic (no hashing, pure byte comparison)
        - Idempotent: canonicalize(canonicalize(G)[2]) = ("I", (0,0), same_grid)
        - No empty grids (undefined behavior if all cells are 0)

    Edge cases:
        - If G is all zeros: anchor = (0, 0) (no translation)
        - If G is 1x1: pose_id = "I" (no D4 ambiguity)

    Example:
        >>> G = [[0, 1], [1, 0]]  # 2x2 checkerboard
        >>> pid, anchor, G_canon, receipts = canonicalize(G)
        >>> # Returns D4 pose with lex-min serialization, anchored to origin, plus receipts
    """
    if not G or not G[0]:
        # Empty grid edge case
        receipts = {
            "frame.inputs": {"H": 0, "W": 0, "colors_order": [0], "nonzero_count": 0},
            "frame.pose": {"pose_id": "I", "pose_tie_count": 1},
            "frame.anchor": {"r": 0, "c": 0, "all_zero": True},
            "frame.bytes": {"hash_before": "", "hash_after": ""}
        }
        return ("I", (0, 0), G, receipts)

    H = len(G)
    W = len(G[0])

    # Extract color palette
    color_set = set()
    for row in G:
        for val in row:
            color_set.add(val)

    # Ensure 0 is in color set (required by order_colors)
    if 0 not in color_set:
        color_set.add(0)

    colors = order_colors(color_set)

    # Frozen pose order (WO-01)
    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]

    # Phase 1: Enumerate all 8 poses and find lex-min serialization
    min_bytes = None
    min_pid = None
    min_grid = None
    pose_tie_count = 0  # Track how many poses tie on lex-min

    for pid in pose_ids:
        # Apply pose to grid via bit-planes
        G_posed = _pose_grid_via_planes(G, H, W, colors, pid)
        H_posed = len(G_posed)
        W_posed = len(G_posed[0]) if G_posed else 0

        # Serialize using WO-00
        grid_bytes = serialize_grid_be_row_major(G_posed, H_posed, W_posed, colors)

        # Lexicographic comparison
        if min_bytes is None or grid_bytes < min_bytes:
            min_bytes = grid_bytes
            min_pid = pid
            min_grid = G_posed
            pose_tie_count = 1  # First pose with this minimum
        elif grid_bytes == min_bytes:
            pose_tie_count += 1  # Another pose ties

    # Phase 2: Anchor to origin
    assert min_grid is not None and min_pid is not None

    # Find first nonzero cell (row-major)
    r_first, c_first = _find_first_nonzero(min_grid)

    all_zero = (r_first is None)

    if all_zero:
        # All zeros: no translation needed
        anchor = (0, 0)
        G_canon = min_grid
    else:
        # Translate so (r_first, c_first) → (0, 0)
        anchor = (r_first, c_first)
        G_canon = _translate_grid(min_grid, -r_first, -c_first)

    # Generate receipts (always)
    # Count nonzero cells
    nonzero_count = sum(1 for row in G for val in row if val != 0)

    # Hash before canonicalization
    hash_before = blake3_hash(serialize_grid_be_row_major(G, H, W, colors))

    # Hash after canonicalization
    H_canon = len(G_canon)
    W_canon = len(G_canon[0]) if G_canon else 0
    color_set_canon = set()
    for row in G_canon:
        for val in row:
            color_set_canon.add(val)
    if 0 not in color_set_canon:
        color_set_canon.add(0)
    colors_canon = order_colors(color_set_canon)
    hash_after = blake3_hash(serialize_grid_be_row_major(G_canon, H_canon, W_canon, colors_canon))

    # Note: idempotence check NOT done here to avoid recursion
    # Test code should verify: canonicalize(G_canon)[0:3] == ("I", (0,0), G_canon)

    receipts = {
        "frame.inputs": {
            "H": H,
            "W": W,
            "colors_order": colors,
            "nonzero_count": nonzero_count
        },
        "frame.pose": {
            "pose_id": min_pid,
            "pose_tie_count": pose_tie_count
        },
        "frame.anchor": {
            "r": anchor[0],
            "c": anchor[1],
            "all_zero": all_zero
        },
        "frame.bytes": {
            "hash_before": hash_before,
            "hash_after": hash_after
        }
    }

    return (min_pid, anchor, G_canon, receipts)


def apply_pose_anchor(
    planes: dict[int, List[int]],
    pid: str,
    anchor: Tuple[int, int],
    H: int,
    W: int,
    colors_order: List[int]
) -> Tuple[dict[int, List[int]], int, int]:
    """
    Apply frame transformation (pose + anchor) to bit-planes.

    Args:
        planes: Dict mapping color → list of H row masks.
        pid: Pose ID (D4 transform).
        anchor: Translation offset (r_offset, c_offset).
        H: Source height.
        W: Source width.
        colors_order: Ordered list of colors.

    Returns:
        Tuple of (planes', H', W'):
          - planes': Transformed bit-planes
          - H': Output height
          - W': Output width

    Spec:
        WO-03: Frame transformation via WO-01 ops.

        Algorithm:
          1. Apply pose to each plane (using pose_plane from WO-01)
          2. Apply translation (using shift_plane from WO-01)
          3. Rebuild color 0 plane as complement of non-zero planes (preserves exclusivity)

        Invariants:
          - Exclusivity: union of all planes == full mask, pairwise overlaps == 0
          - This is equivalent to transforming the grid with zero-fill, then repacking
          - Color 0 will increase where translation creates empty cells (not a bug)

    Raises:
        ValueError: If exclusivity is violated after transformation.
    """
    r_offset, c_offset = anchor

    # Step 1: Apply pose to each plane
    planes_posed = {}
    H_posed, W_posed = H, W

    for color in colors_order:
        plane = planes[color]
        plane_posed, H_out, W_out = pose_plane(plane, pid, H, W)
        planes_posed[color] = plane_posed

        # All planes should have same output shape
        H_posed, W_posed = H_out, W_out

    # Step 2: Apply translation (shift by negative offset to move anchor to origin)
    planes_anchored = {}
    for color in colors_order:
        plane_posed = planes_posed[color]
        # Shift by (-r_offset, -c_offset) to move anchor to origin
        plane_anchored = shift_plane(plane_posed, -r_offset, -c_offset, H_posed, W_posed)
        planes_anchored[color] = plane_anchored

    # Step 3: Fix exclusivity - cells with no color should be set to background (0)
    # This is needed because shift_plane zero-fills, leaving some cells with no color
    if 0 in planes_anchored:
        # Build mask of all cells that have any color set
        all_colors_mask = [0] * H_posed
        for color in colors_order:
            if color != 0:
                for r in range(H_posed):
                    all_colors_mask[r] |= planes_anchored[color][r]

        # Set background (0) for cells with no color
        for r in range(H_posed):
            empty_cells = ((1 << W_posed) - 1) & ~all_colors_mask[r]
            planes_anchored[0][r] |= empty_cells

    # Step 4: Validate exclusivity (defensive check)
    # Verify: union == full_mask and pairwise overlaps == 0
    full_mask = (1 << W_posed) - 1 if W_posed > 0 else 0

    for r in range(H_posed):
        # Check union covers all cells
        union = 0
        for color in colors_order:
            union |= planes_anchored[color][r]

        if union != full_mask:
            raise ValueError(
                f"Exclusivity violation at row {r}: union {union:b} != full_mask {full_mask:b}. "
                f"Not all cells have a color assigned after frame transformation."
            )

        # Check pairwise overlaps are zero
        for i, color_i in enumerate(colors_order):
            for color_j in colors_order[i+1:]:
                overlap = planes_anchored[color_i][r] & planes_anchored[color_j][r]
                if overlap != 0:
                    raise ValueError(
                        f"Exclusivity violation at row {r}: colors {color_i} and {color_j} overlap "
                        f"with mask {overlap:b}. Multiple colors assigned to same cell."
                    )

    return (planes_anchored, H_posed, W_posed)


# ============================================================================
# Helper Functions
# ============================================================================

def _pose_grid_via_planes(
    G: List[List[int]],
    H: int,
    W: int,
    colors: List[int],
    pid: str
) -> List[List[int]]:
    """
    Apply D4 pose to grid via bit-plane operations.

    Args:
        G: Input grid.
        H: Height.
        W: Width.
        colors: Color palette (ordered).
        pid: Pose ID.

    Returns:
        list[list[int]]: Posed grid.

    Spec:
        WO-03 helper: Use WO-01 pack/pose/unpack pipeline.
    """
    # Pack to planes
    planes = pack_grid_to_planes(G, H, W, colors)

    # Apply pose to each plane
    planes_posed = {}
    H_posed, W_posed = H, W

    for color in colors:
        plane = planes[color]
        plane_posed, H_out, W_out = pose_plane(plane, pid, H, W)
        planes_posed[color] = plane_posed
        H_posed, W_posed = H_out, W_out

    # Unpack back to grid
    G_posed = unpack_planes_to_grid(planes_posed, H_posed, W_posed, colors)

    return G_posed


def _find_first_nonzero(G: List[List[int]]) -> Tuple[Optional[int], Optional[int]]:
    """
    Find first nonzero cell in row-major order.

    Args:
        G: Grid.

    Returns:
        Tuple of (r, c) or (None, None) if all zeros.

    Spec:
        WO-03 helper: Row-major scan (r=0..H-1, c=0..W-1).
    """
    H = len(G)
    if H == 0:
        return (None, None)

    W = len(G[0]) if G else 0

    for r in range(H):
        for c in range(W):
            if G[r][c] != 0:
                return (r, c)

    return (None, None)


def _translate_grid(G: List[List[int]], dy: int, dx: int) -> List[List[int]]:
    """
    Translate grid by (dy, dx) with zero-fill.

    Args:
        G: Input grid.
        dy: Vertical translation (negative = up).
        dx: Horizontal translation (negative = left).

    Returns:
        list[list[int]]: Translated grid (same shape as input).

    Spec:
        WO-03 helper: Crop/pad with zeros to maintain shape.

        For dy < 0 (shift up):
          - Output row r comes from input row r - dy
          - Top rows are from input, bottom rows are zeros

        For dx < 0 (shift left):
          - Output col c comes from input col c - dx
          - Left cols are from input, right cols are zeros
    """
    H = len(G)
    if H == 0:
        return G

    W = len(G[0]) if G else 0
    if W == 0:
        return G

    # Build translated grid
    G_trans = []
    for r_out in range(H):
        row_out = []
        for c_out in range(W):
            # Source coordinates
            r_src = r_out - dy
            c_src = c_out - dx

            # Check bounds
            if 0 <= r_src < H and 0 <= c_src < W:
                row_out.append(G[r_src][c_src])
            else:
                row_out.append(0)

        G_trans.append(row_out)

    return G_trans
