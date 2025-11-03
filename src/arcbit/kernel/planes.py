"""
WO-01 Component: Bit-Planes (PACK/UNPACK)

Grid ↔ per-color bit-planes with mutual exclusivity.

Plane representation:
  - list[int] of length H
  - Each entry is a Python int with W least-significant bits
  - Bit j (0-indexed) corresponds to column j
  - bit j == 1 ⟺ cell (r, j) has that color

Spec: WO-01 sections 1-2.
"""


def order_colors(C: set[int]) -> list[int]:
    """
    Return ascending list of colors; 0 must be present.

    Args:
        C: Set of integer colors (must include 0).

    Returns:
        list[int]: Colors in ascending order.

    Raises:
        ValueError: If 0 is not in C.

    Spec:
        Matches WO-00 order_colors but defined here for kernel independence.
    """
    if 0 not in C:
        raise ValueError("Color 0 must be present in universe.")
    return sorted(C)


def pack_grid_to_planes(
    G: list[list[int]],
    H: int,
    W: int,
    colors_order: list[int]
) -> dict[int, list[int]]:
    """
    For each color c in colors_order, build a plane with row masks.

    Exactly one color bit is set per cell (mutual exclusivity enforced).

    Args:
        G: Grid as list of H rows, each row a list of W integers.
        H: Height (must match len(G)).
        W: Width (must match len(G[0]) for all rows).
        colors_order: Ordered list of colors (ascending integers).

    Returns:
        dict[int, list[int]]: Mapping color → plane (list of H row masks).

    Raises:
        ValueError: If H/W mismatch, or grid cell color not in colors_order.

    Spec:
        WO-01 section 2: PACK.
        Bit j (0-indexed) = column j.

    Invariant:
        Exactly one color per cell.
    """
    # Validate dimensions
    if len(G) != H:
        raise ValueError(f"Grid height mismatch: expected {H}, got {len(G)}")
    if H > 0:
        if any(len(row) != W for row in G):
            raise ValueError(f"Grid width mismatch: expected {W}")

    # Validate all colors are in colors_order
    color_set = set(colors_order)
    for r in range(H):
        for c in range(W):
            if G[r][c] not in color_set:
                raise ValueError(
                    f"Color {G[r][c]} at ({r},{c}) not in colors_order"
                )

    # Build planes
    planes = {}
    for color in colors_order:
        plane = []
        for r in range(H):
            mask = 0
            for c in range(W):
                if G[r][c] == color:
                    mask |= (1 << c)  # bit c = column c
            plane.append(mask)
        planes[color] = plane

    return planes


def unpack_planes_to_grid(
    planes: dict[int, list[int]],
    H: int,
    W: int,
    colors_order: list[int]
) -> list[list[int]]:
    """
    Reconstruct color grid from per-color planes.

    For each (r, c), exactly one color must have its bit set.

    Args:
        planes: Dict mapping color → list of H row masks.
        H: Height (number of rows).
        W: Width (number of columns).
        colors_order: Ordered list of colors.

    Returns:
        list[list[int]]: Reconstructed grid (H rows × W columns).

    Raises:
        ValueError: If multiple colors are ON at a cell, or no color is ON.
        ValueError: If any plane has wrong row count or bits outside 0..W-1.

    Spec:
        WO-01 section 2: UNPACK.
        Mutual exclusivity enforced (exactly one color per cell).

    Invariant:
        unpack(pack(G)) == G (round-trip identity).
    """
    # Validate planes structure
    for color in colors_order:
        if color not in planes:
            raise ValueError(f"Color {color} missing from planes dict")
        if len(planes[color]) != H:
            raise ValueError(
                f"Plane for color {color} has {len(planes[color])} rows, expected {H}"
            )
        # Check no bits outside [0..W-1]
        for r, mask in enumerate(planes[color]):
            if mask >> W != 0:
                raise ValueError(
                    f"Plane for color {color} row {r} has bits outside [0..{W-1}]: {mask:b}"
                )

    # Reconstruct grid
    G = []
    for r in range(H):
        row = []
        for c in range(W):
            # Find which color has bit set at (r, c)
            active_colors = []
            for color in colors_order:
                if (planes[color][r] >> c) & 1:
                    active_colors.append(color)

            # Enforce mutual exclusivity
            if len(active_colors) == 0:
                raise ValueError(
                    f"No color is ON at cell ({r},{c}). Planes are not exclusive."
                )
            if len(active_colors) > 1:
                raise ValueError(
                    f"Multiple colors {active_colors} are ON at cell ({r},{c}). "
                    f"Planes are not exclusive."
                )

            row.append(active_colors[0])
        G.append(row)

    return G
