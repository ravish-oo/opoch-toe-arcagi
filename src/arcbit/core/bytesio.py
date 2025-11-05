"""
WO-00 Component: Byte Serialization (Big-Endian, Row-Major)

Stable, deterministic byte serialization for grids and planes.

Bit mapping (frozen):
  - Within each byte: bit 7 → col 0, bit 6 → col 1, ..., bit 0 → col 7
  - Row-major order: rows serialized sequentially
  - Big-endian for multi-byte integers (dims, color count)

No timestamps, no padding beyond ceil(W/8) per row.
Spec: WO-00, sections 2-4.
"""

import math


def order_colors(C: set[int]) -> list[int]:
    """
    Return colors sorted ascending; 0 must be present and is included.

    Args:
        C: Set of integer colors (0 must be present).

    Returns:
        list[int]: Colors in ascending order.

    Raises:
        SerializationError: If 0 is not in C.

    Spec:
        color_ordering = "ascending-int" (frozen in param_registry).
    """
    if 0 not in C:
        raise SerializationError("Color 0 must be present in universe.")
    return sorted(C)


def serialize_grid_be_row_major(
    G: list[list[int]],
    H: int,
    W: int,
    colors_order: list[int]
) -> bytes:
    """
    Encode a grid as a deterministic byte stream for hashing.

    Format (exact):
      - 4 ASCII bytes tag: b"GRD1"
      - 2 bytes H (uint16, big-endian)
      - 2 bytes W (uint16, big-endian)
      - 1 byte K (number of colors)
      - K bytes: color ids (uint8 each, in ascending order)
      - Payload: for each row r in 0..H-1, for each color c in colors_order:
          emit ceil(W/8) bytes with bit mapping:
            bit 7 → col 0, bit 6 → col 1, ..., bit 0 → col 7
            next byte continues with col 8 at bit 7, etc.
          A bit is 1 iff G[r][col] == c.

    Args:
        G: Grid as list of H rows, each row a list of W integers.
        H: Height (must match len(G)).
        W: Width (must match len(G[0]) for all rows).
        colors_order: Ordered list of colors (ascending integers).

    Returns:
        bytes: Deterministic serialization.

    Raises:
        SerializationError: If dimensions mismatch or color not in colors_order.

    Spec:
        WO-00 section 3: serialize_grid_be_row_major.
    """
    # Validate dimensions
    if len(G) != H:
        raise SerializationError(f"Grid height mismatch: expected {H}, got {len(G)}")
    if H > 0 and any(len(row) != W for row in G):
        raise SerializationError(f"Grid width mismatch: expected {W}")

    # Validate all colors are in colors_order
    color_set = set(colors_order)
    for r in range(H):
        for c in range(W):
            if G[r][c] not in color_set:
                raise SerializationError(
                    f"Color {G[r][c]} at ({r},{c}) not in colors_order"
                )

    K = len(colors_order)
    if K > 255:
        raise SerializationError(f"Too many colors: {K} > 255")
    if H > 65535 or W > 65535:
        raise SerializationError(f"Dimensions too large: H={H}, W={W}")

    # Build byte stream
    stream = bytearray()

    # Tag (4 ASCII bytes)
    stream.extend(b"GRD1")

    # Dimensions (big-endian uint16)
    stream.extend(H.to_bytes(2, byteorder='big'))
    stream.extend(W.to_bytes(2, byteorder='big'))

    # Color count and ids
    stream.append(K)
    for color in colors_order:
        if color < 0 or color > 255:
            raise SerializationError(f"Color {color} out of uint8 range")
        stream.append(color)

    # Payload: per-row, per-color masks
    bytes_per_row = math.ceil(W / 8)

    for r in range(H):
        for color in colors_order:
            # Build bit mask for this (row, color)
            mask_bytes = bytearray(bytes_per_row)
            for col in range(W):
                if G[r][col] == color:
                    byte_idx = col // 8
                    bit_pos = 7 - (col % 8)  # bit 7 → col 0
                    mask_bytes[byte_idx] |= (1 << bit_pos)
            stream.extend(mask_bytes)

    return bytes(stream)


def serialize_planes_be_row_major(
    planes: dict[int, list[int]],
    H: int,
    W: int,
    colors_order: list[int]
) -> bytes:
    """
    Encode per-color planes as row-packed masks.

    Format (exact):
      - 4 ASCII bytes tag: b"PLN1"
      - 2 bytes H (uint16, big-endian)
      - 2 bytes W (uint16, big-endian)
      - 1 byte K (number of colors)
      - K bytes: color ids (uint8 each, in ascending order)
      - Payload: for each color c in colors_order, for each row r:
          emit ceil(W/8) bytes with bit mapping:
            bit 7 → col 0, bit 6 → col 1, ..., bit 0 → col 7
          A bit is 1 iff planes[c][r] has that bit set.

    Args:
        planes: Dict mapping color → list of H row masks (unsigned ints).
        H: Height (number of rows per plane).
        W: Width (number of columns per plane).
        colors_order: Ordered list of colors (ascending integers).

    Returns:
        bytes: Deterministic serialization.

    Raises:
        SerializationError: If dimensions mismatch or color missing.

    Spec:
        WO-00 section 4: serialize_planes_be_row_major.

    Notes:
        - planes[c] is a list of H unsigned integers, one per row.
        - Each integer encodes W bits (low W bits used).
        - This format is identical to serialize_grid_be_row_major's payload,
          just starting from plane representation instead of grid.
    """
    K = len(colors_order)
    if K > 255:
        raise SerializationError(f"Too many colors: {K} > 255")
    if H > 65535 or W > 65535:
        raise SerializationError(f"Dimensions too large: H={H}, W={W}")

    # Validate all colors present
    for color in colors_order:
        if color not in planes:
            raise SerializationError(f"Color {color} missing from planes dict")
        if len(planes[color]) != H:
            raise SerializationError(
                f"Plane for color {color} has {len(planes[color])} rows, expected {H}"
            )

    # Build byte stream
    stream = bytearray()

    # Tag (4 ASCII bytes)
    stream.extend(b"PLN1")

    # Dimensions (big-endian uint16)
    stream.extend(H.to_bytes(2, byteorder='big'))
    stream.extend(W.to_bytes(2, byteorder='big'))

    # Color count and ids
    stream.append(K)
    for color in colors_order:
        if color < 0 or color > 255:
            raise SerializationError(f"Color {color} out of uint8 range")
        stream.append(color)

    # Payload: per-row, per-color masks (same order as grid serialization)
    bytes_per_row = math.ceil(W / 8)

    for r in range(H):
        for color in colors_order:
            row_mask = planes[color][r]
            # Convert row mask (unsigned int) to bytes
            mask_bytes = bytearray(bytes_per_row)
            for col in range(W):
                if row_mask & (1 << col):
                    byte_idx = col // 8
                    bit_pos = 7 - (col % 8)  # bit 7 → col 0
                    mask_bytes[byte_idx] |= (1 << bit_pos)
            stream.extend(mask_bytes)

    return bytes(stream)


def serialize_scope_be_row_major(
    S: list[int],
    H: int,
    W: int
) -> bytes:
    """
    Encode scope (row masks) as deterministic byte stream.

    Format (exact):
      - 4 ASCII bytes tag: b"SCP1"
      - 2 bytes H (uint16, big-endian)
      - 2 bytes W (uint16, big-endian)
      - Payload: for each row r in 0..H-1:
          emit ceil(W/8) bytes with bit mapping:
            bit 7 → col 0, bit 6 → col 1, ..., bit 0 → col 7
          A bit is 1 iff S[r] has that bit set.

    Args:
        S: List of H row masks (unsigned ints).
        H: Height (number of rows).
        W: Width (number of columns per row).

    Returns:
        bytes: Deterministic serialization.

    Raises:
        SerializationError: If dimensions mismatch.

    Spec:
        Debug arrays WO: serialize_scope_be_row_major.
    """
    if len(S) != H:
        raise SerializationError(f"Scope length mismatch: expected {H}, got {len(S)}")
    if H > 65535 or W > 65535:
        raise SerializationError(f"Dimensions too large: H={H}, W={W}")

    # Build byte stream
    stream = bytearray()

    # Tag (4 ASCII bytes)
    stream.extend(b"SCP1")

    # Dimensions (big-endian uint16)
    stream.extend(H.to_bytes(2, byteorder='big'))
    stream.extend(W.to_bytes(2, byteorder='big'))

    # Payload: per-row masks
    bytes_per_row = math.ceil(W / 8)

    for r in range(H):
        row_mask = S[r]
        # Convert row mask (unsigned int) to bytes
        mask_bytes = bytearray(bytes_per_row)
        for col in range(W):
            if row_mask & (1 << col):
                byte_idx = col // 8
                bit_pos = 7 - (col % 8)  # bit 7 → col 0
                mask_bytes[byte_idx] |= (1 << bit_pos)
        stream.extend(mask_bytes)

    return bytes(stream)


class SerializationError(Exception):
    """Raised when serialization encounters invalid dimensions or colors."""
    pass
