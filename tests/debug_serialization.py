"""
Debug script to investigate Grid ↔ Planes serialization mismatch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import (
    order_colors,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
)


def debug_serialization():
    # Simple 2x3 grid for easy manual verification
    grid = [
        [0, 1, 0],
        [1, 0, 1],
    ]

    H, W = 2, 3
    colors_order = [0, 1]  # already sorted

    print("=" * 70)
    print("GRID:")
    for row in grid:
        print(f"  {row}")
    print()

    # Serialize as grid
    grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)

    # Manually construct planes
    # For color 0: row 0 = [1,0,1] → mask, row 1 = [0,1,0] → mask
    # For color 1: row 0 = [0,1,0] → mask, row 1 = [1,0,1] → mask

    planes = {}
    for c in colors_order:
        plane_rows = []
        for r in range(H):
            row_mask = 0
            for col in range(W):
                if grid[r][col] == c:
                    row_mask |= (1 << col)  # bit 0 = col 0, bit 1 = col 1, etc.
            plane_rows.append(row_mask)
            print(f"  Color {c}, Row {r}: mask = {row_mask:03b} (0b{row_mask:b})")
        planes[c] = plane_rows

    print()

    # Serialize as planes
    planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)

    # Compare
    print(f"Grid serialization:   {len(grid_bytes)} bytes")
    print(f"Planes serialization: {len(planes_bytes)} bytes")
    print()

    # Hex dump both
    print("Grid bytes (hex):")
    print(f"  {grid_bytes.hex()}")
    print()

    print("Planes bytes (hex):")
    print(f"  {planes_bytes.hex()}")
    print()

    # Detailed breakdown
    print("Grid bytes breakdown:")
    for i, b in enumerate(grid_bytes):
        print(f"  [{i:2d}] 0x{b:02x} = {b:3d} = 0b{b:08b}  {_interpret_byte(i, grid_bytes)}")

    print()

    print("Planes bytes breakdown:")
    for i, b in enumerate(planes_bytes):
        print(f"  [{i:2d}] 0x{b:02x} = {b:3d} = 0b{b:08b}  {_interpret_byte(i, planes_bytes)}")

    print()

    # Find first difference
    for i in range(min(len(grid_bytes), len(planes_bytes))):
        if grid_bytes[i] != planes_bytes[i]:
            print(f"❌ FIRST DIFFERENCE at byte {i}:")
            print(f"  Grid:   0x{grid_bytes[i]:02x} = 0b{grid_bytes[i]:08b}")
            print(f"  Planes: 0x{planes_bytes[i]:02x} = 0b{planes_bytes[i]:08b}")
            break
    else:
        if len(grid_bytes) == len(planes_bytes):
            print("✅ Bytes are IDENTICAL")
        else:
            print(f"❌ Lengths differ: Grid {len(grid_bytes)}, Planes {len(planes_bytes)}")


def _interpret_byte(i, data):
    """Interpret byte position in serialization format."""
    if i < 4:
        return f"tag[{i}] = {chr(data[i]) if 32 <= data[i] < 127 else '?'}"
    elif i == 4:
        return "H high byte"
    elif i == 5:
        return "H low byte"
    elif i == 6:
        return "W high byte"
    elif i == 7:
        return "W low byte"
    elif i == 8:
        return "K (num colors)"
    elif 9 <= i < 9 + data[8]:
        return f"color[{i-9}]"
    else:
        # Payload
        return "payload mask"


if __name__ == "__main__":
    debug_serialization()
