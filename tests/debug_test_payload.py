"""Debug the test itself to see why it's failing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import (
    order_colors,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
    blake3_hash,
)


grid = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]

H, W = 3, 3
colors_set = {0, 1}
colors_order = order_colors(colors_set)

print(f"Colors: {colors_order}")
print(f"H={H}, W={W}")
print()

# Serialize as grid
grid_bytes = serialize_grid_be_row_major(grid, H, W, colors_order)

# Convert to planes
planes = {}
for c in colors_order:
    plane_rows = []
    for r in range(H):
        row_mask = 0
        for col in range(W):
            if grid[r][col] == c:
                row_mask |= (1 << col)
        plane_rows.append(row_mask)
    planes[c] = plane_rows

# Serialize as planes
planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)

# Header calculation from test
header_size = 4 + 2 + 2 + 1 + len(colors_order)
print(f"Header size: {header_size}")
print(f"Grid total: {len(grid_bytes)} bytes")
print(f"Planes total: {len(planes_bytes)} bytes")
print()

grid_payload = grid_bytes[header_size:]
planes_payload = planes_bytes[header_size:]

print(f"Grid payload length: {len(grid_payload)}")
print(f"Planes payload length: {len(planes_payload)}")
print()

print(f"Grid payload hex:   {grid_payload.hex()}")
print(f"Planes payload hex: {planes_payload.hex()}")
print()

print(f"Grid payload hash:   {blake3_hash(grid_payload)}")
print(f"Planes payload hash: {blake3_hash(planes_payload)}")
print()

print(f"Payloads match: {grid_payload == planes_payload}")
