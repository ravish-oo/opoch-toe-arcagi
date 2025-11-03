"""
Core foundation: receipts, hashing, serialization, parameter registry.

WO-00 implementation - frozen constants and deterministic byte-level I/O.
"""

from .registry import param_registry, RegistryError
from .hashing import blake3_hash
from .bytesio import (
    order_colors,
    serialize_grid_be_row_major,
    serialize_planes_be_row_major,
    SerializationError
)
from .receipts import (
    Receipts,
    assert_double_run_equal,
    ReceiptError,
    DeterminismError
)

__all__ = [
    # Registry
    "param_registry",
    "RegistryError",

    # Hashing
    "blake3_hash",

    # Serialization
    "order_colors",
    "serialize_grid_be_row_major",
    "serialize_planes_be_row_major",
    "SerializationError",

    # Receipts
    "Receipts",
    "assert_double_run_equal",
    "ReceiptError",
    "DeterminismError",
]
