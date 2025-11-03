"""
WO-00 Component: Parameter Registry

Frozen constants for deterministic solver operation.
All global parameters (pose order, AC-3 policy, engine priority, etc.)
are defined here with exact values from spec v1.5.

No randomness, no environment leakage, no optionals.
"""


def param_registry() -> dict:
    """
    Returns a frozen mapping of all global constants used by the solver.

    Keys and values are JSON-serializable primitives or lists/tuples.
    This registry is hashed into every section receipt to prove parametric consistency.

    Spec: v1.5 addendum, section A.3 and throughout.

    Returns:
        dict: Frozen parameter mapping with exact keys and values.

    Raises:
        RegistryError: If any required key is missing (internal consistency check).
    """
    registry = {
        # Core version binding
        "spec_version": "1.5",
        "endianness": "BE",  # big-endian bit packing within bytes

        # D4 group frozen order (8 elements)
        # Used for lex-min canonicalization tie-breaks
        "pose_order": ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"],

        # AC-3 arc consistency settings
        "ac3_neighbor": "4",  # 4-neighbor graph (no diagonals, no wrap)
        "ac3_queue_order": "row-major-pq-fifo",  # queue initialization and re-enqueue order

        # Engine selection priority (frozen)
        # When multiple engines fit training equally, choose by this order
        # T1 Witness excluded (always first bucket)
        # T2 Unanimity excluded (always third bucket)
        "engine_priority": ["T3", "T5", "T4", "T6", "T7", "T8", "T9", "T10", "T11"],

        # Color and domain defaults
        "bottom_color": 0,  # guaranteed in universe; fallback selection

        # Lattice/period defaults
        "period_phase_origin": [0, 0],  # phase fixed at (0,0) for period detection

        # Downscaling mode
        "strict_downscale": True,  # constant blocks only; no majority voting

        # Hashing
        "hash_algo": "BLAKE3",

        # Color ordering in all serialization
        "color_ordering": "ascending-int",

        # Byte frame tags for serialization (ASCII 4-byte tags)
        "byte_frame_tags": {
            "GRID": "GRD1",
            "PLANES": "PLN1"
        }
    }

    # Consistency check: ensure all required keys are present
    required_keys = {
        "spec_version", "endianness", "pose_order", "ac3_neighbor",
        "ac3_queue_order", "engine_priority", "bottom_color",
        "period_phase_origin", "strict_downscale", "hash_algo",
        "color_ordering", "byte_frame_tags"
    }

    actual_keys = set(registry.keys())
    if actual_keys != required_keys:
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        raise RegistryError(
            f"param_registry() key mismatch. Missing: {missing}, Extra: {extra}"
        )

    return registry


class RegistryError(Exception):
    """Raised when param_registry() has missing or unexpected keys."""
    pass
