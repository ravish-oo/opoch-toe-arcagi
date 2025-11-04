"""
ARC-AGI Emitters

Witness learning and admit generation.
"""

from .witness_learn import learn_witness, Piece, WitnessResult
from .witness_emit import emit_witness

__all__ = [
    "learn_witness",
    "Piece",
    "WitnessResult",
    "emit_witness",
]
