"""
ARC-AGI Emitters

Witness learning and admit generation.
"""

from .witness_learn import learn_witness, Piece, WitnessResult

__all__ = [
    "learn_witness",
    "Piece",
    "WitnessResult",
]
