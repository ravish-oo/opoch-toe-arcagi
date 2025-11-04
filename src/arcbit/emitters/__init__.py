"""
ARC-AGI Emitters

Witness learning and admit generation.
"""

from .witness_learn import learn_witness, Piece, WitnessResult
from .witness_emit import emit_witness
from .output_transport import emit_output_transport, TransportReceipt
from .unanimity import emit_unity, emit_unanimity, UnanimityReceipt

__all__ = [
    "learn_witness",
    "Piece",
    "WitnessResult",
    "emit_witness",
    "emit_output_transport",
    "TransportReceipt",
    "emit_unity",
    "emit_unanimity",
    "UnanimityReceipt",
]
