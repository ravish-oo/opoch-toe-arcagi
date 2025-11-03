"""
WO-00 Component: BLAKE3 Hashing

Deterministic hash function for all receipts and grid serialization.

No seeding, no randomness, no timestamps.
Spec: v1.5 addendum A.3.
"""

import blake3


def blake3_hash(data: bytes) -> str:
    """
    Return hex-encoded BLAKE3 digest of the byte stream.

    Args:
        data: Raw bytes to hash.

    Returns:
        str: Hexadecimal digest (64 characters for BLAKE3-256).

    Spec:
        - No seeding or personalization.
        - Output is always lowercase hex.
        - Deterministic: same bytes → same hash, always.

    Example:
        >>> blake3_hash(b"test")
        '4878ca0425c739fa427f7eda20fe845f6b2e46ba5fe2a14df5b1e32f50603215'
    """
    hasher = blake3.blake3()
    hasher.update(data)
    return hasher.hexdigest()
