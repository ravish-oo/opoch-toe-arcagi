"""
WO-00 Component: Section Receipts & Double-Run Checker

Deterministic, ordered receipts for every WO section.
All receipts include param_registry_hash binding and section_hash for comparison.

No timestamps, no memory addresses, no environment leakage.
Spec: WO-00 sections 6-7.
"""

import json
from typing import Any, Callable

from .registry import param_registry
from .hashing import blake3_hash


class Receipts:
    """
    Section-scoped receipt builder.

    Each section (WO unit) creates a Receipts instance, logs key/value pairs,
    and produces a digest with:
      - section identifier
      - spec_version
      - param_registry_hash (proves parametric consistency)
      - payload (ordered key/value pairs)
      - section_hash (cryptographic commitment to all above)

    Forbidden in payload:
      - floats (spec disallows floats in logic)
      - timestamps or wall-clock
      - random values
      - object reprs or memory addresses

    Only allowed:
      - int, bool, str, None
      - list, tuple, dict (of allowed types)
    """

    def __init__(self, section: str):
        """
        Initialize receipt collector for a logical section.

        Args:
            section: ASCII identifier (e.g., "WO-06-witness").

        Spec:
            Section names should be human-readable and match WO numbering.
        """
        self.section = section
        self.payload = []  # list of (key, value) to preserve insertion order

    def put(self, key: str, value: Any) -> None:
        """
        Insert key/value pair into receipts.

        Args:
            key: Unique string key.
            value: JSON-serializable value (int/bool/str/list/dict/None only).

        Raises:
            ReceiptError: If key is duplicate or value is invalid type.

        Spec:
            Keys must be unique within a section.
            Values must be deterministic (no floats, no timestamps).
        """
        # Check for duplicate keys
        existing_keys = [k for k, _ in self.payload]
        if key in existing_keys:
            raise ReceiptError(f"Duplicate key in receipts: '{key}'")

        # Validate value type (recursively)
        _validate_receipt_value(value, key)

        self.payload.append((key, value))

    def digest(self) -> dict:
        """
        Returns the complete receipt digest with section_hash.

        Format:
          {
            "section": section,
            "spec_version": "1.5",
            "param_registry_hash": blake3_hash(stable_json(param_registry())),
            "payload": {key: value for key, value in self.payload},
            "section_hash": blake3_hash(stable_json({section, spec_version, param_registry_hash, payload}))
          }

        Returns:
            dict: Receipt digest ready for logging or comparison.

        Spec:
            stable_json = sorted keys, UTF-8, no whitespace variation.
            Double-run must produce identical section_hash.
        """
        # Get param registry hash (binds all frozen constants)
        registry = param_registry()
        registry_bytes = _stable_json_bytes(registry)
        registry_hash = blake3_hash(registry_bytes)

        # Build payload dict (preserves order from insertion)
        payload_dict = {k: v for k, v in self.payload}

        # Pre-digest structure (what gets hashed)
        pre_digest = {
            "section": self.section,
            "spec_version": "1.5",
            "param_registry_hash": registry_hash,
            "payload": payload_dict
        }

        # Compute section hash
        pre_digest_bytes = _stable_json_bytes(pre_digest)
        section_hash = blake3_hash(pre_digest_bytes)

        # Final digest includes the hash
        final_digest = {
            **pre_digest,
            "section_hash": section_hash
        }

        return final_digest


def assert_double_run_equal(build_section_callable: Callable[[], Receipts]) -> None:
    """
    Calls build_section_callable() twice and verifies identical section_hash.

    Args:
        build_section_callable: Function that builds and returns a Receipts instance.

    Raises:
        DeterminismError: If section_hash differs between runs.

    Spec:
        This is the mechanical proof of determinism.
        Same inputs → same bytes → same hash.

    Example:
        >>> def build():
        ...     r = Receipts("test")
        ...     r.put("value", 42)
        ...     return r
        >>> assert_double_run_equal(build)  # passes
    """
    # First run
    receipts_a = build_section_callable()
    digest_a = receipts_a.digest()
    hash_a = digest_a["section_hash"]

    # Second run
    receipts_b = build_section_callable()
    digest_b = receipts_b.digest()
    hash_b = digest_b["section_hash"]

    # Compare
    if hash_a != hash_b:
        # Find first differing key in payload
        payload_a = digest_a["payload"]
        payload_b = digest_b["payload"]

        differing_key = None
        for key in set(payload_a.keys()) | set(payload_b.keys()):
            val_a = payload_a.get(key, "<MISSING>")
            val_b = payload_b.get(key, "<MISSING>")
            if val_a != val_b:
                differing_key = key
                break

        raise DeterminismError(
            section=digest_a["section"],
            first_differing_key=differing_key,
            value_a=val_a if differing_key else None,
            value_b=val_b if differing_key else None,
            hash_a=hash_a,
            hash_b=hash_b
        )


def _stable_json_bytes(obj: Any) -> bytes:
    """
    Serialize object to deterministic JSON bytes.

    Rules:
      - Sorted keys (recursively)
      - UTF-8 encoding
      - No whitespace variation (compact)
      - No ensure_ascii (allows raw UTF-8)

    Args:
        obj: JSON-serializable object.

    Returns:
        bytes: Stable UTF-8 JSON representation.
    """
    json_str = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':')
    )
    return json_str.encode('utf-8')


def _validate_receipt_value(value: Any, key: str) -> None:
    """
    Recursively validate that value contains only allowed types.

    Allowed:
      - int, bool, str, None
      - list, tuple (of allowed types)
      - dict (keys must be str, values must be allowed types)

    Forbidden:
      - float (spec: "no floats in logic")
      - Any other type (no datetime, no object reprs, etc.)

    Args:
        value: Value to validate.
        key: Key name (for error messages).

    Raises:
        ReceiptError: If value contains forbidden types.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return

    if isinstance(value, float):
        raise ReceiptError(
            f"Floats forbidden in receipts (key: '{key}'). Spec: no floats in logic."
        )

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _validate_receipt_value(item, f"{key}[{i}]")
        return

    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ReceiptError(
                    f"Dict keys must be strings in receipts (key: '{key}', dict_key: {k})"
                )
            _validate_receipt_value(v, f"{key}.{k}")
        return

    # Anything else is forbidden
    raise ReceiptError(
        f"Invalid type in receipts: {type(value).__name__} (key: '{key}'). "
        f"Allowed: int, bool, str, None, list, tuple, dict."
    )


class ReceiptError(Exception):
    """Raised when receipt construction violates spec (duplicate key, invalid type, etc.)."""
    pass


class DeterminismError(Exception):
    """Raised when double-run produces different section hashes."""

    def __init__(
        self,
        section: str,
        first_differing_key: str | None,
        value_a: Any,
        value_b: Any,
        hash_a: str,
        hash_b: str
    ):
        self.section = section
        self.first_differing_key = first_differing_key
        self.value_a = value_a
        self.value_b = value_b
        self.hash_a = hash_a
        self.hash_b = hash_b

        msg = (
            f"Double-run hash mismatch in section '{section}'.\n"
            f"  First differing key: '{first_differing_key}'\n"
            f"  Value A: {value_a}\n"
            f"  Value B: {value_b}\n"
            f"  Hash A: {hash_a}\n"
            f"  Hash B: {hash_b}"
        )
        super().__init__(msg)
