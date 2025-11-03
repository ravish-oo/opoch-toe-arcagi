"""
WO-01 SPEC CONFORMANCE ISSUE INVESTIGATION

ISSUE: minimal_period_row() returns period=1, but spec says "2 <= p <= W"

Spec clause (WO-01 section 6):
  "Return p (2 <= p <= W) if the row's bitstring is an exact repetition
   with minimal period p; else None."

Implementation behavior:
  - Returns 1 for "000000" (all zeros)
  - Returns 1 for "111111" (all ones)

Spec example (WO-01 section 6):
  >>> minimal_period_row(0b111111, 6)  # "111111" → period 1 (but we require >= 2)
  1

AMBIGUITY:
  - Spec docstring says "2 <= p <= W" (p must be >= 2)
  - Spec example shows returning 1, with note "(but we require >= 2)"
  - Implementation returns 1 (does not filter for >= 2)

QUESTION: Should minimal_period_row() filter out period=1?

TESTING:
  - All-zeros "000000": Implementation returns 1, test expects None
  - All-ones "111111": Implementation returns 1, test expects 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.kernel import minimal_period_row


def investigate():
    print("=" * 70)
    print("WO-01 PERIOD SPEC CONFORMANCE INVESTIGATION")
    print("=" * 70)
    print()

    test_cases = [
        ("000000 (all zeros)", 0b000000, 6),
        ("111111 (all ones)", 0b111111, 6),
        ("101010 (stripe 2)", 0b101010, 6),
        ("110110 (stripe 3)", 0b110110, 6),
        ("101011 (no period)", 0b101011, 6),
    ]

    print("IMPLEMENTATION BEHAVIOR:")
    print("-" * 70)

    for label, mask, W in test_cases:
        p = minimal_period_row(mask, W)
        print(f"  {label:25} mask={bin(mask):10} W={W} → period={p}")

    print()
    print("SPEC ANALYSIS:")
    print("-" * 70)
    print("  Spec docstring: 'Return p (2 <= p <= W)'")
    print("  This REQUIRES p >= 2")
    print()
    print("  Spec example: minimal_period_row(0b111111, 6) → 1")
    print("  Example note: '(but we require >= 2)'")
    print()
    print("  Implementation code: if t < W and W % t == 0: return t")
    print("  This ALLOWS t=1 (no filtering)")
    print()

    print("VERDICT:")
    print("-" * 70)
    print("  ❌ SPEC NON-CONFORMANCE or SPEC AMBIGUITY")
    print()
    print("  Either:")
    print("    1. Implementation VIOLATES spec (should filter >= 2)")
    print("    2. Spec is AMBIGUOUS (docstring vs example)")
    print("    3. Spec example note is INCORRECT")
    print()
    print("  Recommendation: Clarify spec or fix implementation")
    print("=" * 70)


if __name__ == "__main__":
    investigate()
