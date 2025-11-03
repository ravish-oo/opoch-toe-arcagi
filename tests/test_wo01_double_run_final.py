"""
WO-01 Double-Run Determinism Verification (After PERIOD Fix)

Verify that the PERIOD fix does not break determinism.
All hashes must remain stable across double-runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbit.core import assert_double_run_equal, Receipts, blake3_hash
from arcbit.kernel import minimal_period_row


def test_period_double_run():
    """Verify minimal_period_row is deterministic after fix."""

    test_masks = [
        (0b000000, 6),   # all zeros → None
        (0b111111, 6),   # all ones → None
        (0b101010, 6),   # stripe 2 → 2
        (0b110110, 6),   # stripe 3 → 3
        (0b101011, 6),   # no period → None
    ]

    def build_receipts():
        r = Receipts("test-period-double-run")

        for mask, W in test_masks:
            p = minimal_period_row(mask, W)
            r.put(f"period_{mask:08b}_W{W}", p)

        return r

    try:
        assert_double_run_equal(build_receipts)
        print("✅ PASS: PERIOD double-run determinism verified")
        return True
    except Exception as e:
        print(f"❌ FAIL: PERIOD double-run determinism broken: {e}")
        return False


if __name__ == "__main__":
    success = test_period_double_run()
    sys.exit(0 if success else 1)
