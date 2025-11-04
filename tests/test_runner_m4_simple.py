#!/usr/bin/env python3
"""
M4 Integration Test - Simple Smoke Test

Verifies runner with M4 (forbids + LFP) on a minimal task.

Spec: M4 v1.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve

print("=" * 70)
print("M4 RUNNER INTEGRATION - SMOKE TEST")
print("=" * 70)

# =============================================================================
# Test: Simple Identity Task (minimal training)
# =============================================================================

print("\n" + "=" * 70)
print("TEST: Simple Identity Task")
print("=" * 70)

# Minimal task: 1 training, identity transformation
task_json = {
    "train": [
        {
            "input": [[1, 2], [2, 1]],
            "output": [[1, 2], [2, 1]]
        }
    ],
    "test": [
        {
            "input": [[1, 2], [2, 1]]
        }
    ]
}

try:
    # Run solver with M4 enabled
    Y_out, receipts = solve(task_json, with_witness=False)

    print("\n✅ Solver completed successfully")
    print(f"Output: {Y_out}")

    # Receipts are nested in payload
    payload = receipts.get('payload', {})
    print(f"\nReceipt payload keys: {list(payload.keys())}")

    # Verify M4 receipts are present
    assert "forbids" in payload, "Missing forbids receipts"
    assert "lfp" in payload, "Missing LFP receipts"
    assert "lfp_status" in payload, "Missing LFP status"

    print(f"\n📊 M4 Receipts:")
    print(f"  Forbids:")
    print(f"    - matrix_hash: {payload['forbids']['matrix_hash'][:16]}...")
    print(f"    - edges_count: {payload['forbids']['edges_count']}")
    print(f"  LFP:")
    print(f"    - status: {payload['lfp_status']}")
    print(f"    - admit_passes: {payload['lfp']['admit_passes']}")
    print(f"    - ac3_passes: {payload['lfp']['ac3_passes']}")
    print(f"    - total_admit_prunes: {payload['lfp']['total_admit_prunes']}")
    print(f"    - total_ac3_prunes: {payload['lfp']['total_ac3_prunes']}")
    print(f"    - empties: {payload['lfp']['empties']}")
    print(f"    - domains_hash: {payload['lfp']['domains_hash'][:16]}...")

    # Verify LFP status is SUCCESS
    assert payload['lfp_status'] == "SUCCESS", f"Expected SUCCESS, got {payload['lfp_status']}"

    # Verify LFP completed (empties == 0)
    assert payload['lfp']['empties'] == 0, "LFP should not have empty domains"

    # Verify output is correct (identity)
    expected_output = [[1, 2], [2, 1]]
    assert Y_out == expected_output, f"Expected {expected_output}, got {Y_out}"

    print("\n✅ All M4 invariants verified")
    print("=" * 70)
    print("✅ SMOKE TEST PASSED")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
