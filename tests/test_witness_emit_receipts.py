#!/usr/bin/env python3
"""Quick test to verify witness_emit receipts are in the bundle."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve

# Load one task
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

# Pick a simple task
task = all_tasks["007bbfb7"]

print("Running solver on task 007bbfb7...")
Y_out, receipts = solve(task, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                        with_witness=True, with_unanimity=False)

print("\n" + "="*70)
print("RECEIPTS STRUCTURE")
print("="*70)

payload = receipts["payload"]

print("\nTop-level keys in receipts['payload']:")
for key in sorted(payload.keys()):
    print(f"  - {key}")

print("\n" + "-"*70)
print("Checking witness receipts:")
print("-"*70)

# Check witness_learn
if "witness_learn" in payload:
    print(f"✓ witness_learn: {len(payload['witness_learn'])} trainings")
    if payload["witness_learn"]:
        sample = payload["witness_learn"][0]
        print(f"  Sample keys: {list(sample.keys())}")
else:
    print("❌ witness_learn: MISSING")

# Check witness_emit (should be there according to WO-07)
# BUT: emit_witness calls receipts.digest() internally and doesn't return it
# So we need to check if it's logged in the global receipts system

# Let me check what's actually in the payload
print("\n" + "-"*70)
print("Full payload structure:")
print("-"*70)
print(json.dumps({k: type(v).__name__ for k, v in payload.items()}, indent=2))

# Check selection
if "selection" in payload:
    print(f"\n✓ selection receipts present")
    print(f"  Keys: {list(payload['selection'].keys())}")
    print(f"  Counts: {payload['selection']['counts']}")
else:
    print("\n❌ selection: MISSING")

# Check unanimity_evaluated
if "unanimity_evaluated" in payload:
    print(f"\n✓ unanimity_evaluated: {payload['unanimity_evaluated']}")
else:
    print("\n❌ unanimity_evaluated: MISSING")

print("\n" + "="*70)
print("ISSUE CHECK")
print("="*70)

# The issue: witness_emit generates its own Receipts object internally
# and calls digest(), but doesn't return it to runner.py
# So runner.py never gets the witness_emit receipts section

if "witness_emit" not in payload:
    print("⚠️  PROBLEM FOUND:")
    print("  witness_emit receipts are generated in emit_witness()")
    print("  but NOT returned to runner.py!")
    print()
    print("  Current code in witness_emit.py:")
    print("    receipts = Receipts('witness_emit')")
    print("    receipts.put(...)")
    print("    _ = receipts.digest()  # ← Called but not returned")
    print("    return A_wit, S_wit    # ← Receipts not in return")
    print()
    print("  Fix needed: emit_witness should return receipts dict")
    print("  OR: runner should extract witness_emit receipts from global log")
