#!/usr/bin/env python3
"""Verify witness_emit receipts contents."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve

arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

task = all_tasks["007bbfb7"]

Y_out, receipts = solve(task, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                        with_witness=True, with_unanimity=False)

print("="*70)
print("WITNESS_EMIT RECEIPTS VERIFICATION")
print("="*70)

witness_emit = receipts["payload"]["witness_emit"]

print("\nwitness_emit keys:")
for key in sorted(witness_emit.keys()):
    print(f"  - {key}")

print("\n" + "-"*70)
print("Required fields (per M3 spec):")
print("-"*70)

required = {
    "inputs": ["R_out", "C_out", "num_trainings", "num_pieces_total"],
    "scopes": ["per_training_scope_bits", "scope_bits"],
    "admits": ["A_wit_hash"],
    "section_hash": None,
}

all_ok = True

for field, subfields in required.items():
    if field in witness_emit:
        print(f"✓ {field}")
        if subfields:
            for subfield in subfields:
                if subfield in witness_emit[field]:
                    val = witness_emit[field][subfield]
                    print(f"    ✓ {subfield}: {val}")
                else:
                    print(f"    ❌ {subfield}: MISSING")
                    all_ok = False
    else:
        print(f"❌ {field}: MISSING")
        all_ok = False

print("\n" + "="*70)
if all_ok:
    print("✅ ALL REQUIRED FIELDS PRESENT")
else:
    print("❌ SOME FIELDS MISSING")

print("\nFull witness_emit structure:")
print(json.dumps(witness_emit, indent=2))
