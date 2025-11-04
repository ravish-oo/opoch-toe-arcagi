#!/usr/bin/env python3
"""Verify M2 fix - check if unanimity_hash == repaint_hash after fix"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve

# Load data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

BUG_TASKS = ['4522001f', 'bdad9b1f']

print("=" * 70)
print("M2 FIX VERIFICATION - FRESH RUN")
print("=" * 70)

all_passing = True

for task_id in BUG_TASKS:
    print(f"\nTask: {task_id}")
    print("-" * 70)

    task = all_tasks[task_id]

    try:
        Y_out, receipts = solve(task, families=["H1", "H2", "H3", "H4", "H5", "H6", "H7"])

        # Extract hashes
        unanimity_hash = receipts['payload']['unanimity']['unanimity_hash']
        repaint_hash = receipts['payload']['selection']['repaint_hash']

        # Extract counts
        unanimous_pixels = receipts['payload']['unanimity']['unanimous_pixels']
        total_covered = receipts['payload']['unanimity']['total_covered_pixels']
        counts = receipts['payload']['selection']['counts']

        # Check dimensions
        R_out = receipts['payload']['working_canvas']['R_out']
        C_out = receipts['payload']['working_canvas']['C_out']
        total_pixels = R_out * C_out

        print(f"Canvas: {R_out}×{C_out} = {total_pixels} pixels")
        print(f"Unanimous pixels: {unanimous_pixels}/{total_covered}")
        print(f"Selection counts: unanimity={counts['unanimity']}, bottom={counts['bottom']}")
        print()
        print(f"unanimity_hash: {unanimity_hash}")
        print(f"repaint_hash:   {repaint_hash}")

        hashes_match = (unanimity_hash == repaint_hash)
        full_unanimity = (unanimous_pixels == total_pixels)
        full_from_unanimity = (counts['unanimity'] == total_pixels and counts['bottom'] == 0)

        print()
        print(f"✓ Full unanimity: {full_unanimity}")
        print(f"✓ Selector picks unanimity everywhere: {full_from_unanimity}")
        print(f"✓ Hashes match: {hashes_match}")

        if hashes_match and full_unanimity and full_from_unanimity:
            print()
            print("✅ INVARIANT SATISFIED")
        else:
            print()
            print("❌ INVARIANT VIOLATED")
            all_passing = False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passing = False

print()
print("=" * 70)
if all_passing:
    print("✅ FIX VERIFIED - All invariants satisfied")
else:
    print("❌ FIX INCOMPLETE - Invariants still violated")
print("=" * 70)
