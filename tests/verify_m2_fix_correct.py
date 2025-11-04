#!/usr/bin/env python3
"""Verify M2 fix - checking CORRECT hashes (unanimity_grid_hash vs repaint_hash)"""

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
print("M2 FIX VERIFICATION - CHECKING CORRECT HASHES")
print("=" * 70)
print()
print("Implementer's claim:")
print("  - unanimity_hash (planes format) vs repaint_hash → DIFFERENT (expected)")
print("  - unanimity_grid_hash (grid format) vs repaint_hash → SAME (required)")
print()
print("=" * 70)

all_passing = True

for task_id in BUG_TASKS:
    print(f"\nTask: {task_id}")
    print("-" * 70)

    task = all_tasks[task_id]

    try:
        Y_out, receipts = solve(task, families=["H1", "H2", "H3", "H4", "H5", "H6", "H7"])

        # Extract ALL unanimity hashes
        unanimity_data = receipts['payload']['unanimity']
        selection_data = receipts['payload']['selection']

        unanimity_hash = unanimity_data.get('unanimity_hash')
        unanimity_grid_hash = unanimity_data.get('unanimity_grid_hash')  # NEW field
        repaint_hash = selection_data['repaint_hash']

        # Extract counts
        unanimous_pixels = unanimity_data['unanimous_pixels']
        total_covered = unanimity_data['total_covered_pixels']
        counts = selection_data['counts']

        # Check dimensions
        R_out = receipts['payload']['working_canvas']['R_out']
        C_out = receipts['payload']['working_canvas']['C_out']
        total_pixels = R_out * C_out

        print(f"Canvas: {R_out}×{C_out} = {total_pixels} pixels")
        print(f"Unanimous pixels: {unanimous_pixels}/{total_covered}")
        print(f"Selection counts: unanimity={counts['unanimity']}, bottom={counts['bottom']}")
        print()

        print("Hash comparison:")
        print(f"  unanimity_hash (planes):     {unanimity_hash}")
        if unanimity_grid_hash:
            print(f"  unanimity_grid_hash (grid):  {unanimity_grid_hash}")
        else:
            print(f"  unanimity_grid_hash (grid):  ❌ FIELD MISSING")
        print(f"  repaint_hash (grid):         {repaint_hash}")
        print()

        # Check OLD comparison (my original verification)
        old_match = (unanimity_hash == repaint_hash)
        print(f"OLD check (unanimity_hash == repaint_hash): {old_match}")

        # Check NEW comparison (implementer's claim)
        if unanimity_grid_hash:
            new_match = (unanimity_grid_hash == repaint_hash)
            print(f"NEW check (unanimity_grid_hash == repaint_hash): {new_match}")
        else:
            new_match = False
            print(f"NEW check: ❌ Cannot verify - unanimity_grid_hash field missing")

        print()

        full_unanimity = (unanimous_pixels == total_pixels)
        full_from_unanimity = (counts['unanimity'] == total_pixels and counts['bottom'] == 0)

        print(f"✓ Full unanimity: {full_unanimity}")
        print(f"✓ Selector picks unanimity everywhere: {full_from_unanimity}")

        if full_unanimity and full_from_unanimity:
            # For full unanimity cases, the NEW check must pass
            if unanimity_grid_hash and new_match:
                print()
                print("✅ INVARIANT SATISFIED (unanimity_grid_hash == repaint_hash)")
            elif not unanimity_grid_hash:
                print()
                print("❌ FIX INCOMPLETE - unanimity_grid_hash field missing")
                all_passing = False
            else:
                print()
                print("❌ INVARIANT VIOLATED (unanimity_grid_hash ≠ repaint_hash)")
                all_passing = False
        else:
            print()
            print("⏭️  NOT FULL UNANIMITY - checking not required")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passing = False

print()
print("=" * 70)
if all_passing:
    print("✅ FIX VERIFIED - Implementer is correct!")
    print("   unanimity_grid_hash == repaint_hash for full unanimity cases")
else:
    print("❌ FIX INCOMPLETE - Invariants still violated or fields missing")
print("=" * 70)
