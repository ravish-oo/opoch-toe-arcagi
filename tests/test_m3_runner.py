#!/usr/bin/env python3
"""
M3 Runner Integration Test

Verifies M3 receipts structure and selector output.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve

# Load ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

# Test on a simple task
task_id = "00576224"
task = all_tasks[task_id]

print(f"Testing M3 runner on task: {task_id}")
print("="*70)

# Run solver with H1-7
Y, receipts = solve(
    task,
    families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
    with_witness=True,
    with_unanimity=False
)

payload = receipts["payload"]

# Check all M3 sections
print("\n✓ RECEIPTS STRUCTURE:")
required_sections = [
    "color_universe.colors_order",
    "pack_unpack",
    "frames.canonicalize",
    "frames.apply_pose_anchor",
    "features.trainings",
    "working_canvas",
    "components",
    "witness_learn",
    "witness_emit",
    "unanimity_evaluated",
    "selection",
]

for section in required_sections:
    if section in payload:
        print(f"  ✓ {section}")
    else:
        print(f"  ❌ {section} MISSING")
        sys.exit(1)

# Check witness_emit structure
print("\n✓ WITNESS_EMIT SECTION:")
we = payload["witness_emit"]
required_we_fields = ["inputs", "scopes", "admits", "families", "per_training_piece_counts"]
for field in required_we_fields:
    if field in we:
        print(f"  ✓ {field}")
    else:
        print(f"  ❌ {field} MISSING")
        sys.exit(1)

print(f"    - R_out={we['inputs']['R_out']}, C_out={we['inputs']['C_out']}")
print(f"    - scope_bits={we['scopes']['scope_bits']}")
print(f"    - A_wit_hash={we['admits']['A_wit_hash'][:16]}...")

# Check selection structure
print("\n✓ SELECTION SECTION:")
sel = payload["selection"]
required_sel_fields = ["precedence", "counts", "containment_verified", "repaint_hash"]
for field in required_sel_fields:
    if field in sel:
        print(f"  ✓ {field}")
    else:
        print(f"  ❌ {field} MISSING")
        sys.exit(1)

print(f"    - precedence: {sel['precedence']}")
print(f"    - counts: {sel['counts']}")
print(f"    - containment_verified: {sel['containment_verified']}")

# Check invariants
print("\n✓ INVARIANTS:")

# Invariant 1: Pixel count balance
R_out = we["inputs"]["R_out"]
C_out = we["inputs"]["C_out"]
total_pixels = R_out * C_out
counts = sel["counts"]
pixels_covered = counts["witness"] + counts["unanimity"] + counts["bottom"]

print(f"  Total pixels: {total_pixels}")
print(f"  Covered: {pixels_covered} (witness={counts['witness']}, unanimity={counts['unanimity']}, bottom={counts['bottom']})")

assert pixels_covered == total_pixels, f"Pixel count mismatch: {pixels_covered} != {total_pixels}"
print(f"  ✓ Pixel count balance: {pixels_covered} == {total_pixels}")

# Invariant 2: Containment verified
assert sel["containment_verified"] == True, "Containment must be verified"
print(f"  ✓ Containment verified: True")

# Invariant 3: Unanimity OFF at M3
assert payload["unanimity_evaluated"] == False, "Unanimity should be OFF at M3"
print(f"  ✓ Unanimity OFF: unanimity_evaluated=False")

# Invariant 4: Y_out shape matches working canvas
assert len(Y) == R_out, f"Y_out height mismatch: {len(Y)} != {R_out}"
assert len(Y[0]) == C_out, f"Y_out width mismatch: {len(Y[0])} != {C_out}"
print(f"  ✓ Y_out shape: {len(Y)}×{len(Y[0])} matches working canvas")

print("\n" + "="*70)
print("✅ M3 RUNNER TEST PASSED")
print("="*70)
print(f"\nTask {task_id}:")
print(f"  Training pairs: {len(task['train'])}")
print(f"  Working canvas: {R_out}×{C_out}")
print(f"  Witness scope: {we['scopes']['scope_bits']} pixels")
print(f"  Selection: {counts['witness']} witness, {counts['unanimity']} unanimity, {counts['bottom']} bottom")
