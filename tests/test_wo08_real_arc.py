#!/usr/bin/env python3
"""
WO-08 Real ARC Data Test

Test output_transport and unanimity on actual ARC tasks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.output_transport import emit_output_transport
from src.arcbit.emitters.unanimity import emit_unity


def load_task(task_id):
    """Load task from ARC dataset."""
    data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
    with open(data_path) as f:
        all_tasks = json.load(f)
    return all_tasks.get(task_id)


def test_real_task():
    """Test on a real ARC task with mixed-size outputs."""
    print("=" * 80)
    print("WO-08 REAL ARC DATA TEST")
    print("=" * 80)
    print()

    # Use a task with simple size pattern
    task_id = "00576224"
    task = load_task(task_id)

    if not task:
        print(f"Task {task_id} not found in dataset")
        return

    # Extract training outputs
    Y_list = [pair["output"] for pair in task["train"]]

    # Working canvas: assume 6×6 for this test
    R_out, C_out = 6, 6

    # Create dummy frames (all identity at origin for simplicity)
    frames_out = [("I", (0, 0)) for _ in Y_list]
    pi_out_star = ("I", (0, 0))

    # Get all colors from trainings
    all_colors = set()
    for Y_i in Y_list:
        for row in Y_i:
            all_colors.update(row)
    colors_order = sorted(all_colors)

    print(f"Task: {task_id}")
    print(f"Trainings: {len(Y_list)}")
    print(f"Working canvas: {R_out}×{C_out}")
    print(f"Colors: {colors_order}")
    print()

    # Test output transport
    print("Testing output_transport...")
    A_out_list, S_out_list, receipts, section = emit_output_transport(
        Y_list, frames_out, R_out, C_out, colors_order, pi_out_star
    )

    print(f"✓ Output transport completed")
    print(f"  n_included: {section['payload']['n_included']}")
    print()

    # Show per-training results
    for i, receipt in enumerate(receipts):
        H_i = len(Y_list[i])
        W_i = len(Y_list[i][0]) if Y_list[i] else 0
        print(f"  Training {i}: {H_i}×{W_i}")
        print(f"    norm_kind: {receipt['norm_kind']}")
        if receipt['norm_kind'] != 'silent':
            print(f"    s_r={receipt['s_r']}, s_c={receipt['s_c']}")
            print(f"    scope_bits: {receipt['scope_bits']}")
        else:
            print(f"    (silent - no exact integer relation)")
        print()

    # Test unanimity
    print("Testing unanimity...")
    A_uni, S_uni, uni_receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    print(f"✓ Unanimity completed")
    print(f"  included_train_ids: {uni_receipt['included_train_ids']}")
    print(f"  unanimous_pixels: {uni_receipt['unanimous_pixels']}")
    print(f"  total_covered_pixels: {uni_receipt['total_covered_pixels']}")
    print(f"  empty_scope_pixels: {uni_receipt['empty_scope_pixels']}")
    print()

    # Verify receipts
    print("Verifying receipts...")
    assert "transports_hash" in section["payload"], "Missing transports_hash"
    assert "section_hash" in section, "Missing section_hash"
    assert "unanimity_hash" in uni_receipt, "Missing unanimity_hash"
    assert "scope_hash" in uni_receipt, "Missing scope_hash"
    print("✓ All receipt fields present")
    print()

    # Verify invariants
    total_pixels = R_out * C_out
    covered_plus_empty = (
        uni_receipt["total_covered_pixels"] + uni_receipt["empty_scope_pixels"]
    )
    assert (
        covered_plus_empty == total_pixels
    ), f"Invariant violated: covered({uni_receipt['total_covered_pixels']}) + empty({uni_receipt['empty_scope_pixels']}) ≠ total({total_pixels})"
    print("✓ Invariant: covered + empty == total_pixels")

    assert (
        uni_receipt["unanimous_pixels"] <= uni_receipt["total_covered_pixels"]
    ), "Invariant violated: unanimous > covered"
    print("✓ Invariant: unanimous ≤ covered")
    print()

    print("=" * 80)
    print("REAL ARC DATA TEST PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_real_task()
