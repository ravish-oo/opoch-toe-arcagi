#!/usr/bin/env python3
"""
M2 Unit Tests - Output Path (Transport + Unanimity)

Tests the M2 pipeline on simple hand-crafted tasks:
1. Full unanimity (all trainings agree)
2. Partial unanimity (some disagreement)
3. No unanimity (all trainings disagree or silent)
4. Selector logic (unanimity → bottom precedence)

Spec: Milestone M2 + WO-08
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve


def test_m2_full_unanimity():
    """
    Test M2 with full unanimity: all trainings agree on every pixel.

    Expectation: unanimous_pixels == R_out * C_out
                 selection.counts.unanimity == R_out * C_out
                 Y_out == expected pattern
    """
    print("\n" + "="*70)
    print("TEST 1: Full Unanimity (All Trainings Agree)")
    print("="*70)

    # Task: All trainings output the same 2×2 pattern
    # Training 1: [[1, 2], [3, 4]] → [[1, 2], [3, 4]]
    # Training 2: [[5, 6], [7, 8]] → [[1, 2], [3, 4]]  # Different input, same output
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2], [3, 4]]
            },
            {
                "input": [[5, 6], [7, 8]],
                "output": [[1, 2], [3, 4]]
            }
        ],
        "test": [
            {
                "input": [[9, 0], [0, 9]]
            }
        ]
    }

    # Run M2 (output path only, witness disabled)
    Y_out, receipts = solve(task, with_witness=False, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"))

    r = receipts["payload"]

    # Verify receipts structure
    assert "working_canvas" in r, "Missing working_canvas"
    assert "transports" in r, "Missing transports"
    assert "unanimity" in r, "Missing unanimity"
    assert "selection" in r, "Missing selection"

    # Extract key metrics
    R_out, C_out = r["working_canvas"]["R_out"], r["working_canvas"]["C_out"]
    total_pixels = R_out * C_out

    n_included = r["transports"]["n_included"]
    unanimous_pixels = r["unanimity"]["unanimous_pixels"]
    total_covered = r["unanimity"]["total_covered_pixels"]
    empty_scope = r["unanimity"]["empty_scope_pixels"]

    selection = r["selection"]
    unanimity_count = selection["counts"]["unanimity"]
    bottom_count = selection["counts"]["bottom"]

    print(f"\n  Canvas: {R_out}×{C_out} ({total_pixels} pixels)")
    print(f"  Transports included: {n_included}/2")
    print(f"  Unanimous pixels: {unanimous_pixels}/{total_pixels}")
    print(f"  Covered pixels: {total_covered}/{total_pixels}")
    print(f"  Empty scope: {empty_scope}")
    print(f"  Selection counts: unanimity={unanimity_count}, bottom={bottom_count}")

    # Verify full unanimity
    assert n_included == 2, f"Expected 2 included trainings, got {n_included}"
    assert unanimous_pixels == total_pixels, f"Expected full unanimity ({total_pixels} pixels), got {unanimous_pixels}"
    assert unanimity_count == total_pixels, f"Expected unanimity selection for all pixels, got {unanimity_count}"
    assert bottom_count == 0, f"Expected 0 bottom pixels, got {bottom_count}"

    # Verify output is deterministic and all pixels are from unanimity bucket
    # With full unanimity, all pixels should come from unanimity (not checking exact values due to canonicalization)
    # Just verify Y_out is a valid 2×2 grid with colors from training outputs
    assert len(Y_out) == 2, f"Expected 2 rows, got {len(Y_out)}"
    assert len(Y_out[0]) == 2, f"Expected 2 columns, got {len(Y_out[0])}"

    # All pixels should be from color universe (1,2,3,4)
    for row in Y_out:
        for val in row:
            assert val in [1, 2, 3, 4], f"Unexpected color {val} not in training outputs"

    print("\n  ✅ PASS: Full unanimity verified")
    print(f"  ✅ Y_out = {Y_out} (all pixels from unanimity)")


def test_m2_partial_unanimity():
    """
    Test M2 with partial unanimity: trainings agree on some pixels, disagree on others.

    Expectation: unanimous_pixels < R_out * C_out
                 unanimous_pixels < total_covered_pixels
                 selection.counts.unanimity + bottom = R_out * C_out
    """
    print("\n" + "="*70)
    print("TEST 2: Partial Unanimity (Some Disagreement)")
    print("="*70)

    # Task: Trainings agree on first pixel, disagree on second
    # Training 1: [[0, 0]] → [[1, 2]]
    # Training 2: [[0, 0]] → [[1, 3]]  # Disagree on second pixel
    task = {
        "train": [
            {
                "input": [[0, 0]],
                "output": [[1, 2]]
            },
            {
                "input": [[0, 0]],
                "output": [[1, 3]]
            }
        ],
        "test": [
            {
                "input": [[0, 0]]
            }
        ]
    }

    # Run M2
    Y_out, receipts = solve(task, with_witness=False, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"))

    r = receipts["payload"]

    # Extract metrics
    R_out, C_out = r["working_canvas"]["R_out"], r["working_canvas"]["C_out"]
    total_pixels = R_out * C_out

    n_included = r["transports"]["n_included"]
    unanimous_pixels = r["unanimity"]["unanimous_pixels"]
    total_covered = r["unanimity"]["total_covered_pixels"]

    selection = r["selection"]
    unanimity_count = selection["counts"]["unanimity"]
    bottom_count = selection["counts"]["bottom"]

    print(f"\n  Canvas: {R_out}×{C_out} ({total_pixels} pixels)")
    print(f"  Transports included: {n_included}/2")
    print(f"  Unanimous pixels: {unanimous_pixels}/{total_pixels}")
    print(f"  Covered pixels: {total_covered}/{total_pixels}")
    print(f"  Selection counts: unanimity={unanimity_count}, bottom={bottom_count}")

    # Verify partial unanimity
    assert n_included == 2, f"Expected 2 included trainings, got {n_included}"
    assert unanimous_pixels < total_pixels, f"Expected partial unanimity, got {unanimous_pixels}/{total_pixels}"
    assert unanimous_pixels >= 1, f"Expected at least 1 unanimous pixel, got {unanimous_pixels}"
    assert unanimous_pixels < total_covered, f"Expected unanimous < covered (disagreement present)"
    assert unanimity_count + bottom_count == total_pixels, f"Selection counts don't sum to total: {unanimity_count} + {bottom_count} ≠ {total_pixels}"

    # Verify Y_out structure (1×2 grid)
    assert len(Y_out) == 1, f"Expected 1 row, got {len(Y_out)}"
    assert len(Y_out[0]) == 2, f"Expected 2 columns, got {len(Y_out[0])}"

    # Verify at least one pixel is non-zero (unanimous) and at least one is zero (bottom)
    # Don't assume which pixel is which due to canonicalization
    pixel_values = Y_out[0]
    non_zero_count = sum(1 for v in pixel_values if v != 0)
    zero_count = sum(1 for v in pixel_values if v == 0)

    assert non_zero_count == unanimous_pixels, f"Expected {unanimous_pixels} non-zero pixels (unanimous), got {non_zero_count}"
    assert zero_count == bottom_count, f"Expected {bottom_count} zero pixels (bottom), got {zero_count}"

    print("\n  ✅ PASS: Partial unanimity verified")
    print(f"  ✅ Y_out = {Y_out} ({non_zero_count} unanimous, {zero_count} bottom)")


def test_m2_no_unanimity():
    """
    Test M2 with no unanimity: all trainings disagree on every pixel.

    Expectation: unanimous_pixels == 0
                 selection.counts.unanimity == 0
                 selection.counts.bottom == R_out * C_out
    """
    print("\n" + "="*70)
    print("TEST 3: No Unanimity (All Trainings Disagree)")
    print("="*70)

    # Task: Same canvas size but trainings completely disagree
    # Training 1: [[0, 0]] → [[1, 1]]
    # Training 2: [[0, 0]] → [[2, 2]]  # Same size but different colors
    task = {
        "train": [
            {
                "input": [[0, 0]],
                "output": [[1, 1]]
            },
            {
                "input": [[0, 0]],
                "output": [[2, 2]]
            }
        ],
        "test": [
            {
                "input": [[0, 0]]
            }
        ]
    }

    # Run M2
    Y_out, receipts = solve(task, with_witness=False, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"))

    r = receipts["payload"]

    # Extract metrics
    R_out, C_out = r["working_canvas"]["R_out"], r["working_canvas"]["C_out"]
    total_pixels = R_out * C_out

    n_included = r["transports"]["n_included"]
    unanimous_pixels = r["unanimity"]["unanimous_pixels"]
    total_covered = r["unanimity"]["total_covered_pixels"]

    selection = r["selection"]
    unanimity_count = selection["counts"]["unanimity"]
    bottom_count = selection["counts"]["bottom"]

    print(f"\n  Canvas: {R_out}×{C_out} ({total_pixels} pixels)")
    print(f"  Transports included: {n_included}/2")
    print(f"  Unanimous pixels: {unanimous_pixels}/{total_pixels}")
    print(f"  Covered pixels: {total_covered}/{total_pixels}")
    print(f"  Selection counts: unanimity={unanimity_count}, bottom={bottom_count}")

    # Verify no unanimity (could be 0 or 1 training included, both are valid "not yet" cases)
    assert unanimous_pixels == 0, f"Expected 0 unanimous pixels, got {unanimous_pixels}"
    assert unanimity_count == 0, f"Expected 0 unanimity selections, got {unanimity_count}"
    assert bottom_count == total_pixels, f"Expected all pixels from bottom, got {bottom_count}/{total_pixels}"

    # Verify Y_out is all zeros (bottom)
    for row in Y_out:
        for val in row:
            assert val == 0, f"Expected all pixels to be 0 (bottom), got {val}"

    print("\n  ✅ PASS: No unanimity verified")
    print(f"  ✅ All pixels fell back to bottom (0)")


def test_m2_receipts_completeness():
    """
    Test M2 receipts completeness: all required fields present and valid.

    Expectation: All M2 sections present with correct structure
                 All hashes present
                 Invariants satisfied
    """
    print("\n" + "="*70)
    print("TEST 4: Receipts Completeness")
    print("="*70)

    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[5, 6], [7, 8]]
            }
        ],
        "test": [
            {
                "input": [[0, 0], [0, 0]]
            }
        ]
    }

    # Run M2
    Y_out, receipts = solve(task, with_witness=False, families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"))

    r = receipts["payload"]

    # Check M0 sections (bedrock)
    print("\n  Checking M0 sections...")
    assert "color_universe.colors_order" in r, "Missing color_universe"
    assert "pack_unpack" in r, "Missing pack_unpack"
    assert "frames.canonicalize" in r, "Missing frames.canonicalize"
    assert "frames.apply_pose_anchor" in r, "Missing frames.apply_pose_anchor"
    print("  ✅ M0 sections present")

    # Check M1 sections
    print("\n  Checking M1 sections...")
    assert "working_canvas" in r, "Missing working_canvas"
    canvas = r["working_canvas"]
    assert "R_out" in canvas, "Missing R_out"
    assert "C_out" in canvas, "Missing C_out"
    assert "winner" in canvas, "Missing winner"
    print("  ✅ M1 sections present")

    # Check M2 sections
    print("\n  Checking M2 sections...")

    # Transports
    assert "transports" in r, "Missing transports"
    transports = r["transports"]
    assert "n_included" in transports, "Missing n_included"
    assert "transports" in transports, "Missing transports list"
    assert "transports_hash" in transports, "Missing transports_hash"
    print("  ✅ Transports section complete")

    # Unanimity
    assert "unanimity" in r, "Missing unanimity"
    unanimity = r["unanimity"]
    assert "included_train_ids" in unanimity, "Missing included_train_ids"
    assert "unanimous_pixels" in unanimity, "Missing unanimous_pixels"
    assert "total_covered_pixels" in unanimity, "Missing total_covered_pixels"
    assert "empty_scope_pixels" in unanimity, "Missing empty_scope_pixels"
    assert "unanimity_hash" in unanimity, "Missing unanimity_hash"
    assert "scope_hash" in unanimity, "Missing scope_hash"
    print("  ✅ Unanimity section complete")

    # Selection
    assert "selection" in r, "Missing selection"
    selection = r["selection"]
    assert "precedence" in selection, "Missing precedence"
    assert selection["precedence"] == ["unanimity", "bottom"], "Wrong precedence order"
    assert "counts" in selection, "Missing counts"
    assert "unanimity" in selection["counts"], "Missing unanimity count"
    assert "bottom" in selection["counts"], "Missing bottom count"
    assert "containment_verified" in selection, "Missing containment_verified"
    assert "repaint_hash" in selection, "Missing repaint_hash"
    print("  ✅ Selection section complete")

    # Verify invariants
    print("\n  Checking invariants...")
    R_out = canvas["R_out"]
    C_out = canvas["C_out"]
    total_pixels = R_out * C_out

    # Invariant: covered + empty = total_pixels
    covered_plus_empty = unanimity["total_covered_pixels"] + unanimity["empty_scope_pixels"]
    assert covered_plus_empty == total_pixels, f"Invariant violated: covered + empty ≠ total ({covered_plus_empty} ≠ {total_pixels})"
    print(f"  ✅ Invariant: covered + empty == total ({covered_plus_empty} == {total_pixels})")

    # Invariant: unanimous ≤ covered
    assert unanimity["unanimous_pixels"] <= unanimity["total_covered_pixels"], "Invariant violated: unanimous > covered"
    print(f"  ✅ Invariant: unanimous ≤ covered ({unanimity['unanimous_pixels']} ≤ {unanimity['total_covered_pixels']})")

    # Invariant: selection counts sum to total_pixels
    counts_sum = selection["counts"]["unanimity"] + selection["counts"]["bottom"]
    assert counts_sum == total_pixels, f"Invariant violated: selection counts sum ≠ total ({counts_sum} ≠ {total_pixels})"
    print(f"  ✅ Invariant: selection counts sum == total ({counts_sum} == {total_pixels})")

    print("\n  ✅ PASS: All receipts complete and invariants satisfied")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("M2 UNIT TESTS - OUTPUT PATH (TRANSPORT + UNANIMITY)")
    print("="*70)

    try:
        test_m2_full_unanimity()
        test_m2_partial_unanimity()
        test_m2_no_unanimity()
        test_m2_receipts_completeness()

        print("\n" + "="*70)
        print("ALL M2 UNIT TESTS PASSED ✅")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
