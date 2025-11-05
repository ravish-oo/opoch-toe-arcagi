#!/usr/bin/env python3
"""
M0 Runner Verification Tests

Tests the M0 "Bedrock" runner on real ARC tasks to verify:
1. Color universe construction
2. PACK↔UNPACK identity for all grids
3. Canonicalize idempotence for all grids
4. apply_pose_anchor equivalence for X*
5. Determinism (double-run hash equality)

Spec: Milestone M0.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve, solve_with_determinism_check


def load_arc_tasks(task_file: str, num_tasks: int = 5) -> dict:
    """Load a subset of ARC tasks for testing."""
    with open(task_file, 'r') as f:
        all_tasks = json.load(f)

    # Take first num_tasks
    subset = {}
    for i, (task_id, task_data) in enumerate(all_tasks.items()):
        if i >= num_tasks:
            break
        subset[task_id] = task_data

    return subset


def test_m0_basic():
    """Test M0+M2 runner on a single simple task."""
    print("Testing M0+M2 runner on simple task...")

    # Simple task: 2x2 grid
    task = {
        "train": [
            {
                "input": [[0, 1], [1, 0]],
                "output": [[1, 0], [0, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 2], [2, 0]]
            }
        ]
    }

    # Run M2 (output path only, witness disabled)
    Y, receipts = solve(task, with_witness=False)

    # M2 produces actual predictions using unanimity
    # Just verify it's a valid 2x2 grid
    assert len(Y) == 2 and len(Y[0]) == 2, "Y must be 2x2 grid"

    # Extract payload for easier access
    r = receipts["payload"]

    # Verify receipts structure
    assert "color_universe.colors_order" in r
    assert "pack_unpack" in r
    assert "frames.canonicalize" in r
    assert "frames.apply_pose_anchor" in r
    assert "working_canvas" in r, "M1: working_canvas section must exist"
    assert "transports" in r, "M2: transports section must exist"
    assert "unanimity" in r, "M2: unanimity section must exist"
    assert "selection" in r, "M2: selection section must exist"

    # Verify color universe
    colors = r["color_universe.colors_order"]
    assert 0 in colors, "Color 0 must be in universe"
    assert 1 in colors, "Color 1 must be in universe (from training)"
    assert 2 in colors, "Color 2 must be in universe (from test)"

    # Verify pack_unpack identity
    pack_receipts = r["pack_unpack"]
    assert len(pack_receipts) == 3, "Must have 3 grids (train input, train output, test input)"
    for pr in pack_receipts:
        assert pr["pack_equal"] == True, f"PACK↔UNPACK identity failed for {pr['split']}[{pr['idx']}].{pr['io_type']}"

    # Verify canonicalize idempotence
    frame_receipts = r["frames.canonicalize"]
    assert len(frame_receipts) == 3, "Must have 3 frame receipts"
    for fr in frame_receipts:
        assert fr["idempotent"] == True, f"Canonicalize idempotence failed for {fr['split']}[{fr['idx']}].{fr['io_type']}"

    # Verify apply_pose_anchor equivalence
    apply_receipt = r["frames.apply_pose_anchor"]
    assert apply_receipt["equivalence_ok"] == True, "apply_pose_anchor equivalence failed"
    assert apply_receipt["hash_equal"] == True, "apply_pose_anchor hash mismatch"

    # Verify M4.5 selection receipts (domain-driven)
    selection = r["selection"]
    assert selection["precedence"] == ["domain_singleton", "domain_multi_min", "bottom"], \
        "M4.5 precedence must be domain-driven"
    assert "counts" in selection, "Selection must have counts"
    assert "repaint_hash" in selection, "Selection must have repaint_hash"

    print("✓ Basic M0+M2 test passed")


def test_m0_determinism():
    """Test M2 determinism check (double-run)."""
    print("Testing M2 determinism (double-run)...")

    task = {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[5, 4, 3], [2, 1, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [1, 0]]
            }
        ]
    }

    # Run M2 with determinism check
    Y, receipts = solve_with_determinism_check(task, with_witness=False)

    # Verify determinism flags
    assert "determinism.double_run_ok" in receipts, "Missing determinism.double_run_ok"
    assert receipts["determinism.double_run_ok"] == True, "Determinism check failed"
    assert receipts["determinism.sections_checked"] > 0, "No sections checked"

    print(f"✓ Determinism test passed ({receipts['determinism.sections_checked']} sections checked)")


def test_m0_on_arc_tasks():
    """Test M0 runner on real ARC training tasks."""
    print("Testing M0 runner on real ARC tasks...")

    # Load subset of ARC tasks
    arc_file = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
    if not arc_file.exists():
        print("  ⚠ Skipping: ARC training challenges not found")
        return

    tasks = load_arc_tasks(str(arc_file), num_tasks=5)

    print(f"  Loaded {len(tasks)} tasks from ARC training set")

    passed = 0
    failed = 0

    for task_id, task_data in tasks.items():
        try:
            # Run M2 (output path only) with determinism check
            Y, receipts = solve_with_determinism_check(task_data, with_witness=False)

            # M2 produces actual predictions, not X*
            # Just verify it's a valid grid
            X_star = task_data["test"][0]["input"]
            assert isinstance(Y, list) and len(Y) > 0, f"Task {task_id}: Y must be non-empty list"

            # Verify determinism
            assert receipts["determinism.double_run_ok"] == True, f"Task {task_id}: Determinism check failed"

            # Extract payload
            r = receipts["payload"]

            # Verify all pack_unpack identities
            for pr in r["pack_unpack"]:
                assert pr["pack_equal"] == True, f"Task {task_id}: PACK↔UNPACK identity failed"

            # Verify all canonicalize idempotence
            for fr in r["frames.canonicalize"]:
                assert fr["idempotent"] == True, f"Task {task_id}: Canonicalize idempotence failed"

            # Verify apply_pose_anchor equivalence
            assert r["frames.apply_pose_anchor"]["equivalence_ok"] == True, \
                f"Task {task_id}: apply_pose_anchor equivalence failed"

            print(f"  ✓ Task {task_id}: PASS")
            passed += 1

        except Exception as e:
            print(f"  ✗ Task {task_id}: FAIL - {e}")
            failed += 1

    print(f"  Results: {passed} passed, {failed} failed")

    if failed > 0:
        raise AssertionError(f"M0 failed on {failed} ARC tasks")

    print("✓ All ARC tasks passed M0 validation")


def test_m0_edge_cases():
    """Test M2 runner on edge cases."""
    print("Testing M2 runner on edge cases...")

    # Edge case 1: All zeros (constant grid)
    task1 = {
        "train": [
            {
                "input": [[0, 0], [0, 0]],
                "output": [[0, 0], [0, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 0], [0, 0]]
            }
        ]
    }

    Y1, receipts1 = solve(task1, with_witness=False)
    assert Y1 == [[0, 0], [0, 0]]  # Unanimity should agree on all zeros
    assert receipts1["payload"]["color_universe.colors_order"] == [0]
    print("  ✓ Edge case 1: All zeros")

    # Edge case 2: Single pixel
    task2 = {
        "train": [
            {
                "input": [[5]],
                "output": [[7]]
            }
        ],
        "test": [
            {
                "input": [[3]]
            }
        ]
    }

    Y2, receipts2 = solve(task2, with_witness=False)
    assert len(Y2) == 1 and len(Y2[0]) == 1  # Valid 1x1 grid
    r2 = receipts2["payload"]
    assert 0 in r2["color_universe.colors_order"]  # Background always included
    assert 3 in r2["color_universe.colors_order"]  # From test
    assert 5 in r2["color_universe.colors_order"]  # From training input
    assert 7 in r2["color_universe.colors_order"]  # From training output
    print("  ✓ Edge case 2: Single pixel")

    # Edge case 3: Large color palette (K=10)
    task3 = {
        "train": [
            {
                "input": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                "output": [[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 1, 2], [3, 4, 5]]
            }
        ]
    }

    Y3, receipts3 = solve(task3, with_witness=False)
    assert len(Y3) > 0 and len(Y3[0]) > 0  # Valid grid (size determined by working canvas)
    assert receipts3["payload"]["color_universe.K"] == 10  # 0..9
    print("  ✓ Edge case 3: Large color palette (K=10)")

    # Edge case 4: Rectangular grid (non-square)
    task4 = {
        "train": [
            {
                "input": [[1, 2, 3, 4, 5, 6]],
                "output": [[6, 5, 4, 3, 2, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 1], [2, 3], [4, 5]]
            }
        ]
    }

    Y4, receipts4 = solve(task4, with_witness=False)
    assert len(Y4) > 0 and len(Y4[0]) > 0  # Valid grid (size determined by working canvas)
    print("  ✓ Edge case 4: Rectangular grid")

    print("✓ All edge cases passed")


if __name__ == "__main__":
    print("=" * 60)
    print("M0 Runner Verification Tests")
    print("=" * 60)
    print()

    test_m0_basic()
    test_m0_determinism()
    test_m0_edge_cases()
    test_m0_on_arc_tasks()

    print()
    print("=" * 60)
    print("✅ All M0 tests passed!")
    print("=" * 60)
    print()
    print("M0 Implementation Complete:")
    print("  - Color universe (A.1) with receipts")
    print("  - PACK↔UNPACK identity (all grids)")
    print("  - Canonicalize idempotence (all grids)")
    print("  - apply_pose_anchor equivalence (X* proof)")
    print("  - Determinism (double-run hash equality)")
    print("  - No heuristics, no floats, no RNG")
    print("  - Receipts-first: every step logged")
    print("  - Fail-closed: raise on any violation")
