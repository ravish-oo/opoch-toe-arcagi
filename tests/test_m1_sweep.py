#!/usr/bin/env python3
"""
M1 Canvas Online - Sweep Test on Real ARC Data

Tests M1 runner integration on 50 tasks (then 1000 if successful):
  1. Returns X* as placeholder (Y = X*)
  2. Always includes working_canvas receipts
  3. M0 sections unchanged
  4. Only failure mode: SIZE_UNDETERMINED with attempts trail
  5. Determinism check passes

SIZE_UNDETERMINED Triage (3-step):
  1. Inspect attempts trail (exhaustive H1-H7, full bounds)
  2. Cross-check with oracle (if needed)
  3. Check first_counterexample (bug vs genuine incompatibility)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve, solve_with_determinism_check
from src.arcbit.canvas import SizeUndetermined, parse_families, FULL_FAMILY_SET


# Load real ARC data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)


def verify_m0_sections(receipts, task_id):
    """Verify M0 sections are present and well-formed."""
    errors = []

    # Receipts have a payload wrapper
    payload = receipts.get("payload", {})

    # Check color_universe
    if "color_universe.colors_order" not in payload:
        errors.append(f"{task_id}: Missing color_universe.colors_order")
    if "color_universe.K" not in payload:
        errors.append(f"{task_id}: Missing color_universe.K")

    # Check pack_unpack
    if "pack_unpack" not in payload:
        errors.append(f"{task_id}: Missing pack_unpack")
    else:
        pack_unpack = payload["pack_unpack"]
        if not isinstance(pack_unpack, list):
            errors.append(f"{task_id}: pack_unpack should be list")
        else:
            for entry in pack_unpack:
                if "pack_equal" not in entry or not entry["pack_equal"]:
                    errors.append(f"{task_id}: pack_equal=False in pack_unpack")

    # Check frames
    if "frames.canonicalize" not in payload:
        errors.append(f"{task_id}: Missing frames.canonicalize")
    if "frames.apply_pose_anchor" not in payload:
        errors.append(f"{task_id}: Missing frames.apply_pose_anchor")

    return errors


def verify_working_canvas_section(receipts, task_id):
    """Verify working_canvas section is present and complete."""
    errors = []

    payload = receipts.get("payload", {})

    if "working_canvas" not in payload:
        errors.append(f"{task_id}: Missing working_canvas section")
        return errors

    canvas = payload["working_canvas"]

    # Required fields
    required = [
        "num_trainings",
        "features_hash_per_training",
        "test_input_shape",
        "attempts",
        "total_candidates_checked",
        "R_out",
        "C_out"
    ]

    for field in required:
        if field not in canvas:
            errors.append(f"{task_id}: Missing working_canvas.{field}")

    # Check winner present (if no SIZE_UNDETERMINED)
    if "winner" in canvas and canvas["winner"] is not None:
        winner = canvas["winner"]
        if "family" not in winner or "params" not in winner:
            errors.append(f"{task_id}: Malformed winner")

    # Check attempts is ordered list
    if "attempts" in canvas:
        if not isinstance(canvas["attempts"], list):
            errors.append(f"{task_id}: attempts should be list")

    return errors


def verify_no_lcm_fields(receipts, task_id):
    """Verify no LCM fields anywhere (v1.6 forbids LCM)."""
    receipts_str = json.dumps(receipts, sort_keys=True)

    forbidden = ["lcm_canvas", "lcm_shape", "upscale", "reduce_", "downscale"]
    found = []

    for term in forbidden:
        if term in receipts_str:
            found.append(term)

    if found:
        return [f"{task_id}: Found forbidden LCM terms: {found}"]
    return []


def analyze_size_undetermined(task_id, exception):
    """
    Analyze SIZE_UNDETERMINED to determine if it's a bug or genuine.

    3-Step Triage:
      1. Inspect attempts trail (exhaustive H1-H7?)
      2. Check for missing families or truncated bounds
      3. Examine first_counterexample
    """
    receipts = exception.receipts

    if "payload" not in receipts:
        return "BUG", "No receipts payload in SIZE_UNDETERMINED"

    payload = receipts["payload"]

    # Step 1: Check attempts trail
    if "attempts" not in payload:
        return "BUG", "No attempts trail in SIZE_UNDETERMINED"

    attempts = payload["attempts"]

    # Count attempts per family
    family_counts = {}
    for att in attempts:
        family = att["family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    # Expected counts (with full bounds):
    # H1: 8*8 = 64
    # H2: 17*17 = 289
    # H3: 8*8*17*17 = 18496
    # H4: 30*30 = 900
    # H5: 8*8 = 64
    # H6: 4*4 = 16
    # H7: 4*4 = 16

    expected = {
        "H1": 64,
        "H2": 289,
        "H3": 18496,
        "H4": 900,
        "H5": 64,
        "H6": 16,
        "H7": 16
    }

    # Check if all families present
    missing_families = []
    for family in expected:
        if family not in family_counts:
            missing_families.append(family)

    if missing_families:
        return "BUG", f"Missing families: {missing_families}"

    # Check if any family has truncated bounds
    truncated = []
    for family, expected_count in expected.items():
        actual_count = family_counts.get(family, 0)
        if actual_count < expected_count:
            truncated.append(f"{family}: {actual_count}/{expected_count}")

    if truncated:
        return "BUG", f"Truncated bounds: {truncated}"

    # Step 2: Check first_counterexample
    if "first_counterexample" not in payload:
        return "GENUINE", "All families tried, no counterexample provided (may be real)"

    counterexample = payload["first_counterexample"]

    # If counterexample shows conflicting requirements, it's genuine
    # This is a heuristic - deeper analysis would need oracle
    return "GENUINE", f"All H1-H7 tried with full bounds, counterexample: {counterexample}"


def test_task(task_id, task_json, use_determinism_check=False, families=FULL_FAMILY_SET, skip_h8h9_if_area1=False):
    """
    Test M1 runner on a single task.

    Returns:
        Tuple of (status, details):
            status: "PASS" | "SIZE_UNDETERMINED" | "ERROR"
            details: dict with test results
    """
    try:
        if use_determinism_check:
            Y, receipts = solve_with_determinism_check(task_json, families=families, skip_h8h9_if_area1=skip_h8h9_if_area1)
        else:
            Y, receipts = solve(task_json, families=families, skip_h8h9_if_area1=skip_h8h9_if_area1)

        # Verify Y = X* (placeholder)
        X_star = task_json["test"][0]["input"]
        if Y != X_star:
            return "ERROR", {"error": "Y != X* (should be identity at M1)"}

        # Verify M0 sections
        m0_errors = verify_m0_sections(receipts, task_id)
        if m0_errors:
            return "ERROR", {"error": f"M0 sections invalid: {m0_errors[0] if m0_errors else 'unknown'}"}

        # Verify working_canvas section
        canvas_errors = verify_working_canvas_section(receipts, task_id)
        if canvas_errors:
            return "ERROR", {"error": "working_canvas invalid", "details": canvas_errors}

        # Verify no LCM fields
        lcm_errors = verify_no_lcm_fields(receipts, task_id)
        if lcm_errors:
            return "ERROR", {"error": "LCM fields found", "details": lcm_errors}

        # Extract key info
        payload = receipts.get("payload", {})
        canvas = payload.get("working_canvas", {})
        winner = canvas.get("winner")
        R_out = canvas.get("R_out")
        C_out = canvas.get("C_out")

        return "PASS", {
            "winner": winner["family"] if winner else None,
            "size": f"{R_out}×{C_out}",
            "determinism": payload.get("determinism.double_run_ok", False)
        }

    except SizeUndetermined as e:
        # Analyze if bug or genuine
        classification, reason = analyze_size_undetermined(task_id, e)

        return "SIZE_UNDETERMINED", {
            "classification": classification,
            "reason": reason
        }

    except Exception as e:
        return "ERROR", {
            "error": str(e),
            "type": type(e).__name__
        }


def run_sweep(num_tasks=50, use_determinism_check=True, families=FULL_FAMILY_SET, skip_h8h9_if_area1=False):
    """
    Run M1 sweep on num_tasks.

    Args:
        num_tasks: Number of tasks to test (50 or 1000)
        use_determinism_check: Whether to run double-run determinism check
        families: Tuple of family IDs to evaluate (e.g., ("H1", ..., "H7"))
        skip_h8h9_if_area1: Skip H8/H9 if area=1 found
    """
    print("=" * 70)
    families_str = f"H1-{families[-1][1]}" if len(families) > 1 and families == tuple(f"H{i}" for i in range(1, int(families[-1][1])+1)) else str(families)
    print(f"M1 CANVAS ONLINE - SWEEP TEST ({num_tasks} tasks, families={families_str})")
    print("=" * 70)
    print()

    task_ids = sorted(all_tasks.keys())[:num_tasks]

    results = {
        "PASS": [],
        "SIZE_UNDETERMINED": [],
        "ERROR": []
    }

    size_undetermined_bugs = []
    size_undetermined_genuine = []

    for i, task_id in enumerate(task_ids):
        task_json = all_tasks[task_id]

        status, details = test_task(task_id, task_json, use_determinism_check, families, skip_h8h9_if_area1)
        results[status].append((task_id, details))

        if status == "PASS":
            winner = details.get("winner", "?")
            size = details.get("size", "?")
            det = "✓" if details.get("determinism") else "✗"
            print(f"  ✅ {task_id}: {winner} → {size} [det:{det}]")

        elif status == "SIZE_UNDETERMINED":
            classification = details.get("classification")
            reason = details.get("reason", "")

            if classification == "BUG":
                size_undetermined_bugs.append((task_id, reason))
                print(f"  🐛 {task_id}: SIZE_UNDETERMINED (BUG) - {reason}")
            else:
                size_undetermined_genuine.append((task_id, reason))
                print(f"  ⚠️  {task_id}: SIZE_UNDETERMINED (genuine) - {reason}")

        else:  # ERROR
            error = details.get("error", "Unknown")
            print(f"  ❌ {task_id}: ERROR - {error}")

        # Progress indicator every 10 tasks
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_tasks} tasks processed]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    num_pass = len(results["PASS"])
    num_size_und = len(results["SIZE_UNDETERMINED"])
    num_error = len(results["ERROR"])

    print(f"  ✅ PASS: {num_pass}/{num_tasks}")
    print(f"  ⚠️  SIZE_UNDETERMINED: {num_size_und}/{num_tasks}")
    if size_undetermined_bugs:
        print(f"     🐛 Bugs: {len(size_undetermined_bugs)}")
    if size_undetermined_genuine:
        print(f"     ✓ Genuine: {len(size_undetermined_genuine)}")
    print(f"  ❌ ERROR: {num_error}/{num_tasks}")

    # Detailed error report
    if num_error > 0:
        print("\n" + "-" * 70)
        print("ERROR DETAILS")
        print("-" * 70)
        for task_id, details in results["ERROR"][:10]:  # Show first 10
            error = details.get("error", "Unknown")
            print(f"  {task_id}: {error}")

    # SIZE_UNDETERMINED BUG report
    if size_undetermined_bugs:
        print("\n" + "-" * 70)
        print("SIZE_UNDETERMINED BUGS (Need fixing)")
        print("-" * 70)
        for task_id, reason in size_undetermined_bugs[:10]:
            print(f"  {task_id}: {reason}")

    # Final verdict
    print("\n" + "=" * 70)

    if num_error == 0 and len(size_undetermined_bugs) == 0:
        print(f"✅ M1 SWEEP PASSED ({num_pass + len(size_undetermined_genuine)}/{num_tasks} functional)")
        print(f"   {num_pass} tasks passed, {len(size_undetermined_genuine)} genuine SIZE_UNDETERMINED")
        return 0
    else:
        print(f"❌ M1 SWEEP FAILED")
        if num_error > 0:
            print(f"   {num_error} errors need fixing")
        if size_undetermined_bugs:
            print(f"   {len(size_undetermined_bugs)} SIZE_UNDETERMINED bugs need fixing")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M1' Sweep Test with Family Gating")
    parser.add_argument("num_tasks", type=int, nargs="?", default=50, help="Number of tasks (default: 50)")
    parser.add_argument("--families", type=str, default=None, help="Family allow-list: 'H1-7' or 'H1,H2,H5'")
    parser.add_argument("--skip-h8h9-if-area1", action="store_true", help="Skip H8/H9 if area=1 found")
    parser.add_argument("--no-determinism-check", action="store_true", help="Skip determinism check (faster)")

    args = parser.parse_args()

    # Parse families
    families = parse_families(args.families) if args.families else FULL_FAMILY_SET
    use_determinism_check = not args.no_determinism_check

    exit_code = run_sweep(
        num_tasks=args.num_tasks,
        use_determinism_check=use_determinism_check,
        families=families,
        skip_h8h9_if_area1=args.skip_h8h9_if_area1
    )
    sys.exit(exit_code)
