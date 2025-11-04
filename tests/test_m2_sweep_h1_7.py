#!/usr/bin/env python3
"""
M2 Sweep on H1-7 Tasks (861 tasks that passed M1')

Classifies each task using receipts (M2 taxonomy):
  - SOLVED_BY_UNANIMITY: Full unanimity + Y_out == ground truth
  - NOT_YET_NORMALIZATION: No trainings normalize (n_included=0)
  - NOT_YET_DISAGREEING_OUTPUTS: Coverage but disagreement
  - BUG_TRANSPORT_NORMALIZE: Full unanimity but Y_out ≠ GT
  - BUG_RECEIPT_COUNTING: Counting logic errors

Spec: Milestone M2 (Output path only)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve
from src.arcbit.canvas import SizeUndetermined


# Load data
arc_data_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"
arc_solutions_path = Path(__file__).parent.parent / "data" / "arc-agi_training_solutions.json"

with open(arc_data_path, "r") as f:
    all_tasks = json.load(f)

with open(arc_solutions_path, "r") as f:
    all_solutions = json.load(f)


# Load M1' passing tasks (from previous milestone log)
def load_m1_prime_passing_tasks():
    """Load list of tasks that passed M1' with H1-7."""
    log_path = Path("/tmp/m1_prime_h1-7_1000.log")
    if not log_path.exists():
        print("⚠️  M1' log not found, will test all tasks")
        return None

    passing = []
    with open(log_path, "r") as f:
        for line in f:
            if "✅" in line and "→" in line:
                # Extract task ID
                parts = line.strip().split()
                if len(parts) >= 2:
                    task_id = parts[1].rstrip(":")
                    passing.append(task_id)

    return passing


def classify_task_result(task_id, Y_out, Y_expected, receipts):
    """
    Classify task result using receipts (M2 taxonomy).

    Returns: (status, details dict)
      status ∈ {SOLVED_BY_UNANIMITY, NOT_YET_NORMALIZATION, NOT_YET_DISAGREEING_OUTPUTS,
                BUG_TRANSPORT_NORMALIZE, BUG_RECEIPT_COUNTING, NO_GT}

    Classification logic (per M2 spec):
      1. SOLVED_BY_UNANIMITY:
         - unanimous_pixels == R_out * C_out
         - selection.counts.unanimity == R_out * C_out
         - Y_out == GT

      2. NOT_YET_NORMALIZATION:
         - transports.n_included == 0 OR unanimity.total_covered_pixels == 0
         - selection.counts.unanimity == 0, all pixels from bottom
         - Expected at M2 (H2/H6/H7 choices or decimation failures)

      3. NOT_YET_DISAGREEING_OUTPUTS:
         - total_covered_pixels > 0 and unanimous_pixels < total_covered_pixels
         - selection.counts.bottom > 0
         - Trainings conflict, needs M4/M5

      4. BUG_TRANSPORT_NORMALIZE:
         - unanimous_pixels == R_out * C_out (full unanimity)
         - but Y_out != GT
         - Real bug: transport/pose/anchor or selector broken

      5. BUG_RECEIPT_COUNTING:
         - Selection counts don't match scope/admits
         - Counting logic error
    """
    if Y_expected is None:
        return "NO_GT", {}

    # Exact match check
    exact_match = (Y_out == Y_expected)

    # Extract receipt signals
    r = receipts["payload"]
    selection = r["selection"]
    transports = r["transports"]
    unanimity = r["unanimity"]

    R_out = r["working_canvas"]["R_out"]
    C_out = r["working_canvas"]["C_out"]
    total_pixels = R_out * C_out

    n_included = transports["n_included"]
    unanimous_pixels = unanimity["unanimous_pixels"]
    total_covered = unanimity["total_covered_pixels"]
    empty_scope = unanimity["empty_scope_pixels"]

    unanimity_count = selection["counts"]["unanimity"]
    bottom_count = selection["counts"]["bottom"]

    # Compute coverage
    unanimity_coverage = unanimous_pixels / total_pixels if total_pixels > 0 else 0

    details = {
        "match": exact_match,
        "R_out": R_out,
        "C_out": C_out,
        "total_pixels": total_pixels,
        "n_included": n_included,
        "unanimous_pixels": unanimous_pixels,
        "total_covered": total_covered,
        "empty_scope": empty_scope,
        "unanimity_coverage": unanimity_coverage,
        "unanimity_count": unanimity_count,
        "bottom_count": bottom_count,
    }

    # Check for counting bugs (BUG_RECEIPT_COUNTING)
    # Invariant: covered + empty = total_pixels
    if total_covered + empty_scope != total_pixels:
        return "BUG_RECEIPT_COUNTING", {
            **details,
            "bug_reason": f"covered({total_covered}) + empty({empty_scope}) ≠ total({total_pixels})"
        }

    # Invariant: unanimous ≤ covered
    if unanimous_pixels > total_covered:
        return "BUG_RECEIPT_COUNTING", {
            **details,
            "bug_reason": f"unanimous({unanimous_pixels}) > covered({total_covered})"
        }

    # Invariant: selection counts sum to total
    if unanimity_count + bottom_count != total_pixels:
        return "BUG_RECEIPT_COUNTING", {
            **details,
            "bug_reason": f"selection counts sum ({unanimity_count + bottom_count}) ≠ total({total_pixels})"
        }

    # Classification logic (M2 spec)

    # 1. SOLVED_BY_UNANIMITY
    if unanimous_pixels == total_pixels and unanimity_count == total_pixels and exact_match:
        return "SOLVED_BY_UNANIMITY", details

    # 2. NOT_YET_NORMALIZATION (no coverage)
    if n_included == 0 or total_covered == 0:
        return "NOT_YET_NORMALIZATION", details

    # 3. BUG_TRANSPORT_NORMALIZE (full unanimity but mismatch)
    if unanimous_pixels == total_pixels and unanimity_count == total_pixels and not exact_match:
        return "BUG_TRANSPORT_NORMALIZE", details

    # 4. NOT_YET_DISAGREEING_OUTPUTS (partial unanimity or no unanimity with coverage)
    if total_covered > 0 and unanimous_pixels < total_covered:
        return "NOT_YET_DISAGREEING_OUTPUTS", details

    # Default: NOT_YET (other cases)
    return "NOT_YET_OTHER", details


def run_sweep(task_ids=None, max_tasks=None, verbose=False):
    """Run M2 sweep on specified tasks."""

    if task_ids is None:
        task_ids = sorted(all_tasks.keys())

    if max_tasks:
        task_ids = task_ids[:max_tasks]

    results = {
        "SOLVED_BY_UNANIMITY": [],
        "NOT_YET_NORMALIZATION": [],
        "NOT_YET_DISAGREEING_OUTPUTS": [],
        "NOT_YET_OTHER": [],
        "BUG_TRANSPORT_NORMALIZE": [],
        "BUG_RECEIPT_COUNTING": [],
        "NO_GT": [],
        "SIZE_UNDETERMINED": [],
        "ERROR": [],
    }

    for i, task_id in enumerate(task_ids):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(task_ids)}] processed...")

        task = all_tasks[task_id]

        try:
            # Run solver with H1-7, M2 only (witness disabled)
            Y_out, receipts = solve(
                task,
                families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                with_witness=False
            )

            # Get ground truth (first test output)
            Y_expected = all_solutions.get(task_id, [None])[0]

            # Classify
            status, details = classify_task_result(task_id, Y_out, Y_expected, receipts)

            results[status].append({
                "task_id": task_id,
                **details
            })

            if verbose:
                if status == "SOLVED_BY_UNANIMITY":
                    print(f"  ✅ {status}: {task_id}")
                elif status.startswith("BUG"):
                    print(f"  🐛 {status}: {task_id}")

        except SizeUndetermined as e:
            # SIZE_UNDETERMINED: expected for some tasks with H1-7
            results["SIZE_UNDETERMINED"].append({
                "task_id": task_id,
                "error": "SIZE_UNDETERMINED"
            })
            if verbose:
                print(f"  ⏭️  SIZE_UNDETERMINED: {task_id}")

        except Exception as e:
            results["ERROR"].append({
                "task_id": task_id,
                "error": str(e)
            })
            if verbose:
                print(f"  ❌ ERROR: {task_id}: {str(e)[:80]}")

    return results


def print_summary(results):
    """Print sweep summary."""
    print("\n" + "="*70)
    print("M2 SWEEP SUMMARY (H1-7)")
    print("="*70)

    total = sum(len(v) for v in results.values())

    print(f"\n✅ SOLVED_BY_UNANIMITY: {len(results['SOLVED_BY_UNANIMITY'])}/{total}")
    print(f"⏳ NOT_YET_NORMALIZATION: {len(results['NOT_YET_NORMALIZATION'])}/{total}")
    print(f"⏳ NOT_YET_DISAGREEING_OUTPUTS: {len(results['NOT_YET_DISAGREEING_OUTPUTS'])}/{total}")
    print(f"⏳ NOT_YET_OTHER: {len(results['NOT_YET_OTHER'])}/{total}")
    print(f"⏭️  SIZE_UNDETERMINED: {len(results['SIZE_UNDETERMINED'])}/{total}")
    print(f"🐛 BUG_TRANSPORT_NORMALIZE: {len(results['BUG_TRANSPORT_NORMALIZE'])}/{total}")
    print(f"🐛 BUG_RECEIPT_COUNTING: {len(results['BUG_RECEIPT_COUNTING'])}/{total}")
    print(f"- NO_GT: {len(results['NO_GT'])}/{total}")
    print(f"❌ ERROR: {len(results['ERROR'])}/{total}")

    # Show solved tasks
    if results["SOLVED_BY_UNANIMITY"]:
        print(f"\n{'-'*70}")
        print(f"SOLVED TASKS ({len(results['SOLVED_BY_UNANIMITY'])}):")
        print(f"{'-'*70}")
        for entry in results["SOLVED_BY_UNANIMITY"][:20]:  # Show first 20
            task_id = entry["task_id"]
            cov = entry["unanimity_coverage"]
            n_inc = entry["n_included"]
            print(f"  {task_id}: coverage={cov:.1%}, n_included={n_inc}")

        if len(results["SOLVED_BY_UNANIMITY"]) > 20:
            print(f"  ... and {len(results['SOLVED_BY_UNANIMITY']) - 20} more")

    # Show bugs
    if results["BUG_TRANSPORT_NORMALIZE"]:
        print(f"\n{'-'*70}")
        print(f"BUG: TRANSPORT_NORMALIZE (full unanimity but mismatch) - {len(results['BUG_TRANSPORT_NORMALIZE'])}:")
        print(f"{'-'*70}")
        for entry in results["BUG_TRANSPORT_NORMALIZE"][:10]:
            task_id = entry["task_id"]
            n_inc = entry["n_included"]
            print(f"  {task_id}: n_included={n_inc}, unanimous={entry['unanimous_pixels']}/{entry['total_pixels']}")

    if results["BUG_RECEIPT_COUNTING"]:
        print(f"\n{'-'*70}")
        print(f"BUG: RECEIPT_COUNTING - {len(results['BUG_RECEIPT_COUNTING'])}:")
        print(f"{'-'*70}")
        for entry in results["BUG_RECEIPT_COUNTING"][:10]:
            print(f"  {entry['task_id']}: {entry['bug_reason']}")

    # Show NOT_YET breakdown
    print(f"\n{'-'*70}")
    print(f"NOT YET BREAKDOWN (expected at M2):")
    print(f"{'-'*70}")
    print(f"  Normalization fails: {len(results['NOT_YET_NORMALIZATION'])} (no integer relation)")
    print(f"  Disagreeing outputs: {len(results['NOT_YET_DISAGREEING_OUTPUTS'])} (trainings conflict)")
    print(f"  Other: {len(results['NOT_YET_OTHER'])} (miscellaneous)")

    # Show errors
    if results["ERROR"]:
        print(f"\n{'-'*70}")
        print(f"ERRORS ({len(results['ERROR'])}):")
        print(f"{'-'*70}")
        for entry in results["ERROR"][:10]:
            print(f"  {entry['task_id']}: {entry['error'][:80]}")


def save_results(results, output_path="/tmp/m2_sweep_h1_7_results.json"):
    """Save detailed results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    print("M2 SWEEP - H1-7 TASKS (OUTPUT PATH ONLY)")
    print("="*70)

    # Check if we should use M1' passing tasks or all tasks
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("Testing ALL tasks...")
        task_ids = sorted(all_tasks.keys())
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
        print(f"Testing first {n} tasks...")
        task_ids = sorted(all_tasks.keys())[:n]
    else:
        # Try to load M1' passing tasks
        task_ids = load_m1_prime_passing_tasks()
        if task_ids:
            print(f"Testing {len(task_ids)} tasks that passed M1' with H1-7...")
        else:
            # Default: first 100 tasks
            print("Testing first 100 tasks...")
            task_ids = sorted(all_tasks.keys())[:100]

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Running sweep on {len(task_ids)} tasks...\n")

    results = run_sweep(task_ids, verbose=verbose)

    print_summary(results)

    save_results(results)

    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)
