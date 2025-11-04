#!/usr/bin/env python3
"""
M3 Sweep on H1-7 Tasks (861 tasks that passed M1')

Classifies each task using receipts:
  - SOLVED_BY_WITNESS: Y_out == ground truth
  - NOT_YET: scope_bits == 0 (needs M4/M5)
  - BUG_WITNESS_CONJ: full witness coverage but mismatch
  - BUG_WITNESS_LEARN: partial witness coverage
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.runner import solve


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
    Classify task result using receipts (receipts-driven classification).

    Returns: (status, details dict)
      status ∈ {SOLVED_BY_WITNESS, NOT_YET, BUG_WITNESS_CONJ, BUG_WITNESS_LEARN, NO_GT}
    """
    if Y_expected is None:
        return "NO_GT", {}

    # Exact match check
    exact_match = (Y_out == Y_expected)

    # Extract receipt signals
    selection = receipts["payload"]["selection"]
    witness_learns = receipts["payload"]["witness_learn"]
    witness_emit = receipts["payload"]["witness_emit"]

    R_out = len(Y_out)
    C_out = len(Y_out[0]) if Y_out else 0
    total_pixels = R_out * C_out

    witness_pixels = selection["counts"]["witness"]
    bottom_pixels = selection["counts"]["bottom"]

    # Get scope_bits from witness_emit (more authoritative than selection counts)
    scope_bits = witness_emit["scopes"]["scope_bits"]

    # Compute witness coverage
    witness_coverage = witness_pixels / total_pixels if total_pixels > 0 else 0

    # Check sigma and overlap status
    all_bijection_ok = all(wl["sigma_bijection_ok"] for wl in witness_learns)
    any_overlap_conflict = any(wl["overlap_conflict"] for wl in witness_learns)
    num_silent = sum(1 for wl in witness_learns if wl["silent"])

    details = {
        "match": exact_match,
        "witness_pixels": witness_pixels,
        "scope_bits": scope_bits,
        "total_pixels": total_pixels,
        "witness_coverage": witness_coverage,
        "sigma_bijection_ok": all_bijection_ok,
        "overlap_conflict": any_overlap_conflict,
        "num_silent": num_silent,
        "num_trainings": len(witness_learns),
    }

    # Classification logic (per M3 spec)
    if exact_match:
        return "SOLVED_BY_WITNESS", details

    # Check if witness is absent (needs M4/M5) - use scope_bits
    if scope_bits == 0:
        return "NOT_YET", details

    # Check if witness determined full canvas (bug if mismatch)
    if witness_pixels == total_pixels and all_bijection_ok and not any_overlap_conflict:
        return "BUG_WITNESS_CONJ", details

    # Partial witness coverage (learning issue)
    return "BUG_WITNESS_LEARN", details


def run_sweep(task_ids=None, max_tasks=None, verbose=False):
    """Run M3 sweep on specified tasks."""

    if task_ids is None:
        task_ids = sorted(all_tasks.keys())

    if max_tasks:
        task_ids = task_ids[:max_tasks]

    results = {
        "SOLVED_BY_WITNESS": [],
        "NOT_YET": [],
        "BUG_WITNESS_CONJ": [],
        "BUG_WITNESS_LEARN": [],
        "NO_GT": [],
        "ERROR": [],
    }

    for i, task_id in enumerate(task_ids):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(task_ids)}] processed...")

        task = all_tasks[task_id]

        try:
            # Run solver with H1-7
            Y_out, receipts = solve(
                task,
                families=("H1", "H2", "H3", "H4", "H5", "H6", "H7"),
                with_witness=True,
                with_unanimity=False
            )

            # Get ground truth (first test output)
            Y_expected = all_solutions.get(task_id, [None])[0]

            # Classify
            status, details = classify_task_result(task_id, Y_out, Y_expected, receipts)

            results[status].append({
                "task_id": task_id,
                **details
            })

            if verbose and status in ["SOLVED_BY_WITNESS", "BUG_WITNESS_CONJ"]:
                print(f"  {status}: {task_id}")

        except Exception as e:
            results["ERROR"].append({
                "task_id": task_id,
                "error": str(e)
            })
            if verbose:
                print(f"  ERROR: {task_id}: {str(e)}")

    return results


def print_summary(results):
    """Print sweep summary."""
    print("\n" + "="*70)
    print("M3 SWEEP SUMMARY (H1-7)")
    print("="*70)

    total = sum(len(v) for v in results.values())

    print(f"\n✅ SOLVED_BY_WITNESS: {len(results['SOLVED_BY_WITNESS'])}/{total}")
    print(f"⏳ NOT_YET (needs M4/M5): {len(results['NOT_YET'])}/{total}")
    print(f"🐛 BUG_WITNESS_CONJ: {len(results['BUG_WITNESS_CONJ'])}/{total}")
    print(f"🐛 BUG_WITNESS_LEARN: {len(results['BUG_WITNESS_LEARN'])}/{total}")
    print(f"- NO_GT: {len(results['NO_GT'])}/{total}")
    print(f"❌ ERROR: {len(results['ERROR'])}/{total}")

    # Show solved tasks
    if results["SOLVED_BY_WITNESS"]:
        print(f"\n{'-'*70}")
        print(f"SOLVED TASKS ({len(results['SOLVED_BY_WITNESS'])}):")
        print(f"{'-'*70}")
        for entry in results["SOLVED_BY_WITNESS"][:20]:  # Show first 20
            task_id = entry["task_id"]
            cov = entry["witness_coverage"]
            print(f"  {task_id}: witness_coverage={cov:.1%}")

        if len(results["SOLVED_BY_WITNESS"]) > 20:
            print(f"  ... and {len(results['SOLVED_BY_WITNESS']) - 20} more")

    # Show bugs
    if results["BUG_WITNESS_CONJ"]:
        print(f"\n{'-'*70}")
        print(f"BUG: WITNESS_CONJ (full coverage but mismatch) - {len(results['BUG_WITNESS_CONJ'])}:")
        print(f"{'-'*70}")
        for entry in results["BUG_WITNESS_CONJ"][:10]:
            task_id = entry["task_id"]
            σ_ok = entry["sigma_bijection_ok"]
            overlap = entry["overlap_conflict"]
            print(f"  {task_id}: σ_bijection={σ_ok}, overlap={overlap}")

    if results["BUG_WITNESS_LEARN"]:
        print(f"\n{'-'*70}")
        print(f"BUG: WITNESS_LEARN (partial coverage) - {len(results['BUG_WITNESS_LEARN'])} (showing sample):")
        print(f"{'-'*70}")
        for entry in results["BUG_WITNESS_LEARN"][:10]:
            task_id = entry["task_id"]
            cov = entry["witness_coverage"]
            silent = entry["num_silent"]
            print(f"  {task_id}: witness_coverage={cov:.1%}, silent={silent}/{entry['num_trainings']}")

    # Show errors
    if results["ERROR"]:
        print(f"\n{'-'*70}")
        print(f"ERRORS ({len(results['ERROR'])}):")
        print(f"{'-'*70}")
        for entry in results["ERROR"][:10]:
            print(f"  {entry['task_id']}: {entry['error']}")


def save_results(results, output_path="/tmp/m3_sweep_h1_7_results.json"):
    """Save detailed results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    print("M3 SWEEP - H1-7 TASKS")
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
