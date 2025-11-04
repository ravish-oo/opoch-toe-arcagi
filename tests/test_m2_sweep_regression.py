#!/usr/bin/env python3
"""
M2 Sweep with Regression Detection

Runs M2 on all 861 H1-7 tasks and compares with previous results.
Uses CORRECT hash comparison: unanimity_grid_hash vs repaint_hash
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
                parts = line.strip().split()
                if len(parts) >= 2:
                    task_id = parts[1].rstrip(":")
                    passing.append(task_id)

    return passing


def classify_task_result(task_id, Y_out, Y_expected, receipts):
    """
    Classify task result using receipts (M2 taxonomy).

    CORRECTED: Uses unanimity_grid_hash (not unanimity_hash) for comparison.

    Returns: (status, details dict)
    """
    try:
        # Extract receipt fields
        canvas = receipts['payload']['working_canvas']
        transports = receipts['payload']['transports']
        unanimity = receipts['payload']['unanimity']
        selection = receipts['payload']['selection']

        R_out = canvas['R_out']
        C_out = canvas['C_out']
        total_pixels = R_out * C_out

        n_trainings = canvas['num_trainings']
        n_included = transports['n_included']

        unanimous_pixels = unanimity['unanimous_pixels']
        total_covered = unanimity['total_covered_pixels']
        unanimity_hash_planes = unanimity.get('unanimity_hash')  # Planes encoding
        unanimity_grid_hash = unanimity.get('unanimity_grid_hash')  # Grid encoding (CORRECT)

        counts = selection['counts']
        repaint_hash = selection['repaint_hash']

        # Check exact match
        if Y_expected is None:
            exact_match = None
        else:
            exact_match = (Y_out == Y_expected)

        # Classification logic
        details = {
            'R_out': R_out,
            'C_out': C_out,
            'n_trainings': n_trainings,
            'n_included': n_included,
            'unanimous_pixels': unanimous_pixels,
            'total_covered_pixels': total_covered,
            'counts': counts,
            'exact_match': exact_match,
            'unanimity_hash_planes': unanimity_hash_planes,
            'unanimity_grid_hash': unanimity_grid_hash,
            'repaint_hash': repaint_hash,
        }

        # Check if all trainings normalized
        all_trainings_normalized = (n_included == n_trainings)

        # Check full unanimity
        full_unanimity = (unanimous_pixels == total_pixels)

        # Check hash match (CORRECTED: use unanimity_grid_hash)
        if unanimity_grid_hash:
            hash_match = (unanimity_grid_hash == repaint_hash)
        else:
            hash_match = None

        details['all_trainings_normalized'] = all_trainings_normalized
        details['full_unanimity'] = full_unanimity
        details['hash_match'] = hash_match

        # Classification
        if Y_expected is None:
            return 'NO_GT', details

        # BUG: Full unanimity but hash mismatch (ONLY when all trainings normalized)
        if all_trainings_normalized and full_unanimity and hash_match == False:
            return 'BUG_TRANSPORT_NORMALIZE', details

        # BUG: Counting logic errors
        if unanimous_pixels > total_covered:
            return 'BUG_RECEIPT_COUNTING', details
        if counts['unanimity'] > total_pixels or counts['bottom'] > total_pixels:
            return 'BUG_RECEIPT_COUNTING', details

        # SOLVED: Full unanimity + exact match
        if full_unanimity and exact_match:
            return 'SOLVED_BY_UNANIMITY', details

        # NOT_YET: No trainings normalized
        if n_included == 0:
            return 'NOT_YET_NORMALIZATION', details

        # NOT_YET: Disagreeing outputs (partial unanimity)
        if unanimous_pixels < total_covered:
            return 'NOT_YET_DISAGREEING_OUTPUTS', details

        # NOT_YET: Full unanimity but wrong answer (some trainings silent)
        if full_unanimity and not exact_match:
            if all_trainings_normalized:
                # This should be a bug, but hash match says it's working correctly
                # Likely the hypothesis (H1-7) is wrong for this task
                return 'NOT_YET_WRONG_HYPOTHESIS', details
            else:
                # Some trainings didn't normalize, so can't determine yet
                return 'NOT_YET_PARTIAL_TRAININGS', details

        # Fallback
        return 'NOT_YET_OTHER', details

    except Exception as e:
        return 'ERROR', {'error': str(e)}


def main():
    print("=" * 70)
    print("M2 SWEEP - H1-7 TASKS (OUTPUT PATH ONLY)")
    print("WITH REGRESSION DETECTION")
    print("=" * 70)

    # Load previous results
    old_results_path = Path("/tmp/m2_sweep_h1_7_results.json")
    old_results = {}
    if old_results_path.exists():
        with open(old_results_path, "r") as f:
            old_data = json.load(f)
            for item in old_data.get('results', []):
                old_results[item['task_id']] = item['status']
        print(f"Loaded {len(old_results)} previous results for comparison")
    else:
        print("⚠️  No previous results found - will not check for regressions")

    print()

    # Load passing tasks
    passing_tasks = load_m1_prime_passing_tasks()
    if passing_tasks:
        print(f"Testing {len(passing_tasks)} tasks that passed M1' with H1-7...")
        task_ids = passing_tasks
    else:
        print(f"Testing all {len(all_tasks)} tasks...")
        task_ids = list(all_tasks.keys())

    print()
    print("Running sweep on {} tasks...".format(len(task_ids)))
    print()

    results = []
    taxonomy = {}
    regressions = []
    improvements = []

    for i, task_id in enumerate(task_ids):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(task_ids)}] processed...")

        task = all_tasks[task_id]
        Y_expected = all_solutions.get(task_id, [None])[0] if task_id in all_solutions else None

        try:
            Y_out, receipts = solve(task, families=["H1", "H2", "H3", "H4", "H5", "H6", "H7"])
            status, details = classify_task_result(task_id, Y_out, Y_expected, receipts)
        except SizeUndetermined as e:
            status = 'SIZE_UNDETERMINED'
            details = {'error': str(e)}
        except Exception as e:
            status = 'ERROR'
            details = {'error': str(e)}

        results.append({
            'task_id': task_id,
            'status': status,
            'details': details
        })

        taxonomy[status] = taxonomy.get(status, 0) + 1

        # Check for regression or improvement
        if task_id in old_results:
            old_status = old_results[task_id]
            if old_status != status:
                if status in ['ERROR', 'BUG_TRANSPORT_NORMALIZE', 'BUG_RECEIPT_COUNTING']:
                    regressions.append({
                        'task_id': task_id,
                        'old': old_status,
                        'new': status,
                        'details': details
                    })
                elif old_status in ['ERROR', 'BUG_TRANSPORT_NORMALIZE', 'BUG_RECEIPT_COUNTING']:
                    improvements.append({
                        'task_id': task_id,
                        'old': old_status,
                        'new': status
                    })

    print()
    print("=" * 70)
    print("M2 SWEEP SUMMARY (H1-7)")
    print("=" * 70)
    print()

    # Print taxonomy
    for status in ['SOLVED_BY_UNANIMITY', 'NOT_YET_NORMALIZATION',
                   'NOT_YET_DISAGREEING_OUTPUTS', 'NOT_YET_WRONG_HYPOTHESIS',
                   'NOT_YET_PARTIAL_TRAININGS', 'NOT_YET_OTHER',
                   'SIZE_UNDETERMINED', 'BUG_TRANSPORT_NORMALIZE',
                   'BUG_RECEIPT_COUNTING', 'NO_GT', 'ERROR']:
        count = taxonomy.get(status, 0)
        if count > 0:
            emoji = {
                'SOLVED_BY_UNANIMITY': '✅',
                'NOT_YET_NORMALIZATION': '⏳',
                'NOT_YET_DISAGREEING_OUTPUTS': '⏳',
                'NOT_YET_WRONG_HYPOTHESIS': '⏳',
                'NOT_YET_PARTIAL_TRAININGS': '⏳',
                'NOT_YET_OTHER': '⏳',
                'SIZE_UNDETERMINED': '⏭️',
                'BUG_TRANSPORT_NORMALIZE': '🐛',
                'BUG_RECEIPT_COUNTING': '🐛',
                'NO_GT': '-',
                'ERROR': '❌'
            }.get(status, '?')
            print(f"{emoji} {status}: {count}/{len(task_ids)}")

    print()

    # Print bugs
    bugs = [r for r in results if r['status'] in ['BUG_TRANSPORT_NORMALIZE', 'BUG_RECEIPT_COUNTING']]
    if bugs:
        print("-" * 70)
        print(f"BUG DETAILS ({len(bugs)} bugs found):")
        print("-" * 70)
        for bug in bugs[:20]:  # Show first 20
            task_id = bug['task_id']
            status = bug['status']
            d = bug['details']
            print(f"  {task_id}: {status}")
            print(f"    n_included={d.get('n_included')}/{d.get('n_trainings')}, " +
                  f"unanimous={d.get('unanimous_pixels')}/{d.get('R_out', 0) * d.get('C_out', 0)}, " +
                  f"hash_match={d.get('hash_match')}")
        if len(bugs) > 20:
            print(f"  ... and {len(bugs) - 20} more")
        print()

    # Print regressions
    if regressions:
        print("-" * 70)
        print(f"⚠️  REGRESSIONS DETECTED ({len(regressions)} tasks):")
        print("-" * 70)
        for reg in regressions[:20]:
            print(f"  {reg['task_id']}: {reg['old']} → {reg['new']}")
        if len(regressions) > 20:
            print(f"  ... and {len(regressions) - 20} more")
        print()

    # Print improvements
    if improvements:
        print("-" * 70)
        print(f"✨ IMPROVEMENTS ({len(improvements)} tasks):")
        print("-" * 70)
        for imp in improvements[:20]:
            print(f"  {imp['task_id']}: {imp['old']} → {imp['new']}")
        if len(improvements) > 20:
            print(f"  ... and {len(improvements) - 20} more")
        print()

    # Save results
    output = {
        'total_tasks': len(task_ids),
        'taxonomy': taxonomy,
        'regressions': regressions,
        'improvements': improvements,
        'results': results
    }

    output_path = Path("/tmp/m2_sweep_post_fix_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Detailed results saved to: {output_path}")
    print()
    print("=" * 70)
    print("REGRESSION CHECK:")
    if regressions:
        print(f"❌ {len(regressions)} REGRESSIONS DETECTED")
    else:
        print("✅ NO REGRESSIONS")

    if improvements:
        print(f"✨ {len(improvements)} IMPROVEMENTS")
    print("=" * 70)


if __name__ == "__main__":
    main()
