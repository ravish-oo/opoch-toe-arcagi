#!/usr/bin/env python3
"""
M2 Pixel Match Analysis

Analyzes pixel match % between M2 outputs and ground truth.
Categorizes results into buckets: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%, 100%
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


def calculate_pixel_match(Y_out, Y_expected):
    """
    Calculate pixel match percentage between output and expected.

    Returns: (match_percent, total_pixels, matching_pixels)
    """
    if Y_expected is None:
        return None, None, None

    # Check shape match
    if len(Y_out) != len(Y_expected):
        return 0.0, len(Y_expected) * len(Y_expected[0]) if Y_expected else 0, 0

    if len(Y_out) > 0 and len(Y_out[0]) != len(Y_expected[0]):
        return 0.0, len(Y_expected) * len(Y_expected[0]), 0

    # Count matching pixels
    total = 0
    matching = 0

    for r in range(len(Y_expected)):
        for c in range(len(Y_expected[0])):
            total += 1
            if r < len(Y_out) and c < len(Y_out[0]) and Y_out[r][c] == Y_expected[r][c]:
                matching += 1

    match_percent = (matching / total * 100) if total > 0 else 0.0
    return match_percent, total, matching


def get_bucket(match_percent):
    """Categorize match percentage into buckets."""
    if match_percent is None:
        return "NO_GT"
    elif match_percent == 100.0:
        return "100%"
    elif match_percent >= 80.0:
        return "80-99%"
    elif match_percent >= 60.0:
        return "60-79%"
    elif match_percent >= 40.0:
        return "40-59%"
    elif match_percent >= 20.0:
        return "20-39%"
    else:
        return "0-19%"


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


def main():
    print("=" * 70)
    print("M2 PIXEL MATCH ANALYSIS")
    print("=" * 70)
    print()

    # Load passing tasks
    passing_tasks = load_m1_prime_passing_tasks()
    if passing_tasks:
        print(f"Analyzing {len(passing_tasks)} tasks that passed M1' with H1-7...")
        task_ids = passing_tasks
    else:
        print(f"Analyzing all {len(all_tasks)} tasks...")
        task_ids = list(all_tasks.keys())

    print()
    print("Running M2 and calculating pixel match %...")
    print()

    results = []
    buckets = {
        "100%": [],
        "80-99%": [],
        "60-79%": [],
        "40-59%": [],
        "20-39%": [],
        "0-19%": [],
        "NO_GT": []
    }

    for i, task_id in enumerate(task_ids):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(task_ids)}] processed...")

        task = all_tasks[task_id]
        Y_expected = all_solutions.get(task_id, [None])[0] if task_id in all_solutions else None

        try:
            Y_out, receipts = solve(task, families=["H1", "H2", "H3", "H4", "H5", "H6", "H7"])
            match_percent, total_pixels, matching_pixels = calculate_pixel_match(Y_out, Y_expected)

            bucket = get_bucket(match_percent)

            result = {
                'task_id': task_id,
                'match_percent': match_percent,
                'total_pixels': total_pixels,
                'matching_pixels': matching_pixels,
                'bucket': bucket
            }

            results.append(result)
            buckets[bucket].append(task_id)

        except Exception as e:
            result = {
                'task_id': task_id,
                'error': str(e),
                'bucket': 'ERROR'
            }
            results.append(result)
            if 'ERROR' not in buckets:
                buckets['ERROR'] = []
            buckets['ERROR'].append(task_id)

    print()
    print("=" * 70)
    print("PIXEL MATCH DISTRIBUTION")
    print("=" * 70)
    print()

    # Print distribution
    bucket_order = ["100%", "80-99%", "60-79%", "40-59%", "20-39%", "0-19%", "NO_GT", "ERROR"]

    for bucket in bucket_order:
        if bucket in buckets:
            count = len(buckets[bucket])
            percent = (count / len(task_ids) * 100) if len(task_ids) > 0 else 0

            # Emoji based on bucket
            if bucket == "100%":
                emoji = "✅"
            elif bucket in ["80-99%", "60-79%"]:
                emoji = "🟢"
            elif bucket in ["40-59%"]:
                emoji = "🟡"
            elif bucket in ["20-39%"]:
                emoji = "🟠"
            elif bucket in ["0-19%"]:
                emoji = "🔴"
            elif bucket == "NO_GT":
                emoji = "⚪"
            else:
                emoji = "❌"

            bar_length = int(percent / 2)  # Scale to 50 chars max
            bar = "█" * bar_length

            print(f"{emoji} {bucket:>8}: {count:4}/{len(task_ids)} ({percent:5.1f}%) {bar}")

    print()

    # Print some examples from each bucket
    print("-" * 70)
    print("SAMPLE TASKS BY BUCKET:")
    print("-" * 70)
    for bucket in bucket_order:
        if bucket in buckets and len(buckets[bucket]) > 0:
            sample_tasks = buckets[bucket][:5]  # Show first 5
            print(f"\n{bucket} ({len(buckets[bucket])} tasks):")
            for task_id in sample_tasks:
                result = [r for r in results if r['task_id'] == task_id][0]
                if 'match_percent' in result and result['match_percent'] is not None:
                    print(f"  {task_id}: {result['match_percent']:.1f}% " +
                          f"({result['matching_pixels']}/{result['total_pixels']} pixels)")
                else:
                    print(f"  {task_id}")
            if len(buckets[bucket]) > 5:
                print(f"  ... and {len(buckets[bucket]) - 5} more")

    print()

    # Statistics
    valid_results = [r for r in results if r.get('match_percent') is not None]
    if valid_results:
        match_percents = [r['match_percent'] for r in valid_results]
        avg_match = sum(match_percents) / len(match_percents)
        median_match = sorted(match_percents)[len(match_percents) // 2]

        print("-" * 70)
        print("STATISTICS:")
        print("-" * 70)
        print(f"Average pixel match: {avg_match:.2f}%")
        print(f"Median pixel match:  {median_match:.2f}%")
        print(f"Perfect matches:     {len(buckets['100%'])}")
        print(f"High quality (≥80%): {len(buckets['100%']) + len(buckets.get('80-99%', []))}")
        print(f"Good (≥60%):         {len(buckets['100%']) + len(buckets.get('80-99%', [])) + len(buckets.get('60-79%', []))}")

    print()

    # Save results
    output = {
        'total_tasks': len(task_ids),
        'buckets': {k: len(v) for k, v in buckets.items()},
        'bucket_tasks': buckets,
        'results': results
    }

    output_path = Path("/tmp/m2_pixel_match_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Detailed results saved to: {output_path}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
