#!/usr/bin/env python3
"""Get fresh receipts from both bug tasks."""

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

bug_tasks = ['4522001f', 'bdad9b1f']

for task_id in bug_tasks:
    print(f"Generating fresh receipt for {task_id}...")

    task_data = all_tasks[task_id]
    GT = all_solutions[task_id][0]

    try:
        Y_out, receipts = solve(
            task_data,
            families=('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7'),
            with_witness=False
        )

        # Save full receipt
        receipt_path = f"/tmp/receipt_{task_id}_latest.json"
        with open(receipt_path, "w") as f:
            json.dump(receipts, f, indent=2)
        print(f"  ✅ Saved to {receipt_path}")

        # Save compact issue summary
        payload = receipts['payload']
        compact = {
            "task_id": task_id,
            "problem": "repaint_hash != unanimity_hash",
            "working_canvas": {
                "R_out": payload['working_canvas']['R_out'],
                "C_out": payload['working_canvas']['C_out']
            },
            "transports": {
                "n_included": payload['transports']['n_included'],
                "transports": payload['transports']['transports']
            },
            "unanimity": {
                "included_train_ids": payload['unanimity']['included_train_ids'],
                "unanimous_pixels": payload['unanimity']['unanimous_pixels'],
                "total_covered_pixels": payload['unanimity']['total_covered_pixels'],
                "unanimity_hash": payload['unanimity']['unanimity_hash']
            },
            "selection": {
                "precedence": payload['selection']['precedence'],
                "counts": payload['selection']['counts'],
                "repaint_hash": payload['selection']['repaint_hash']
            },
            "hash_mismatch": {
                "unanimity_hash": payload['unanimity']['unanimity_hash'],
                "repaint_hash": payload['selection']['repaint_hash'],
                "match": payload['unanimity']['unanimity_hash'] == payload['selection']['repaint_hash']
            },
            "training_outputs": [task_data['train'][i]['output'] for i in range(len(task_data['train']))],
            "Y_out": Y_out,
            "GT": GT,
            "Y_out_matches_GT": (Y_out == GT)
        }

        compact_path = f"/tmp/compact_{task_id}_latest.json"
        with open(compact_path, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"  ✅ Saved compact to {compact_path}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n✅ Fresh receipts generated")
