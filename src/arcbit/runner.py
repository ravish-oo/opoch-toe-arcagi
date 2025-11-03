"""
ARC-AGI Deterministic Runner

Milestone-based runner that evolves incrementally:
  - M0 (current): Bedrock validation (color universe, pack/unpack, canonicalize)
  - M1+: Size prediction, emitters, LFP, selection (to be added)

Current M0 Implementation:
  - Wires foundational modules (WO-00, WO-01, WO-03)
  - Proves byte-level correctness of color-plane packing/unpacking
  - Proves frame canonicalization (D4 lex-min + anchor)
  - Output: Y_placeholder = X* (identity), plus receipts bundle
  - No solving yet (validation only)

Future sections will be appended below M0 sections as milestones progress.
"""

from typing import Tuple, Dict, List
from .core import Receipts, blake3_hash
from .core.bytesio import serialize_grid_be_row_major, serialize_planes_be_row_major
from .kernel import (
    order_colors,
    pack_grid_to_planes,
    unpack_planes_to_grid,
    canonicalize,
    apply_pose_anchor
)


def solve(task_json: dict) -> Tuple[List[List[int]], Dict]:
    """
    ARC-AGI deterministic solver (milestone-based).

    Current M0 Implementation:
      Returns test input unchanged as Y_placeholder (validation only).
      Steps:
        1) Build color universe C = {0} ∪ colors(X*) ∪ ⋃colors(X_i,Y_i)
        2) PACK→UNPACK identity check on all trainings and on X*
        3) Canonicalize frames (Π_in for all X_i and X*, Π_out for all Y_i)
        4) apply_pose_anchor round-trip proof on X* planes
        5) Seal receipts (no timestamps)

    Future Milestones (to be added):
      - M1+: Size prediction, emitters, LFP, selection

    Args:
        task_json: ARC task dict with keys:
            - "train": list of {"input": grid, "output": grid}
            - "test": list of {"input": grid}

    Returns:
        Tuple of (Y, receipts_bundle):
          - Y: Predicted output grid (M0: returns X* unchanged)
          - receipts_bundle: dict with sectioned receipts

    Invariants (all milestones):
        - No heuristics, no floats, no RNG
        - No minted non-zero bits
        - Color exclusivity preserved
        - Receipts-first: every step logs BLAKE3 hashes
        - Deterministic: double-run produces identical hashes

    Raises:
        ValueError: On shape mismatch, bit overflow, bad pose ID
        RuntimeError: On determinism check failure (double-run mismatch)
    """
    # Extract grids
    train_pairs = task_json.get("train", [])
    test_inputs = task_json.get("test", [])

    if len(test_inputs) == 0:
        raise ValueError("Runner: task_json must have at least one test input")

    # X* = first test input
    X_star = test_inputs[0]["input"]

    # ========================================================================
    # M0 Step 1: Color Universe (A.1)
    # ========================================================================

    # Collect all colors from test input, training inputs/outputs
    color_set = {0}  # Always include background

    # Add colors from X*
    for row in X_star:
        for val in row:
            color_set.add(val)

    # Add colors from training pairs
    for pair in train_pairs:
        X_i = pair["input"]
        Y_i = pair["output"]

        for row in X_i:
            for val in row:
                color_set.add(val)

        for row in Y_i:
            for val in row:
                color_set.add(val)

    # Sort to get colors_order (ascending ints)
    colors_order = sorted(color_set)

    # Track which colors came from test (for receipts)
    test_colors = set()
    for row in X_star:
        for val in row:
            test_colors.add(val)

    added_from_test = sorted(test_colors - {0})

    # Build receipts
    receipts = Receipts("M0-bedrock")  # Section label (will evolve with milestones)

    receipts.put("color_universe.colors_order", colors_order)
    receipts.put("color_universe.added_from_test", added_from_test)
    receipts.put("color_universe.K", len(colors_order))

    # ========================================================================
    # M0 Step 2: PACK↔UNPACK Identity (WO-01)
    # ========================================================================

    pack_unpack_receipts = []

    # Check all training pairs + X*
    all_grids = []
    for i, pair in enumerate(train_pairs):
        all_grids.append(("train", i, "input", pair["input"]))
        all_grids.append(("train", i, "output", pair["output"]))

    all_grids.append(("test", 0, "input", X_star))

    for (split, idx, io_type, G) in all_grids:
        H = len(G)
        W = len(G[0]) if G else 0

        # Pack to planes
        planes = pack_grid_to_planes(G, H, W, colors_order)

        # Unpack back to grid
        G2 = unpack_planes_to_grid(planes, H, W, colors_order)

        # Assert identity
        if G2 != G:
            raise ValueError(
                f"Runner: PACK↔UNPACK identity failed for {split}[{idx}].{io_type}: "
                f"G != G2 (shapes: {H}x{W})"
            )

        # Compute hashes (payload only, skip magic number to compare content)
        grid_bytes = serialize_grid_be_row_major(G, H, W, colors_order)
        planes_bytes = serialize_planes_be_row_major(planes, H, W, colors_order)

        # Header size: tag (4) + H (2) + W (2) + K (1) + colors (K)
        K = len(colors_order)
        header_size = 4 + 2 + 2 + 1 + K

        # Extract payloads (skip headers with different magic numbers)
        grid_payload = grid_bytes[header_size:]
        planes_payload = planes_bytes[header_size:]

        # Hash payloads (content equivalence)
        hash_grid_payload = blake3_hash(grid_payload)
        hash_planes_payload = blake3_hash(planes_payload)

        # Hash equality check (serialization equivalence theorem)
        pack_equal = (hash_grid_payload == hash_planes_payload)

        # Also keep full hashes for audit
        hash_grid = blake3_hash(grid_bytes)
        hash_planes = blake3_hash(planes_bytes)

        pack_unpack_receipts.append({
            "split": split,
            "idx": idx,
            "io_type": io_type,
            "H": H,
            "W": W,
            "hash_grid": hash_grid,
            "hash_planes": hash_planes,
            "pack_equal": pack_equal
        })

        if not pack_equal:
            raise ValueError(
                f"Runner: Hash mismatch for {split}[{idx}].{io_type}: "
                f"hash_grid != hash_planes (serialization equivalence violated)"
            )

    receipts.put("pack_unpack", pack_unpack_receipts)

    # ========================================================================
    # M0 Step 3: Canonical Frames (WO-03; B.1)
    # ========================================================================

    frames_receipts = []

    for (split, idx, io_type, G) in all_grids:
        H = len(G)
        W = len(G[0]) if G else 0

        # Canonicalize
        pid, anchor, G_canon, canon_receipts = canonicalize(G)

        # Re-canonicalize to check idempotence
        pid2, anchor2, G_canon2, _ = canonicalize(G_canon)

        # Idempotence theorem: C(C(G)) = ("I", (0,0), C(G))
        idempotent = (pid2 == "I" and anchor2 == (0, 0) and G_canon2 == G_canon)

        if not idempotent:
            raise ValueError(
                f"Runner: Canonicalize idempotence failed for {split}[{idx}].{io_type}: "
                f"pid2={pid2}, anchor2={anchor2} (expected: I, (0,0))"
            )

        # Extract key fields from canonicalize receipts
        frame_receipt = {
            "split": split,
            "idx": idx,
            "io_type": io_type,
            "frame.inputs": canon_receipts["frame.inputs"],
            "frame.pose": canon_receipts["frame.pose"],
            "frame.anchor": canon_receipts["frame.anchor"],
            "frame.bytes": canon_receipts["frame.bytes"],
            "idempotent": idempotent
        }

        frames_receipts.append(frame_receipt)

    receipts.put("frames.canonicalize", frames_receipts)

    # ========================================================================
    # M0 Step 4: Frame Apply Equivalence (WO-03; X* only)
    # ========================================================================

    # For X* only: prove apply_pose_anchor(planes) == canonicalize(grid)
    H_star = len(X_star)
    W_star = len(X_star[0]) if X_star else 0

    # Get canonical frame for X*
    pid_star, anchor_star, G_canon_star, _ = canonicalize(X_star)

    # Pack X* to planes
    planes_star = pack_grid_to_planes(X_star, H_star, W_star, colors_order)

    # Apply pose + anchor to planes
    planes_transformed, H_trans, W_trans = apply_pose_anchor(
        planes_star, pid_star, anchor_star, H_star, W_star, colors_order
    )

    # Unpack transformed planes to grid
    G_from_planes = unpack_planes_to_grid(planes_transformed, H_trans, W_trans, colors_order)

    # Transformation equivalence theorem: unpack(apply(pack(X*))) = C(X*)
    equivalence_ok = (G_from_planes == G_canon_star)

    if not equivalence_ok:
        raise ValueError(
            f"Runner: apply_pose_anchor equivalence failed for X*: "
            f"G_from_planes != G_canon_star (shapes: {H_trans}x{W_trans} vs {len(G_canon_star)}x{len(G_canon_star[0])})"
        )

    # Compute hashes for cross-check (payload only, skip magic numbers)
    grid_from_planes_bytes = serialize_grid_be_row_major(
        G_from_planes, H_trans, W_trans, colors_order
    )

    H_canon = len(G_canon_star)
    W_canon = len(G_canon_star[0]) if G_canon_star else 0
    grid_canon_bytes = serialize_grid_be_row_major(G_canon_star, H_canon, W_canon, colors_order)

    # Header size: tag (4) + H (2) + W (2) + K (1) + colors (K)
    K_star = len(colors_order)
    header_size = 4 + 2 + 2 + 1 + K_star

    # Extract payloads
    grid_from_planes_payload = grid_from_planes_bytes[header_size:]
    grid_canon_payload = grid_canon_bytes[header_size:]

    # Hash payloads (content equivalence)
    hash_grid_from_planes_payload = blake3_hash(grid_from_planes_payload)
    hash_grid_canon_payload = blake3_hash(grid_canon_payload)

    hash_equal = (hash_grid_from_planes_payload == hash_grid_canon_payload)

    # Also keep full hashes for audit
    hash_grid_from_planes = blake3_hash(grid_from_planes_bytes)
    hash_grid_canon = blake3_hash(grid_canon_bytes)

    apply_receipt = {
        "split": "test",
        "idx": 0,
        "apply.pose_id": pid_star,
        "apply.anchor": list(anchor_star),
        "shape_before": [H_star, W_star],
        "shape_after": [H_trans, W_trans],
        "hash_grid_from_planes": hash_grid_from_planes,
        "hash_grid_canon": hash_grid_canon,
        "hash_equal": hash_equal,
        "equivalence_ok": equivalence_ok
    }

    if not hash_equal:
        raise ValueError(
            f"Runner: Hash mismatch after apply_pose_anchor: "
            f"hash_grid_from_planes != hash_grid_canon"
        )

    receipts.put("frames.apply_pose_anchor", apply_receipt)

    # ========================================================================
    # M0 Step 5: Seal receipts and return
    # ========================================================================

    # Seal receipts (no timestamps)
    receipts_bundle = receipts.digest()

    # Y_placeholder = X* (identity)
    Y_placeholder = X_star

    return (Y_placeholder, receipts_bundle)


def solve_with_determinism_check(task_json: dict) -> Tuple[List[List[int]], Dict]:
    """
    Run solve() twice and verify determinism (all section hashes match).

    Args:
        task_json: ARC task dict.

    Returns:
        Tuple of (Y, receipts_bundle) with determinism flags added.

    Raises:
        RuntimeError: If double-run hashes differ.

    Spec:
        Determinism check (A.3): Verifies referential transparency.
        Applicable to all milestones.
    """
    # Run 1
    Y1, receipts1 = solve(task_json)

    # Run 2
    Y2, receipts2 = solve(task_json)

    # Compare Y (must be identical)
    if Y1 != Y2:
        raise RuntimeError("M0: Determinism check failed: Y1 != Y2")

    # Compare all section hashes
    # Extract all keys from receipts
    all_keys = set(receipts1.keys()) | set(receipts2.keys())

    for key in all_keys:
        if key not in receipts1:
            raise RuntimeError(f"Runner: Determinism check failed: key '{key}' missing in run 1")
        if key not in receipts2:
            raise RuntimeError(f"Runner: Determinism check failed: key '{key}' missing in run 2")

        val1 = receipts1[key]
        val2 = receipts2[key]

        if val1 != val2:
            raise RuntimeError(
                f"Runner: Determinism check failed: receipts differ at key '{key}'\n"
                f"  Run 1: {val1}\n"
                f"  Run 2: {val2}"
            )

    # Add determinism summary to receipts
    receipts_final = receipts1.copy()
    receipts_final["determinism.double_run_ok"] = True
    receipts_final["determinism.sections_checked"] = len(all_keys)

    return (Y1, receipts_final)
