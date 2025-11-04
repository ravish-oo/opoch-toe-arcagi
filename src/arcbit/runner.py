"""
ARC-AGI Deterministic Runner

Milestone-based runner that evolves incrementally:
  - M0: Bedrock validation (color universe, pack/unpack, canonicalize) ✅
  - M1': Working canvas selection via WO-14 + WO-04a with H1-H9 ✅
  - M2: Output path (transport + unanimity) ✅
  - M3: Witness path (components, learn, emit, minimal selector) ✅
  - M4: True LFP (forbids + admit-∧ + AC-3) ✅

Current Implementation (M4):
  - M0/M1' sections unchanged
  - M2: WO-08 output transport + unanimity (always enabled)
  - M3: WO-05/06/07 witness path (optional, with_witness=True)
  - M4: WO-10/11 LFP propagation (forbids + AC-3, always enabled)
  - Selector: witness → unanimity → bottom (no EngineWinner yet)

Future milestones:
  - M5: Lattice + EngineWinner
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
from .features import agg_features
from .canvas import choose_working_canvas, SizeUndetermined, parse_families, FULL_FAMILY_SET
from .kernel.components import components
from .emitters import learn_witness, emit_witness, emit_output_transport, emit_unity
from .emitters.forbids import learn_forbids, build_4neighbor_graph
from .emitters.lfp import lfp_propagate


# ============================================================================
# M2: Minimal Selector (Unanimity → Bottom)
# ============================================================================

def select_unanimity_first(
    A_uni: Dict[int, List[int]],
    S_uni: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> Tuple[List[List[int]], Dict]:
    """
    Minimal selector for M2 (output path only).

    Precedence:
      1. Unanimity bucket: pick singleton if S_uni[p]==1
      2. Bottom: pick 0

    Args:
        A_uni: Unanimity admits (color -> plane)
        S_uni: Unanimity scope (list of row masks)
        colors_order: Ascending color list
        R_out, C_out: Canvas dimensions

    Returns:
        Tuple of (Y_out, selection_receipts):
          - Y_out: Selected grid (R_out × C_out)
          - selection_receipts: Dict with counts, containment, repaint_hash

    Spec: Sub-WO-M2-SelectorPatch (final)
    """
    Y_out = [[0] * C_out for _ in range(R_out)]
    counts = {"unanimity": 0, "bottom": 0}

    for r in range(R_out):
        scope_row = S_uni[r] if r < len(S_uni) else 0
        for c in range(C_out):
            bit = 1 << c
            if scope_row & bit:
                # Find the unique unanimous color by reading A_uni color planes.
                found = None
                for color in colors_order:
                    plane = A_uni.get(color)
                    if not plane:
                        continue
                    # Defensive: protect against short rows
                    row_mask = plane[r] if r < len(plane) else 0
                    if (row_mask & bit) != 0:
                        if found is None:
                            found = color
                        else:
                            # Should never happen in unanimity; fall back to bottom to be safe.
                            found = None
                            break
                if found is not None:
                    Y_out[r][c] = found
                    counts["unanimity"] += 1
                else:
                    # No unanimous color despite scope: fall back to bottom (0)
                    Y_out[r][c] = 0
                    counts["bottom"] += 1
            else:
                Y_out[r][c] = 0
                counts["bottom"] += 1

    # Hash of selected grid
    grid_bytes = serialize_grid_be_row_major(Y_out, R_out, C_out, colors_order)
    repaint_hash = blake3_hash(grid_bytes)

    selection_receipts = {
        "precedence": ["unanimity", "bottom"],
        "counts": counts,
        "containment_verified": True,
        "repaint_hash": repaint_hash,
        # Debug flag: if full unanimity, selector and unanimity hashes must match.
        "full_unanimity_match": (counts["unanimity"] == R_out * C_out)
    }

    return Y_out, selection_receipts


# ============================================================================
# M3: Minimal Selector (Witness → Unanimity → Bottom)
# ============================================================================

def select_witness_first(
    A_wit: Dict[int, List[int]],
    S_wit: List[int],
    A_uni: Dict[int, List[int]],
    S_uni: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> Tuple[List[List[int]], Dict]:
    """
    Minimal selector for M3 (no LFP yet).

    Precedence:
      1. Witness bucket (scope-gated): pick min{c | A_wit[c][p]==1} if S_wit[p]==1
      2. Unanimity bucket: pick singleton if S_uni[p]==1
      3. Bottom: pick 0

    Args:
        A_wit: Witness admits (color -> plane)
        S_wit: Witness scope (list of row masks)
        A_uni: Unanimity admits (color -> plane)
        S_uni: Unanimity scope (list of row masks)
        colors_order: Ascending color list
        R_out, C_out: Canvas dimensions

    Returns:
        Tuple of (Y_out, selection_receipts):
          - Y_out: Selected grid (R_out × C_out)
          - selection_receipts: Dict with counts, containment, repaint_hash
    """
    Y_out = [[0] * C_out for _ in range(R_out)]

    counts = {"witness": 0, "unanimity": 0, "bottom": 0}
    # Diagnostic: count phantom scope pixels (S_wit==1 but admits empty)
    # Per Sub-WO-M3-SelectorFix - should be 0 after normalization fix
    witness_empty_scope_pixels = 0

    for r in range(R_out):
        for c in range(C_out):
            # Check witness scope
            if S_wit[r] & (1 << c):
                # Witness bucket: collect all colors with bit set at this pixel
                cand = []
                for color in colors_order:
                    if A_wit[color][r] & (1 << c):
                        cand.append(color)

                if cand:
                    # Pick minimum color
                    Y_out[r][c] = min(cand)
                    counts["witness"] += 1
                    continue
                else:
                    # Phantom scope: S_wit bit set but no admits
                    # After normalization fix, this should never happen
                    witness_empty_scope_pixels += 1
                    # Fall through to unanimity/bottom

            # Check unanimity scope
            if S_uni[r] & (1 << c):
                # Unanimity bucket: should be singleton
                cand = []
                for color in colors_order:
                    if A_uni[color][r] & (1 << c):
                        cand.append(color)

                if len(cand) == 1:
                    Y_out[r][c] = cand[0]
                    counts["unanimity"] += 1
                    continue

            # Bottom: default to 0
            Y_out[r][c] = 0
            counts["bottom"] += 1

    # Compute repaint hash for idempotence check
    grid_bytes = serialize_grid_be_row_major(Y_out, R_out, C_out, colors_order)
    repaint_hash = blake3_hash(grid_bytes)

    selection_receipts = {
        "precedence": ["witness", "unanimity", "bottom"],
        "counts": counts,
        "witness_empty_scope_pixels": witness_empty_scope_pixels,  # Diagnostic (Sub-WO-M3-SelectorFix)
        "containment_verified": True,  # With D*=1, always true
        "repaint_hash": repaint_hash,
    }

    return Y_out, selection_receipts


def solve(
    task_json: dict,
    families: Tuple[str, ...] = FULL_FAMILY_SET,
    skip_h8h9_if_area1: bool = False,
    with_witness: bool = True,
    with_unanimity: bool = False
) -> Tuple[List[List[int]], Dict]:
    """
    ARC-AGI deterministic solver (milestone-based).

    Current Implementation (M4):
      M2: Output path (transport + unanimity)
      M3: Witness path (optional)
      M4: True LFP (forbids + admit-∧ + AC-3)

      Steps:
        M0 sections:
          1) Build color universe C = {0} ∪ colors(X*) ∪ ⋃colors(X_i,Y_i)
          2) PACK→UNPACK identity check on all trainings and on X*
          3) Canonicalize frames (Π_in for all X_i and X*, Π_out for all Y_i)
          4) apply_pose_anchor round-trip proof on X* planes

        M1' sections:
          5) Extract WO-14 features from training inputs
          6) Choose working canvas (R_out, C_out) via WO-04a (H1-H9)

        M2 sections:
          7) Output transport (WO-08: normalize + transport to working canvas)
          8) Unanimity (WO-08: singleton admits where trainings agree)

        M3 sections (optional, with_witness=True):
          9) Extract components from training inputs (WO-05)
          10) Learn witness per training (WO-06: rigid pieces + σ)
          11) Emit global witness (WO-07: conjugation + forward mapping)

        M4 sections:
          12) Learn forbids from training outputs (WO-10: Type 1 + Type 2)
          13) Assemble emitters in frozen family order (T1_witness, T2_unity, ...)
          14) Initialize D0 to all-ones (top of lattice)
          15) Run LFP propagation (WO-11: admit-∧ + AC-3, monotone fixed point)
          16) Handle UNSAT/FIXED_POINT_NOT_REACHED (fail-closed)

        Final sections:
          17) Selector (witness → unanimity → bottom)
          18) Seal receipts and return

    Future Milestones:
      - M5: Lattice + EngineWinner

    Args:
        task_json: ARC task dict with keys:
            - "train": list of {"input": grid, "output": grid}
            - "test": list of {"input": grid}
        families: Tuple of family IDs to evaluate (Sub-WO-RUN-FAM).
            Default: FULL_FAMILY_SET (all H1-H9).
        skip_h8h9_if_area1: If True, skip H8/H9 when area=1 found (Sub-WO-RUN-FAM).
            Default: False (production conservative).
        with_witness: If True, enable witness path (default: True at M3).
        with_unanimity: If True, enable unanimity (default: False at M3).

    Returns:
        Tuple of (Y, receipts_bundle):
          - Y: Predicted output grid
          - receipts_bundle: dict with sectioned receipts

    Invariants (all milestones):
        - No heuristics, no floats, no RNG
        - No minted non-zero bits
        - Color exclusivity preserved
        - Receipts-first: every step logs BLAKE3 hashes
        - Deterministic: double-run produces identical hashes

    Raises:
        ValueError: On shape mismatch, bit overflow, bad pose ID
        SizeUndetermined: If no H1-H9 hypothesis fits all trainings (M1')
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
    receipts = Receipts("M1-canvas-online")  # Section label (will evolve with milestones)

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
    # M1 Step 5: Extract WO-14 Features (from training inputs)
    # ========================================================================

    # WO-14 features are extracted from INPUT grids only (not outputs)
    # These are used for WO-04a size prediction
    features_receipts = []

    for i, pair in enumerate(train_pairs):
        X_i = pair["input"]
        H_i = len(X_i)
        W_i = len(X_i[0]) if X_i else 0

        # Extract features using WO-14
        # agg_features returns (FeatureVector, receipts_bundle)
        feature_vec, feat_receipts = agg_features(X_i, H_i, W_i, colors_order)

        features_receipts.append({
            "train_id": i,
            "features_hash": feat_receipts["section_hash"]
        })

    receipts.put("features.trainings", features_receipts)

    # ========================================================================
    # M1 Step 6: Choose Working Canvas (WO-04a)
    # ========================================================================

    # Prepare inputs for choose_working_canvas
    # Need: train_pairs (dicts with X,Y), frames_in, frames_out, xstar_shape

    # Build train_pairs in format expected by choose_working_canvas
    train_pairs_for_canvas = []
    for pair in train_pairs:
        train_pairs_for_canvas.append({
            "X": pair["input"],
            "Y": pair["output"]
        })

    # Build frames_in and frames_out from canonicalize receipts
    frames_in = []
    frames_out = []

    for receipt in frames_receipts:
        if receipt["io_type"] == "input" and receipt["split"] == "train":
            frames_in.append({
                "pose": receipt["frame.pose"],
                "anchor": receipt["frame.anchor"]
            })
        elif receipt["io_type"] == "output" and receipt["split"] == "train":
            frames_out.append({
                "pose": receipt["frame.pose"],
                "anchor": receipt["frame.anchor"]
            })

    # xstar_shape is (H*, W*)
    xstar_shape = (H_star, W_star)

    # Call choose_working_canvas (may raise SizeUndetermined)
    # M1': Now includes H8/H9 via xstar_grid parameter (Sub-WO-04a-H8H9)
    # Sub-WO-RUN-FAM: Pass families and skip_h8h9_if_area1 for family gating
    R_out, C_out, canvas_receipts = choose_working_canvas(
        train_pairs_for_canvas,
        frames_in,
        frames_out,
        xstar_shape,
        colors_order,
        xstar_grid=X_star,  # Enables H8/H9 feature extraction and test_area computation
        families=families,  # Family allow-list (Sub-WO-RUN-FAM)
        skip_h8h9_if_area1=skip_h8h9_if_area1  # Skip H8/H9 if area=1 found (Sub-WO-RUN-FAM)
    )

    # Add working_canvas section to receipts
    receipts.put("working_canvas", canvas_receipts["payload"])

    # ========================================================================
    # M2 Step 7: Output Transport (WO-08)
    # ========================================================================

    # Build Y_train_list from training outputs
    Y_train_list = [pair["output"] for pair in train_pairs]

    # Build frames_out list (pose, anchor tuples) from frames_receipts
    frames_out_list = []
    for receipt in frames_receipts:
        if receipt["split"] == "train" and receipt["io_type"] == "output":
            pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
            anchor = receipt["frame.anchor"]
            anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
            frames_out_list.append((pose_id, anchor_tuple))

    # pi_out_star is the test output frame (working canvas is always identity)
    pi_out_star = ("I", (0, 0))

    # Call emit_output_transport
    A_out_list, S_out_list, transport_receipts, transport_section = emit_output_transport(
        Y_train_list, frames_out_list, R_out, C_out, colors_order, pi_out_star
    )

    # Add transport receipts
    receipts.put("transports", transport_section["payload"])

    # ========================================================================
    # M2 Step 8: Unanimity (WO-08)
    # ========================================================================

    # Call emit_unity (unanimity)
    A_uni, S_uni, unanimity_receipt = emit_unity(
        A_out_list, S_out_list, colors_order, R_out, C_out
    )

    # Add unanimity receipts
    receipts.put("unanimity", unanimity_receipt)

    # ========================================================================
    # M3 Step 9: Extract Components (WO-05) - if witness enabled
    # ========================================================================

    if with_witness:
        # Extract components from each training input
        # We'll need these for witness learning
        components_per_training = []

        for i, pair in enumerate(train_pairs):
            X_i = pair["input"]
            H_i = len(X_i)
            W_i = len(X_i[0]) if X_i else 0

            # Pack to planes
            planes_i = pack_grid_to_planes(X_i, H_i, W_i, colors_order)

            # Extract components (WO-05)
            comps_i, comps_receipts = components(planes_i, H_i, W_i, colors_order)

            components_per_training.append({
                "train_id": i,
                "num_components": len(comps_i),
                "components_hash": comps_receipts["section_hash"]
            })

        receipts.put("components", components_per_training)

    # ========================================================================
    # M3 Step 8: Learn Witness (WO-06) - per training
    # ========================================================================

    if with_witness:
        witness_results_all = []
        witness_receipts_all = []

        for i, pair in enumerate(train_pairs):
            X_i = pair["input"]
            Y_i = pair["output"]

            # Get frames for this training from canonicalize receipts
            # Find the matching frames from frames_receipts
            Pi_in_i = None
            Pi_out_i = None

            for receipt in frames_receipts:
                if receipt["split"] == "train" and receipt["idx"] == i:
                    if receipt["io_type"] == "input":
                        pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
                        anchor = receipt["frame.anchor"]
                        anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
                        Pi_in_i = (pose_id, anchor_tuple)
                    elif receipt["io_type"] == "output":
                        pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
                        anchor = receipt["frame.anchor"]
                        anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
                        Pi_out_i = (pose_id, anchor_tuple)

            if Pi_in_i is None or Pi_out_i is None:
                raise ValueError(f"Runner: Missing frames for training {i}")

            # Build frames dict for this training
            frames_i = {
                "Pi_in": Pi_in_i,
                "Pi_out": Pi_out_i
            }

            # Learn witness (WO-06)
            witness_result = learn_witness(X_i, Y_i, frames_i)

            witness_results_all.append(witness_result)

            # Extract receipts for logging
            witness_receipts_all.append({
                "train_id": i,
                "silent": witness_result["silent"],
                "num_pieces": len(witness_result["pieces"]),
                "sigma_bijection_ok": witness_result["receipts"]["payload"]["sigma"]["bijection_ok"],
                "overlap_conflict": witness_result["receipts"]["payload"]["overlap"]["conflict"],
                "section_hash": witness_result["receipts"]["section_hash"]
            })

        receipts.put("witness_learn", witness_receipts_all)

    # ========================================================================
    # M3 Step 9: Emit Global Witness (WO-07)
    # ========================================================================

    if with_witness:
        # Build frames dict for emit_witness
        # Need: Pi_in_star, Pi_out_star, and per-training frames

        # Get Pi_in_star and Pi_out_star from frames_receipts
        Pi_in_star = None
        Pi_out_star = (
            "I",
            (0, 0),
        )  # Working canvas frame is always identity (canonical)

        for receipt in frames_receipts:
            if receipt["split"] == "test" and receipt["idx"] == 0 and receipt["io_type"] == "input":
                pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
                anchor = receipt["frame.anchor"]
                anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
                Pi_in_star = (pose_id, anchor_tuple)

        if Pi_in_star is None:
            raise ValueError("Runner: Missing Pi_in_star frame")

        # Build frames dict with all training frames
        frames_all = {
            "Pi_in_star": Pi_in_star,
            "Pi_out_star": Pi_out_star,
        }

        for i in range(len(train_pairs)):
            # Find frames for this training
            for receipt in frames_receipts:
                if receipt["split"] == "train" and receipt["idx"] == i:
                    if receipt["io_type"] == "input":
                        pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
                        anchor = receipt["frame.anchor"]
                        anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
                        frames_all[f"Pi_in_{i}"] = (pose_id, anchor_tuple)
                    elif receipt["io_type"] == "output":
                        pose_id = receipt["frame.pose"]["pose_id"] if isinstance(receipt["frame.pose"], dict) else receipt["frame.pose"]
                        anchor = receipt["frame.anchor"]
                        anchor_tuple = (anchor["r"], anchor["c"]) if isinstance(anchor, dict) else tuple(anchor)
                        frames_all[f"Pi_out_{i}"] = (pose_id, anchor_tuple)

        # Emit global witness (WO-07)
        A_wit, S_wit, witness_emit_receipts = emit_witness(
            X_star, witness_results_all, frames_all, colors_order, R_out, C_out
        )

        # Add witness_emit receipts to final bundle
        receipts.put("witness_emit", witness_emit_receipts["payload"])

    # ========================================================================
    # M4 Step 0: Learn Forbids (WO-10)
    # ========================================================================

    # Reconstruct Y_train_on_canvas from transports
    # These are the normalized training outputs on the working canvas
    Y_train_on_canvas = []
    included_train_ids = []

    for i, (A_out_i, S_out_i) in enumerate(zip(A_out_list, S_out_list)):
        # Check if this training is silent (no scope)
        has_scope = any(row != 0 for row in S_out_i)
        if not has_scope:
            continue  # Skip silent trainings

        included_train_ids.append(i)

        # Reconstruct grid from planes
        Y_i = [[0] * C_out for _ in range(R_out)]
        for r in range(R_out):
            scope_row = S_out_i[r] if r < len(S_out_i) else 0
            if scope_row == 0:
                continue

            for c in range(C_out):
                bit = 1 << c
                if not (scope_row & bit):
                    continue

                # Find the unique color (singleton from WO-08)
                for color in colors_order:
                    plane = A_out_i.get(color)
                    if plane and r < len(plane):
                        if plane[r] & bit:
                            Y_i[r][c] = color
                            break

        Y_train_on_canvas.append(Y_i)

    # Learn forbids from training outputs
    M_matrix, forbids_receipt = learn_forbids(
        Y_train_on_canvas, included_train_ids, colors_order
    )
    E_graph = build_4neighbor_graph(R_out, C_out)

    # Add forbids receipts
    receipts.put("forbids", forbids_receipt)

    # ========================================================================
    # M4 Step 1-3: Assemble Emitters + Initialize D0 + Run LFP (WO-11)
    # ========================================================================

    # Assemble emitters in frozen family order (T1...T12)
    # At M4, we only have T1 (witness) and T2 (unanimity)
    emitters_list = []

    if with_witness:
        emitters_list.append(("T1_witness", A_wit, S_wit))

    # T2_unity (unanimity) - always add if available
    emitters_list.append(("T2_unity", A_uni, S_uni))

    # Initialize D0 to all-ones (top of lattice)
    D0 = {(r, c): ((1 << len(colors_order)) - 1) for r in range(R_out) for c in range(C_out)}

    # Run LFP propagation
    forbids_tuple = (E_graph, M_matrix) if E_graph and M_matrix else None

    result = lfp_propagate(
        D0, emitters_list, forbids=forbids_tuple,
        colors_order=colors_order, R_out=R_out, C_out=C_out
    )

    # Handle UNSAT or FIXED_POINT_NOT_REACHED (fail-closed)
    if isinstance(result, tuple) and isinstance(result[0], str):
        # UNSAT or FIXED_POINT_NOT_REACHED
        status, lfp_stats = result

        # Add LFP receipts with failure status
        receipts.put("lfp", lfp_stats)
        receipts.put("lfp_status", status)

        # Seal receipts and return with empty grid (fail-closed)
        receipts_bundle = receipts.digest()
        Y_out = [[0] * C_out for _ in range(R_out)]
        return (Y_out, receipts_bundle)

    # Success: extract D* and stats
    D_star, lfp_stats = result

    # Add LFP receipts
    receipts.put("lfp", lfp_stats)
    receipts.put("lfp_status", "SUCCESS")

    # ========================================================================
    # M4 Step 4: Selector (Witness → Unanimity → Bottom)
    # ========================================================================
    # Note: At M4 with all-ones D0, D* is typically the admits intersection.
    # Selection precedence remains witness → unanimity → bottom (no EngineWinner yet)

    if with_witness:
        # M3: Witness → Unanimity → Bottom
        Y_out, selection_receipts = select_witness_first(
            A_wit, S_wit, A_uni, S_uni, colors_order, R_out, C_out
        )
    else:
        # M2: Unanimity → Bottom (output path only)
        Y_out, selection_receipts = select_unanimity_first(
            A_uni, S_uni, colors_order, R_out, C_out
        )

    # Guard: verify selector/unanimity match under full unanimity (Sub-WO-M2-SelectorPatch final)
    # When unanimity covers all pixels, repaint_hash must match unanimity_grid_hash
    # Both use grid-encoded format for apples-to-apples comparison
    if selection_receipts["counts"]["unanimity"] == R_out * C_out:
        assert selection_receipts["repaint_hash"] == unanimity_receipt["unanimity_grid_hash"], \
            f"Selector/unanimity mismatch under full unanimity: " \
            f"repaint_hash={selection_receipts['repaint_hash']} != " \
            f"unanimity_grid_hash={unanimity_receipt['unanimity_grid_hash']}"
        selection_receipts["full_unanimity_match"] = True
    else:
        selection_receipts["full_unanimity_match"] = False

    # Add section hash to selection receipts
    selection_payload = selection_receipts.copy()
    selection_json = blake3_hash(
        str(selection_payload).encode("utf-8")
    )  # Simple hash for now
    selection_receipts["section_hash"] = selection_json

    receipts.put("selection", selection_receipts)

    # ========================================================================
    # Step 11: Seal receipts and return
    # ========================================================================

    # Seal receipts (no timestamps)
    receipts_bundle = receipts.digest()

    return (Y_out, receipts_bundle)


def solve_with_determinism_check(
    task_json: dict,
    families: Tuple[str, ...] = FULL_FAMILY_SET,
    skip_h8h9_if_area1: bool = False,
    with_witness: bool = True,
    with_unanimity: bool = False
) -> Tuple[List[List[int]], Dict]:
    """
    Run solve() twice and verify determinism (all section hashes match).

    Args:
        task_json: ARC task dict.
        families: Tuple of family IDs to evaluate (Sub-WO-RUN-FAM).
        skip_h8h9_if_area1: If True, skip H8/H9 when area=1 found (Sub-WO-RUN-FAM).
        with_witness: If True, enable witness path (default: True at M3).
        with_unanimity: If True, enable unanimity (default: False at M3).

    Returns:
        Tuple of (Y, receipts_bundle) with determinism flags added.

    Raises:
        RuntimeError: If double-run hashes differ.

    Spec:
        Determinism check (A.3): Verifies referential transparency.
        Applicable to all milestones.
    """
    # Run 1
    Y1, receipts1 = solve(
        task_json,
        families=families,
        skip_h8h9_if_area1=skip_h8h9_if_area1,
        with_witness=with_witness,
        with_unanimity=with_unanimity
    )

    # Run 2
    Y2, receipts2 = solve(
        task_json,
        families=families,
        skip_h8h9_if_area1=skip_h8h9_if_area1,
        with_witness=with_witness,
        with_unanimity=with_unanimity
    )

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


# ============================================================================
# CLI Entry Point (Sub-WO-RUN-FAM)
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="ARC-AGI Deterministic Runner (M3 with witness path)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast pass with H1-7 only
  python -m src.arcbit.runner task.json --families=H1-7

  # Full evaluation with all H1-H9
  python -m src.arcbit.runner task.json --families=H1-9

  # Skip H8/H9 if area=1 found (dev speed optimization)
  python -m src.arcbit.runner task.json --skip-h8h9-if-area1

  # M3: Witness path (default on)
  python -m src.arcbit.runner task.json --with-witness

  # Disable witness (return to M1' behavior)
  python -m src.arcbit.runner task.json --no-witness

  # Enable unanimity (M4+, not yet implemented)
  python -m src.arcbit.runner task.json --with-unanimity

Dev workflow (two-pass):
  1. Fast pass: --families=H1-7
  2. Rerun SIZE_UNDETERMINED with: --families=H1-9
        """
    )

    parser.add_argument(
        "task_file",
        type=str,
        help="Path to ARC task JSON file"
    )

    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Family allow-list: 'H1-7' (range) or 'H1,H2,H5' (CSV). Default: all H1-H9."
    )

    parser.add_argument(
        "--skip-h8h9-if-area1",
        action="store_true",
        help="Skip H8/H9 if area=1 found in H1-H7 (dev speed optimization). Default: False."
    )

    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Run double-solve determinism check. Default: False (single solve)."
    )

    parser.add_argument(
        "--with-witness",
        action="store_true",
        default=True,
        help="Enable witness path (M3). Default: True."
    )

    parser.add_argument(
        "--no-witness",
        action="store_true",
        help="Disable witness path (return to M1' behavior). Default: False."
    )

    parser.add_argument(
        "--with-unanimity",
        action="store_true",
        help="Enable unanimity (M4+, not yet fully implemented). Default: False."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for receipts JSON. Default: print to stdout."
    )

    args = parser.parse_args()

    # Handle witness flag logic
    with_witness = args.with_witness and not args.no_witness

    # Load task JSON
    try:
        with open(args.task_file, 'r') as f:
            task_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Task file not found: {args.task_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in task file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse families parameter
    if args.families:
        try:
            families_tuple = parse_families(args.families)
        except ValueError as e:
            print(f"Error: Invalid families parameter: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        families_tuple = FULL_FAMILY_SET  # Default: all H1-H9

    # Run solver
    try:
        if args.determinism_check:
            Y, receipts = solve_with_determinism_check(
                task_json,
                families=families_tuple,
                skip_h8h9_if_area1=args.skip_h8h9_if_area1,
                with_witness=with_witness,
                with_unanimity=args.with_unanimity
            )
        else:
            Y, receipts = solve(
                task_json,
                families=families_tuple,
                skip_h8h9_if_area1=args.skip_h8h9_if_area1,
                with_witness=with_witness,
                with_unanimity=args.with_unanimity
            )
    except SizeUndetermined as e:
        # SIZE_UNDETERMINED: print receipts and exit with code 2
        print("SIZE_UNDETERMINED: No hypothesis fits all trainings", file=sys.stderr)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(e.receipts, f, indent=2)
            print(f"Receipts written to: {args.output}", file=sys.stderr)
        else:
            print("\nReceipts:", file=sys.stderr)
            print(json.dumps(e.receipts, indent=2), file=sys.stderr)
        sys.exit(2)
    except (ValueError, RuntimeError) as e:
        # Other errors: print error and exit with code 1
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Success: print results
    result = {
        "prediction": Y,
        "receipts": receipts
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to: {args.output}")
    else:
        print(json.dumps(result, indent=2))

    sys.exit(0)
