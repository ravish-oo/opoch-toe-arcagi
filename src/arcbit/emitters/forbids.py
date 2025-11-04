"""
WO-10 Component: Forbids Learning + AC-3

Learn forbid matrix M from training outputs and apply AC-3 arc consistency
to prune domains.

Spec: WO-10 v1.6
"""

from typing import Dict, List, Tuple, Optional, TypedDict, Set
from collections import deque
from ..core.hashing import blake3_hash


class ForbidsReceipt(TypedDict):
    forbid_symmetric: bool
    matrix_hash: str
    edges_count: int
    colors_with_forbids: List[int]  # colors with M[c][c]=1 or M[c][d]=1


class AC3Stats(TypedDict):
    queue_init_len: int
    arcs_processed: int
    prunes: int
    passes: int
    empties: int
    section_hash: str


def build_4neighbor_graph(R_out: int, C_out: int) -> List[Tuple[int, int, int, int]]:
    """
    Build directed 4-neighbor edges in frozen row-major order.

    For each pixel (r,c) in row-major order, check neighbors in fixed order:
    UP, LEFT, RIGHT, DOWN. Add directed arc (pr,pc,qr,qc) if in-bounds.

    Args:
        R_out, C_out: Canvas dimensions

    Returns:
        E_graph: List of directed edges (pr, pc, qr, qc)

    Spec: WO-10 v1.6 section "E_graph"
    """
    E_graph = []

    # Fixed neighbor order: UP, LEFT, RIGHT, DOWN
    # UP: (r-1, c), LEFT: (r, c-1), RIGHT: (r, c+1), DOWN: (r+1, c)
    neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    # Iterate pixels in row-major order
    for pr in range(R_out):
        for pc in range(C_out):
            # Check each neighbor in fixed order
            for dr, dc in neighbor_offsets:
                qr = pr + dr
                qc = pc + dc

                # Check bounds
                if 0 <= qr < R_out and 0 <= qc < C_out:
                    # Add directed arc p->q
                    E_graph.append((pr, pc, qr, qc))

    return E_graph


def learn_forbids(
    Y_train_list: List[List[List[int]]],
    included_train_ids: List[int],
    colors_order: List[int]
) -> Tuple[Dict[int, Set[int]], ForbidsReceipt]:
    """
    Learn forbid matrix M from training outputs.

    Type 1 (Universal differ): M[c][c] = 1 if:
      - NO training has adjacent (c,c) pair, AND
      - Color c appeared on at least one edge (non-vacuous)

    Type 2 (Directed): M[c][d] = 1 (c≠d) if:
      - ALL FOUR orientations prove M[c][d]=1, where for each orientation:
        - Color c appeared at source position (non-vacuous), AND
        - NEVER observed (c at p, d at q) for that orientation

    Args:
        Y_train_list: Per-training integer grids on working canvas
        included_train_ids: IDs of non-silent trainings
        colors_order: Global color order

    Returns:
        M: Forbid matrix as dict[c, set[d]] where d in M[c] means (c,d) forbidden
        receipt: ForbidsReceipt

    Spec: WO-10 v1.6 section "Learning forbids"
    """
    if not Y_train_list:
        # No trainings → empty forbid matrix
        M = {c: set() for c in colors_order}
        receipt = ForbidsReceipt(
            forbid_symmetric=True,
            matrix_hash=_hash_forbid_matrix(M, colors_order),
            edges_count=0,
            colors_with_forbids=[]
        )
        return M, receipt

    # BUG FIX #2: Validate uniform dimensions
    R_out = len(Y_train_list[0])
    C_out = len(Y_train_list[0][0]) if Y_train_list[0] else 0

    for i, Y_i in enumerate(Y_train_list):
        if len(Y_i) != R_out:
            raise ValueError(
                f"Training {i} has {len(Y_i)} rows, expected {R_out}. "
                f"All trainings must have uniform dimensions after WO-08 transport."
            )
        for r, row in enumerate(Y_i):
            if len(row) != C_out:
                raise ValueError(
                    f"Training {i} row {r} has {len(row)} cols, expected {C_out}. "
                    f"All trainings must have uniform dimensions after WO-08 transport."
                )

    # Scan all trainings to collect per-orientation adjacency evidence
    # For Type 2 directed forbids, we track evidence per orientation

    # Fixed neighbor offsets: UP, LEFT, RIGHT, DOWN
    neighbor_offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    orientation_names = ["UP", "LEFT", "RIGHT", "DOWN"]

    # For each orientation: track observed (c, d) pairs and source colors
    # per_orientation[orientation_idx] = {
    #   'observed_pairs': set of (c, d),
    #   'source_colors': set of c
    # }
    per_orientation = [
        {'observed_pairs': set(), 'source_colors': set()}
        for _ in range(4)
    ]

    # Also track global adjacency for Type 1
    adjacent_pairs = set()  # (c, d) pairs that appear adjacent (any orientation)
    colors_on_edges = set()  # colors that appear on at least one edge

    for Y_i in Y_train_list:
        # Scan all edges
        for r in range(R_out):
            for c in range(C_out):
                color_p = Y_i[r][c]

                # Check each neighbor with its orientation
                for orient_idx, (dr, dc) in enumerate(neighbor_offsets):
                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < R_out and 0 <= nc < C_out:
                        color_q = Y_i[nr][nc]

                        # Record for this orientation
                        per_orientation[orient_idx]['observed_pairs'].add((color_p, color_q))
                        per_orientation[orient_idx]['source_colors'].add(color_p)

                        # Record global adjacency (directed)
                        adjacent_pairs.add((color_p, color_q))

                        # Record that these colors appeared on edges
                        colors_on_edges.add(color_p)
                        colors_on_edges.add(color_q)

    # Learn forbids
    M = {c: set() for c in colors_order}

    # Type 1: M[c][c] = 1 (universal differ)
    for c in colors_order:
        # Check conditions for M[c][c] = 1:
        # 1. c appeared on at least one edge (non-vacuous)
        # 2. NO adjacent (c,c) pair observed
        if c in colors_on_edges and (c, c) not in adjacent_pairs:
            # Universal differ proved for c
            M[c].add(c)

    # Type 2: M[c][d] = 1 for c≠d (directed, very conservative)
    # Only add if ALL FOUR orientations prove M[c][d]=1
    for c in colors_order:
        for d in colors_order:
            if c == d:
                continue  # Type 1 already handled this

            # Check if ALL four orientations prove M[c][d]=1
            all_orientations_forbid = True

            for orient_idx in range(4):
                source_colors = per_orientation[orient_idx]['source_colors']
                observed_pairs = per_orientation[orient_idx]['observed_pairs']

                # For this orientation to prove M[c][d]=1:
                # 1. Color c must have appeared at source (non-vacuous)
                # 2. Pair (c, d) must NEVER have been observed

                if c in source_colors:
                    # c appeared at source, check if (c,d) observed
                    if (c, d) in observed_pairs:
                        # Observed (c,d) → this orientation does NOT forbid
                        all_orientations_forbid = False
                        break
                    # else: c appeared but (c,d) never seen → this orientation forbids
                else:
                    # c never appeared at source for this orientation → vacuous, no evidence
                    # Conservative: treat as "does not prove forbid"
                    all_orientations_forbid = False
                    break

            # If all four orientations prove M[c][d]=1, add it
            if all_orientations_forbid:
                M[c].add(d)

    # Build receipt
    colors_with_forbids = [c for c in colors_order if len(M[c]) > 0]
    forbid_symmetric = all(len(M[c]) == 0 or M[c] == {c} for c in colors_order)

    E_graph = build_4neighbor_graph(R_out, C_out)
    edges_count = len(E_graph)

    receipt = ForbidsReceipt(
        forbid_symmetric=forbid_symmetric,
        matrix_hash=_hash_forbid_matrix(M, colors_order),
        edges_count=edges_count,
        colors_with_forbids=colors_with_forbids
    )

    return M, receipt


def ac3_prune(
    D: Dict[Tuple[int, int], int],
    E_graph: List[Tuple[int, int, int, int]],
    M: Dict[int, Set[int]],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> Tuple[bool, AC3Stats]:
    """
    AC-3 arc consistency with frozen FIFO queue.

    For each arc (p->q), remove c from D[p] if it has no support in D[q]
    (i.e., all d in D[q] are forbidden: M[c][d] = 1).

    Args:
        D: Per-pixel domain bitmask over colors_order
        E_graph: Directed 4-neighbor edges
        M: Forbid matrix
        colors_order: Global color order
        R_out, C_out: Canvas dimensions

    Returns:
        changed: True if any prune happened
        stats: AC3Stats

    Spec: WO-10 v1.6 section "AC-3"
    """
    # Initialize queue with all arcs
    queue = deque(E_graph)
    queue_init_len = len(queue)

    arcs_processed = 0
    prunes = 0
    empties = 0
    passes = 1  # Count outer loop iterations

    # Build reverse adjacency for efficient predecessor lookup
    # predecessors[(qr, qc)] = [(pr, pc), ...] for all arcs pr,pc -> qr,qc
    predecessors = {}
    for pr, pc, qr, qc in E_graph:
        if (qr, qc) not in predecessors:
            predecessors[(qr, qc)] = []
        predecessors[(qr, qc)].append((pr, pc))

    changed = False

    while queue:
        # Pop arc FIFO
        pr, pc, qr, qc = queue.popleft()
        arcs_processed += 1

        p = (pr, pc)
        q = (qr, qc)

        # Get domains
        D_p = D.get(p, 0)
        D_q = D.get(q, 0)

        if D_p == 0:
            # Domain already empty, skip
            continue

        # Check each color in D[p] for support in D[q]
        pruned_this_arc = False

        for c in colors_order:
            bit_c = 1 << colors_order.index(c)

            # Is c in D[p]?
            if not (D_p & bit_c):
                continue

            # Check if c has support in D[q]
            # Support means: exists d in D[q] such that M[c][d] = 0 (not forbidden)
            has_support = False

            for d in colors_order:
                bit_d = 1 << colors_order.index(d)

                # Is d in D[q]?
                if not (D_q & bit_d):
                    continue

                # Is (c,d) allowed (not forbidden)?
                if d not in M.get(c, set()):
                    # Found support!
                    has_support = True
                    break

            if not has_support:
                # Remove c from D[p]
                D[p] = D_p & ~bit_c
                D_p = D[p]
                prunes += 1
                pruned_this_arc = True
                changed = True

        # If we pruned at p, enqueue all arcs r->p
        if pruned_this_arc:
            # Check if D[p] became empty
            if D[p] == 0:
                empties += 1

            # Enqueue predecessors
            for rr, rc in predecessors.get(p, []):
                queue.append((rr, rc, pr, pc))

    # Build stats
    stats = AC3Stats(
        queue_init_len=queue_init_len,
        arcs_processed=arcs_processed,
        prunes=prunes,
        passes=passes,
        empties=empties,
        section_hash=_hash_ac3_stats(queue_init_len, arcs_processed, prunes, passes, empties)
    )

    return changed, stats


# ============================================================================
# Helpers
# ============================================================================


def _hash_forbid_matrix(M: Dict[int, Set[int]], colors_order: List[int]) -> str:
    """
    Hash forbid matrix in canonical row-major order.

    For each c in colors_order, serialize a bitmask over colors_order
    where bit d is 1 iff d in M[c].
    """
    result = b""

    for c in colors_order:
        # Build bitmask for this row
        row_mask = 0
        for d in colors_order:
            if d in M.get(c, set()):
                bit_idx = colors_order.index(d)
                row_mask |= (1 << bit_idx)

        # Serialize to bytes
        num_bytes = (len(colors_order) + 7) // 8
        result += row_mask.to_bytes(num_bytes, "big")

    return blake3_hash(result)


def _hash_ac3_stats(
    queue_init_len: int,
    arcs_processed: int,
    prunes: int,
    passes: int,
    empties: int
) -> str:
    """Hash AC-3 stats in deterministic order."""
    stats_str = f"{queue_init_len},{arcs_processed},{prunes},{passes},{empties}"
    return blake3_hash(stats_str.encode("utf-8"))
