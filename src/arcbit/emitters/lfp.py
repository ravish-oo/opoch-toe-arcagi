"""
WO-11 Component: Domain Tensor & LFP Propagator

Compute the least fixed point (LFP) of constraints via monotone iteration:
  1. Admit-intersect in frozen family order (T1...T12)
  2. AC-3 prune with forbids
  3. Repeat until no changes (fixed point)

Spec: WO-11 v1.6
"""

from typing import Dict, List, Tuple, Optional, TypedDict, Set
from copy import deepcopy
from ..core.hashing import blake3_hash
from .forbids import ac3_prune


class LFPStats(TypedDict):
    admit_passes: int
    ac3_passes: int
    total_admit_prunes: int     # sum of bits removed by admits
    total_ac3_prunes: int       # sum of AC-3 prunes
    empties: int                # pixels with empty domain (>0 = UNSAT)
    singleton_pixels: int       # pixels with exactly 1 color in domain
    multi_pixels: int           # pixels with >1 colors in domain
    empty_pixels: int           # pixels with 0 colors (alias for empties for clarity)
    lfp_prunes_by_family: Dict[str, int]  # prunes contributed by each family (T1, T2, T3, ...)
    domains_hash: str           # BLAKE3 of final domain tensor
    section_hash: str


# Frozen defaults (spec WO-11 v1.6)
MAX_PROPAGATION_ITERATIONS = 1000

# Frozen family order (spec WO-11 v1.6)
FROZEN_FAMILY_ORDER = [
    "T1_witness",
    "T2_unity",
    "T3_lattice",
    "T4_kron",
    "T5_conv",
    "T6_morph",
    "T7_logic",
    "T8_param",
    # T9 is size; no admits
    "T10_forbids",  # placeholder; AC-3 consumes forbids separately
    "T11_csp",
    "T12_strata",
]


def lfp_propagate(
    D0: Dict[Tuple[int, int], int],
    emitters_list: List[Tuple[str, Optional[Dict[int, List[int]]], Optional[List[int]]]],
    forbids: Optional[Tuple[List[Tuple[int, int, int, int]], Dict[int, Set[int]]]] = None,
    colors_order: Optional[List[int]] = None,
    R_out: Optional[int] = None,
    C_out: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int], int], LFPStats] | Tuple[str, LFPStats]:
    """
    Run monotone loop to compute least fixed point of constraints.

    Args:
        D0: Initial domain tensor (r,c) -> bitmask over colors_order
        emitters_list: List of (family_name, A, S) admits in any order
        forbids: Optional (E_graph, M_matrix) from WO-10
        colors_order: Ascending color list (required for admits/AC-3)
        R_out, C_out: Canvas dimensions (inferred from D0 if not provided)

    Returns:
        (D*, stats) on success
        ("UNSAT", stats) if any domain becomes empty
        ("FIXED_POINT_NOT_REACHED", stats) if cap exceeded

    Spec: WO-11 v1.6
    """
    # Infer dimensions if not provided
    if R_out is None or C_out is None:
        if not D0:
            R_out = R_out or 0
            C_out = C_out or 0
        else:
            max_r = max(r for r, c in D0.keys())
            max_c = max(c for r, c in D0.keys())
            R_out = R_out or (max_r + 1)
            C_out = C_out or (max_c + 1)

    # Default colors_order if not provided
    if colors_order is None:
        colors_order = [0]

    # Unpack forbids
    E_graph = None
    M_matrix = None
    if forbids is not None:
        E_graph, M_matrix = forbids

    # Reorder emitters_list to frozen family order
    emitters_ordered = _reorder_emitters(emitters_list)

    # Initialize working domain (deep copy)
    D = deepcopy(D0)

    # Stats accumulators
    admit_passes = 0
    ac3_passes = 0
    total_admit_prunes = 0
    total_ac3_prunes = 0
    empties = 0
    prunes_by_family: Dict[str, int] = {}

    # Monotone loop
    for iter_idx in range(1, MAX_PROPAGATION_ITERATIONS + 1):
        changed = False

        # (1) Admit pass in frozen family order
        admit_prunes_this = 0

        for family, A, S in emitters_ordered:
            # Skip T10_forbids (not an admit layer)
            if family == "T10_forbids":
                continue

            # Skip if absent or empty
            if not A or not S:
                continue

            # Scope-gated intersect
            family_prunes = _admit_intersect(
                D, A, S, colors_order, R_out, C_out
            )
            admit_prunes_this += family_prunes

            # Track prunes by family (accumulate across all iterations)
            prunes_by_family[family] = prunes_by_family.get(family, 0) + family_prunes

        admit_passes += 1
        total_admit_prunes += admit_prunes_this
        changed = changed or (admit_prunes_this > 0)

        # Early UNSAT check (any empty domain)
        empties = sum(1 for rc, mask in D.items() if mask == 0)
        if empties > 0:
            return "UNSAT", _seal_stats(
                admit_passes, ac3_passes, total_admit_prunes,
                total_ac3_prunes, empties, prunes_by_family, D, R_out, C_out, colors_order
            )

        # (2) AC-3 prune (if forbids present)
        if forbids is not None:
            changed_ac3, ac3_stats = ac3_prune(
                D, E_graph, M_matrix, colors_order, R_out, C_out
            )
            ac3_passes += 1
            total_ac3_prunes += ac3_stats["prunes"]
            empties = ac3_stats["empties"]
            changed = changed or changed_ac3

            if empties > 0:
                return "UNSAT", _seal_stats(
                    admit_passes, ac3_passes, total_admit_prunes,
                    total_ac3_prunes, empties, prunes_by_family, D, R_out, C_out, colors_order
                )

        # Fixed point?
        if not changed:
            return D, _seal_stats(
                admit_passes, ac3_passes, total_admit_prunes,
                total_ac3_prunes, 0, prunes_by_family, D, R_out, C_out, colors_order
            )

    # Cap guard
    return "FIXED_POINT_NOT_REACHED", _seal_stats(
        admit_passes, ac3_passes, total_admit_prunes,
        total_ac3_prunes, 0, prunes_by_family, D, R_out, C_out, colors_order
    )


# ============================================================================
# Helpers
# ============================================================================


def _reorder_emitters(
    emitters_list: List[Tuple[str, Optional[Dict[int, List[int]]], Optional[List[int]]]]
) -> List[Tuple[str, Optional[Dict[int, List[int]]], Optional[List[int]]]]:
    """
    Reorder emitters to frozen family order T1...T12.

    Each family may appear at most once. Raises ValueError if duplicates detected.

    Args:
        emitters_list: List of (family_name, A, S) in any order

    Returns:
        Reordered list following FROZEN_FAMILY_ORDER

    Raises:
        ValueError: If duplicate family names are detected

    Spec: WO-11 v1.6 frozen family order
    """
    # Validate: no duplicate family names
    family_names = [family for family, _, _ in emitters_list]
    duplicates = {name for name in family_names if family_names.count(name) > 1}

    if duplicates:
        raise ValueError(
            f"Duplicate emitter families detected: {sorted(duplicates)}. "
            f"Each Tx family must be unique by design; merge duplicates inside "
            f"the emitter before M4."
        )

    # Build dict for fast lookup (safe now; no duplicates)
    emitters_dict = {family: (A, S) for family, A, S in emitters_list}

    # Reorder according to frozen order
    result = []
    for family in FROZEN_FAMILY_ORDER:
        if family in emitters_dict:
            A, S = emitters_dict[family]
            result.append((family, A, S))

    return result


def _admit_intersect(
    D: Dict[Tuple[int, int], int],
    A: Dict[int, List[int]],
    S: List[int],
    colors_order: List[int],
    R_out: int,
    C_out: int
) -> int:
    """
    Scope-gated admit intersect for one emitter.

    For each pixel (r,c) where S[r]&(1<<c)==1:
      - Compute admit_mask = ⋁_k (A[color_k][r]&(1<<c) ? (1<<k) : 0)
      - Intersect: D[(r,c)] &= admit_mask
      - Count bit prunes

    Args:
        D: Domain tensor (modified in place)
        A: Admit planes (color -> row-masks)
        S: Scope mask (row-masks)
        colors_order: Ascending color list
        R_out, C_out: Canvas dimensions

    Returns:
        Number of bits pruned

    Spec: WO-11 v1.6 section 0 & 1
    """
    prunes = 0

    for r in range(R_out):
        scope_row = S[r] if r < len(S) else 0
        if scope_row == 0:
            continue

        for c in range(C_out):
            bit = 1 << c
            if not (scope_row & bit):
                continue

            # Build admit bitmask at (r,c) from color planes
            admit_mask = 0
            for k, color in enumerate(colors_order):
                plane = A.get(color)
                if plane and r < len(plane):
                    if (plane[r] & bit) != 0:
                        admit_mask |= (1 << k)

            # Intersect
            old = D.get((r, c), 0)
            new = old & admit_mask

            if new != old:
                D[(r, c)] = new
                # Count bit prunes
                prunes += bin(old ^ new).count("1")

    return prunes


def _seal_stats(
    admit_passes: int,
    ac3_passes: int,
    total_admit_prunes: int,
    total_ac3_prunes: int,
    empties: int,
    prunes_by_family: Dict[str, int],
    D: Dict[Tuple[int, int], int],
    R_out: int,
    C_out: int,
    colors_order: List[int]
) -> LFPStats:
    """
    Seal LFPStats with deterministic hashes.

    Args:
        admit_passes, ac3_passes, total_admit_prunes, total_ac3_prunes, empties: Stats
        prunes_by_family: Dict mapping family names to total prunes
        D: Final domain tensor
        R_out, C_out: Canvas dimensions
        colors_order: Color order for hash canonicality

    Returns:
        LFPStats

    Spec: WO-11 v1.6 receipts section
    """
    # Count singleton/multi/empty pixels for diagnostics
    singleton_pixels = 0
    multi_pixels = 0
    empty_pixels = 0

    for r in range(R_out):
        for c in range(C_out):
            mask = D.get((r, c), 0)
            if mask == 0:
                empty_pixels += 1
            elif mask & (mask - 1) == 0:  # exactly one bit set (power of 2)
                singleton_pixels += 1
            else:  # multiple bits set
                multi_pixels += 1

    domains_hash = _hash_domains(D, R_out, C_out, colors_order)
    section_hash = _hash_lfp_stats(
        admit_passes, ac3_passes, total_admit_prunes,
        total_ac3_prunes, empties, singleton_pixels, multi_pixels, empty_pixels,
        prunes_by_family, domains_hash
    )

    return LFPStats(
        admit_passes=admit_passes,
        ac3_passes=ac3_passes,
        total_admit_prunes=total_admit_prunes,
        total_ac3_prunes=total_ac3_prunes,
        empties=empties,
        singleton_pixels=singleton_pixels,
        multi_pixels=multi_pixels,
        empty_pixels=empty_pixels,
        lfp_prunes_by_family=prunes_by_family,
        domains_hash=domains_hash,
        section_hash=section_hash
    )


def _hash_domains(
    D: Dict[Tuple[int, int], int],
    R_out: int,
    C_out: int,
    colors_order: List[int]
) -> str:
    """
    Hash domain tensor in canonical row-major order.

    Prefix: (R_out, C_out, K) as 3 × 4-byte big-endian ints
    For each (r,c) in row-major: domain_bitmask as ceil(K/8) bytes big-endian

    Args:
        D: Domain tensor
        R_out, C_out: Canvas dimensions
        colors_order: Color order (K = len(colors_order))

    Returns:
        BLAKE3 hex digest

    Spec: WO-11 v1.6 section 0 (helpers)
    """
    K = len(colors_order)
    num_bytes = (K + 7) // 8

    # Prefix with dimensions
    result = b""
    result += R_out.to_bytes(4, "big")
    result += C_out.to_bytes(4, "big")
    result += K.to_bytes(4, "big")

    # Row-major domain bytes
    for r in range(R_out):
        for c in range(C_out):
            domain_mask = D.get((r, c), 0)
            result += domain_mask.to_bytes(num_bytes, "big")

    return blake3_hash(result)


def _hash_lfp_stats(
    admit_passes: int,
    ac3_passes: int,
    total_admit_prunes: int,
    total_ac3_prunes: int,
    empties: int,
    singleton_pixels: int,
    multi_pixels: int,
    empty_pixels: int,
    prunes_by_family: Dict[str, int],
    domains_hash: str
) -> str:
    """Hash LFP stats in deterministic order."""
    # Serialize prunes_by_family deterministically (sorted by key)
    family_str = ",".join(f"{k}:{v}" for k, v in sorted(prunes_by_family.items()))

    stats_str = (
        f"{admit_passes},{ac3_passes},{total_admit_prunes},"
        f"{total_ac3_prunes},{empties},{singleton_pixels},{multi_pixels},{empty_pixels},"
        f"{family_str},{domains_hash}"
    )
    return blake3_hash(stats_str.encode("utf-8"))
