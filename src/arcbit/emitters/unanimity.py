"""
WO-08 Component: Unanimity Emitter

Emit singleton color admits where all included trainings agree.

Spec: WO-08 v1.6 (section 3)
"""

from typing import List, Dict, Tuple, TypedDict
from ..core.hashing import blake3_hash


class UnanimityReceipt(TypedDict):
    included_train_ids: List[int]
    unanimous_pixels: int
    total_covered_pixels: int
    empty_scope_pixels: int
    unanimity_hash: str
    scope_hash: str


def emit_unity(
    A_out_list: List[Dict[int, List[int]]],
    S_out_list: List[List[int]],
    colors_order: List[int],
    R_out: int,
    C_out: int,
) -> Tuple[Dict[int, List[int]], List[int], UnanimityReceipt]:
    """
    Emit unanimous color admits where all included trainings agree.

    For each pixel p:
      - Let I_p = {i : S_out_i[p] == 1} (included trainings at p)
      - Let U_p = {u : A_out_{i,u}[p] == 1, i ∈ I_p} (colors from included trainings)
      - If |U_p| == 1 (singleton {u}): S_uni[p] = 1, A_uni_u[p] = 1
      - Else (empty or disagree): S_uni[p] = 0

    Args:
        A_out_list: Per-training admits (color -> row masks)
        S_out_list: Per-training scope masks
        colors_order: Ascending color list
        R_out, C_out: Canvas dimensions

    Returns:
        A_uni: Unanimous admits (singleton at unanimous pixels)
        S_uni: Unanimous scope (1 where all agree)
        receipt: UnanimityReceipt

    Spec: WO-08 v1.6 section 3
    """
    # Initialize unanimity admits and scope
    A_uni = {c: [0] * R_out for c in colors_order}
    S_uni = [0] * R_out

    # Track statistics
    unanimous_pixels = 0
    total_covered_pixels = 0
    empty_scope_pixels = 0

    # Determine which trainings are included (have any scope)
    included_train_ids = []
    for i, S_i in enumerate(S_out_list):
        has_scope = any(row != 0 for row in S_i)
        if has_scope:
            included_train_ids.append(i)

    # Process each pixel
    for r in range(R_out):
        for c_bit in range(C_out):
            bit = 1 << c_bit

            # Find trainings that speak at this pixel
            I_p = []
            for i in included_train_ids:
                if S_out_list[i][r] & bit:
                    I_p.append(i)

            if not I_p:
                # No training speaks at this pixel
                empty_scope_pixels += 1
                continue

            # This pixel is covered by at least one training
            total_covered_pixels += 1

            # Collect colors from all speaking trainings
            U_p = set()
            for i in I_p:
                # Find which color is admitted by training i at pixel (r, c_bit)
                for color in colors_order:
                    if A_out_list[i][color][r] & bit:
                        U_p.add(color)

            # Check for unanimity
            if len(U_p) == 1:
                # Singleton - unanimous!
                u = list(U_p)[0]
                S_uni[r] |= bit
                A_uni[u][r] |= bit
                unanimous_pixels += 1
            # else: disagreement or empty → S_uni[p] remains 0

    # Compute hashes
    unanimity_hash = _hash_planes(A_uni, R_out, C_out, colors_order)
    scope_hash = _hash_scope(S_uni, C_out)

    receipt = UnanimityReceipt(
        included_train_ids=included_train_ids,
        unanimous_pixels=unanimous_pixels,
        total_covered_pixels=total_covered_pixels,
        empty_scope_pixels=empty_scope_pixels,
        unanimity_hash=unanimity_hash,
        scope_hash=scope_hash,
    )

    return A_uni, S_uni, receipt


# Alias for backward compatibility
emit_unanimity = emit_unity


# ============================================================================
# Helpers
# ============================================================================


def _hash_planes(
    planes: Dict[int, List[int]], R: int, C: int, colors_order: List[int]
) -> str:
    """Hash color planes in BE row-major order."""
    result = b""
    for c in colors_order:
        plane = planes.get(c, [0] * R)
        for row_mask in plane:
            result += row_mask.to_bytes((C + 7) // 8, "big")
    return blake3_hash(result)


def _hash_scope(S: List[int], C: int) -> str:
    """Hash scope mask in BE row-major order."""
    result = b""
    for row_mask in S:
        result += row_mask.to_bytes((C + 7) // 8, "big")
    return blake3_hash(result)
