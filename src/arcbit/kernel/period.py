"""
WO-01 & WO-02: Minimal Period Detection (KMP)

1D and 2D period detection using exact integer KMP (Knuth-Morris-Pratt).

Components:
  - minimal_period_row: 1D period for binary row (WO-01)
  - minimal_period_1d: Alias for minimal_period_row (WO-02)
  - period_2d_planes: 2D period for multi-color planes (WO-02)

Spec: WO-01 section 6, WO-02.
"""

import math
from typing import Dict, List, Tuple, Optional


def minimal_period_row(mask: int, W: int) -> int | None:
    """
    Return p (2 <= p <= W) if the row's bitstring is an exact repetition
    with minimal period p; else None.

    Args:
        mask: Row mask (W least-significant bits).
        W: Width (number of columns).

    Returns:
        int | None: Minimal non-trivial period p (p >= 2), or None.

    Spec:
        WO-01 section 6: PERIOD.

        Definition:
          - Convert row to length-W bitstring s[0..W-1] with s[j] = ((mask >> j) & 1)
          - Compute prefix function pi (textbook KMP)
          - Let t = W - pi[W-1]
          - If t >= 2 and t < W and W % t == 0: return t
          - Else: return None

        Period 1 (constant rows) is EXCLUDED (returns None).

    Examples:
        >>> minimal_period_row(0b101010, 6)  # "101010" → period 2
        2
        >>> minimal_period_row(0b110110, 6)  # "110110" → period 3
        3
        >>> minimal_period_row(0b111111, 6)  # constant row → None (period 1 excluded)
        None
        >>> minimal_period_row(0b000000, 6)  # constant row → None (period 1 excluded)
        None
        >>> minimal_period_row(0b101011, 6)  # no period
        None
    """
    if W == 0:
        return None

    # Convert mask to bitstring s[0..W-1]
    s = [(mask >> j) & 1 for j in range(W)]

    # Compute KMP prefix function
    pi = _kmp_prefix_function(s)

    # Minimal period candidate
    t = W - pi[W - 1]

    # Check if t is a valid non-trivial period (p >= 2)
    # Period 1 (constant rows) is excluded per spec
    if t >= 2 and t < W and W % t == 0:
        return t
    else:
        return None


def _kmp_prefix_function(s: list[int]) -> list[int]:
    """
    Compute KMP prefix function for a sequence.

    The prefix function pi[i] is the length of the longest proper prefix
    of s[0..i] that is also a suffix of s[0..i].

    Args:
        s: Sequence (list of integers, typically 0/1).

    Returns:
        list[int]: Prefix function values pi[0..len(s)-1].

    Spec:
        Textbook KMP (Knuth-Morris-Pratt) prefix function.
        See Cormen et al., "Introduction to Algorithms", section 32.4.

    Complexity:
        O(len(s)) time, O(len(s)) space.
    """
    n = len(s)
    if n == 0:
        return []

    pi = [0] * n
    k = 0  # length of previous longest prefix suffix

    for q in range(1, n):
        # While we have a mismatch and k > 0, fall back
        while k > 0 and s[k] != s[q]:
            k = pi[k - 1]

        # If we have a match, increment k
        if s[k] == s[q]:
            k += 1

        pi[q] = k

    return pi


# ============================================================================
# WO-02: 1D Period (alias) + 2D Multi-Color Period
# ============================================================================

def minimal_period_1d(mask_bits: int, W: int) -> int | None:
    """
    Alias for minimal_period_row. Return minimal period p (2 <= p <= W).

    Args:
        mask_bits: Row mask (W least-significant bits).
        W: Width (number of columns).

    Returns:
        int | None: Minimal non-trivial period p (p >= 2), or None.

    Spec:
        WO-02: minimal_period_1d.
        Identical to minimal_period_row from WO-01.
    """
    return minimal_period_row(mask_bits, W)


def period_2d_planes(
    planes: Dict[int, List[int]],
    H: int,
    W: int,
    colors_order: List[int]
) -> Tuple[Optional[int], Optional[int], List[List[int]]]:
    """
    Compute minimal 2D periods (p_r, p_c) for multi-color grid.

    Each period is >= 2 (proper/non-trivial) or None.
    Also returns residue-class masks with phase fixed at (0,0).

    Args:
        planes: Dict mapping color → list of H row masks.
        H: Height (number of rows).
        W: Width (number of columns).
        colors_order: Ordered list of colors (ascending integers).

    Returns:
        Tuple of (p_r, p_c, residues):
          - p_r: Row period (>= 2) or None
          - p_c: Column period (>= 2) or None
          - residues: List of residue masks (list[int] per residue)
                      Empty if both periods are None.
                      Ordered row-major: residue i*p_c + j for i in [0..p_r-1], j in [0..p_c-1]

    Spec:
        WO-02: period_2d_planes.

        Algorithm:
          1. Build K-tuple symbols per (row, col) across all colors
          2. For each row: compute minimal 1D period of column symbols (KMP)
          3. Take LCM of non-None row periods → p_c_candidate
          4. Validate p_c globally; if fails → p_c = None
          5. Symmetric for rows (transpose) → p_r
          6. Build residue masks with phase (0,0)

    Raises:
        ValueError: If plane shapes mismatch or bits outside [0..W-1].

    Invariants:
        - Pure tuple equality (no hashing)
        - Phase fixed at (0,0)
        - Deterministic
        - No minted bits (only masks)
    """
    # Validate inputs
    K = len(colors_order)
    for color in colors_order:
        if color not in planes:
            raise ValueError(f"Color {color} missing from planes")
        if len(planes[color]) != H:
            raise ValueError(f"Plane for color {color} has wrong height: {len(planes[color])} != {H}")
        # Check no bits outside [0..W-1]
        for r, mask in enumerate(planes[color]):
            if mask >> W != 0:
                raise ValueError(f"Plane color {color} row {r} has bits outside [0..{W-1}]: {mask:b}")

    # Edge cases: small grids
    if H < 2 or W < 2:
        # Cannot have proper period (>= 2) on axis with length < 2
        p_r = None if H < 2 else None  # will be computed if H >= 2
        p_c = None if W < 2 else None  # will be computed if W >= 2

        # If either dimension too small, no proper 2D period possible
        if H < 2 or W < 2:
            return (None, None, [])

    # ========================================================================
    # Column period (p_c): analyze rows
    # ========================================================================

    row_periods = []  # Non-None periods from each row

    for r in range(H):
        # Build symbol sequence for row r across columns
        # sym_r[c] = tuple of bits (one per color) at (r, c)
        sym_r = []
        for c in range(W):
            k_tuple = tuple((planes[color][r] >> c) & 1 for color in colors_order)
            sym_r.append(k_tuple)

        # Compute minimal period of this symbol sequence
        t_r = _minimal_period_tuples(sym_r)

        if t_r is not None:
            row_periods.append(t_r)

    # Compute LCM of row periods
    if len(row_periods) == 0:
        p_c_candidate = None
    else:
        p_c_candidate = _lcm_list(row_periods)

    # Validate p_c globally
    p_c = None
    if p_c_candidate is not None:
        valid = True
        for r in range(H):
            # Build symbols for this row
            sym_r = []
            for c in range(W):
                k_tuple = tuple((planes[color][r] >> c) & 1 for color in colors_order)
                sym_r.append(k_tuple)

            # Check periodicity: sym_r[c] == sym_r[c + p_c] for all valid c
            for c in range(W - p_c_candidate):
                if sym_r[c] != sym_r[c + p_c_candidate]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            p_c = p_c_candidate

    # ========================================================================
    # Row period (p_r): analyze columns
    # ========================================================================

    col_periods = []  # Non-None periods from each column

    for c in range(W):
        # Build symbol sequence for column c across rows
        # sym_c[r] = tuple of bits (one per color) at (r, c)
        sym_c = []
        for r in range(H):
            k_tuple = tuple((planes[color][r] >> c) & 1 for color in colors_order)
            sym_c.append(k_tuple)

        # Compute minimal period of this symbol sequence
        t_c = _minimal_period_tuples(sym_c)

        if t_c is not None:
            col_periods.append(t_c)

    # Compute LCM of column periods
    if len(col_periods) == 0:
        p_r_candidate = None
    else:
        p_r_candidate = _lcm_list(col_periods)

    # Validate p_r globally
    p_r = None
    if p_r_candidate is not None:
        valid = True
        for c in range(W):
            # Build symbols for this column
            sym_c = []
            for r in range(H):
                k_tuple = tuple((planes[color][r] >> c) & 1 for color in colors_order)
                sym_c.append(k_tuple)

            # Check periodicity: sym_c[r] == sym_c[r + p_r] for all valid r
            for r in range(H - p_r_candidate):
                if sym_c[r] != sym_c[r + p_r_candidate]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            p_r = p_r_candidate

    # ========================================================================
    # Build residue masks
    # ========================================================================

    residues = _build_residue_masks(p_r, p_c, H, W)

    return (p_r, p_c, residues)


def _minimal_period_tuples(symbols: List[tuple]) -> Optional[int]:
    """
    Compute minimal period of a sequence of tuples using KMP.

    Args:
        symbols: List of tuples (K-tuples of bits).

    Returns:
        int | None: Minimal period p (>= 2), or None.

    Spec:
        WO-02: KMP over tuple equality (not hash).
        Same algorithm as minimal_period_1d but for tuples.
    """
    n = len(symbols)
    if n == 0:
        return None
    if n < 2:
        return None  # Cannot have proper period >= 2

    # Compute KMP prefix function over tuple equality
    pi = _kmp_prefix_function_tuples(symbols)

    # Minimal period candidate
    t = n - pi[n - 1]

    # Check if t is a valid non-trivial period (p >= 2)
    if t >= 2 and t < n and n % t == 0:
        return t
    else:
        return None


def _kmp_prefix_function_tuples(symbols: List[tuple]) -> List[int]:
    """
    Compute KMP prefix function for a sequence of tuples.

    Uses tuple EQUALITY (not hash) for comparisons.

    Args:
        symbols: List of tuples.

    Returns:
        list[int]: Prefix function values.

    Spec:
        WO-02: Pure equality logic, no hashing.
    """
    n = len(symbols)
    if n == 0:
        return []

    pi = [0] * n
    k = 0

    for q in range(1, n):
        # Fall back on mismatch
        while k > 0 and symbols[k] != symbols[q]:
            k = pi[k - 1]

        # Match: increment
        if symbols[k] == symbols[q]:
            k += 1

        pi[q] = k

    return pi


def _lcm_list(numbers: List[int]) -> int:
    """
    Compute LCM of a list of positive integers.

    Args:
        numbers: List of positive integers.

    Returns:
        int: LCM of all numbers.

    Spec:
        WO-02: Exact integer LCM.
    """
    if len(numbers) == 0:
        raise ValueError("Cannot compute LCM of empty list")

    result = numbers[0]
    for n in numbers[1:]:
        result = _lcm_two(result, n)

    return result


def _lcm_two(a: int, b: int) -> int:
    """
    Compute LCM of two positive integers.

    Args:
        a, b: Positive integers.

    Returns:
        int: LCM(a, b).

    Spec:
        LCM(a, b) = abs(a * b) // GCD(a, b)
    """
    return abs(a * b) // math.gcd(a, b)


def _build_residue_masks(
    p_r: Optional[int],
    p_c: Optional[int],
    H: int,
    W: int
) -> List[List[int]]:
    """
    Build residue-class masks with phase (0,0).

    Args:
        p_r: Row period (>= 2) or None.
        p_c: Column period (>= 2) or None.
        H: Height.
        W: Width.

    Returns:
        List of residue masks (each is list[int] of length H).
        Ordered row-major: residue i*p_c + j for i in [0..p_r-1], j in [0..p_c-1].

    Spec:
        WO-02: Phase fixed at (0,0).
        Residue mask for (i, j) has bit set at (r, c) iff:
          r % p_r == i and c % p_c == j
    """
    if p_r is None and p_c is None:
        return []

    # Internally treat missing period as 1 for mask construction
    p_r_internal = p_r if p_r is not None else 1
    p_c_internal = p_c if p_c is not None else 1

    residues = []

    for i in range(p_r_internal):
        for j in range(p_c_internal):
            # Build mask for residue (i, j)
            mask_rows = []
            for r in range(H):
                row_mask = 0
                for c in range(W):
                    # Check if (r, c) belongs to residue (i, j)
                    if (r % p_r_internal == i) and (c % p_c_internal == j):
                        row_mask |= (1 << c)
                mask_rows.append(row_mask)

            residues.append(mask_rows)

    return residues
