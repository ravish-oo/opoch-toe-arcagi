"""
WO-01 Component: Minimal Period Detection (KMP)

1D minimal period detection using exact integer KMP (Knuth-Morris-Pratt).

Spec: WO-01 section 6.
"""


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
