#!/usr/bin/env python3
"""
WO-11 LFP Propagator - Integration Test

Demonstrates full LFP computation with multiple emitters and forbids.

Spec: WO-11 v1.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arcbit.emitters.lfp import lfp_propagate
from src.arcbit.emitters.forbids import build_4neighbor_graph, learn_forbids


print("=" * 70)
print("WO-11 LFP PROPAGATOR - INTEGRATION TEST")
print("=" * 70)

# =============================================================================
# Scenario: Checkerboard constraint propagation
# =============================================================================

print("\n" + "=" * 70)
print("SCENARIO: Checkerboard Constraint Propagation")
print("=" * 70)

colors_order = [0, 1, 2]
R_out, C_out = 4, 4

# Initial domain: all colors at all pixels
D0 = {(r, c): 0b111 for r in range(R_out) for c in range(C_out)}

print(f"\nInitial domain: {R_out}×{C_out} canvas, all pixels have colors {colors_order}")

# =============================================================================
# Emitters Setup
# =============================================================================

# T1_witness: Partial checkerboard (top-left 2×2)
# Forces (0,0)→1, (0,1)→2, (1,0)→2, (1,1)→1
A_witness = {
    0: [0b0000, 0b0000, 0b0000, 0b0000],
    1: [0b0001, 0b0010, 0b0000, 0b0000],  # (0,0) and (1,1)
    2: [0b0010, 0b0001, 0b0000, 0b0000],  # (0,1) and (1,0)
}
S_witness = [0b0011, 0b0011, 0b0000, 0b0000]  # Top-left 2×2 only

print("\nT1_witness: Checkerboard pattern on top-left 2×2")
print("  (0,0)→1, (0,1)→2, (1,0)→2, (1,1)→1")

# T3_lattice: Periodic extension (period 2×2) - forces same pattern everywhere
# Since we have a 4×4 grid with period 2×2, this forces all (even,even) and (odd,odd) to color 1
# and all (even,odd) and (odd,even) to color 2
A_lattice = {
    0: [0b0000, 0b0000, 0b0000, 0b0000],
    1: [0b0101, 0b1010, 0b0101, 0b1010],  # (0,0), (0,2), (1,1), (1,3), (2,0), (2,2), (3,1), (3,3)
    2: [0b1010, 0b0101, 0b1010, 0b0101],  # (0,1), (0,3), (1,0), (1,2), (2,1), (2,3), (3,0), (3,2)
}
S_lattice = [0b1111, 0b1111, 0b1111, 0b1111]  # Full coverage

print("\nT3_lattice: Periodic extension (2×2) across full 4×4 canvas")
print("  Forces checkerboard pattern everywhere")

# Emitters list (frozen order: T1, T3)
emitters_list = [
    ("T1_witness", A_witness, S_witness),
    ("T3_lattice", A_lattice, S_lattice),
]

print("\nEmitters enabled: T1_witness, T3_lattice")

# =============================================================================
# Forbids Setup
# =============================================================================

# Learn forbids from checkerboard training
Y_train = [
    [1, 2, 1, 2],
    [2, 1, 2, 1],
    [1, 2, 1, 2],
    [2, 1, 2, 1],
]

M, forbids_receipt = learn_forbids([Y_train], [0], colors_order)
E_graph = build_4neighbor_graph(R_out, C_out)
forbids = (E_graph, M)

print("\nForbids learned from checkerboard training:")
print(f"  M[1] forbids: {M[1]} (Type 1: universal differ)")
print(f"  M[2] forbids: {M[2]} (Type 1: universal differ)")
print(f"  Total forbid edges: {len(E_graph)}")

# =============================================================================
# Run LFP Propagation
# =============================================================================

print("\n" + "=" * 70)
print("RUNNING LFP PROPAGATION...")
print("=" * 70)

D_final, stats = lfp_propagate(
    D0, emitters_list, forbids=forbids, colors_order=colors_order,
    R_out=R_out, C_out=C_out
)

print(f"\n✅ CONVERGED TO FIXED POINT")
print(f"\nStatistics:")
print(f"  Admit passes: {stats['admit_passes']}")
print(f"  AC-3 passes: {stats['ac3_passes']}")
print(f"  Total admit prunes: {stats['total_admit_prunes']}")
print(f"  Total AC-3 prunes: {stats['total_ac3_prunes']}")
print(f"  Empty domains: {stats['empties']}")
print(f"  Domains hash: {stats['domains_hash'][:32]}...")

# =============================================================================
# Verify Results
# =============================================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Decode final domains to check singletons
def decode_domain(mask, colors_order):
    """Decode bitmask to list of colors."""
    return [colors_order[k] for k in range(len(colors_order)) if mask & (1 << k)]

print("\nFinal domains (decoded):")
for r in range(R_out):
    row_str = ""
    for c in range(C_out):
        domain = decode_domain(D_final[(r, c)], colors_order)
        if len(domain) == 1:
            row_str += f" {domain[0]} "
        else:
            row_str += f"[{','.join(map(str, domain))}]"
    print(f"  Row {r}: {row_str}")

# Check invariants
print("\nInvariants:")

# 1. All domains should be singletons (converged to unique solution)
all_singletons = all(
    bin(D_final[(r, c)]).count("1") == 1
    for r in range(R_out) for c in range(C_out)
)
print(f"  ✓ All singletons: {all_singletons}")

# 2. Should match checkerboard pattern
expected_pattern = [[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]]
matches_pattern = all(
    D_final[(r, c)] == (1 << colors_order.index(expected_pattern[r][c]))
    for r in range(R_out) for c in range(C_out)
)
print(f"  ✓ Matches checkerboard: {matches_pattern}")

# 3. No adjacent equal colors (forbids satisfied)
satisfies_forbids = True
for pr, pc, qr, qc in E_graph:
    domain_p = D_final[(pr, pc)]
    domain_q = D_final[(qr, qc)]
    # Extract singleton colors
    color_p = colors_order[[i for i in range(len(colors_order)) if domain_p & (1 << i)][0]]
    color_q = colors_order[[i for i in range(len(colors_order)) if domain_q & (1 << i)][0]]
    if color_p == color_q:
        satisfies_forbids = False
        break
print(f"  ✓ Satisfies forbids: {satisfies_forbids}")

# 4. Determinism check
D_check, stats_check = lfp_propagate(
    D0, emitters_list, forbids=forbids, colors_order=colors_order,
    R_out=R_out, C_out=C_out
)
deterministic = (stats['domains_hash'] == stats_check['domains_hash'])
print(f"  ✓ Deterministic: {deterministic}")

# =============================================================================
# Summary
# =============================================================================

if all_singletons and matches_pattern and satisfies_forbids and deterministic:
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 70)
    print("\nWO-11 LFP propagator correctly:")
    print("  • Applied admit constraints in frozen family order (T1→T3)")
    print("  • Propagated arc consistency via AC-3 with learned forbids")
    print("  • Converged to unique fixed point (checkerboard solution)")
    print("  • Maintained determinism across runs")
    sys.exit(0)
else:
    print("\n❌ INTEGRATION TEST FAILED")
    sys.exit(1)
