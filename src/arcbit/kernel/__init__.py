"""
WO-01 & WO-02: Bit-Planes & Kernel Ops

Minimal kernel of pure operations on packed bit-planes.

Components:
  - planes: PACK/UNPACK (grid to bit-planes)
  - ops: SHIFT, POSE, BITWISE (AND/OR/ANDN)
  - period: minimal_period_row, minimal_period_1d, period_2d_planes (KMP-based)
"""

from .planes import (
    order_colors,
    pack_grid_to_planes,
    unpack_planes_to_grid
)
from .ops import (
    shift_plane,
    pose_plane,
    pose_inverse,
    plane_and,
    plane_or,
    plane_andn
)
from .period import (
    minimal_period_row,
    minimal_period_1d,
    period_2d_planes,
    period_receipts
)
from .frames import (
    canonicalize,
    apply_pose_anchor
)

__all__ = [
    # Planes
    "order_colors",
    "pack_grid_to_planes",
    "unpack_planes_to_grid",

    # Ops
    "shift_plane",
    "pose_plane",
    "pose_inverse",
    "plane_and",
    "plane_or",
    "plane_andn",

    # Period (WO-01 & WO-02)
    "minimal_period_row",
    "minimal_period_1d",
    "period_2d_planes",
    "period_receipts",

    # Frames (WO-03)
    "canonicalize",
    "apply_pose_anchor",

    # Receipts
    "kernel_receipts",
]


def kernel_receipts(section_label: str, fixtures: list[dict]) -> dict:
    """
    Generate receipts for kernel operations using fixed fixtures.

    Args:
        section_label: ASCII identifier (e.g., "WO-01-kernel").
        fixtures: List of dicts with keys:
            - "grid": list[list[int]]
            - "H": int
            - "W": int
            - "colors": list[int]
            - "label": str (description)

    Returns:
        dict: Receipt digest with kernel operation proofs.

    Spec:
        WO-01 receipts section.
        Uses WO-00 Receipts for logging.
    """
    from ..core import Receipts, blake3_hash
    from ..core.bytesio import serialize_grid_be_row_major, serialize_planes_be_row_major

    receipts = Receipts(section_label)

    # 1. kernel.params_hash: Hash of D4 mapping table + inverses
    pose_ids = ["I", "R90", "R180", "R270", "FX", "FXR90", "FXR180", "FXR270"]
    # NOTE: FXR90 and FXR270 are self-inverse (verified algebraically)
    pose_inverses = {
        "I": "I", "R90": "R270", "R180": "R180", "R270": "R90",
        "FX": "FX", "FXR90": "FXR90", "FXR180": "FXR180", "FXR270": "FXR270"
    }
    d4_table = {
        "pose_ids": pose_ids,
        "inverses": pose_inverses,
        "coord_mapping": {
            "I": "r=r', c=c'",
            "R90": "r=H-1-c', c=r'",
            "R180": "r=H-1-r', c=W-1-c'",
            "R270": "r=c', c=W-1-r'",
            "FX": "r=r', c=W-1-c'",
            "FXR90": "r=H-1-c', c=W-1-r'",
            "FXR180": "r=H-1-r', c=c'",
            "FXR270": "r=c', c=r'"
        }
    }
    import json
    d4_bytes = json.dumps(d4_table, sort_keys=True).encode('utf-8')
    receipts.put("kernel.params_hash", blake3_hash(d4_bytes))

    # 2. pack_consistency: Pack/unpack round-trip for each fixture
    pack_consistency = []
    for fix in fixtures:
        G = fix["grid"]
        H, W = fix["H"], fix["W"]
        colors = fix["colors"]
        label = fix["label"]

        # Pack to planes
        planes = pack_grid_to_planes(G, H, W, colors)

        # Unpack back to grid
        G_reconstructed = unpack_planes_to_grid(planes, H, W, colors)

        # Check round-trip
        match = (G == G_reconstructed)

        # Serialize planes for hash
        planes_bytes = serialize_planes_be_row_major(planes, H, W, colors)
        planes_hash = blake3_hash(planes_bytes)

        pack_consistency.append({
            "label": label,
            "planes_hash": planes_hash,
            "roundtrip_ok": match
        })

    receipts.put("pack_consistency", pack_consistency)

    # 3. pose_inverse_ok: Test pose o inv = id for each fixture
    pose_inverse_tests = []
    for fix in fixtures:
        G = fix["grid"]
        H, W = fix["H"], fix["W"]
        colors = fix["colors"]
        label = fix["label"]

        planes = pack_grid_to_planes(G, H, W, colors)

        # Test each pose ID
        for pid in pose_ids:
            # Pick first color's plane for testing
            plane = planes[colors[0]]

            # Apply pose
            plane_fwd, H_fwd, W_fwd = pose_plane(plane, pid, H, W)

            # Apply inverse
            inv_pid = pose_inverse(pid)
            plane_back, H_back, W_back = pose_plane(plane_fwd, inv_pid, H_fwd, W_fwd)

            # Check round-trip
            ok = (H_back == H and W_back == W and plane_back == plane)

            pose_inverse_tests.append({
                "label": label,
                "pid": pid,
                "inv_pid": inv_pid,
                "ok": ok
            })

    all_ok = all(t["ok"] for t in pose_inverse_tests)
    receipts.put("pose_inverse_ok", all_ok)
    receipts.put("pose_inverse_tests", pose_inverse_tests)

    # 4. shift_boundary_counts: Bits dropped at edges
    shift_tests = []
    for fix in fixtures:
        G = fix["grid"]
        H, W = fix["H"], fix["W"]
        colors = fix["colors"]
        label = fix["label"]

        planes = pack_grid_to_planes(G, H, W, colors)
        plane = planes[colors[0]]

        # Count initial bits
        initial_bits = sum(bin(mask).count('1') for mask in plane)

        # Shift right by 1 and count bits
        plane_r1 = shift_plane(plane, 0, 1, H, W)
        bits_r1 = sum(bin(mask).count('1') for mask in plane_r1)
        dropped_r1 = initial_bits - bits_r1

        # Shift down by 1 and count bits
        plane_d1 = shift_plane(plane, 1, 0, H, W)
        bits_d1 = sum(bin(mask).count('1') for mask in plane_d1)
        dropped_d1 = initial_bits - bits_d1

        shift_tests.append({
            "label": label,
            "initial_bits": initial_bits,
            "shift_right_1_dropped": dropped_r1,
            "shift_down_1_dropped": dropped_d1
        })

    receipts.put("shift_boundary_counts", shift_tests)

    # 5. period_kmp_examples: Canonical rows
    period_examples = [
        {"label": "solid_0", "mask": 0b000000, "W": 6},
        {"label": "solid_1", "mask": 0b111111, "W": 6},
        {"label": "stripe_2", "mask": 0b101010, "W": 6},
        {"label": "stripe_3", "mask": 0b110110, "W": 6},
        {"label": "no_period", "mask": 0b101011, "W": 6},
    ]

    period_results = []
    for ex in period_examples:
        p = minimal_period_row(ex["mask"], ex["W"])
        period_results.append({
            "label": ex["label"],
            "mask": ex["mask"],
            "W": ex["W"],
            "period": p
        })

    receipts.put("period_kmp_examples", period_results)

    return receipts.digest()
