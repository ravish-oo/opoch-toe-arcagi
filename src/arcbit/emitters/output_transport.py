"""
WO-08 Component: Output Transport Emitter

Normalize each training output to the working canvas via integer replication
or exact block-constancy decimation, then transport to test output frame.

Spec: WO-08 v1.6
"""

from typing import List, Dict, Tuple, Optional, TypedDict
from ..core import Receipts
from ..core.hashing import blake3_hash
from ..kernel.planes import pack_grid_to_planes
from ..kernel.ops import pose_plane, shift_plane, pose_compose, pose_inverse


class TransportReceipt(TypedDict):
    train_id: int
    norm_kind: str  # "replicate" | "decimate" | "silent"
    s_r: Optional[int]
    s_c: Optional[int]
    block_constancy_ok: Optional[bool]
    pose_src: str
    anchor_src: Tuple[int, int]
    pose_dst: str
    anchor_dst: Tuple[int, int]
    scope_bits: int
    transport_hash: str


def emit_output_transport(
    Y_list: List[List[List[int]]],
    frames_out: List[Tuple[str, Tuple[int, int]]],
    R_out: int,
    C_out: int,
    colors_order: List[int],
    pi_out_star: Tuple[str, Tuple[int, int]],
) -> Tuple[List[Dict[int, List[int]]], List[List[int]], List[TransportReceipt], Dict]:
    """
    Normalize training outputs to working canvas and transport to test frame.

    Args:
        Y_list: Training outputs (each H_i × W_i)
        frames_out: Π_out_i for each training (pose_id, anchor)
        R_out, C_out: Working canvas dimensions
        colors_order: Ascending color list
        pi_out_star: Π_out* test output frame (pose_id, anchor)

    Returns:
        A_out_list: Per-training admits (singleton color planes)
        S_out_list: Per-training scope masks
        per_train_receipts: List of TransportReceipt
        section_receipt: Overall transport section

    Spec: WO-08 v1.6
    """
    receipts = Receipts("output_transport")

    A_out_list = []
    S_out_list = []
    per_train_receipts = []
    n_included = 0

    pose_dst, anchor_dst = pi_out_star

    for i, Y_i in enumerate(Y_list):
        H_i = len(Y_i)
        W_i = len(Y_i[0]) if Y_i else 0

        pose_src, anchor_src = frames_out[i]

        # Step 1: Normalize Y_i to (R_out, C_out)
        Y_norm, norm_kind, s_r, s_c, block_ok = _normalize_to_canvas(
            Y_i, H_i, W_i, R_out, C_out
        )

        # Step 2: Transport to test output frame (if not silent)
        if norm_kind == "silent":
            # Silent training: no scope, no admits
            A_i = {c: [0] * R_out for c in colors_order}
            S_i = [0] * R_out
            transport_hash = blake3_hash(b"silent")
        else:
            # Build color planes from normalized output
            planes_norm = pack_grid_to_planes(Y_norm, R_out, C_out, colors_order)

            # Transport from Π_out_i to Π_out*
            A_i, S_i = _transport_to_test_frame(
                planes_norm,
                R_out,
                C_out,
                pose_src,
                anchor_src,
                pose_dst,
                anchor_dst,
                colors_order,
            )

            # Compute transport hash
            transport_bytes = _serialize_transport(A_i, S_i, R_out, C_out, colors_order)
            transport_hash = blake3_hash(transport_bytes)
            n_included += 1

        A_out_list.append(A_i)
        S_out_list.append(S_i)

        # Build receipt for this training
        receipt = TransportReceipt(
            train_id=i,
            norm_kind=norm_kind,
            s_r=s_r,
            s_c=s_c,
            block_constancy_ok=block_ok,
            pose_src=pose_src,
            anchor_src=anchor_src,
            pose_dst=pose_dst,
            anchor_dst=anchor_dst,
            scope_bits=_popcount_scope(S_i),
            transport_hash=transport_hash,
        )
        per_train_receipts.append(receipt)

    # Compute overall transports hash
    all_hashes = b"".join(r["transport_hash"].encode("utf-8") for r in per_train_receipts)
    transports_hash = blake3_hash(all_hashes)

    # Build section receipt
    receipts.put("transports", per_train_receipts)
    receipts.put("n_included", n_included)
    receipts.put("transports_hash", transports_hash)

    section_receipt = receipts.digest()

    return A_out_list, S_out_list, per_train_receipts, section_receipt


# ============================================================================
# Normalization (Replicate / Decimate / Silent)
# ============================================================================


def _normalize_to_canvas(
    Y: List[List[int]], H: int, W: int, R_out: int, C_out: int
) -> Tuple[List[List[int]], str, Optional[int], Optional[int], Optional[bool]]:
    """
    Normalize Y to (R_out, C_out) via replication or decimation.

    Returns:
        Y_norm: Normalized grid (R_out × C_out)
        norm_kind: "replicate" | "decimate" | "silent"
        s_r, s_c: Factors (or None if silent)
        block_ok: True only for decimate when verified (None otherwise)

    Spec: WO-08 section 1
    """
    # Try replication: R_out = s_r * H and C_out = s_c * W
    if R_out % H == 0 and C_out % W == 0:
        s_r = R_out // H
        s_c = C_out // W
        Y_norm = _replicate(Y, H, W, s_r, s_c)
        return Y_norm, "replicate", s_r, s_c, None

    # Try decimation: H = s_r * R_out and W = s_c * C_out
    if H % R_out == 0 and W % C_out == 0:
        s_r = H // R_out
        s_c = W // C_out
        Y_norm, block_ok = _decimate(Y, H, W, s_r, s_c, R_out, C_out)
        if block_ok:
            return Y_norm, "decimate", s_r, s_c, True
        else:
            # Block constancy failed → silent
            return [], "silent", s_r, s_c, False

    # No exact integer relation → silent
    return [], "silent", None, None, None


def _replicate(
    Y: List[List[int]], H: int, W: int, s_r: int, s_c: int
) -> List[List[int]]:
    """
    Integer replication (Kronecker product).

    Y_i↑[r,c] = Y_i[⌊r/s_r⌋, ⌊c/s_c⌋]

    Spec: WO-08 section 1 (replicate)
    """
    R_out = s_r * H
    C_out = s_c * W

    Y_norm = []
    for r in range(R_out):
        row = []
        r_src = r // s_r
        for c in range(C_out):
            c_src = c // s_c
            row.append(Y[r_src][c_src])
        Y_norm.append(row)

    return Y_norm


def _decimate(
    Y: List[List[int]], H: int, W: int, s_r: int, s_c: int, R_out: int, C_out: int
) -> Tuple[List[List[int]], bool]:
    """
    Exact block-constancy decimation.

    Every s_r × s_c block must be constant.

    Returns:
        Y_norm: Decimated grid (R_out × C_out) if successful, else []
        block_ok: True if all blocks constant, False otherwise

    Spec: WO-08 section 1 (decimate)
    """
    Y_norm = []

    for r_out in range(R_out):
        row = []
        for c_out in range(C_out):
            # Extract s_r × s_c block
            r_start = r_out * s_r
            c_start = c_out * s_c

            # Check block constancy
            block_color = Y[r_start][c_start]
            is_constant = True

            for dr in range(s_r):
                for dc in range(s_c):
                    if Y[r_start + dr][c_start + dc] != block_color:
                        is_constant = False
                        break
                if not is_constant:
                    break

            if not is_constant:
                # Block constancy failed
                return [], False

            row.append(block_color)
        Y_norm.append(row)

    return Y_norm, True


# ============================================================================
# Transport (D4 Pose + Anchor Shift)
# ============================================================================


def _transport_to_test_frame(
    planes: Dict[int, List[int]],
    R_out: int,
    C_out: int,
    pose_src: str,
    anchor_src: Tuple[int, int],
    pose_dst: str,
    anchor_dst: Tuple[int, int],
    colors_order: List[int],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Transport color planes from Π_out_i to Π_out*.

    The normalized grid Y_norm is encoded as planes in IDENTITY/RAW orientation.
    We transform it to the test output frame Π_out* via:
      Step 0: Apply pose_src to bring planes into source frame Π_out_i
      Step 1: Unanchor from source (shift by -anchor_src)
      Step 2: Apply relative pose (pose_dst ∘ inv(pose_src))
      Step 3: Re-anchor to destination (shift by +anchor_dst)

    All operations maintain the fixed (R_out, C_out) canvas.
    Dimension swaps from R90/R270/FXR90/FXR270 are handled by verifying
    that the canvas is square (R_out == C_out) when swaps occur.

    Returns:
        A_i: Transported admits (singleton per pixel)
        S_i: Scope mask (union of all color planes)

    Spec: WO-08 section 2
    """
    # Compute relative pose transform: T = pose_dst ∘ inv(pose_src)
    pose_src_inv = pose_inverse(pose_src)
    T_pose = pose_compose(pose_dst, pose_src_inv)

    # Check if relative pose swaps dimensions
    # R90, R270, FXR90, FXR270 swap H/W
    swaps_dims = T_pose in ["R90", "R270", "FXR90", "FXR270"]

    if swaps_dims and R_out != C_out:
        # Dimension swap on non-square canvas → incompatible
        # Return silent (no scope, no admits)
        A_i = {c: [0] * R_out for c in colors_order}
        S_i = [0] * R_out
        return A_i, S_i

    A_i = {}

    for c in colors_order:
        plane = planes.get(c, [0] * R_out)

        # Step 0: Bring normalized raw planes into Π_out_i pose
        # The planes from pack_grid_to_planes(Y_norm) are in IDENTITY/RAW orientation
        # We need to apply pose_src to get them into the source training frame
        plane_src, H0, W0 = pose_plane(plane, pose_src, R_out, C_out)
        if H0 != R_out or W0 != C_out:
            # Dimension mismatch → incompatible
            A_i[c] = [0] * R_out
            continue

        # Step 1: Unanchor from source (shift by -anchor_src)
        # Remove the anchor offset to move content to origin
        plane_unanchored = shift_plane(plane_src, -anchor_src[0], -anchor_src[1], R_out, C_out)

        # Step 2: Apply relative pose transform (pose_dst ∘ inv(pose_src))
        # Transform from source frame to destination frame
        plane_transformed, H_new, W_new = pose_plane(plane_unanchored, T_pose, R_out, C_out)

        # Verify dimensions match canvas after pose
        if H_new != R_out or W_new != C_out:
            # Dimension mismatch after pose → incompatible
            A_i[c] = [0] * R_out
            continue

        # Step 3: Re-anchor to destination (shift by +anchor_dst)
        # Apply the destination anchor offset
        plane_final = shift_plane(plane_transformed, +anchor_dst[0], +anchor_dst[1], R_out, C_out)

        A_i[c] = plane_final

    # Compute scope as union of all color planes
    # Always use fixed R_out (not variable length)
    S_i = [0] * R_out
    for c in colors_order:
        for r in range(R_out):
            S_i[r] |= A_i[c][r]

    return A_i, S_i


# ============================================================================
# Helpers
# ============================================================================


def _popcount_scope(S: List[int]) -> int:
    """Count total bits set in scope mask."""
    return sum(bin(row).count("1") for row in S)


def _serialize_transport(
    A: Dict[int, List[int]],
    S: List[int],
    R_out: int,
    C_out: int,
    colors_order: List[int],
) -> bytes:
    """Serialize transported admits for hashing."""
    # Concatenate all color planes in order + scope
    result = b""
    for c in colors_order:
        plane = A.get(c, [0] * R_out)
        for row_mask in plane:
            result += row_mask.to_bytes((C_out + 7) // 8, "big")

    # Add scope
    for row_mask in S:
        result += row_mask.to_bytes((C_out + 7) // 8, "big")

    return result
