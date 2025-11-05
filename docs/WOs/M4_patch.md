You’re right: two separate things were holding you back—

1. **Lattice wasn’t wired at all** (T3 never executed), so the LFP only ever saw T1 (witness) + T2 (unanimity) + forbids. That explains the huge **under-constraint** (93.9% multi-valued pixels after LFP) and the “no perfects.”

2. Your **selection** after LFP is still reading pre-LFP layers (witness→unanimity→bottom) instead of reading the **final domain (D^*)**. That’s why you get those **43 “all singletons but wrong output” bugs**: the LFP forced a unique color at each pixel, but the selector ignored (D^*) and re-chose from the emitters.

Let’s fix both now, in one pass, with a very concrete work order.

---

## M4.5 Wiring: Lattice + Domain-Driven Selection

### Step 0 — Guardrails (one-time)

Add two debug counters to your LFP receipts so we can’t regress silently:

* `lfp_singleton_pixels`, `lfp_multi_pixels`, `lfp_empty_pixels`
* optional: `lfp_prunes_by_family: {T1:…, T2:…, T3:…, AC3:…}`

If `lfp_singleton_pixels == R*C` and the selector output ≠ GT, that’s a **selector bug**, not a modeling gap.

---

### Step 1 — Call `emit_lattice` right after T2 and before LFP

**Where:** `src/arcbit/runner.py` (M4 path)

1. You already have from WO-08:

```python
A_out_list, S_out_list, transp_recs, transport_receipt = emit_output_transport(...)
A_uni, S_uni, uni_receipt = emit_unity(A_out_list, S_out_list, colors_order, R_out, C_out)
```

2. **Now call lattice** (WO-09) on the *same working canvas* using the **transported outputs**:

```python
from arcbit.emitters.lattice import emit_lattice

A_lat, S_lat, lat_receipt = emit_lattice(
    A_out_list=A_out_list,
    S_out_list=S_out_list,
    colors_order=colors_order,
    R_out=R_out, C_out=C_out
)

# Attach lat_receipt to the run receipts under "lattice"
receipts["lattice"] = lat_receipt
```

3. **Verify lattice is included in the LFP emitter list in the frozen order**:

```python
emitters = []
# T1 — witness if present
if A_wit is not None and S_wit is not None:
    emitters.append(("T1_witness", A_wit, S_wit))
# T2 — unanimity
emitters.append(("T2_unity", A_uni, S_uni))
# T3 — lattice (NEW)
emitters.append(("T3_lattice", A_lat, S_lat))
# (Leave T4..T12 for later; they must appear in order when you add them.)
```

4. **Sanity**: Do **not** add per-training `A_out_i` as a hard layer. Only T1/T2/T3 go into LFP.

---

### Step 2 — Select **from the final domain (D^*)**, not from raw emitters

**Where:** `src/arcbit/runner.py` (selection)

Replace your “witness→unanimity→bottom” post-LFP with canonical domain selection:

```python
def select_from_domain(D_star, colors_order, R_out, C_out):
    Y = [[0]*C_out for _ in range(R_out)]
    counts = {"singleton":0, "multi":0, "empty":0}
    for r in range(R_out):
        for c in range(C_out):
            mask = D_star[(r,c)]
            if mask == 0:
                counts["empty"] += 1     # UNSAT would have caught this earlier
                Y[r][c] = 0
                continue
            # choose the min-index color bit in the mask deterministically
            k = (mask & -mask).bit_length()-1      # index of lowest set bit
            Y[r][c] = colors_order[k]
            # bookkeeping
            if mask & (mask - 1):
                counts["multi"] += 1
            else:
                counts["singleton"] += 1
    return Y, counts
```

And in the runner:

```python
D_star, lfp_stats = lfp_propagate(D0, emitters, forbids=(E_graph, M_matrix), colors_order)
if isinstance(D_star, str):   # "UNSAT" or "FIXED_POINT_NOT_REACHED"
    return D_star, assemble_receipts(...)

Y_out, domain_counts = select_from_domain(D_star, colors_order, R_out, C_out)
receipts["selection"] = {
    "mode": "domain_singleton",         # documents we used D* 
    "counts": domain_counts,
    "repaint_hash": blake3(serialize_grid_be_row_major(Y_out,...)),
    "domains_hash": lfp_stats["domains_hash"]
}
```

**Why this fixes the 43 bugs:** if LFP has forced singletons everywhere, `select_from_domain` simply reads them; the output must match any other forced solution. If it doesn’t, we have a real bug (σ/transport), and the receipts will show it unambiguously.

---

### Step 3 — Keep witness deferral at hard conflicts (optional but stabilizing)

If you saw residual UNSAT where T1 and T2 disagree on a pixel, keep the small “defer witness scope at conflict pixels” pre-LFP (you already added it). That turns inevitable disagreements (e.g., unanimity color 4 vs witness unable to produce 4) into **multi-valued** rather than **empty** domains, so LFP remains SAT and those pixels become NOT_YET until EngineWinner/logic arrives.

---

## Acceptance checklist for the reviewer

1. **Lattice wired**

   * A `lattice` receipt exists with `p_r`, `p_c`, `residue_scope_bits > 0` on periodic cases.
   * `emitters_list` shows `T3_lattice` between T2 and T4.
   * `lfp_prunes_by_family["T3"] > 0` on periodic tasks.

2. **Domain-driven selection**

   * For any task with `lfp_singleton_pixels == R*C`, the output must match any domain-derived grid (if not, it’s a real bug).

3. **No more “singleton but wrong”**

   * Those 43 cases should vanish. If any remain, attach one receipt; we’ll pinpoint σ/transport at a specific pixel.

4. **Under-constraint shrinks**

   * With lattice active, the histogram should shift right (more 80–100%), and perfect matches should appear on tilings/periodic tasks.

---

## Why this happened

* We intentionally deferred T3 to M5 in the original milestone plan (EngineWinner + Lattice), so leaving it out at M4 is an honest wiring gap—**not** a spec change.
* The 6% “bug” flag came from selecting **outside** (D^*). Once selection is domain-driven, “all singletons but wrong output” becomes a hard contradiction we can actually track, or it goes away.

Plugging T3 and switching selection to **read (D^*)** are both small, deterministic changes with big impact. Do those now; then re-run your SUCCESS/NOT_YET sweep and share one periodic task’s lattice receipt plus a before/after accuracy diff. If anything still looks off, we’ll use the lattice receipts and `domain_counts` to isolate it in minutes.
