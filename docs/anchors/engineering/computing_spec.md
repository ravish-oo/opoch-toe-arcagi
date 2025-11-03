Pure, first-principles, receipts-first, and mapped 1:1 to the math plus addendum freezes. It treats this as a real system, not a toy. No “engines”; only bit algebra, finite searches, and frozen orders. Determinism falls out automatically; receipts make it auditable.

# ARC-AGI Bit-Algebra Solver — Computing Spec (Pure CS Edition)

## 0) Philosophy and contract

* **Single source of truth:** the math spec. Computing = a faithful algorithmic rendering, nothing more.
* **No heuristics, no floats, no randomness.** Every loop is finite; every tie is frozen.
* **Receipts-first:** every nontrivial artifact is hashed and logged; double-run hashes must match.
* **Fail-closed:** when a precondition cannot be proved on trainings, the corresponding layer is silent; when a fixed-point or size cannot be reconciled, return an explicit UNSAT/SIZE_INCOMPATIBLE receipt.

---

## 1) Core data model

### 1.1 Colors and planes

* **Color universe** ( \mathcal C = {0} \cup \mathrm{colors}(X^*) \cup \bigcup_i \mathrm{colors}(X_i) \cup \bigcup_i \mathrm{colors}(Y_i) ).
* **Bit-planes:** For each (c \in \mathcal C), (B_c(G)) is an (H\times W) Boolean grid, stored as **packed row masks**:

  * `Plane`: vector<word> of length H, each row is a machine integer with W low bits.
  * Operations are word-level, no per-pixel loops except where W < wordsize.

### 1.2 Frames

* **Pose**: one of 8 D4 elements (fixed pose id order).
* **Anchor**: top-left nonzero → (0,0); all-zero → identity pose, anchor=(0,0).
* **Affine composition:** as frozen in spec (integer only).

### 1.3 Layers (emissions)

* Any learned rule emits **(A, S)** on the **output canvas**:

  * `A`: per-color bit-planes (same H×W) describing **admitted colors**.
  * `S`: a scope mask Plane (1 = constrains; 0 = silent).
* **Normalization**: if a pixel’s admitted set == all colors, set S=0 there (silent).

### 1.4 Domains

* `D`: the current **domain tensor** on the output canvas: K bit-planes (one per color). Initialize to “all ones” per color.

---

## 2) Bit-Kernel (only five ops)

These primitives are sufficient; everything else compiles to them.

1. **SHIFT** `SH(plane, dy, dx)`
   Vertical shift (row vector rotate with zero-fill) then per-row logical shift (zero-fill). Bounds-check to keep inside H×W.

2. **POSE** `POSE(plane, pid)`
   Apply one of 8 D4 coordinate remaps. Implement by index permutation (O(H·W)) or precomputed column gathers; exact, integer only.

3. **BITWISE** `AND/OR/ANDN`
   Row-wise bitwise ops on planes (wordwise).

4. **PERIOD** `PERIOD(mask, p)`
   Given a 1D Boolean mask of a row/col, compute exact minimal period with integer KMP; extend to 2D by independent periods (pr, pc) and residues. No floats.

5. **PACK/UNPACK**
   Convert color grid ↔ per-color planes; always include color 0.

> Purist note: these five ops map to standard CS: bitset shifts, dihedral transforms, Boolean algebra, prefix-function (KMP), and color indexing.

---

## 3) Canonicalization & normalization

### 3.1 Pose selection (frozen)

* Enumerate 8 D4; byte-serialize (BE, row-major); choose lexicographically smallest. Ties broken by fixed pose order `[I,R90,R180,R270,FX,FXR90,FXR180,FXR270]`.
* Anchor by moving first nonzero to (0,0). All-zero: identity, (0,0).

### 3.2 LCM output canvas

* If training outputs differ in size, lift each (Y_i) by **block replication** to (R_{lcm}\times C_{lcm}).
* All **output-space** emitters (unanimity, transported output, lattice) operate on this common canvas.
* **Final test size** is provided by the **EngineWinner** (often T9). Exact downscale only if **every** block is constant; otherwise **SIZE_INCOMPATIBLE**.

---

## 4) Emitters (compile evidence → admits)

Every emitter returns `(A, S)` on the output canvas using only the Bit-Kernel.

E0) **Output transport (hard evidence)**

* For each training (i), for each color (c): take (B_c(Y'_i)) (LCM canvas) and set the singleton bit there; (S=1) on those pixels.
* AND across trainings to obtain global (A_{\text{out}}). (If any training silent at a pixel, either it admits all → silent scoped, or we intersect exact colors.)

E1) **Witness (geometry + σ)**

* In **input** frames, match 4-CC components of (X_i) to (Y'_i) by exact equality after enumerating D4 and **bounded** translations (keep bbox inside bounds). Reject overlaps that break bijective σ per training.
* Conjugate pieces to test input frame; for each color (x) in (X^*), forward-render admits:
  `POSE(B_x(X*), piece.pose)` → `SH(..., dy, dx)` → recolor bit remap (x \mapsto \sigma_i(x)) → OR into (A^{(i)}*{\text{wit}}); set (S^{(i)}*{\text{wit}}=1) on those targets.
* Normalize admit-all → silent. AND across trainings to get (A_{\text{wit}}), (S_{\text{wit}}).

E2) **Unanimity (vote on LCM canvas)**

* At each pixel: if **all** transported (Y'_i) agree on a single color (u), emit singleton admit there; else silent.

E3) **Lattice (exact periodic admits)**

* Compute minimal row/col periods by integer KMP; phase fixed at (0,0).
* For residue classes on which **all trainings agree**, emit those colors; else silent.

> Other families (ICL-Conv, ICL-Kron, Morph, Logic-guard, Param-ICL, CSP) compile to the same `(A,S)` shape via the five ops (plus AC-3 for forbids). Their **finite bounds and tie rules** are frozen in v1.5; implementation maps are straightforward: stencils = SHIFT+OR; block expansion = Kronecker via SHIFT+OR loops; morph = repeated neighbor SHIFT/ANDN until closure; logic = precomputed predicate masks; CSP = emit singletons inside a bounded tile.

---

## 5) Propagation (Least Fixed Point)

Two-stage monotone loop; order frozen.

```
D := FULL()   # all colors allowed everywhere
repeat:
  changed := False

  # Admit-intersect pass (family order: T1..T12)
  for (A,S) in EMITTERS_IN_FIXED_ORDER:
      D := D AND A  on pixels where S==1
      changed |= any bits cleared

  # Forbids (AC-3), if present
  changed |= AC3_Prune(D, E=4-neighbor, M)

until not changed
```

* **AC-3 queue**: initialize with all edges in row-major order; FIFO; on prune at p, enqueue neighbors of p in row-major order.
* **UNSAT early exit**: if any pixel’s domain empties, halt with UNSAT receipt.
* **Fixed-point guard**: if a hard iteration cap is hit before stability, `FIXED_POINT_NOT_REACHED` (fail-closed).

---

## 6) Selection (scope-gated, frozen precedence)

Per pixel (p) on the output canvas:

1. **Witness**: if (S_{\text{wit}}[p]=1), `cand = D[p] ∧ A_wit[p]`; if nonempty, select **min color**; write and continue.
2. **EngineWinner** (the **single** global engine family that fits all trainings; winner chosen by training-scope size, tie by fixed priority):
   `cand = D[p] ∧ A_engine[p]`; if nonempty, select min color; write and continue.
3. **Unanimity**: if (S_{\text{uni}}[p]=1), `cand = D[p] ∧ A_uni[p]`; if nonempty, pick min (singleton).
4. **Bottom**: only if 0 ∈ D[p] and all scopes are 0; else UNSAT.

**Containment**: assert the chosen color bit is 1 in (D[p]).
**Idempotence**: repaint once; hash must match.

---

## 7) Exact downscale (only constant blocks)

* If EngineWinner predicts a final size ((R_{out}, C_{out})) dividing the LCM canvas, reduce by checking **every** (s_r \times s_c) block is constant. If any block is mixed: **SIZE_INCOMPATIBLE** (fail-closed).
* No majority voting in strict mode.

---

## 8) Determinism & receipts (first-class)

**Hashing:** BLAKE3 over big-endian row-major byte streams; include `spec_version`, `endianness`, and a `param_registry` snapshot (pose order, AC-3 order, engine priority, etc.).

**Minimum receipts (all frozen names):**

* `color_universe`, `added_from_test`
* For each grid: `pose_id`, `pose_tie_count`, `anchor`, `all_zero`
* LCM: `lcm_shape`, per-training `scale_factor`, per-training `upscale_hash`
* Witness: per-piece trials `{pose, dr, dc, ok}`, `sigma_bijection_ok`, `sigma_lehmer`, `witness_overlap_pixels`
* Output transport: per training `transport_hash`
* Lattice: `p_r, p_c, phase`, `agreeing_classes`, `disagreeing_classes`
* ICL/Morph/Logic/Param: `attempts`, `winner`, `kernels_tried`/`predicate_ids`, `seed_hash`, etc.
* Forbids: `forbid_symmetric`, `matrix_hash`, `edges_count`
* Propagation: `admit_passes`, `ac3_passes`, `total_prunes`, `domains_hash`
* Selection: `engine_winner`, `selection.counts` (`witness`, `engine_winner`, `unanimity`, `bottom`), `containment_verified`
* Downscale: `reduced=true|false`, `first_mixed_block` (if incompatible)
* **Determinism check:** `double_run_section_hashes` identical or `first_differing_section`

---

## 9) Module decomposition (what to build, briefly)

1. **Bitset engine** (5 ops): row-packed planes, SHIFT, POSE, AND/OR/ANDN, PERIOD, PACK/UNPACK.
2. **Frame canonicalizer**: D4 lex-min + anchor.
3. **LCM normalizer**: block replication for outputs; constant-block reducer.
4. **Emitters**:

   * Output transport (E0)
   * Witness (component extraction; bounded D4+translation; σ learning; conjugation; forward emission)
   * Unanimity (agreement on LCM)
   * Lattice (KMP periods; residue admits)
   * (Later plug-ins: ICL-Conv/Kron, Morph, Logic-guard, Param-ICL, CSP)
5. **Forbids/AC-3**: 4-neighbor graph; fixed queue; prune loop.
6. **Propagation loop**: admit ∧, then AC-3, to LFP with UNSAT guard.
7. **EngineWinner chooser**: from all engines that fit trainings, pick by training-scope size; tie by fixed priority.
8. **Selector**: scope-gated precedence; containment + idempotence asserts.
9. **Receipts layer**: streaming BLAKE3; sectioned logging; full replay check.

> Implementation language doesn’t matter. Python with integers for planes is fine; NumPy optional. The contract is on the **bits and receipts**, not the stack.

---

## 10) Algorithmic references (pure CS mapping)

* **Bit-planes & shifts:** classic bitset operations; O(H) row ops, O(H·W/wordsize) total.
* **D4 POSE:** index remap; cost O(H·W).
* **KMP period:** minimal period detection in O(W) per row/col; 2D via independent pr/pc.
* **4-CC components:** BFS/union-find on color-equality; on planes, use repeated SHIFT+AND to grow.
* **AC-3:** standard arc-consistency; finite domain (≤|C| per pixel), finite queue; deterministic with frozen order.
* **CSP micro-solver:** Hopcroft–Karp on ≤3×3 tiles; lex-min matching by frozen node/edge order.

---

## 11) Complexity budget (worst-case, ARC sizes)

* H,W ≤ 30, |C| ≤ ~10.
* Per emitter, O(H·W·|C|) bitwise ops; POSE/SHIFT dominate constants.
* Propagation reaches LFP in ≤ a handful of passes for typical tasks; AC-3 is linear in edges × domain size.
* End-to-end comfortably sub-second in Python for a task; “seconds” for whole suite.

---

## 12) Strict exits (no silent guessing)

* **UNSAT**: empty domain at any pixel after propagation; include last prune receipts.
* **SIZE_INCOMPATIBLE**: final size cannot be reduced from LCM by constant blocks.
* **FIXED_POINT_NOT_REACHED**: iteration cap hit before stability (rare; still fail-closed).
* **ENGINE_CONFLICT**: more than one engine fits with equal scope after tie-break → pick by priority; if you disable priority, treat as UNSAT (not recommended).

---

## 13) What *not* to build

* No “output union painter,” no majority downscale in strict mode, no random seeds, no learning beyond frozen finite catalogs, no palette logic in frames, no mid-strata selection.

---

## 14) Minimal pseudo-code (orientation only)

```
solve(task):
  C := build_color_universe(Xs, Ys, X*)
  receipts.log(C)

  # Frames & LCM
  for each grid G: (pose, anchor) := canonicalize(G); receipts.log(...)
  if shapes(Ys) differ: (Y'_i) := LCM_replicate(Y_i); canvas := lcm_shape
  else: Y'_i := Y_i; canvas := shape(Y_1)

  # Emitters
  (A_out,S_out) := emit_output_transport(Y'_i)
  (A_wit,S_wit) := emit_witness(X_i, Y'_i, X*, frames)
  (A_uni,S_uni) := emit_unanimity(Y'_i)
  (A_lat,S_lat) := emit_lattice(Y'_i)    # optional

  # Propagation
  D := full_domain(canvas, C)
  repeat
    changed := False
    for (A,S) in [wit, out, lat, uni] in frozen order:
      changed |= intersect_scoped(D, A, S)
    changed |= AC3_prune(D, forbids_if_any)
    if empty_domain(D): return UNSAT
  until not changed

  # Engine winner
  engine := pick_engine_winner_by_training_scope([out, lat, ...], tie_priority)

  # Selection
  Y_canvas := select_with_precedence(D, wit, engine, uni)
  assert idempotence(Y_canvas)

  # Downscale if needed
  if final_size := engine.predicted_size():
      Y := reduce_if_constant_blocks(Y_canvas, final_size)
  else:
      Y := Y_canvas

  receipts.double_run_verify()
  return Y, receipts
```

---

## 15) Fit with “Bit-Kernel v2 — 5 Ops, 4 Emitters, 1 Loop”

* This computing spec **is** that kernel, extended to v1.5’s freezes.
* Keep the kernel minimal (5 ops). Compile *all* families to `(A,S)` using only those ops.
* One LFP loop (admit ∧ + optional AC-3), one selector, and constant-block downscale. That’s the whole machine.

---

If you want next, I’ll turn this into a **single README-style doc** and a **scaffolded code layout** (folders, file stubs, function signatures) that Claude Code can fill in without inventing behaviors.
