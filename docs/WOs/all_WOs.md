Good plan. We’ll build it bottom-up in small, receipts-first work orders (WOs), each ≤ ~500 LOC, no forward dependencies, no stubs. Every WO ships with: a tiny spec, stable interfaces, invariants, receipts, and a self-test. When a later WO needs capabilities, it **imports only the published interface** of earlier WOs.

Below is the high-level computing WBS you can hand to Claude Code. It maps 1:1 to the math+computing spec we locked.

---

# ARC-AGI Bit-Algebra — Receipts-First Work Orders (v1.5)

Conventions for all WOs

* Language: Python 3.12. No randomness in logic.
* Style: pure functions where possible, small modules, no side effects except receipts.
* Receipts: BLAKE3 (vendored) over big-endian row-major bytes; log `spec_version="1.5"` and `param_registry` snapshot.
* Tests: each WO ships a `tests_woXX.py` that runs offline and verifies determinism by a double-run hash equality.

---

## WO-00 — Param Registry & Receipts Core (foundation) ✅ COMPLETED

**Goal:** One tiny library to hash, serialize planes, and record receipts consistently.

**Interface**

* `hash_bytes(be_row_major_bytes) -> str`
* `hash_planes(planes: dict[color]->Plane) -> str`
* `Receipts(section: str)` with `.log(dict)` and `.seal() -> dict`
* `serialize_grid_to_be_bytes(G: list[list[int]], H, W) -> bytes`

**Invariants**

* BLAKE3 BE only.
* Double-run same inputs → same hashes.

**Receipts**

* `blake3_version`, `endianness:"BE"`, `spec_version:"1.5"`, `param_registry`.

**Tests**

* Known bytes -> known hash; double-run equality.

---

## WO-01 — Bit-Planes & Kernel Ops (5 primitives) ✅ COMPLETED

**Goal:** Implement PACK/UNPACK and the five ops: SHIFT, POSE, AND/OR/ANDN, PERIOD skeleton.

**Interface**

* `pack_grid(G, C) -> dict[color]->Plane`
* `unpack_planes(planes, C, H, W) -> G`
* `shift(plane, dy, dx, H, W) -> plane`
* `pose(plane, pid, H, W) -> plane`  (pid in `[I,R90,R180,R270,FX,FXR90,FXR180,FXR270]`)
* `bit_and(p1,p2)->plane`, `bit_or(...)`, `bit_andn(p,mask)->plane`

**Invariants**

* Zero-fill on shifts; no wrap.
* POSE is exact; inverse(pose(pid)) exists.

**Receipts**

* Hashes of op tables (D4 matrices), unit op self-checks.

**Tests**

* Round-trip POSE ∘ inverse ∘ POSE == identity; shift bounds.

---

## WO-02 — PERIOD (1D minimal period + 2D residues) ✅ COMPLETED

**Goal:** Integer KMP minimal period for rows/cols; build 2D residue masks.

**Interface**

* `minimal_period_1d(mask_bits: int, W:int) -> int|None`
* `period_2d_planes(planes) -> (p_r, p_c, residue_masks)`

**Invariants**

* Minimal >1 if non-trivial, else None. Phase fixed at (0,0).

**Receipts**

* `p_r,p_c,phase`, `chosen_period`, any ties tested and resolved.

**Tests**

* Checkerboard, stripes, solids.

---

## WO-03 — Frame Canonicalizer (D4 lex-min + anchor)

**Goal:** Freeze poses and anchors deterministically.

**Interface**

* `canonicalize(G) -> (pid, anchor_rc, G_canon)`
* `apply_pose_anchor(planes, pid, anchor) -> planes`

**Edge cases**

* All-zero grid → pid=I, anchor=(0,0).

**Receipts**

* `pose_id`, `pose_tie_count`, `anchor`, `all_zero`.

**Tests**

* Symmetric cases; double-run identical.

---

### Milestone M0 (after WO-00 + WO-01 + WO-03): “Bedrock”

* Wire: `core/registry, core/receipts, kernel/{planes,ops}, kernel/frames`.
* Runner does: color universe, pack↔unpack, pose canonicalize, receipts only.
* Real-ARC check: 5 tasks → pack/unpack identity receipts; pose inverse receipts.

---

## WO-04 — LCM Normalizer & Strict Downscale

**Goal:** Lift differing output sizes to LCM canvas; strict reduction to final size by constant blocks only.

**Interface**

* `lcm_normalize_outputs(Y_list) -> (R_lcm,C_lcm, Yprime_list, scale_factors)`
* `reduce_if_constant(Y_canvas, Rout, Cout) -> Y or SIZE_INCOMPATIBLE(first_mixed_block)`

**Receipts**

* `lcm_shape`, per-training `scale_factor`, `upscale_hash`; on reduce, `reduced:true/false`.

**Tests**

* Mixed shapes; reduction fail on non-uniform blocks.

---

### Milestone M1 (after WO-04 + WO-02): “Canvas online”

* Wire: `normalization/lcm_canvas`, `normalization/downscale_strict`, `kernel/period` (exposed but not used yet).
* Runner does: LCM-lift trainings; strict downscale probe disabled by default.
* Real-ARC check: tasks with differing output sizes → deterministic `lcm_shape` receipts; reduction fails on mixed blocks.

---
## WO-05 — 4-CC Components & Shape Invariants

**Goal:** Extract per-color components using bit ops; bbox, area, perim4, outline hash.

**Interface**

* `components(planes) -> list[Component]` with fields `{color, bbox, mask_plane, area, perim4, outline_hash}`

**Receipts**

* Component table with invariant tuples.

**Tests**

* Known shapes; disjoint unions.

---

## WO-06 — Witness Matcher (per training)

**Goal:** Learn rigid pieces and σ per training with exact equality; resolve overlaps.

**Interface**

* `learn_witness(Xi, Yi_prime, frames) -> WitnessResult`

  * `pieces: list[{pid, dy, dx, bbox}]`
  * `sigma: dict[color_in]->color_out` or silent
  * `silent: bool`

**Rules (frozen)**

* Enumerate 8 D4 and translations that keep bbox inside canvas.
* Reject if overlapping pieces imply different colors at any pixel.
* σ must be bijection on touched colors; conflicts → silent.

**Receipts**

* Per-piece trials `{pid,dy,dx,ok}`, `sigma_bijection_ok`, `witness_overlap_pixels`.

**Tests**

* Positive match; overlap conflict → silent; σ conflict.

---

## WO-07 — Conjugation & Forward Witness Emitter

**Goal:** Conjugate pieces into test frame; emit `(A_wit,S_wit)` from `X*` via σ.

**Interface**

* `emit_witness(X_star, witness_results_all_trainings, frames) -> (A_wit,S_wit)`

**Normalization**

* Admit-all ⇒ scope=0.

**Receipts**

* `A_wit_hash`, `scope_bits`.

**Tests**

* Small synthetic cases; scope correctness.

---

### Milestone M3 (after WO-05 + WO-06 + WO-07): “Witness path”

* Wire: `evidence/components`, `emitters/witness_learn`, `emitters/witness_emit`.
* Runner: enables witness; selection precedence becomes **witness → unanimity → bottom** (no LFP yet).
* Real-ARC check: 5 simple geometric copy/flip tasks solved; σ bijection receipts; no overlaps.

---

## WO-08 — Output Transport & Unanimity Emitters

**Goal:** Transport normalized `Y'_i` onto LCM canvas; emit unanimity.

**Interface**

* `emit_output_transport(Yprime_list) -> (A_out,S_out)`
* `emit_unanimity(Yprime_list) -> (A_uni,S_uni)`

**Receipts**

* Per-training transport hashes; unanimity mask hash, counts.

**Tests**

* Agreement/disagreement; admit-all ⇒ silent handling.

---

### Milestone M2 (after WO-08): “Output path”

* Wire: `emitters/output_transport`, `emitters/unanimity`.
* Runner: build emits list with these two; keep witness/lattice disabled.
* Selection precedence: engine_winner not present yet → selection uses unanimity only (witness absent).
* Real-ARC check: 5 consensus tasks solved bit-exact; receipts show unanimity counts, EngineWinner absent.

---

## WO-09 — Lattice Emitter (exact periodic admits)

**Goal:** Emit residue admits where **all trainings** agree.

**Interface**

* `emit_lattice(Yprime_list) -> (A_lat,S_lat)`

**Receipts**

* `p_r,p_c`, agreeing/disagreeing residue classes.

**Tests**

* Periodic outputs; mixed class → silent.

---

## WO-10 — Forbids + AC-3 (4-neighbor)

**Goal:** Learn forbid matrix M(c,d) (directed if needed) and run frozen AC-3.

**Interface**

* `learn_forbids(train_pairs) -> (E_graph, M_matrix)`
* `ac3_prune(D, E, M) -> (D_changed:bool, stats)`

**Frozen**

* E = 4-neighbor; queue order row-major; FIFO; on prune enqueue neighbors.

**Receipts**

* `forbid_symmetric:true|false`, `matrix_hash`, `edges_count`, `prune_counts`.

**Tests**

* Simple Sudoku-like pruning; determinism.

---

## WO-11 — Domain Tensor & LFP Propagator

**Goal:** The monotone loop: admit-intersect in fixed order, then AC-3, to LFP.

**Interface**

* `lfp_propagate(D0, emitters_list, forbids=None) -> (D*, stats or UNSAT)`

**Frozen**

* Family order: T1..T12.
* Early exit on empty domain → UNSAT.
* Cap guard: FIXED_POINT_NOT_REACHED → fail-closed.

**Receipts**

* `admit_passes`, `ac3_passes`, `total_prunes`, `domains_hash`.

**Tests**

* Converges in 1–3 passes on toy; UNSAT path.

---

### Milestone M4 (after WO-10 + WO-11): “True LFP”

* Wire: `constraints/forbids` (can return None), `constraints/ac3`, `propagation/lfp`.
* Runner: domains init → admit-∧ (witness, output, unanimity) → AC-3 → fixed point → selection.
* Real-ARC check: Stage M2/M3 tasks yield identical outputs pre- vs post-LFP; UNSAT path receipts OK.

---

## WO-12 — EngineWinner Chooser (global, training-scope)

**Goal:** If multiple engines fit trainings, pick the single winner deterministically.

**Interface**

* `choose_engine_winner(candidates: list[EmitterStatsTrain]) -> winner_id`

**Frozen**

* Compare **training** scope totals; tie by fixed priority:
  `T3 Lattice > T5 ICL-Conv > T4 ICL-Kron > T6 Morph > T7 Logic > T8 Param-ICL > T9 AggMap > T10 Forbids > T11 CSP`.

**Receipts**

* `engines_fitting`, `engine_scope_train_total`, `engine_winner`.

**Tests**

* Tie cases across two families.

---

### Milestone M5 (after WO-09 + WO-12): “EngineWinner + Lattice”

* Wire: `emitters/lattice`, `selection/engine_winner`.
* Runner: now passes **witness → engine_winner (global) → unanimity → bottom**.
* Real-ARC check: tiling task solved; receipts show `(p_r,p_c)` and EngineWinner chosen by training scope, tie priority frozen.

---

## WO-13 — Selector (witness → engine_winner → unanimity → bottom)

**Goal:** Scope-gated pick with containment + idempotence asserts.

**Interface**

* `select(D_star, Awit, Swit, Aeng, Seng, Auni, Suni) -> Y_canvas or UNSAT`

**Frozen**

* Bottom only if 0 ∈ D*[p] and all scopes 0; else UNSAT.

**Receipts**

* `selection.counts`, `containment_verified:true`, `repaint_hash`.

**Tests**

* Each bucket exercised; bottom precondition.

---

## WO-14 — Aggregate Mapping (T9) — Features & Size Predictor

**Goal:** Compute frozen features; search frozen hypothesis class; emit size (and optional color) mapping.

**Interface**

* `agg_features(X) -> FeatureVector`
* `agg_size_fit(train_pairs) -> (family, params) or NONE`
* `predict_size(X_star, fit) -> (R_out, C_out)`

**Receipts**

* `features_hash`, `size_fit {winner, attempts}`.

**Tests**

* Affine, period multiple, CC-linear; tie-breaking.

---

## WO-15 — Final Downscale Integrator

**Goal:** Combine Y_canvas with T9 size; exact reduce or fail.

**Interface**

* `finalize_output(Y_canvas, size_fit_opt) -> Y or SIZE_INCOMPATIBLE`

**Receipts**

* `reduced:true|false`, first mixed block if any.

**Tests**

* Passing and failing reductions.

---

### Milestone M6 (after WO-14 + WO-15): “Size finalize”

* Wire: `evidence/features`, `sizing/aggmap_size`, `normalization/downscale_strict` final step.
* Runner: if T9 size present, perform strict block-constant reduction; else keep canvas size.
* Real-ARC check: argmax-uniform + size-mapping tasks solved; receipts show `size_fit.attempts` and `reduced:true` (constant blocks only).

---

## WO-16 — Optional Engines (each its own ≤500 LOC)

Pick and schedule only as needed; each compiles to `(A,S)` using WO-01 ops.

* **WO-16A ICL-Conv (stencils)**: offset kernels from X*, bounds |K|≤9.
* **WO-16B ICL-Kron (block expansion)**: factors within [1..5], exact Kronecker.
* **WO-16C Morph (reachability)**: N=4, seeds/barriers from frozen hypotheses, cap H+W.
* **WO-16D Logic-Guard**: predicate catalog + truth table with leftmost/topmost anchor.
* **WO-16E Param-ICL**: guards n from test, values in training set, lex-min on multi-fire.
* **WO-16F CSP micro-solver**: ≤3×3 tiles, Hopcroft–Karp, lex-min matching.

Each optional WO ships its **own** attempts log and winner receipts.

---

### Milestone M7+ (each optional engine from WO-16x)

* Wire one emitter at a time (e.g., `icl_conv`, `icl_kron`, `morph`, `logic_guard`, `param_icl`).
* Runner: add to emits; EngineWinner decides globally.
* Real-ARC check: 3–5 tasks per engine; receipts list attempts and chosen kernel/factors/predicates.

---

## WO-17 — End-to-End Runner (no stubs)

**Goal:** Wire WOs 00–15 into one deterministic pipeline.

**Interface**

* `solve(task_json) -> (Y, receipts_bundle or explicit FAIL)`
* Steps: build 𝒞; frames; LCM; emitters (witness, output, unanimity, lattice, +optional); LFP; choose EngineWinner; select; T9 size; strict reduce; double-run hash check.

**Receipts**

* Sectioned bundle; `first_differing_section` if double-run fails.

**Tests**

* The three problems we solved by hand; a few negative UNSAT/SIZE_INCOMPATIBLE cases.

---

## Dependency Graph (bottom-up)

* 00 → 01 → 02/03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 17
* 16x branches off after 01/05; plugs into 11 & 12.

---

## Authoring Guidance for Claude Code

* Keep each WO in a dedicated module, ≤ ~500 LOC.
* Export only the described interfaces; everything else `__all__`-hidden.
* No TODOs, no stubs. If something can’t be proved, the function returns **silent** `(A=all, S=0)` or an explicit **FAIL** per spec.
* Every WO must include:

  * a README-style docstring with invariants and failure modes,
  * tests that assert double-run determinism,
  * receipts checks (e.g., hashes must change when inputs change, and not otherwise).

---

## Why this plan avoids silent drift

* Each WO is self-contained, receipts-logged, and consumable by later WOs without “future promises.”
* The kernel (WO-01) and frames/LCM (WO-03/04) are frozen first, so geometry, periods, and emitters cannot improvise.
* The LFP (WO-11) and Selector (WO-13) are late, so they bind semantics only after all emitters are stable.
* EngineWinner (WO-12) makes “only one engine speaks” explicit; receipts show the winner.
* T9 (WO-14/15) is isolated; downscale is strict, preserving “no minted bits.”

If you want, I can now expand **WO-01** with function signatures, row-mask packing format, and exact unit cases so Claude Code can start coding without guessing.


====
You’re thinking correctly: don’t leave integration to the end. Keep a **single runner file** and grow it incrementally as WOs land. That’s efficient and safer, provided we freeze the runner’s contract now and treat missing layers as **silent** (A=all, S=0) rather than stubbing behavior.

Here’s exactly how to “squeeze” progressive integration into the current WO plan (WO-02 is WIP).

# Freeze the runner contract now (one file, grown incrementally)

`src/arcbit/runner/solve.py` must exist **now** with this stable flow (no logic beyond wiring):

```
def solve(task_json):
  C = color_universe(task_json)                     # includes 0 and test colors
  frames = canonicalize_all(task_json)              # D4 + anchor
  Yprime, lcm_shape = lcm_normalize_outputs(train_outputs)

  emits = []
  if ENABLE_OUTPUT:   emits += [emit_output_transport(Yprime)]
  if ENABLE_WITNESS:  emits += [emit_witness(X_star, learn_witness(...))]
  if ENABLE_UNANIMITY: emits += [emit_unanimity(Yprime)]
  if ENABLE_LATTICE:  emits += [emit_lattice(Yprime)]
  # optional engines gated similarly

  Dstar = lfp_propagate(full_domain(C), emits, forbids_opt=None)
  engine = pick_engine_winner(train_scope_stats(emits))  # returns None until added
  Y_canvas = select(Dstar, wit_layer(emits), engine_layer(emits), uni_layer(emits))
  Y_final = reduce_if_constant(Y_canvas, predicted_size_opt)  # size predictor None until added

  return Y_final, receipts_bundle
```

Anything not implemented yet returns a **silent** emitter `(A=all, S=0)` or `None` (for size), never a behavioral stub. Receipts are still produced for every section (even when silent).

# Progressive integration milestones (tie to your WOs)

You’re mid WO-02; use these milestones. Each one is tiny and just wires the runner to the newly-finished WOs, then tests on real ARC slices.







# Where this changes your current WO flow

* **Don’t wait for WO-17**. The runner exists from M0 and never changes its shape; you only flip feature flags as WOs complete.
* For each WO completion, create a tiny **integration PR** that:

  1. imports the new module into `solve.py`,
  2. adds a feature flag defaulting **on**,
  3. drops 3–5 real ARC JSONs into `fixtures/arc/` for that milestone.
     No test framework needed beyond our receipts and final grid equality.

# Why a single file grown over time is safe here

* The runner contract is **frozen**; missing pieces return **silent** admits or `None`. You can wire without stubs that change behavior later.
* Receipts are **sectioned** and exist from M0, so every integration is algebraically traceable (hash diffs localize regressions).
* Each milestone exercises **real tasks** for that path only, catching interface drift early.

# Tiny instructions per role

**Implementer (per milestone):**
Wire the new module into `solve.py` behind a feature flag; return silent layer if inputs are missing; emit receipts for the new section; keep ≤ ~50 LOC changes.

**Reviewer (per milestone, real ARC only):**
Run `scripts/solve_task.py` on the curated slice; verify double-run identical section hashes; check milestone-specific receipts (e.g., unanimity counts at M2, σ receipts at M3, LFP passes at M4); confirm no minted bits and strict downscale only.

This gives you continuous end-to-end integration without ever having to “merge 15 pieces at once,” and it aligns perfectly with the math and computing specs.
