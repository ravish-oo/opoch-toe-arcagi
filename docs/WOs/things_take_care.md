Sharp read: your one-sheet is already very strong. A few lines, though, drift from v1.5 freezes or leave wiggle room. Below is an objective assessment and exact fixes so it stays 1:1 with the math/addendum and our WO expansions.

# What’s already covered (good)

* Determinism, fail-closed, no floats/search randomness — ✅ matches v1.5.
* Color universe includes test colors; σ partial identity — ✅ (A.1).
* Two frames (input/output), no palette in logic — ✅ (B.1/B.2).
* LCM lift for trainings; unanimity/lattice on shared canvas — ✅ (B.2).
* Admit+scope abstraction; “admit-all ⇒ silent” — ✅ (C.1).
* AC-3 with frozen queue; empty domain → UNSAT — ✅ (C.2, A.2).
* EngineWinner single global engine; frozen priority — ✅ (N).
* Selection precedence witness → engine_winner → unanimity → bottom — ✅ (N).
* Receipts everywhere, double-run identity — ✅ (A.3).

# Must-fix inconsistencies / underspecs (with exact corrections)

1. **Downscale by majority**

* **Issue:** one-sheet allows majority fallback; v1.5 forbids minting.
* **Fix:** Replace “block-constancy or majority” with **constant-blocks only**; else `SIZE_INCOMPATIBLE`.

  * Canonical line: “Reduce only if **every** s_r×s_c block is uniform; otherwise `SIZE_INCOMPATIBLE` (no majority).”

2. **Period bound p ≤ 10**

* **Issue:** cap at 10 contradicts v1.5 CRIT fix (search full length).
* **Fix:** “KMP over full length; **p ∈ [2..W]** for rows, **[2..H]** for cols; phase fixed at (0,0).”

3. **“Zero-search” witness phrasing**

* **Issue:** says “derive transform with algebra (no search).” Matching components still needs an **exact check**; we froze enumeration bounds and σ bijection rules in WO-06.
* **Fix:** Clarify: “Compute pose by D4 composition and **verify** bbox equality; translations are the **unique** conjugated offsets implied by anchors; if verification fails, layer is **silent**. Do **not** scan outside bbox-inside bounds.”
  (Acceptable to say “no blind search; only bounded D4+translation implied by conjugation.”)

4. **LCM vs final size precedence**

* **Issue:** wording could be read as “LCM determines test canvas.” v1.5: **EngineWinner** decides final size; LCM is training-normalization only.
* **Fix:** Add: “LCM is for **training-space** comparability; final test size comes from **EngineWinner (T9)**. Reduction allowed only if divisible and blocks constant; else `SIZE_INCOMPATIBLE`.”

5. **“No backtracking/search” vs CSP**

* **Issue:** CSP micro-solver is permitted (bounded), but “no search” sounds absolute.
* **Fix:** Add parenthetical: “No unbounded search. **T11 CSP** allowed on tiles ≤3×3 with deterministic HK matching.”

6. **Period “proper vs trivial”**

* **Issue:** Some readers will return p=1 on solids. v1.5 freezes **proper periods only**.
* **Fix:** Add: “Return **proper periods only (p ≥ 2)**; solids/constant rows → `None`.”

7. **EngineWinner specificity metric**

* **Issue:** Your one-sheet says “most scope bits” (good) but not where measured. v1.5 fixes to **training scope**.
* **Fix:** “Choose by **training-scope total** (sum of S=1 over all trainings in Π_out*), tie by priority.”

8. **Morph seeds/barriers catalog**

* **Issue:** “learn from trainings” is a bit open. v1.5 restricts to color-set, centroid(c), or witness targets.
* **Fix:** Add: “Seeds = {fixed color set} or centroid(c) or witness targets; Barriers = union of fixed colors; hypothesis must be **identical across trainings** or morph is **silent**.”

9. **Forbids symmetry**

* **Issue:** one-sheet states symmetric; v1.5 permits directed if asymmetry observed.
* **Fix:** “Learn directed; if both directions agree across trainings, store symmetric; else keep directed M.”

10. **Background color note**

* **Issue:** Background inference was removed in v1.5 (to avoid heuristics).
* **Fix:** Delete the background bullet entirely (or mark “viz-only, never used in logic”).

# Small clarifications to eliminate wiggle room

* **Colors ordering in all serializers/receipts:** ascending ints (log it).
* **Bit mapping:** bit j = column j (planes); serializers emit BE bytes with bit7→col0 … bit0→col7 (log this in BYTES_AND_HASHING.md).
* **Phase:** always (0,0) for lattice residues; do not search phase.
* **Admit-all handling:** must set S=0 **before** propagation.
* **Bottom pick:** only if 0 ∈ D*[p] and all scopes are 0; else **UNSAT**.

# Where to reflect these in WOs (so we catch them in code)

* **WO-02 PERIOD:** enforce **p≥2**; receipts list `row_periods_nontrivial` without 1.
* **WO-04 Downscale:** strict constant-block only; majority path does not exist.
* **WO-06 Witness:** no free translation scan; overlaps→silent; σ bijection or silent.
* **WO-09 Lattice:** no 10-cap; KMP over full length.
* **WO-10 Forbids:** permit directed; record `forbid_symmetric=true|false`.
* **WO-16C Morph:** restrict seed/barrier hypotheses as above.

# Reviewer’s 1–2 line live checks (real ARC tasks)

* **Downscale:** On any mixed block reduction case, runner must return `SIZE_INCOMPATIBLE`, not majority output.
* **Periods:** On solids/constant rows, `minimal_period_row` returns `None`; on stripes, returns the true divisor and reconstructs via residues.

If you update the one-sheet with those edits, it will be fully congruent with v1.5 and the expanded WOs, and you won’t get “spec vs code” mismatches like the WO-01 period hiccup again.


==== more common pitfalls===
WO-04 → WO-04a: Working canvas provider calling WO-14 (T9) early to compute 𝑅 𝑜 𝑢 𝑡 , 𝐶 𝑜 𝑢 𝑡 R out ​ ,C out ​ . No LCM logic. 
WO-08 (Unanimity): add per-training normalize-to-working-canvas (replicate/decimate/silent). 
WO-09 (Lattice): read normalized/transported 𝑌 𝑖 Y i ​ on working canvas; compute KMP periods there. 
WO-12 (EngineWinner): compute training-scope on working canvas; tie by priority. 
WO-14 (AggMap size): add H1..H7 with receipts; expose predict_size(X*) → (R_out,C_out). 
WO-15 (Finalize): no post-solve reduction; simply return Y on working canvas.

======common pit falls main diff====
Strong proposal — but it collides with parts of v1.5 and a few details need pinning. Here’s an objective assessment, exactly what it fixes, what it breaks, and the minimal changes to the math/computing spec + WOs so we can adopt it cleanly.

# Bottom line

* **Adopt** Patch v1.6 (“single working canvas; no LCM”) — it resolves the LCM ambiguity and unifies where all emitters place admits.
* You must **freeze** a few underspecified points (below) so it stays receipts-tight and consistent with v1.5’s “no minted bits” and “fail-closed” rules.
* We’ll need **surgical edits** to WO-08 (unanimity), WO-09 (lattice), WO-12 (EngineWinner scope computation), WO-14/15 (T9 size + finalize), and the runner wiring. Everything else remains intact.

---

## What v1.6 fixes (good)

1. **Canvas ambiguity (GAP #4)** — all layers now emit on **one** canvas ((R_{out},C_{out})) chosen **once** by T9. No “dual canvas” logic.
2. **LCM inconsistencies (GAP #2)** — eliminated completely; no lift/reduce loop.
3. **Downscale coverage (CRITICAL ISSUE #1)** — adds exact stride-downsample hypotheses (H6/H7) under T9, with **integer-only** verification, no heuristics.
4. **Witness alignment** — φ* and σ emit directly on the same working canvas; conjugation math is cleaner.

---

## What v1.6 breaks (must replace)

* v1.5 §B.2 LCM normalization (training-space) and §B.3 strict downscale after solve → **remove entirely**.
* WO-04 (LCM normalize + strict downscale) → **split** into:

  * **WO-04a**: *define working canvas* (from T9, see below);
  * **WO-15** already handles strict downscale at the end; under v1.6 this reduces to “no post-solve downscale,” because size is picked **up front** and all layers already operate on it.

---

## Underspecs in v1.6 to freeze now (critical/ high)

### C1. Size predictor order & bounds (critical)

* Your H1..H7 list is good, but **must be frozen** like v1.5 did.
  **Freeze:**

1. H1 multiplicative: (a,c\in{1..8})
2. H2 additive: (b,d\in{0..16})
3. H3 mixed affine: (a,c\in{1..8},\ b,d\in{0..16})
4. H4 constant ((R_0,C_0))
5. H5 lcm/gcd of **proper** periods (from KMP; p≥2; phase=(0,0))
6. H6 floor-stride: (k_r,k_c\in{2..5})
7. H7 ceil-stride: (k_r,k_c\in{2..5})
   **Tie rule**: smallest area (R' \cdot C'), then family id, then params lex.
   **Receipts**: `size_fit.attempts` (ordered), `winner`.

### C2. Unanimity inclusion/exclusion (critical)

* “Exclude trainings that can’t map to (R_out,C_out)” is correct, but must be deterministic.
  **Freeze:**
* Mapping operator per training is **one** of {replicate, decimate, silent}.
* **Replicate** requires integer factors (s_r,s_c) such that (R_{out}=s_r R_i, C_{out}=s_c C_i).
* **Decimate** requires exact **block-constancy** on every (s_r \times s_c) block with (R_i = s_r R_{out}, C_i = s_c C_{out}).
* If both replicate and decimate are possible (rare), pick **replicate** (record this tie rule).
* If neither holds → **silent** for that training.
* **Unanimity vote** uses only **included** trainings; if included set empty → unanimity silent everywhere.
  **Receipts:** per-training `norm_kind: "replicate"|"decimate"|"silent"`, `s_r,s_c`, `block_constancy_ok`, `aligned_hash`.

### C3. EngineWinner scope metric (high)

* With no LCM, scope must be computed **on (R_out,C_out)**.
  **Freeze:** EngineWinner is chosen by **training-scope total** on the working canvas (sum of S=1 over all trainings after normalization), tie by frozen family priority.

### C4. Lattice periods with single canvas (high)

* Lattice must compute periods directly on the **normalized/transported** training outputs in ((R_{out},C_{out})), not a separate canvas.
  **Freeze:** KMP over full rows/cols (p≥2), phase=(0,0), emit residues only where **all included trainings** agree on the color.
  **Receipts:** `p_r,p_c`, `agreeing_classes`, `disagreeing_classes`, `included_trainings`.

### C5. Witness conjugation domain (med)

* Clarify φ* computation references **Π_in** and **Π_out** established under the single canvas. No translation search; only exact **bbox equality verify** per training.
  **Receipts:** unchanged.

### C6. SIZE_UNDETERMINED exit (critical)

* If no H1..H7 fits sizes across trainings **exactly**, we must **fail-closed** before any layer runs.
  **Freeze:** return `SIZE_UNDETERMINED` with receipts: hypothesis attempts list + first counterexample.

---

## Spec diffs to publish (math + computing)

* **Delete** §B.2 LCM normalization and §B.3 post-solve downscale.
* **Add** §B.0 “Working canvas prediction” with H1..H7 + tie rule + SIZE_UNDETERMINED.
* **Edit** T2 (Unanimity) to include replicate/decimate inclusion logic and exclusion rule.
* **Note** in T1 (Witness): emits on ((R_{out},C_{out})), no LCM.
* **Note** in T3 (Lattice): KMP and agreement computed on the working canvas.

---

## Minimal changes to WOs & runner

### WO changes

* **WO-04 → WO-04a**: *Working canvas* provider calling WO-14 (T9) early to compute (R_{out},C_{out}). No LCM logic.
* **WO-08 (Unanimity)**: add per-training normalize-to-working-canvas (replicate/decimate/silent).
* **WO-09 (Lattice)**: read normalized/transported (Y_i) on working canvas; compute KMP periods there.
* **WO-12 (EngineWinner)**: compute training-scope on working canvas; tie by priority.
* **WO-14 (AggMap size)**: add H1..H7 with receipts; expose predict_size(X*) → (R_out,C_out).
* **WO-15 (Finalize)**: no post-solve reduction; simply return Y on working canvas.

### Runner wiring

* **Stage order change:** size prediction must happen **before** any emitter.

```
C = color_universe(...)
frames_in/out = canonicalize(...)
R_out,C_out, size_fit = predict_size_from_trainings(...)
emits = []
emits += [emit_witness(..., canvas=(R_out,C_out))]
emits += [emit_unanimity(normalize_train_outputs_to_canvas(..., R_out,C_out))]
emits += [emit_lattice(... on working canvas ...)]
D* = lfp_propagate(...)
engine = pick_engine_winner(...)
Y = select(...)
return Y  # no finalize downscale
```

* **Stage naming:** replace previous “LCM” mentions with “working canvas.”

---

## Receipts additions/changes

* `working_canvas`: `{"R_out":…, "C_out":…, "size_hypothesis":"Hk", "params":{...}}`
* Unanimity per-training: `norm_kind`, `s_r,s_c`, `block_constancy_ok`, `aligned_hash`
* Drop all `lcm_shape` fields anywhere.
* Keep double-run identity and section hashes unchanged.

---

## Risks & mitigations

* **Fewer trainings included in unanimity** (when neither replicate nor decimate works). That’s acceptable because other engines (witness, lattice, ICL) remain; unanimity is **optional** by design.
* **Edge tasks using heterogeneous size transforms** across trainings may yield `SIZE_UNDETERMINED`. That’s correct per “no guessing.”

---

## Reviewer’s 2-line checks (real ARC)

* A task where outputs differ in size must yield a **single** ((R_{out},C_{out})) via H1..H7 with a fully logged `size_fit.attempts`.
* Unanimity receipts must show some trainings **excluded** when they cannot be replicated or decimated to the working canvas; no LCM fields anywhere.

---

## Final recommendation

Proceed with v1.6. It simplifies implementation, removes the biggest conceptual ambiguity, and stays faithful to the “no heuristics, receipts-first” contract. Apply the freezes above, update the doc, and patch the affected WOs/runner exactly as listed.
