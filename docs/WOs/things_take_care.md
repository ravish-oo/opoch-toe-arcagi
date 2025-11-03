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
