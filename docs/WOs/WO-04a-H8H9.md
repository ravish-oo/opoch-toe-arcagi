# Sub-WO-04a-H8H9 — Plug H8 & H9 into `choose_working_canvas`

## Purpose

Extend **WO-04a** so `choose_working_canvas(...)` evaluates **H1…H9** (not just H1…H7) and picks a single working canvas ((R_{out}, C_{out})) using the same frozen tie rules and receipts. No other interfaces change.

## Interfaces (unchanged)

* `choose_working_canvas(train_pairs, frames_in, frames_out, xstar_shape) -> (R_out, C_out, size_fit_receipts)`
* Uses `agg_features` (WO-14) for each training; WO-14 now emits H8/H9 attempts/winner logic (you already added).

## Exact changes (mechanical)

1. **Import & feature prep (unchanged)**

   * Keep using **WO-14** `agg_features(X_i, C_order)` for all trainings; keep `test_features = agg_features(X*, C_order)` to compute test area.

2. **Attempt enumeration**

   * **Call WO-14’s enumerators** (or reuse the same deterministic loops) to generate attempts for:

     * H1…H7 (existing)
     * **H8:** feature-linear (bounded small-integer), using the **frozen feature basis** and loop order you added in Sub-WO-14b
     * **H9:** guarded piecewise over the **frozen guard set** and clause families {H1,H2,H6,H7}
   * For each attempt:

     * Compute **fit_all** over **all trainings** (sizes only; no content checks).
     * Record `{"family":"Hk","params":{...},"ok_train_ids":[...],"fit_all":bool}` in `attempts` **in the exact loop order**.
     * If `fit_all` is true, compute **test-area** (R^*\cdot C^*) using the frozen rule:

       * H1…H7: as before
       * **H8:** evaluate rows/cols from (\phi(X^*)) with the chosen coefficients
       * **H9:** evaluate the **guard on (X^*)** and apply the winning clause family’s params to ((H^*,W^*))
     * Append `("Hk", params, test_area)` to `candidates`.

3. **Winner selection (unchanged rule)**

   * Choose the candidate with **smallest test-area**, then by **family id order** `H1 < … < H7 < H8 < H9`, then by **param tuple lex** (as frozen in WO-14).
   * Produce `(R_out, C_out)` by applying the winning family to ((H^*, W^*)`.

4. **Failure**

   * If **no** candidate fit all trainings, raise `SIZE_UNDETERMINED` with the **full `attempts` trail** and `first_counterexample` (unchanged schema).

## Determinism guards (must hold)

* **Family order** now includes `H8`, `H9` **after** `H7`.
* **Attempt order** for each family exactly matches WO-14 (especially H8 rows-then-cols nesting, and H9 guard→true-clause→false-clause nesting).
* Tie rule unchanged: **min test-area**, then **family id**, then **params lex**.

## Receipts (add-only; same section)

* In `working_canvas`:

  * `attempts`: now contains H8/H9 entries with their params (no schema change).
  * `winner`: may now have `"family": "H8"` or `"H9"`.
  * Keep all existing keys (`features_hash_per_training`, `verified_train_ids`, `section_hash`).
* **Do not** rename or remove any keys; this is backward-compatible.

## Pitfalls to avoid (based on past bugs)

* **H5 one-sided periods:** still use **identity** on missing axes (already fixed).
* **H3 bounds:** keep (a,c \in 1..8), (b,d \in 0..16).
* **H8/H9 loops:** do **not** prune or early-exit; exhaustive bounded enumeration only.
* **Test-area for H9:** evaluate guard on **test input** (X^*), not on trainings.

## Reviewer quick check (real ARC; 1–2 lines)

* On a “size = #objects” task, `working_canvas.winner.family == "H8"` and all trainings are `fit_all=true`; `attempts` include H8 in proper order.
* On a “if periodic then scale else identity” task, `winner.family == "H9"`, and the guard partitions trainings consistently; no `SIZE_UNDETERMINED`.

## Runner impact

* None beyond `choose_working_canvas` now potentially returning H8/H9 winners. The runner’s M1 flow remains: build features → choose working canvas → log receipts → return placeholder Y.

---

This is purely **plumbing**: you’re just letting WO-04a consider the two new families you already defined in WO-14. No schema breaks, no LCM resurrected, and receipts stay exhaustive and deterministic.
