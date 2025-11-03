# Sub-WO-14b — Add H8 (Feature-Linear) and H9 (Guarded Piecewise)

## Purpose

Extend the **size hypothesis class** with:

* **H8**: small-integer **feature-linear** laws (covers “#objects”, “#pixels”, etc.).
* **H9**: **guarded piecewise** laws over a frozen predicate set (covers “if periodic then…, else …”).

Everything remains trainings-only, finite, deterministic, and receipts-tight.

---

## Interfaces (unchanged)

You **do not** change public signatures. You only extend:

* `agg_size_fit(train_pairs, features) -> SizeFit | None` to include H8, H9 attempts and winner.
* `predict_size(H*,W*,fit) -> (R_out,C_out)` works unchanged (applies winner to test).

Receipts: the existing `size_fit.attempts` and `winner` now include `"family": "H8"` or `"H9"` entries when relevant. No schema breaks.

---

## Frozen additions

### A) Feature basis (for H8/H9, deterministic)

Use only these **input-derived** scalars (integers), computed from WO-14/WO-05:

* `1` (intercept)
* `H`, `W`
* `sum_nonzero` := (\sum_{c≠0} h(c))
* `ncc_total` := (\sum_{c≠0} n_{cc}(c))
* Up to **Kc = 2** per-color counts for the two **most frequent non-zero colors across all trainings combined**, breaking ties by ascending color id. Call them `count_c1`, `count_c2`. (If fewer than 2 non-zero colors appear, missing features are treated as 0.)

Let the **feature vector** be:
[
\phi = [1,\ H,\ W,\ \mathrm{sum_nonzero},\ \mathrm{ncc_total},\ \mathrm{count_c1},\ \mathrm{count_c2}]
]
with a fixed index order ([f_0..f_6]).

> This is finite, deterministic, and maps directly to common size laws like “width equals number of objects” (use `ncc_total`) or “height equals count of a color.”

---

### B) **H8 — Feature-Linear (bounded integer)**

**Definition:**

* Predict rows and columns **independently** with up to **two** non-intercept features and small integer coefficients:

[
R' = \alpha_0\ +\ \alpha_{i_1} f_{i_1}\ +\ \alpha_{i_2} f_{i_2}\quad (i_1<i_2,\ i_k \in {1..6},\ k\in{1,2})
]
[
C' = \beta_0\ +\ \beta_{j_1} f_{j_1}\ +\ \beta_{j_2} f_{j_2}\quad (j_1<j_2,\ j_k \in {1..6})
]

**Bounds (frozen):**

* Intercepts: (\alpha_0,\beta_0 \in [0..32])
* Coefficients: (\alpha,\beta \in {-3,-2,-1,1,2,3}) (no 0 for selected features)
* Feature selection: (k_R,k_C \in {0,1,2}).

  * If (k_R=0) then (R'=\alpha_0) (pure constant), same for (C').

**Fit criterion:**
For a parameter tuple ((\alpha, \beta)), for **every** training (i), compute ((R'_i, C'_i)) from (\phi(X_i)). It **fits** iff ((R'_i,C'_i) = \mathrm{shape}(Y_i)) for all trainings.

**Attempts enumeration order (frozen):**

* **Rows** first, then **cols** (so `attempts` is stable):

  1. Enumerate (k_R \in {0,1,2}). For (k_R=0): loop (\alpha_0) only; else choose feature index sets (S_R) of size (k_R) in ascending lex order from ({1..6}); then (\alpha_0) in 0..32; then coefficients for the selected features in ({-3,-2,-1,1,2,3}) lex order.
  2. For each row choice, enumerate **column** side in the same pattern for (k_C), (S_C), (\beta_0), (\beta).
* Each attempt appends:

  ```json
  {"family":"H8","params":{"R":{"k":kR,"S":S_R,"a0":α0,"a":coeffs}, "C":{"k":kC,"S":S_C,"b0":β0,"b":coeffs}},
   "ok_train_ids":[...], "fit_all": true|false }
  ```

**Test tie-area:**

* For the **test input** (X^*), compute (\phi^* = \phi(X^*)); then:
  (R^* = \alpha_0 + \sum \alpha_{i} f_{i}^*,\ C^* = \beta_0 + \sum \beta_{j} f_{j}^*); tie-area is (R^*\cdot C^*).

> This keeps H8 finite, still powerful enough to capture the 139 cases in practice (objects/pixels/#colors), and remains receipts-tight.

---

### C) **H9 — Guarded piecewise (single guard, two clauses)**

**Guards (frozen; computed from input features of (X)):**

* (g_1): has_row_period := (`periods.row_min` is not None)
* (g_2): has_col_period := (`periods.col_min` is not None)
* (g_3): ncc_total > 1
* (g_4): sum_nonzero > (\lfloor H\cdot W/2 \rfloor)   (majority non-zero)
* (g_5): H > W

**Clause families allowed (to keep finite):** ({\mathbf{H1}, \mathbf{H2}, \mathbf{H6}, \mathbf{H7}}) with the same bounds as in WO-14.

**Definition:**
Pick one guard (g \in {g_1..g_5}). Choose **two** clause families (F_{\text{true}}, F_{\text{false}}) from the set above, with their own parameters (\theta_T, \theta_F). The size law is:
[
(R',C') =
\begin{cases}
F_{\text{true}}(H,W;\ \theta_T) & \text{if } g(X)\ \text{is true} \
F_{\text{false}}(H,W;\ \theta_F) & \text{otherwise}
\end{cases}
]

**Fit criterion:**
For **all trainings**, evaluate the guard on (X_i) and apply the corresponding clause; the resulting ((R'_i,C'_i)) must equal ((R_i,C_i)). Fit only if **every** training matches.

**Attempts enumeration order (frozen):**

* Guards outer (g1..g5). For each guard:

  * `F_true` family in order H1→H2→H6→H7 with their param loops; nested inside, `F_false` family in the same order with its param loops.
* Each attempt appends:

  ```json
  {"family":"H9",
   "params":{"guard":"g3","true":{"family":"H6","params":{"kr":2,"kc":3}},
                     "false":{"family":"H2","params":{"b":0,"d":2}}},
   "ok_train_ids":[...], "fit_all": true|false}
  ```

**Test tie-area:**

* Evaluate the guard on (X^*). Apply the corresponding clause family to ((H^*,W^*)) to get ((R^*,C^*)). Tie-area is (R^*\cdot C^*).

> H9 adds the minimum guarded expressiveness without a catalog explosion.

---

## Tie rule (unchanged)

If multiple families fit (including H8/H9), choose by:

1. **Smallest test area** (R^*C^*),
2. Then **family id order**: H1 < H2 < H3 < H4 < H5 < **H6 < H7 < H8 < H9**,
3. Then **parameter lex** (frozen encoding; for H8, encode (k, S, coeffs); for H9, (`guard_id`, `true.family`, `true.params`…, `false.family`, `false.params`)).

> Adding H8/H9 at the end preserves prior priorities.

---

## Invariants

* **Trainings-only selection** (no test leakage beyond tie-area computation).
* **Finite search**, deterministic orders as frozen.
* **Integer-only**, no heuristics, no learning beyond bounded enumeration.

---

## Receipts (add-only)

Continue to log in `size_fit`:

* Each new attempt with `"family":"H8"` or `"H9"` and their `"params"`.
* `winner` may now be H8 or H9.
* **Optional** (nice audit): append a `"families_present": ["H1","H2",...,"H9"]`.

No breaking changes.

---

## Failure modes

If **no** family (H1…H9) fits: `agg_size_fit` returns `None` (WO-04a will surface `SIZE_UNDETERMINED` with the attempts trail). That remains the honest fail-closed path.

---

## Reviewer quick-verification (real ARC; 1–2 lines)

* On a task where size = number of objects (e.g., (C' = ncc_total)), verify that `size_fit.winner.family == "H8"` and all trainings fit exactly.
* On a task with “if periodic then scale, else identity,” verify `size_fit.winner.family == "H9"` and guard truth partitions the trainings consistently.

---

## Notes on complexity (kept small but exact)

* H8 searches at most:

  * Rows: choose ≤2 features from 6 → (1 + 6 + \binom{6}{2} = 22) selections; each with (\alpha_0\in 33) and coefficients (\in 6^{k_R}).
  * Columns similarly → worst-case attempts in the low millions across 1000 tasks, but each check is **just integers** and ARC sizes are tiny.
* H9 composes two of {H1,H2,H6,H7} under 5 guards; the combined attempts are manageable and fully deterministic.

If any of the 139 tasks are still `SIZE_UNDETERMINED` after H8/H9, the receipts will make it clear what law they need, and we can consider one more bounded family (e.g., motif-based via witness) **add-only** without breaking anything.
