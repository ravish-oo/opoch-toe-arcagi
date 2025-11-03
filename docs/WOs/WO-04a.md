WO-04a you asked for, locked to v1.6 (“single working canvas”), pure CS, and receipts-first. I’ll also address your downscale pitfall note at the end.

# WO-04a — Working Canvas Provider (calls T9 early)

## Purpose

Pick the **single working canvas** size ((R_{\text{out}}, C_{\text{out}})) **once, up front**, using the frozen T9 size hypothesis class (H1–H7) evaluated **only on trainings**. No LCM anywhere.

---

## Interface (pure; no I/O; receipts-first)

```python
def choose_working_canvas(train_pairs, frames_in, frames_out, xstar_shape) -> (int, int, dict):
    """
    Inputs:
      train_pairs : list of dicts [{ "X": Gx_i, "Y": Gy_i }, ...]  (raw integer grids)
      frames_in   : list of Π_in objects for each X_i (pose_id, anchor)  # from WO-03
      frames_out  : list of Π_out objects for each Y_i (pose_id, anchor) # from WO-03
      xstar_shape : (H*, W*)  # test input dimensions (raw grid shape)

    Process:
      1) Build frozen T9 features for each training input/output (WO-14), integers only.
      2) Evaluate H1..H7 in the frozen order. A hypothesis FITS iff it reproduces *each* training Y_i.size exactly.
      3) Tie-break winners by smallest area (R'·C'), then family id ("H1".."H7"), then param tuple lex.
      4) Apply the winning formula to the test input size (H*, W*) to yield (R_out, C_out).

    Outputs:
      R_out, C_out  # working canvas
      size_fit      # receipts payload: winner, params, ordered attempts, verified ids

    Fail-closed:
      If no hypothesis fits all trainings → raise SizeUndetermined with receipts.
    """
```

**Notes**

* `frames_in/frames_out` are *not* used to change sizes; they are included for parity with the overall pipeline and for receipts (pose/anchor provenance). Size hypotheses act on **raw H,W** only.

---

## Frozen hypothesis class (v1.6; exact bounds)

**Order of evaluation (must not change):**

1. **H1 (multiplicative):** (R' = aR,\ C' = cW,\ a,c \in {1..8})
2. **H2 (additive):** (R' = R + b,\ C' = W + d,\ b,d \in {0..16})
3. **H3 (mixed affine):** (R' = aR + b,\ C' = cW + d,\ a,c \in {1..8},\ b,d \in {0..16})
4. **H4 (constant):** (R' = R_0,\ C' = C_0) (same for all trainings)
5. **H5 (period lcm/gcd, integer):** derive from **proper** (p≥2) KMP row/col periods; phase=(0,0)
6. **H6 (floor stride):** (R'=\lfloor R/k_r \rfloor,\ C'=\lfloor W/k_c \rfloor,\ k_r,k_c \in {2..5})
7. **H7 (ceil stride):** (R'=\lceil R/k_r \rceil,\ C'=\lceil W/k_c \rceil,\ k_r,k_c \in {2..5})

**Fit criterion (strict):** For a candidate family+params, compute ((R'_i,C'_i)) from each training input ((R_i,W_i)); it **fits** iff ((R'_i,C'_i) = \text{shape}(Y_i)) for **every** training (i). No content checks here (content checks live later in T2 normalization or family-specific verifiers).

**Tie rule (frozen):**

* First, smallest area (R' \cdot C') (computed on the test input’s ((H^*, W^*))).
* Then, smaller family id (“H1” < “H2” < …).
* Then, parameter tuple lex (e.g., ((a,c)), then ((b,d)), then ((a,b,c,d)), etc).

**If no fit:** `SIZE_UNDETERMINED` (fail-closed), with receipts showing all attempts and the first counterexample.

---

## Invariants (must hold)

* **Trainings-only**: No test leakage in hypothesis selection.
* **Determinism**: Given identical inputs, winner and params are identical (frozen order, frozen tie rules).
* **No LCM**: This WO does not compute or reference any LCM canvas.
* **No heuristics, no floats**: Integer arithmetic only (counts, H,W, KMP periods, divisions, floors/ceils in H6/H7).

---

## Receipts (first-class; additive; stable)

Section name: `"working_canvas"` (single section per task).

**Required fields:**

* `features_hash_per_training`: array of BLAKE3 hashes of each training’s frozen T9 feature vector (from WO-14), in index order.
* `attempts`: ordered array; each element:

  ```json
  {
    "family": "Hk",
    "params": { ... },
    "ok_train_ids": [0,1,...],     // trainings for which sizes matched
    "fit_all": true|false
  }
  ```
* `winner`: `{ "family": "Hk", "params": {...} }`  (present only if fits)
* `R_out`: int, `C_out`: int
* `verified_train_ids`: `[0,1,...]`  // should be all trainings when a fit exists
* `section_hash`: string (BLAKE3 over the canonical JSON per WO-00)

**Failure receipts (on SIZE_UNDETERMINED):**

* Include `attempts` as above,
* `first_counterexample`: `{ "train_id": i, "expected": [Ryi, Cyi], "predicted": [R'i, C'i], "family": "Hk", "params": {...} }`

---

## Exact algorithm (reference pseudocode)

```python
def choose_working_canvas(train_pairs, frames_in, frames_out, xstar_shape):
    # 0) Collect raw sizes
    sizes_in  = [(len(p["X"]), len(p["X"][0])) for p in train_pairs]
    sizes_out = [(len(p["Y"]), len(p["Y"][0])) for p in train_pairs]
    Hs, Ws = zip(*sizes_in)
    YsH, YsW = zip(*sizes_out)
    Hstar, Wstar = xstar_shape

    # 1) Features per training (WO-14)
    feats = [agg_features(p["X"], p["Y"]) for p in train_pairs]  # frozen contents: counts, H,W, CC stats, proper periods...

    # 2) Enumerate hypotheses H1..H7 in order; collect attempts
    attempts = []
    winner = None

    for family, param_iter in frozen_param_generators(feats, sizes_in, sizes_out):
        for params in param_iter:
            ok_ids = []
            fits_all = True
            for i, (H, W) in enumerate(sizes_in):
                R_pred, C_pred = apply_hypothesis(family, params, H, W, feats[i])
                if (R_pred, C_pred) == sizes_out[i]:
                    ok_ids.append(i)
                else:
                    fits_all = False
            attempts.append({"family": family, "params": params, "ok_train_ids": ok_ids, "fit_all": fits_all})

            if fits_all and (winner is None or better_by_tie_rules(family, params, winner, Hstar, Wstar)):
                winner = {"family": family, "params": params}

    if winner is None:
        raise SizeUndetermined(receipts=seal_receipts(...attempts..., first_counterexample(...)))

    # 3) Apply winner to test input size
    R_out, C_out = apply_hypothesis(winner["family"], winner["params"], Hstar, Wstar, feats=None)

    # 4) Seal receipts and return
    receipts = {
       "features_hash_per_training": [blake3_hash(json_dumps(f, stable=True)) for f in feats],
       "attempts": attempts,
       "winner": winner,
       "R_out": R_out, "C_out": C_out,
       "verified_train_ids": list(range(len(train_pairs)))
    }
    return R_out, C_out, seal_receipts("working_canvas", receipts)
```

**Where:**

* `frozen_param_generators(...)` produce bounded integer grids for each Hk (e.g., `a,c ∈ {1..8}` etc), or compute period-based candidates for H5.
* `apply_hypothesis(...)` is pure arithmetic on integers; **no content checks** at WO-04a.

---

## Edge cases (fully specified)

* **Single training**: allowed; hypotheses must fit that one.
* **Zero trainings**: invalid task for ARC; return `SIZE_UNDETERMINED`.
* **All outputs equal to inputs**: H1 with (a=c=1) fits; tie rules still apply.
* **Inconsistent trainings** (no shared Hk): `SIZE_UNDETERMINED` (fail-closed).
* **Period extraction (H5)**: use WO-02 KMP minimal proper periods (p≥2) on the **training outputs**; phase=(0,0); combine by lcm/gcd deterministically; then compute sizes from those integers.

---

## Underspecificity risks (closed here)

* **Hypothesis order or bounds drifting**: frozen above; do not add/remove families here.
* **Tie criterion ambiguity**: frozen (min area on test input, then family id, then param lex).
* **Content checks in WO-04a**: explicitly **not** done here; they live in later modules (e.g., T2 normalization decimation needs constant blocks).
* **Frames involvement**: sizes use raw grid shapes; frames are receipts-only at this stage.

---

## Relation to the “strict downscale only” pitfall

Your earlier pitfall **WO-04: downscale must be constant-block only** referred to the old LCM normalization and post-solve reduction. Under v1.6, **WO-04 no longer exists**. Where the **strict constant-block** rule now applies:

* **T2 Unanimity normalization**: when *including* a training whose Y_i must be mapped to the working canvas by **decimation**, we require **exact block-constancy**; otherwise that training is **excluded** from unanimity (silent).
* **H6/H7 (stride)** in WO-04a: these are **size** hypotheses only; they do **not** apply content downsampling here. (If you later decide to verify stride content in sizing, do it add-only in WO-14 with receipts; not needed to pick the canvas.)

So: **WO-04a does not perform any downscale**, strict or otherwise. The strictness lives in T2 (later).

---

## Developer checklist (Implementer)

* Enumerate H1–H7 **in the frozen order** and within the specified integer bounds.
* Compare **only sizes** against each training’s Y_i.
* Apply tie rules exactly as written.
* On no fit, raise `SizeUndetermined` with receipts including `first_counterexample`.
* Do **not** inspect pixel content here; no LCM, no resampling, no heuristics.

---

## Reviewer quick-verification (real ARC; 1–2 lines)

* On mixed-size trainings, `choose_working_canvas` must yield a single ((R_{\text{out}},C_{\text{out}})) with an ordered `attempts` trail and a clear `winner`; there must be **no** `lcm_*` fields anywhere.
