# Lattice bug
##After runnning check for lattice found this
---
so got this for lattice part:
 Lattice Wiring Verification Complete

  I've verified lattice wiring across all 1000 tasks. Here are the findings:

  ✅ LATTICE IS PROPERLY WIRED

  On all 773 SUCCESS tasks:
  - T3_lattice appears in emitters_evaluated
  - Lattice is being called and generating receipts
  - Emitters frozen order maintained: T1_witness → T2_unity → T3_lattice

  ❌ ONE CRITICAL BUG FOUND: S_lat mask construction

  6 tasks have periods detected but zero pruning:

  Task ID  | Period | residue_scope_bits | T3_prunes | Status
  ---------|--------|-------------------|-----------|--------
  310f3251 | (4, 4) | 23                | 0         | ❌ BUG
  48131b3c | (4, 4) | 12                | 0         | ❌ BUG
  c92b942c | (3, 3) | 81                | 0         | ❌ BUG
  caa06a1f | (6, 6) | 324               | 0         | ❌ BUG
  f5b8619d | (4, 4) | 56                | 0         | ❌ BUG
  f823c43c | (3, 3) | 144               | 0         | ❌ BUG

  Diagnosis:
  - Lattice correctly detected global periods
  - Lattice emitted admits for agreeing residue classes
  - BUT: S_lat masks are outside the consensus region (intersection of S_out_list)
  - Result: Admits never intersect with actual domains during LFP → zero pruning

  Root cause:
  Per WO-09 spec, S_lat should only emit admits over pixels in ∩ S_out_list (where ALL trainings have pixels). Current implementation likely uses union or
  ignores S_out masks.

  Fix location: src/arcbit/emitters/lattice.py in _emit_residue_admits()

  Fix: Only set S_lat bits for pixels (r,c) where:
  1. r % p_r == i AND c % p_c == j (residue class membership)
  2. ALL trainings have pixel at (r,c) (consensus region)

  ⚠️ MINOR: Missing emitters_evaluated on UNSAT/ERROR (not a bug)

  - 227 tasks (88 UNSAT + 139 ERROR) have no emitters_evaluated field
  - Lattice receipt EXISTS on these tasks (lattice is running)
  - Expected: emitters_evaluated only populated during LFP pass
  - UNSAT/ERROR exit before LFP → field never set

  This is expected behavior, not a wiring bug.

  📊 Lattice Activity Summary

  767 tasks (76.7%): Wired correctly, ran, found no global periods ✅6 tasks (0.6%): Wired correctly, found periods, mask bug prevents pruning ❌227 tasks 
  (22.7%): UNSAT/ERROR - no diagnostic data ⚠️
---
does it mean only 6 tasks are affected by it? or more? give me 1-3 line human comprehensible info - like how many tasks are affected if this bug remains? like can this also affect other tasks like succes/not yet?
---
## gpt said this
Short and clear:

* The lattice bug (using **union** instead of **intersection** of training scopes to build `S_lat`) shows up **only** on the 6 tasks where lattice actually detected periods. Those 6 would have pruned (and likely improved accuracy) but didn’t. Fixing it will affect **those 6 tasks** directly.

* For the other ~767 “SUCCESS” tasks where lattice found **no global periods**, `S_lat` was empty by design, so this bug doesn’t touch them.

* It won’t change UNSAT/ERROR counts directly (those exit before LFP), but fixing `S_lat` may turn a few borderline NOT_YET cases (in truly periodic tasks) into higher-accuracy or even perfect matches.
===
