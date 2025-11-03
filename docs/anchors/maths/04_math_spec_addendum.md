ARC-AGI Bit-Algebra Solver — Spec Addendum v1.5 (Final Freezes)

This document extends and locks the core spec (v1.4) so that all code paths are uniquely determined. Anything not listed here is understood to follow v1.4 verbatim.

Global rule: When an invariant cannot be satisfied exactly on all trainings, that layer is silent (no admits). We never mint bits.

⸻

A. Universals

A.1 Color universe (includes test)

\boxed{\ \mathcal C \;=\;\{0\}\ \cup\ \mathrm{colors}(X^\*)\ \cup\ \bigcup_i \mathrm{colors}(X_i)\ \cup\ \bigcup_i \mathrm{colors}(Y_i)\ }
	•	σ is partial: learned only on touched colors; identity elsewhere.
	•	Receipts: color_universe, added_from_test.

A.2 Empty domain (UNSAT)

If any pixel p reaches D^\*[p]=\varnothing at the fixed point, stop and fail-closed:
	•	status: "UNSAT", receipts log: unsat_pixel=(r,c), last_prune_layer, last_prune_edge, domains_hash_before, domains_hash_after.

A.3 Hash & determinism
	•	Hash: BLAKE3, big-endian byte streams, row-major.
	•	Double run: if any section hash differs → FAIL with first_differing_section.
	•	Receipts include: blake3_version, endianness:"BE", spec_version:"1.5", param_registry.

⸻

B. Frames, LCM & Size

B.1 Canonical D4 frame (pose)

Given grid G\in\mathbb Z^{H\times W}:
	1.	Apply 8 D4 transforms T\in\{I,R90,R180,R270,FX,FXR90,FXR180,FXR270\}.
	2.	Flatten each G_T to bytes (BE) row-major.
	3.	Choose lexicographically smallest sequence; ties broken by pose id order:
[I=0, R90=1, R180=2, R270=3, FX=4, FXR90=5, FXR180=6, FXR270=7].
	4.	Anchor: find first nonzero pixel by row-major; translate so it becomes (0,0).
Receipts: pose_id, pose_tie_count, anchor=(dr,dc).

B.2 LCM normalization for unanimity
	•	If training outputs’ shapes differ, compute R_{lcm}=\mathrm{LCM}(R_i), C_{lcm}=\mathrm{LCM}(C_i).
	•	Lift each Y_i to (R_{lcm},C_{lcm}) by block replication (exact; no interpolation).
	•	Unanimity/transport works on LCM canvas.
	•	Final test size comes from EngineWinner (usually T9). If EngineWinner size divides LCM, block-downsample (see B.3). Otherwise SIZE_INCOMPATIBLE (fail-closed).

B.3 Downscaling (LCM → final)

Partition into k_r\times k_c blocks:
	•	If uniform color in the block → write that color.
	•	Else majority vote; tie → lex-min color.
Receipts: downscale_majority_used: true|false, lcmsize, finalsize.

B.4 Size predictor (T9) — explicit & finite

Features (frozen; integers only):
	•	H,W
	•	per-color counts h(c)
	•	per-color 4-CC counts n_{cc}(c) and area stats (min,max,sum)
	•	minimal row/col periods (p_r,p_c), plus \mathrm{lcm}_r,\mathrm{lcm}_c,\mathrm{gcd}_r,\mathrm{gcd}_c

Hypotheses (order frozen; first fit wins):
	1.	Affine: R=aH+b,\ C=cW+d, a,c\in[-4..4],\ b,d\in[-64..64].
	2.	Period multiples: R=k_r\cdot \mathrm{lcm}_r,\ C=k_c\cdot \mathrm{lcm}c,\ k\bullet\in[1..8].
	3.	CC-linear: R=\alpha_0+\sum_c \alpha_c n_{cc}(c), \alpha_\bullet\in[0..4]. (Same for C.)
	4.	Scale: R=s_r H,\ C=s_c W,\ s_\bullet\in\{2,3,4\}.
	5.	Tiled-grid: R=t_r r_0,\ C=t_c c_0 using motif dims from lattice (if present), t_\bullet\in[1..8].

Ties: minimal \sum|\text{params}|; then family id; then param bytes lex.
Receipts: agg.size_fit = {family, params, verified_train_ids, attempts}.

⸻

C. Propagation & AC-3

C.1 Admit-intersect pass
	•	Intersect admits in fixed family order: T1..T12.
	•	Intersect only where scope S=1; normalize admit-all ⇒ S=0.

C.2 AC-3 pruning (if forbids present)
	•	Graph E: 4-neighbor on Π_out* canvas (no diagonals, no wrap).
	•	Initialize queue with all edges (p,q), sorted lex by ((p.r,p.c),(q.r,q.c)).
	•	FIFO processing; on prune at p, enqueue (r,p) for all neighbors r in increasing (r,c).
	•	Iterate passes until no domain bit changes.
	•	Bounds: MAX_PROPAGATION_ITERATIONS = 1000; if hit → propagation_hit_limit:true but hashes still deterministic.
Receipts: admit_passes, ac3_passes, queue_init_len, total_prunes.

⸻

D. Witness (T1)

D.1 Translation bounds

For a source bbox of size h\times w in train-Π frame and target canvas R_{out}\times C_{out}:
	•	Enumerate translations t=(dr,dc)\in[-R_{out}..R_{out}]\times[-C_{out}..C_{out}].
	•	D4 poses: all 8.
	•	Accept a piece only if exact pixel equality holds over the bbox (after pose+shift).
Receipts: per-piece trials:[{pose,dr,dc,ok}].

D.2 Overlaps & σ
	•	If two witness pieces target the same pixel with different colors in the same training, reject the conflicting set (no σ), → that training’s witness silent (S=0).
	•	σ per training must be bijective over touched colors; conflicts → silent.
	•	σ learned by votes at matched positions; tie → lex-min output color; identity for untouched colors.
Receipts: sigma = {domain_colors, lehmer, moved_count, bijection_ok}.

⸻

E. Lattice (T3)

E.1 Period search (bounded)

Search p_r\in[2..\min(R,10)], p_c\in[2..\min(C,10)], accept the minimal pair that exactly reconstructs. Phase fixed at (0,0).
Receipts: p_r,p_c,phase, chosen_period, ties.

E.2 Multi-color classes

Only emit admits for residue classes where all trainings agree on the color; else silent for that class.
Receipts: agreeing_classes, disagreeing_classes.

⸻

F. ICL-Kron (T4) & ICL-Conv (T5)

F.1 Kernel & factor bounds
	•	Kron: factor dims r_0,c_0\in[1..5]; tiling counts t_r,t_c\in[1..8].
	•	Conv (stencil): offsets u come from \{-2..2\}\times\{-2..2\}; limit |K|\le 9.
	•	Fit must reconstruct all trainings exactly.

F.2 Ties & overlap
	•	If multiple kernels/factors fit → pick smallest |K| (or smallest r_0\cdot c_0), then lex on offsets (or factors).
	•	Overlapping hits produce unions of admits; conflicts resolved only after propagation/selection.
Receipts: kernels_tried, kernel_winner, overlap_pixels.

⸻

G. Morph (T6)

G.1 Neighborhood & pass cap
	•	Neighborhood = 4-neighbors (frozen).
	•	Iterate R^{t+1} = R^t \cup \{p: p\in\mathcal N(R^t),\ p\notin \text{barriers}\} until no change or \;t=H+W.
	•	Seeds/barriers learned only if identical hypothesis fits all trainings; otherwise silent.
Receipts: N_type="4", passes, seed_hash, barrier_hash, grew_pixels.

⸻

H. Logic-Guard (T7)

H.1 Predicate set (explicit, bounded)
	•	Windows: 3\times 3, plus 1\times k and k\times 1 with k\in\{2,3,4\}.
	•	Local predicates (max 12 total):
	•	3×3 color hash (BE bytes; exact hash match)
	•	5×1 / 1×5 window hash
	•	4-neighbor color set (encoded as bitmask over \mathcal C)
	•	per-color neighbor count \in\{0,1,2,3,4\} for top 4 colors by training counts
	•	Positional (max 5): row index, col index, row parity, col parity, Manhattan distance to nearest edge.
	•	Global guards (subset of T9): \max_c h(c), \arg\max_c h(c), n_{cc}(\text{nonzero}), p_r,p_c.
	•	Build truth table only over predicate assignments observed in trainings; if any assignment maps to different colors across trainings → silent on that assignment.
	•	Anchor policy when multiple hits: leftmost in row mode; topmost in column mode; per-row count is frozen to the count observed in trainings (if applicable).
Receipts: predicate_ids, window_sizes, truth_rows, anchor_policy, per_row_counts.

⸻

I. Param ICL (T8)
	•	Guards n are strictly those observed in trainings.
	•	If multiple guards fire at test: choose lex-min n.
	•	If none fire: layer silent (no extrapolation).
Receipts: n_values_seen, n_fired_test, chosen_n_or_silent.

⸻

J. Aggregate Mapping (T9) — colors & size
	•	Feature set: as in B.4; no floats.
	•	Size hypotheses: as in B.4; order and parameter bounds are frozen.
	•	Color: can emit a constant color chosen by feature table; ties → lex-min color.
Receipts: features_hash, size_fit (winner & attempts), color_map_decision (if used), tie_broken.

⸻

K. Forbids (T10) + AC-3
	•	Graph E = 4-neighbor (frozen).
	•	Learn forbid matrix M(c,d) only from explicit violations in trainings; unseen pairs are permissive (0); symmetric by default.
	•	AC-3 prunes domains with the queue policy in C.2.
Receipts: forbid_symmetry:"symmetric", forbid_matrix_hash, edges_count.

⸻

L. CSP Micro-solvers (T11)
	•	Only for tiles \le 3\times 3; candidate motifs \le 9.
	•	Solve by Hopcroft–Karp; if multiple max matchings, choose lex-min by edge tuple list.
	•	Emit singleton admits only if compatible with current domains; otherwise silent (never force UNSAT by T11).
Receipts: tile_bbox, candidate_count, matchings_count, chosen_idx, cells_forced, cells_skipped_due_to_domain.

⸻

M. Strata (T12 — optional)
	•	Stratum order: S0 Evidence → S1 Witness/Lattice/Kron/Conv → S2 Morph/Logic/Param-ICL → S3 CSP.
	•	No selection between strata; each stratum reads planes from earlier ones and emits admits; global LFP at the end.
Receipts: strata_plane_hashes[S0..S3].

⸻

N. Selection (scope-gated; global EngineWinner)
	•	Buckets in order: Witness → EngineWinner → Unanimity → Bottom.
	•	EngineWinner is the single engine family that reconstructs all trainings; if multiple candidates fit, pick by specificity (most scope bits); tie by fixed priority:
T3 Lattice > T5 ICL-Conv > T4 ICL-Kron > T6 Morph > T7 Logic-guard > T8 Param ICL > T9 AggMap > T10 Forbids > T11 CSP.
	•	Bottom 0 chosen only if 0 \in D^\*[p] and all buckets are silent; otherwise UNSAT (A.2).
Receipts: engines_fitting, engine_winner, selection.counts, containment_verified:true.

⸻

O. Missing items now specified
	•	Search bounds: T1 translations, T3 period range, T5 kernel window/size, T7 predicates/windows, T11 tile/matching bounds — all frozen above.
	•	D4 lex-min algorithm: B.1.
	•	σ inference w/ overlaps: D.2 rules.
	•	Component invariants: integers only (area, bbox, perim4, outline D4-min hash).
	•	Morph learning: hypothesis must be identical across trainings; otherwise silent (FROZEN N=4, pass-cap H+W).
	•	Downscale rule: B.3.
	•	AC-3 queue order: C.2.
	•	Multi-engine ties: N.
	•	Background color (when needed): argmax(train input+output histogram); tie → 0 if present, else lex-min; receipt background_color.
	•	Output Union layer removed (only Unanimity exists).
	•	EngineWinner name frozen (replaces vague “Law”).

⸻

P. Reviewer’s Checklist (per task)
	1.	Universals: color_universe includes 0 and test colors; σ partial identity.
	2.	Frames: pose_id/anchor logged; LCM applied when trainings differ; downscale receipts.
	3.	Witness: poses & translations bounded; σ bijection OK; overlap rules respected.
	4.	Lattice/ICL/Morph/Logic/ParamICL: search bounds present; receipts show attempts and winner; silence on disagreement.
	5.	Forbids + AC-3: E=4; symmetric; queue order & prune counts logged.
	6.	Propagation: admit passes & ac3 passes; propagation_hit_limit:false; no empties unless UNSAT.
	7.	Selection: witness→engine_winner→unanimity→bottom; containment asserted; counts match receipts.
	8.	Determinism: all section hashes equal across double run.
	9.	Size: T9 chosen size matches trainings; LCM reducible or SIZE_INCOMPATIBLE.

⸻

Q. Why this now “just works”
	•	Every “enumerate/try” now has finite bounds and frozen order.
	•	Every tie has a deterministic resolution.
	•	Every ambiguous learning case falls back to silence (no minted bits).
	•	Propagation + AC-3 have fixed queue order and termination cap.
	•	Selection is scope-gated and containment-checked.
	•	Receipts are complete to reproduce any discrepancy.

This addresses—explicitly—the search bounds, predicate catalog, CSP triggers, lex-min frame, σ with overlaps, morph learning, downscaling, AC-3, engine ties, background, and the UNSAT path. With v1.5 in place, a clean implementation won’t stall on ambiguity or drift into heuristics.
