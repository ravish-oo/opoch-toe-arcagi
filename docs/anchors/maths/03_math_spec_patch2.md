Spec Patch v1.4 — Close the Last Gaps

P-1. Output size prediction (T9) — make it explicit

1) What T9 does

T9 (Aggregate Mapping) doesn’t just pick a constant color; it also predicts the output dimensions from input features with a finite, frozen hypothesis class. It returns:
	•	a color (optional; constant-color families),
	•	and a size function (R_{\text{out}}, C_{\text{out}}) = F(X).

Both must exactly fit all trainings; otherwise T9 doesn’t apply.

2) Frozen feature vector \phi(X)

Compute once per input X (original color space \mathcal C):
	•	Counts: h(c)=\sum_{r,c} \mathbf 1[X= c] for all c\in\mathcal C.
	•	Connected components (4-CC) per color:
	•	n_{cc}(c), multiset \{ \text{areas}(c)\} summarized by (\min,\max,\sum).
	•	Periods: row/col minimal periods (p_r, p_c) and their LCM/GCD: \mathrm{lcm}_r,\mathrm{lcm}_c,\mathrm{gcd}_r,\mathrm{gcd}_c.
	•	Input dims: H, W.
	•	(Optional but frozen) object lattice vectors if available: (\Delta r_k, \Delta c_k) for small k.

Receipt: agg.features = {counts, n_cc, area_stats, periods, H, W}.

3) Frozen hypothesis class \mathcal H for size

Evaluate these in order and pick the first that fits all trainings exactly:

Affine & multiples
	•	R=aH+b,\ \ C=cW+d with a,b,c,d\in\mathbb Z,\ |a|,|c|\le 4,\ |b|,|d|\le 64.
	•	Multiples of periods: R = k_r \cdot \mathrm{lcm}_r,\ C=k_c \cdot \mathrm{lcm}c, 1\le k{\bullet}\le 8.
	•	Product of CC counts with small constants:
	•	R = \alpha_0 + \sum_{c\in\mathcal C_{\text{seen}}}\alpha_c n_{cc}(c) with \alpha\in\{0,1,2,3,4\}.
	•	Same for C.

Canonical templates (common ARC families)
	•	Scale: R=s_r H,\ C=s_c W with s_{\bullet}\in\{2,3,4\}.
	•	Tall/skinny: R=n_{cc}(\text{nonzero}),\ C=1 or R=1,\ C=n_{cc}(\text{nonzero}).
	•	Tiled-grid: R = t_r \cdot r_0,\ C= t_c \cdot c_0 where r_0,c_0 are motif sizes from lattice layer (if available), t_{\bullet}\in[1..8].

Tie rules (frozen):
	1.	Smallest tuple (|a|+|b|+|c|+|d|) or (k_r+k_c) etc.
	2.	Then lex by family id.
	3.	Then lex by parameter bytes.

Receipt:
agg.size_fit = {family:"affine"|"period_lcm"|..., params, verified_train_ids:[...], attempts:[...tried...]}.

If none fits all trainings → status:"NONE" and T9 doesn’t speak about size (runner must use other layers or fail-closed).

⸻

P-2. Selection Step “Law” naming — unambiguously define the layer

The selector has three scoped buckets, in this order:
	1.	Witness (geometric + σ admits)
	2.	EngineWinner (the single co-observed engine that proved fit on all trainings and emitted admits: ICL-Kron/Conv, Morph, Lattice, AggMap-constant, etc.)
	3.	Unanimity (block constants)

Replace the vague “Law” by EngineWinner everywhere. In code & receipts:

"selection": {
  "precedence": ["witness","engine_winner","unanimity","bottom"],
  "counts": {"witness": N1, "engine_winner": N2, "unanimity": N3, "bottom": N0}
}

Selector (scope-gated):
	•	cand = D*[p] ∩ A_w[p] if S_w[p] else ∅
	•	else cand = D*[p] ∩ A_E[p] if S_E[p] else ∅
	•	else cand = D*[p] ∩ A_uni[p] if S_uni[p] else ∅
	•	else select bottom (color 0; guaranteed in \mathcal C)

Containment assertion: chosen color \in D^\*_p.

⸻

P-3. Theoretical completeness — scope, and how the spec meets it

You’re also right: some ARC tasks are higher-order programs. Here’s how we square that with a deterministic, seconds-fast solver:

What the spec does cover (deterministically)
	•	Geometry & recolor (T1 Witness, T10 GL(2,ℤ) bounded): D4 + integer affine blocks.
	•	Consensus / repetition / tiling: T2 Output union, T4 Unanimity, T3 Lattice.
	•	Scaling & convolutional replication: T5 ICL-Kron, T6 ICL-Conv.
	•	Fixed-point morphologies: T7 Morph (flood-fill, gravity, hull, etc.) with bounded iterations, receipts-tight.
	•	Local logic with global guards: T8 Logic-guard (k×k windows + learned truth table) with optional global predicates from T9 features.
	•	Global constant mapping: T9 AggMap (color + size prediction), now explicit.
	•	Normalization for mixed output sizes: LCM/GCD scale-up/down—all trainings align for unanimity; downscale receipts prove reversibility.
	•	Admit & Propagate: monotone intersection + scope gating; no minted bits.

In practice, these layers cover the vast majority of ARC patterns—tilings, flips, scalings, palette swaps, fills, counting-based sizes and constant outputs, “apply seen kernel,” etc.—with no heuristics.

What may remain (and how we handle it)
	•	Program-like tasks (“apply the transform from top half to bottom half”, multi-step pipelines, backtracking constraints).
We handle these by explicit extensions (if needed), frozen in scope:
	1.	Sequential Composition (T12): allow 2–3 staged passes: stage 1 emits admits (e.g., extract objects), propagate; stage 2 consumes stage-1 selection as features, emits new admits (e.g., sorted placement). Receipts log stage graphs. Bounded (max 3 stages).
	2.	Exclusion Constraints (T13): express “different-neighbor color” as negative admits F[p] and intersect D \leftarrow D \cap \overline{F}. Receipts: forbidden bits per pixel.
	3.	CSP Backtracking (T14, optional): only if all positive layers leave multi-valued domains and a finite, low-branching search exists (e.g., 4–9 choices total). Deterministic order; receipts include tree trace and proof the chosen leaf satisfies all constraints.

These three are off by default. They are not needed for the canonical ARC families, but they are spelled out if you decide to support “program-ish” edge cases.

⸻

One-page engineer view (what to change in code)
	1.	Color universe (simple fix)
\mathcal C = \{0\} \cup \mathrm{colors}(X^\*) \cup \bigcup_i \mathrm{colors}(X_i)\cup \bigcup_i \mathrm{colors}(Y_i).
Receipt: color_universe, added_from_test.
	2.	T9 size predictor (add)

	•	Build frozen features; evaluate hypothesis functions; pick the first that fits all trainings; receipts record attempts & winner.
	•	On test, compute (R_{\text{out}},C_{\text{out}}) from the chosen function.

	3.	Rename selection’s “Law” → “EngineWinner”

	•	Only the single engine that proves fit contributes admits (others remain silent).
	•	Update receipts & counts accordingly.

	4.	Keep everything else as frozen

	•	Witness on input frames (no palette).
	•	Output union/unanimity on output frames (LCM-normalized sizes if needed).
	•	Lattice, ICL-Kron/Conv, Morph, Logic-guard emit (A,S) admits, not painted grids.
	•	Admit & Propagate: intersect only where S=1; normalize “admit-all ⇒ S=0”.
	•	Selector: witness → engine_winner → unanimity → bottom; assert containment.

⸻

Why this resolves the raised issues
	•	Output size mapping is now explicit (fits trainings or it doesn’t).
	•	“Law layer mystery” is gone (it’s the winner engine; named and receipts-logged).
	•	“Completeness” is stated honestly: core ARC is covered by T1–T11; if you choose to support rarer program-like tasks, T12–T14 give you bounded, receipts-tight paths (sequential, exclusion, optional CSP). There’s no silent guessing anywhere.

⸻

Sanity check on your two examples
	•	“Most frequent color” → covered by T9 AggMap (constant color from histogram).
	•	Variable output sizes → covered by LCM normalization for unanimity/union plus T9 size predictor for the test case.
	•	Tiling + flip with a new test-only color → already works: we include \(\mathrm{colors}(X^\)\) in \mathcal C; witness geometry acts on positions, σ is partial identity, lattice/ICL-Conv emit admits from \(X^\\) bit-planes in original colors.

⸻
