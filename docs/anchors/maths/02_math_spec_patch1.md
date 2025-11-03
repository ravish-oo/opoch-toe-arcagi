⸻

What to change (spec + code), exactly

1) Freeze the color universe

Replace the universe definition with:

\boxed{\ \mathcal C := \{0\}\ \cup\ \mathrm{colors}(X^*)\ \cup\ \bigcup_i \mathrm{colors}(X_i)\ \cup\ \bigcup_i \mathrm{colors}(Y_i)\ }

Receipts (must log):
	•	color_universe: sorted list of integers
	•	added_from_test: \mathrm{colors}(X^*) \setminus (\bigcup_i \mathrm{colors}(X_i)\cup\mathrm{colors}(Y_i))

Implementation:

def build_color_universe(X_star, X_trains, Y_trains):
    C = {0}
    C |= set(int(v) for v in np.unique(X_star))
    for X in X_trains: C |= set(int(v) for v in np.unique(X))
    for Y in Y_trains: C |= set(int(v) for v in np.unique(Y))
    return sorted(C)  # keep stable ascending order for min-selection

Why: this guarantees you can represent any color that appears in the test input, even if it never appeared in training. That’s the only way your forward-emitting layers (lattice/ICL-Conv/morph/logic) can legally place those colors.

2) Keep σ partial and identity elsewhere
	•	Learn \sigma_i only on colors it touches in training X_i \to Y_i.
	•	For any c \in \mathcal C not in \mathrm{domain}(\sigma_i), define \sigma_i(c)=c (identity).
	•	This way, forward admits from X^* can carry test-only colors through witness and other layers without being “remapped” to nonsense.

Receipts:
	•	sigma.domain_colors, sigma.lehmer (only for touched colors), moved_count
	•	No fictitious entries for unseen colors.

3) Default forbids = permissive
	•	Your admit layers are positive constraints (“what’s allowed”).
	•	If a pairwise relation M(c,d) was not learned, do not forbid it. That keeps \mathcal C bits alive in the domain D until a real proof shrinks them.
	•	This matches the lattice/intersection semantics and prevents accidental killing of new colors.

⸻

Why this fully resolves “new color in test” concerns
	•	T3 Lattice / T5 ICL-Conv / Morph / Logic-guard read bit-planes from X^* and emit those colors as singleton admits. With \mathcal C\supseteq \mathrm{colors}(X^*), those exact integers are representable.
	•	T1 Witness geometry works on positions; recolor \sigma is only applied where proven. Unseen test colors pass unchanged (identity), so forward admits stay correct.
	•	Output-transport / Unanimity never ban “new” colors. They emit singletons only where scoped; otherwise they’re silent (“admit-all ⇒ scope=0”), which doesn’t intersect away test-only color bits.
	•	The fixed-point intersection shrinks D only where some layer truly speaks; selection then picks a color inside D^\*. No invention, no loss.

⸻

Edge cases you also close with this patch
	1.	0 not present in data → still included so the bottom path stays inside the lattice (containment invariant stays true).
	2.	New constant color in Y^* not in \mathrm{colors}(X^*):
	•	If it’s author-intent (present in all Y_i), output-transport/unanimity will place it.
	•	If it’s a feature-constant (“paint 9 if #objects=3”), your Aggregate Mapping layer covers it (learned constant from global features).
	3.	Determinism: the color index ordering is now stable (ascending integers), so “minbit” precedence and receipts stay reproducible.

⸻

Quick sanity tests to prove it
	•	“Test-only color” check: Construct a toy task with training colors \{1,3\}, test X^* includes 2 and the rule is a 2×2 lattice tiling. With \mathcal C patched, your T3 admits will place 2; without the patch they can’t.
	•	Containment check: Confirm selection never picks a bit outside D^\*. With 0 forced into \mathcal C, the bottom path is always contained.
	•	Witness identity: Show a case where \sigma swaps 1\leftrightarrow 3 but test offers only 2; selected pixels with color 2 pass through unchanged (identity on 2). Receipts show \sigma.domain = \{1,3\}.

⸻

Answer to “pattern-based color inference”

You don’t need a new inference layer. Once \mathcal C includes \mathrm{colors}(X^), all pattern families that read X^ (lattice/ICL-Conv/morph/logic-guard) can place those colors exactly. Output-side families (transport/unanimity) constrain positions, not values, so they won’t “erase” new colors (they’re silent off-scope). The lattice fixed-point guarantees you only collapse to colors supported by the proofs you’ve emitted.

⸻

TL;DR
	•	Yes—add \mathrm{colors}(X^*) to \mathcal C.
	•	Keep \sigma partial identity.
	•	Default forbids permissive (emit only what you can prove).
	•	Nothing else is required; your existing engines/witness + admits + fixed-point will handle test-only colors deterministically.
