# P1_P2 keystone — minimal-delta-V2 rank-lift: cell-type targeting NOT established (worm)

**Date:** 2026-06-12 · **Verdict: worm Match#3 = NULL_DEFLATE** (conserved cell-type spread at the 46th
percentile — unremarkable). The rank lift is real and correct; it does not surface conserved-target
cell-type specificity on worm. Prereg `audits/phase2/P1_P2/prereg.json` + `AMENDMENT_2026-06-12.md`.

## What the keystone is

WF1/WF2 identified the **minimal-delta-V2 rank-lift** as the single substrate change that lifts the
cell-type-targeting impossibility: replace the rank-2 operator's all-ones broadcast with CeNGEN per-class
expression vectors `x_c` (7×300), raising operator rank 2 → min(7,N) so that *which cell types* each mechanism
class targets becomes a real degree of freedom — making the Match#3 (cell-type-spread) test, undecidable on V1,
finally testable.

## It was built correctly (non-destructive, verified)

- New operator path; frozen `apply_anesthetic` untouched. **G_BIT_IDENTITY PASS** — bit-identical to V1 when
  x_c=ones (max per-neuron error 3.55e-15 pA across 36 cases).
- **G0_RANK_LIFT_REALIZED PASS** — two profiles matched on (total_pa, snare_factor) now produce per-neuron drive
  differing by L2 = 31 pA (17× the V1 value), i.e. the rank is genuinely lifted.
- `x_c` is real CeNGEN expression (complex_i/ii all-ones by biology as prereg'd; the other 5 classes carry
  genuine ~0.22–0.33 per-neuron expression).
- **SOL7 able-to-fail screen PASS** and earned its place: it caught that the prereg'd η² statistic is degenerate
  (x_c rows class-mean-constant ⇒ η²≡1) and fell back to the participation ratio PR (which G1 already used). A
  real statistic correction surfaced by the screen, not a silent loosening.

## The result: NULL

With the rank lifted and the spread statistic frozen first, the conserved halothane profile's cell-type spread
(PR = 0.875) sits at the **46th percentile** of magnitude+SNARE-matched random surrogates → **G1_MATCH3_SPATIAL_SPECIAL
= NULL_DEFLATE** (the (1%,10%] PASS band was not entered; >10% is the NULL/deflate zone). G2_MATCH3_NOT_ENTAILED
PASS confirms the surrogates do vary (Var 2.5e-3), so the null is a real test, not a pinned constant.

**Reading:** lifting the operator rank did **not** reveal hidden conserved-target cell-type specificity on worm.
The conserved targets do not spread across cell types in a way distinguishable from a magnitude+SNARE-matched
random target-set. The keystone test that was supposed to be able to rescue the target-specificity claim returns
a null.

## Scope limit (honest, not a result)

- **Fly cannot be tested** — there is no Drosophila cell-type-expression atlas (CeNGEN is C.-elegans-only); the
  workflow's fly arm was a fabrication (worm relabeled) and has been deleted + guarded. Testing fly Match#3
  requires acquiring/building a fly expression atlas (new data, out of scope). Fly was the a-priori payoff
  organism, so this is a genuine open gap, not a closed negative.
- **Mouse excluded** — generic random graph has no cell types.

## Cumulative implication (with P8, P3)

Three independent lines now converge against the strong "specific conserved targets are load-bearing" claim:
- **P8** — the conserved target set is only weakly special even on the two magnitude coordinates (worm/fly
  26–28%, mouse median).
- **P3** — the mouse-at-median is a derived mean-field consequence.
- **P1_P2** — lifting the operator rank to make cell-type targeting testable yields a NULL on worm.

The honest standing claim is unchanged from the prior session synthesis and is now better-evidenced: these are
biologically-motivated targets that anesthetics demonstrably bind (Wave-P) and whose multi-class quorum
reproduces MAC, but the project's own substrate — even after the rank-lift built specifically to test it — does
not establish that the *specific cell-type-resolved identities* are load-bearing on worm. Fly remains the one
untested payoff, gated on data the project does not yet have.
