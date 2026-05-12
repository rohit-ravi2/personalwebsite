# Path 2 v1 cell validation — Phase 6 deliverable

**Status:** Phase 6 of §7.3.5 Path 2. Per-cell validation under §8.6
reframed criteria. **Outcome: 0/4 cells PASS.** Phase 6 failure routes to
Option β refinement (NOT Option α ship).

**Date:** 2026-05-12

**Reference:** `docs/path2_channel_validation.md` Phase 5 + `docs/layer1_design_decisions.md`
§8.6 reframed validation criteria.

---

## 1 · Validation framework (reframed per §8.6)

Per Rohit's 2026-05-12 Option α authorization with reframed validation
criteria. Path 2 cells are NOT validated against Nicoletti's specific
gbar values (those are non-unique fits per §8.6 uniqueness audit). They
are validated against:

- **Per-cell rest stability:** [K]_in stable within ±2% over 5s,
  [Na]_in stable, [Cl]_in physiological (3-7 mM), [Ca]_in near target,
  V_rest within published range per cell
- **Cell-level I-V envelope match** against Nicoletti's published data
  (with SEM where available) — channels' kinetic parameters preserved
  from Nicoletti; only gbar values change to Path 2 derived
- **Cross-cell consistency:** biological differentiation preserved

---

## 2 · Per-cell results

```
cell    V_rest    Δ[K]      Δ[Na]      Δ[Cl]      Δ[Ca]            verdict
AVAL    -3.5 mV   -80.0%    +381.5%    +38.5%     +1,212,473%      FAIL
AVAR    -4.9 mV   -88.0%    +706.6%    +54.4%     +1,314,764%      FAIL
AIY     +6.5 mV   -97.8%    +1,598.3%  +9.8%      +126,330%        FAIL
RIM     +7.4 mV   -97.7%    +728.2%    +85.9%     +1,063,557%      FAIL
```

**All four cells fail catastrophically.** V depolarizes to near 0 mV in
all cells; [K]_in crashes 80-98%; [Na]_in rises 4-17×; [Ca]_in rises
1,000-13,000× into millimolar range. Rest homeostasis lost completely.

---

## 3 · Failure pattern diagnosis

The catastrophic failure across all four cells is more severe than §7.3
showed with Nicoletti gbars (which failed but with V_rest near published
range and [K]_in within ~20%, not 80-98%). Path 2 v1 deployment makes
cell behavior WORSE than Nicoletti's parameterization, not better.

### 3.1 Dominant cause: Path 2 deploys channels Nicoletti omitted

**Critical observation:** Wave 2 AVAL cell builder has `g_nca = 0`
explicitly in Nicoletti's parameterization. CeNGEN T2 says nca-2 = 153.2
TPM in AVA → Path 2 derives `gbar_NCA_AVAL = 1.33e-5 S/cm²` (non-zero).

Path 2 v1 thus includes a substantial NCA channel that Nicoletti's fit
converged with set to zero. NCA is a non-specific cation channel (`e_NCA
= +30 mV`) — at V_rest = -50 mV, driving force is -80 mV → strong inward
Na current.

**NCA in Path 2 AVAL contributes ~1.06e-3 mA/cm² inward Na current** at
the initial V_rest. This is much larger than other ion currents and
drives massive depolarization.

Same pattern applies in AVAR (Nicoletti gbar_NCA = 4.4e-6; Path 2 gbar =
1.33e-5 — also higher than Nicoletti, plus the broader IRK over-channeling).

Per the §8.6 uniqueness audit reframing: Nicoletti's choice to set NCA
= 0 in AVAL was one possible non-unique fit (his optimization converged
to a parameter set without NCA contribution). Path 2 says CeNGEN expression
data justifies including NCA. Without independent evidence, both views
are defensible.

**Path 2 v1 deployment exposes this as a load-bearing question:** what
should the substrate do when CeNGEN says a channel is expressed (TPM > 0
at threshold 2) but Nicoletti's degenerate fit omitted it?

### 3.2 Three possible refinement interpretations

**Interpretation A: Nicoletti's gbar=0 omissions are silent channels.**
Gene is expressed at mRNA level but functionally silenced (post-
translational modification, trafficking, regulation). Refinement:
zero out E_translation specifically for channels Nicoletti omitted
→ effectively per-channel E_translation that respects Nicoletti's
inclusion decisions.

**Risk:** Defeats the biology-derived methodology contribution by
deferring to Nicoletti's choices on which channels to include.

**Interpretation B: Path 2 is correct, but γ values are wrong.**
NCA's literature-gap γ = 5 pS placeholder may be wildly overestimated.
Refinement: refit γ_NCA much smaller (e.g., 0.5 pS) → derived NCA
gbar drops 10× → less aggressive Na influx → more compatible with
Nicoletti-like rest behavior.

**Risk:** Refitting γ to make cells work feels like inverse-engineering
the answer (Nicoletti gbar → "true" γ that produces it).

**Interpretation C: Per-cell-family C_global needed**
(canonical Option β v2 refinement). Refinement: separate C_global for
AVA-class vs AIY-class vs RIM-class cells.

**Risk:** Adds 3 free parameters (one per cell-family) while only
mitigating one of the two patterns (small-cell under-channeling); the
NCA-in-AVAL issue is orthogonal to cell size.

### 3.3 Secondary failure pattern: AIY/RIM small-cell under-channeling (Phase 5 Pattern A)

Even setting aside NCA, AIY and RIM remain catastrophically under-channeled
relative to Nicoletti's per-cell-fit values. AIY has all 4 channels deriving
gbar << Nicoletti by 50-1000×; RIM similarly under by 6-450×. With these
small derived gbars, the per-cm² K efflux is too low to balance Na influx
(through whatever Na conductance exists) and V depolarizes.

This is the small-cell-systematic pattern documented in Phase 5 §4. v2
candidate: per-cell-family C_global (Interpretation C).

### 3.4 Tertiary failure pattern: IRK over-channeling in AVAL/AVAR (Phase 5 Pattern B)

In AVAL/AVAR, derived IRK gbar is 8-19× HIGHER than Nicoletti. This
should produce MORE K efflux and HYPERPOLARIZATION, not depolarization.
The NCA-included issue dominates over IRK over-channeling such that
AVAL/AVAR still depolarize.

---

## 4 · Option β refinement routing

Phase 6 outcome triggers Option β per Rohit's framework. Phase 5 already
flagged: "if Phase 6 reveals load-bearing issues → deploy Option β
refinement." Phase 6 confirms load-bearing issues across all four cells.

### 4.1 Targeted Option β candidates (in increasing scope)

**β-1 (smallest scope): per-channel E_translation respecting Nicoletti omissions.**

For channels Nicoletti explicitly set to gbar = 0 in a cell's
parameterization (e.g., NCA in AVAL), Path 2 honors that as
`E_translation[channel][cell] = 0` regardless of TPM. This represents
Interpretation A.

Cost: 1 free parameter per (channel, cell) combination where Nicoletti
omitted. For Wave 2 cells, this is only AVAL NCA (Nicoletti g=0).

Methodology cost: Deviates from "all channels Path 2 derives based on
biology." Becomes "Path 2 derives gbar magnitude per CeNGEN but
inclusion/exclusion per Nicoletti's fit choices."

**β-2 (medium scope): refit γ_NCA + γ_IRK.**

γ_NCA = 5 pS in Phase 2 is a placeholder for the documented literature
gap. Refit to ~0.5-1 pS (much smaller). γ_IRK = 25 pS chord conductance
may overestimate physiological per-channel current; refit to ~10 pS
slope or ~15 pS average.

Cost: 2 γ refits + re-derivation + re-validation. Bounded.

Methodology cost: Maintains "all channels from CeNGEN" but adjusts
per-channel intrinsic γ. Empirically valid if other Cav2/Cav3/Kv γ values
in the inventory can be cross-checked for consistency.

**β-3 (full scope): per-cell-family C_global.**

C_global_AVA = 1.73e4 (current); C_global_AIY = ~5e5 (~30× larger);
C_global_RIM = ~2e5 (~10× larger). Calibrated against Nicoletti's
per-cell gbars for one anchor channel per cell.

Cost: 3 calibration anchors (one per cell family) instead of 1.

Methodology cost: Larger; loses the "single global constant" contribution
of v1. But cell-family differences are biologically motivated (different
membrane composition, different protein-density-per-mRNA ratios).

### 4.2 Recommended order

Deploy **β-2 first** (smallest methodology-cost refinement that targets
the dominant failure mechanism):

1. Refit γ_NCA to ~0.5 pS (NALCN literature gap; conservative estimate
   reflecting NALCN's known low conductance)
2. Refit γ_IRK to ~12 pS (slope conductance midpoint)
3. Re-derive all gbars
4. Re-run Phase 5 (channel-level) + Phase 6 (cell-level)

If β-2 fixes AVAL/AVAR (NCA + IRK refits address Pattern B + the
Nicoletti-omission issue) but AIY/RIM still fail:
- Deploy **β-3** (per-cell-family C_global) for small-cell systematic

If β-2 + β-3 still fails:
- Consider **β-1** (per-channel E_translation honoring Nicoletti omissions)
- Or **Option γ** (Path 1 fallback — refit Nicoletti under physiological Nernst)

---

## 5 · Cross-cutting methodology observation

Phase 6 surfaces a **third standing methodology lesson** complementing
the state-variable audit (§7.3) and uniqueness audit (§7.3.5 Phase 5):

**Channel-inclusion audit (Phase 6 finding):** Inherited fits encode
not just parameter values but also IMPLICIT CHANNEL INCLUSION/EXCLUSION
decisions. When Nicoletti sets `gbar = 0` for a channel, that's an
inclusion decision: "this channel is functionally negligible in this
cell." Path 2 derivation overrides this with TPM-based inclusion.
**Whether to honor Nicoletti's inclusion decisions or override them
based on gene expression is a methodology decision that needs explicit
authorization.**

For Layer 2-7 forward-looking application: any inherited model that
sets parameters to zero is making an inclusion decision; the receiving
substrate methodology must decide whether to honor or override.

---

## 6 · Phase 6 acceptance criteria status

Per methodology / roadmap:

- [x] All four cells validated under §8.6 reframed criteria
- [x] No cells pass reframed criteria (0/4)
- [x] Failure patterns diagnosed and routed to Option β
- [ ] **Option α ship: BLOCKED** (no cells pass)
- [ ] **Phase 7 commit: BLOCKED** pending Option β outcome

**Phase 6 SHIPPED with comprehensive failure diagnosis.** Routes to
Option β refinement; specific refinement choice (β-1 / β-2 / β-3 /
combination) requires Rohit's direction.

---

## 7 · Files of record

- This document: `docs/path2_cell_validation.md`
- Phase 6 validation script: `scripts/brain/wave2/validate_path2_cells.py`
- Phase 5 channel validation: `docs/path2_channel_validation.md`
- Phase 5 HARD_STOP record: `scripts/brain/wave2/artifacts/HARD_STOP_path2_phase5.md`
- Methodology reference: `docs/channel_parameter_derivation_methodology.md`
- Phase 6 checkpoint: `scripts/brain/wave2/artifacts/path2_phase6_checkpoint.json`
- Progress tracking: `scripts/brain/wave2/artifacts/path2_progress_summary.md`
