# Phase G — standing followups (deferred work blocks)

**Date:** 2026-05-12
**Status:** Documentation only. No work-block deployment authorized; these
are deliberate-planning items awaiting Rohit's review.

---

## 1 · Phase G v2 architecture re-spec — next major work block

The Phase G LIFBrain integration work block (commits `32fb25b` + `da679dc`)
surfaced 3 implementation gaps in Phase G v1's perturbation manager:

1. **ModulationLayer overwrites I_ext** every step → complex_i_block +
   k2p_potentiation broken on production substrate.
2. **NT string equality fails** ('ACh' != 'Acetylcholine (ACh)') → nAChR
   antagonism silently no-ops on all 159 cholinergic neurons.
3. **Missing hook implementations** for glucl_potentiation,
   complex_ii_block, nca_block — defined in architecture doc but never
   landed in apply_to_brain code.

Per the CP2 hard-stop diagnosis at `CALIBRATION_GAP.md`, fixing in-place
is achievable in ~2-3 hours code + 35 min compute. But the
implementation gap reflects a methodology gap: Phase G v1 shipped without
substrate-realistic testing, because the demo network's aggregate-I_ext
bypass masked all 3 issues silently.

**Phase G v2 re-spec scope (proposed):**
- Compose-point architecture: confirm `brain.ablation_current_pA` is the
  right channel for hyperpolarizing perturbations (additive into modulation
  layer's I_total), or design a separate Phase-G-specific channel.
- NT-matching protocol: substring vs canonical lookup table vs LIFBrain
  API change to expose presynaptic NT class consistently.
- Biological-judgment input for the 3 missing hooks:
  - `complex_ii_block` — does it parallel `complex_i_block` in K-ATP
    consequence (Phase F coupling), or have distinct downstream
    physiology that requires a different substrate analog?
  - `nca_block` — NCA-1 is a sodium leak channel; LIFBrain has no explicit
    leak channels. What's the right substrate proxy: ablation_current_pA
    additive (treats NCA loss like depolarization removal), or W_syn
    multiplicative (treats NCA loss like reduced effective drive)?
  - `glucl_potentiation` — symmetric to gaba_potentiation but operates
    on Glu→GluCl-expressing edges. Needs presynaptic-Glu identification +
    postsynaptic GluCl-receptor channel expression mapping.
- W_chem-to-Brian2 sync architecture: confirm the current approach
  (mutate `_W_chem_runtime` then sync to syn_exc.w / syn_inh.w via
  helper) is production-grade for repeated perturbations, vs bake
  perturbation into LIFBrain construction with a `perturbation_profile`
  kwarg.
- Substrate-realistic testing protocol: require that every Phase G v2
  hook is validated on production LIFBrain with measurable behavioral
  effect before declaring the hook "shipped." Don't repeat the v1 mistake
  of accepting demo-network apparent dose-response as evidence of
  correctness.

**Scoping:** deliberate-planning work block. Estimated ~3-5 hours
discussion + ~6-8 hours implementation + ~2 hours validation. Not
autonomous execution — needs Rohit's biological-judgment input on the 3
missing hooks.

## 2 · Bottom-up substrate redesign under consideration

Per project trajectory discussion (Category 1 + Category 2 from earlier
session): the Phase G CP2 calibration failure raises a deeper substrate
question. The horizontal rebase (Phase 0-3) locked M2-pure as the
brain-side sign mode and proved the recalibrated stack works for direct
neural readouts (cascade firing, AVA-ablation dFWD signal). But:

- **Category 1 — substrate scale:** is the 300-neuron LIFBrain the right
  substrate for anesthetic dose-response, or does the production substrate
  need cellular-level granularity (Wave 2 channel detail) for the binding
  pipeline's mechanism-class engagement to map onto behavioral suppression
  faithfully?
- **Category 2 — substrate biology:** does the substrate need explicit
  K-ATP channels, leak channels (NCA), and GABA/GluCl conductance-level
  detail (vs the current sign-based binary representation) to be a
  legitimate Phase G consumer?

The Phase G v2 re-spec (item 1 above) is the bounded fix-in-place path.
Bottom-up substrate redesign is the alternative path that addresses both
Category 1 and Category 2 simultaneously. Either is defensible; the
choice depends on:
- Whether the rock-solid mechanism contribution (Wave P) genuinely
  requires sub-cellular substrate detail or can ship on LIFBrain.
- Whether the resource cost of substrate redesign is justified by the
  ablation-experiment-deployment value that Phase G unlocks.

**Recommended timing:** discuss before Phase G v2 implementation. The
choice between v2 re-spec vs bottom-up redesign affects what gets built
next.

## 3 · AVA → dFWD literature precedent check (task #15)

Deferred from horizontal-rebase Phase 3 doc-fix sweep
(commit `ce9c7c9`). The Phase 2.5 default tier surfaced a new direct
phenotype finding: AVA-ablation produces a robust dFWD signal under
M2-pure (Cohen's d ≈ 0.93, 7/10 negative seeds at n=10×60s). Forward-
locomotion suppression rather than the canonical Chalfie 1985 reversal
abolition.

Catalog (`docs/state_of_claims_2026-05-02.md`) classifies this as
**Direct — alternative phenotype finding** with cautious framing pending
literature precedent verification.

Bounded research-block items:
- Wang/Liu/Chen 2020 *Nat Commun* 11:5076 — does the AVA-AVB
  gap-junction-coupling work report behavioral consequences of AVA
  ablation on forward locomotion?
- Gao 2015 *Nat Commun* 6:6323 (NCA-1 + AVA persistent activity) — any
  forward-locomotion phenotype quantified?
- Pirri / Alkema 2009 (tyraminergic gating) — adjacent literature on
  AVA-AVB recurrent control.
- Eshel Ben-Jacob / Connor Mooney / other C. elegans simulators — prior
  AVA-ablation forward-suppression results?

Outcome decides the project page framing: precedent exists → strengthens
to validation claim; no precedent → cautious novel-finding framing stays
+ headline-result decision waits for robustness checks.

## 4 · Cross-thread integration validation deferred

The ablation harness (commit `b27dcf2`, May 2026) consumes Phase G output
as substrate-agnostic. CP4 of the LIFBrain integration work block was
designed to verify the harness API contract holds when Phase G consumes
the production substrate. CP4 is blocked behind CP2 resolution.

Once Phase G v2 lands (or bottom-up substrate redesign provides a new
Phase G substrate), the harness consumption verification becomes a
~1-2 hour work block:
- Halothane × UNC-49 ablation on production substrate (3 seeds × n=2
  conditions = 6 runs ~30 min)
- Verify Phase G output fields match harness expectations
- Document production-substrate baseline as reference for full-scale
  ablation experiments

No deployment until Phase G substrate question resolves.

---

## Cross-references

- Phase G LIFBrain integration commits: `32fb25b` (substrate factory +
  sync helper) + `da679dc` (CP2 calibration gap diagnosis)
- Horizontal rebase deliverables: `docs/state_of_claims_2026-05-02.md`,
  `docs/brain_v3.5_locked.md`, `docs/phase2_preflight.md`
- Phase G v1 architecture: `artifacts/phase_g/phase_g_architecture.md`
- CP2 hard stop diagnosis: `artifacts/phase_g/CALIBRATION_GAP.md`
- Work block summary: `artifacts/phase_g/phase_g_lifbrain_integration_summary.md`
- Pending tasks (#15 dFWD literature, #17 CP4 harness, #19 CP3 cross-anesthetic,
  #20 CP5 doc-commit) all blocked behind Phase G v2 architecture decision.

---

*No work-block deployment authorized in this commit. Standing followups
recorded for deliberate-planning when ready.*
