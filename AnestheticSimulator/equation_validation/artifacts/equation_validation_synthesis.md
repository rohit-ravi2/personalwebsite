# Equation validation — three-path synthesis

**Date:** 2026-04-28 (Wave P / Session 2 / equation-derived integration)
**Status:** Synthesis across Paths A + B + C (CP D.1)

---

## How Paths A, B, C complement each other

The three paths together form a complete equation-derived validation methodology that complements primary-source-grounded empirical validation:

**Path A — equation-derived sanity checks** (Nernst, GHK, power balance, cable equation): tests whether the *implementation* is self-consistent with the canonical equations the implementation rests on. Catches parameter errors, units mistakes, and edge-case behaviors. Cheap to run; reusable as production CI gates.

**Path B — dynamical systems analysis** (phase planes, bifurcation, H-H universality): tests whether the *dynamics* emerge as the canonical formalism predicts. Distinguishes "matches empirical traces" from "matches the mathematical structure of the framework underlying the biology." Strengthens the production-grade verdict for cells that pass both empirical AND dynamical validation.

**Path C — CeNGEN-equation-coupling** (gene expression → channel densities → cell models): tests whether the canonical equations + transcriptomics can extend biophysical grounding past the literature cap. Produces falsifiable equation-derived predictions for un-validated cells with explicit awaiting-empirical-validation labeling.

Together they're stronger than any single path:

```
Empirical validation (Nicoletti)        ← what the cell IS, measured
+ Path A (Nernst/GHK/power/cable)        ← parameter set self-consistent
+ Path B (phase plane/bifurcation/H-H)   ← dynamics emerge canonically
+ Path C (CeNGEN coupling)               ← reach extends past literature cap
═════════════════════════════════════════
= Production-grade verdict refinement    + Falsifiable predictions for ~270 cells
```

## What's now mathematically validated about the simulator

### Per-cell verdict refinement

| Cell | Empirical (Nicoletti) | Path A | Path B | Path C ground truth | Combined verdict |
|---|---|---|---|---|---|
| AVAL | ✓ | ✓ within physio | ✓ Mellem regime FP | calibration cell | **PRODUCTION GROUNDED** |
| AVAR | ✓ | ✓ within physio | ✓ Mellem regime FP | calibration cell | **PRODUCTION GROUNDED** |
| AIY | ✓ | ✓ | ✓ structurally; extrapolated parameters | calibration cell | **GROUNDED EMPIRICAL, EXTRAPOLATED EQUATION** |
| RIM | ✓ | ✓ | ✓ structurally; extrapolated parameters | calibration cell (LOO outlier) | **GROUNDED EMPIRICAL, EXTRAPOLATED EQUATION; LOO HOLD-OUT POOR** |

AVAL/AVAR are now the most rigorously grounded cells in the Wave 2 panel — primary-source-anchored on both empirical and equation-derived axes.

AIY/RIM are grounded empirically but with extrapolated parameters on the equation-derived side. RIM specifically is the LOO outlier in Path C calibration (mean |log10_err| 1.26 when held out), suggesting RIM's channel suite is genuinely distinct from AVA-class + AIY in ways that CP C.2's linear scaling doesn't capture.

### Equation-level findings

- **Mellem 2008 voltage regime confirmed at equation level:** AVAL/AVAR fixed-point voltages (-29 / -27 mV) are produced naturally by the GHK + phase-plane analysis. The cells aren't tuned to be depolarized; their channel suites place them there.
- **Single-compartment Nicoletti models defensible:** cable equation analysis shows λ >> r_soma for all 4 cells (3000+ μm for AVA-class, 200-415 μm for AIY/RIM). Multi-compartment treatment isn't required for the validated phenotypes.
- **Phase F refinement opportunity surfaced:** RIM has 4.5× higher steady-state ATP cost than AIY (CP A.3); Phase F's uniform K_BASE_CONSUMPTION underestimates differential metabolic vulnerability. Future Phase F refinement should use cell-class-specific consumption rates.

## What's now potentially extendable past literature

**Path C demonstrates that CeNGEN-equation-coupling is structurally viable but quantitatively marginal at v1.**

Linear scaling g_nS = α × TPM produces leave-one-out predictions within ~3.6× on average (mean |log10_err| 0.56). Adequate for order-of-magnitude predictions; not adequate for tight point estimates.

Equation-derived models produced for 3 representative un-validated cells (AVBL, PVCL, ASHL):

- All have biologically-reasonable channel suites consistent with their roles
- Predicted V_rest values fall in plausible physiological range (-45 to -66 mV)
- Indirect evidence (Atanas 2023 calcium imaging, behavioral genetics, connectome) qualitatively consistent

The methodology produces FALSIFIABLE PREDICTIONS for wet-lab follow-up, NOT validated models for production deployment. The labeling discipline matters.

### Recommended refinement path for production-grade Path C

1. Expand calibration cell panel (currently 4 cells; target 8-12 if remaining literature mined)
2. Switch from linear to Hill function scaling
3. Per-channel-class calibration (K vs Ca channels likely have different α scaling)
4. Expand CeNGEN gene panel for missing channels (IRK family, TWK channels)
5. Empirical validation on cells with partial indirect data (e.g., match AVB equation-derived dynamics to Atanas 2023 calcium traces)

## Methodology contribution

This work block adds **equation-derived validation** as a complement to primary-source-grounded empirical validation. The two are complementary:

- Empirical validation establishes "matches measured biology"
- Equation-derived validation establishes "consistent with canonical mathematical frameworks"

Cells that pass both are more rigorously grounded than cells that pass only one. The methodology pattern that's been load-bearing throughout the project (honest documentation > overclaiming; pause-with-documentation > push-through; primary-source verification when applicable) extends naturally to equation-derived work.

### Methodology catches surfaced during this run

1. **CP A.3 power-balance hand-written prose error:** initial print statement claimed "AVA-class cells are 30-100× higher cost than AIY/RIM." Computed numbers showed 4.5× spread with RIM (not AVA) at the high end. Caught at run time; corrected in `path_a_summary.md`. CSV + checkpoint JSON have correct numbers; only the interpretive prose was wrong.

2. **CP B.3 spike-detector false positive:** adaptive-threshold spike detection flagged "1 spike" per cell from initial-condition transient (V step from -55 mV to steady state). Real Wave 2 cells don't show repeated regenerative spiking under +200 pA — graded mode confirmed. Limitation documented; broader conclusion robust.

3. **CP C.3 ASE substitution:** ASE not in CeNGEN panel; ASHL substituted as polymodal-sensory analog. Documented explicitly so the ASHL prediction isn't conflated with an ASE-specific claim.

These are the methodology pattern continuing to work — surfacing errors in interpretation/scope rather than in computed numbers.

## Per Wave 2 cell production-grade verdict refinement

The 4 production cells shipping in Wave 2 with the rigor-tightened verdict structure:

- **AVAL — PRODUCTION GROUNDED** — empirical (Nicoletti) + Path A within physio range + Path B Mellem-regime fixed point + Path C calibration cell. All axes anchored to primary sources or directly validated.
- **AVAR — PRODUCTION GROUNDED** — same as AVAL.
- **AIY — GROUNDED EMPIRICAL, EXTRAPOLATED EQUATION** — Nicoletti recordings exist; Path B phase-plane parameters (V_half, k, τ for slow gate) are extrapolated from cell-builder validation per WB3 Decision 3 caveat. Sensitivity sweep on V_half ± 5 mV deferred but flagged.
- **RIM — GROUNDED EMPIRICAL, EXTRAPOLATED EQUATION; LOO HOLD-OUT POOR** — Nicoletti recordings exist; Path B parameters extrapolated; **Path C linear-scaling calibration generalizes poorly to RIM** (LOO |log10_err| 1.26). RIM's channel suite is distinct enough from AVA-class + AIY that the linear-scaling assumption breaks down. This is informative about RIM's biophysical regime, not a defect in the cell model.

## Recommended next-step trajectory

1. **AIY/RIM sensitivity sweep on V_half ± 5 mV** (Path B follow-up) — bounded analysis, ~1 hour. Tests whether extrapolated parameter dynamics are robust to parameter perturbation.

2. **Multi-slow-variable phase plane for AVAL/AVAR** (Path B follow-up) — bounded analysis, ~3 hours. Single-slow-variable misses Wicks-style hysteretic bistability; full multi-gate phase plane in (V, h_egl19, n_unc103) 3D may expose it.

3. **Phase F cell-class-specific K_BASE_CONSUMPTION** (Path A follow-up) — refines metabolic-vulnerability prediction. Bounded ~2 hours.

4. **Path C Hill-function calibration** — promote linear scaling to Hill if a 5-cell calibration set can be assembled. Bounded ~4-6 hours including refit + LOO + viability re-assessment.

5. **Path C empirical match against Atanas 2023 indirect data** for AVB — match equation-derived AVBL dynamics to observed calcium traces under controlled stimuli. Bounded ~3-4 hours.

These are bounded follow-up work blocks; none are in scope for this overnight run.

## Out-of-scope items (not addressed in this run, per prompt)

- Modifying production Wave 2 cell models (read-only reference)
- Modifying production simulator (LIFBrain, etc.)
- Phase G LIFBrain integration (separate work block, blocked on Session 1's WB3)
- Wave 2 expansion beyond 4 production cells (separate work)
- Methodology paper material (off-table per Rohit's earlier instruction)
- Modifications to wave2_overlay_v2.json or Phase G perturbation manager

## Cross-thread coordination

This work block touches Wave 2 cellular-layer territory (read-only) but writes new infrastructure under `AnestheticSimulator/equation_validation/`. No conflict with Session 1's Wave V/Wave 2 production code; the two threads are coordinated via clean boundary at `cell_params.py` (mirrors Session 1's parameter constants without mutating them).
