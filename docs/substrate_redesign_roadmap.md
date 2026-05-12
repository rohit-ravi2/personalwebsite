# Substrate redesign — Layers 1-7 roadmap

**Status:** Layer 1 in progress (§7.1 foundation work block, 2026-05-12).
Layers 2-7 are placeholders that get refined as their work blocks approach.

**Date:** 2026-05-12 (initialized)

**Motivation:** Phase G LIFBrain CP2 calibration hard-stopped on 2026-05-12
with 0% behavioral suppression across 5 orders of dose. Root cause:
substrate adequacy — LIFBrain represents synapses as signed scalar weights
with no GluCl conductance, K-ATP channel, or NCA leak channel for halothane
/ sevoflurane / halogen mechanism classes to engage with. The fix-in-place
Phase G v2 path would ship hooks that proxy biology not present in the
substrate. Bottom-up substrate redesign rebuilds the production substrate
from ion-channel-level biophysics up, so Phase G perturbations engage real
conductances.

**Total budget:** ~30-44 work blocks across Layers 1-7.

**Cross-cutting principle (per Layer 1 §2.8):** Each parameter ships with
an explicit epistemic label (empirically grounded / biophysically derived /
approximation from adjacent biology / free parameter with sensitivity
sweep). The substrate is biophysically grounded in *structure* with
explicit uncertainty quantification in *parameters*. Refinement happens
when empirical data lands; the substrate ships rather than blocking on
absent measurements.

---

## Layer 1 — Ion concentration state + dynamic Nernst + pumps

**Scope:** Per-cell `[K]_in`, `[Na]_in`, `[Cl]_in`, `[Ca]_in` as Brian2
state variables on the 4 production-grade Wave 2 cells (AVAL, AVAR, AIY,
RIM). Fixed extracellular reservoir. Single whole-cell compartment.
Na/K-ATPase (Hill-ATP) + lumped Ca extrusion (PMCA+NCX+SERCA) + KCC-2 +
ABTS-1 (lumped electroneutral). Dynamic Nernst per Brian2 dt. Pump
current → Phase F's restructured metabolic balance.

**Design doc:** `docs/layer1_design_decisions.md` (resolved 2026-05-12)

**Work blocks:**
- §7.1 — Foundation: `scripts/brain/wave2/ion_dynamics.py` + geometry +
  Nernst/GHK helpers + Brian2 equation fragment. **SHIPPED 2026-05-12.**
- §7.2 — Pump module: Hill-ATP Na/K + Payne KCC-2 + approximate ABTS-1 +
  threshold-MM Ca clearance. **SHIPPED 2026-05-12 (v2 after Path-b
  thermodynamic-form upgrade).** AVAL anchor passes acceptance; AIY/RIM
  drift surfaced as TPM-scaling finding under placeholder leaks.
- §7.3 — Per-cell integration: AVAL → AVAR → RIM → AIY. **SHIPPED
  2026-05-12 with structural finding (see §7.3.5 entry below).**
  Infrastructure landed; acceptance criteria unmet pending channel refit.
- **§7.3.5 — Channel-Substrate Consistency Audit and Refit (NEW, BLOCKS §7.4).**
  See dedicated entry below.
- §7.4 — Phase F restructure: `phase_f_analytic.py` preserved +
  `phase_f_layer1.py` as Layer 1 consumer. **BLOCKED by §7.3.5.**
- §7.5 — Validation gauntlet + r_eff sensitivity sweep {0.25, 0.5, 1.0 μm}.
- §7.6 — Commit + roadmap doc population.

**Estimate:** 10-13 work blocks through commit (revised from 6-8; the
inserted §7.3.5 adds 3-5 blocks for channel audit + refit + validation).

**Status:** §7.1 + §7.2 SHIPPED. §7.3 SHIPPED with structural finding.
§7.3.5 not yet started (pre-flight scoping pending). §7.4 blocked.

**Acceptance criteria (per design doc §5):**
1. Per-cell rest stability over 60s (|Δ[X]| within tolerance).
2. Nernst-bound voltage: `min(E_K, E_Cl) ≤ V ≤ max(E_Na, E_Ca)`.
3. GHK rest prediction within ±5 mV of measured V_rest.
4. Voltage-clamp + current-clamp envelopes within Nicoletti SEM tolerance.
5. ATP balance: pump consumption rate matches production at rest.
6. Recovery from perturbation on biological timescales.
7. Cl perturbation dress rehearsal (Phase G dry run): KCC-2+ABTS-1
   lumped extruders return [Cl]_in to rest after GluCl conductance
   pulse.

---

## §7.3.5 (Layer 1.5) — Channel-Substrate Consistency Audit and Refit

**Status:** Inserted 2026-05-12 from §7.3 finding. Not yet scoped (deliberate
pre-flight required). BLOCKS §7.4 and downstream work.

**Motivation:** §7.3 integration surfaced that Nicoletti 2024's channel
parameterization assumes fixed `E_Ca = 60 mV`, which under physiological
`[Ca]_out = 2 mM` requires `[Ca]_in ≈ 17 μM` — 340× the mammalian-default
50 nM. Under Layer 1's physiological `[Ca]_in = 50 nM` (E_Ca = 134 mV), the
inherited gbar values produce ~70% larger Ca driving force than Nicoletti's
fit assumed, causing runaway Ca accumulation in all four cells.

Full finding documentation: `docs/layer1_design_decisions.md` §8,
`scripts/brain/wave2/artifacts/layer1_7_3_findings.md`.

**Scope (anticipated):**

1. **Methodology document.** Codify the "parameter audit before integration"
   methodology surfaced in design doc §2.8 + §8. Include audit checklist
   (state variables assumed, reference E_X values, fitting target). This
   becomes a transferable pattern for Layers 2-7.

2. **Nicoletti audit.** Per-channel inventory of fitted gbar values + the
   E_X assumed in their fit. For each Ca channel (EGL-19, CCA-1, UNC-2):
   document the original Nicoletti fitting target (published I-V curves) +
   the assumed E_Ca. Audit similarly for K channels (assumed E_K) where
   relevant.

3. **Channel re-fit under physiological Nernst.** For each Ca channel,
   refit gbar to match Nicoletti's published voltage-clamp envelopes
   under physiological E_Ca = 134 mV at rest. Expected outcome: gbar
   values drop by factor ~0.45 (60/134 driving-force ratio); precise value
   depends on per-channel kinetics. K channels likely need smaller
   correction (E_K = −90 vs assumed −80 → ~10% driving-force change at
   typical V_rest).

4. **Channel re-validation.** Each refit channel re-validated against
   Nicoletti's published I-V envelopes (within SEM, per §2.2 acceptance
   criteria). The refit cells should match Nicoletti's published phenotypes
   AND maintain stable rest under Layer 1's physiological substrate.

5. **Per-cell re-integration.** Re-run §7.3 with refit channels. Acceptance
   criteria from §7.3 should now pass.

6. **Forward-looking flags.** Document the parameter-audit step for other
   inherited parameter sets known to be at risk (Wicks 1996 graded release,
   Nicoletti Ca pool dynamics, peptide release constants — see design doc
   §8.5).

**Acceptance criteria:**
- Refit channels reproduce Nicoletti's published voltage-clamp I-V curves
  within SEM tolerance.
- Each Layer 1 cell (AVAL, AVAR, AIY, RIM) achieves stable rest at
  physiological [Ca]_in (50-200 nM steady-state range) with ±2% K stability.
- V_rest in published range per cell (Mellem 2008 for AVA; Nicoletti 2024
  envelopes for AIY/RIM).
- Methodology document complete and reviewable.

**Dependency chain:** §7.3.5 BLOCKS §7.4 (Phase F restructure depends on
correct Ca dynamics feeding ATP consumption via Ca-ATPase). §7.3.5 also
informs Layer 2 (channel-rewire-to-dynamic-Nernst), which would otherwise
inherit the same inconsistency.

**Estimate:** 3-5 work blocks. Substantial — channel re-fitting + per-channel
re-validation is the bulk; methodology document is bounded; per-cell
re-integration is straightforward once channels refit.

**Pre-flight scoping required before deployment:**
- Define refit objective function precisely (match peak I, steady-state I,
  inactivation timescale, or a weighted combination?)
- Decide refit method (manual gradient + visual match against published
  figures, or systematic least-squares against digitized I-V points?)
- Decide whether to refit K channels or accept their smaller E_K shift
- Decide validation tolerance (per-feature SEM, or relative L2 error against
  digitized traces?)

Pre-flight design discussion is its own work block before implementation.

---

## Layer 2 — Existing Wave 2 channels rewired to dynamic Nernst (PLACEHOLDER)

**Scope (anticipated):** The 14 NMODL-translated Brian2 channel modules
currently use fixed reversal potentials (e.g. `eca=60 mV` constant for
EGL-19, `ek=-80 mV` constant for IRK). Layer 2 rewires them to read
Layer 1's `E_K_mV`, `E_Ca_mV`, etc. as dynamic variables.

**Affected modules (anticipated):** `channels/{egl19,irk,nca,unc2,cca1,
egl2,kqt1,shl1,kvs1,unc103,slo2,...}.py`.

**Key open question:** Do any channels need re-fitting under dynamic
Nernst, or does the constant-Nernst → dynamic-Nernst swap leave channel
kinetics intact? Likely intact for kinetics (m∞, τ_m functions are
voltage-dependent, not Nernst-dependent) but resting balance may shift.

**Estimate:** 3-5 work blocks.

**Status:** Not started. Refined when Layer 1 §7.5 validation lands.

---

## Layer 3 — New channel classes for Phase G mechanisms (PLACEHOLDER)

**Scope (anticipated):** Channels Phase G perturbations need to engage
that don't exist in Nicoletti / Wave 2:
- **K-ATP channel** (gain in halothane / sevoflurane via Phase F ATP
  drop). Currently a phenomenological coupling in Phase F analytic.
  Layer 3 makes it explicit at the cell membrane.
- **NCA-1 leak** (Na leak for halothane block). Wave 2 has NCA with
  gbar=0 in AVAL; Layer 3 sets explicit conductance + makes it block-
  able.
- **GluCl explicit conductance**: GLC-3, GLC-4, AVR-14 currently represented
  by edge sign in LIFBrain. Layer 3 makes them explicit Cl conductances
  on the receiving cells.
- **GABA-A explicit conductance**: UNC-49 currently represented by edge
  sign. Layer 3 makes it explicit Cl conductance.
- **K2P channels** (background K leak, including UNC-58's Na-permeable
  variant per Wojtovich 2024 PNAS).

**Estimate:** 4-6 work blocks. Largest layer in the redesign.

**Status:** Not started. Mechanism class scoping depends on Phase G v3
target mechanism list.

---

## Layer 4 — Compartmentalization (CONDITIONAL — PLACEHOLDER)

**Scope (anticipated):** Soma + dendrite + axon compartments for cells
where single-compartment fails validation. May not be needed if Layer 1-3
single-compartment cells pass validation. Pre-spec: AVAL is process-
dominated; AIY has spatially restricted ER-released Ca (Layer 1 lumps
this away). If validation surfaces residuals, Layer 4 lands.

**Gate decision:** Defer until Layer 1 §7.5 validation result is known.
Possible alternative: Layer 4 ships as "AVAL multi-compartment only" if
that's the only cell that fails single-compartment.

**Estimate:** 4-6 work blocks if needed; 0 if single-compartment
validation passes for all four cells.

**Status:** Not started. Conditional on Layer 1-3 validation.

---

## Layer 5 — Network integration (PLACEHOLDER)

**Scope (anticipated):** Substitute Layer 1-3 (and optionally Layer 4)
cells back into the LIFBrain network via Wave2HybridBrain. The four
production-grade cells become biophysical; the other 296 cells remain
LIF. Cross-coupling: presynaptic LIF spikes drive postsynaptic
graded Boltzmann release into the biophysical cells (WB3 release rule);
postsynaptic biophysical-cell σ-magnitude readout drives downstream LIF
cells.

**Open question:** WB3 capacitance + release-event rule pinned during
Layer 1 (substrate redesign may invalidate previous WB3 calibration).

**Estimate:** 3-5 work blocks.

**Status:** Not started. Inherits from `wave2/integration/wave2_hybrid_brain.py`.

---

## Layer 6 — Phase G v3 rebuilt against real substrate (PLACEHOLDER)

**Scope (anticipated):** Phase G perturbations engage real conductances
on Layer 1-5 cells. Each mechanism class hooks into the substrate
mechanism it names:
- `gaba_potentiation` → UNC-49 explicit conductance scaling
- `glucl_potentiation` → GLC-3 / GLC-4 / AVR-14 explicit conductance scaling
- `nachr_antagonism` → ACR / UNC-29 / UNC-38 conductance scaling
- `complex_i_block` → Phase F Complex I rate → ATP → K-ATP coupling
  (now load-bearing on Layer 3's explicit K-ATP channel)
- `complex_ii_block` → Phase F Complex II rate
- `k2p_potentiation` → K2P conductance scaling (Layer 3)
- `snare_cooperativity` → release rule modulation (Layer 5 cross-coupling)
- `nca_block` → NCA-1 conductance scaling (Layer 3)

Each hook is **substrate-realistic-tested** before declared shipped (Phase
G v1's methodology gap was that the demo network's aggregate-I_ext bypass
masked named-hook failures silently — Phase G v3 must validate every hook
against the production substrate with measurable behavioral effect).

**Estimate:** 3-5 work blocks. Replaces Phase G v2 fix-in-place path.

**Status:** Not started. Inherits Phase G architecture but rewrites
apply_to_brain against Layer 1-5 substrate.

---

## Layer 7 — Validation + closed-loop integration (PLACEHOLDER)

**Scope (anticipated):**
- Cross-anesthetic verification (halothane, sevoflurane, isoflurane,
  desflurane, propofol, etomidate, ketamine, dexmedetomidine,
  α-chloralose) on Layer 1-6 substrate.
- Ablation harness CP4 verification against Layer 6 Phase G v3 output.
- Closed-loop env integration: substitute Layer 1-6 cells into
  `ClosedLoopEnv` and validate behavioral outputs match published
  C. elegans anesthesia phenotypes (immobilization, recovery, gas-1
  hypersensitivity).
- Public-facing documentation update: anesthesia-pipeline web page
  reflects Phase F v2 + Layer 1-7 substrate.

**Estimate:** 3-5 work blocks.

**Status:** Not started.

---

## Cross-cutting tracks

These run in parallel with Layer work blocks and don't gate any single
layer:

- **CeNGEN panel extension.** `public/data/cengen-panel.json` currently
  contains 41 channel/receptor genes. Layer 1 v2 broadcast to remaining
  cells needs the transporter genes (kcc-2, nkcc-1, abts-1, eat-6,
  mca-3, sca-1, ncx-1/2/3/4, pmr-1) added. Bounded data-engineering
  block.
- **Documentation propagation.** Public-facing `c-elegans-multimodal.mdx`
  + `anesthesia-pipeline.mdx` updates after each major Layer ships.
  Honest scope labels per Layer's epistemic status.
- **Citation discipline.** Wave 2's primary-source verification protocol
  applies to every new biology citation introduced in Layers 1-7.
- **Parameter audit before integration (NEW, surfaced 2026-05-12 from
  §7.3).** Before composing any inherited parameter set into the substrate,
  audit what state variables and reversal potentials its fit assumed;
  verify those assumptions are consistent with current substrate state; if
  not, flag for refit before composition. Standing methodology step
  applied to:
    - Nicoletti 2024 channels → §7.3.5 audit + refit (in flight)
    - Wicks 1996 graded-release Boltzmann → audit before any Layer 3+
      WB3-equivalent reuse
    - Nicoletti calcium pool dynamics → audit before Layer 4 ER integration
    - Peptide release rate-coupling → audit before Layer 5+ neuromodulation
  See design doc `docs/layer1_design_decisions.md` §8 for the
  methodology + per-set flag rationale.

---

## Status log

- **2026-05-12** — Roadmap initialized during Layer 1 §7.1 foundation
  work block. Layer 1 design doc resolved + authorized; Layers 2-7
  placeholders.
- **2026-05-12 (later same day)** — §7.1, §7.2 v2, §7.3 all SHIPPED in
  one extended work block. §7.3 surfaces inherited-parameter audit
  finding (Nicoletti E_Ca = 60 mV / [Ca]_in ≈ 17 μM inconsistency).
  §7.3.5 (Layer 1.5) inserted to address. §7.4+ blocked until refit.
  Cross-cutting "parameter audit before integration" track added.

---

## Files of record

- This document: `docs/substrate_redesign_roadmap.md`
- Layer 1 design: `docs/layer1_design_decisions.md`
- Phase G CALIBRATION_GAP (load-bearing context):
  `AnestheticSimulator/artifacts/phase_g/CALIBRATION_GAP.md`
- Phase G standing followups:
  `AnestheticSimulator/artifacts/phase_g/standing_followups.md`
- Brain v3.5 lock: `docs/brain_v3.5_locked.md`
- State of claims (rebase deliverable): `docs/state_of_claims_2026-05-02.md`
