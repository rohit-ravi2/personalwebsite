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

## §7.3.5 (Layer 1.5) — Channel-Substrate Consistency: Path 2 (gene-expression-derived parameterization)

**Status:** Phase 1 in progress (methodology document SHIPPED 2026-05-12).
BLOCKS §7.4 and downstream work until all 7 phases complete.

**Motivation:** §7.3 integration surfaced that Nicoletti 2024's channel
parameterization assumes fixed `E_Ca = 60 mV` (implying `[Ca]_in ≈ 17 μM`,
340× the physiological 50 nM). Under Layer 1's dynamic Nernst this
inconsistency causes runaway Ca accumulation in all four cells. Two
resolution paths considered:

- **Path 1 (rejected):** Refit Nicoletti gbar values under physiological
  Nernst. Preserves Nicoletti as calibration anchor; preserves the
  inherited-fit dependency. Tractable but doesn't address the methodology
  pattern.
- **Path 2 (authorized 2026-05-12):** Replace inherited fits with
  derivation from biology — γ × TPM × E_translation × C_global. One global
  scaling constant calibrated once. Nicoletti's I-V curves become
  validation targets, not anchors.

Full finding documentation: `docs/layer1_design_decisions.md` §8,
`scripts/brain/wave2/artifacts/layer1_7_3_findings.md`. Path 2 methodology
codified in `docs/channel_parameter_derivation_methodology.md`.

**Pre-authorized design decisions (Rohit, 2026-05-12):**
- Single global `C_global` scaling constant for v1
- Literature-fallback γ values with explicit epistemic labels
- Uniform `E_translation = 1.0` for v1
- Reference: EGL-19 in AVAL
- Refinement triggers deferred to Phase 5 evidence

**Phase 1 pushback resolutions (Rohit, 2026-05-12):**
- **Unit pipeline:** Path B intensive formulation; surface area NOT in gbar
  formula (enters only via cell-builder membrane equation downstream)
- **Multi-gene channel aggregation:** default = paralog-separate (each gene
  forms its own channel); exception = min-across-pore-forming for documented
  heteromers; auxiliary subunits ignored in density estimation
- **C_global plausibility:** sanity-check-based hard-stop (negative,
  infinity, <1 channel/cell, >10^7 channels/cm²), not fixed numerical range
- **AVA voltage anchor:** Liu/Chen/Wang 2020 *Nat Commun* 11:5076 (NOT
  Mellem 2008; corrected per Wave 2 Mellem investigation)

**Scope — 7 phases:**

1. **Phase 1 (SHIPPED 2026-05-12)** — Methodology document.
   Deliverable: `docs/channel_parameter_derivation_methodology.md` with 7
   sections (architectural principle, per-channel formula, calibration
   protocol, validation hierarchy, failure modes, epistemic labeling,
   forward-looking application). Pushback resolutions baked in. Reference
   material for the entire substrate redesign trajectory.

2. **Phase 2** — Single-channel γ literature scoping. Per-channel
   inventory of γ (pS) values with citation + epistemic label
   (empirically grounded / biophysically derived / approximation from
   adjacent biology). Direct C. elegans → heterologous expression of the
   C. elegans gene → mammalian/Drosophila homolog fallback hierarchy.
   Deliverable: `docs/channel_gamma_inventory.md`.

3. **Phase 3** — CeNGEN TPM extension + translation efficiency. Audit
   coverage of channel genes in local CeNGEN panel; pull missing TPMs;
   document `E_translation = 1.0` v1 uniform assumption; verify per-cell
   surface area from §7.1. Deliverable: `docs/channel_tpm_inventory.md`.

4. **Phase 4** — `C_global` calibration. Compute from EGL-19 in AVAL
   reference with explicit unit pipeline; sanity-check biophysical
   plausibility. Deliverable: `docs/channel_calibration_protocol.md`
   + named constant in Layer 1 code.

5. **Phase 5** — Derive all channel parameters + per-channel validation.
   Build `scripts/brain/wave2/channels/derived_channel_parameters.py`.
   Per-(channel, cell): compute derived gbar; compare to Nicoletti
   (2× / 5× / beyond bands); run I-V validation against published
   voltage-clamp envelopes. Critical decision point: assess methodology
   adequacy. Deliverable: `scripts/brain/wave2/artifacts/path2_channel_validation.md`.

6. **Phase 6** — Per-cell integration + cell-level validation. Replace
   hardcoded Nicoletti gbar values in AVAL/AVAR/AIY/RIM with derived
   values (kinetic parameters preserved). Re-run §7.3 acceptance:
   rest stability (±2% on ions, [Cl]_in 3-7 mM, [Ca]_in near target),
   V_rest in published range, voltage-clamp envelopes within tolerance,
   cross-cell consistency.

7. **Phase 7** — Documentation + commit + push. Update design doc §8.6 +
   §8.7 with Path 2 outcomes + forward-looking application. Four commit
   groups (A: methodology, B: inventories, C: calibration + derivation,
   D: cell integration). Push to origin/main. Save memory.

**Acceptance criteria (Path 2 work block as a whole):**
- Methodology document peer-readable, reference-quality (Phase 1) ✓
- γ inventory complete for every channel in Wave 2 builders with per-
  entry citation + epistemic label (Phase 2)
- TPM inventory complete for every (channel, cell) combination (Phase 3)
- `C_global` computed; biophysically sensible; reference verified by
  construction (Phase 4)
- Per-channel validation: most channels within 2-5× of Nicoletti; >5×
  outliers documented as substantive findings (Phase 5)
- Per-cell validation: all four cells stable rest with physiological
  [Ca]_in; V_rest in published range; voltage-clamp envelopes within
  tolerance; cross-cell consistency (Phase 6)
- All documented and committed cleanly (Phase 7)

**Dependency chain:** §7.3.5 BLOCKS §7.4 (Phase F restructure depends on
correct Ca dynamics). Path 2 methodology also informs Layer 2 (kinetic
parameter audit) and Layer 3+ (Wicks 1996 receptor parameter audit).

**Estimate:** 8-12 work blocks total. Per-phase:
- Phase 1: 1 (SHIPPED)
- Phase 2: 1-2 (literature scoping, may need multiple passes)
- Phase 3: 1 (CeNGEN extension is bounded)
- Phase 4: 1 (calibration math + sanity check)
- Phase 5: 2 (derivation + per-channel validation, possibly with refinement)
- Phase 6: 1-2 (cell integration + validation, possibly with substantive findings)
- Phase 7: 0.5 (documentation + commit)

**Hard-stop conditions:**
- CeNGEN data inaccessible or schema changed (Phase 3)
- Single-channel γ values cannot be sourced for >50% of channels (Phase 2)
- `C_global` biophysically nonsensical (Phase 4)
- > 50% channels beyond 5× discrepancy (Phase 5)
- Cells fail stable rest with derived parameters after acceptable channel-
  level validation (Phase 6)
- Brian2 codegen failures or repeated failure patterns

Each hard stop writes `HARD_STOP.txt` with diagnosis + terminates cleanly.

**Files of record (Phase 1):**
- `docs/channel_parameter_derivation_methodology.md` (the methodology)
- `docs/channel_parameter_derivation_methodology_pushback.md` (pre-flight)
- `scripts/brain/wave2/artifacts/path2_progress_summary.md` (live tracking)
- `scripts/brain/wave2/artifacts/path2_phase1_checkpoint.json` (resumability)

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
- **Parameter audit before integration (FOUR AUDITS, surfaced
  2026-05-12 across §7.3, §7.3.5 Phase 5/6, and v2 reorientation).**
  All four audits required before composing any inherited parameter set:

    1. **State-variable audit** (introduced §7.3): does the fit's implicit
       ionic state match the current substrate?
    2. **Uniqueness audit** (introduced §7.3.5 Phase 5): does the paper
       report error bars / sensitivity analyses / uniqueness evidence?
    3. **Channel-inclusion audit** (introduced §7.3.5 Phase 6): does
       the inherited model set parameters to zero, implicitly making
       channel inclusion/exclusion decisions?
    4. **Measurement-vs-fit audit** (introduced §7.3.5 v2 reorientation):
       does the paper publish underlying measurements (raw traces,
       protocols, I-V data with SEM)? Use measurements; treat
       inherited fits as one of many possible parameterizations.

  Standing methodology applied to:
    - **Nicoletti 2024 channels:** all four audits failed
      (state inconsistency §7.3 + non-uniqueness §7.3.5 Phase 5 +
      channel-inclusion ambiguity §7.3.5 Phase 6 + measurements
      available so consume them §7.3.5 v2)
    - **Wicks 1996 graded-release Boltzmann:** all four audits required
      before any Layer 3+ WB3-equivalent reuse
    - **Nicoletti calcium pool dynamics:** all four audits required
      before Layer 4 ER integration
    - **Peptide release rate-coupling:** all four audits required
      before Layer 5+ neuromodulation

  **Foundational methodology principle (§2.9 design doc, "machine-code up"):**
  Biological validity arises from accurate biophysical modeling of the
  underlying structure, not from injection of measurement-matched
  parameters. Each layer of the substrate is built with full mechanistic
  sophistication; cell-level, network-level, and behavior-level phenomena
  are emergent consequences of the biophysics. Validation occurs against
  measured biology at multiple scales; parameters derived from biology
  with minimal explicit calibration.

  See design doc `docs/layer1_design_decisions.md` §2.9 (foundational
  principle) + §8.6 (uniqueness audit) + §8.10 (channel-inclusion audit)
  + §8.11 (measurement-vs-fit audit) + §8.12 (reorientation summary)
  for full methodology framework. Methodology doc
  `docs/channel_parameter_derivation_methodology.md` §3.0 + §4.0
  documents v2 calibration + validation framework under reorientation.

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
