# Commit groupings proposal — Wave P (CP2 of WB1)

**Date:** 2026-04-28
**Status:** PAUSE FOR REVIEW. No `git add` or `git commit` run.
**Predecessor:** `commit_proposal.md` (CP1 — `.gitignore` patch applied; file count 6,250 → 303).
**Repo:** `rohit-ravi2/personalwebsite` (PUBLIC), branch `main`, 4 commits ahead of `origin/main`.

---

## Overview

303 untracked files in `AnestheticSimulator/` organized into **15 logical commits (A–O)**. Each commit represents one cohesive unit of work. Order is dependency-respecting: foundation/scaffold → pipeline phases → calibration → rigor pass → downstream propagation → documentation.

Honest scope labels per the project's methodology pattern. Where the v1 framing was downgraded by the rigor pass, the commit message says so explicitly. Where work is scaffold (not shipped end-to-end), the commit message marks it scaffold.

Conventional Commits format (`feat:`, `chore:`, `docs:`, `refactor:`) per home `CLAUDE.md` repo convention. All messages use imperative mood.

---

## Group A — Foundation: project scaffold, architecture, preregistration

**Files (33):**
- `.gitignore` (the CP1-patched version)
- `README.md`, `WAVE_P_ARCHITECTURAL_PLAN.md`
- `infrastructure/{compute_budget,dependencies,directory_structure,setup_colab}.md`, `infrastructure/setup_local.sh`
- `timeline/timeline.md`, `risk/risk_register.md`, `papers/wave_p_paper_outline.md`
- `integration/{notebooks_handoff,production_simulator_handoff,wave2_handoff}.md`
- `preregistration/phase_{a,b,c,d,e,f,g,h,i,j}_*.md` (10 phase preregistration docs)
- `validation/{empirical_anchors.md,mutant_panel.csv}`
- `targets/{tier1_targets.csv,tier1_targets_corrected.csv,tier2_targets.csv,target_panel_rationale.md,pocket_residues_homolog.csv}`
- `src/correct_target_csvs.py`
- `artifacts/{logs,runs,validation}/README.md`

**Proposed message:**
```
chore(wave-p): project scaffold + preregistered phase plan

Architectural plan, infrastructure docs ($0 external spend, RTX 4060 Ti
constrained), preregistration for Phases A-J (binding pipeline → kinetic
shifts → network perturbation → empirical validation → inverse design),
target panel (Tier 1 30 C. elegans anesthetic targets + Tier 2 candidates),
mutant panel, integration handoff stubs to production simulator + Wave 2
brain + parent C-Elegans notebooks.

Phases A-D shipped end-to-end (separate commits); Phases E/F implemented
as scaffolds; Phases G partially shipped (separate commits); Phases H/I/J
scaffolded only.

The original Tier 1 target CSV had 30/32 wrong UniProt IDs caught during
Phase A re-fetch; corrected version shipped alongside. correct_target_csvs.py
produces the corrected table from the audit.
```

---

## Group B — Anesthetic + negative control panels + ligand prep

**Files (10):**
- `anesthetics/anesthetic_panel.csv` — 6 anesthetics (halothane, isoflurane, sevoflurane, propofol, etomidate, ketamine)
- `anesthetics/negative_control_panel.csv` — 8 negative-control ligands (benzene, methanol, n-pentane, cyclohexane, dimethyl ether, hexafluoroethane, cis-1,2-DCE, trans-1,2-DCE)
- `anesthetics/anesthetic_smiles/*.smi` (6 files — canonical SMILES)
- `anesthetics/prepare_ligands.py`

**Proposed message:**
```
feat(wave-p): anesthetic + negative-control ligand panels

6 clinical anesthetics (3 halogenated volatiles + 3 IV), 8 negative
controls including the 4 Eger 2001 non-immobilizers (cis vs trans
1,2-DCE pair, hexafluoroethane). Canonical SMILES checked in;
RDKit/Meeko-prepared .sdf and .pdbqt ligand intermediates are
gitignored as regeneratable.

prepare_ligands.py runs RDKit 3D embedding + GAFF/AM1-BCC charge
assignment + Meeko PDBQT export.
```

---

## Group C — Phase A: AlphaFold structure fetch + pocket detection + citation audit

**Files (~120):**
- `src/phase_a_{fetch_alphafold_db,pocket_detect,structures,esmfold_missed}.py`
- `artifacts/structures/<gene>_<acc>_out/{*.pml, *.tcl, *_PYMOL.sh, *_VMD.sh, *_info.txt}` for 23 genes (PDB/PQR/CIF binaries are gitignored; only fpocket auxiliary scripts kept)
- `artifacts/structures/{README.md, pocket_summary.csv, uniprot_id_audit.csv, esmfold_missed.log}`
- `CITATION_AUDIT_2026-04-27.md`, `REVISION_LOG_2026-04-27.md`

**Proposed message:**
```
feat(phase-a): AlphaFold DB v6 fetch + fpocket pocket detection (30/32)

UniProt → AlphaFold DB v6 latestVersion API → per-gene pocket detection
via fpocket. 30/32 Tier-1 targets have AlphaFold structures; NCA-1
(Q6Q762) and UNC-80 (Q9XV66) are missing from AF DB and deferred to
ColabFold T4 fallback per R14 mitigation.

ESMFold local fallback attempted for the 2 missing — OOMed on 8GB
VRAM (RTX 4060 Ti); skipped at this work block.

Citation audit (CITATION_AUDIT_2026-04-27.md) catalogs corrections to
6 wrong/fabricated PMIDs in original target CSV including
Crowder 1996 (8855256→8873562), Morgan & Sedensky (1995→1994 PMID
7943840), Sedensky 1992 (1346264 → Sedensky & Meneely 1987 PMID
3576211), van Swinderen 1999 (cited for unc-13, actually unc-64),
Sedensky 2001 PMID 11756669 (twk-18 — fabricated PMID), Boddington
2017 (propofol — fabricated). REVISION_LOG_2026-04-27.md tracks the
propagation through dependent docs.

Per-gene fpocket auxiliary files kept in artifacts/structures/<gene>_<acc>_out/;
PDB/PQR binaries gitignored as regeneratable.
```

---

## Group D — Phase B: Vina docking pipeline

**Files (6):**
- `src/phase_b_{dock,dock_pipeline,prepare_ligands}.py`
- `src/scan_pose_affinities.py`
- `artifacts/binding/{README.md, vina_results.csv, vina_results_from_poses.csv}`

**Proposed message:**
```
feat(phase-b): AutoDock Vina pipeline — 540 dockings (6 ligands × 30 targets × 3 poses)

Receptor preparation via Meeko mk_prepare_receptor; ligand prep via
mk_prepare_ligand from RDKit 3D structures; Vina 1.1.2 docking
across all (anesthetic, target) pairs at default exhaustiveness.

Top-3 poses per pair retained; binding affinity in kcal/mol logged
to vina_results.csv. scan_pose_affinities.py provides quick re-tabulation
from the per-pose log files.

Vina poses (PDBQT) gitignored as regeneratable; vina_results.csv ships
the affinities used by downstream Phase C.
```

---

## Group E — Phase C/D: occupancy matrix + kinetic shifts → wave2_overlay v1

**Files (10):**
- `src/phase_c_occupancy.py`
- `src/phase_d_kinetic_shifts.py`
- `src/finalize_phase_a_to_d.py`
- `artifacts/occupancy/{best_pocket_per_target.csv, gate_c1_summary.md, occupancy_matrix.csv, README.md}`
- `artifacts/kinetics/{kinetic_shifts_at_1xEC50.csv, phase_d_summary.md, wave2_overlay.json, README.md}`
- `WAVE_P_PHASE_ABCD_MILESTONE.md`

**Proposed message:**
```
feat(phase-c-d): occupancy + kinetic shifts → wave2_overlay.json (v1)

Phase C: Vina ΔG → predicted Kd (Kd = exp(ΔG/RT) × 1e6 at 298 K),
amplified by membrane-partition K_p per anesthetic (halothane Kp=250,
etc.); Hill-equation occupancy at clinical EC50.

Phase D: kinetic-shift translation per mechanism class (snare_cooperativity
→ n_Ca_delta, k2p_potentiation → g_KATP_max shift, gaba_potentiation →
τ_decay shift, complex_i_block → rate_factor, nca_block → leak shift,
nachr_antagonism → effective conductance reduction, glucl_potentiation →
τ_decay shift).

wave2_overlay.json packages the 6 anesthetics × 30 targets = 180
(occupancy_1xEC50 + kinetic shift parameters) for downstream Phase E/F/G
consumption.

WAVE_P_PHASE_ABCD_MILESTONE.md captures the end-to-end Phase A-D
pass-through with the full milestone framing as it stood at completion.
finalize_phase_a_to_d.py is the consolidator that produces the
milestone artifacts.

Note: this is the v1 overlay; the post-allosteric-correction v2 ships
in a later commit (Group K) with corrected occupancies per CP5/CP7.
```

---

## Group F — Phase E/F implementations (synapse + metabolic ATP)

**Files (10):**
- `src/phase_e_markov_synapse.py` (canonical implementation)
- `src/phase_e_markov_synapses.py` (earlier scaffold; kept for reference)
- `src/phase_f_metabolic_layer.py` (canonical implementation)
- `src/phase_f_metabolic.py` (earlier scaffold; kept for reference)
- `artifacts/markov/{anesthetic_perturbation.csv, baseline_calibration.csv, phase_e_summary.md, README.md}`
- `artifacts/metabolic/{atp_steady_states.csv, gas1_ec50_prediction.csv, phase_f_summary.md, README.md}`

**Proposed message:**
```
feat(phase-e-f): Markov synapse + metabolic ATP layer (initial implementations)

Phase E (phase_e_markov_synapse.py): Gillespie SSA-style stochastic
simulation of a single C. elegans NMJ synapse with cooperative
Ca-SNARE binding (Hill exponent n) → fusion → recycling. Anesthetic
perturbation: shift n by n_Ca_delta from wave2_overlay.json. Reproduces
Stewart 2000 PMID 11095753 / van Swinderen 1999 PMID 10051668
release-p reduction band 0.3-0.7 at clinical halothane concentration.

Phase F (phase_f_metabolic_layer.py): analytic ATP steady-state +
K-ATP coupling + V-shift threshold. Predicts gas-1(fc21) hypersensitivity
ratio versus WT halothane EC50.

Both phases scaffolded with literature-grounded parameters; Phase F's
predicted 2.48× gas-1 ratio at GAS1_COMPLEX_I_FACTOR=0.4 lands in
Morgan & Sedensky 1994 PMID 7943840's 2-3× target band — but the
rigor pass (CP1, separate commit) later showed this is parameter-locked
(block_factor cancels in d_WT/d_g1 ratio). The implementation here
is correct; the validation framing is downgraded by CP1.

Older scaffold variants (phase_e_markov_synapses.py plural,
phase_f_metabolic.py without _layer suffix) retained for reference
as earlier work products; canonical implementations are the singular
phase_e and the _layer-suffixed phase_f.
```

---

## Group G — Phase H/I/J scaffolds + v1 validation table

**Files (5):**
- `src/phase_h_validation.py`
- `src/phase_h_validation_consolidator.py`
- `src/phase_i_inverse_jax.py`
- `src/phase_j_signature.py`
- `WAVE_P_PHASE_H_VALIDATION.md`

**Proposed message:**
```
feat(phase-h-i-j): empirical-validation + inverse-design + network-signature scaffolds

Phase H (phase_h_validation.py + phase_h_validation_consolidator.py):
8-anchor validation table comparing predictions against published
C. elegans anesthesia phenotypes (halothane EC50, isoflurane EC50,
gas-1 hypersensitivity, unc-79/80 resistance, twk-18 sensitivity,
unc-13 hypersensitivity, propofol immobilization).

Phase I (phase_i_inverse_jax.py): JAX-based inverse-design scaffold —
gradient flow from desired phenotype → ligand parameter optimization.
Scaffolded only; no shipped predictions.

Phase J (phase_j_signature.py): network-signature scaffold for
distinguishing anesthetic-class-specific brain dynamics signatures.
Scaffolded only.

WAVE_P_PHASE_H_VALIDATION.md ships the v1 5/5 PASS verdict table.
This framing was DOWNGRADED by the CP1-CP8 rigor pass (separate
commit Group J): Phase F gas-1 ratio is structurally parameter-locked,
twk-18 anchor had inverted biological direction, propofol C. elegans
EC50 anchor had no primary source. The CP8 verdict (7+1+5+3+2 four-
category structure) supersedes this v1 5/5 PASS framing. Document
retained for history; cp8_rigor_tightened_verdict.md is the current
authoritative verdict.
```

---

## Group H — Calibration v1 (7-stage pipeline)

**Files (~21):**
- `src/calibration_dock_runner.py`
- `src/calibration_prep_negative_controls.py`
- `src/calibration_pull_mammalian_homologs.py`
- `src/calibration_scan_negative_poses.py`
- `src/calibration_stage4_dual_table.py`
- `src/calibration_stage5_discriminative.py`
- `src/calibration_stage6_rank_correlation.py`
- `src/calibration_cp6_verdict.py`
- `artifacts/calibration/calibration_summary.md`
- `artifacts/calibration/calibration_comparison_{raw,withKp}.csv`
- `artifacts/calibration/ground_truth_Kd_table.csv`
- `artifacts/calibration/{mammalian_homolog_structures,mammalian_vina_results,negative_vina_results}.csv`
- `artifacts/calibration/stage4_summary.md`
- `artifacts/calibration/stage5_discriminative.{csv,md}`
- `artifacts/calibration/stage6_rank_correlation.{csv,md}`

**Proposed message:**
```
feat(calibration-v1): 7-stage calibration with discriminative power test

Calibration pipeline against 30 anchor entries from published
mammalian-homolog electrophysiology + mitochondrial assays (no strict
radioligand-Kd entries; functional EC50/IC50 only — flagged in CP4
of subsequent rigor pass).

Stages:
  1. Pull mammalian homolog structures (calibration_pull_mammalian_homologs.py)
  2. Prepare negative-control ligands (calibration_prep_negative_controls.py)
  3. Dock anesthetics + negative controls against mammalian homologs +
     C. elegans Tier-1 targets (calibration_dock_runner.py +
     calibration_scan_negative_poses.py)
  4. Dual-table comparison: Vina-predicted Kd vs literature EC50/IC50
     (calibration_stage4_dual_table.py)
  5. Discriminative power test: anesthetic vs negative-control ligand
     class engagement gap (calibration_stage5_discriminative.py)
  6. Spearman rank correlation between Vina ΔG and experimental
     log(EC50) (calibration_stage6_rank_correlation.py)
  cp6. Consolidated verdict (calibration_cp6_verdict.py)

Stage 5 reports discriminative gap = 28 targets between anesthetic and
negative-control ligand classes. Stage 6 reports Spearman ρ = +0.93
across 22 entries. Stage 4 reports calibration within 10× tolerance
band for 75% of entries.

Verdict at this stage was DISCRIMINATIVE_AND_CALIBRATED (calibration_summary.md).
Subsequent CP1-CP8 rigor pass (separate commits) refined this to the
four-category structure and applied an allosteric correction
(f_allo = 2.50×) that shifts within-10× to 94% on the strict T1 subset.
```

---

## Group I — Pre-flight pushback documents + sensitivity-test scaffolding

**Files (3):**
- `artifacts/calibration/calibration_pushback.md`
- `artifacts/calibration/rigor_tightening_pushback.md`
- `src/preflight_phase_f_saturation.py`

**Proposed message:**
```
docs(rigor): pre-flight pushback documents + Phase F saturation diagnostic

calibration_pushback.md: pre-flight pushback for the calibration work
block. Surfaced Kd-vs-EC50 conflation concern + parameter-locking risk
+ citation chain corrections needed before claiming calibration pass.

rigor_tightening_pushback.md: pre-flight pushback for the CP1-CP8 rigor
work block. Surfaced Phase F structural parameter-lock (block_factor
cancels in d_WT/d_g1 ratio analytically), twk-18 direction inversion
(GoF is HYPERSENSITIVE per Singaram 2011 PMID 22137475, not RESISTANT
as originally claimed), propofol C. elegans EC50 anchor lacks primary
source, NCA-1/UNC-80 structures missing from AlphaFold DB.

preflight_phase_f_saturation.py: canonical sensitivity-sweep script
that produces the joint block_factor × GAS1_COMPLEX_I_FACTOR table
showing predicted ratio is structurally invariant to block_factor
across 19× variation.

These are the methodology infrastructure documents that make the
CP1-CP8 rigor pass reproducible.
```

---

## Group J — CP1-CP8 rigor-tightening pass

**Files (~23):**
- `src/calibration_phase_e_sensitivity.py` (CP2)
- `src/calibration_cp3_dce_diagnostic.py`
- `src/calibration_cp4_strict_kd_subset.py`
- `src/calibration_cp5_strict_recalibration.py`
- `src/calibration_cp7_allosteric_correction.py`
- `artifacts/calibration/phase_f_structural_diagnosis.md` (CP1)
- `artifacts/calibration/phase_e_sensitivity.{csv,md}` (CP2)
- `artifacts/calibration/dce_concentration_sweep.csv` (CP3)
- `artifacts/calibration/dce_diagnostic_summary.md` (CP3)
- `artifacts/calibration/cp4_directness_tiers.csv`
- `artifacts/calibration/cp4_strict_kd_summary.md`
- `artifacts/calibration/cp4_strict_subset.csv`
- `artifacts/calibration/cp5_strict_recalibration.{csv,md}`
- `artifacts/calibration/cp6_anchor_classification.md`
- `artifacts/calibration/cp7_class_stratified.csv`
- `artifacts/calibration/cp7_corrected.csv`
- `artifacts/calibration/cp7_summary.md`
- `artifacts/calibration/cp8_rigor_tightened_verdict.md`

**Proposed message:**
```
refactor(calibration): CP1-CP8 rigor pass — replaces v1 5/5 PASS framing

Eight-checkpoint rigor-tightening pass that downgrades several v1 PASS
verdicts and surfaces the systematic biases in the original calibration
table.

CP1 (phase_f_structural_diagnosis.md): Phase F gas-1 hypersensitivity
ratio is structurally parameter-locked. Joint sensitivity sweep over
block_factor [0.05-0.95] × GAS1_COMPLEX_I_FACTOR [0.3-0.7] shows the
predicted ratio at GAS1=0.4 varies by only 0.05 across 19× block_factor
variation. Analytical proof: (1-bf) cancels in d_WT/d_g1 ratio. Verdict
downgraded PASS_5/6 → PASS_PARAMETER_TUNED.

CP2 (phase_e_sensitivity): Phase E CLINICAL_EFFECTIVE_OCCUPANCY sweep
[0.10-0.70] shows Stewart band reproduced across [0.10, 0.30] occupancy
range. Verdict ROBUST within sensitivity envelope.

CP3 (dce_concentration_sweep + dce_diagnostic_summary): cis-1,2-DCE vs
trans-1,2-DCE conformational specificity test (Eger 2001). Max gap = 0
across 0.1-30 mM concentration grid; at Eger anesthetic-range (1 mM)
trans engages slightly more than cis. Verdict FAIL — pipeline cannot
distinguish stereoisomers of small halogenated alkanes.

CP4 (cp4_strict_kd_summary + directness_tiers): all 30 anchor entries
are functional EC50/IC50, not strict-Kd. Directness-tier framework
classifies T1 (recombinant electrophys, n=17) / T2 (native, n=4) /
T3 (mitochondrial O2, n=3). Strict T1 subset signed median log_err =
+0.399, consistent with PAM allosteric coupling η ≈ 0.4 (Forman &
Miller 2016 PMID 27749338).

CP5 (cp5_strict_recalibration): single-parameter allosteric correction
f_allo = 10^0.399 = 2.50× shifts T1 signed mean log_err to +0.13.
Within-10× rate 76% → 94%. Leave-one-anesthetic-out cross-validation
(mean held-out signed = +0.097) confirms correction generalizes across
anesthetic chemotypes.

CP6 (cp6_anchor_classification): four-category anchor reframe replacing
binary PASS/DEFERRED — VERIFIED / STRUCTURALLY_GROUNDED_BY_HOMOLOG /
STRUCTURALLY_GROUNDED_AWAITING_WETLAB / STRUCTURALLY_UNCALIBRATED.
twk-18 anchor direction corrected per Singaram 2011 PMID 22137475
(K2P GoF is HYPERSENSITIVE, not RESISTANT as originally claimed).
Propofol C. elegans EC50 anchor reframed via mammalian homolog
(no whole-animal primary source; closest is Heuer 2014 PMID 24501356
oocyte channel-level IC50).

CP7 (cp7_corrected + cp7_class_stratified + cp7_summary): per-chemical-
class stratification post-correction. 4/5 classes 100% within 10×
(ALKANE_HALOGENATED, ETHER_HALOGENATED, IV_IMIDAZOLE, IV_ARYLCYCLOHEXYLAMINE);
IV_PHENOL (propofol) 67% within 10× due to GABA-A allosteric outlier.
Hexafluoroethane non-immobilizer test: engages 30/30 targets vs cis-DCE
22/30 — confirms binding pipeline lacks Eger non-immobilizer
discrimination. Boundary FAIL documented.

CP8 (cp8_rigor_tightened_verdict): rolled-up verdict — 7 verified +
1 homolog-grounded + 5 awaiting-wetlab + 3 uncalibrated + 2 boundary
FAIL. Replaces v1 5/5 PASS headline.

The honest-framing principle applied throughout: don't conflate "no
wet-lab validation" with "no scientific value"; document boundaries
explicitly rather than chasing them with parameter tuning.
```

---

## Group K — wave2_overlay_v2 + Phase E/F v2 propagation

**Files (4):**
- `artifacts/kinetics/wave2_overlay_v2.json`
- `src/phase_ef_v2_recompute.py`
- `artifacts/calibration/phase_ef_v2_propagation.{csv,md}`

**Proposed message:**
```
feat(kinetics): wave2_overlay_v2.json — post-allosteric-correction occupancies

Apply CP5 allosteric correction f_allo = 2.50× to wave2_overlay.json:
each (anesthetic, target) entry's occupancy_1xEC50 recomputed from
corrected Kd via Hill equation. Original occupancy retained as
occupancy_1xEC50_v1 for trace auditing; new occupancy in
occupancy_1xEC50; correction_applied tag = "f_allo_2.50x_CP5".

Parameter values (n_Ca_delta, rate_factor, etc.) NOT modified — these
are mechanism-class kinetic shifts independent of the binding-affinity
correction.

phase_ef_v2_recompute.py runs Phase E and Phase F against both v1 and v2
overlays and verifies bitwise-identical predictions (max |Δfold_change|
= 0.0000 in Phase E, max |Δratio| = 0.0000 in Phase F). This empirically
confirms CP1's analytical claim that Phase F is parameter-locked: even
with corrected occupancies, the ratio is invariant.

To make Phase E genuinely consume CP7-corrected occupancies,
phase_e_markov_synapse.py would need to switch from CLINICAL_EFFECTIVE_OCCUPANCY
(hand-tuned 0.30) to per-target overlay occupancy. Documented as a
Phase G design decision in the propagation report; deferred to a
separate work block.
```

---

## Group L — Phase G architecture + perturbation manager + halothane demo

**Files (6):**
- `artifacts/phase_g/phase_g_architecture.md`
- `src/phase_g_network_perturbation.py`
- `src/phase_g_network_runs.py` (earlier scaffold)
- `artifacts/phase_g/phase_g_dose_response_summary.md`
- `artifacts/phase_g/phase_g_halothane_dose_response.csv`
- `artifacts/phase_g/phase_g_smoke_test.json`

**Proposed message:**
```
feat(phase-g): network perturbation manager + halothane dose-response demo

Phase G architecture (phase_g_architecture.md): per-mechanism-class
hook mapping (gaba_potentiation → enhance W_chem inhibitory edges;
glucl_potentiation → same on GluCl-expressing post; nachr_antagonism →
reduce excitatory ACh edges; k2p_potentiation → add hyperpolarizing
I_ext on K2P-expressing; complex_i_block → uniform K-ATP-coupled
hyperpolarizing I_ext; snare_cooperativity → scale W_syn globally).
Channel-to-neuron expression mapping v1 simplified hand-curated;
CeNGEN integration deferred to v2.

AnestheticPerturbation class (phase_g_network_perturbation.py)
consumes wave2_overlay_v2.json and produces per-(anesthetic, dose)
PerturbationProfile. Hill-equation dose scaling. apply_to_brain()
mutates Brian2 brain in-place; revert() restores. Designed as wrapper
on LIFBrain; no Wave 2 brain code modification required.

Halothane @ 1× EC50 smoke test: 8 mechanism classes engaged, 30 targets
at occupancy > 10%, max class occupancy 0.998.

Halothane dose-response sweep on 50-neuron Brian2 LIF demo network
(40 E + 10 I, recurrent E↔I with E→I, E→E, I→E synapses). 50%-firing-
rate suppression at 0.01× clinical EC50 — ~100× tighter than Crowder
1996 PMID 8873562 behavioral EC50 anchor. Documented as honest gap:
binding pipeline saturates targets at clinical EC50 (occupancy ≈ 1
across all 30 targets), so dose-response shape is determined by
network coupling sensitivity. Behavioral threshold calibration is
the gap, deferred to LIFBrain integration work block.

phase_g_network_runs.py is an earlier scaffold variant; the canonical
implementation is phase_g_network_perturbation.py.
```

---

## Group M — Wave P × Wave 2 integration scoping

**Files (1):**
- `artifacts/phase_g/wave_p_wave_2_integration_scoping.md`

**Proposed message:**
```
docs(phase-g): Wave P × Wave 2 integration scoping for Phase δ-expanded substrate

Substrate landscape (LIFBrain, GradedBrain, Phase δ-projected expansions);
Phase G mechanism class × Phase δ cell integration matrix; touch cascade
× anesthesia predictions (halothane vs etomidate qualitative differences);
mutant phenotype predictions for gas-1, twk-18(cn110), sup-9, unc-13(s69)
with primary-source anchors; 5-test execution plan (touch cascade
discrimination, gas-1 hypersensitivity, twk-18 GoF, hexafluoroethane null,
per-anesthetic dose-response sweep); cross-thread coordination requirements
with Session 1's Phase δ overnight; risk register; standing follow-ups.

Test 4 (hexafluoroethane null perturbation) is the most informative — it
genuinely tests whether network-level integration captures Eger non-
immobilizer discrimination that binding alone misses (CP3, CP7 both said
"no" at the binding-pipeline level; Phase G is the next bet).
```

---

## Group N — Methodology paper case studies

**Files (5):**
- `artifacts/methodology_paper/case_study_phase_f_parameter_lock.md`
- `artifacts/methodology_paper/case_study_kd_ec50_conflation.md`
- `artifacts/methodology_paper/case_study_eger_nonimmobilizer.md`
- `artifacts/methodology_paper/case_study_twk18_direction_inversion.md`
- `artifacts/methodology_paper/case_study_preflight_pushback.md`

**Proposed message:**
```
docs(methodology): 5 case study drafts for AI-assisted-research methodology paper

Drafts (~6700 words total) documenting load-bearing methodology patterns
that surfaced during the Wave P rigor pass.

1. case_study_phase_f_parameter_lock.md (~1117 words) — sensitivity-sweep
   methodology surfaces structural parameter-lock; (1-bf) cancellation
   analytical proof; downgrade verdict to PASS_PARAMETER_TUNED.

2. case_study_kd_ec50_conflation.md (~1261 words) — directness-tier audit
   reveals all 30 ground-truth entries are functional EC50, not strict-Kd;
   f_allo = 2.50× allosteric correction; LOO-CV validates correction
   generalizes; 76% → 94% within-10× post-correction.

3. case_study_eger_nonimmobilizer.md (~1379 words) — CP3 cis/trans-DCE
   FAIL + CP7 hexafluoroethane FAIL → binding pipeline lacks Eger
   non-immobilizer discrimination. Documented as boundary, not bug.

4. case_study_twk18_direction_inversion.md (~1250 words) — original
   Anchor 6 had fabricated PMID + inverted biological direction (claimed
   RESISTANT, real is HYPERSENSITIVE per Singaram 2011 PMID 22137475).

5. case_study_preflight_pushback.md (~1731 words) — umbrella thesis
   that systematic pre-flight pushback is cost-effective methodology
   for AI-assisted scientific work. Cumulative catch-list >37 citation
   issues + 1 parameter-lock + 1 direction-inversion + Kd/EC50
   conflation + saturation collapse documented.

Drafts only — not yet integrated into a paper manuscript. Outline for
integration sketched in wave_p_wave_2_integration_scoping.md (Group M).
```

---

## Group O — Wake-up summary + STATUS + commit-process tracking

**Files (4):**
- `artifacts/calibration/wave_p_overnight_summary_2026-04-28.md`
- `STATUS.md`
- `SETUP_COMPLETE.md`
- `commit_proposal.md`, `commit_groupings.md` (this document)

**Proposed message:**
```
docs: overnight summary + STATUS + commit-process tracking

wave_p_overnight_summary_2026-04-28.md: consolidated wake-up summary
from the Stage A-D overnight run. Stage A (Phase E/F v2 verified) +
Stage B (Phase G architecture + perturbation + dose-response demo) +
Stage C (5 methodology case studies) + Stage D (Wave P × Wave 2
integration scoping) all landed; standing followups documented in
priority order.

STATUS.md: cumulative status log across the project. SETUP_COMPLETE.md
is the initial environment setup record (Wave-p-docking conda env,
RDKit, Meeko, Vina, BioPython, fpocket installed).

commit_proposal.md + commit_groupings.md: pre-flight + commit-grouping
proposal documents for this commit cycle. Tracked in repo for
reproducibility of the commit-propagation methodology.
```

---

## Summary table

| Group | Topic | Files | Approx commit size |
|---|---|---|---|
| A | Foundation: scaffold + preregistration | 33 | medium |
| B | Anesthetic + negative-control panels | 10 | small |
| C | Phase A: AlphaFold + fpocket + citation audit | ~120 | large |
| D | Phase B: Vina docking pipeline | 6 | small |
| E | Phase C/D: occupancy + kinetic shifts | 10 | small |
| F | Phase E/F: synapse + metabolic implementations | 10 | small |
| G | Phase H/I/J scaffolds + v1 validation table | 5 | small |
| H | Calibration v1 (7-stage) | 21 | medium |
| I | Pre-flight pushback documents | 3 | tiny |
| J | CP1-CP8 rigor pass | 23 | medium-large |
| K | wave2_overlay_v2 + propagation | 4 | tiny |
| L | Phase G architecture + perturbation + demo | 6 | small |
| M | Wave P × Wave 2 integration scoping | 1 | tiny |
| N | Methodology paper case studies | 5 | small |
| O | Wake-up summary + STATUS + commit tracking | 4 | tiny |
| **Total** | | **~261 + per-gene structures = 303** | |

---

## What the user is asked to approve

**Question 1 (load-bearing):** approve the 15-group structure (A-O) and the dependency-respecting order shown above?

If revisions wanted to grouping or order, flag and I'll re-propose.

**Question 2 (load-bearing):** approve the proposed commit messages as-is, or are there edits to the framing?

The messages were drafted with honest-framing in mind:
- Group F flags Phase F prediction as parameter-tuned (downgraded by CP1)
- Group G flags v1 5/5 PASS as superseded by CP8 four-category verdict
- Group H labels v1 verdict DISCRIMINATIVE_AND_CALIBRATED as the v1 framing, refined by rigor pass
- Group L documents the 100× behavioral-EC50 gap honestly as a known calibration gap, not a pass

If any of these flagged-as-honest framings should be tightened or loosened, flag now.

**Question 3 (lower-stakes):** OK to use Conventional Commits format (`feat:`, `chore:`, `docs:`, `refactor:`) per home `CLAUDE.md` repo convention? Default: yes.

---

## What happens after approval

For each approved group A-O:
1. `git add <files in group>`
2. `git status` to verify staging
3. `git commit -m "<approved message>"`
4. `git log --oneline -1` to verify commit landed
5. Append commit hash to a running `commit_log_summary.md`

After all 15 commits land:
- `git status` — verify working tree clean within Wave P scope
- `git log --oneline -16` — verify all 15 + foundation visible
- `git push origin main` — pushes 15 new commits + the 4 already-ahead pre-existing commits

**Strict pause-for-review behavior** — no `git commit` will run until the approval is received.

---

## Time-budget note

CP2 elapsed: ~30 minutes (within 30-45 min budget).

CP3 (commit execution) post-approval: ~45-60 minutes for 15 commits, mostly waiting for `git add` on the Phase A structures group.

CP4 (final verification + push) post-CP3: ~10 minutes.
