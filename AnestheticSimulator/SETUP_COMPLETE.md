# Wave P — Setup complete

**Date:** 2026-04-27
**Status:** Kickoff package complete. Phase A not yet started.
**External spend:** **$0.** Wave P runs entirely on local hardware (RTX 4060 Ti, 8 GB VRAM) plus free-tier overflow. No cloud bursts, no FEP cloud spend, no Colab Pro, no commercial licenses.

---

## ZERO EXTERNAL SPEND declaration

The Wave P plan as scoped requires **no out-of-pocket cost** beyond hardware and electricity already paid for. The cost-elimination decisions are:

1. **Cloud bursts dropped entirely** ($0 instead of $200-400). FEP top-10 confirmation is moved to a deferred / speculative appendix in `preregistration/phase_b_binding_pose.md` §13. The canonical Phase B cascade is Vina + DiffDock + GNINA, with GNINA-derived Kd ranking sufficient for the multi-target occupancy framing at Gate C.1.
2. **Colab dependency softened to overflow-only.** Default path is RTX 4060 Ti local. Colab free tier (T4, ~12 hr/day) is reserved for pentameric edge cases that don't fit in 8 GB. Cumulative Colab budget across the program: ~30 hours = ~3 calendar days at the cap.
3. **Open-source structure-prediction substitutes added as primary fallbacks.** ESMFold (Lin et al. 2023, MIT), OpenFold (Ahdritz et al. 2024, Apache 2.0), and Boltz-1 (Wohlwend et al. 2024, MIT) replace AF-Multimer / RFAA as the load-bearing layer. AF-Multimer / RFAA remain available for academic-use cross-validation but are non-load-bearing.
4. **Complex I full-assembly dropped.** Phase A targets **single-subunit-per-anesthetic-site** modeling (GAS-1 primary per Morgan & Sedensky 1995; NUO-1 through NUO-6 individually). Full ~45-subunit assembly is DEFERRED / SPECULATIVE.
5. **Storage** uses pre-existing `/mnt/ssd4tb/` 4 TB SSD; ~120 GB peak allocation; cost $0.

If the user later reverses any of these decisions, the deferred enhancements are documented in `infrastructure/compute_budget.md` §4 and `preregistration/phase_b_binding_pose.md` §13.

---

## File inventory

**Total files:** 58 (after cleanup of smoke-test artifacts)
**Total size:** ~460 KB of plain markdown / CSV / Python / shell.

Breakdown:

- 4 top-level docs (README, WAVE_P_ARCHITECTURAL_PLAN, STATUS, SETUP_COMPLETE)
- 1 .gitignore
- 10 preregistration documents (one per phase A through J)
- 5 infrastructure documents (dependencies, compute budget, directory structure, setup_local.sh, setup_colab.md)
- 4 target/anesthetic CSVs + 1 target rationale + 1 pocket-residues CSV
- 1 anesthetic-panel CSV + 6 SMILES files + 1 ligand prep skeleton
- 10 Python skeletons (one per phase A through J)
- 2 validation docs (empirical_anchors.md, mutant_panel.csv)
- 3 integration docs (wave2, notebooks, production_simulator handoffs)
- 1 risk register
- 1 timeline
- 1 paper outline
- 9 artifacts/ subdirectory READMEs

The whole kickoff package fits in ~460 KB and contains zero binary or large files. Everything is text and version-controllable.

---

## What is ready for Phase A kickoff

### Plans in place

- `preregistration/phase_a_structural_priors.md` — full plan with goal, method, compute budget, success criteria (Gate A.1: ≥ 22/25 with pLDDT > 70 at pocket; ≥ 22/25 with TM-score ≥ 0.5 vs homolog; ≥ 22/25 with oligomeric PAE ≤ 10 Å), halting rules, falsifiability checks.
- `targets/tier1_targets.csv` — 25 Tier-1 targets with WormBase IDs, UniProt IDs, predicted oligomer state, AlphaFold DB URLs, mammalian PDB homologs, pocket compartment assignments.
- `targets/target_panel_rationale.md` — paper-by-paper justification for each Tier-1 inclusion.
- `targets/pocket_residues_homolog.csv` — homolog pocket residue mapping skeleton (Phase A populates the *C. elegans*-side residues).
- `infrastructure/setup_local.sh` — bash setup script for Wave P environments + AlphaFold DB monomer downloads.
- `infrastructure/setup_colab.md` — Colab pipeline notes for AlphaFold-Multimer multimer runs.
- `validation/empirical_anchors.md` — citation hygiene matrix; 8 PMIDs are blocking-items at Phase A entry.

### Code skeletons in place

- `src/phase_a_structures.py` — runnable: `python phase_a_structures.py --dry-run` lists the 25 targets; `--pull-alphafold-db` pulls AF DB monomers (real download, scaffold but functional); `--gate-evaluation` writes a placeholder verdict JSON.
- All 9 other phase scaffolds runnable with `--dry-run`. All print "PHASE X SCAFFOLD — implementation pending — see preregistration/phase_x_*.md".
- `anesthetics/prepare_ligands.py` — runnable; RDKit prep is functional, AM1-BCC requires AmberTools antechamber.

### What runs without further setup

- `python src/phase_c_occupancy.py --verbose` — produces a placeholder occupancy matrix using Kd=100uM defaults; demonstrates the Hill + partition pipeline; smoke-test confirms the math is correct (when real Kd values arrive from Phase B, the matrix becomes meaningful).
- `python src/phase_f_metabolic.py --smoke-test` — produces placeholder ATP / K-ATP / V-shift estimates for WT, gas-1, mev-1, atp-2 baselines; demonstrates the metabolic ODE skeleton works.

---

## What needs user review BEFORE Phase A kickoff

After applying the zero-external-spend decisions, the blocking-items list collapses to **4 desk-work / local-check tasks** with $0 cost. The previous license-verification, Colab-quota, and FEP-cloud-burst items are no longer load-bearing (see "ZERO EXTERNAL SPEND declaration" above).

### 1. Citation pre-flight queue (~1-2 hours, free PubMed lookup)

8 PMIDs are blocking items at Phase A entry (or at the phase that cites them):

- Rahman 2022 Torpedo nAChR PDB 7QL5 — Phase A.
- Yip 2013 propofol GABA-A photolabel — Phase B.
- Jayakar 2014 propofol GABA-A photolabel — Phase B.
- Trudell isoflurane GABA-A photolabel (specific paper) — Phase B.
- Morgan 1995 isoflurane EC50 — Phase C, G, H.
- Boddington 2017 propofol — Phase C, H.
- van Swinderen 2004 Ca cooperativity — Phase D, E, G.
- van Swinderen 1999 unc-13 hypersensitivity — Phase G, H.
- Kayser 2001 gas-1 Complex I — Phase F.

Resolution effort: ~1-2 hours of focused PubMed lookup. The full queue is in `validation/empirical_anchors.md`. Cost: $0 (PubMed is free).

### 2. UniProt ID re-verification (~1-2 hours, free UniProt + WormBase)

The UniProt IDs in `targets/tier1_targets.csv` were assigned best-effort at kickoff. Some may be incorrect or refer to outdated entries. Pre-Phase-A, re-verify each against current WormBase + UniProt across all 25 Tier-1 targets. Effort: ~1-2 hours. Cost: $0.

### 3. Wave 2 IRK + UNC-103 ship-status check (~30 min internal status)

Wave 2 had IRK and UNC-103 in flight as of 2026-04-26. Wave P's Phase G uses these for AVA simulation; until they ship, Wave P uses a hybrid Brian2+NEURON path (documented in `integration/wave2_handoff.md`). **Does not block until Phase G in month 4.** Effort: ~30 min internal status check. Cost: $0.

### 4. Storage allocation confirmation on existing `/mnt/ssd4tb/` (~5 min verification)

Wave P will produce ~120 GB of intermediate artifacts during peak (months 3-5). The user has `/mnt/ssd4tb/` (4 TB SSD) already mounted. Verify free space ≥ 200 GB and decide on a top-level project subdirectory path. Effort: ~5 min. Cost: $0.

---

**Items removed from blocking list (no longer load-bearing):**

- ~~License verification of AF-Multimer / RFAA~~ — non-load-bearing in revised plan; ESMFold / OpenFold / Boltz-1 (MIT / Apache 2.0) are the load-bearing predictors.
- ~~ColabFold quota / Colab Pro escalation~~ — Colab is overflow-only, free-tier T4 sufficient (~30 cumulative hours total).
- ~~FEP cloud-burst budget approval ($200)~~ — FEP dropped from canonical Phase B; deferred to `preregistration/phase_b_binding_pose.md` §13.

---

## Concrete first-week task list

After user approves the kickoff. All work is local / desk-based; **$0 spend**.

**Day 1 — Desk-work blocking items (4 items, all free)**
- Resolve all 8 blocking PMIDs in `validation/empirical_anchors.md` (~1-2 hr PubMed).
- Re-verify UniProt IDs in `tier1_targets.csv` against current WormBase + UniProt (~1-2 hr).
- Quick ship-status check on Wave 2 IRK / UNC-103 (~30 min).
- `df -h /mnt/ssd4tb/` storage allocation verification (~5 min).
- Update `validation/empirical_anchors.md` with verified PMIDs.

**Day 2 — Environment setup (local, no Colab quota check needed)**
- Run `bash infrastructure/setup_local.sh --phase A` to set up `~/venvs/wave-p-dock/` and `~/venvs/wave-p-md/`.
- Install ESMFold (`pip install fair-esm[esmfold]`), OpenFold (clone + pip install), Boltz-1 (`pip install boltz`) into the structure-prediction venv.
- Verify each predictor loads on the 4060 Ti and runs a smoke-test inference on a small monomer.
- ColabFold install is optional (overflow path only).

**Day 3 — AlphaFold DB monomer pulls + ESMFold smoke runs**
- Run `bash infrastructure/setup_local.sh --download-structures` to pull all 25 monomer entries from the AlphaFold DB.
- Verify each PDB loads in PyMOL and looks structurally reasonable.
- Run ESMFold locally on 2-3 *C. elegans* sequences not in AF DB to confirm the local pipeline works end-to-end.
- Document any AF DB failures or stale model_v3 entries.

**Day 4-5 — Local pentamer pipeline (Boltz-1 primary)**
- First pentamer (UNC-49) on Boltz-1 locally; expected ~2-6 hr on 4060 Ti.
- Compare result to mammalian 6X3X via PyMOL alignment.
- If Boltz-1 fails, try ESMFold; if both fail, escalate to free-tier Colab T4.
- Schedule remaining pentamers across week 2 with the working predictor.

**Day 6-7 — Pocket residue identification**
- For each Tier-1 target, identify the homolog pocket residues via sequence alignment.
- Populate `targets/pocket_residues_homolog.csv` with *C. elegans*-side residue numbers.
- Document any targets where homolog mapping is ambiguous (UNC-79, UNC-80, NLF-1).

End of week 1: Phase A is in progress; first 1-2 pentamer Boltz-1 results in hand; Gate A.1 evaluation begins week 2. **Cumulative external spend through end of week 1: $0.**

---

## Honest list of stubs that aren't truly implementable yet

The following components are **scaffolds** that need empirical tuning, missing dependencies, or methodological elaboration before they become functional:

1. **`src/phase_b_dock.py` cascade execution.** The Vina + DiffDock + GNINA cascade requires GNINA installed (conda recommended) and DiffDock cloned + dependencies built. Skeleton is functional for `--dry-run`, `--gate-evaluation` only. **Real cascade is implemented at Phase B work block.**

2. **`src/phase_d_kinetic_shifts.py` MD pipeline.** OpenMM driver, CHARMM-GUI Membrane Builder integration, and HOLE / pore-radius analysis are not implemented. **Implementation is the bulk of Phase D work (week 1-2 of month 3).**

3. **`src/phase_e_markov_synapses.py` Brian2 module.** The `MarkovSynapseModule` class skeleton exists; full Brian2 ODE + event-based Gillespie SSA is not implemented. **Implementation is the Phase E work block.**

4. **`src/phase_g_network_runs.py` 2,400-run grid execution.** The grid-driver structure is in place; per-run integration with Wave 2 channels + Markov synapses + metabolic layer is not implemented. **Implementation is the Phase G work block (~80 GPU-hours).**

5. **`src/phase_i_inverse_jax.py` JAX simulator.** Manual JAX reimplementation of LIF + 7 channels + Markov + metabolic is required. ~2 weeks engineering. **Deferred until Phase H ≥ 6/8 anchors.**

6. **fpocket parameter tuning (Phase B).** fpocket has empirical druggability-score thresholds that need calibration on anesthetic-class molecules. The default (`-d 1`) is for general drug discovery; anesthetic pockets are smaller and more diffuse. **Empirical tuning needed when Phase B activates.**

7. **K-ATP channel parameters (Phase F).** `K_ATP_uM` and `n_ATP` defaults are textbook neuronal values. Wave 2 has not yet translated K-ATP from the worm. The phenomenological coupling in Phase F is acceptable for proof-of-concept but **Wave 2 K-ATP translation is the eventual production-grade source.**

8. **FSM IMMOBILIZED threshold (Phase G).** Calibrated against WT control runs; threshold not pre-determined. **Sensitivity analysis at ±10% threshold is part of Phase G's work.**

9. **Hill coefficient n=1 default (Phase C).** Pentameric receptors may have n > 1 due to multiple symmetric binding sites. Wave P uses n=1 default and runs sensitivity analysis at n=2; **per-target measured n is a Tier-2 refinement.**

10. **AM1-BCC partial charges (anesthetics/prepare_ligands.py).** Requires AmberTools antechamber binary. The skeleton falls back to Gasteiger charges if antechamber is unavailable; acceptable for the canonical Vina + DiffDock + GNINA cascade. (Note: AM1-BCC would be needed if the user later authorizes the deferred FEP path per `phase_b_binding_pose.md` §13, but is not required for the canonical Phase B.)

11. **UNC-79 / UNC-80 / NLF-1 structures.** These auxiliary subunits are large (>1000 aa for UNC-79 and UNC-80) with significant intrinsically disordered regions. AlphaFold confidence will be poor at full length. **Phase A focuses on structured domains; Phase B targets the structured anesthetic-binding domain (if any). Some failures here are expected and pre-accepted in Gate A.1.**

12. **Mammalian-control MD calibration (Phase D).** Three control systems (TREK-1, GABA-A, NALCN) need to produce shifts within 2× of published values for the *C. elegans* MD pipeline to be trusted. **If controls fail, fall back to literature-only kinetic shifts (reduced coverage).**

---

## Wave P kickoff is complete

The kickoff package establishes the scaffolding for a 6-month research program. Nothing has shipped. Phase A activates upon user approval of the items in §"What needs user review BEFORE Phase A kickoff" above.

The first program-level decision point is **Gate C.1 at end of month 2**: does multi-target binding at clinical anesthetic concentrations actually involve multiple targets? The kickoff package is structured around making that decision honestly, with the Plan B (single-target pivot, publishable negative result) documented if it goes the other way.

The overall program-level success criterion is **≥ 4 of 8 anchor predictions match published wet-lab data within 2× tolerance** AND **per-target lesion analysis fails to reproduce the full anesthetic effect** (multi-target framing supported). Below these thresholds, Wave P pivots or reframes; either outcome is valuable scientifically.
