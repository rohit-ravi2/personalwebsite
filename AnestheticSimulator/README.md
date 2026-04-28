# Wave P — Anesthetic Simulator (digital pharmacology platform)

**Status:** SCAFFOLDED (kickoff package, 2026-04-27). No phase has shipped.
**Project lead:** Rohit Ravi (NYU Data Science / Philosophy minor).
**Parent program:** C. elegans biophysical-simulator (Waves 1, 2, P running in parallel).
**Sibling tracks:**

- **Wave 2** (channel translation) at `/home/rohit/Desktop/website/personalwebsite/scripts/brain/`. Produces validated Brian2 channel implementations. Wave P consumes them.
- **Notebook pipeline** (structural connectomics) at `/home/rohit/Desktop/C-Elegans/New Notebooks/`. Produces Witvliet 2020 + CeNGEN + Bentley 2016 + Cook 2019 + Brittin contact priors. Wave P consumes them.

---

## Intent

Wave P treats general anesthesia as a **multi-target network-level phenomenon** rather than a single-receptor phenomenon. The motivating science is that volatile and IV anesthetics simultaneously bind dozens of channels, receptors, SNARE-machinery proteins, and Complex-I subunits at clinically relevant concentrations; the immobilization end-point emerges from the *parallel sum* of those small effects, not from any single dominant target.

The simulator becomes a **digital pharmacology platform**: in-silico perturbation of a multi-target binding profile produces falsifiable network-level predictions (loss of locomotion, EC50 curves, mutant phenotypes) that can be compared against published *C. elegans* anesthetic phenotype data (Crowder 1996, Morgan & Sedensky 1995, Sedensky 1992/2001, van Swinderen 1999/2004, Boddington 2017, Kayser 2001).

The strongest version of the falsifiability claim sits at **Gate C.1**: if a structurally grounded occupancy estimate at clinical EC50 shows only one target above 10% occupancy, the multi-target premise is falsified and Wave P pivots to a single-target validation framework. That gate is load-bearing and is reached early in the program (month 2).

---

## Scope discipline

This directory is **completely separate** from the existing simulator at `scripts/brain/` and the notebook pipeline at `New Notebooks/`. The whole point of the separation is:

- Wave P does not modify Wave 2's channel translations. It overlays anesthetic-induced kinetic shifts on top of them.
- Wave P does not modify the notebook pipeline. It consumes its CSV/NPZ artifacts as biological priors.
- Wave P is a research program kickoff, not a production simulator. Results from Wave P are evaluated, then either fed back into the production simulator (`scripts/brain/`) via a documented handoff or left as standalone evidence.

Wave P preserves Wave 2's discipline:

- Plan-first. Each phase has a preregistration document.
- Falsifiability before elaboration. Each phase has explicit kill-criteria.
- Citation hygiene. Every cited paper carries PMID or DOI. (Wave 2 caught four primary-source misattributions; Wave P enforces verification from day 1.)
- Honest scope tags: SHIPPED / SCAFFOLDED / CALIBRATION-PENDING / DEFERRED / SPECULATIVE.

---

## How to navigate

Top-level documents:

- `README.md` (this file) — overview, intent, scope, navigation.
- `WAVE_P_ARCHITECTURAL_PLAN.md` — the master plan. Read second.
- `STATUS.md` — current phase, gate state, next concrete action. Updated per work block.
- `SETUP_COMPLETE.md` — kickoff inventory, blocking items for the user, week-1 task list.

Subdirectories:

- `preregistration/` — one document per phase (A through J). Each is an independent, self-contained planning document with goal, method, compute budget, success criteria, halting rules, falsifiability checks, integration points.
- `infrastructure/` — dependency manifest, compute budget summary, directory-structure guide, local + Colab setup scripts.
- `targets/` — Tier-1 (25) + Tier-2 (~25) target panels as CSVs, plus paper-by-paper rationale.
- `anesthetics/` — anesthetic panel CSV with SMILES + clinical EC50 + log P, plus per-anesthetic SMILES files and ligand-prep skeleton.
- `src/` — implementation skeletons, one Python file per phase. Runnable stubs that print phase scaffolding messages and reference their preregistration documents.
- `validation/` — empirical-anchor matrix per published paper, per-mutant phenotype panel.
- `integration/` — handoff documents to Wave 2, the notebook pipeline, and the production simulator.
- `risk/` — full risk register with mitigations.
- `timeline/` — month-by-month 6-month breakdown.
- `papers/` — Wave P paper outline (target venues: Cell Systems / Neuron / eLife / PLOS Computational Biology).
- `artifacts/` — empty placeholders for phase outputs. MD trajectories not git-tracked.

---

## Phase index (Phase A through J)

| Phase | Name | Document | Status |
|---|---|---|---|
| A | Structural priors | `preregistration/phase_a_structural_priors.md` | SCAFFOLDED |
| B | Binding pose prediction | `preregistration/phase_b_binding_pose.md` | SCAFFOLDED |
| C | Occupancy matrix | `preregistration/phase_c_occupancy_matrix.md` | SCAFFOLDED (load-bearing gate) |
| D | Per-target kinetic shifts | `preregistration/phase_d_kinetic_shifts.md` | SCAFFOLDED |
| E | Markov synaptic transmission | `preregistration/phase_e_markov_synapses.md` | SCAFFOLDED |
| F | Metabolic state layer | `preregistration/phase_f_metabolic_layer.md` | SCAFFOLDED |
| G | Network-level perturbation | `preregistration/phase_g_network_perturbation.md` | SCAFFOLDED |
| H | Empirical validation | `preregistration/phase_h_empirical_validation.md` | SCAFFOLDED |
| I | Inverse design (stretch) | `preregistration/phase_i_inverse_design.md` | DEFERRED |
| J | Network signatures (stretch) | `preregistration/phase_j_network_signature.md` | DEFERRED |

Phase ordering is sequential through G; H aggregates G's outputs; I and J are deferred / stretch goals that only activate after H lands.

---

## Hardware and compute envelope

- Local: NVIDIA RTX 4060 Ti 8 GB VRAM, Linux. Realistic month-1 throughput ~120 GPU-hours. **Primary compute path.**
- Colab free tier (T4 GPU, ~12 hr/day): overflow only, ~30 cumulative hours across the program. Reserved for pentameric edge cases that don't fit in 8 GB locally.
- **No cloud bursts.** External spend is $0 in the canonical plan. If the user later authorizes FEP top-10 confirmation on cloud, see `preregistration/phase_b_binding_pose.md` §13.
- Storage: `/mnt/ssd4tb/` (4 TB SSD, already mounted), ~120 GB peak allocation.
- Brain conda env at `/home/rohit/miniconda3/envs/ml/` is shared with Wave 2 / notebook pipeline. Wave P installs are in **isolated venvs** (`~/venvs/wave-p-md/`, `~/venvs/wave-p-dock/`, `~/venvs/wave-p-jax/`) to prevent conflicts. See `infrastructure/dependencies.md`.

---

## Reading order for a fresh session

1. `README.md` (this file) — five minutes.
2. `WAVE_P_ARCHITECTURAL_PLAN.md` — twenty minutes.
3. `STATUS.md` — current phase, current gate, next concrete action.
4. The active phase's preregistration document (e.g., `preregistration/phase_a_structural_priors.md` if Phase A is current).
5. `integration/wave2_handoff.md` and `integration/notebooks_handoff.md` if the work block touches consumption of those tracks' outputs.

---

## Citation discipline declaration

Every primary-source citation in Wave P documents carries a PMID or DOI. Where a PMID could not be verified at scaffolding time, the citation is marked `(PMID lookup needed)` rather than fabricated. The user has explicit standing instructions to flag missing PMIDs as a citation-hygiene blocker (Wave 2 lost 3-4 weeks of work to a single misattribution; Wave P does not repeat that pattern).

The Wave P paper drafts will not enter manuscript prep until 100% of cited papers in the active phase preregistrations have verified PMIDs/DOIs.

---

## License and reproducibility note

The load-bearing structural-prediction tools in the revised plan are **MIT / Apache 2.0**: ESMFold (MIT, Lin et al. 2023 *Science*), OpenFold (Apache 2.0, Ahdritz et al. 2024 *Nat Methods*), and Boltz-1 (MIT, Wohlwend et al. 2024). These have no commercial-use restriction.

AlphaFold-Multimer (CC BY-NC 4.0) and RoseTTAFold-AllAtom (custom academic license) remain available as non-load-bearing cross-validation tools but no longer gate Phase A entry. The license-verification deliverable in Phase A is retained as bookkeeping only. See `infrastructure/dependencies.md` §3 for the full license matrix.
