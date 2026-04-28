# Wave P — Directory structure

**Location:** `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/`

```
AnestheticSimulator/
|
|-- README.md                              # project overview (read first)
|-- WAVE_P_ARCHITECTURAL_PLAN.md           # master plan (read second)
|-- STATUS.md                              # current phase, gates, next action
|-- SETUP_COMPLETE.md                      # kickoff inventory + week-1 tasks
|-- .gitignore                             # excludes data, MD trajectories, runs
|
|-- preregistration/                       # one .md per phase
|   |-- phase_a_structural_priors.md
|   |-- phase_b_binding_pose.md
|   |-- phase_c_occupancy_matrix.md        # GATE C.1 LOAD-BEARING
|   |-- phase_d_kinetic_shifts.md
|   |-- phase_e_markov_synapses.md
|   |-- phase_f_metabolic_layer.md
|   |-- phase_g_network_perturbation.md
|   |-- phase_h_empirical_validation.md
|   |-- phase_i_inverse_design.md          # stretch (deferred)
|   `-- phase_j_network_signature.md       # stretch (deferred)
|
|-- infrastructure/
|   |-- dependencies.md                    # tool inventory + versions + licenses
|   |-- compute_budget.md                  # per-phase compute / Colab / cloud
|   |-- directory_structure.md             # this file
|   |-- setup_local.sh                     # local environment setup script
|   `-- setup_colab.md                     # Colab pipeline notes
|
|-- targets/
|   |-- tier1_targets.csv                  # 25 Tier-1 targets with WB IDs, oligomer, AF DB URL, mammalian PDB
|   |-- tier2_targets.csv                  # ~25 Tier-2 (deferred panel)
|   |-- target_panel_rationale.md          # paper-by-paper Tier-1 justification
|   `-- pocket_residues_homolog.csv        # Phase A populates with per-target pocket residues
|
|-- anesthetics/
|   |-- anesthetic_panel.csv               # name, smiles, MW, log P, Kp, EC50, anchor papers
|   |-- prepare_ligands.py                 # RDKit prep skeleton (3D, AM1-BCC, .pdbqt + .sdf)
|   `-- anesthetic_smiles/                 # one .smi per anesthetic
|       |-- halothane.smi
|       |-- isoflurane.smi
|       |-- sevoflurane.smi
|       |-- propofol.smi
|       |-- ketamine.smi
|       `-- etomidate.smi
|
|-- src/                                    # implementation skeletons
|   |-- phase_a_structures.py
|   |-- phase_b_dock.py
|   |-- phase_c_occupancy.py
|   |-- phase_d_kinetic_shifts.py
|   |-- phase_e_markov_synapses.py
|   |-- phase_f_metabolic.py
|   |-- phase_g_network_runs.py
|   |-- phase_h_validation.py
|   |-- phase_i_inverse_jax.py
|   `-- phase_j_signature.py
|
|-- validation/
|   |-- empirical_anchors.md               # per-paper validation matrix; CITATION HYGIENE LIVES HERE
|   `-- mutant_panel.csv                   # every mutant the simulator predicts
|
|-- integration/
|   |-- wave2_handoff.md                   # how Wave P consumes Wave 2 outputs
|   |-- notebooks_handoff.md               # how Wave P consumes notebook pipeline outputs
|   `-- production_simulator_handoff.md    # how a deployed Wave P plugs into LIFBrain
|
|-- risk/
|   `-- risk_register.md                   # 8-12 specific risks with mitigations
|
|-- timeline/
|   `-- timeline.md                        # 6-month month-by-month breakdown
|
|-- papers/
|   `-- wave_p_paper_outline.md            # paper outline; target venues
|
`-- artifacts/                              # phase outputs (mostly NOT git-tracked)
    |-- structures/                         # Phase A
    |   `-- README.md
    |-- binding/                            # Phase B
    |   `-- README.md
    |-- occupancy/                          # Phase C
    |   `-- README.md
    |-- kinetics/                           # Phase D (NOT git-tracked: trajectories)
    |   `-- README.md
    |-- markov/                             # Phase E
    |   `-- README.md
    |-- metabolic/                          # Phase F
    |   `-- README.md
    |-- runs/                               # Phase G (NOT git-tracked: NPZ runs)
    |   `-- README.md
    |-- validation/                         # Phase H
    |   `-- README.md
    |-- traces/                             # MD trajectories (NOT git-tracked)
    |   `-- README.md
    `-- logs/                               # per-phase log files (NOT git-tracked)
        `-- README.md
```

---

## Per-directory purpose

### `preregistration/`

The master planning document set. Each phase's preregistration is a self-contained, falsifiable plan with goal, method, compute, success criteria, halting rules, output deliverables. Modifications during execution require an amendment block at the bottom of the file with date and rationale.

**Read order:** A → B → C → D → E → F → G → H → I → J. Phase C is load-bearing (Gate C.1 is the first program-level falsifiability checkpoint).

### `infrastructure/`

Operational documents about how Wave P runs. Dependencies, compute, setup scripts.

### `targets/`

The target panel CSVs and per-target rationale. `tier1_targets.csv` is the entry point. `pocket_residues_homolog.csv` is populated by Phase A as predictions complete.

### `anesthetics/`

The anesthetic panel and ligand-preparation pipeline. Six anesthetics: halothane, isoflurane, sevoflurane, propofol, ketamine (NMDA control), etomidate (GABA-A specificity control).

### `src/`

Implementation skeletons. Each `phase_<letter>_*.py` is a runnable stub with substantial docstring, argparse CLI, logging, and gate-criteria assertions. At kickoff, each prints "PHASE X SCAFFOLD — implementation pending" when run.

### `validation/`

The per-paper empirical-anchor matrix lives here. **Citation hygiene's load-bearing artifact.** Every quantitative biological claim in any Wave P document maps to a row in `empirical_anchors.md`.

### `integration/`

How Wave P interacts with Wave 2, the notebook pipeline, and the production simulator. Documented file paths and protocols.

### `risk/`

Wave P risk register. Updated per work block.

### `timeline/`

6-month plan, month by month.

### `papers/`

Paper outline. Target venues: Cell Systems / Neuron / eLife / PLOS Computational Biology.

### `artifacts/`

Phase outputs. Most subdirectories are not git-tracked (large NPZ / MD trajectories). Each has a placeholder `README.md` describing what will go there.

---

## File-naming conventions

- Markdown documents: `lower_snake_case.md` (or descriptive UPPERCASE.md for top-level).
- Python skeletons: `phase_<letter>_<short_name>.py`.
- CSVs: `lower_snake_case.csv` with header row.
- NPZ artifacts: `<descriptor>_<config>.npz`. Phase G uses `<anesthetic>_<dose>_<genotype>_<scenario>_<seed>.npz`.
- Logs: `phase_<letter>_<DATE>.log` where DATE is `YYYYMMDD`.
- Reports: `phase_<letter>_completion.md` for end-of-block; `gate_<letter><n>_evaluation.md` for gate-evaluation artifacts.

---

## Path conventions

All paths in Wave P documents are **absolute**, rooted at `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/`. Relative paths are not used.

Paths external to Wave P (Wave 2, notebook pipeline) are also absolute:

- Wave 2 channels: `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/`.
- Notebook pipeline data: `/home/rohit/Desktop/C-Elegans/New Notebooks/data_derived/`.
