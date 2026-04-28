# Phase V Wave 2 — Existing-Data Inventory (narrow scope)

**Scope:** items already on disk that change Path A execution plan. Not a comprehensive re-inventory of the full data ecosystem (covered in earlier sessions).

**Methodology:** searched `~/Desktop/`, `~/Downloads/`, project artifacts and references for: NeuroML files, NEURON `.mod` files, c302/Nicoletti/MetaWorm repository contents, channel kinetic parameter tables, validation datasets, supplementary materials with parameters.

---

## Headline finding

**c302 framework (Python + NeuroML + cell morphologies + connectome readers) is already cloned locally at `~/Desktop/C-Elegans/simulation/`.** This is a substantial piece of Path A's foundation already in hand — but with a critical gap: the biophysical channel kinetic definitions (`.channel.nml` files) for non-leak channels are NOT in the local copy. The local clone is at parameters_B-level (LIF/IAF + double exponential synapses) plus the parameter framework for C and D levels but missing the channel implementations they reference.

**Path A execution plan implication:** we don't need to build c302 NeuroML infrastructure from scratch. We need to: (1) acquire missing biophysical channel definitions from upstream openworm/CElegansNeuroML, and (2) separately acquire Nicoletti 2019/2024 code (not in local copy at all). The Cook 2019 connectome integration is already present via `Cook2019DataReader.py`.

---

## c302 local clone — `~/Desktop/C-Elegans/simulation/`

### What's present (useful for Path A)

| Component | Path | Content | Useful for |
|---|---|---|---|
| **c302 Python framework** | `simulation/c302_code/` | 82 Python files: c302.py main, c302_Full.py, c302_Pharyngeal.py, c302_TapWithdrawal.py, c302_Oscillator.py, c302_Muscles.py, c302_RIA.py, c302_Kato.py, c302_Social.py, c302_Syns.py, c302_AWC.py, etc. (specialized variants) | Network generation logic, parameter management, model I/O |
| **Parameter sets A through D** | `c302_code/parameters_*.py` | parameters_A (basic), parameters_B (LIF + double-exp synapses), parameters_BC1 (intermediate), parameters_C, C0, C1, C2 (HH-like channels), parameters_D, D1 (multicompartmental + HH), parameters_W2D | Multiple fidelity tiers documented; calibration knobs visible |
| **Connectome readers** | `Cook2019DataReader.py`, `OpenWormReader.py`, `ConnectomeReader.py` | Already-written parsing of Cook 2019 herm_full_edgelist.csv, OpenWorm PyOpenWorm interface | Don't re-implement connectome integration |
| **Bioparameter tracking** | `bioparameters.py` | BioParameter class with name, value, source, certainty fields | Provenance-aware parameter management (this project hasn't done this) |
| **LEMS templates** | `LEMS_c302_A_Syns.xml`, `LEMS_c302_C_Oscillator.xml` | Simulator instruction files for jNeuroML/NEURON execution | Direct execution templates, runnable if backend supports LEMS |
| **Cell morphologies** | `simulation/cells/` (607 files) | One `.cell.nml` per neuron, with full morphology + axon segments (e.g., `AVAL.cell.nml` has soma + 30+ axon segment definitions) | Multicompartmental cells already defined; could replace simulator's simple-soma scaffold for cells where morphology matters |
| **Synapse NML definitions** | `simulation/synapses/` (10 files) | Acetylcholine, Acetylcholine_Tyramine, Dopamine, FMRFamide, GABA, Glutamate, Octapamine, Serotonin, Serotonin_Acetylcholine, Serotonin_Glutamate | Per-neurotransmitter synapse kinetics (likely double-exponential conductance) |
| **Network NML files** | `simulation/networks/` | Multiple full-network instantiations: c302_A_Full.net.nml, c302_B_Full.net.nml, c302_C0/C1/C2/D_Full.net.nml, plus specialized (Pharyngeal, Oscillator, Muscles, Social, Syns, IClamp variants) | Pre-generated networks at each fidelity level |

### What's NOT present (acquisition required for Path A)

| Missing | Where to get it | Path A impact |
|---|---|---|
| **Biophysical channel `.channel.nml` files** (k_slow, k_fast, ca_boyle, CaPool, others referenced in parameters_C/D) | github.com/openworm/CElegansNeuroML or github.com/openworm/c302 NeuroML2 channels directory | High — without these, parameters_C/D can't be instantiated; parameters_B is the working ceiling locally |
| **Nicoletti 2019/2024 code** | PLOS ONE 2019 supplementary, ModelDB 267187, Mailler-Wang/Nicoletti GitHub | High — the high-fidelity HH channel implementations (SHL-1, SHK-1, EGL-19 specific, EGL-36, KQT-3, etc.) are Nicoletti's, not c302's |
| **NEURON `.mod` files** (any) | Various ModelDB sources, openworm/c302 repository | Conditional on backend choice — needed for Path 3C (NEURON backend) |
| **MetaWorm code** (Liang 2024) | Their bioRxiv supplementary or GitHub if released | TBD per Phase 2C |

### Channel-level inventory (this is the gap)

Local `simulation/channels/` directory has **only**:
- `LeakConductance.channel.nml` (passive ionChannelPassive, conductance="10pS")
- `Generic_GJ.nml` (gap junction)

`parameters_C.py` references channels named `Leak`, `k_slow`, `k_fast`, `ca_boyle`, plus a `CaPool` concentration model. These channel definitions are not in the local repo — they're imported from c302's main package or upstream NeuroML2 channels.

`parameters_D.py` (multicompartmental + HH) references the same channel set. So Path A using c302 alone caps at simplified Boyle 2012-style channels (3 channels: k_fast, k_slow, ca_boyle), not the 12+ specific worm K+/Ca channels Nicoletti 2019 implements.

**c302's biophysical fidelity ceiling is intermediate — better than IAF, less detailed than Nicoletti.** The full ion channel diversity (per the previous audit's Category A) requires Nicoletti separately.

---

## Other relevant assets on disk

### Validation datasets

| Dataset | Path | Useful for |
|---|---|---|
| **Kato 2015 calcium imaging** | `data/paper_supplements/pnas_1507110112_kato2015/sd01-16` | Whole-brain Ca²⁺ imaging in immobilized worms — validation against Atanas-style traces |
| **NIHMS2095963 supplementary tables** | `data/paper_supplements/NIHMS2095963-supplement-TableS1-S19` | Likely connectome-related (large file sizes — TableS14 alone is 69 MB); content unverified, may have parameters |
| **41467_2025_58293 supplementary** | `data/paper_supplements/41467_2025_58293_MOESM15_ESM.xlsx` | Recent (2025) Nature Communications supplement — content unverified |
| **eLife 2023 Supplementary file 2** | `data/paper_supplements/Supplementary_file_2_eLife_2023.xlsx` | Recent eLife supplement — content unverified |
| **Atanas 2023 reference worms** | `scripts/brain/artifacts/atanas_worm_0{01..10}.npz` | 10 worms × 60s; already integrated as 18-readout source |
| **Witvliet 2020 / Brittin 2021 connectome data** | searches did not find Witvliet developmental L1-adult connectomes locally; project's data tree has Cook 2019 only | Acquisition needed if developmental connectomes wanted |

### Adjacent organism data (out of immediate scope)

- **Platynereis 3D connectome 2024** at `~/Desktop/C-Elegans/data/platynereis/` — annelid worm, comparative connectomics, not Path A target

### Project's own simulator infrastructure (already known)

- LIFBrain, GradedBrain, compartmental scaffold (`scripts/brain/`)
- 18-readout neuron set (Atanas 2023 strict intersection) hardcoded
- Modulator layer with 9 modulators (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5HT, DA, OA, TA)
- Sensory transduction cascades for ASE/AWC/ASH/AFD/ALM
- v3 LIF + voltage-fix patches + h_kca patches (graded_brain_h_kca.py, compartmental_neurons_kca.py)
- 7-entry DOCUMENTED_SIGN_EXCEPTIONS registry
- Phase 0 plateau diagnostic infrastructure (voltage-clamp pattern via Brian2 @network_operation)

---

## Path A execution plan changes from this Phase 1

### Items that change the plan

1. **c302 framework is locally present** — saves substantial setup work. Don't re-implement c302 infrastructure from scratch.
2. **607 cell morphologies are present** — multicompartmental cell definitions already exist for every neuron with full axon segment morphology. Could replace project's simple-soma scaffold for cells where morphology matters (especially RIA with documented axonal compartmentalization, AVA with long descending process, AWC with compartmentalized cilium). This is a substantial asset.
3. **Cook 2019 reader is present** — we have parsing logic that's been tested in production. If the project ever wants to switch from `connectome.npz` (which loaded from Cook 2019 SI 5 XLS) to c302's edgelist-based reader, the code is ready.
4. **LEMS templates and network NML at multiple fidelity levels** — enables jNeuroML or NEURON execution paths with minimal new infrastructure.

### Items that don't change the plan

1. **Biophysical channel definitions still need acquisition** — c302's local clone is at parameters_B/C-infrastructure level but missing channel.nml definitions. Path A still requires fetching these from upstream OR pulling Nicoletti.
2. **No NEURON .mod files locally** — Path 3C (backend switch to NEURON) still requires acquisition of Nicoletti's NEURON files or c302's full upstream repo.
3. **No Nicoletti code locally** — separate acquisition step regardless.

### Adjusted Path A execution plan recommendation (preliminary, refined in Phase 4)

1. **Clone openworm/CElegansNeuroML or openworm/c302 GitHub repos to acquire missing channel.nml** files. ~1-2 hours including verification.
2. **Acquire Nicoletti 2019 code** from PLOS supplementary or ModelDB 267187. ~1-2 hours.
3. **Choose backend path** (3A/B/C/D per Phase 3 analysis) based on what's been acquired.
4. **Integration work** (covered in Phase 4 synthesis).

The local c302 clone reduces step 1's impact: we already have the framework to use those channels, so the channel acquisition is the load-bearing step, not "set up c302 from zero."

---

## Honest gaps in this Phase 1

1. **I haven't verified the c302 local clone's git status.** Is it a tracked clone of upstream, or a snapshot copied at some past date? Determines whether `git pull` updates it or whether full re-clone is needed. To verify: `cd ~/Desktop/C-Elegans/simulation/c302_code && git log` if it's a git repo.
2. **NIHMS supplementary tables haven't been opened.** TableS14 at 69 MB might contain relevant connectome or expression data; without opening, can't characterize. Low-priority for Path A specifically.
3. **The `simulation/` directory's relationship to `~/Desktop/C-Elegans/`** isn't fully traced — who put it there, when, whether it's pinned to a specific c302 release.

These don't change the headline finding (c302 framework present, channels missing). They're documentation gaps to close if Path A commitment proceeds.

