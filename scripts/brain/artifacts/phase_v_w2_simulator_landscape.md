# Phase V Wave 2 — Simulator Landscape Investigation (partial: 2A + 2B)

**This document covers Sub-phases 2A (c302/OpenWorm deep) + 2B (Nicoletti deep)** completed in Session A. Sub-phases 2C (MetaWorm/BAAIWorm) + 2D (other simulators) deferred to Session B.

---

## 2A — c302 / OpenWorm deep characterization

### Repository state

- **github.com/openworm/c302** — Active development. Latest release v0.12.0, March 31, 2026. 548 commits on master. MIT license. Continuous integration workflows (ci.yml, non_omv.yml). 12 releases tracked. Active development per release cadence.
- **github.com/openworm/CElegansNeuroML** — 735 commits, 141 stars, 53 forks. Substantial legacy framework. The README explicitly states: *"An accurate representation of the ion channels and their distributions in each neuron has not yet been attempted."* This is the OpenWorm honest disclaimer that's been present for years. Synaptic implementations also acknowledged as simplistic.
- **github.com/openworm/ChannelWorm** — **Archived August 27, 2018.** Read-only. Originally intended to build HH ion channel models from worm experimental data; abandoned partway. Models present: SHL-1, SHK-1, EGL-19, SLO-2 (4 channels). Many anticipated channels not modeled (EGL-36, KQT, NCA, BK as separate from SLO-2, HCN, TWK).

### Parameter sets (architectural fidelity tiers)

| Level | Description | Fidelity | Production-ready? |
|---|---|---|---|
| **A** | Simple two-neuron and muscle networks with current inputs (LEMS event-based) | toy | Demo only |
| **B** | LIF/IAF cells + double-exponential conductance synapses | comparable to current LIFBrain | Yes for connectivity studies |
| **C** | Single-compartment HH-style channels (Boyle 2012 muscle channels: k_fast, k_slow, ca_boyle + leak) | intermediate biophysical | Yes for muscle dynamics |
| **D** | Multicompartmental + HH (NEURON simulator only) | higher biophysical | Yes via NEURON backend |

The local clone at `~/Desktop/C-Elegans/simulation/` includes all parameter modules (parameters_A/B/BC1/C/C0/C1/C2/D/D1/W2D) but **only the leak channel definition** in the channels directory. Higher-fidelity channels (k_fast, k_slow, ca_boyle, CaPool concentration model) are referenced by parameters_C/D Python but their `.channel.nml` files are not bundled in the local copy — they live in c302's main repo at `c302/c302/NeuroML2/`.

### Channel coverage (the load-bearing gap)

Inspecting c302's master branch: `AVAL.cell.nml` (and presumably other cell files at the standard NeuroML2 level) **include only `LeakConductance.channel.nml`**. No active channels in standard cell files. The Boyle 2012 channels (k_fast, k_slow, ca_boyle) appear in muscle models specifically; ChannelWorm's 4 channels (SHL-1, SHK-1, EGL-19, SLO-2) are separate and require manual integration.

**c302 + ChannelWorm combined ceiling: ~7 channels** (3 Boyle muscle + 4 ChannelWorm neuron).

This is significantly less than Nicoletti 2024's 22 channels.

### Connectome integration

c302 has multiple connectome readers in `c302/data/`:
- `aconnectome_white_1986_A/L4/whole.csv` — White 1986 historic connectome
- `herm_full_edgelist.csv` + modified version — current standard
- `SI 5 Connectome adjacency matrices.xlsx` — Cook 2019
- `witvliet_2020_1.xlsx` through `witvliet_2020_8.xlsx` — Witvliet developmental connectomes (L1-adult stages)
- `synapse_count_matrices.xlsx`
- `wormwiring_N2U.txt`
- `Bentley_et_al_2016_expression.csv` — peptide-receptor expression

`Cook2019DataReader.py` and `OpenWormReader.py` (PyOpenWorm-based) are tested production readers.

**Witvliet 2020 connectomes are present in c302 but missing from this project's data tree.** Acquiring c302's data folder fills a connectome gap as well as providing channels.

### Synaptic implementations

10 synapse types in `simulation/synapses/`: Acetylcholine, Acetylcholine_Tyramine, Dopamine, FMRFamide, GABA, Glutamate, Octapamine, Serotonin, Serotonin_Acetylcholine, Serotonin_Glutamate. Per-neurotransmitter NeuroML synapse definitions (likely double-exponential conductance with type-specific reversal potentials).

These are not at the level of receptor binding kinetics (Markov state schemes). They're per-NT synaptic types. Useful for connectivity studies and behavioral simulation; insufficient for receptor-pharmacology-level claims.

### Cell morphologies

`simulation/cells/` has 607 `.cell.nml` files. Each contains soma + axon segments (often 30+ segments) + dendrite branches with diameters. Source likely WormAtlas / OpenWorm geometry reconstruction. Substantial asset for any multicompartmental work.

### Licensing and integration considerations

- **MIT license** (favorable for project integration)
- Active development — can pull updates without compatibility breaks (12 releases, regular cadence)
- NeuroML2 standard format — interchangeable with NeuroML-aware tools (jNeuroML, jLEMS, NEURON, Brian2 via converters)
- Documentation at https://docs.openworm.org/Projects/c302/ and the Phil Trans B 2018 paper (Gleeson et al.)

### What c302 doesn't provide

- Comprehensive biophysical channels (only ~7 with ChannelWorm)
- Receptor binding kinetics at Markov state level
- CeNGEN-coupled per-cell channel densities
- Peptide processing kinetics (modulator layer is generic)
- Plateau termination mechanism (no K_Ca + h coupling done)
- Per-cell parameters validated against Mellem 2008 voltage-clamp targets

These are gaps the project would have to fill itself, regardless of c302 import.

---

## 2B — Nicoletti deep characterization

### Publication trail

- **Nicoletti M, Loppini A, Chiodo L, Folli V, Ruocco G, Filippi S (2019)** — "Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD." PLoS ONE 14(7): e0218738. PMID 31260485. ModelDB 267187.
- **Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, et al. (2024)** — "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families." PLOS ONE 19(3): e0298105. PMID 38551921. ModelDB 2017403.
- Affiliated with Department of Engineering, Campus Bio-Medico University Rome + Center for Life Nano Science CLNS@Sapienza, IIT Rome.

### 2019 paper (foundation)

**Scope:** AWCon (chemosensory) and RMD (motor) — 2 neurons.
**Simulation environment:** XPPAUT (older modeling tool, less maintained). ModelDB 267187 download has XPPAUT files.
**Channels modeled:**
- Voltage-gated K+: SHL-1, KVS-1, SHK-1, IRK-1/3, KQT-3, EGL-36, EGL-2
- Voltage-gated Ca: EGL-19, UNC-2, CCA-1
- Ca-activated K+: SLO-1/SLO-2 (BK class) and KCNL (SK class)
- Passive: NCA (Na+ leak), generic leak

Total: 12 channels.

### 2024 paper (extension)

**Scope:** Extends to AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 — 7 additional neurons.
**Simulation environment:** **NEURON** (Python wrapper). Move from XPPAUT to NEURON is significant — NEURON is the de facto standard for biophysical neural simulation and has an active community.
**Code repository:** **github.com/ModelDBRepository/2017403** (public).
**File structure:**
- NEURON `.mod` files for **22 ionic currents** (extends 2019's 12 with EXP-2, UNC-103, KQT-1, plus the original set)
- Python files per neuron: `<NeuronName>_simulation.py` (main), `_simulation_vclamp.py` (voltage clamp protocols), `_simulation_iclamp.py` (current clamp protocols)
- Validation against published voltage- and current-clamp data
- Knockout simulations per channel (suppress each ionic current at a time)

### Architectural notes

- **Single-compartment cylindrical approximation.** Neuron modeled as a cylinder with surface area from Neuromorpho database. *"This approximation was adopted because of the limited information available on the specific distribution of the ionic channels in different regions of the neuron."* — directly addresses why compartmentalization wasn't pursued.
- All channels combined per cell with cell-specific densities; single-compartment dynamics emerge from the combined currents.
- Validates against published worm voltage-clamp data; reproduces current-clamp responses including bistable regimes (RMD), regenerative responses (AWCon), command-neuron tonic dynamics (AVA).
- **Knockout simulations** are a notable validation feature — by suppressing each ionic current, they identify mechanism per-channel for each neuron's behavior.

### License (presumed)

ModelDB entries typically permit academic use with attribution. Standard practice. Direct verification by checking the GitHub repository's LICENSE file is needed before formal commitment. For an investigation document, treating this as "MIT-equivalent academic use" is reasonable.

### Integration considerations

- **Backend question:** Nicoletti 2024 is NEURON-based. Importing into Brian2 requires equation translation (per Path 3A). Importing as-is requires NEURON backend (per Path 3C).
- **Conversion tools:** `pyNeuroML` provides NEURON → NeuroML conversion utilities. NEURON `.mod` to NeuroML2 ChannelML translation is somewhat automated. The reverse (NeuroML to NEURON) is maturer.
- **Single-compartment match to graded mode:** Nicoletti's single-compartment approximation matches GradedBrain's structure. The 22-channel library could in principle slot into GradedBrain as a higher-fidelity replacement of the current `m_Ca + h + I_KCa` patch — replacing handcrafted equations with channel-by-channel HH dynamics.
- **Cell coverage:** 7 specific neurons (plus AWCon and RMD from 2019 = 9 total) have published validated parameters. Other ~290 neurons would need calibration by analogy or via CeNGEN expression scaling.

### What Nicoletti doesn't provide

- Network-level connectivity (uses bare cells, not networks)
- Synaptic input / receptor binding kinetics (cells are tested in voltage clamp / current injection, not synaptic stimulation)
- Cell morphologies (single-compartment cylindrical approximation)
- Connectome integration
- Modulator layer
- Body / locomotion driver
- Sensory transduction beyond what's needed for AWC validation
- The 290+ non-modeled neurons

These are gaps that c302 (network, connectome, cell morphologies) and the project's own existing infrastructure (sensory transduction cascades, modulator layer) fill.

---

## Combined picture: c302 + Nicoletti complementarity

| Capability | c302 | Nicoletti 2024 | This project | Coverage if all imported |
|---|---|---|---|---|
| Cell morphologies (607 cells) | ✓ | — | — | ✓ from c302 |
| Network structure / connectome readers | ✓ | — | partial | ✓ from c302 |
| HH ion channel kinetics (12-22 channels) | partial (~7 with ChannelWorm) | ✓ (22 channels) | absent | ✓ from Nicoletti |
| Neurons with validated biophysics | small subset | 9 (AWC, RMD, AVA, AIY, RIM, VA5, VB6, VD5) | absent | ✓ partial (9 of 302) |
| Per-neuron channel densities | — | ✓ for 9 | absent | ✓ for 9 |
| Validated against voltage-clamp data | — | ✓ | partial | ✓ from Nicoletti |
| KO simulations | — | ✓ | absent | ✓ from Nicoletti |
| Receptor binding kinetics | — | — | absent | gap (project must add) |
| CeNGEN-coupled channel densities | — | — | partial (data loaded) | gap (project must add) |
| Modulator layer | — | — | ✓ | ✓ from this project |
| Sensory transduction | — | — | ✓ (5 cascades) | ✓ from this project |
| MuJoCo body integration | — | — | ✓ | ✓ from this project |
| Compartmental scaffold | — | — | ✓ (built, not deployed) | ✓ from this project (where morphology matters) |

**The combined import provides a complete biophysical foundation for the 9 neurons Nicoletti modeled.** The remaining 290+ neurons require calibration via cell type, expression scaling, or analogous parameters — gaps that exist regardless of import path.

---

## Cross-architecture comparison

### c302's biophysical fidelity vs Nicoletti's

| Dimension | c302 (with ChannelWorm) | Nicoletti 2024 |
|---|---|---|
| Channel count | ~7 | 22 |
| Channel diversity | Boyle 2012 muscle + 4 ChannelWorm | comprehensive worm-specific |
| Cell coverage | 607 morphologies, mostly leak-only | 9 fully validated |
| Validation against voltage-clamp data | limited | comprehensive |
| Knockout simulation | not available | per-channel KO |
| Active development | yes (March 2026) | yes (2024 paper, code public) |
| Backend | NeuroML2 (jNeuroML, NEURON, others) | NEURON-only |

### What this means for Path A

The **two-source import is necessary**, not optional. c302 alone leaves channel diversity insufficient for serious mechanistic claims. Nicoletti alone leaves network structure missing. Both together provide the foundation; the project's own work fills the remaining 290+ neurons via expression scaling and the application-layer infrastructure (modulator, sensory, body, scenarios, FSM/classifier).

### Alternative: BAAIWorm/MetaWorm (Liang 2024)

Search results surfaced **BAAIWorm** as an integrative model (Liang 2024, Nature Computational Science): "biophysically detailed neuronal model capable of replicating the zigzag movement... digitally reconstructed models of five representative neurons: AWC, AIY, AVA, RIM and VD5."

These five neurons overlap exactly with Nicoletti 2024's set. BAAIWorm/MetaWorm likely uses Nicoletti's models (or a derivative). Sub-phase 2C (deferred to Session B) will characterize whether BAAIWorm provides additional value beyond Nicoletti, or whether it's an integration project that ports Nicoletti's models into a fuller framework.

---

## Acquisition checklist (Path A logistics)

### What we already have locally

- c302 framework snapshot (July 2025) at `~/Desktop/C-Elegans/simulation/c302_code/`
- 607 cell morphologies at `~/Desktop/C-Elegans/simulation/cells/`
- 10 synapse NML at `~/Desktop/C-Elegans/simulation/synapses/`
- Multiple network NML at `~/Desktop/C-Elegans/simulation/networks/`
- Cook 2019 connectome reader at `~/Desktop/C-Elegans/simulation/c302_code/Cook2019DataReader.py`

### What we need to acquire

1. **Updated c302 from upstream** (`git clone github.com/openworm/c302`) — newer than July 2025 snapshot, includes any recent updates. Brings `c302/data/` files including witvliet_2020 connectomes, Bentley 2016 expression. ~30 min to clone + verify.
2. **ChannelWorm models** (`git clone github.com/openworm/channelworm`) — archived but accessible. SHL-1, SHK-1, EGL-19, SLO-2 in NeuroML2 format. ~5 min.
3. **Nicoletti 2024 ModelDB 2017403** (`git clone github.com/ModelDBRepository/2017403`) — NEURON .mod files for 22 channels, Python wrappers for 7 neurons, voltage/current clamp protocols. ~10 min.
4. **(Optional) Nicoletti 2019 ModelDB 267187** — XPPAUT-based; likely superseded by 2024 work. Acquire if specific AWC/RMD validation needed.

Total acquisition cost: ~1 hour. **Storage:** Nicoletti 2024 + ChannelWorm + c302 fresh clone is probably <500 MB total.

### License status (preliminary; verify before production commitment)

- c302: MIT (verified)
- ChannelWorm: presumed MIT/Apache (OpenWorm convention; verify from LICENSE file)
- Nicoletti 2024: presumed academic-use-with-attribution per ModelDB convention; verify from GitHub

All three should be license-compatible with this project's likely public release. **Formal license verification is a Wave 2 implementation prerequisite, not a Phase 2 finding.**

---

## Honest gaps from Sub-phases 2A + 2B

1. **Haven't actually downloaded any of the three packages.** Phase is investigation, not setup. Acquisition is Wave 2 implementation work.
2. **Haven't verified Nicoletti's 22 channels work as cited** without running the code. Trust ModelDB's review process for now.
3. **Haven't characterized BAAIWorm/MetaWorm** beyond surfacing its existence. That's Session B's Sub-phase 2C.
4. **Haven't characterized other landscape simulators** (PyNN, jNeuroML standalone, NetPyNE, Bionet, etc.). Session B's Sub-phase 2D.
5. **Backend architecture analysis pending.** Phase 3 in Session B.

