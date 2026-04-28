# Notebook pipeline → Wave P handoff

**Status:** SCAFFOLDED. Consumption protocol documented; data not yet loaded into Wave P.

---

## What Wave P consumes from the notebook pipeline

The notebook pipeline at `/home/rohit/Desktop/C-Elegans/New Notebooks/` runs 56 notebooks producing structural connectomics priors from Witvliet 2020 (222 neurons), CeNGEN expression, Bentley 2016 peptide-receptor mapping, Cook 2019, and Brittin contact matrix.

Wave P consumes a specific subset of artifacts as **biological priors** for the 290 non-Nicoletti neurons. (Wave 2 covers 9 Nicoletti-validated neurons; Wave P covers the rest via the priors below.)

### File-level consumption

| Artifact | Path | Wave P use |
|---|---|---|
| `connectome_adult.npz` | `New Notebooks/data_derived/connectome_adult.npz` | Phase G connectivity matrix |
| `expression_tpm.npz` | `New Notebooks/data_derived/expression_tpm.npz` | Per-neuron channel-density scaling (Phase G) |
| `nb12_peptide_adjacency.npz` | `New Notebooks/data_derived/nb12_peptide_adjacency.npz` | A_peptide modulator matrix (Phase G) |
| `nb28_neuron_profile.csv` | `New Notebooks/data_derived/nb28_neuron_profile.csv` | 8 multiplex archetypes (Phase D / G) |
| `nb29_channel_rulebook.csv` | `New Notebooks/data_derived/nb29_channel_rulebook.csv` | Per-neuron channel rules |
| `nb44_*` | `New Notebooks/data_derived/nb44_*` | Per-neuron functional priors |

Wave P imports these artifacts **read-only**:

```python
import numpy as np
import pandas as pd

NB_DATA = "/home/rohit/Desktop/C-Elegans/New Notebooks/data_derived"
connectome = np.load(f"{NB_DATA}/connectome_adult.npz")
expression = np.load(f"{NB_DATA}/expression_tpm.npz")
peptide_adj = np.load(f"{NB_DATA}/nb12_peptide_adjacency.npz")
neuron_profile = pd.read_csv(f"{NB_DATA}/nb28_neuron_profile.csv")
channel_rulebook = pd.read_csv(f"{NB_DATA}/nb29_channel_rulebook.csv")
```

---

## Per-artifact use cases

### `connectome_adult.npz` — synaptic connectivity

The 222-neuron Witvliet adjacency matrix. Wave P uses this directly as the network connectivity for Phase G.

The simulator extends to 300 neurons (Witvliet 222 + supplementary Cook 2019 cells); the additional 78 cells come from Wave 2 / production-simulator code, which already integrates Cook for the missing cells. Wave P simply reuses what's already in production-simulator land.

### `expression_tpm.npz` — per-neuron channel-density scaling

CeNGEN per-neuron transcript counts. Wave P uses these to scale per-neuron channel densities:

```python
# For each neuron and each channel, the effective g_max scales with TPM
g_max_neuron = g_max_default * (TPM_neuron_channel / TPM_reference) ** alpha
```

where `alpha ~ 0.5` is a phenomenological exponent (sub-linear scaling per Wave 2 / production-simulator convention). Wave P uses the same alpha as the production simulator.

This is the primary mechanism for handling the 290 non-Nicoletti neurons: the 7 Wave 2 channels are scaled by their CeNGEN TPM in each neuron, producing a CeNGEN-coupled channel-density profile per neuron.

### `nb12_peptide_adjacency.npz` — A_peptide modulator matrix

Bentley 2016 ligand-receptor pairs aggregated into a per-neuron-pair peptide-modulation strength matrix. Wave P uses this for the modulator layer in Phase G:

- Neurons release peptides at firing-rate-dependent rates.
- Receiver neurons are modulated by accumulated peptide concentration (slow time constant).
- Anesthetic effect on peptide processing (Tier 2 targets EGL-3, EGL-21) modifies these rates.

### `nb28_neuron_profile.csv` — 8 multiplex archetypes

Per-neuron archetype assignment into 8 categories (Wired_receiver, Multiplex_integrator, etc.). Wave P uses this for **archetype-level anesthetic-sensitivity prediction**:

- Wired_receiver and Multiplex_integrator archetypes have high SNARE load (chemical synapse-heavy).
- Predicted to be more SNARE-anesthesia-sensitive at clinical concentrations.
- Phase G aggregates per-archetype loss-of-locomotion and confirms the predicted ranking.

### `nb29_channel_rulebook.csv` — channel-density rules

Per-neuron rules for which channels are present (boolean) and at what relative density (continuous). Wave P uses this to instantiate the 7-channel subset per neuron:

```python
# For neuron AVA: channels = {IRK, LEAK, EGL-19, NCA, UNC-103}
# For neuron AIY: channels = {SHL-1, KQT-3, NCA, EGL-19, ...}
```

The rulebook is consumed in Phase G's per-neuron configuration step. Anesthetic effects then apply per-channel.

### `nb44_*` — per-neuron functional priors

Phase G applies per-neuron functional priors (resting V, baseline firing rate, etc.) from the nb44 outputs. Anesthetic effects shift these priors via the metabolic layer (Phase F) and Markov synapse module (Phase E).

---

## Version pinning

Like the Wave 2 handoff, Wave P pins the notebook pipeline to a specific git commit at the start of each Wave P phase. Logged in:

- `STATUS.md` per Wave P phase entry.
- `artifacts/<phase>/notebook_pipeline_commit.txt` per phase output.

The notebook pipeline is in active development (per `~/.claude/projects/.../MEMORY.md`); Wave P should not assume the artifacts are stable across weeks. Pinning is mandatory.

---

## Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| Notebook artifact path moves | FileNotFoundError | Update path in `src/phase_g_network_runs.py`; check notebook pipeline's data_derived/ for renames |
| Notebook artifact format changes | KeyError on np.load access | Adapter layer; document the version mismatch |
| Notebook pipeline ships a regression (e.g., wrong connectome) | Phase G smoke-test fails | Revert pin; surface to notebook pipeline maintainer |
| Connectome includes neuron not in Wave 2 | Channel dispatch fails | Use Wave 2's existing fallback (LIF-only neuron); document |

---

## Reverse handoff

Wave P does not feed back into the notebook pipeline. The notebook pipeline produces structural priors; Wave P consumes them. Discovery of a notebook pipeline bug is documented and surfaced; bug fixes belong in the notebook pipeline's repo.

Exception: Wave P's Phase G predictions about per-archetype anesthetic sensitivity may be of interest to the notebook pipeline's archetype-validation work. If so, Wave P documents the connection in `papers/wave_p_paper_outline.md` for later cross-citation.
