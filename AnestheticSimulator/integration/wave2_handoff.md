# Wave 2 → Wave P handoff

**Status:** SCAFFOLDED. The handoff protocol is documented; consumption hasn't started.

---

## What Wave P consumes from Wave 2

Wave 2 owns per-channel Brian2 implementations. Wave P consumes them as the baseline channel set, applies anesthetic-induced kinetic shifts on top, and runs network simulation. Wave P does **not** modify any Wave 2 file.

### File-level handoff (read-only)

Wave 2 channel implementations live at:

```
/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/
    egl19.py
    shk1.py
    shl1.py
    nca.py
    kqt3.py
    slo1_iso.py
    slo1_egl19_coupled.py
```

Each file is a Python module that exposes:

- A `<CHANNEL>_eqs` Brian2 equation string.
- A `<CHANNEL>_namespace` dict of parameter values.
- Optional `validate_<CHANNEL>(brian2_neurongroup)` smoke-test.

Wave P imports these modules **read-only**:

```python
import sys
sys.path.insert(0, "/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels")
from egl19 import egl19_eqs, egl19_namespace
```

Wave P then constructs a Brian2 NeuronGroup with the consumed channel equations and applies its own anesthetic-shift overlay at runtime. Wave P does not modify the consumed equations; it modifies the **parameter values in the namespace dict** before instantiation.

### Anesthetic-shift overlay protocol

For each channel × each anesthetic, Wave P applies a Phase D-derived multiplier:

```python
from copy import deepcopy
ns = deepcopy(egl19_namespace)
ns["g_max"] = ns["g_max"] * (1 - 0.3 * occupancy_EGL19_halothane)  # example block
neurons = NeuronGroup(N, eqs=egl19_eqs, namespace=ns, ...)
```

The overlay protocol is implemented in `src/phase_g_network_runs.py`'s `apply_kinetic_shifts()` function (scaffold).

---

## Wave 2 status as of Wave P kickoff (2026-04-27)

Per the most recent Wave 2 handoff (`scripts/brain/artifacts/handoffs/session_3_handoff_2026-04-26.md`):

- **7 essential channels validated** at < 1% divergence from Nicoletti 2024 NEURON reference:
  - EGL-19 (L-type Ca)
  - SHK-1 (Kv1)
  - SHL-1 (Kv4)
  - NCA (Na leak)
  - KQT-3 (M-type)
  - SLO-1 isolated (BK)
  - SLO-1 + EGL-19 coupled

- **In-flight (not yet shipped):**
  - IRK (inwardly rectifying K)
  - UNC-103 (Erg-like K)

AVA cell uses **5 channels: IRK + LEAK + EGL-19 + NCA + UNC-103**. Of these, EGL-19 and NCA are Wave 2-validated; LEAK is trivial; IRK and UNC-103 are not yet translated.

### Wave P workaround for missing IRK + UNC-103

Until Wave 2 ships IRK and UNC-103, Wave P uses **Nicoletti's NEURON reference** for AVA simulations involving these channels:

- Wave P imports the `.mod` file via Nicoletti's NEURON setup at `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`.
- For Phase G runs, AVA simulations route through a hybrid Brian2 (3 channels) + NEURON (2 channels) cell model.
- This is acceptable for Phase G because Phase G's evaluation point is network-level EC50, not per-channel kinetics.
- When Wave 2 ships IRK and UNC-103, Wave P switches AVA to pure-Brian2.

The hybrid path is documented in `src/phase_g_network_runs.py` as `--use-hybrid-AVA` flag (scaffold).

---

## Handoff version pinning

Wave P pins to a specific Wave 2 git commit at the start of each Wave P phase. The pinned commit is logged in:

- `STATUS.md` per Wave P phase entry.
- `artifacts/<phase>/wave2_commit.txt` per phase output.

This guards against Wave 2 making channel-translation changes mid-Wave-P that would invalidate Wave P's runs.

When Wave 2 ships an updated channel (e.g., EGL-19 v2 with calibration tweak), Wave P phases that already executed remain pinned to the original commit; new Wave P phases pick up the latest.

---

## Reverse handoff: Wave P → Wave 2

Wave P does not feed back into Wave 2's channel translations. Wave 2 owns its own validation against Nicoletti; anesthetic-perturbed validation is a Wave P concern, not a Wave 2 concern.

Exception: if Wave P discovers a Wave 2 channel implementation has a bug (e.g., wrong unit conversion), Wave P opens an issue with Wave 2 and **does not fix the bug locally**. Bug fixes belong in Wave 2's repo.

---

## Imports cheat sheet

```python
# Wave P imports Wave 2 channels (read-only)
import sys
WAVE2_PATH = "/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2"
sys.path.insert(0, WAVE2_PATH)
sys.path.insert(0, f"{WAVE2_PATH}/channels")

# Use case 1: import a single channel
from egl19 import egl19_eqs, egl19_namespace

# Use case 2: import the full validation harness
from voltage_clamp_harness import voltage_clamp_compare
from plateau_harness import current_clamp_plateau

# Use case 3: import NEURONReference for hybrid path
from neuron_reference import NEURONReference
nr = NEURONReference("/home/rohit/Desktop/C-Elegans/simulation/upstream/nicoletti_2024",
                     mech_compiled=True)
nr.run_voltage_clamp("AVAL", v_holds=[-80, -40, 0])
```

---

## Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| Wave 2 file moved / renamed | ImportError at Wave P startup | Update path in `src/phase_g_network_runs.py`; pin to last-known-good commit |
| Wave 2 namespace dict format changes | KeyError on parameter access | Add adapter layer; surface to Wave 2 maintainer |
| Wave 2 ships a regression that breaks validation | Phase G smoke-test fails | Revert Wave P pin; surface bug to Wave 2 |
| IRK / UNC-103 ship with bugs | Wave P AVA simulations diverge from baseline | Use NEURON reference (hybrid path) until Wave 2 fixes |

---

## Coordination protocol

Wave P and Wave 2 are owned by the same user. Coordination:

- Wave P opens an issue / writes a note in `STATUS.md` if Wave 2 needs to ship something for Wave P to proceed.
- Wave 2's pace is independent of Wave P. Wave P should not block on Wave 2 for non-essential channels.
- If a Wave P phase is blocked on a Wave 2 deliverable, Wave P documents the dependency in the relevant `phase_*_completion.md`.

The first known dependency: **Wave 2 IRK and UNC-103 translations** are needed for Phase G AVA simulations. Until those ship, Wave P uses the NEURON reference path.
