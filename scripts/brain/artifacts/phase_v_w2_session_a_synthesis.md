# Phase V Wave 2 — Session A Interim Synthesis

**Scope:** Phase 1 (existing-data inventory) + Sub-phase 2A (c302/OpenWorm deep) + Sub-phase 2B (Nicoletti deep). Sub-phase 2C (MetaWorm/BAAIWorm), Sub-phase 2D (other simulators), Phase 3 (backend architecture analysis), and Phase 4 (Wave 2 plan) deferred to Session B.

---

## Three load-bearing findings

### 1. Local c302 clone reduces Path A setup cost ~50%

`~/Desktop/C-Elegans/simulation/c302_code/` is a complete c302 Python framework snapshot from July 2025 with 607 cell morphologies, 10 synapse NMLs, multiple network NMLs at fidelity levels A through D, and Cook 2019 connectome reader. **The framework is in hand; only channel definitions and updates are needed from upstream.**

### 2. Nicoletti 2024 is more comprehensive than the previous audit assumed

The published code at `github.com/ModelDBRepository/2017403` provides:
- **NEURON .mod files for 22 ionic currents** (extends 2019's 12 with EXP-2, UNC-103, KQT-1)
- **Python simulation wrappers for 9 neurons** (AWC + RMD from 2019; AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 from 2024)
- **Voltage-clamp + current-clamp protocols implemented per neuron**
- **Knockout simulations per channel** (mechanism identification)
- **Validated against published electrophysiology**

This is genuinely a more complete biophysical channel library than I'd characterized in Wave 1's audit. Combined with c302's framework, the import provides a full foundation for 9 of 302 neurons; the other 290+ require calibration via expression scaling.

### 3. c302 + ChannelWorm alone caps biophysical fidelity at ~7 channels

OpenWorm's biophysical channel coverage is bounded:
- c302's standard cell files include only LeakConductance
- Boyle 2012 muscle channels (k_fast, k_slow, ca_boyle) — 3 channels
- ChannelWorm (archived August 2018, no longer maintained): SHL-1, SHK-1, EGL-19, SLO-2 — 4 channels

Total ~7 channels. Significantly less than Nicoletti's 22. **Without Nicoletti, Path A's biophysical ceiling is intermediate.** Nicoletti is the higher-leverage acquisition; c302 provides infrastructure (cell morphologies, network structure, connectome integration); both together are necessary.

---

## Path A acquisition logistics (refined from Phase 1+2)

| Step | Action | Cost |
|---|---|---|
| 1 | Clone fresh c302 from `github.com/openworm/c302` (newer than local July 2025 snapshot) | ~30 min |
| 2 | Clone ChannelWorm (archived but accessible) — adds SHL-1, SHK-1, EGL-19, SLO-2 in NeuroML2 | ~5 min |
| 3 | Clone Nicoletti 2024 from `github.com/ModelDBRepository/2017403` — 22 channels NEURON .mod | ~10 min |
| 4 | Verify licenses (c302 MIT confirmed; ChannelWorm + Nicoletti presumed academic-use-with-attribution) | ~15 min |
| 5 | Validate downloaded code structure matches publications | ~30 min |

Total: ~1.5 hours of acquisition work. Storage <500 MB.

---

## Strategic implications for Path A execution

### What Path A buys us

- **For the 9 Nicoletti-modeled neurons** (AWC, RMD, AVAL, AVAR, AIY, RIM, VA5, VB6, VD5): full biophysical channel-level dynamics with worm-validated parameters.
- **For the other 290+ neurons:** infrastructure (morphologies from c302, connectome from c302's Cook 2019 reader); biophysical parameters require calibration via cell type or CeNGEN expression scaling.
- **Network-level NeuroML structure:** importable via c302; replaces project's `connectome.npz` if backend is changed.
- **Existing voltage-clamp + current-clamp protocols** for the 9 neurons: Nicoletti 2024 includes them, can validate any backend's implementation against published traces.
- **Per-channel KO simulations:** Nicoletti 2024 provides them; addresses Mellem-style mechanism dissection at cellular level.

### What Path A doesn't buy us

- **Receptor binding kinetics** at Markov state level — not in c302 (per-NT generic synapses) or Nicoletti (focuses on intrinsic channels, not synaptic input).
- **CeNGEN-coupled per-cell channel densities** — neither c302 nor Nicoletti uses CeNGEN.
- **Modulator layer / peptide processing** — entirely the project's own work.
- **Sensory transduction cascades** — entirely the project's own work (5 cascades in `sensory_transduction.py`).
- **MuJoCo body integration** — entirely the project's own work.
- **Anesthesia-specific allosteric framework** — outside scope of all three packages.
- **Per-cell biophysics for the 290+ non-Nicoletti neurons** — calibration burden falls on this project.

This delimitation matches the previous audit's framing: the project's distinguishing potential is integration above the channel layer. Path A acquires the channel layer; the project still owns the layers above.

---

## What Session B needs to address

### Sub-phase 2C: MetaWorm/BAAIWorm

Phase 2B surfaced that **BAAIWorm** (Liang et al. 2024, Nature Computational Science) is the integrative simulator Sub-phase 2C is meant to characterize. It "replicates zigzag movement" with biophysical models of 5 neurons (AWC, AIY, AVA, RIM, VD5) — exactly overlapping Nicoletti's set. Likely uses Nicoletti's models or derivatives. Session B should determine: (a) is BAAIWorm's code public? (b) does it add value beyond Nicoletti at the cellular level, or is it primarily an integration framework? (c) what architectural patterns are worth borrowing for our integration?

### Sub-phase 2D: other simulators

Survey-level scan for other relevant simulators: PyNN, NetPyNE, Bionet, jNeuroML standalone, FlyWire/Drosophila biophysical models, mammalian neuron simulators with novel architectural approaches. Bounded scope; the deep dives (c302, Nicoletti, MetaWorm) are the load-bearing characterizations.

### Phase 3: backend architecture analysis

This is the load-bearing decision Session B needs to make. The four paths (3A: Brian2 + parameter import, 3B: multi-framework, 3C: NEURON backend, 3D: NeuroML2-based) need cost-benefit assessment with concrete what-breaks analysis for each. Phase 1 + 2 findings inform this:

- **3A (Brian2 + import):** translate Nicoletti's 22 NEURON .mod files into Brian2 equation strings. ~weeks of work but preserves all existing infrastructure (LIFBrain/GradedBrain integration, MuJoCo, scenario JSON, dashboard, FSM/classifier). Brian2 has limited support for some NEURON features (e.g., active conductances with complex gating).
- **3B (multi-framework):** run NEURON + Brian2 in parallel coordinated by ClosedLoopEnv. NEURON for 9 Nicoletti-modeled cells, Brian2 for the 290+ others. Increased complexity, sync at MuJoCo's body cadence.
- **3C (NEURON backend):** rewrite simulator brain layer in NEURON. Leverage NEURON's mature biophysics. Substantial code surface change. Brian2 expertise less relevant.
- **3D (NeuroML2-based):** use jNeuroML or similar declarative-model-driven simulator. Most portable but unfamiliar; less mature for biophysical work.

Each path's break analysis requires careful examination of the simulator's coupling points (modulator layer, FSM, classifier, sensory cascades, scenario pipeline, dashboard, MuJoCo coupling). Session B's Phase 3 will do this work explicitly.

### Phase 4: Wave 2 architectural plan

After Sub-phase 2C, 2D, and Phase 3, the synthesis produces:
- Recommended Wave 2 primary path with sequencing and gates
- Fallback paths and trigger criteria
- Risk assessment
- Resource implications
- Honest "what import achieves vs what gaps remain" framing
- Connection to paper trajectories (paper 2 behavioral, paper 3 mechanistic, methodology paper)

---

## Recommended decisions before Session B

If the user accepts the Phase 1+2 findings, the following decisions can be made now (don't require Session B):

1. **Acquire c302, ChannelWorm, Nicoletti 2024 from upstream.** ~1.5 hours. Doesn't commit to Path A execution but unblocks Session B's Phase 3 backend analysis (Phase 3 needs to inspect Nicoletti's actual .mod files to assess translation effort).
2. **Verify license terms** for ChannelWorm and Nicoletti 2024 from their GitHub LICENSE files. Standard practice.
3. **Decide whether BAAIWorm/MetaWorm investigation is worth the depth Session A originally planned for it.** Phase 2B suggests BAAIWorm is likely Nicoletti-derived; if so, the deep dive may produce limited new info. Session B can scope accordingly.

---

## Honest gaps from Session A

1. **Haven't downloaded any packages.** Phase is investigation only. Acquisition is Wave 2 implementation work, gated on architectural commitment.
2. **License verification is preliminary.** Formal verification before production commitment.
3. **Sub-phase 2C/2D not done.** Session B coverage.
4. **Phase 3 backend analysis not done.** Session B coverage. **This is the load-bearing remaining decision.**
5. **Phase 4 synthesis not done.** Session B coverage.
6. **Haven't tested whether Nicoletti's NEURON .mod files actually compile and run** — investigation is read-only per prompt; this is engineering work.

---

## Connection to the project's strategic positioning

The previous audit's recommendation: *"the simulator's distinguishing potential is integration above the channel layer."* Session A's findings sharpen that:

- **Integration above the channel layer** = modulator + sensory transduction + CeNGEN-coupled densities + behavioral/network closure + body coupling.
- **Channel layer itself = importable** from Nicoletti (channels) + c302 (infrastructure) + project's existing compartmental scaffold (where morphology matters beyond Nicoletti's single-compartment approximation).

This delimitation is **stable and consistent** between the previous audit and Session A's deeper investigation. Path A is the right direction; the question Session B answers is execution mechanics.

