# Phase V Wave 2 — Backend Architecture Analysis (Phase 3)

**Goal:** four-path cost-benefit for backend strategy, with concrete what-breaks analysis. Inputs: Phase 1 (existing-data inventory), Phase 2A-D (c302, Nicoletti, BAAIWorm, broader landscape).

---

## Current simulator's coupling surface

The simulator's blast radius for backend change is bounded by these coupling points:

### Brian2-specific code

| File | Brian2 dependency |
|---|---|
| `lif_brain.py` | NeuronGroup, Synapses, StateMonitor, Network, network_operation, equation strings, units (mV, ms, pA, nS), `_brian2_seed` |
| `graded_brain.py` | Same as above + Poisson groups for sensory injection |
| `graded_brain_h_kca.py` | Same; the Wave 1 sandbox |
| `compartmental_neurons.py` | NeuronGroup with multi-compartment state variables |
| `compartmental_neurons_kca.py` | Same |
| `modulation_layer.py` | network_operation rewrite of `neurons.I_ext` periodically |
| `phase0_param_lhs_runner.py` and other diagnostics | Brian2 unit imports |

### Backend-agnostic code (preserved across any backend)

| File | Coupling to brain |
|---|---|
| `closed_loop_env.py` | calls `brain.run()`, reads `brain.spikes.t/.i`, `brain.output_rates()`, `brain.idx`, `brain.names`, `brain.N`; calls `brain.ablate()`, `brain.set_proprioception()`, `brain.set_sensory_rate()`, `brain.inject_poisson()` |
| `sensory_transduction.py` | runs external ODEs, injects via `brain.set_sensory_rate()` (interface call, not Brian2-internal) |
| `modulation_layer.py` (peptide release/decay logic only) | scales modulator concentrations from firing rates |
| `behavioral_fsm.py`, `activity_fsm.py` | reads firing rates from brain output |
| `neural_classifier_bank.py` | reads firing rates |
| `phase0_audit.py`, `phase1_*`, `phase3_*`, `phase6_*` | audit harness; backend-agnostic if brain interface preserved |
| MuJoCo body integration | uses `mujoco.MjModel` directly; **backend-independent** |
| Scenario JSON pipeline | reads from ClosedLoopEnv output |
| Dashboard | reads scenario JSONs |

**The brain interface to ClosedLoopEnv is small and clean.** ~10 methods/properties. Backend swap is feasible if interface is preserved.

### Lines of Brian2-coupled code

`grep -c "from brian2"` across the simulator scripts shows ~25 import sites across ~15 files. Most files have one import block. The Brian2-specific lines (equation strings, NeuronGroup construction, StateMonitor wiring, network_operation hooks) are concentrated in the brain class implementations and the Phase 0 diagnostic scripts.

**Estimated total Brian2-specific code: ~2000-3000 lines.** Substantial but bounded.

---

## Path 3A — Brian2 + parameter import

Translate Nicoletti's NEURON `.mod` files into Brian2 equation form. No backend change. Per-channel additive.

### Translation pattern (verified from sample inspection)

NMODL `egl19.mod` structure:
```
NEURON { SUFFIX egl19 USEION ca READ eca WRITE ica RANGE gbar }
PARAMETER { gbar=1.55 (S/cm2), va_egl19=5.6 (mV), ka_egl19=7.50 (mV), ... }
ASSIGNED { ica, minf, hinf, mtau, htau }
STATE { m h }
INITIAL { ... }
BREAKPOINT { SOLVE states METHOD cnexp; ica = gbar * m * h * (v - eca) }
DERIVATIVE states { rates(v); m' = (minf - m)/mtau; h' = (hinf - h)/htau }
PROCEDURE rates(v) { minf = ...; mtau = ...; hinf = ...; htau = ... }
```

Translates to Brian2:
```python
eqs = """
dv/dt = ... + g_egl19/C_mem : volt
g_egl19 = gbar * m * h * (E_Ca - v) * area : siemens
dm/dt = (minf - m)/mtau : 1
dh/dt = (hinf - h)/htau : 1
minf = 1/(1 + exp(-(v - va_egl19)/ka_egl19)) : 1
mtau = pdg1 + pdg5*exp(-((v-pdg7)/pdg6)**2)*(...) : second
hinf = ... (multi-component, paper specifies)
htau = ...
"""
```

**Per-channel translation effort:** ~2-4 hours per channel including validation against Nicoletti's published traces. 22 channels = ~44-88 hours focused work, likely faster as patterns emerge after first few channels.

### What breaks

- **Nothing in current infrastructure.** All existing files (LIFBrain, GradedBrain, ClosedLoopEnv, modulator, sensory, FSM, classifier, scenarios, dashboard, audit harness, calibration) preserved.
- The compartmental scaffold can be enhanced incrementally: replace handcrafted `m_Ca + h + I_KCa` equations with the imported HH channels, neuron by neuron.

### What's gained

- **22 worm-validated HH channels** in Brian2 form
- **Brian2's compute advantage** preserved (1-2 orders of magnitude faster than NEURON on CPU per benchmarks; relevant for RTX 4060 Ti compute budget)
- **Brian2GeNN / Brian2CUDA** GPU acceleration paths available if scaling needed
- Per-channel additive: prioritize 6 most-load-bearing (EGL-19 + h, SHK-1, SHL-1, KQT-3, SLO-1, NCA) and add others over time
- Single debugging surface

### Realistic execution path

1. Translate EGL-19 first (load-bearing for plateau dynamics; replaces current m_Ca + h)
2. Validate against Nicoletti's voltage-clamp traces side-by-side
3. Translate SLO-1 (BK with Ca coupling) to fix plateau termination properly
4. Translate SHK-1, SHL-1, NCA, KQT-3 (5 channels = essential set)
5. Add remaining 16 channels as needed per cell-type calibration
6. Validate full set of 9 Nicoletti neurons reproduce her traces

### Rollback

At any point, revert `lif_brain.py` / `graded_brain_h_kca.py` to current state. Each channel is additive; no all-or-nothing commitment. **Lowest-risk path.**

### Cost-benefit

| Dimension | Score |
|---|---|
| Engineering effort | medium (~44-88 hrs translation + validation) |
| Risk | low (rollback per-channel) |
| Compute cost | low (Brian2 fast) |
| Preserves existing work | **yes, fully** |
| Long-term flexibility | good (extensible) |

---

## Path 3B — Multi-framework (Brian2 + NEURON parallel)

Run Brian2 for ~290 cells (current LIFBrain/GradedBrain), NEURON for 9 cells using Nicoletti's `.mod` files natively.

### What breaks

- **ClosedLoopEnv orchestration becomes substantially more complex.** Two simulation backends each with their own time stepping, integration methods, event handling. Synchronization at MuJoCo's 50 ms cadence requires:
  - Both backends advance to t+50ms in sync
  - State exchange between backends (V values for cross-backend synapses, spike events)
  - Coordinated stimulus injection (sensory cascades feeding into both)
  - Coordinated output reading (FSM/classifier merging spike rasters from both)
- **Spike/voltage exchange protocol** needs careful design. Nicoletti cells output continuous V (graded); Brian2 cells output spike events. Mapping spikes ↔ voltage at the synapse-by-synapse level is non-trivial.
- **Modulator layer** rewrites `I_ext` on Brian2 neurons; needs equivalent path for NEURON neurons. Likely requires custom NEURON mechanism for modulation.
- **Sensory transduction cascades** would need to drive both backends.
- **Audit harness** (phase0_audit.py etc.) needs to merge spike data from both backends.
- **Calibration infrastructure** (LHS, sweeps) needs to handle two backends' parameter spaces.

### What's gained

- Don't need to translate Nicoletti's channels to Brian2 (preserves her validated implementations exactly)
- NEURON's biophysical maturity for the 9 specific cells

### Realistic execution path

1. Establish NEURON installation alongside Brian2 (pip install neuron + mod compilation)
2. Build a bridge module: `dual_backend_brain.py` that holds both NEURON and Brian2 sub-networks
3. Implement timestep sync at the brain-level run() interface
4. Implement V-state exchange for cross-backend gap junctions
5. Implement spike-event exchange for cross-backend chemical synapses
6. Validate timestep alignment + state exchange under simple scenarios
7. Then run full audit suite

### Rollback

Possible but expensive: revert to Brian2-only by removing NEURON sub-network. The bridge code is dead weight unless fully removed.

### Cost-benefit

| Dimension | Score |
|---|---|
| Engineering effort | high (~3-6 weeks for bridge + validation) |
| Risk | high (sync bugs, state-exchange bugs, two debugging surfaces) |
| Compute cost | medium (NEURON slow for 9 cells; Brian2 fast for rest; net OK) |
| Preserves existing work | partially |
| Long-term flexibility | medium (couples to NEURON's evolution) |

**Not recommended for this project's constraints.**

---

## Path 3C — Backend switch to NEURON

Rewrite simulator brain layer in NEURON. Use NetPyNE for declarative network construction. Keep MuJoCo body driver, dashboard pipeline.

### What breaks

- **All Brian2-coupled brain code rewritten:** lif_brain.py, graded_brain.py, graded_brain_h_kca.py, compartmental_neurons.py, compartmental_neurons_kca.py — replaced with NEURON-equivalent.
- **modulation_layer.py** — NetPyNE has its own approach to current injection; needs port.
- **sensory_transduction.py** — external ODEs injecting current into NEURON cells; needs new wiring.
- **All Phase 0 diagnostic scripts** (phase0_audit.py, phase0_plateau_diagnostic.py, phase1_plateau_calibrate.py etc.) — many are Brian2-specific; need NEURON-equivalent.
- **The 7-entry DOCUMENTED_SIGN_EXCEPTIONS registry** — needs to be re-implemented in NEURON's connectivity setup.
- **The voltage-fix patches** — NEURON has its own per-neuron parameter handling.
- **The Wave 1 cellular validation work** (h_kca patch, calibrated α_Ca etc.) — needs to be redone in NEURON or replaced wholesale by Nicoletti's models.

### What's gained

- **Direct import of Nicoletti's 22 channels + BAAIWorm's 14 channels + ChannelWorm's 4 channels** — no translation work
- **NetPyNE declarative API** for network construction; well-tested patterns
- **CoreNEURON GPU support** via NetPyNE — alternative to Brian2GeNN
- **Standard biophysical-modeling community alignment**

### Realistic execution path

1. Set up NEURON + NetPyNE environment
2. Port simulator's neuron list, connectome, signs to NetPyNE declarative format
3. Import Nicoletti's mechanisms; assign per-cell as Wave 1 cellular work supports
4. Implement modulator layer in NEURON (custom mechanism or current injection)
5. Implement sensory transduction → NEURON wiring
6. Re-implement audit harness using NetPyNE's batch-run + analysis tools
7. Validate against existing Brian2 results for connectome and behavior reproduction
8. Re-validate Wave 1 cellular tests
9. Re-validate scenario sweeps

### Rollback

Difficult. Once committed to NEURON, reverting requires re-doing the bridge work in reverse. Effectively a one-way decision.

### Cost-benefit

| Dimension | Score |
|---|---|
| Engineering effort | very high (~3-6 months full rewrite + revalidation) |
| Risk | high (substantial code rewrite, behavior preservation hard) |
| Compute cost | medium (NEURON slower than Brian2 on CPU; CoreNEURON GPU helps) |
| Preserves existing work | low (most brain code rewritten) |
| Long-term flexibility | good (NEURON is community standard) |

**Not recommended given project constraints (single undergraduate, no PhD route, RTX 4060 Ti compute).** The 3-6 month rewrite cost dominates the channel-import benefit.

---

## Path 3D — NeuroML2-based simulation

Use jNeuroML or similar declarative-model-driven simulator. Most portable; targets NEURON / NEST / MOOSE for execution.

### What breaks

- **All Brian2-coupled code** (same as Path 3C; substantial rewrite)
- **MuJoCo body integration** — no clear NeuroML2 ↔ MuJoCo bridge; would need custom development
- **Modulator layer** — NeuroML2 has limited neuromodulation primitives
- **Sensory transduction cascades** — external ODE coupling to jNeuroML execution unclear
- **The simulator becomes Java-dependent** for jNeuroML execution

### What's gained

- **Most portable across simulator backends**
- **NeuroML2 is the community standard** for sharing biophysical models
- **Future-proof** as community converges

### Realistic execution path

Substantial unknown — jNeuroML has limited adoption beyond OpenWorm and a few academic groups; documentation is sparser than NEURON or Brian2.

### Rollback

Worse than Path 3C. Java + multiple-backend abstraction makes reversal harder.

### Cost-benefit

| Dimension | Score |
|---|---|
| Engineering effort | very high + uncertainty |
| Risk | very high (least-mature ecosystem for our coupling needs) |
| Compute cost | unclear |
| Preserves existing work | low |
| Long-term flexibility | best in theory; worst for short-term productivity |

**Not recommended for this project's needs.** May be revisited if community ecosystem matures or if collaboration with jNeuroML-using labs emerges.

---

## Recommended primary path: Path 3A (Brian2 + parameter import)

### Reasoning

1. **Preserves all existing infrastructure.** ClosedLoopEnv, modulator layer, sensory transduction, MuJoCo body, FSM, classifier, scenario pipeline, dashboard, audit harness, calibration scripts, 7-entry override registry, voltage-fix patches, Wave 1 cellular validation work — all stay. This is substantial value to preserve.

2. **Compute advantage matches constraint.** Brian2's CPU performance is 1-2 orders of magnitude faster than NEURON for many workloads (per recent benchmarks). On RTX 4060 Ti, Brian2 + numpy backend handles 302 cells comfortably; NEURON with 302 multi-compartmental cells (BAAIWorm-scale) requires Nvidia 3090 minimum per BAAIWorm's docs — exceeds available compute.

3. **Translation is bounded and additive.** ~44-88 hours total for 22 channels, but per-channel additive — start with EGL-19, SHK-1, SHL-1, NCA, KQT-3, SLO-1 (6 most-load-bearing) and add others as cell calibration requires. Don't need full set before progress can be made.

4. **Validation pathway exists.** Nicoletti's Python wrappers (`AVAL_simulation.py` etc.) reproduce specific voltage-clamp and current-clamp traces. Side-by-side comparison validates each translated channel against her published behavior.

5. **Rollback per-channel.** No all-or-nothing commitment. If translation of a particular channel proves problematic, fall back to current handcrafted approach for that channel.

6. **Path 3A doesn't preclude Path 3C later.** If the project ever commits to NEURON backend (e.g., for a paper requiring NetPyNE or for collaboration with NEURON-using labs), Path 3C remains an option. Path 3A is the lower-cost first step.

### Recommended execution sequencing

| Step | Content | Gate |
|---|---|---|
| 1 | Set up Brian2-NEURON validation harness (run Nicoletti's NEURON models locally, capture voltage-clamp traces) | License verification, Nicoletti `.mod` compilation in NEURON |
| 2 | Translate EGL-19 + h with full Nicoletti dynamics to Brian2; validate against her trace | Step 1 complete |
| 3 | Translate SLO-1 (BK with Ca coupling) — addresses plateau termination | Step 2 success |
| 4 | Translate SHK-1, SHL-1, NCA, KQT-3 (essential 4 more) | Step 3 success |
| 5 | Per-cell calibration in compartmental scaffold for AVAL/AVAR using Nicoletti's per-cell parameters | Step 4 success |
| 6 | Validate Wave 1 cellular targets (Mellem 600 ms plateau) under imported channels | Step 5 success |
| 7 | Add remaining 16 channels per cell-type need | Step 6 success |
| 8 | Network-level validation: re-run scenario sweeps; compare vs current LIFBrain/GradedBrain | Step 7 sufficient |

### Fallback paths

**If Path 3A's translation work hits unexpected blockers** (e.g., a specific channel's NMODL idiom doesn't translate cleanly):
- Per-channel fallback: keep that channel as handcrafted Brian2; translate others. Most channels follow the same pattern as EGL-19; bespoke handling per channel is unlikely.
- Architecture fallback: switch to **Path 3B (multi-framework)** for the specific cells using problematic channels. Run NEURON for AVAL/AVAR specifically while keeping Brian2 elsewhere. ~3-6 week bridge implementation.

**Trigger criteria for fallback:**
- More than 4 of 22 channels resist clean Brian2 translation
- Performance regression > 5× from current Brian2 + handcrafted approach (unlikely given Brian2 speed)
- Validation traces consistently disagree with Nicoletti by > 20% in characteristic features after careful translation

### What Path 3A doesn't solve

The previous audit identified gaps Path A doesn't address:

- **Receptor binding kinetics** — Markov state schemes for major ligand-gated channels (UNC-49 GABA_A, GluCl, iGluR family, nAChR). Not in Nicoletti or c302. Project must implement separately if mechanistic synaptic claims are needed.
- **CeNGEN-coupled per-cell channel densities** — neither package uses CeNGEN. Project must integrate scaling via TPM data.
- **Modulator layer / peptide processing refinement** — entirely project's own work.
- **Anesthesia-specific allosteric framework** — outside scope of all packages.
- **Per-cell biophysics for the 290+ non-Nicoletti neurons** — calibration burden falls on this project.

These remain Wave 2+ work for the project regardless of backend choice.

---

## Summary

| Path | Effort | Risk | Compute fit | Preserves work | Recommendation |
|---|---|---|---|---|---|
| **3A** Brian2 + import | medium | low | excellent | full | **PRIMARY** |
| **3B** Multi-framework | high | high | medium | partial | fallback if 3A blocks |
| **3C** NEURON backend | very high | high | medium (CoreNEURON GPU mitigates) | low | not recommended |
| **3D** NeuroML2-based | very high + unknown | very high | unclear | low | not recommended |

**Primary recommendation: Path 3A.** Bounded translation work, preserves existing infrastructure, matches compute constraints, supports per-channel rollback, doesn't preclude Path 3C later.

