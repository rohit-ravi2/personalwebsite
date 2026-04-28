# Phase δ network integration scoping — mid-flight findings

**Mode:** investigation + architectural design (not implementation).
**Spec:** `phase_v_w2_phase_delta_scoping_prompt.md`.
**Output:** `phase_delta_scoping.md` is the main decision-grade document.

---

## Pre-flight acknowledgment

Plan accepted. No pre-flight pushback warranted. Approach:

1. Read production simulator code (`lif_brain.py`, `graded_brain.py`,
   `graded_brain_h_kca.py`, `compartmental_neurons_kca.py`,
   `closed_loop_env.py`, scenario JSON files, `connectome.npz` consumers,
   FSM/classifier/dashboard hooks). Document actual structure. Primary
   source wins over prompt assumptions.
2. Read 1-2 wave2 production cells (`option_alpha_ava_cell.py`, etc.) to
   characterize the Brian2-NeuronGroup integration shape.
3. Phase 2 (alternatives), Phase 3 (compute), Phase 4 (validation),
   Phase 5 (work-block decomposition) flow from Phase 1's primary-source
   findings.
4. Synthesis (Section 6) pulls everything into a recommended trajectory.

Findings appended below as work proceeds.

---

## Section 1 findings — production simulator architecture (complete)

**Primary-source verified.** The prompt's high-level claims about
"production simulator infrastructure" are substantively accurate, but
the actual structure is significantly cleaner and more constrained
than the prompt suggested. Headlines:

- `LIFBrain` and `GradedBrain` are **already Brian2 NeuronGroup-based**
  (one NeuronGroup of N=300, plus three Synapses objects: exc / inh /
  gap). `prefs.codegen.target = "numpy"` is hardcoded in both at module
  scope (not cython). They are NOT a separate cell-object architecture
  competing with Brian2 — they ARE Brian2.
- Connectome: `artifacts/connectome.npz`, 300 neurons (CANL/CANR
  excluded), arrays `names`, `nt_primary`, `sign`, `W_chem_raw`,
  `W_chem_per_edge`, `W_gap`, `iGluR_expr`, `GluCl_expr`,
  `post_sign_glu`, `klass`. 3707 chemical edges, 2188 gap edges.
- Run interface: `brain.run(duration_ms)` advances Brian2 internals
  (default dt=0.1 ms). `ClosedLoopEnv.step_sync()` calls run(50 ms),
  reads spikes via `_read_spike_rates()`, advances MuJoCo 100 sub-
  steps at 0.5 ms each. **Sync cadence = 50 ms.**
- "Cell representation" — for LIF/Graded, a "cell" is one row in a
  NeuronGroup with shared eqs. There is **no per-cell-type
  parameterization in current production**: every neuron is the same
  LIF or graded-Boltzmann unit, distinguished only by its index +
  per-neuron state arrays (e.g., `g_Ca_local`, `tau_h`,
  `ablation_current_pA`).
- **Wave 2 cells, by contrast, are 1-cell NeuronGroups with 4-7
  voltage-dependent channels and 50+ state variables per cell.**
  Each cell has its own NeuronGroup, its own equations, its own
  Network. They are not designed to be batched into a single
  NeuronGroup with the rest of the brain.
- Modulation: `ModulationLayer` overlays a 9-channel slow
  neuromodulator system on top via `@network_operation(dt=50 ms)`.
  Reads spikes from `brain.spikes`, writes `neurons.I_ext`.
  Non-coercive — works on any brain class with `.spikes` + `.neurons`
  + `.I_ext` interface.
- Sensory: two modes. Default `injection`: PoissonGroup → Synapses
  with `on_pre="v_post += W*mV"`. Alternative `transduction`:
  ODE-based cascades that compute Poisson rates fed in the same way.
  Both are LIF-spike-trigger oriented.
- Classifier + FSM: read `brain.spikes` (or graded's _FakeSpikeMonitor
  shim that thresholds σ at 0.5). Output is per-neuron rate vector,
  then fed to either Atanas-trained classifier (8 events) or
  ActivityFSM (z-scored direct-from-rate). FSM emits CPG params for
  MuJoCo.
- Body: MuJoCo wormbody.xml, position-actuated CPG, drag from Phase
  2a. Curvature → Poisson rate to PDE/PDA/DVA via `brain.set_proprioception()`.

**Scenario "JSON":** there are no input scenario JSON specs. Scenarios
are Python dicts: `stim_schedule = [(t_s, preset_name, intensity), ...]`
passed to `env.run(duration_s, stim_schedule=...)`. The "scenario JSON"
files at `public/data/wormbody-brain-*.json` are **OUTPUT** trace
files, not input specs. Presets are hardcoded in `sensory_injection.py`
(SENSORY_PRESETS dict) and in `_PRESET_TO_CASCADE` dict in
`closed_loop_env.py`.

### Production-side I/O contract (the load-bearing interface)

ClosedLoopEnv expects from a brain object:

```
brain.names       : list[str], len N
brain.idx         : dict[str, int]
brain.N           : int
brain.spikes      : object with .t (Brian2 quantity, units = second) and
                    .i (np int array). Real or _FakeSpikeMonitor shim.
brain.neurons     : Brian2 NeuronGroup with .I_ext settable
brain.run(ms)     : advance simulator
brain.time_ms()   : return wall t in ms
brain.set_proprioception(curv_mag) : mutate proprio Poisson rates
brain.set_sensory_rate(name, rate_hz, weight_mv) : add/update Poisson drive
brain.inject_poisson(name, rate_hz, weight_mv)   : non-destructive poisson stim
brain.ablate(names, current_pA)                  : silence neurons by
                                                   constant hyperpolarizing I_ext
brain.ablation_current_pA : np.float32 (N,) — read by ModulationLayer
```

**This is the integration contract Phase δ must satisfy.** Anything
that exposes this interface and steps a Brian2 Network drops in.

### Wave 1 vs Wave 2 cellular layer relationship

There are TWO cellular-detail layers in the existing codebase:

1. `compartmental_neurons.py` + `compartmental_neurons_kca.py` (Wave 1
   sandbox era) — multi-compartment neurons with h-inactivation and
   K_Ca, parameterized roster with 8 cells (AVA, AVE, AVB, PVC, RIS,
   DVA, AIY*). Uses Brian2 NeuronGroup. Treats each compartmental cell
   as a separate group, then hand-couples via axial g_ax. **Sandbox,
   not production-wired.** Not consumed by ClosedLoopEnv.
2. `graded_brain_h_kca.py` — same era, single-compartment graded brain
   variants (base / h_only / h_kca). Three variants of all-300-neurons
   with optional h + KCa. **Sandbox, also not production-wired.**

Production today is `LIFBrain` (default) and `GradedBrain` (T1a opt-in).
Both with no per-cell channel detail. The Wave 1 compartmental work was
methodology validation that didn't ship into ClosedLoopEnv.

**This means Phase δ's "validated cellular layer connecting to
production simulator infrastructure" is really:**

- Take Wave 2's 3 production-grade cells (Brian2 NeuronGroups with
  Nicoletti-validated channel sets), AND
- Connect them to the rest of the 300-neuron LIF/Graded simulator via
  the connectome, AND
- Preserve the ClosedLoopEnv I/O contract.

### Production cells and codegen

- Wave 2 `option_alpha_*_cell.py` factories already set
  `prefs.codegen.target = "cython"` post-cython migration.
- LIFBrain and GradedBrain set `prefs.codegen.target = "numpy"`
  at module import. **This is a global setting; the last `prefs`
  assignment wins.** Phase δ cannot run cython-Wave-2 cells alongside
  numpy-LIF in the same Python process unless this is reconciled.
  This is a real but small-effort coupling concern.

### Performance baselines from cython migration

- AVAL (4-channel cython): ~0.9 s wall / s simulated, dt=0.025 ms.
- AIY (7-channel cython): ~1.6 s wall / s simulated, dt=0.025 ms.
- RIM (7-channel cython): ~1.8 s wall / s simulated, dt=0.025 ms.
- LIFBrain at 300 neurons: not benchmarked in artifacts I read, but
  scenario JSON outputs are 30-60 s simulated, ClosedLoopEnv ships
  these in routine generation.

### F18 implications at network scale

F18 (asymmetric multi-USEION-ca trigger) is a NEURON-side issue.
Brian2 cells don't have ion_style — `eca_mV` is a hardcoded constant
in our channel translations. Network-scale F18 risk: zero, **as long
as we keep Brian2 channels as our forward path.** The risk vector
would be Layer A re-validations against NEURON references for new
cells (RMD etc.) — out of scope for Phase δ.

---

## Section 2 findings — integration alternatives (complete; full
analysis in `phase_delta_scoping.md` §2)

Headlines:

- **Alternative A (full Wave 2 brain) is structurally infeasible
  without ~297 additional cell-validation work blocks.** Wave 2 has
  3 production-grade cells; the other 297 have no validated
  cellular detail. "Replace LIF entirely with Wave 2" requires
  scaffold cells for the unvalidated 297 — those scaffolds would be
  functionally LIF-equivalent (with ad hoc parameters), making
  Alternative A reduce to Alternative B in practice.
- **Alternative B (hybrid) is the correct integration.** All cells
  in **one Brian2 Network** with multiple NeuronGroups (one per
  Wave 2 cell type or per cell instance, plus one big LIF group for
  the 294 unvalidated cells). No Python-side adapter; Brian2
  Synapses cross NeuronGroups natively. The prompt's "coupling at
  the connectome layer with adapter passing voltages/currents
  between systems" framing was misleading — it's just Brian2
  Synapses.
- **Alternative C reduces to Alternative B at the limit.** As a
  staging strategy *within* B (one Wave 2 cell at a time), it's
  exactly the right risk decomposition.

The unique design surface in B: **the "release event" definition
for Wave 2 cells.** Wave 2 cells are graded (Nicoletti's neurons
plateau, not spike). Need a translation rule: when does a
presynaptic Wave 2 cell deliver a postsynaptic LIF spike-arrival
event? Two reasonable choices: (a) V threshold crossing (e.g., V
crosses -25 mV with refractory); (b) graded Boltzmann release
(matches GradedBrain pattern). Both are defensible; both need
calibration — possibly a dedicated WB3.5.

---

## Section 3 findings — compute envelope (complete)

- Alternative B at 60 s simulated: ~10 minutes wall-clock cython.
  Bottleneck is the 6 Wave 2 cells (cython baseline 1.5 s/s/cell ×
  6 × 60 ≈ 540 s). LIF scaffold negligible by comparison.
- dt mismatch (Wave 2 dt=0.025 ms, LIF dt=0.1 ms) handled via
  Brian2 per-NeuronGroup `clock` keyword. Standard idiom.
- Cython codegen target collision (LIF: numpy hardcode; Wave 2:
  cython) is the load-bearing blocker. Fixed in WB1.

---

## Section 4 findings — validation strategy (complete)

Three-layer validation:

- **Layer A (preserve prior behavior):** 6 scenarios; FSM state
  distribution within 10% of LIF baseline; event probability
  correlation > 0.7. Spontaneous, touch, osmotic_shock, food,
  chemotaxis, aerotaxis.
- **Layer B (biological enrichment):** AVA plateau under sustained
  ASH (Mellem 2008); AIY graded response to food intensity sweep;
  RIM activity correlation with REVERSE FSM state.
- **Layer C (cross-cell numerical sanity):** confirm AIY eca=127.59
  and RIM eca=60 are independent in the network. F18-style
  network-scale check. Quick gate, likely passes by construction.

---

## Section 5 findings — work block decomposition (complete)

Six work blocks (possibly 7 with WB3.5 release-event calibration):

- **WB1:** Integration prep + cython unification + namespace audit.
- **WB2:** `Wave2HybridBrain` class skeleton (LIF-only equivalent
  initially), ClosedLoopEnv accepts `brain_class='wave2_hybrid'`.
- **WB3:** AVA pair as Wave 2 cells. **WB3.5 (contingency):**
  release-event calibration if WB3 surfaces ambiguity.
- **WB4:** AIY pair extension.
- **WB5:** RIM pair extension; Layer C namespace audit.
- **WB6:** Phase δ summary + Layer A/B trace artifacts.

Strict sequencing; no parallel work within Phase δ. Critical path
WB1 → WB6.

---

## Discrepancies vs prompt assumptions (summary)

| Prompt | Actual | Severity |
|---|---|---|
| "Scenario JSON files driving touch/food/osmotic" | Scenarios are Python `[(t, preset, intensity)]` lists; JSON files are output traces | Reduces Phase δ scope (no JSON parser to write) |
| "302 neurons full connectome" | 300 (CANL/CANR excluded) | Bookkeeping |
| Implies LIF is "existing cellular layer" Wave 2 supersedes | LIF is a 1-equation per neuron model; Wave 2 cells are 4-7-channel multi-state cells. Not a replacement; an enrichment for 3 specific cells. | Conceptual — Phase δ is "first cellular detail layer" not "replace existing cellular layer" |
| "compartmental_neurons_kca.py existing cellular implementations" | Wave 1 sandbox files, NOT production-wired into ClosedLoopEnv | Phase δ doesn't need to maintain compatibility with these |
| "Connectome data: connectome.npz referenced in earlier project work" | Fully verified: 300 names, 4 weight matrices, 3707 chem edges, 2188 gap edges | Confirmed |
| "modulation layer (9 modulators per project memory)" | Confirmed (FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5HT, DA, TA, OA) | Confirmed |

**No discrepancies severe enough to invalidate Phase δ feasibility.**
The prompt's broad architecture description is correct. The minor
corrections sharpen Phase δ scope rather than breaking it.

---

## Summary

Scoping complete. Recommended trajectory: **Alternative B (hybrid),
Alternative C staged. WB1 first (cython unification + namespace
audit). 6 work blocks total.**

Main scoping output: `wave2/artifacts/phase_delta_scoping.md` (this
is the morning-review entry point and the launching pad for Phase δ
implementation work blocks).


