# Wave 2 Phase δ network integration — scoping document

**Date:** 2026-04-26
**Mode:** investigation + architectural design (NOT implementation).
**Spec:** `phase_v_w2_phase_delta_scoping_prompt.md`.
**Companion findings:** `phase_delta_scoping_findings.md` (mid-flight).

This document is decision-grade scoping output for the Phase δ
implementation work blocks. It does NOT propose code changes, channel
translations, or new infrastructure. It surveys the production
simulator, evaluates three integration alternatives against
primary-source structure, estimates compute, designs validation, and
decomposes the recommended trajectory into bounded work blocks.

---

## Section 1 — Production simulator architecture survey

Primary-source verified by reading `lif_brain.py`, `graded_brain.py`,
`graded_brain_h_kca.py`, `closed_loop_env.py`, `modulation_layer.py`,
`behavioral_fsm.py`, `activity_fsm.py` (referenced), the connectome
artifact via `np.load`, an output scenario JSON, and the sensory
preset surface. The prompt's "production simulator infrastructure"
list is broadly accurate but several structural claims warrant
correction or clarification.

### 1.1 — Brain layer architecture

`LIFBrain` (`lif_brain.py`, ~600 lines):

- **Class.** Single `LIFBrain` class. `__init__` loads
  `artifacts/connectome.npz`, builds one Brian2 `NeuronGroup` of
  N=300 with stochastic LIF eqs (`dv/dt = (v_rest - v)/tau + (I_gap +
  I_ext)/C_mem + noise_sigma * xi / sqrt(tau)`), then constructs
  three `Synapses` objects (exc, inh, gap), a `SpikeMonitor`, and
  combines into one `Network`.
- Spike threshold `v > v_thr` triggers `v = v_reset` reset. No
  per-neuron LIF parameter heterogeneity — every cell is the same
  Mellem-2008-grounded LIF (v_rest=-25 mV, v_thr=-10 mV, v_reset=-30
  mV, tau=10 ms, t_ref=2 ms).
- **Codegen target:** `prefs.codegen.target = "numpy"` set at module
  import.

`GradedBrain` (`graded_brain.py`, ~440 lines):

- Drop-in API replacement for LIFBrain. NeuronGroup with continuous
  graded eqs: `dv/dt = (v_rest - v)/tau + (I_syn_exc + I_syn_inh +
  I_gap + I_ext + I_Ca)/C_mem + noise_sigma * xi / sqrt(tau)` plus
  `sigma = 1 / (1 + exp(-(v - v_half)/k_half))`.
- Per-neuron `g_Ca_local` array assigns L-type Ca conductance only
  to a hand-curated `PLATEAU_NEURONS` list (14 names: AVA, AVE, AVD,
  AVB, PVC, AIY, RIS, DVA pairs).
- **No actual spikes.** `_FakeSpikeMonitor` shim polls σ via
  `@network_operation(dt=10*ms)` for σ rising through 0.5 and emits
  pseudo-spike events, so the rest of ClosedLoopEnv (which reads
  `brain.spikes.i` and `brain.spikes.t`) works unchanged.
- `prefs.codegen.target = "numpy"`.

`GradedBrain h_kca` (`graded_brain_h_kca.py`, sandbox, NOT wired into
ClosedLoopEnv): same shape but with optional h-inactivation +
intracellular [Ca] pool + I_KCa current. Three variants ('base',
'h_only', 'h_kca'). Built via `build_neuron_group(variant, N, names)`
factory; not class-based.

### 1.2 — Cell representation

A "cell" in production is **one row of a single 300-row NeuronGroup**.
There is no per-cell-type structure beyond:

- A per-neuron sign override array (LIFBrain) and per-edge sign
  exception map.
- A per-neuron `g_Ca_local` (GradedBrain only — 14 plateau cells
  vs 286 non-plateau).
- A per-neuron `tau_h` (only in `graded_brain_h_kca` sandbox).
- A per-neuron `ablation_current_pA` (LIFBrain + GradedBrain).

**Wave 2 cellular layer is structurally different.** Each Wave 2
cell (`build_brian2_aval_4channel`, `build_brian2_aiy_*`,
`build_brian2_rim_*`) returns a factory that builds a 1-cell
NeuronGroup with eqs concatenated from per-channel modules in
`wave2/channels/`. Each Wave 2 cell has its own equations, its own
50+ state variables, its own monitors, its own Network.

### 1.3 — Time loop

There is no custom timestep loop in the brain layer. `brain.run(
duration_ms)` simply calls `self.net.run(duration_ms * ms)` —
Brian2's internal scheduler handles all integration.

`ClosedLoopEnv.step_sync()` (the orchestrator):

1. (Optional) Advance transduction cascades by 50 ms (Python ODE).
2. `brain.run(50 * ms)` — Brian2 advances internally at default
   dt=0.1 ms (so 500 internal sub-steps).
3. `_read_spike_rates()` → spike counts per neuron per 50 ms window.
4. Maintain calcium buffer (every 600 ms, build calcium proxy from
   spike counts, IIR-smooth, calibrate, classify).
5. Update FSM (classifier mode or activity mode).
6. Step MuJoCo `STEPS_PER_SYNC = 100` times at 0.5 ms = 50 ms.
7. Record body frame.
8. Compute body curvature → `brain.set_proprioception()`.
9. (Optional) update environment + chemo gradient → write
   `brain.set_sensory_rate(...)` rates.

**Bedrock cadence: 50 ms.** Brain timestep, body timestep, and sync
all share this granularity.

### 1.4 — State variables

Brain layer maintains:

- `neurons.v` (volts) — membrane potential per neuron.
- `neurons.I_gap`, `neurons.I_ext` (amperes) — accumulated currents.
- `neurons.I_syn_exc`, `neurons.I_syn_inh` (graded only).
- `neurons.sigma` (graded only) — Boltzmann output.
- `SpikeMonitor.t`, `SpikeMonitor.i` — spike history.
- `ablation_current_pA` (numpy, persistent).
- `_sensory_groups` dict and `_stim_cache` list — Brian2 PoissonGroup
  + Synapses objects added to the Network for sensory drive.
- `proprio_group` (PoissonGroup) and `proprio_syn` (Synapses).

Synapses store weights (`syn_exc.w`, `syn_inh.w`, `syn_gap.w_gap`) —
loaded once at construction and not retuned at runtime.

ModulationLayer maintains:

- `concentrations` (numpy, M=9) — per-modulator dimensionless conc.
- Internal release/target weight matrices loaded from
  `artifacts/modulator_tables.npz`.
- `@network_operation(dt=50 ms)` increments concentrations from
  spikes and writes `neurons.I_ext` accordingly.

### 1.5 — I/O — readout interface

`brain.spikes` is the load-bearing readout. `_read_spike_rates`
slices new spikes since last call and counts per neuron:

```python
all_t = self.brain.spikes.t[:]
all_i = self.brain.spikes.i[:]
new = slice(self._prev_spike_len, len(all_t))
counts = np.zeros(self.brain.N, dtype=np.float32)
np.add.at(counts, all_i[new], 1)
return counts[self.readout_idx]   # 18 readout subset for classifier
```

Calcium proxy per neuron: 600 ms aggregated counts, single-tap IIR
(τ=0.5 s), per-neuron affine calibration onto Atanas ΔF/F moments
loaded from `artifacts/calibration.npz`.

Classifier (`neural_classifier_bank.py`, not read in detail in this
work block) consumes the 18-neuron readout and emits 8 event
probabilities per 600 ms tick. FSM consumes those probs (or the
direct rate vector under `fsm_mode='activity'`).

Dashboard / output: `ClosedLoopEnv.export()` writes the trace JSON
files at `public/data/wormbody-brain-*.json`. The `meta` block lists
`readout_neurons`, `num_neurons`, `events_tracked`, `states`, etc.
The trace lists `frames`, `raster`, `full_raster`, `event_probs`,
`fsm_states`, `stim_log`, `modulator_concentrations`, etc.

### 1.6 — Discrepancies vs prompt assumptions

Documented for transparency:

| Prompt claim | Actual structure | Implication |
|---|---|---|
| "Scenario JSON files driving touch/food/osmotic/etc." | Scenarios are **Python dicts** (`[(t_s, preset, intensity), ...]`) passed to `env.run()`. The JSON files are **output traces** for the website, not inputs. | Phase δ has no input scenario format to integrate against; just `stim_schedule` lists. |
| "compartmental_neurons_kca.py existing cellular implementations" | Sandbox files. Not consumed by ClosedLoopEnv in any code path I found. | Phase δ doesn't replace anything in production cellular detail; it's the **first** production-grade cellular detail layer. |
| "9 modulators per project memory" | Confirmed: FLP-11, FLP-1, FLP-2, NLP-12, PDF-1, 5HT, DA, TA, OA. | OK. |
| "5 sensory transduction cascades" | Confirmed: ASE, AWC, ASH, AFD, ALM/AVM (`sensory_transduction.py` referenced). | OK. |
| "302 neurons full connectome" | **300** neurons (CANL/CANR excluded — no characterised synaptic output). | Phase δ planning should use N=300, not 302. |
| "MuJoCo coupling" | Confirmed: `wormbody.xml`, position-actuator CPG, drag from Phase 2a, 50 ms sync, no force feedback brain→body. | OK. |

The first two are non-trivial — they shrink Phase δ scope (no
scenario JSON parser to write; no compartmental cellular layer to
maintain compatibility with). The "300 not 302" is bookkeeping but
should be stated.

### 1.7 — Cython codegen consideration

LIFBrain and GradedBrain set `prefs.codegen.target = "numpy"` at
module scope. Wave 2 production cells set
`prefs.codegen.target = "cython"` at factory invocation time.
**The two cannot coexist as-is in the same Python process** —
whichever was set last wins. This is a real but bounded issue:

- Either set the target to `cython` once globally (cython migration
  summary §5 already flags this as the "operational followup":
  remove the numpy hardcode in the 17 wave2 files OR change the
  brain layer to cython).
- Or the integration alternative chosen needs to handle this —
  e.g., Alternative A entirely replaces the brain so the LIF
  numpy hardcode becomes irrelevant.

This is a **5-minute fix** when Phase δ implementation begins, but
worth flagging.

---

## Section 2 — Wave 2 cell integration shape analysis

Three architectural alternatives evaluated against the actual
production structure documented in Section 1. The prompt's
alternatives map cleanly with one structural caveat noted below.

### 2.1 — Alternative A: replace LIFBrain entirely with Wave 2 cells

**Concept.** `Wave2Brain` class loads connectome.npz, instantiates
each of 300 neurons as a Wave 2 cell, wires connectome via Brian2
`Synapses` objects, exposes the ClosedLoopEnv I/O contract.

**Issue: not all 300 neurons have Wave 2 cellular detail.** Wave 2
has 3 production-grade cells: AVAL, AIY, RIM. The other ~297 cells
have no Wave 2 implementation. Pure Alternative A as described is
**impossible without ~297 more cell-validation work blocks** (not
in scope, would multiply Phase δ ~100×).

**Reformulation:** "Replace LIFBrain entirely with Wave 2 cells
where validated, and use a simple-Brian2-cell scaffold for the
remainder, all in one network". This is essentially Alternative B
with 'all in one Brian2 Network' as a clarifying constraint.

| Dimension | Assessment |
|---|---|
| Implementation complexity | High — requires designing the "scaffold" cells for 297 unvalidated neurons (likely a generic 1-2 channel Brian2 cell), then coupling all cells via Brian2 Synapses, then exposing I/O. Plus per-cell-type parameter heterogeneity in NeuronGroups (Brian2 supports this but eqs cannot vary structurally within one group). |
| Coupling points | Connectome wired entirely in Brian2 Synapses (simpler than B). Modulation, sensory, FSM, classifier, body — all unchanged because the I/O contract is preserved. |
| Risk profile | High. (a) The 297 scaffold cells need parameters; with no Layer A reference they are fundamentally tunable hyperparameters of the network. (b) Brian2 networks with diverse equation structures across cells require careful NeuronGroup design (one group per equation-class). (c) Validation: how do you tell scaffold-cells-misbehaving from scaffold-cells-correctly-mediating between validated cells? |
| Performance | Best case: Brian2 batches 297 scaffold cells into 1-N groups → fast amortized vectorization. Worst case: per-channel diversity across cells multiplies parameter values, breaks vectorization. Cython baseline: 1.0-1.8 s/s for 7-channel cells; for 300 cells with diverse channel sets, plausibly 30-100 s wall per s simulated even with cython. |
| Path to Phase δ proper | Long. Most work is on the 297 unvalidated cells, which is research-grade work requiring its own work blocks. |

**Verdict:** Architecturally clean, but requires extensive
infrastructure (~297 scaffold cells, all needing parameters from
somewhere). **Not recommended for Phase δ.**

### 2.2 — Alternative B: hybrid (Wave 2 for 3 cells, LIF/Graded for 297)

**Concept.** AVA, AIY, RIM run as Brian2 NeuronGroups with full
Wave 2 channel sets (cython codegen). Other 297 cells run as
LIFBrain or GradedBrain (numpy/cython codegen). Coupling at
connectome layer via adapter passing voltage / spikes / synaptic
currents between the two systems.

**Issue: "two systems" are both Brian2.** If you embed all of these
in **one** Brian2 Network, Brian2 handles cross-NeuronGroup
Synapses natively. The hybrid alternative reduces to: build N+1
NeuronGroups in one Brian2 Network, where N is the number of
Wave 2 cell types (3) and the +1 is the LIF/Graded scaffold for
the other 297 cells, then connect via Synapses with appropriate
on_pre rules.

**This is the correct way to read Alternative B and it is much
simpler than the prompt's framing implies.** No adapter layer
between Python systems is needed; just one Brian2 Network with
several NeuronGroups.

| Dimension | Assessment |
|---|---|
| Implementation complexity | Medium. Build one NeuronGroup per Wave 2 cell type (so 6 single-cell NeuronGroups: AVAL, AVAR, AIYL, AIYR, RIML, RIMR — each existing Wave 2 factory builds one). Build one NeuronGroup of 294 cells running LIF/Graded for the rest. Wire all chemical and gap synapses, distinguishing pre- and post-side group identity. |
| Coupling points | (1) **Wave 2 ↔ LIF chem syn:** Wave 2 cells are not LIF — they don't spike in the LIF threshold sense. Need a "spike detection" criterion (v rising through some threshold? σ rising through 0.5?). (2) **LIF → Wave 2 chem syn:** when a LIF spikes, deliver a current pulse / voltage bump to the Wave 2 cell. Must define what `v_post += W_syn * w` means for a multi-channel Wave 2 cell where v is a continuous variable being integrated by rk4. (3) **Gap junctions:** electrical coupling between Wave 2 (mV-resolved) and LIF (mV-resolved but with spike resets). Brian2 Synapses with `(summed)` keyword can do this if both neurons expose `v` in volts. (4) **Sensory input** to the 6 Wave 2 cells (PoissonGroup → Synapses with on_pre). (5) **Modulation** writing `I_ext` to all cells — Wave 2 cells use I_inj naming, LIF uses I_ext; need rename or shim. |
| Risk profile | Medium-high. Spike-detection-on-graded-Wave-2-cells is the load-bearing risk. AVAL doesn't really spike (no Na+ — no fast AP). It plateaus. So "AVA spike → downstream LIF" requires defining what a "release event" means at the Wave 2 cell level. Two reasonable choices: (a) σ-style threshold crossing on V (e.g., V crosses -25 mV); (b) graded transmitter release proportional to V via Boltzmann (matches GradedBrain pattern). Either is defensible; both need calibration. |
| Performance | Wave 2 cython baseline: 6 cells × 60 s × ~1.5 s/s ≈ 540 s wall. LIF for 294 cells in one NeuronGroup is trivial (Brian2 vectorizes; current LIFBrain for 300 runs scenarios in seconds-to-minutes per JSON). Total: bounded by Wave 2 cells, ~10 minutes per 60 s scenario. |
| Path to Phase δ proper | Cleanest. Each new Wave 2 cell (RMD, future cells) drops in by adding another NeuronGroup. Existing infrastructure unchanged. |

**Verdict:** **Recommended.** This is the integration shape Phase δ
should pursue.

### 2.3 — Alternative C: Wave 2 as drop-in for specific cells

**Concept.** Existing brain layer (LIFBrain or GradedBrain) remains
mostly intact. Wave 2 cells "inject" as alternate implementations of
AVAL/AIY/RIM specifically. Other infrastructure unchanged.

**Issue: there is no obvious surgical-injection point.** A "cell"
in LIFBrain is one row of a NeuronGroup. You cannot replace a single
row of a NeuronGroup with a different NeuronGroup that has different
equations. You'd need to:

- Remove that row from the LIF NeuronGroup (Brian2 doesn't support
  this; you'd rebuild the group with N-1).
- Add a separate Wave 2 NeuronGroup containing just that cell.
- Re-wire connectome edges that pre-/post that cell to/from the new
  group.

**This is structurally identical to Alternative B.** The "drop-in"
framing was misleading; the actual operation is "remove cell from
LIF group; add as separate Wave 2 group; re-wire". After all 3 cells
are extracted, Alternative C IS Alternative B.

| Dimension | Assessment |
|---|---|
| Implementation complexity | Same as B, but staged: do it for one cell first (AVAL only, AIY+RIM stay in LIF) as a proof of concept, then generalize. |
| Coupling points | Same as B. |
| Risk profile | Lower-risk-per-step than B because you can validate one cell at a time. Higher-risk-overall because the staged approach may surface incompatibilities only at the final cell (RIM with its UNC-2/CCA-1 channels and 11000-ms-tau s-gate dynamics may behave differently in coupling than AVAL's 4-channel setup did). |
| Performance | Same envelope as B (per-cell Wave 2 cost dominates). |
| Path to Phase δ proper | Identical to B at the limit. |

**Verdict:** Useful as a **staging strategy within Alternative B**.
Not a separate alternative. The first work block of Phase δ should
in fact follow this staging logic: integrate AVA pair first, then
AIY, then RIM. This decomposes the integration risk.

### 2.4 — Synthesis

The three alternatives reduce to one architecturally:

- All Wave 2 cells coexist with a LIF (or Graded) scaffold for the
  unvalidated cells in **one Brian2 Network**.
- ClosedLoopEnv I/O contract is preserved by the wrapping
  `Wave2HybridBrain` class.
- Staging: integrate cells one at a time (Alternative C as work-block
  decomposition strategy), but the end state is Alternative B.

**The unique new design surface is the "release event" definition
for Wave 2 cells.** Wave 2 cells are graded (Nicoletti's neurons
plateau, not spike) — they need a translation rule for what
`presynaptic event → postsynaptic LIF spike-arrival` means.

---

## Section 3 — Compute envelope and scaling analysis

Baseline cython per-cell per-second-simulated cost from
`cython_migration_summary.md`:

| Cell | s wall / s simulated, dt=0.025 ms |
|---|---|
| AVAL (4 chan) | ~0.9 |
| AIY (7 chan) | ~1.6 |
| RIM (7 chan) | ~1.8 |

LIFBrain at 300 cells: not benchmarked in artifacts, but scenario
generation (60 s simulated, default dt=0.1 ms) ships routinely in
the website pipeline. Inferred ~1-5 s wall per 60 s simulated for
the LIF brain on existing hardware (8GB 4060 Ti, numpy codegen).

### 3.1 — Compute per alternative (60 s simulated scenario)

| Alternative | 6 Wave 2 cells | 294 LIF cells | Coupling overhead | Total wall-clock |
|---|---|---|---|---|
| A (full Wave 2) | ~9-12 min if scaffold works at LIF speeds; **unbounded** if scaffold needs higher per-cell channel detail | ≥6× more if scaffold is multi-channel (~70-100 s wall per s simulated for 300 cells with diverse channels) | Brian2 internal | Worst case 30-90 min per 60 s scenario |
| B (hybrid) | 6 × 60 × ~1.5 = **540 s = 9 min** (cython, dt=0.025 ms) | <60 s (numpy or cython, vectorized) | Brian2 internal Synapses (negligible) | **~9-10 min per 60 s scenario** |
| C (staged B) | 2 × 60 × ~0.9 = 108 s for AVA only; same as B at full integration | <60 s | Brian2 internal | **~3 min for AVA-only first work block; reaches B's 9-10 min at full** |

### 3.2 — dt sensitivity

Wave 2 cells run at dt=0.025 ms (cellular validation timestep). LIF
runs at dt=0.1 ms (Brian2 default). **One Brian2 Network has one
defaultclock per scheduler.** Either:

- Run everything at dt=0.025 ms — LIF cost increases ~4×.
- Run everything at dt=0.1 ms — Wave 2 cellular validation is
  invalidated for this dt; would need re-validation.
- Use Brian2's per-NeuronGroup `clock` keyword to give different
  groups different timesteps. Brian2 supports this; multi-clock
  scheduling has some overhead but is the correct fix.

**Recommendation:** dt=0.025 ms for Wave 2 NeuronGroups, dt=0.05 ms
or 0.1 ms for the LIF scaffold via per-group `clock`. Test in early
work block; this is a known Brian2 idiom.

### 3.3 — Compute bottleneck identification

Under Alternative B, the Wave 2 cells dominate. Per-cell channels
expand to ~25-50 state variables each. A 7-channel cell has
order-of-magnitude 10× the per-cell compute of a 1-equation LIF cell.
Across 6 cells, this is 6 × 10 ≈ 60 cell-equivalents of compute. The
LIF group at 294 vectorizes well in cython/numpy, so its cost is
negligible by comparison.

Optimization levers if Phase δ runs too slow:

1. **Already-cython Wave 2 cells** — done.
2. **Larger dt for non-Wave-2 group** — recommended via per-group
   clocks.
3. **Reduce monitor cadence** in the Wave 2 cells (currently
   `record=True` on multiple variables; could limit to v + currents).
4. **Batch all Wave 2 cells of the same type into one NeuronGroup**
   — e.g., AVAL + AVAR in one group rather than two separate
   instances. Halves cell-instance count from 6 to 3, with same
   per-cell-type eqs.

### 3.4 — Compute envelope verdict

**Alternative B at 60 s simulated: ~10 min wall-clock cython, ~3-4
hour numpy.** This is acceptable for development iteration. For
website-ready scenario regeneration this would 10× the current
LIF-only generation time, but is bounded.

---

## Section 4 — Validation strategy for Phase δ

Two validation layers per spec, plus a third needed by
F1-F18 lessons.

### 4.1 — Layer A network: integration preserves prior behavior

**Reference target:** for each of `wormbody-brain-{spontaneous,
touch, osmotic_shock, food, chemotaxis, aerotaxis}.json` scenarios,
the Wave-2-hybrid-brain trace produces:

- Same FSM state distribution (within 5% of LIF baseline) over the
  scenario duration.
- Same broad event-probability profile (correlation > 0.7 between
  Wave-2-hybrid event probs and LIF baseline event probs at matched
  timepoints).
- Same locomotion outcome (forward/reverse/omega bouts within 1 SD
  of LIF baseline).

**Test scenarios (priority order):**

1. `spontaneous` (no stim) — baseline behavior. Easy to fail; no
   external drive means the simulator's intrinsic dynamics speak.
2. `touch_anterior` — ALM/AVM → AVA reversal pathway. AVA is now
   Wave-2-cellular; this is the first scenario where the upgrade
   bites.
3. `osmotic_shock` — ASH → AVA/AIB → reversal. RIM tyramine
   modulation matters here. RIM is now Wave-2-cellular.
4. `food` — NSM 5HT → dwelling. AIY is Wave-2-cellular and
   participates in food signaling.
5. `chemotaxis`, `aerotaxis` — slower modulator-driven scenarios.
   Higher tolerance for divergence.

**Failure modes (Layer A):**

| Symptom | Plausibility | Diagnosis next step |
|---|---|---|
| FSM state distribution shifts >>5% | likely (Wave 2 cells have different effective conductance at rest than LIF spike-output) | Tune the "release event" calibration on Wave 2 cells |
| Event probs uncorrelated | likely if release-event mapping is wrong | Bisect: AVA Wave 2 only, then add AIY, then RIM |
| Behavior absent (no reversal at all) | indicates broken coupling | Check connectome wiring, confirm spike events from Wave 2 cells reach LIF targets |
| dt mismatch artifacts | possible | Confirm per-group clocks set, or unify dt |

### 4.2 — Layer B network: integration produces biologically meaningful behavior

**Reference target:** Wave-2-hybrid produces behavior that is **more
faithful to literature** than LIF baseline on the targets where
Wave 2 cells matter:

- AVA plateau dynamics under sustained ASH stim. LIF AVA produces
  high-rate firing (artifact); Wave 2 AVA should plateau then
  inactivate (Mellem 2008). Specifically:
  - 5 s sustained ASH stim → AVA voltage trace shows initial
    depolarization to ~+5 mV → plateau ~3 s → inactivation back
    to baseline.
  - LIF baseline produces flat ~22 Hz firing throughout.
- AIY graded response to food signal. LIF AIY fires at flat rate;
  Wave 2 AIY should show smooth voltage modulation tracking input
  intensity (Clark 2006, Beverly 2011).
- RIM tyramine modulation: not directly testable at the cellular
  layer (modulator layer handles release kinetics), but RIM voltage
  trace under aversive stim should match RIM characterizations
  (sustained bursts during reversal, silent during forward).

**Test scenarios (priority order):**

1. **AVA voltage trace under ASH stim** — direct comparison vs
   Mellem 2008 Figure 2 traces. Single-cell-level reference target.
2. **AIY response gradient to food intensity** — sweep
   `food_signal` intensity from 0.2 to 1.0; AIY should show
   monotonic graded response.
3. **RIM activity correlation with reversal bouts** — under
   `osmotic_shock`, RIM should be active during REVERSE FSM state,
   silent during FORWARD.

**Failure modes (Layer B):**

| Symptom | Plausibility | Diagnosis next step |
|---|---|---|
| AVA doesn't plateau in network context | possible (sufficient ASH→AVA drive needed; depends on coupling) | Check effective ASH→AVA current at Wave 2 AVA membrane |
| AVA plateaus but doesn't inactivate | possible (h-inactivation params correct in Wave 2 cell?) | Cellular-level Wave 2 validation has shown AVA inactivates in isolation; in network it should too unless ongoing drive prevents it. Verify against Wave 2 standalone trace. |
| AIY response is binary not graded | likely if AIY's dynamic range is being clipped by LIF-style release | This is a mild form of failure; investigate the release event mapping |
| Network goes wild (saturates / silent) | medium-likely (interaction effects between Wave 2 cells and LIF scaffold not predictable from cellular work) | Expected; iterate on coupling parameters |

### 4.3 — Layer C: F18-style cross-cell numerical sanity (NEW, not in prompt)

Wave 2 cellular validation surfaced F18: asymmetric multi-USEION-ca
declarations trigger NEURON eca override. This is a NEURON-side
issue. Brian2 doesn't have ion_style; eca_mV is constant. **At
network scale**, the analogous risk is not eca per se but:

- **Cross-cell ion accounting.** AIY and RIM both use Ca channels
  but under different reversal potentials (AIY eca=127.59 mV, RIM
  eca=60 mV). In the network, they don't interact via shared Ca
  pool (Brian2 cells don't share state except via Synapses), so this
  is fine. **Layer C check: confirm eca_mV is per-cell-type, not
  global.**
- **Per-cell-type parameter visibility in shared Brian2
  namespace.** When multiple NeuronGroups exist with overlapping
  variable names (e.g., g_egl19_Scm2 in AVAL group and AIY group),
  Brian2 keeps these per-group. Confirm this works as expected with
  cython codegen.

This layer is a single test, not a battery: instantiate the hybrid
brain, confirm AIY's eca and RIM's eca are independent, and confirm
no shared-namespace surprises. Likely passes by construction; worth
checking once.

### 4.4 — Validation prioritization

**Layer A first** (integration didn't break anything). If Layer A
fails, Layer B failures are uninformative — they could be coupling
failures masquerading as cellular failures.

**Layer C as a sanity gate** before Layer A: 30 minutes of work,
catches systematic namespace issues early.

**Layer B last** (integration adds value beyond LIF). Once Layer A
passes, Layer B speaks to the actual scientific motivation for
Phase δ — does the cellular detail change the network's emergent
behavior in ways that are biologically meaningful?

---

## Section 5 — Phase δ work block decomposition

Decomposition assumes **Alternative B (hybrid)** with Alternative-C
staging (one cell at a time first). Each work block fits ~1-3 hours
single-session execution, produces output the next consumes, has an
explicit gate to determine whether to proceed.

### 5.1 — Work block list (sequential)

**WB1 — Integration prep + namespace audit + cython unification**

*Concrete deliverables:*

- `wave2/integration/` directory created.
- `wave2/integration/check_namespace_compat.py` — instantiate Wave 2
  AVAL factory + LIFBrain in same Python process; check that
  `prefs.codegen.target` is consistently `cython` and confirm both
  build without errors.
- Decision: change LIFBrain/GradedBrain `prefs.codegen.target` to
  `"cython"` (or remove the line). One-line change × 2 files +
  smoke test that LIFBrain.smoke_test() still passes under cython.
- `wave2/artifacts/integration_namespace_findings.md` — document
  shared-namespace surprises if any.

*Gate:* both Wave 2 AVAL factory and LIFBrain instantiate cleanly
in same process under cython. Existing LIFBrain.smoke_test()
unchanged behavior. **If gate fails: stop, surface for review.**

*Dependencies:* none.

---

**WB2 — Wave2HybridBrain class skeleton (no Wave 2 cells yet)**

*Concrete deliverables:*

- `wave2/integration/wave2_hybrid_brain.py` — `Wave2HybridBrain`
  class. Initially identical-behavior to LIFBrain (no Wave 2 cells
  embedded), exposing the ClosedLoopEnv I/O contract:
  `names, idx, N, neurons, spikes, run, time_ms,
  set_proprioception, set_sensory_rate, inject_poisson, ablate,
  ablation_current_pA`.
- Class loads connectome.npz, builds one LIF NeuronGroup of 300,
  builds Synapses, exposes the contract. Internal architecture
  designed to support adding Wave 2 cell groups (placeholder list
  `self.wave2_groups: list[Brian2Group]`).
- ClosedLoopEnv accepts `brain_class='wave2_hybrid'` and
  instantiates this. All scenario regeneration tests pass with
  identical-or-near-identical output (5% tolerance — same numerical
  difference as cython vs numpy LIF).

*Gate:* `python closed_loop_env.py` smoke test passes with
`brain_class='wave2_hybrid'` and produces qualitatively similar
behavior to default LIF. **Specifically:** 10 s spontaneous run,
state distribution within 10% of LIF baseline.

*Dependencies:* WB1.

---

**WB3 — AVA pair as Wave 2 cells in hybrid brain**

*Concrete deliverables:*

- `Wave2HybridBrain` instantiates AVAL and AVAR via
  `option_alpha_ava_cell.build_brian2_aval_4channel`. The other 298
  cells stay in the LIF NeuronGroup.
- Connectome edges where AVAL or AVAR is pre or post: re-wire to
  go between LIF group and the AVA Wave 2 cells. Specifically:
  - LIF → AVAL chemical: PoissonGroup-style or Synapse with on_pre
    rule that delivers a current pulse to AVA's I_inj.
  - AVAL → LIF chemical: define spike-detection threshold on AVA
    voltage (proposal: V crosses -25 mV with 5 ms refractory). On
    detected event, deliver `v_post += W_syn * w` to LIF target.
  - LIF ↔ AVAL gap: Brian2 `(summed)` Synapses, `g_gap * w_gap *
    (v_pre - v_post)` works as long as both groups expose `v` in
    volts. Need to confirm AVA's `v_mV = v / mV : 1` aux variable
    doesn't conflict.
  - AVA ↔ AVAR gap: same group-to-group Synapses.
- Wave 2 AVA cells' I_ext receive modulation from ModulationLayer
  (rename their I_inj to I_ext or have ModulationLayer write to
  both).
- Sensory injection to AVA (rare) — handle via `set_sensory_rate`
  routing to AVA cells when name is "AVAL" or "AVAR".

*Gate:* Layer A validation for `touch_anterior` scenario. State
distribution within 10% of LIF baseline; reversal bouts present at
expected times. **Layer B target:** AVA voltage trace shows
plateau-then-inactivation under 5 s sustained ASH stim (Mellem
2008-style).

*Dependencies:* WB2.

---

**WB4 — AIY pair extension**

*Concrete deliverables:*

- Add AIYL, AIYR via `build_brian2_aiy_*` factory.
- Connectome re-wiring for AIY edges (AIY has many fewer high-weight
  edges than AVA; lighter task than WB3).
- Spike-detection threshold for AIY: AIY plateaus less dramatically
  than AVA; possibly use σ-style Boltzmann release (graded
  release) instead of threshold spike. Decide and implement.
- Re-validate Layer A scenarios (now AVA + AIY are Wave 2).

*Gate:* `food` scenario Layer A passes; AIY graded response to
food intensity sweep is monotonic (Layer B).

*Dependencies:* WB3.

---

**WB5 — RIM pair extension**

*Concrete deliverables:*

- Add RIML, RIMR via `build_brian2_rim_*` factory.
- Connectome re-wiring (RIM has dense connections — UNC-2 P/Q-type
  Ca and CCA-1 T-type matter for fast transient release;
  consider this in spike-detection design).
- Modulation: RIM is itself a tyramine-releasing neuron; verify
  modulator-tables release weights apply correctly to Wave 2 RIM
  cells.
- Re-validate all Layer A scenarios (now all three are Wave 2).
- Layer C namespace audit: confirm AIY eca=127.59 mV and RIM eca=60
  mV are independent across the network.

*Gate:* `osmotic_shock` Layer A passes. Layer B: RIM voltage
correlates with REVERSE FSM state (active during reversal).

*Dependencies:* WB4.

---

**WB6 — Phase δ summary + Layer A/B trace artifacts**

*Concrete deliverables:*

- Run all 6 standard scenarios (spontaneous, touch, osmotic_shock,
  food, chemotaxis, aerotaxis) under `wave2_hybrid` and produce
  output JSON traces.
- Comparison report: state distributions, event probability
  correlations, AVA/AIY/RIM voltage trace excerpts vs LIF
  baseline.
- `wave2/artifacts/phase_delta_summary.md` — outcome doc.

*Gate:* none — this is the documentation work block.

*Dependencies:* WB5.

---

### 5.2 — Critical path and parallel-able work

**Critical path:** WB1 → WB2 → WB3 → WB4 → WB5 → WB6. Strict
sequencing because each work block depends on the prior's hybrid-
brain state.

**No work is parallel-able within Phase δ.** The hybrid-brain class
is a single object with single state; its internal structure
evolves monotonically across WB2-WB5.

**Possible parallel work in subsequent waves (NOT Phase δ):**
RMD acquisition (Nicoletti 2019 paper) → cellular validation work
block → eventual addition to hybrid brain in a Wave 3 work block.

### 5.3 — Estimated total scope

**Six work blocks, ~6 sessions of focused 1-3 hour work.**

This is "small Phase δ" — the goal is a working hybrid brain with
3 Wave 2 cells embedded, validated against existing scenarios.

If Phase δ scope expands to include re-tuning of LIF parameters or
modulator strengths post-integration, add WB7-WB8. If it expands to
include new Wave 2 cells (RMD, etc.), those are Wave 3 work blocks.

If WB3 reveals that the spike-detection mapping is subtle and needs
its own diagnostic work block (likely), insert WB3.5: "Wave 2 cell
release-event mapping calibration". This brings total to 7 work
blocks.

### 5.4 — Decision gates summary

| WB | Gate criterion | Action if fails |
|---|---|---|
| WB1 | Cython unification + smoke pass | Surface for review; Phase δ pauses |
| WB2 | LIF-equivalent hybrid brain works | Surface; debug class skeleton |
| WB3 | touch + AVA Layer A + Layer B plateau | Iterate on AVA release-event mapping; possibly insert WB3.5 |
| WB4 | food + AIY Layer A + Layer B graded response | Iterate on AIY release; reduce LayerB threshold if marginal |
| WB5 | osmotic_shock + RIM Layer A + Layer B reversal corr | Iterate on RIM release; check namespace collisions |
| WB6 | Documentation only | n/a |

---

## Section 6 — Recommended Phase δ trajectory

### 6.1 — Recommended alternative

**Alternative B (hybrid Wave 2 + LIF), with Alternative C staging
(integrate cells one at a time within B).**

Rationale:

- Alternative A is structurally infeasible without ~297 additional
  cell-validation work blocks.
- Alternative B is the "obvious" architecture and reduces to "one
  Brian2 Network with multiple NeuronGroups" — a well-supported
  Brian2 idiom.
- Alternative C as standalone is illusory (it reduces to B); but
  Alternative C as a *staging strategy within B* is exactly how
  the integration risk should be decomposed.
- The result: 3 Wave 2 cells (AVAL, AVAR), then 5 (+AIY), then 7
  (+RIM), each at a checkpoint. ClosedLoopEnv I/O contract
  preserved throughout. LIF scaffold for the 293-294 unvalidated
  cells preserved throughout.

### 6.2 — First work block to deploy

**WB1 — Integration prep + namespace audit + cython unification.**

Single session, ~1-2 hours. Concrete deliverables:

1. `wave2/integration/` directory.
2. Namespace compatibility check script.
3. Decision-recorded one-line cython-target update for LIFBrain
   and GradedBrain.
4. Findings doc.

Gate: Wave 2 AVAL factory + LIFBrain coexist in same Python
process under cython, no namespace surprises, LIFBrain smoke test
unchanged.

**Why this first:** the Wave 2 cells use cython; LIFBrain is
hardcoded numpy. The two cannot coexist in the same Python process
as currently configured. This is a 5-minute fix but it gates
everything downstream. It's the load-bearing pre-flight for Phase
δ.

### 6.3 — Decision gates for subsequent work blocks

See §5.4. The structural pattern is:

- Each WB has one Layer A gate (didn't break LIF baseline) and
  optionally one Layer B gate (Wave 2 cell adds biological fidelity).
- WB-level gate failures are diagnosed before advancing — never
  "claim it works and move on" pattern.
- F18-style methodology lock-in applies: each WB's findings get a
  named entry in a Phase δ findings doc, mid-flight.

### 6.4 — Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wave 2 cell "release event" definition is ambiguous and requires per-cell calibration | High | Medium | Plan for WB3.5 (release event calibration) as a contingency. Document calibration choices in findings. |
| Brian2 multi-clock scheduling has surprises with cython + Synapses | Medium | Medium | Test in WB1 with toy 2-cell setup before WB2. Has known idioms in Brian2 docs. |
| Wave 2 cell's continuous voltage doesn't drive LIF cells the way ASH→AVA pathway expects (e.g., AVA plateaus at +5 mV but LIF expects 100Hz spike train) | High | High | This IS the central design surface. Layer B validation directly tests it. Mitigation: graded-release Boltzmann mapping rather than threshold spike. |
| ModulationLayer writes to `I_ext` but Wave 2 cells use `I_inj`. Naming collision. | Certain | Low | Rename in Wave 2 cell factories to expose I_ext alias, OR change ModulationLayer to write both. 5-min fix. |
| dt=0.025 ms (Wave 2) vs dt=0.1 ms (LIF) sync issues | Medium | Medium | Use Brian2 per-NeuronGroup `clock` keyword. Standard pattern. |
| Layer A scenarios produce qualitatively different behavior under Wave 2 hybrid vs LIF baseline (e.g., FSM state distributions diverge) | High | Medium-High | Expected to some degree — document delta as a positive finding (Wave 2 cells change behavior in ways consistent with biology). Falsifiable: if delta is unbounded chaos rather than systematic biological enrichment, the integration is wrong. |
| F19 (KQT-1 slow-gate drift in AIY at -15 pA) surfaces in network context | Low | Low | KQT-1 drift was specific to long sustained sub-threshold injection; in network context, AIY cycles between drive states, accumulated drift unlikely to manifest. Defer mitigation unless surfaces. |
| Cython compile cache thrashing across multiple NeuronGroups slows iteration | Medium | Low | Standard Brian2 cython behavior; first run is slow, subsequent runs cached. Document if surfaces. |
| Connectome edges where Wave 2 cells are pre or post are mis-rewired | Medium | High | WB3 is the first place this surfaces. Add a smoke test that compares pre-integration LIFBrain edge counts to post-integration hybrid-brain edge counts (per-pair-of-groups breakdown). |
| Phase δ surfaces that Wave 2 cells don't actually improve network behavior on Layer B targets | Medium | Medium | This is a publishable methodology finding either way (pro or con). Document carefully. |

### 6.5 — Success criteria for Phase δ overall

Phase δ succeeds when:

1. `Wave2HybridBrain` class exists, exposes ClosedLoopEnv I/O
   contract, embeds AVAL/AVAR/AIYL/AIYR/RIML/RIMR as Wave 2
   NeuronGroups, runs LIF for the other 294 cells, all in one
   Brian2 Network.
2. ClosedLoopEnv accepts `brain_class='wave2_hybrid'` and produces
   output JSON traces structurally identical to LIF/Graded.
3. Layer A: 6 standard scenarios produce FSM state distributions
   within 10% of LIF baseline (or, where divergent, the divergence
   is documented as biologically motivated).
4. Layer B: at least one of {AVA plateau, AIY graded response, RIM
   reversal correlation} demonstrably improved over LIF baseline.
5. Compute: 60 s simulated scenario completes in ≤ 15 minutes
   wall-clock under cython.
6. Layer C namespace audit clean.
7. `wave2/artifacts/phase_delta_summary.md` exists, documents
   outcome.

Phase δ explicitly does NOT need to:

- Add new Wave 2 cells beyond the 3 already validated.
- Re-tune LIF parameters or modulator strengths globally.
- Replace the LIF scaffold for the 294 unvalidated cells with
  anything more sophisticated.
- Validate against new biological references beyond what Wave 2
  cellular work already validated.

These belong to Wave 3 or later phases.

### 6.6 — Phase δ scope summary

**6 work blocks (possibly 7 with WB3.5 release-event calibration).
~6-12 hours total focused work distributed across multiple
sessions. Single integration architecture. Hybrid Wave 2 + LIF
Brian2 brain. 3 Wave 2 cells embedded (6 NeuronGroups: L+R for
each). 294 LIF scaffold cells. ClosedLoopEnv I/O contract
preserved.**

**First action: WB1 — namespace audit + cython unification.**
Other actions follow gates.

---

## Appendix — file-path index

Production simulator (verified primary source):

- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/lif_brain.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/graded_brain.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/graded_brain_h_kca.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/closed_loop_env.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/modulation_layer.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/behavioral_fsm.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/activity_fsm.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/sensory_injection.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/sensory_transduction.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/run_perturbations.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/compartmental_neurons_kca.py`
  (Wave 1 sandbox, NOT production-wired)

Connectome artifact:

- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/artifacts/connectome.npz`
  (300 neurons, 12 arrays incl W_chem_raw, W_chem_per_edge, W_gap)

Wave 2 cellular layer (Phase δ inputs):

- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/option_alpha_ava_cell.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/option_alpha_aiy_cell.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/option_alpha_rim_cell.py`
- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/channels/`
  (14 channels)

Phase δ output target:

- `/home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/integration/`
  (to be created in WB1)

Output JSON scenarios (Layer A reference):

- `/home/rohit/Desktop/website/personalwebsite/public/data/wormbody-brain-{spontaneous,touch,osmotic_shock,food,chemotaxis,aerotaxis}.json`
