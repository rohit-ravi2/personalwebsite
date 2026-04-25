# Tier 2 / Tier 4 execution plan

*Living document. Updated at phase boundaries.*

Companion to `docs/current-state-summary.md` (narrative state) and
`scripts/brain/artifacts/phase0_baseline_report.md` (measured
baselines + ratified thresholds). This file describes the **execution
sequence** with concrete file paths, entry/exit criteria, and
compute budgets.

## Sequencing decision (2026-04-21)

Original T4 plan (drafted pre-Phase-0) put T4-4 CeNGEN-conductance
mid-sequence and merged T2-#4 into T4-3. User agreed to the
counter-sequence after push-back: low-compute-first subject to T4-3
being foundational. The agreed order is:

0. **Phase 0** — baseline measurement + audit infra (1 week, ~12 hrs compute)
1. **Phase 1** — independent low-compute prep (2-3 weeks, ~4 hrs compute)
2. **Phase 2** — compartmental integration + plateau calibration (2-3 weeks)
3. **Phase 3** — T4-3 synaptic calibration, the T0 fix (3-4 weeks, highest per-phase compute)
4. **Phase 4** — T4-1 motor coupling validation (2 weeks)
5. **Phase 5** — T4-4 CeNGEN + T4-5 INS overlays (3 weeks)
6. **Phase 6** — T4-6 trajectory correlation capstone (2 weeks)

Total: 14-17 weeks at observed velocity.

## Compute budget (ratified against 3.06× wall/sim ratio)

| phase | per-phase compute | gating audit |
|---|---|---|
| 0 | ~12 hrs (W0.2 phenotype + W0.3 scenario) | — |
| 1 | ~4 hrs (T4-2 standalone calibration grid + Tierpsy reference build) | — |
| 2 | ~6 hrs (post-integration drift check, n=5 × 60s) | n=10 × 60s phenotype (~6 hrs) |
| 3 | ~12 hrs (ActivityFSM validation × 2 modes) | n=10 × 60s phenotype + n=10 × 60s scenario (~9 hrs) |
| 4 | ~3 hrs (curvature ρ validation, 6 scenarios × 10 seeds) | scenario audit (~3 hrs) |
| 5 | ~18 hrs (baseline + T4-4 + T4-5 interaction matrix) | 2× phenotype audit (~12 hrs) |
| 6 | ~3 hrs (trajectory stats pipeline on existing outputs) | — |
| **total** | **~58 hrs dedicated compute across phases** | across 14-17 weeks |

## Phase 1 — Independent low-compute prep

**Goal:** three tasks that don't depend on the T0 fix and can parallelize.

### T2-#4 sensory cascade calibration (~2 weeks, ~10 min compute)

- **Prerequisite:** digitise 5 published ΔF/F traces into
  `scripts/brain/references/<paper>/trace.csv` via WebPlotDigitizer
  (Thiele 2009, Chalasani 2007, Hilliard 2005, Clark 2006, O'Hagan 2005).
  Meta.json stubs present; expected user time ~2 hrs.
- **Run:** `python scripts/brain/phase0_ingest_references.py` →
  compiles `trace.npz` per paper.
- **Fit:** `python scripts/brain/phase1_calibrate_cascades.py --cascade all`
  → grid-search Frechet distance per cascade.
- **Exit:** post-fit Frechet ≤ 0.10 per cascade; improvement ≥ 0.05
  over defaults. No regression on touch/osmotic/salt/food scenarios
  in the default phenotype audit with `sensory_mode=transduction`.

### T4-2 standalone plateau calibration (~1-2 weeks, ~2 hrs compute)

- **Baseline:** `phase0_plateau_baseline.csv` shows 2/15 pass. AVA
  v_d peak = +4.5 mV (target +20 mV), duration 0 ms (target 600 ms).
- **Fit:** `python scripts/brain/phase1_plateau_calibrate.py
  --grid full` → per-neuron best (g_ca, tau_h, v_ca_half).
  Full grid = 320 combos × 10 neurons × ~1 s per combo ≈ 55 min.
- **Exit:** AVA plateau duration ∈ [480, 720] ms, amplitude ∈ [18, 22]
  mV above rest. Non-plateau neurons (AWC/RMG/ALA) show no sustained
  depolarisation. Paste-back suggested roster params into
  `compartmental_neurons.py:COMPARTMENTAL_ROSTER`.

### T4-1 MJCF refactor + muscle driver (~1 week, minimal compute)

- **Done in Phase 0:** `build_wormbody_v3.py` emits `wormbody_v3.xml`
  with 19 legacy position actuators + 76 Hill `<muscle>` actuators on
  `<tendon><spatial>` spans. `muscle_driver.py` transforms brain
  motor rates → 76 muscle ctrls via the 540-weight innervation matrix.
- **Integrate:** add `body_mjcf="wormbody_v3.xml"` and
  `body_driver="muscle"` flags to `ClosedLoopEnv`. When both are set,
  replace `cpg_ctrl()` with `muscle_driver.step(brain_rates)` applied
  to the muscle_i_Q ctrls. Keep position-actuator path as fallback.
- **Tierpsy reference pool (done in Phase 0):** 5 streaks, 1207 frames
  max, 25 Hz sampling — `artifacts/phase1_tierpsy_reference_pool.npz`.
- **Exit:** MJCF loads cleanly (verified: 21 bodies / 95 actuators /
  76 tendons). `muscle_driver._smoke_test()` produces non-zero forward
  and reverse drive activations (verified).
- **Validation deferred to Phase 4** (needs Phase 3's fixed command
  cascade to drive the body meaningfully).

## Phase 2 — Compartmental integration + plateau calibration

**Goal:** `LIFBrain.replace_neurons_with_compartmental(["AVAL",
"AVAR", ...])` that substitutes the 15 plateau-expressing neurons
with their soma+dendrite equivalents, preserving connectome wiring.

### The architectural refactor

Single-compartment LIF group currently has 300 neurons indexed 0..299.
Compartmental integration adds 15 neurons × 2 compartments = 30 new
Brian2 state entries. Options:

1. **Two-group design** (recommended): keep `self.neurons` (LIF, N=300)
   and add `self.comp_neurons` (compartmental, N=15). The 15 LIF
   indices corresponding to the compartmental neurons are "silenced"
   via ablation-style hyperpolarisation so they don't spike. All
   incoming/outgoing synapses are redirected:
   - LIF → LIF (285 × 285): unchanged existing Synapses
   - LIF → Comp (pre ∈ LIF, post ∈ Comp.soma): new Synapses group
   - Comp → LIF (pre ∈ Comp.soma, post ∈ LIF): new Synapses group
   - Comp → Comp: internal soma-to-soma routing within comp_neurons
2. **Single-group design**: extend `self.neurons` to N=315, with
   compartmental state variables applied only to indices ≥ 300.
   Tricky because Brian2 model equations are uniform per group.

Option 1 is architecturally cleaner; implementation in `lif_brain.py`:
- New method `replace_neurons_with_compartmental(names: list[str])`
- New attribute `self.comp_group`, `self.comp_idx: dict[name, int]`
- New Synapses: `syn_lif_to_comp`, `syn_comp_to_lif`, `syn_comp_gap`
- Apply ablation current to the original LIF indices

### Post-integration drift check

After integration, run a 30s spontaneous simulation at one seed, compare
per-neuron firing rates to pre-refactor. Expected: non-plateau neurons
≤ 15% drift; plateau neurons may drift more (by design).

### Exit criteria

- Standalone: `phase1_plateau_calibrate.py` reports AVA duration ∈
  [480, 720] ms. Passed before integration.
- In-network: same protocol run on an integrated network: plateau
  survives inhibitory inputs. Duration ≥ 400 ms (allow some shrinkage).
- Rate-drift check: non-plateau neurons' firing rates ≤ 15% drift
  across all 6 scenarios at single seed.

## Phase 3 — T4-3 synaptic calibration

**Goal:** tune W_syn such that ALM→AIB→AVA cascade produces
ΔAVA ≥ +15 Hz on touch_anterior stim.

### Step 1: localise the break

Before tuning weights, determine which edge in the cascade is failing.
New script `scripts/brain/phase3_cascade_diagnostic.py`:
- Direct ALM current injection → measure ALM, AIB, AVA
- Direct AIB current injection → measure AVA (bypasses ALM)
- Direct AVA current injection → measure RIM, AVD, AVE
- Compare each link's transmission efficacy.

Based on Phase 0 touch-scenario rates: ALM→AIB looks weak (AIB
shows ~flat response). AIB→AVA unknown until diagnostic runs.

### Step 2: constrained optimization

Free parameters:
- `W_syn` (global chemical weight scale)
- Per-class multipliers: sensory→interneuron, interneuron→command
- Inhibitory gain on AVA specifically (currently masks cascade)

Fixed constraints:
- Preserve Phase 0's current GNCA-derived relative weight structure
  (weights from `connectome.npz:W_chem_raw`).
- Don't break spontaneous AVA rate (keep < 10 Hz at rest).

Optimizer: Nelder-Mead over 3-5 free scalars, fitness = weighted sum
of (AVA_peri_hz target, spontaneous_AVA_hz target, AVA_peri_minus_pre
target). Expect ~100-300 evaluations × 30s per eval × 3.06 ratio ≈
2.5-8 hrs.

### Step 3: ActivityFSM validation

With tuned weights, run `FSM_MODE=activity` touch audit at n=10 × 60s.
Exit: ΔREV ≤ −0.40, all 10 seeds negative, 95% CI excludes zero.

### Step 4: cross-scenario preservation

Same calibration must not break:
- Spontaneous AVA rate (< 10 Hz)
- Osmotic AVA response (should still fire)
- Food AVB tonic drive preserved

## Phase 4 — T4-1 motor coupling validation

**Goal:** muscle-driven forward bout matches Tierpsy curvature at
median ρ ≥ max(0.6, CPG_baseline + 0.15).

### Step 1: CPG baseline (Phase 0 output)

`artifacts/phase0_scenario_traces/spontaneous_seed{42..51}.npz` has
body_xy trajectories. For each forward-bout window, extract per-segment
curvature and cross-correlate against `phase1_tierpsy_reference_pool.npz`.
Yields CPG baseline ρ distribution.

### Step 2: muscle-driver validation

With `ClosedLoopEnv(body_mjcf="wormbody_v3.xml", body_driver="muscle")`
and Phase 3's calibrated brain:
- Run forward scenario at n=10 × 60s
- Extract forward-bout windows (FSM state = FORWARD)
- Compute per-segment curvature ρ vs Tierpsy pool
- Compare distribution to CPG baseline

### Exit

- Median ρ ≥ max(0.6, CPG_baseline + 0.15)
- Per-segment ρ > 0 for at least 16/20 segments
- No catastrophic regression (any segment ρ < 0.2 flagged)

## Phase 5 — T4-4 CeNGEN conductance + T4-5 INS expansion

### T4-4 CeNGEN-conductance coupling

- Extract channel-gene TPMs per neuron from `public/data/cengen-panel.json`
- Scale per-neuron intrinsic conductances by TPM (ratio to reference
  neuron — AVA with Mellem 2008 values).
- Reference-case 141 non-CeNGEN-profiled neurons: hold at roster defaults.

### T4-5 INS-family expansion

Add 6 INS peptides to `modulation_layer.py` + `modulator_tables.npz`:
INS-1, INS-6, INS-7, INS-17/18, INS-22, DAF-28. Each with:
- Releaser neurons from CeNGEN (ASI, ASJ, URX, BAG, ALA, DAF-2
  downstream)
- Target neurons via DAF-2 receptor expression
- Slow modulation of K+ leak (or voltage threshold) — 10s-100s τ

### Interaction tests

INS-22 ablation under osmotic_shock: ΔQUI ≥ +0.15 (inverse RIS).
INS-7 ablation under olfactory learning (new scenario TBD).
INS-1 ablation under food deprivation.

### Exit

- Phase 3 phenotype reproductions preserved (AVA/Chalfie still passes).
- At least 2/3 INS phenotype tests pass at n=5/10 seeds.
- T4-4 preservation: inter-seed variance in AVA/AVE/AVD firing changes
  by ≤ 30% relative to Phase 3 baseline.

## Phase 6 — T4-6 trajectory correlation capstone

- Extract per-event windows (reversal onset, forward run, omega, quiescence)
  from Atanas recordings (10 worms, pre-parsed to
  `artifacts/atanas_worm_0{1-10}.npz`).
- Match to equivalent simulator windows across all 6 scenarios × 10 seeds.
- For each of the 18 validated-readout neurons × 4 event types,
  compute cross-correlation ρ between real ΔF/F and simulator synthetic
  calcium (from full_raster in scenario traces).
- PCA comparison: simulated 300-neuron activity vs Atanas (after
  resampling simulator to 1.67 Hz to match Atanas sampling).

### Output

Distribution: per-neuron × per-event ρ. The paper-reportable claim is
a distribution, not a threshold.

## Changelog

- 2026-04-21: document created at Phase 0 completion. Phase 0 measured
  3.06× wall/sim ratio, 2/15 plateau pass, swap-jitter σ=4.75ms
  (within tolerance). Tier 2/4 thresholds ratified.
