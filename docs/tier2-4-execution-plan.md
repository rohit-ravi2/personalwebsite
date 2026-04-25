# Tier 2 / Tier 4 execution plan

*Living document. Updated at phase boundaries.*

Companion to `docs/current-state-summary.md` (narrative state) and
`scripts/brain/artifacts/phase0_baseline_report.md` (measured
baselines + ratified thresholds). This file describes the **execution
sequence** with concrete file paths, entry/exit criteria, and
compute budgets.

**2026-04-25 update — partially superseded.** Phase 3 (originally
"T4-3 synaptic calibration, the T0 fix") is no longer accurate.
T0 was resolved at the architectural level by per-edge sign
convention, not weight calibration; the operative cascade is
ALM/AVM → PVC → AVD/AVE → AVA (not ALM→AIB→AVA). Phase 3 has
been rewritten to reflect the actual resolution path and the
follow-on questions it surfaced. Sequencing and compute budgets
in §0 are also revised. Phases 1, 2, 4, 5, 6 are unchanged in
intent (they describe sub-systems that the T0 resolution does not
affect: T2-#4 sensory cascades, T4-2 plateau dynamics, T4-1 motor
coupling, T4-4/T4-5 modulator overlays, T4-6 trajectory
correlation). Canonical T0 record:
`docs/t0_resolution_report.md`.

## Sequencing decision (revised 2026-04-25)

Original T4 plan (drafted pre-Phase-0) put T4-4 CeNGEN-conductance
mid-sequence and merged T2-#4 into T4-3. The April 21 sequencing
treated T4-3 as foundational and budgeted it at 3-4 weeks of focused
synaptic weight calibration. **That framing is now obsolete.** T0
turned out to be a sign-convention default issue, not a weight-
calibration issue, and was resolved by flipping a constructor flag
(`use_per_edge_glu_signs=True`). The revised order is:

0. **Phase 0** — baseline measurement + audit infra (1 week, ~12 hrs compute) — COMPLETE
1. **Phase 1** — independent low-compute prep (2-3 weeks, ~4 hrs compute) — partially done
2. **Phase 2** — compartmental integration + plateau calibration (2-3 weeks) — independent of T0
3. **Phase 3** — T0 resolution + follow-on calibration (cascade fix landed 2026-04-25; PVC/AVB and FSM recalibration follow-ons pending, ~3-6 weeks)
4. **Phase 4** — T4-1 motor coupling validation (2 weeks) — independent of T0
5. **Phase 5** — T4-4 CeNGEN + T4-5 INS overlays (3 weeks)
6. **Phase 6** — T4-6 trajectory correlation capstone (2 weeks)

Total: still ~14-17 weeks at observed velocity, but Phase 3's
character has shifted. Earlier framing assumed Phase 3 would be
3-4 weeks of intensive weight tuning with the highest per-phase
compute budget. Actual Phase 3 (the per-edge sign convention work)
was a single sweep + diagnostic block; Phase 3's remaining time is
in PVC/AVB literature investigation and FSM/classifier
recalibration, neither of which is compute-heavy.

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

## Phase 3 — T0 resolution + follow-on calibration

**2026-04-25 update — this phase has been substantially restructured.**

The original Phase 3 framing ("tune W_syn such that ALM→AIB→AVA
cascade produces ΔAVA ≥ +15 Hz on touch_anterior stim") was wrong
about which cascade was operative AND about what fix category was
needed. Both points were established in the 2026-04-25 T0 diagnostic
block. The original step-by-step plan (cascade-diagnostic localization,
Nelder-Mead weight optimization, ActivityFSM validation, cross-scenario
preservation) is preserved at the bottom of this section as a
historical record but is no longer the active plan.

Canonical record of what actually happened: `docs/t0_resolution_report.md`.

### What was actually done (2026-04-25, COMPLETE)

The cascade-firing question was resolved at the architectural level
by flipping the simulator's glutamate sign convention from per-
presynaptic-neuron NT-sign (Glu = −1 with hand-picked overrides) to
per-edge CeNGEN-derived postsynaptic-receptor signs:

- Constructor flag `use_per_edge_glu_signs=True` on LIFBrain (already
  in codebase, off by default).
- Switching this flag flips ~518 chemical edges (14% of total) where
  glutamate sources target iGluR-dominant postsynaptic neurons.
- Under per-edge mode, the operative touch cascade
  (ALM/AVM → PVC → AVD/AVE → AVA, with PVC as the load-bearing
  first-stage relay) fires at +60 Hz on touch with seed-to-seed
  variance under 1.5 Hz across n=10.

Two suspects were falsified along the way:
- Voltage regime (no-op for LIF dynamics under coordinate translation;
  voltage fix kept in place for biological documentation).
- Gap-junction conductance (increasing g_gap monotonically silenced
  the network via noise averaging).

Goal achievement vs original Phase 3 exit criteria:
- Original: AVAL peri ≥ 20 Hz AND Δ ≥ +15 Hz on ≥ 8/10 seeds.
- Achieved: AVAL peri = 97 Hz, Δ = +60.3 Hz on 10/10 seeds.

### Follow-on questions (Phase 3 remaining work, 2026-04-25 → ongoing)

The architectural fix surfaced new questions that were not in the
original Phase 3 scope. These are the active work items:

**3a — PVC/AVB handling under per-edge mode** (~1-2 weeks).
Under per-edge, PVC fires Δ +60-70 Hz and AVB fires Δ +51-57 Hz on
touch. Canonical biology has anterior touch suppressing forward
locomotion. Two interpretations open:

- **A) CeNGEN expression-vs-function mismatch.** PVC has iGluR
  receptors per CeNGEN but the ALM/AVM synapses onto PVC may be
  functionally GluCl-mediated. If true, per-edge needs targeted
  overrides.
- **B) Canonical biology more nuanced than textbook.** PVC excitation
  on anterior touch may be defensible. If true, the per-edge prediction
  stands.

Resolution path: literature dive on PVC functional sign biology;
possibly a per-edge override sweep to test (A).

**3b — FSM/classifier recalibration under per-edge dynamics**
(~2-4 weeks engineering). The 18-readout classifier bank was trained
on default-mode firing distributions. Under per-edge, AVA's dynamic
range tripled and the AVA-ablation effect shifts FSM channels (dREV
→ dPIR; dPIR mean −0.117, 9/10 negative seeds). Three sub-questions:

1. Does AVA-ablation under correct cascade produce the Chalfie
   phenotype through dPIR (preserved under per-edge), or would
   recalibrated thresholds re-route the signal to dREV?
2. Is the existing 18-readout architecture fundamentally
   incompatible with per-edge dynamics, requiring a wider redesign?
3. Bank retraining (deferred during overnight v2 Track B as
   LOGISTICAL_FAILURE) is the technical prerequisite — pooled-target
   data prep, retrain against per-edge-mode synthetic calcium, swap
   bank path in `ClosedLoopEnv`.

**3c — Network-stability scan under per-edge for non-touch scenarios**
(~1 week). Per-edge changes ~14% of chemical edges; touch scenario
validates one regime. Need to verify osmotic_shock, food, chemotaxis,
aerotaxis, spontaneous don't destabilize.

**3d — Per-edge re-runs of audited phenotypes** (~2 weeks compute).
RIS molecular audit, three-mode taxonomy classifications, Mode 3
modulator results all conducted under default mode. Need re-running
to determine which findings transfer.

**3e — Production sign-mode decision.** Per-edge as default, opt-in,
or hybrid (curated per-edge override list). Depends on 3a and 3b
outcomes.

### Exit criteria (Phase 3 follow-on)

- 3a: PVC/AVB interpretation adjudicated; per-edge override list
  defined OR per-edge accepted as biological prediction.
- 3b: AVA-ablation phenotype recovered through some FSM channel
  with similar statistical robustness to default-mode dREV.
- 3c: All 6 scenarios run cleanly under chosen sign mode.
- 3d: Audit findings under per-edge documented; supersessions
  noted.
- 3e: Production sign-mode set; default flag updated; docs
  reflect new default.

### Compute budget revision

Original Phase 3 estimate: 3-4 weeks at highest per-phase compute
(2.5-8 hrs of Nelder-Mead optimization).

Revised: ~3-6 weeks total wall, but compute is much lower.
- 3a: literature time + possibly 1-2 hours per override-test sweep.
- 3b: classifier bank retraining is the only compute-significant
  item (~few hours per retraining iteration).
- 3c: ~3-6 hours total for the 6-scenario sweep at n=10.
- 3d: ~3-6 hours per audit re-run; total depends on which audits
  need re-running.
- 3e: no compute, decision-only.

---

### Historical: original Phase 3 plan (preserved as record)

The original April 21 Phase 3 plan, preserved here verbatim. Active
plan is above; this is for understanding what was previously believed
and how it changed.

> **Goal:** tune W_syn such that ALM→AIB→AVA cascade produces
> ΔAVA ≥ +15 Hz on touch_anterior stim.
>
> **Step 1: localise the break**
> Before tuning weights, determine which edge in the cascade is
> failing. New script `scripts/brain/phase3_cascade_diagnostic.py`:
> direct ALM current injection → measure ALM, AIB, AVA; direct AIB
> current injection → measure AVA (bypasses ALM); direct AVA current
> injection → measure RIM, AVD, AVE; compare each link's transmission
> efficacy. Based on Phase 0 touch-scenario rates: ALM→AIB looks weak
> (AIB shows ~flat response). AIB→AVA unknown until diagnostic runs.
>
> **Step 2: constrained optimization** — Nelder-Mead over 3-5 free
> scalars (W_syn, per-class multipliers for sensory→interneuron and
> interneuron→command, inhibitory gain on AVA), fitness = weighted
> sum of AVA target rates, ~100-300 evaluations × 30s × 3.06×
> ratio ≈ 2.5-8 hrs.
>
> **Step 3: ActivityFSM validation** — n=10 × 60s, exit ΔREV ≤ −0.40.
>
> **Step 4: cross-scenario preservation** — don't break spontaneous
> AVA rate, osmotic AVA response, food AVB tonic drive.

This plan was wrong in two ways: (1) the ALM→AIB→AVA cascade does
not exist in this connectome (AIB has zero chemical edges to AVD
and ALM has no chemical edges to AIB), and (2) the cascade failure
under default sign convention was a sign-assignment problem, not a
weight problem. The cascade-diagnostic script was never run; the
Nelder-Mead optimization was never run; both became unnecessary
once the sign-convention diagnosis was established.

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

- 2026-04-25: Phase 3 substantially restructured to reflect T0
  resolution. Original Phase 3 framing (T4-3 synaptic weight
  calibration as the operative T0 fix; ALM→AIB→AVA cascade)
  preserved as historical record at the bottom of the Phase 3
  section but no longer the active plan. Sequencing decision and
  compute budget revised — Phase 3 character has shifted from
  weight tuning to PVC/AVB resolution + FSM recalibration.
  Phases 1, 2, 4, 5, 6 unchanged. Header note added at top of
  document. Canonical T0 record: `docs/t0_resolution_report.md`.
- 2026-04-21: document created at Phase 0 completion. Phase 0 measured
  3.06× wall/sim ratio, 2/15 plateau pass, swap-jitter σ=4.75ms
  (within tolerance). Tier 2/4 thresholds ratified.
