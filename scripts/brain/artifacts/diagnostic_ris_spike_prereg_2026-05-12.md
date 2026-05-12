# Diagnostic spike — RIS drive viability under M2-pure (pre-registration)

**Date locked:** 2026-05-12
**Scope:** Go/no-go diagnostic for V8 bridge experiment's natural-quiescence arm.
NOT paper-quality statistics.
**Author / session:** Claude Code (Rohit's V3 simulator work, separate session from V7 paper-finishing).
**Status at lock:** pre-registered before any data collection. Refinements R1/R2/R3
and Outcome-B interpretation distinction (B-i vs B-ii) approved by Rohit prior to lock.

---

## 1 · Question

Under the locked M2-pure stack (brain v3.5 per `docs/brain_v3.5_locked.md`),
does driving RIS to suprabaseline firing rates produce:

1. **A1 — FLP-11 release scaling:** monotone-nondecreasing FLP-11 concentration as
   a function of RIS firing rate, consistent with the C-17 linear-rate-coupling
   verification (which was conducted under the legacy default-mode stack).
2. **A2 — Target inhibition:** measurable firing-rate reduction on the
   FLP-11-receptor-expressing target population.
3. **A3 — Global rate reduction:** measurable network-wide mean firing-rate
   reduction.

The diagnostic decides whether the bridge experiment's Condition A
(natural quiescence via RIS → FLP-11 → broad inhibition) is producible in the
current locked-production simulator.

## 2 · Stack lock — exact configuration

Per `docs/brain_v3.5_locked.md` §1.1 and ClosedLoopEnv interface:

```python
LIFBrain(
    use_per_edge_glu_signs=True,
    sign_exceptions={},          # explicit empty — bypasses DOCUMENTED_SIGN_EXCEPTIONS default
)
ModulationLayer attached with default modulator_tables.npz
g_gap = 0.1 nS (T0-resolution baseline)
seed: set per run (Brian2 + numpy synced via brain._brian2_seed)
```

Classifier bank, calibration, and FSM thresholds are **not used** for this spike —
the measurements are raw spike counts and modulator concentrations, not
classifier-derived calcium or FSM state. (See §6 for runner-architecture rationale.)

## 3 · Protocol

### 3.1 Phase A — input-rate → RIS-output-rate calibration

**Purpose:** find Poisson input rates that produce target RIS firing rates of
approximately {1, 5, 12, 22, 35, 50} Hz.

**Method.** For each seed in {0, 1, 2} and each input rate in {25, 50, 100, 200,
400, 800} Hz:
- Run a fresh LIFBrain (M2-pure) for 12s.
- t=0..2s: settle.
- t=2..12s: `inject_poisson("RISL", rate_hz=r, weight_mv=15.0)` and same for RISR.
- Measure mean (RISL + RISR) / 2 firing rate over t=4..12s window (8s window,
  discarding first 2s of stim as settle).

**Output:** RIS rate (Hz) as a function of input Poisson rate (Hz), 3 seeds per
input level. Build the mapping; pick the input rates that get nearest to the
target output bins {1, 5, 12, 22, 35, 50} Hz. The 1-Hz bin uses zero forcing
(natural baseline).

**Stop-and-report conditions for Phase A:**
- Mapping is **non-monotone** (higher input rate produces lower RIS rate at any
  step ≥ 1 σ): stop and report. Do NOT switch to direct I_ext fallback without
  authorization.
- Maximum achievable RIS rate is **< 30 Hz** at the largest input rate tested
  (800 Hz): stop and report. The pre-registered fallback (direct I_ext helper)
  exists but mechanism-switching is a deviation that needs user sign-off before
  proceeding.

### 3.2 Phase B — main spike

For each of 6 drive levels (calibrated input rates from Phase A targeting
RIS bins {1, 5, 12, 22, 35, 50} Hz) × 3 seeds {0, 1, 2}:

- **0..10s:** settle, no forcing.
- **10..25s:** pre-stim baseline. RIS at natural firing rate. Modulator layer
  active.
- **25s:** activate `inject_poisson("RISL", rate=calibrated_input, weight_mv=15)`
  and `inject_poisson("RISR", ...)`.
- **25..30s:** discard as settle-in (5s).
- **30..45s:** post-stim measurement window (15s).
- **45s:** stop. Total run length = 45s.

Total simulated time: 6 levels × 3 seeds × 45s = **810s simulated**, ~41 min
wall at 3.06× ratio. Within budget.

### 3.3 Measurements

All measurements are reported separately for the **pre-stim window (10..25s)**
and the **post-stim measurement window (30..45s)**, both 15s long. The ΔPost-Pre
column for each measurement is the load-bearing diagnostic.

| Tag | Definition |
|---|---|
| **Mod-FLP11** | Mean FLP-11 concentration over window. Source: `ModulationLayer.history_concentrations`, column 0 of `modulators`. |
| **Mod-all9** | Mean per-modulator concentration for all 9 modulators (sanity check). |
| **RIS** | Mean (RISL + RISR) / 2 firing rate. Source: `brain.spikes` filtered to RISL/RISR indices. |
| **Target-30** | Mean firing rate over the top-30 |target_weights_FLP-11| roster (locked roster in §7). |
| **Target-152** | Mean firing rate over all 152 receptor-expressing neurons. |
| **Cmd** | Mean firing rate over AVAL, AVAR, AVEL, AVER, AVBL, AVBR, PVCL, PVCR (NOT FLP-11 targets — indirect effect indicator). |
| **Global** | Mean firing rate over all 300 neurons. |

Each measurement is also reported in **absolute units** (Hz for rates,
release-units for concentration) at every drive level, per Refinement R2.

## 4 · Outcome decision criteria

### 4.1 Outcome A — natural-quiescence arm is viable. ALL THREE of:

- **A1 (FLP-11 scaling).** Spearman rank correlation between RIS firing rate
  and Mod-FLP11 post-stim concentration across the 6 drive levels (n=6) gives
  ρ with 95% bootstrap CI lower bound ≥ 0.70. Report actual ρ + CI per
  Refinement R1; do not reduce to pass/fail bit.
- **A2 (target inhibition).** Target-30 ΔPost-Pre firing-rate change at the
  RIS≈35 Hz drive level ≤ **−15%** relative to its own pre-stim baseline. At
  the RIS≈1 Hz baseline drive level, |ΔPost-Pre| ≤ 5% (no spurious effect from
  the inject_poisson mechanism alone).
- **A3 (global rate reduction).** Global ΔPost-Pre firing-rate change shows
  monotone-nonincreasing trend across drive levels (Spearman of drive vs ΔPost-Pre
  is negative with CI excluding 0). ΔPost-Pre at RIS≈35 Hz ≤ **−10%** relative
  to its own pre-stim baseline.

### 4.2 Outcome B — partial viability, needs additional work

Exactly ONE of A1 / A2 / A3 fails, OR Spearman ρ in A1 is in [0.70, 0.85]
borderline range, OR A2 reduction is in [−15%, −10%] borderline range.

**Interpretation flag (per Rohit's pre-reg refinement).** Outcome B with
A1 + A2 passing but A3 failing has two distinguishable readings; the report
must explicitly distinguish them:

- **Reading B-i — simulator calibration failure.** FLP-11 release works and
  inhibits its direct receptor-expressing targets, but downstream network
  dynamics don't propagate that inhibition to global rate reduction. Diagnostic:
  examine the FIRING-RATE-REDUCTION-LOCALIZATION metric (defined in §5) — if
  Target-30 shows clear reduction but their downstream first-order synaptic
  targets do not, B-i is supported. Implication: bridge experiment needs
  simulator network-connectivity or receptor-weight recalibration before
  running.
- **Reading B-ii — Turek 2016 framework challenge.** FLP-11 release works and
  inhibits its receptor-expressing targets per the CeNGEN expression mapping,
  but those targets aren't load-bearing for global network quiescence in this
  substrate. Diagnostic: examine WHICH neurons in Target-30 are inhibited and
  whether their inhibition logically should propagate. If the receptor-mapping
  is biologically correct (CeNGEN-faithful) but the receptor-expressing
  population is not a sufficient arousal-suppression circuit under M2-pure, B-ii
  is supported. Implication: bridge experiment can still run with revised scope —
  the natural-quiescence arm becomes "FLP-11 inhibits the CeNGEN-predicted
  receptor-expressing targets" rather than "FLP-11 produces sleep-like
  behavioral quiescence."

Other Outcome B sub-cases (A1 + A3 pass, A2 fails; or A2 + A3 pass, A1 fails)
should be reported with explicit narrative — they don't fit the B-i / B-ii
distinction.

### 4.3 Outcome C — natural-quiescence arm not viable; framework needs revision

Any of:

- A1 fails: FLP-11 doesn't scale monotonically with RIS firing rate (Spearman
  ρ with CI overlapping 0 or negative), OR
- Phase A stop condition triggered: RIS cannot be driven to ≥ 30 Hz at maximum
  input rate, OR
- A2 fails AND A3 fails: release fires but produces no measurable downstream
  effect anywhere.

Outcome C means the framework's natural-quiescence pathway is not testable in
the locked simulator without prior work on release dynamics or receptor mapping.
Report and stop for scoping conversation.

## 5 · Localization metric (for B-i vs B-ii distinction)

Computed post-hoc on Outcome-B data only:

For each neuron `n` in Target-30 that shows ΔPost-Pre ≤ −15% at RIS≈35 Hz:
- Compute mean firing-rate ΔPost-Pre on its direct postsynaptic targets
  (chemical synapses, signed, |W_chem| > threshold).
- Report: (a) ratio of mean downstream Δ to mean own Δ, (b) sign concordance
  (does upstream inhibition produce downstream disinhibition as expected for
  inhibitory→inhibitory or downstream-inhibition for inhibitory→excitatory?).

If downstream effects exist with biologically consistent signs: Reading B-i
(propagation deficit elsewhere — recalibration target identifiable).
If downstream effects are absent or noise-floor only: Reading B-ii (receptor
population isn't load-bearing in this substrate).

## 6 · Runner architecture rationale

This spike will use a **minimal direct LIFBrain + ModulationLayer runner**, not
`ClosedLoopEnv`. Rationale (Rohit's Refinement R3 check):

- ActivityFSM and BehavioralFSM are **pure readouts** — they consume external rate
  vectors and produce state transitions but do not write back to the brain.
  Confirmed by reading `activity_fsm.py` (only brain reference is `brain.names`
  at __init__).
- However, ClosedLoopEnv runs a full MuJoCo body + CPG controller + proprioceptive
  feedback loop. Driving RIS reduces global firing, which reduces motor neuron
  rates, which reduces body curvature, which reduces proprioceptive sensory
  drive — a confounding indirect path back into the brain.
- For diagnostic isolation, the minimal runner bypasses body / CPG / FSM /
  classifier entirely. Brain dynamics + modulation only. Maximum control.

Runner script: `scripts/brain/diagnostic_ris_spike.py` (to be written and
committed alongside this pre-reg's downstream work). Will instantiate LIFBrain
with M2-pure flags, attach ModulationLayer, drive RIS via inject_poisson, and
extract spikes + modulator history.

## 7 · Locked Target-30 roster (FLP-11 most-inhibitory)

Pulled from `artifacts/modulator_tables.npz` `target_weights_FLP-11` at
pre-reg lock time. Reproducibility: re-run `np.argsort(target_weights_FLP-11)[:30]`
on the same npz to recover this list.

```
PVWL, PVWR, RIGR, RIGL, M1, RIH, ASGL, ASGR, I1R, I1L, HSNL, HSNR,
RICL, RICR, URBR, URBL, AVJR, AVJL, PVT, URXL, URXR, PVNL, PVNR,
ASKR, ASKL, RID, AIZL, AIZR, AVDL, AVDR
```

Notable structure:
- Includes **RIH, RICL/R** (head interneurons; RIC implicated in arousal /
  monoaminergic state changes — biologically reasonable FLP-11 targets).
- Includes **M1** (pharyngeal motor — feeding suppression consistent with sleep).
- Includes **HSNL/R** (egg-laying motor — fits broad-quiescence framing).
- Includes **AVDL/R** (only command-neuron-adjacent entries in top-30; reversal
  pre-motor). Notably **AVA, AVE, AVB, PVC are NOT in the top-30** despite
  being canonical command neurons.
- Several arousal / sensory-gating interneurons: RIGL/R, ASGL/R, ASKL/R, URXL/R,
  URBL/R.

The interpretation flag (§4.2) explicitly addresses the concern that this
population is not the canonical sleep-circuit roster.

## 8 · Reproducibility

- **Seeds:** {0, 1, 2} for both Phase A and Phase B (six independent
  Brian2/numpy initializations).
- **Commit-locking:** this pre-reg is committed to git BEFORE the runner script
  is written. The commit message will reference this filename and date. Any
  protocol deviation requires a separate commit with explicit deviation note;
  no silent edits to this file.
- **Source-of-truth artifacts:**
  - Brain v3.5 spec: `docs/brain_v3.5_locked.md`
  - Modulator tables: `scripts/brain/artifacts/modulator_tables.npz`
  - State of claims: `docs/state_of_claims_2026-05-02.md`
- **Output artifact:** the report will be written to
  `scripts/brain/artifacts/diagnostic_ris_spike_report_2026-05-12.md` (and
  raw data to `scripts/brain/artifacts/diagnostic_ris_spike/`).

## 9 · Working discipline (confirmed)

- 3 seeds × 15s post-stim window per drive level — go/no-go scope, not paper
  statistics.
- Mixed outcomes reported as mixed ("A on FLP-11 scaling, B on target
  inhibition"), not forced into a clean verdict.
- Phase A stop-and-report conditions are hard: non-monotone calibration or
  saturation below 30 Hz means STOP, even if the report is unhelpful. Mechanism
  switch to direct I_ext fallback requires explicit user authorization.
- Sensitivity caveats reported as findings, not as failures. Specifically:
  if FLP-11 saturates at RIS rates below 50 Hz (the DEFAULT_RELEASE_GAIN /
  DEFAULT_CONCENTRATION_CAP calibration was tuned for "50 Hz saturates"), report
  the saturation point as a network-feedback finding rather than as A1 failure.

## 10 · Honest scope label

This is V8 / framework-paper diagnostic work, not V7 results. The bridge
experiment itself, if it proceeds, is V8 work. Do not co-locate these results
with V7 sub-questions.

---

**Pre-registration locked.** Refinements R1 (Spearman ρ + bootstrap CI rather
than pass/fail bit), R2 (absolute concentrations alongside ratios), R3 (FSM
feedback check confirmed pure readout; runner uses minimal architecture
bypassing ClosedLoopEnv), and Outcome-B interpretation distinction (B-i vs
B-ii) are baked in.

To be committed before runner script is written.
