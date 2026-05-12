# Phase 2 pre-flight — FSM/classifier recalibration scope

**Date drafted:** 2026-05-03 (Phase 2 pre-flight, post brain-v3.5-lock)
**Predecessor:** Phase 1 closed at M2-pure (`docs/brain_v3.5_locked.md`)
**Status:** scoping draft. No code changes / training runs yet. Awaits Rohit
sign-off on architectural option + readout decision before timeline commitment.

---

## 1 · Data inventory (comprehensive)

### 1.1 Raw Atanas data — present locally

`/home/rohit/Desktop/website/personalwebsite/data/external/atanas2023/`
**(535 GB total)**

| File | Size | Subject |
|---|---:|---|
| atanas_worm_01.nwb | 26.9 GB | (no SWF tag in name) |
| atanas_worm_02.nwb | 27.0 GB | (no SWF tag) |
| atanas_worm_03_sub-2022-06-28-07-SWF702.nwb | 26.8 GB | SWF702 |
| atanas_worm_04..10_sub-…-SWF702.nwb | 26.5–27.0 GB | SWF702 strain |

These are the original DANDI 000776 NWB files: paired GCaMP7f whole-brain
imaging (~1.4-1.7 Hz volumes) + NIR behavior video + NeuroPAL identification.
Per worm: ~134 ROIs identified (varies 109-152 per worm).

### 1.2 Parsed per-worm artifacts — production-grade, ready-to-use

`/home/rohit/Desktop/website/personalwebsite/scripts/brain/artifacts/atanas_worm_*.npz`
(10 files, ~1.5 MB each, **~15 MB total**)

Schema per worm:

| array | shape | dtype | content |
|---|---|---|---|
| `neural` | (1600, N_neurons) | float32 | normalized ΔF/F per identified neuron |
| `neural_raw` | (1600, N) | float32 | raw fluorescence |
| `neuron_ids` | (N,) | object | NeuroPAL labels |
| `t` | (1600,) | float32 | timestamps in seconds |
| `velocity, head_curv, body_curv, ang_vel` | (1600,) each | float32 | kinematics |
| `reversal, pumping` | (1600,) each | float32 | behavioral channels |

**Sampling: dt = 0.5996s (~1.667 Hz)**, total duration ≈ 962s (~16 min) per
worm. Same T=1600 across all 10 worms.

**All 10 worms have AVAL, AVAR, AVDL, AVDR identified.** Confirmed today by
direct measurement — validates catalog claim C-48 (the 18-readout was a
methodology choice, not a data limitation).

Per-worm N_neurons: 109, 128, 130, 132, 133, 134, 142, 142, 144, 152 — wide
spread; the strict cross-worm intersection of 18 dropped many commands.

### 1.3 Atanas published analyses — supplementary

`/home/rohit/Desktop/C-Elegans/data/atanas2023/` **(194 MB)**

`.h5` files: `encoding_changes_corrected`, `fit_ranges`, `neuron_categorization`,
`relative_encoding_strength_median`, `tuning_strength`. Plus
`neuropal_label.json`. These are **derived analysis products** from the
Atanas paper, not raw recordings. Useful as ground-truth labels /
sanity-check targets but not training input.

### 1.4 Other relevant data

- Connectome data: `/home/rohit/Desktop/C-Elegans/data/connectome/` (26 MB)
  — already integrated into `connectome.npz`
- CeNGEN expression: `/home/rohit/Desktop/C-Elegans/data/expression/` (6.5 GB)
  — already integrated; per-edge sign mode uses this
- WormBase WS297: 43 GB; lineage: 5.5 MB; randi2023: 1.3 MB; misc others.
  None directly relevant to Phase 2.

### 1.5 Existing trained artifacts (legacy, default-mode-trained)

`scripts/brain/artifacts/`:
- `classifier_bank.npz` (11 KB) — 8 events × 18 neurons × LogReg weights
- `classifier_bank.json` — metadata (per-event horizon, features, C, AUCs)
- `calibration.npz` (1.7 KB) — Brian2 → Atanas-ΔF/F per-neuron calibration

**These are trained against default-mode dynamics + 18-neuron readout.**
Building fresh for M2-pure does NOT amend these in place — produces new
artifacts (e.g., `classifier_bank_v2_m2pure.npz`) so legacy versions
remain for comparison.

---

## 2 · Code inventory

### 2.1 Data prep pipeline (reusable as-is)

| File | Lines | Role |
|---|---:|---|
| `parse_atanas.py` | 159 | NWB → per-worm npz (single worm) |
| `parse_atanas_all.py` | 123 | NWB → per-worm npz (all 10) |
| `event_extraction.py` | 325 | per-worm npz → event labels (8 events; literature-grounded thresholds) |
| `calibrate_distribution.py` | 188 | Brian2 sim → Atanas-distributed ΔF/F (per-neuron affine) |

**Status: production-grade.** No reason to rewrite. The atanas_worm_*.npz
artifacts are already produced; data prep doesn't need re-running unless we
want to inspect / re-derive.

### 2.2 Existing classifier code (template / reference, NOT the production target)

| File | Lines | Role |
|---|---:|---|
| `neural_classifier_bank.py` | 357 | trains 8-event LogReg bank on 18-neuron readout |

Contains:
- `EVENTS_FOR_BANK` (8 events)
- `EVENT_CONFIGS` (per-event horizon + features + C from harness sweep)
- `calcium_kernel`, `spikes_to_calcium` (synthetic calcium model — GCaMP7f
  τ_rise=0.1s, τ_decay=0.5s)
- `_intersection_all_10` (the strict cross-worm intersection)
- `build_features` (lags / derivs / smoothed feature engineering)

**Reusable components**: synthetic calcium model, feature engineering,
data loading. **What needs redesign**: the pooled-Atanas-data training
loop (since we may want different readout, possibly different model class
or regularization, possibly multi-class instead of per-event binary, and
proper held-out worm CV instead of single training-AUC).

### 2.3 FSM code (downstream consumers)

| File | Lines | Role |
|---|---:|---|
| `behavioral_fsm.py` | 211 | original Phase 3 — classifier-bank-driven FSM |
| `activity_fsm.py` | 363 | P1 #4 upgrade — direct command-neuron firing-rate readout |

Both consume from the brain. Either becomes the validation target for the
new classifier; **decision deferred** (per Phase 2 plan: FSM threshold
recalibration is sub-task 3).

### 2.4 Failed / partial training infrastructure

`overnight_v2_track_b.py` — flagged "LOGISTICAL_FAILURE on full classifier
retraining" per session notes. Cause: "Integration ablation runs deferred
to a dedicated follow-up session" / "API engineering" blocks. Not a data
problem; a code-pathway issue. The fresh-from-scratch plan avoids
inheriting that path.

---

## 3 · Existing classifier — what's there + why fresh

### 3.1 Existing `classifier_bank.npz` content

- **Readout (18 neurons)**: AIBL, ASEL, AUAL, AVEL, AVER, CEPDL, I3, IL2DL,
  M3L, M3R, NSML, NSMR, OLQDL, OLQDR, OLQVL, RMER, SMDVL, URXL
  (notably: AVAL/AVAR/AVDL/AVDR/AVBL/AVBR/PVCL/PVCR all absent)
- **Events (8)**: reversal_onset/offset, forward_run_onset/offset,
  omega_onset, pirouette_entry, quiescence_onset, speed_burst_onset
- **Per-event train AUC** (from `classifier_bank.json`):
  reversal_onset 0.76, reversal_offset 0.85, forward_run_onset 0.83,
  forward_run_offset 0.79, omega_onset 0.90, pirouette_entry 0.75,
  quiescence_onset 0.84, speed_burst_onset 0.80 — range 0.75–0.90
- **Feature engineering**: per-event "horizon" + "features" (derivs / lags),
  C regularization tuned per event

**Important caveat:** the `train_auc` numbers are the *training* fits, not
held-out cross-worm AUC. The Phase 3b harness reportedly validated cross-worm
generalization, but the numbers in classifier_bank.json are training-set fits.

### 3.2 Why "fresh from scratch" makes sense

Per Rohit's instruction:
1. **The existing classifier was trained against Atanas REAL data**, not
   simulator data. So in principle the *classifier* doesn't need retraining
   under M2-pure — just the *Brian2-→-Atanas-ΔF/F calibration* would change.
2. **However** — under M2-pure, AVA dynamic range tripled, RIS silenced,
   per-edge cascade fires. The 18-neuron readout *doesn't include AVA/AVD/PVC*,
   so the classifier may be missing the most informative channels under M2-pure.
3. The existing pipeline has known limitations not fixed at the code level:
   training-AUC reporting (vs held-out-worm CV), per-event binary instead
   of multi-class, no Brier-score / calibration measurement, no clear
   readout-set ablation.
4. **Fresh build** = new training script with clean assumptions + per-readout-set
   training + held-out-worm CV + multi-class option + diagnostics. Existing
   code becomes reference; new artifacts produced under separate filenames.

---

## 4 · Architectural options for the fresh classifier

Four orthogonal axes:

### 4.1 Axis A — Readout-set choice

| Option | Cells | Rationale |
|---|---|---|
| A1 (legacy 18) | 18 strict-intersection (no commands) | reproducible from existing classifier |
| A2 (legacy 18 + commands) | 18 + AVA/AVD/AVE/AVB/PVC = ~28 | minimal extension; preserves legacy + adds the cells M2-pure makes maximally informative |
| A3 (relaxed-intersection ~30+) | cells in ≥7 of 10 worms (per overnight_v2_track_b) | broader coverage; some missing-cell handling needed |
| A4 (per-worm full readout) | each worm's full identified set | maximum information; can't pool naively across worms |

Decision deferred per Rohit's instruction to "after data inventory."

### 4.2 Axis B — Model class

| Option | Notes |
|---|---|
| B1 LogReg per-event (legacy) | fast, interpretable; reproducible |
| B2 LogReg multi-class (single softmax over events + null) | catches mutual exclusion; one model |
| B3 small NN (1-2 hidden layers) | more expressive; risks overfitting on n=10 worms |
| B4 GLM with elastic-net | between B1 and B3; better feature selection |

Default proposal: **B1 + per-event diagnostics first, B2 as comparison**. B3
risks being non-reproducible at n=10; defer.

### 4.3 Axis C — Training data source

| Option | Description |
|---|---|
| C1 Real Atanas only | classifier learns "what Atanas neural activity predicts behavior"; simulator-→-Atanas calibration is the bridge |
| C2 Simulated only | classifier learns "what M2-pure simulator activity predicts behavior"; needs labeled simulator data which doesn't exist |
| C3 Hybrid (real Atanas + augmented from sim) | simulator can generate synthetic data with known event timing; combine with real |

C2 is non-viable (no labels). **C1 is the natural choice.** C3 is interesting
but adds complexity.

### 4.4 Axis D — Cross-validation

| Option | Description |
|---|---|
| D1 Train on all, report training AUC | what classifier_bank.npz currently does — overfits |
| D2 Leave-one-worm-out CV | 10-fold; reports realistic generalization |
| D3 Stratified k-fold within each worm | within-worm CV; doesn't test cross-worm generalization |

**D2 is the right answer** for any decision-grade comparison. D1 is what
classifier_bank.json uses today; D3 is too local.

### 4.5 Default proposal (for Rohit to accept / amend)

**A2 + B1 + C1 + D2** = expanded 28-neuron readout (legacy 18 + 10 commands),
per-event LogReg, trained on real Atanas data, with leave-one-worm-out CV.

This:
- Adds the AVA/AVD/PVC commands that M2-pure makes maximally informative
- Preserves comparability with the legacy 18-readout classifier (subset)
- Doesn't bet on architectural novelty (reproducible)
- Reports real generalization metric (not training AUC)
- Doesn't gate on hybrid-data engineering

---

## 5 · Phase 2 sub-task decomposition (4-5 sub-tasks)

Per the rebase plan, Phase 2 is "FSM / classifier recalibration under chosen
mode." Decomposing:

### Sub-task 2.1 — Readout-set decision (Axis A)

Once data inventory is reviewed, decide A1/A2/A3/A4. Pre-flight gate before
sub-task 2.2.

**Effort:** ½ day (decision-only — based on inventory + brief readout-coverage
overlap analysis).

### Sub-task 2.2 — Build fresh training script + train classifier

New script `scripts/brain/phase2_train_classifier.py` (or similar). Imports
data prep helpers from existing code; clean training loop with leave-one-worm-out
CV; per-event metrics + Brier + calibration plots; output:
`classifier_bank_v2_<readout-tag>.npz` + JSON + CV results.

**Effort:** 2-3 days for code + 1-2 hr training (sklearn LogReg on ~16 min
× 10 worms × 28 features is fast).

### Sub-task 2.3 — Brian2-→-Atanas calibration under M2-pure

The existing `calibrate_distribution.py` produces a per-neuron affine
mapping calibrated under default-mode dynamics. Under M2-pure, AVA dynamic
range tripled, RIS silenced — per-neuron affines need recomputing. Run
M2-pure simulator under matching scenarios, compute per-neuron percentile
matching to Atanas distribution, write `calibration_m2pure.npz`.

**Effort:** 1 day (script + ~3-6 hr Brian2 runs to generate calibration
data).

### Sub-task 2.4 — FSM threshold recalibration

Two sub-options:
- **2.4a** behavioral_fsm — classifier-bank-driven FSM. Once new classifier
  ready, retune state-transition thresholds against Atanas behavioral truth.
- **2.4b** activity_fsm — direct firing-rate readout FSM (P1 #4). Re-tune
  z-score thresholds + EMA baseline τ under M2-pure.

Could do both (independent; useful comparison). Per the Phase 1 lock,
M2-pure has `RIS at 1.08 Hz` — the activity_fsm RIS-quiescence threshold
may need substantial revision.

**Effort:** 1-2 days each FSM (could run in parallel).

### Sub-task 2.5 — Phase 2 gauntlet — validation under recalibrated FSM

Re-run the Phase 1 gauntlet (M2-pure mode only) under the new classifier +
new FSM thresholds. Document which catalog phenotype claims (C-22, C-25,
C-27, etc.) are reproduced under the recalibrated stack.

**Effort:** 1 day setup + 6-12 hr compute (n=10 × 60s × 6 ablations × 1 mode
already at default tier; we know the timing from Phase 1A).

---

## 6 · Decision points before Phase 2 execution

| # | Decision | When |
|---|---|---|
| 1 | Readout-set (A1/A2/A3/A4) | Sub-task 2.1 (pre-flight gate) |
| 2 | Model class default (B1) — accept or amend | Sub-task 2.2 design |
| 3 | Whether to retrain calibration first OR train classifier first | Sub-task 2.2 vs 2.3 ordering |
| 4 | Whether to do both FSM variants (2.4a + 2.4b) or pick one | Sub-task 2.4 |
| 5 | Whether to keep legacy classifier_bank.npz in tree (probably yes, rename to legacy_*) | Sub-task 2.2 output |

---

## 7 · Time estimate (rough; conservative)

If A2 + B1 + C1 + D2 default proposal accepted as-is:

| Sub-task | Code | Compute | Total |
|---|---|---|---|
| 2.1 readout decision | ½ day | 0 | ½ day |
| 2.2 classifier retrain | 2-3 days | 1-2 hr | 2-3 days |
| 2.3 calibration retune | 1 day | 3-6 hr | 1.5 days |
| 2.4a behavioral_fsm | 1 day | minimal | 1 day |
| 2.4b activity_fsm | 1 day | minimal | 1 day |
| 2.5 gauntlet validation | 1 day | 6-12 hr | 1.5 days |

**Total realistic: 7-9 days of focused work + 12-22 hr of compute.**

Phase 1 wall-time overran my estimate by 5x; building in ~2x conservatism
on Phase 2 estimates: **realistically 2-3 weeks of calendar time** with
overnight compute fitting in normally.

If we drop sub-task 2.4a or 2.4b (pick one FSM): -1 day. If we drop
sub-task 2.3 (use legacy calibration as starting point): -1.5 days.
**Minimum viable: ~5 days focused work.**

---

## 8 · What I want from Rohit before timeline commitment

1. **Sign-off on default architectural option (A2 + B1 + C1 + D2)** — or argue
   for a different mix on any axis. Especially: Axis A readout-set is the
   load-bearing decision.
2. **Timeline pressure / context** — if there's a deadline (e.g., NYU
   academic schedule, OMSCS application timing, grant cycle), Phase 2 scope
   should compress. If no pressure, the fuller scope (both FSM variants +
   careful CV + new calibration) is the better path.
3. **Whether to do Phase 2.5 in M2-pure only or also re-test M1 / M2-current
   under new classifier** — comparison runs add ~6-12 hr compute but may
   strengthen the brain-lock decision retroactively.
4. **Wave2HybridBrain investigation thread (C-37) timing** — runs in parallel
   with Phase 2 or after? Per rebase plan §7.2 it's not gating, but adds
   compute load if parallel.

---

## 9 · Standing by

**No code changes / training runs initiated yet.** Pre-flight scope ready.
Tasks #6 (this pre-flight) ready to mark complete; #7 + (Phase 2 sub-tasks)
will be created after Rohit picks the architectural option + timing.
