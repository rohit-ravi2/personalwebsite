# Morning Brief — Overnight Run 2026-04-22

*Generated: 2026-04-22 19:19:24*

## Rigorous findings (Tracks A, B, C)

### Track A — Mode 1 densification

| modulator | status | max conc ctrl | max conc KO | ΔREV | ΔQUI |
|---|---|---|---|---|---|
| FLP-1 | **PASS_MODE_1** | 0.189 | 0.0 | -0.005 | +0.006 |
| NLP-12 | **PASS_MODE_1** | 10.0 | 0.0 | +0.011 | +0.013 |
| TA | **PASS_MODE_1** | 9.398 | 0.0 | +0.011 | +0.015 |
| OA | **PASS_MODE_1** | 0.412 | 0.0 | -0.016 | +0.025 |

### Track B — Readout sensitivity

# Track B — Readout architecture sensitivity

Completed: 2026-04-22 18:23:32
Wall: 0.0 min

## Status: **PARTIAL — prediction-only analysis**

**LOGISTICAL_FAILURE on full classifier retraining.** Integration 
with `neural_classifier_bank.py` training pipeline requires more 
engineering than fit in the budget (custom pooled-target 
preparation, AUC validation harness, modulation-layer-compatible 
bank format, ClosedLoopEnv bank-path override). Documented below.

This task delivers the **prediction part** of the readout-
sensitivity test: given each alternative readout set, what Mode 
does membership-based prediction assign to RIS and AVA ablation?

Full empirical confirmation (with retrained classifier + 
ablation runs) deferred to a dedicated follow-up session.

## Readout set construction

### original — 18 neurons

Neurons: AIBL, ASEL, AUAL, AVEL, AVER, CEPDL, I3, IL2DL, M3L, M3R, NSML, NSMR, OLQDL, OLQDR, OLQVL, RMER, SMDVL, URXL

### permissive — 74 neurons

Neurons: ADAL, ADEL, AIBL, AIBR, AINL, AIZL, ASEL, ASGL, ASHL, AUAL, AVAL, AVAR, AVDL, AVEL, AVER, AVJL, AVJR, AWAR, AWBL, AWCL, CEPDL, CEPDR, CEPVL, CEPVR, FLPL, I1L, I2L, I2R, I3, IL1DL, IL1DR, IL1L, IL1R, IL2DL, IL2DR, IL2L, IL2VL, IL2VR, M3L, M3R, M4, MI, NSML, NSMR, OLLL, OLLR, OLQDL, OLQDR, OLQVL, OLQVR, RIAL, RICL, RID, RIVL, RMDDL, RMDDR, RMDL, RMDR, RMDVL, RMDVR, RMED, RMEL, RMER, RMEV, SMDDL, SMDDR, SMDVL, SMDVR, URBL, URXL, URXR, URYDL, URYVL, URYVR

### command — 27 neurons

Neurons: AIBL, ASEL, AUAL, AVAL, AVAR, AVBL, AVBR, AVDL, AVDR, AVEL, AVER, CEPDL, I3, IL2DL, M3L, M3R, NSML, NSMR, OLQDL, OLQDR, OLQVL, PVCL, PVCR, RIS, RMER, SMDVL, URXL

## Prediction comparison

| readout set | n | RIS in readout? | AVA in readout? | RIS prediction | AVA prediction |
|---|---|---|---|---|---|
| **original** | 18 | ✗ | ✗ | Mode 1 or Mode 3 (releaser not in readout) | Mode 1 or Mode 3 (releaser not in readout) |
| **permissive** | 74 | ✗ | ✓ | Mode 1 or Mode 3 (releaser not in readout) | Mode 2 (readout-trivial) predicted |
| **command** | 27 | ✓ | ✓ | Mode 2 (readout-trivial) predicted | Mode 2 (readout-trivial) predicted |

## Pre-specified prediction check

**Prediction: command-enriched set → AVA shifts to Mode 2**

- Membership-level prediction: **Mode 2 (readout-trivial) predicted** — CONFIRMED at prediction level (AVA is in command-enriched readout, so membership logic predicts Mode 2).

**Empirical confirmation pending full retraining.**

## What was attempted for empirical confirmation

1. Readout-set construction for permissive (≥7/10) and 
   command-enriched sets (implemented).
2. Classifier retraining with custom neuron_order via 
   pooled Atanas worms 1-8 train, 9-10 test (attempted 
   but API mismatch with existing `neural_classifier_bank.py` 
   — multiple non-exported symbols needed).
3. Bank swap-in at `artifacts/classifier_bank.npz` path 
   (implemented but not reached).
4. ClosedLoopEnv ablation runs under alternative bank 
   (deferred).

Next session: add `--readout-set` arg to 
`neural_classifier_bank.py:train_bank()`, saving output as 
`classifier_bank_permissive.npz` / `classifier_bank_command.npz`; 
add `classifier_bank_path` parameter to `ClosedLoopEnv`.


### Track C — Parallel analysis

- **C1 receptor pharmacology:** annotated 37 peptide-receptor pairs; 4 flagged UNVERIFIED pending manual check.
- **C2 molecular baseline:** Operating: 7/9; Inert: 2/9
- **C3 FLP-11 scenario stability:** **Mode stable across scenarios: Mode 1**
- **C4 citation audit:** 7/7 verified on retry.

## Exploratory findings (Tracks E, F) — speculative

**Explicit reminder: Track E and F outputs are exploratory. Interpretation is not yet rigorous. Any follow-up requires dedicated investigation.**

### Track E (GNCA cell fate) — LOGISTICAL_FAILURE

Reason: Sulston lineage data not accessible via WebFetch (Git LFS + paywall blocks). See `speculative/track_e/LOGISTICAL_FAILURE.md` for attempted sources and unblock conditions.

### Track F (HH AVA calibration) — FAIL

| metric | target | tolerance | best result | err | pass |
|---|---|---|---|---|---|
| amplitude (mV) | 20 | ±10% | 10.55 | 0.47 | ✗ |
| duration (ms) | 600 | ±20% | 1.4 | 1.00 | ✗ |
| return (ms) | 1500 | ±30% | 5.5 | 1.00 | ✗ |

Best params g_ca=3 nS, g_k=25 nS, g_leak=0.5 nS, tau_h=400 ms. amp=10.55 mV (target 20, err 0.47); dur=1.4 ms (target 600, err 1.00); ret=5.5 ms (target 1500, err 1.00).

## Failed or ambiguous tasks

- Track E: LOGISTICAL_FAILURE (Sulston lineage data inaccessible)
- Track F: FAIL (HH minimal model cannot produce plateau; duration off by >100× tolerance)

## Recommended morning actions

1. Review Track A Mode 1 densification results — any modulator flagged FAIL_MODE_1 requires re-classification
2. Review Track B prediction check — AVA Mode prediction confirmed or violated?
3. Track F FAIL implies HH minimal model (Ca + K + leak) insufficient for AVA plateau. Do NOT integrate. Follow-up: digitize Mellem Fig 1d trace; add additional K channels (slo-1, shl-1) to the model.
4. Track E unblock: download Packer 2019 supplementary tables locally, then re-run with real Sulston lineage + fates.

## Open questions

- Does Track A's FLP-1/OA AMBIGUOUS status (mechanism inert) change the Mode 1 count from D1? (Task D1 said 5 Mode 1, but Track C2 separates operating vs inert.)
- If Track B confirms AVA → Mode 2 under command-enriched readout, does that open the door to using ActivityFSM on this enlarged readout as a proper behavioral test?
- Track F's failure: is this a minimal-model limitation or an indication that AVA plateau requires channel combinations beyond egl-19+K+leak?
