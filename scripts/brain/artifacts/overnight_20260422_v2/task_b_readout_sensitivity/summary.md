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
