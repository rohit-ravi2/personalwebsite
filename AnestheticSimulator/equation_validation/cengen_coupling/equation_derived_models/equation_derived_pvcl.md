# Equation-derived prediction: PVCL

**Biological role:** PVC additional command interneuron — touch reversal pathway integrator, downstream of ALM/AVM.

**STATUS: EQUATION-DERIVED PREDICTION, AWAITING EMPIRICAL VALIDATION**

This model is generated from CeNGEN gene expression (Taylor et al. 2021) + canonical Hodgkin-Huxley formalism + Wave 2 cell-builder calibration (α-scaling per channel, calibrated on AVAL/AVAR/AIY/RIM). It is NOT validated against published electrophysiology — no Nicoletti recordings exist for PVCL. The model produces falsifiable predictions for wet-lab follow-up, not a model that should be deployed in production simulation.

## Predicted channel densities

| channel | gene | TPM | α used (nS/TPM) | g_predicted (nS) | confidence |
|---|---|---|---|---|---|
| egl19 | egl-19 | 0.1 | 1.072 | 0.1072 | CALIBRATED_HIGH_SPREAD |
| unc2 | unc-2 | 0.27 | 1.429 | 0.3857 | CALIBRATED_SINGLE_CELL |
| shl1 | shl-1 | 0.3 | 12.067 | 3.6201 | CALIBRATED_SINGLE_CELL |
| kvs1 | kvs-1 | 0.07 | 1.250 | 0.0875 | FALLBACK_MEDIAN_ALPHA |
| shk1 | shk-1 | 0.19 | 1.250 | 0.2376 | FALLBACK_MEDIAN_ALPHA |
| unc103 | unc-103 | 0.18 | 1.250 | 0.2251 | FALLBACK_MEDIAN_ALPHA |
| slo1iso | slo-1 | 0.23 | 1.250 | 0.2876 | FALLBACK_MEDIAN_ALPHA |
| slo2 | slo-2 | 0.13 | 1.250 | 0.1625 | FALLBACK_MEDIAN_ALPHA |
| nca_aux | unc-80 | 0.34 | 1.250 | 0.4251 | FALLBACK_MEDIAN_ALPHA |

**Leak conductance:** 0.05 nS (default; CeNGEN doesn't capture leak channels — passive membrane parameter).

## Predicted V_rest

GHK parallel-conductance prediction: **-65.95 mV** (at full channel activation; non-dynamic gate prediction).

## Indirect validation

- **Wicks 1996:** PVC participates in touch reversal cascade; functional ablation studies establish role.
- **Atanas 2023:** PVC shows transient activity during touch-induced reversal sequences.
- **Connectome:** PVC connects to AVA/AVD command interneurons; sign-exception entries documented in Phase 3a (PVC-Glu-iGluR).

## Falsifiability

This prediction is falsifiable by:
1. Whole-cell electrophysiology of PVCL (current-voltage curve, input resistance, V_rest).
2. Channel-specific pharmacology + isolated current measurements.
3. Calcium imaging during sensory/motor protocols.

Prediction failure modes:
- Channel densities off by > 10× → linear scaling insufficient for this cell
- V_rest off by > 15 mV → either channel set wrong or leak parameter wrong
- Cell biophysical phenotype qualitatively different (e.g., spiking when prediction says graded) → CeNGEN-equation-coupling has a fundamental gap for this cell type

## Methodology disclaimer

This is exploratory work testing whether CeNGEN-equation-coupling is a viable path past the C. elegans biophysical literature cap (~20-30 cells with full primary-source validation). The calibration LOO mean |log10_err| is approximately 0.56 — predictions are within ~3.6× on average, with individual channel errors up to 10× possible. **Treat all numbers as order-of-magnitude estimates, not point predictions.**
