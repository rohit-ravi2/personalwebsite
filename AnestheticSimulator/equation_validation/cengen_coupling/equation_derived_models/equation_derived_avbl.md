# Equation-derived prediction: AVBL

**Biological role:** AVB forward-locomotion command interneuron — paired antagonist with AVA, drives forward crawling.

**STATUS: EQUATION-DERIVED PREDICTION, AWAITING EMPIRICAL VALIDATION**

This model is generated from CeNGEN gene expression (Taylor et al. 2021) + canonical Hodgkin-Huxley formalism + Wave 2 cell-builder calibration (α-scaling per channel, calibrated on AVAL/AVAR/AIY/RIM). It is NOT validated against published electrophysiology — no Nicoletti recordings exist for AVBL. The model produces falsifiable predictions for wet-lab follow-up, not a model that should be deployed in production simulation.

## Predicted channel densities

| channel | gene | TPM | α used (nS/TPM) | g_predicted (nS) | confidence |
|---|---|---|---|---|---|
| egl19 | egl-19 | 0.09 | 1.072 | 0.0965 | CALIBRATED_HIGH_SPREAD |
| unc2 | unc-2 | 0.08 | 1.429 | 0.1143 | CALIBRATED_SINGLE_CELL |
| exp2 | exp-2 | 0.17 | 1.250 | 0.2126 | FALLBACK_MEDIAN_ALPHA |
| unc103 | unc-103 | 0.09 | 1.250 | 0.1125 | FALLBACK_MEDIAN_ALPHA |
| slo2 | slo-2 | 0.31 | 1.250 | 0.3876 | FALLBACK_MEDIAN_ALPHA |
| nca | nca-2 | 0.24 | 0.329 | 0.0789 | CALIBRATED_LOW_SPREAD |
| nca_aux | unc-80 | 0.38 | 1.250 | 0.4751 | FALLBACK_MEDIAN_ALPHA |

**Leak conductance:** 0.05 nS (default; CeNGEN doesn't capture leak channels — passive membrane parameter).

## Predicted V_rest

GHK parallel-conductance prediction: **-47.09 mV** (at full channel activation; non-dynamic gate prediction).

## Indirect validation

- **Atanas 2023 calcium imaging:** AVB shows tonic activity correlated with forward locomotion bouts; rapid suppression at reversal onset.
- **Behavioral genetics:** AVB ablation impairs forward locomotion; coupling to muscle motor neurons via gap junctions + chemical synapses.
- **Connectome (Cook 2019):** AVB is a major hub neuron with extensive forward-circuit connectivity.

## Falsifiability

This prediction is falsifiable by:
1. Whole-cell electrophysiology of AVBL (current-voltage curve, input resistance, V_rest).
2. Channel-specific pharmacology + isolated current measurements.
3. Calcium imaging during sensory/motor protocols.

Prediction failure modes:
- Channel densities off by > 10× → linear scaling insufficient for this cell
- V_rest off by > 15 mV → either channel set wrong or leak parameter wrong
- Cell biophysical phenotype qualitatively different (e.g., spiking when prediction says graded) → CeNGEN-equation-coupling has a fundamental gap for this cell type

## Methodology disclaimer

This is exploratory work testing whether CeNGEN-equation-coupling is a viable path past the C. elegans biophysical literature cap (~20-30 cells with full primary-source validation). The calibration LOO mean |log10_err| is approximately 0.56 — predictions are within ~3.6× on average, with individual channel errors up to 10× possible. **Treat all numbers as order-of-magnitude estimates, not point predictions.**
