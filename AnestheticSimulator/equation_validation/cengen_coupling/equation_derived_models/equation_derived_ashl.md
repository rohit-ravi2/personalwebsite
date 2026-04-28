# Equation-derived prediction: ASHL

**Biological role:** ASH polymodal sensory neuron (avoidance) — substitute for ASE which is absent from CeNGEN panel. Detects nociceptive osmotic / chemical / mechanical stimuli; drives avoidance reversal.

**STATUS: EQUATION-DERIVED PREDICTION, AWAITING EMPIRICAL VALIDATION**

This model is generated from CeNGEN gene expression (Taylor et al. 2021) + canonical Hodgkin-Huxley formalism + Wave 2 cell-builder calibration (α-scaling per channel, calibrated on AVAL/AVAR/AIY/RIM). It is NOT validated against published electrophysiology — no Nicoletti recordings exist for ASHL. The model produces falsifiable predictions for wet-lab follow-up, not a model that should be deployed in production simulation.

## Predicted channel densities

| channel | gene | TPM | α used (nS/TPM) | g_predicted (nS) | confidence |
|---|---|---|---|---|---|
| unc2 | unc-2 | 0.1 | 1.429 | 0.1429 | CALIBRATED_SINGLE_CELL |
| shk1 | shk-1 | 0.06 | 1.250 | 0.075 | FALLBACK_MEDIAN_ALPHA |
| exp2 | exp-2 | 0.14 | 1.250 | 0.175 | FALLBACK_MEDIAN_ALPHA |
| slo2 | slo-2 | 0.12 | 1.250 | 0.15 | FALLBACK_MEDIAN_ALPHA |

**Leak conductance:** 0.05 nS (default; CeNGEN doesn't capture leak channels — passive membrane parameter).

## Predicted V_rest

GHK parallel-conductance prediction: **-44.57 mV** (at full channel activation; non-dynamic gate prediction).

## Indirect validation

- **Hart 1995, Hilliard 2002:** ASH responds to high-osmolarity, repellent chemicals, harsh touch; depolarization and Ca2+ rise documented.
- **Atanas 2023:** ASH shows prominent transient calcium response during osmotic shock or mechanical stimulation.

## Falsifiability

This prediction is falsifiable by:
1. Whole-cell electrophysiology of ASHL (current-voltage curve, input resistance, V_rest).
2. Channel-specific pharmacology + isolated current measurements.
3. Calcium imaging during sensory/motor protocols.

Prediction failure modes:
- Channel densities off by > 10× → linear scaling insufficient for this cell
- V_rest off by > 15 mV → either channel set wrong or leak parameter wrong
- Cell biophysical phenotype qualitatively different (e.g., spiking when prediction says graded) → CeNGEN-equation-coupling has a fundamental gap for this cell type

## Methodology disclaimer

This is exploratory work testing whether CeNGEN-equation-coupling is a viable path past the C. elegans biophysical literature cap (~20-30 cells with full primary-source validation). The calibration LOO mean |log10_err| is approximately 0.56 — predictions are within ~3.6× on average, with individual channel errors up to 10× possible. **Treat all numbers as order-of-magnitude estimates, not point predictions.**
