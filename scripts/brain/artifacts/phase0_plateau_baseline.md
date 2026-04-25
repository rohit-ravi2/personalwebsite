# Phase 0 — W0.4a — T4-2 plateau baseline

Current-state measurement of the 15-neuron compartmental scaffold 
(`compartmental_neurons.py`) plateau response to a 50 pA / 100 ms 
somatic current injection. Compares to Gao & Hobert 2020 / Wang 2020 
targets for the plateau-expressing neurons; non-plateau neurons 
(AWC, RMG, ALA) should show no sustained depolarisation.

## Protocol

`t=0..200ms`: settle; `t=200..300ms`: inject 50 pA on soma; 
`t=300..1200ms`: record dendritic plateau dynamics.

Plateau duration = time from injection-release until 
v_d within 5.0 mV of v_rest (-65.0 mV).

## Per-neuron baseline

| neuron | v_d peak (mV) | amp (mV) | dur (ms) | target dur | target amp | gap (dur) | gap (amp) | status |
|---|---|---|---|---|---|---|---|---|
| AVAL | -60.2 | +4.5 | 0.0 | 600 | 20.0 | -600.0 | -15.5 | **FAIL** |
| AVAR | -60.2 | +4.5 | 0.0 | 600 | 20.0 | -600.0 | -15.5 | **FAIL** |
| AVEL | -61.9 | +2.9 | 0.0 | 400 | 15.0 | -400.0 | -12.1 | **FAIL** |
| AVER | -61.9 | +2.9 | 0.0 | 400 | 15.0 | -400.0 | -12.1 | **FAIL** |
| AVBL | -62.2 | +2.6 | 0.0 | 500 | 18.0 | -500.0 | -15.4 | **FAIL** |
| AVBR | -62.2 | +2.6 | 0.0 | 500 | 18.0 | -500.0 | -15.4 | **FAIL** |
| PVCL | -63.2 | +1.7 | 0.0 | 350 | 14.0 | -350.0 | -12.3 | **FAIL** |
| PVCR | -63.2 | +1.7 | 0.0 | 350 | 14.0 | -350.0 | -12.3 | **FAIL** |
| AWCL | -63.8 | +1.2 | 0.0 | 0 | 0.0 | +0.0 | +1.2 | **PASS** |
| AWCR | -63.8 | +1.2 | 0.0 | 0 | 0.0 | +0.0 | +1.2 | **PASS** |
| RMGL | -62.0 | +3.0 | 0.0 | 0 | 0.0 | +0.0 | +3.0 | **FAIL** |
| RMGR | -62.0 | +3.0 | 0.0 | 0 | 0.0 | +0.0 | +3.0 | **FAIL** |
| ALA | -60.5 | +4.5 | 0.0 | 0 | 0.0 | +0.0 | +4.5 | **FAIL** |
| RIS | -61.9 | +3.0 | 0.0 | 700 | 18.0 | -700.0 | -15.0 | **FAIL** |
| DVA | -62.0 | +2.8 | 0.0 | 400 | 16.0 | -400.0 | -13.2 | **FAIL** |

## Interpretation

- **2/15 neurons** currently pass within ±20% of target values.
- **13 fail**: AVAL, AVAR, AVEL, AVER, AVBL, AVBR, PVCL, PVCR, RMGL, RMGR, ALA, RIS, DVA. Expected — `compartmental_neurons.py` docstring marks plateau dynamics as calibration-pending. Current parameters are conservative defaults, not fits to voltage-clamp data.

## T4-2 exit threshold (ratified against this baseline)

- AVA plateau duration within 20% of 600 ms (Gao & Hobert 2020)
- AVA plateau amplitude within 10% of 20 mV (target: 18-22 mV)
- Non-plateau neurons (AWC, RMG, ALA) show no sustained depolarisation (amp < 3 mV, duration < 50 ms).
- All 15 neurons report status=PASS post-calibration.