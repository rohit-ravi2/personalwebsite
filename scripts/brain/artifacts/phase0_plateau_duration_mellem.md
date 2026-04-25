# Plateau duration at Mellem 2008 v_rest

Follow-up to `phase0_plateau_diagnostic.py` Probe 5.

**v_rest set to -25.0 mV** (Mellem 2008 AVA range: −20 to −30 mV). All other compartmental roster parameters unchanged.

## Plateau neurons

| neuron | v_d peak | amplitude | duration | target amp | target dur | @500 ms | status |
|---|---|---|---|---|---|---|---|
| AVAL | -3.7 | +21.3 | 1500 | 20.0 | 600 | -13.38 | **FAIL** |
| AVAR | -3.7 | +21.3 | 1500 | 20.0 | 600 | -13.38 | **FAIL** |
| AVEL | -11.0 | +14.0 | 1500 | 15.0 | 400 | -16.37 | **FAIL** |
| AVER | -11.0 | +14.0 | 1500 | 15.0 | 400 | -16.37 | **FAIL** |
| AVBL | -10.4 | +14.6 | 1500 | 18.0 | 500 | -16.88 | **FAIL** |
| AVBR | -10.4 | +14.6 | 1500 | 18.0 | 500 | -16.88 | **FAIL** |
| PVCL | -15.3 | +9.7 | 1500 | 14.0 | 350 | -18.43 | **FAIL** |
| PVCR | -15.3 | +9.7 | 1500 | 14.0 | 350 | -18.43 | **FAIL** |
| RIS | -9.7 | +15.3 | 1500 | 18.0 | 700 | -18.46 | **FAIL** |
| DVA | -13.7 | +11.3 | 1500 | 16.0 | 400 | -18.37 | **FAIL** |

## Non-plateau neurons (sanity check)

| neuron | v_d peak | amplitude | duration | @500 ms |
|---|---|---|---|---|
| AWCL | -23.8 | +1.2 | 0 | -25.0 |
| AWCR | -23.8 | +1.2 | 0 | -25.0 |
| RMGL | -22.0 | +3.0 | 0 | -25.0 |
| RMGR | -22.0 | +3.0 | 0 | -25.0 |
| ALA | -20.5 | +4.5 | 0 | -25.0 |

## Summary: **0/10 plateau neurons pass** at Mellem v_rest (was 2/15 at scaffold default −65 mV).

Implication: T4-2's primary calibration knob is v_rest, not g_ca / tau_h. A single-parameter change (v_rest: −65 → −25 mV) may resolve most of the plateau gap. Secondary fine-tuning on tau_h and plateau-duration targets per neuron is still needed (most won't hit exact Mellem durations at default tau_h = 350 ms).