# Task 5 — 18-neuron readout peptidergic coverage

Generated: 2026-04-21 09:32:40

For each modulator, counts how many of its receptor-expressing 
target neurons fall inside the 18-neuron classifier readout set. 
This predicts which Mode the modulator's ablation would exhibit 
behaviorally.

## Readout-18 composition

Neurons: AIBL, ASEL, AUAL, AVEL, AVER, CEPDL, I3, IL2DL, M3L, M3R, NSML, NSMR, OLQDL, OLQDR, OLQVL, RMER, SMDVL, URXL

## Peptidergic broadcaster overlap

Ripoll-Sánchez 2023 peptidergic broadcasters: I1, I2, I3, I4, I5, M5, NSM

Broadcasters IN readout:
- **I3** → I3
- **NSM** → NSML
- **NSM** → NSMR

Paper implication: if most peptidergic broadcasters are OUTSIDE the readout, this explains why peptidergic ablations routinely produce behavioral nulls — the simulator's readout architecture systematically excludes the cells that carry the peptidergic signal.

## Per-modulator Mode prediction

| modulator | # conn targets | # in readout | frac | readout hits | predicted Mode |
|---|---|---|---|---|---|
| **FLP-11** | 7 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **FLP-1** | 2 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **FLP-2** | 0 | 0 | 0.00 | - | N/A (no targets detected) |
| **NLP-12** | 0 | 0 | 0.00 | - | N/A (no targets detected) |
| **PDF-1** | 4 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **5HT** | 19 | 3 | 0.16 | I3;M3L;M3R | Mode 3 possible (partial readout overlap) |
| **DA** | 9 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **TA** | 12 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **OA** | 4 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **FLP-13** | 9 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **FLP-18** | 4 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **FLP-21** | 2 | 0 | 0.00 | - | Mode 1 (readout-blind) |
| **NLP-40** | 0 | 0 | 0.00 | - | N/A (no targets detected) |
| **DAF-28** | 0 | 0 | 0.00 | - | N/A (no targets detected) |

## Prediction summary

- **Mode 1 (readout-blind) predicted:** 9/14
- **Mode 3 (readout-cascade) predicted:** 1/14
- **No targets detected:** 4/14

If D1's empirical Mode classification (Task 1) matches 
this prediction table, the B4 readout-overlap predictor 
can be used prospectively for any new modulator.
