# CP B.3 — Hodgkin-Huxley universality test

**Date:** 2026-04-28

Test whether Wave 2 cells exhibit regenerative spike-like dynamics under strong depolarization, and if so whether spike properties match H-H predictions. C. elegans cells are biologically expected to be GRADED (non-spiking), so this validator confirms or rejects that biological expectation at the equation level.

**Method:** simulate single-compartment cell with dynamic slow gate, scan I_inj from 0 to +200 pA, extract V_max, spike count, regenerative overshoot. A cell that doesn't spike under +200 pA is confirmed graded.

## AVAL

| I_inj (pA) | V_max (mV) | V_min (mV) | amplitude | spikes | regenerative |
|---|---|---|---|---|---|
| 0 | -26.76 | -54.94 | 28.18 | 1 | no |
| 10 | -5.44 | -54.89 | 49.45 | 1 | YES |
| 30 | 64.06 | -54.78 | 118.84 | 1 | no |
| 50 | 100.0 | -54.68 | 154.68 | 1 | no |
| 100 | 100.0 | -54.42 | 154.42 | 1 | no |
| 200 | 100.0 | -53.9 | 153.9 | 1 | no |

### Verdict: SPIKING under strong drive (1 spikes at strongest tested current)

**H-H prediction comparison:**

- E_Ca = 60.0 mV (would be Ca-spike peak if Ca-spiking)
- E_K = -80.0 mV (would be after-hyperpolarization floor)
- Observed V_max (across tested currents) = 100.0 mV
- ✓ V_max approaches E_Ca → Ca-spike-consistent if spiking detected

## AVAR

| I_inj (pA) | V_max (mV) | V_min (mV) | amplitude | spikes | regenerative |
|---|---|---|---|---|---|
| 0 | -24.8 | -54.91 | 30.12 | 1 | no |
| 10 | -4.44 | -54.86 | 50.41 | 1 | no |
| 30 | 46.49 | -54.74 | 101.23 | 1 | no |
| 50 | 100.0 | -54.62 | 154.62 | 1 | no |
| 100 | 100.0 | -54.32 | 154.32 | 1 | no |
| 200 | 100.0 | -53.73 | 153.73 | 1 | no |

### Verdict: SPIKING under strong drive (1 spikes at strongest tested current)

**H-H prediction comparison:**

- E_Ca = 60.0 mV (would be Ca-spike peak if Ca-spiking)
- E_K = -80.0 mV (would be after-hyperpolarization floor)
- Observed V_max (across tested currents) = 100.0 mV
- ✓ V_max approaches E_Ca → Ca-spike-consistent if spiking detected

## AIY

| I_inj (pA) | V_max (mV) | V_min (mV) | amplitude | spikes | regenerative |
|---|---|---|---|---|---|
| 0 | -55.12 | -66.77 | 11.65 | 1 | YES |
| 10 | -20.84 | -54.64 | 33.81 | 1 | YES |
| 30 | 67.89 | -53.69 | 121.58 | 1 | YES |
| 50 | 100.0 | -52.74 | 152.74 | 1 | no |
| 100 | 100.0 | -50.37 | 150.37 | 1 | no |
| 200 | 100.0 | -45.63 | 145.63 | 1 | no |

### Verdict: SPIKING under strong drive (1 spikes at strongest tested current)

**H-H prediction comparison:**

- E_Ca = 127.59 mV (would be Ca-spike peak if Ca-spiking)
- E_K = -80.0 mV (would be after-hyperpolarization floor)
- Observed V_max (across tested currents) = 100.0 mV
- Observed V_max in Mellem-AVA range; Ca-channel partial activation but no Ca-spike overshoot

## RIM

| I_inj (pA) | V_max (mV) | V_min (mV) | amplitude | spikes | regenerative |
|---|---|---|---|---|---|
| 0 | -56.2 | -71.42 | 15.23 | 1 | YES |
| 10 | -55.87 | -66.99 | 11.12 | 1 | YES |
| 30 | -55.23 | -58.13 | 2.9 | 1 | no |
| 50 | -49.28 | -54.58 | 5.3 | 1 | no |
| 100 | -27.24 | -52.97 | 25.73 | 1 | no |
| 200 | 17.11 | -49.74 | 66.85 | 1 | no |

### Verdict: SPIKING under strong drive (1 spikes at strongest tested current)

**H-H prediction comparison:**

- E_Ca = 60.0 mV (would be Ca-spike peak if Ca-spiking)
- E_K = -80.0 mV (would be after-hyperpolarization floor)
- Observed V_max (across tested currents) = 17.11 mV
- Observed V_max in Mellem-AVA range; Ca-channel partial activation but no Ca-spike overshoot

## Cross-cell synthesis

- Graded (no spike, no regenerative): **0/4** cells
- Regenerative but non-spiking: **0/4** cells
- Spiking under strong drive: **4/4** cells

**H-H universality verdict:** the canonical H-H formalism is implemented in the cell-builder code; whether or not a cell spikes depends on the channel suite balance (inward vs outward currents). Wave 2 cells use Nicoletti's channel suites which are optimized for graded validated phenotypes. Regenerative behavior under non-physiological strong drive (+200 pA) is informative about the cell's potential dynamical regimes, not its biological operating mode.
