# CP B.2 — Bifurcation analysis under varied input current

**Date:** 2026-04-28

Sweep I_inj from -50 to +50 pA in 2 pA steps for each Wave 2 cell. Compute steady-state V at each I via iterative fixed-point search. Forward + backward sweeps detect hysteresis (signature of bistability). Slope of V-vs-I curve indicates bifurcation type.

## AVAL

### Bifurcation classification: **monotone_smooth (no bifurcation in tested range)**

### Hysteresis detection

- Forward vs backward sweep max difference: **1.28 mV** at I_inj = 20.0 pA
- Hysteresis verdict: **ABSENT**

### V-I curve key points

| I_inj (pA) | V_forward (mV) | V_backward (mV) | diff |
|---|---|---|---|
| -50 | -100.0 | -100.0 | 0.0 |
| -30 | -100.0 | -100.0 | 0.0 |
| -10 | -49.97 | -49.74 | -0.23 |
| 0 | -29.27 | -29.26 | -0.01 |
| 10 | -13.0 | -12.76 | -0.24 |
| 30 | 63.9 | 64.94 | -1.04 |
| 50 | 70.0 | 70.0 | 0.0 |

**Wicks 1996 bistability check:** ⚠ no hysteresis detected at this resolution. AVA-class plateau may be monostable in this single-slow-variable approximation; full bistability could require multiple coupled slow variables (e.g., EGL-19 inactivation + NCA/UNC-103 slow currents) or specific stimulus protocols.

## AVAR

### Bifurcation classification: **monotone_smooth (no bifurcation in tested range)**

### Hysteresis detection

- Forward vs backward sweep max difference: **0.14 mV** at I_inj = 16.0 pA
- Hysteresis verdict: **ABSENT**

### V-I curve key points

| I_inj (pA) | V_forward (mV) | V_backward (mV) | diff |
|---|---|---|---|
| -50 | -100.0 | -100.0 | 0.0 |
| -30 | -91.43 | -91.38 | -0.05 |
| -10 | -45.1 | -45.06 | -0.04 |
| 0 | -26.83 | -26.83 | 0.0 |
| 10 | -8.11 | -8.02 | -0.09 |
| 30 | 46.29 | 46.43 | -0.14 |
| 50 | 70.0 | 70.0 | 0.0 |

**Wicks 1996 bistability check:** ⚠ no hysteresis detected at this resolution. AVA-class plateau may be monostable in this single-slow-variable approximation; full bistability could require multiple coupled slow variables (e.g., EGL-19 inactivation + NCA/UNC-103 slow currents) or specific stimulus protocols.

## AIY

### Bifurcation classification: **monotone_smooth (no bifurcation in tested range)**

### Hysteresis detection

- Forward vs backward sweep max difference: **0.0 mV** at I_inj = None pA
- Hysteresis verdict: **ABSENT**

### V-I curve key points

| I_inj (pA) | V_forward (mV) | V_backward (mV) | diff |
|---|---|---|---|
| -50 | -100.0 | -100.0 | 0.0 |
| -30 | -100.0 | -100.0 | 0.0 |
| -10 | -100.0 | -100.0 | 0.0 |
| 0 | -66.55 | -66.55 | 0.0 |
| 10 | -28.84 | -28.84 | 0.0 |
| 30 | 32.13 | 32.13 | 0.0 |
| 50 | 70.0 | 70.0 | 0.0 |
## RIM

### Bifurcation classification: **monotone_smooth (no bifurcation in tested range)**

### Hysteresis detection

- Forward vs backward sweep max difference: **0.0 mV** at I_inj = None pA
- Hysteresis verdict: **ABSENT**

### V-I curve key points

| I_inj (pA) | V_forward (mV) | V_backward (mV) | diff |
|---|---|---|---|
| -50 | -93.37 | -93.37 | 0.0 |
| -30 | -84.52 | -84.52 | 0.0 |
| -10 | -75.67 | -75.67 | 0.0 |
| 0 | -71.25 | -71.25 | 0.0 |
| 10 | -66.84 | -66.84 | 0.0 |
| 30 | -58.07 | -58.07 | 0.0 |
| 50 | -49.5 | -49.5 | 0.0 |
## Cross-cell synthesis

Bifurcation classifications + hysteresis verdicts per cell summarize the cell's I-V dynamical structure. Cells classified as `monotone_smooth` are monostable in this approximation; cells with `discontinuous` or `rapid_transition` show bistable switching dynamics.

**Caveat:** single-slow-variable approximation is a phase-plane simplification. Full multi-gate Brian2 simulation (separate validator) would expose additional dynamical structure if present. The current sweep is a useful first-pass characterization.
