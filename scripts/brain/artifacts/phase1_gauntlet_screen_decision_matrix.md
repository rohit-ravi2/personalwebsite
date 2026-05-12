# Phase 1 gauntlet — decision matrix

Generated 2026-05-02. Tier: see metadata in JSON sidecar.

## Test 1 — Touch cascade firing (C-13 broadening)

Per-cell Δ peri-touch (Hz, mean across seeds). 'AVA / touch' control runs only.

| cell | M1 | M2-pure | M2-current | M3a |
|---|---|---|---|---|
| ALML | +84.40 | +85.35 | +84.85 | - |
| AVM | +88.20 | +88.00 | +88.10 | - |
| PVCL | -3.55 | +60.35 | -2.50 | - |
| AVDL | -3.95 | +60.35 | -2.50 | - |
| AVAL | -1.85 | +60.20 | +0.20 | - |
| AVAR | -0.60 | +60.10 | -2.30 | - |
| AVBL | -6.00 | +51.65 | -4.15 | - |
| AIBL | +2.20 | +3.70 | +1.50 | - |
| RIML | +3.75 | +11.50 | +0.90 | - |

## Test 2 — AVA→dREV (C-22 default-mode reproduction)

Mean ± SEM, neg-seed count for dREV.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | -0.2137 | 0.1303 | 3/5 |
| M2-pure | +0.0423 | 0.0330 | 2/5 |
| M2-current | +0.0070 | 0.0971 | 3/5 |

## Test 3 — AVA→dPIR (C-21 per-edge channel shift)

Mean ± SEM, neg-seed count for dPIR.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | -0.0563 | 0.0368 | 2/5 |
| M2-pure | -0.0400 | 0.0245 | 2/5 |
| M2-current | -0.0763 | 0.0345 | 3/5 |

## Test 5 — RIS→dQUI (C-25 Turek)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | +0.0043 | 0.0217 | 1/5 |
| M2-pure | +0.0313 | 0.0221 | 1/5 |
| M2-current | +0.0127 | 0.0054 | 0/5 |

## Test 9 — NSM→dQUI counter-finding (C-27)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | +0.3897 | 0.0639 | 0/5 |
| M2-pure | +0.3930 | 0.0527 | 0/5 |
| M2-current | +0.3930 | 0.0558 | 0/5 |

## Test 4 + 6 — RIS baseline + non-touch network stability

RIS baseline rate (Hz, spontaneous) and stability metrics across non-touch scenarios.

### spontaneous

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 18.50 | 30.66 | 0.0 | 0 |
| M2-pure | 10.22 | 1.08 | 0.0 | 0 |

### osmotic_shock

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 52.76 | 72.85 | 69.0 | 0 |
| M2-pure | 27.59 | 0.59 | 32.0 | 0 |

### food

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 25.81 | 42.53 | 0.0 | 0 |
| M2-pure | 14.43 | 0.46 | 0.0 | 0 |

### chemotaxis

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 27.85 | 46.44 | 0.0 | 0 |
| M2-pure | 9.99 | 0.98 | 0.0 | 0 |
