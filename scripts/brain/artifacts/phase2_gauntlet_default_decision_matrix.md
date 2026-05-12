# Phase 1 gauntlet — decision matrix

Generated 2026-05-02. Tier: see metadata in JSON sidecar.

## Test 1 — Touch cascade firing (C-13 broadening)

Per-cell Δ peri-touch (Hz, mean across seeds). 'AVA / touch' control runs only.

| cell | M1 | M2-pure | M2-current | M3a |
|---|---|---|---|---|
| ALML | - | +86.50 | - | - |
| AVM | - | +87.97 | - | - |
| PVCL | - | +60.42 | - | - |
| AVDL | - | +60.42 | - | - |
| AVAL | - | +60.27 | - | - |
| AVAR | - | +60.55 | - | - |
| AVBL | - | +50.60 | - | - |
| AIBL | - | +2.10 | - | - |
| RIML | - | +10.18 | - | - |

## Test 2 — AVA→dREV (C-22 default-mode reproduction)

Mean ± SEM, neg-seed count for dREV.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M2-pure | +0.2285 | 0.1373 | 2/10 |

## Test 3 — AVA→dPIR (C-21 per-edge channel shift)

Mean ± SEM, neg-seed count for dPIR.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M2-pure | -0.0050 | 0.0050 | 1/10 |

## Test 5 — RIS→dQUI (C-25 Turek)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M2-pure | -0.0070 | 0.0264 | 4/10 |

## Test 9 — NSM→dQUI counter-finding (C-27)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M2-pure | +0.2002 | 0.1302 | 0/10 |

## Test 4 + 6 — RIS baseline + non-touch network stability

RIS baseline rate (Hz, spontaneous) and stability metrics across non-touch scenarios.

### spontaneous

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M2-pure | 10.33 | 1.12 | 0.0 | 0 |

### osmotic_shock

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M2-pure | 27.58 | 0.63 | 32.0 | 0 |

### food

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M2-pure | 14.60 | 0.53 | 0.0 | 0 |

### chemotaxis

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M2-pure | 9.32 | 0.89 | 0.0 | 0 |
