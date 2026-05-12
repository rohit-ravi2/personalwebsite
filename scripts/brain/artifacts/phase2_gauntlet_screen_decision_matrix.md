# Phase 1 gauntlet — decision matrix

Generated 2026-05-02. Tier: see metadata in JSON sidecar.

## Test 1 — Touch cascade firing (C-13 broadening)

Per-cell Δ peri-touch (Hz, mean across seeds). 'AVA / touch' control runs only.

| cell | M1 | M2-pure | M2-current | M3a |
|---|---|---|---|---|
| ALML | +85.05 | +85.55 | +84.95 | - |
| AVM | +88.30 | +88.25 | +88.35 | - |
| PVCL | -3.45 | +60.35 | -1.80 | - |
| AVDL | -3.60 | +60.35 | -1.80 | - |
| AVAL | -1.70 | +60.20 | +1.05 | - |
| AVAR | -0.75 | +59.90 | -1.45 | - |
| AVBL | -6.75 | +49.70 | -3.85 | - |
| AIBL | +2.70 | +3.70 | +1.45 | - |
| RIML | +4.50 | +11.35 | +0.45 | - |

## Test 2 — AVA→dREV (C-22 default-mode reproduction)

Mean ± SEM, neg-seed count for dREV.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | +0.0000 | 0.0000 | 0/5 |
| M2-pure | +0.1650 | 0.1071 | 0/5 |
| M2-current | -0.2710 | 0.1530 | 3/5 |

## Test 3 — AVA→dPIR (C-21 per-edge channel shift)

Mean ± SEM, neg-seed count for dPIR.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | +0.1200 | 0.0204 | 0/5 |
| M2-pure | -0.0200 | 0.0200 | 1/5 |
| M2-current | +0.0200 | 0.0374 | 1/5 |

## Test 5 — RIS→dQUI (C-25 Turek)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | +0.0187 | 0.0128 | 0/5 |
| M2-pure | +0.0357 | 0.0412 | 2/5 |
| M2-current | +0.0587 | 0.0397 | 1/5 |

## Test 9 — NSM→dQUI counter-finding (C-27)

Mean ± SEM, neg-seed count for dQUI.

| mode | mean | SEM | neg/N |
|---|---|---|---|
| M1 | -0.1337 | 0.0257 | 5/5 |
| M2-pure | +0.2083 | 0.1890 | 0/5 |
| M2-current | +0.1923 | 0.1923 | 0/5 |

## Test 4 + 6 — RIS baseline + non-touch network stability

RIS baseline rate (Hz, spontaneous) and stability metrics across non-touch scenarios.

### spontaneous

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 19.08 | 31.42 | 0.0 | 0 |
| M2-pure | 10.41 | 1.11 | 0.0 | 0 |
| M2-current | 10.17 | 1.13 | 0.0 | 0 |

### osmotic_shock

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 52.72 | 72.87 | 69.4 | 0 |
| M2-pure | 27.59 | 0.57 | 32.0 | 0 |
| M2-current | 27.53 | 0.59 | 32.0 | 0 |

### food

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 26.12 | 42.72 | 0.0 | 0 |
| M2-pure | 14.63 | 0.47 | 0.0 | 0 |
| M2-current | 14.46 | 0.47 | 0.0 | 0 |

### chemotaxis

| mode | mean rate (Hz) | RIS rate (Hz) | n above 100Hz | errors |
|---|---|---|---|---|
| M1 | 27.54 | 45.73 | 0.0 | 0 |
| M2-pure | 9.42 | 0.93 | 0.0 | 0 |
| M2-current | 9.18 | 0.92 | 0.0 | 0 |
