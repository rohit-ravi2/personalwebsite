# Phase F — metabolic ATP layer + gas-1 hypersensitivity prediction

## Model

Analytic steady-state ATP balance + K-ATP channel coupling. Anesthetic effect on Complex I scaled linearly with dose using `rate_factor` from `artifacts/kinetics/wave2_overlay.json` (Phase D output).

WT Complex I rate constant = 1.0; gas-1 mutant Complex I rate = 0.4 (Kayser 2001 PMID 11278828, mid of 30-50% reduction range).

Behavioral immobilization threshold: 5 mV hyperpolarization from K-ATP opening.

## Predicted dose-to-immobilization

| anesthetic | block_factor@1×EC50 | WT dose | gas-1 dose | ratio (WT/gas-1) |
|---|---|---|---|---|
| etomidate | 0.977 | inf | inf | nan |
| halothane | 0.706 | 2.43 | 0.98 | 2.48 |
| isoflurane | 0.707 | 2.44 | 0.98 | 2.49 |
| ketamine | 0.700 | 2.39 | 0.96 | 2.49 |
| propofol | 0.723 | 2.59 | 1.04 | 2.49 |
| sevoflurane | 0.711 | 2.47 | 1.00 | 2.47 |

## Validation against Morgan & Sedensky 1995 (PMID 7943840)

Target: gas-1 hypersensitivity ratio ~2-3× for volatile anesthetics.

Predicted median (volatiles only): 2.48

**Verdict: PASS** — within Morgan target band
