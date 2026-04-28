# Phase E — Markov synaptic transmission summary

## Method

Single C. elegans NMJ synapse simulated as Gillespie-like SSA. 5 release sites; cooperative Ca-SNARE binding with Hill exponent n; fusion + recycling. Anesthetic perturbation: shift n by `n_Ca_delta` from Phase D `wave2_overlay.json` (SNARE-class kinetic shift).

## Baseline calibration

| n_cooperativity | spont rate Hz | evoked p |
|---|---|---|
| 2.0 | 0.100 | 0.000 |
| 2.5 | 0.200 | 0.000 |
| 3.0 | 0.200 | 0.020 |
| 3.5 | 0.200 | 0.090 |
| 4.0 | 0.200 | 0.230 |
| 5.0 | 0.600 | 0.910 |

WT default: n = 3.5; spont = 0.200 Hz, evoked p = 0.090.

## Anesthetic perturbation

| anesthetic | raw n_Ca_delta | effective n_Ca_delta | n_perturbed | spont Hz | evoked p | fold change |
|---|---|---|---|---|---|---|
| etomidate | -0.209 | -0.063 | 3.44 | 0.200 | 0.090 | 1.000 |
| halothane | -1.454 | -0.436 | 3.06 | 0.200 | 0.030 | 0.333 |
| isoflurane | -1.467 | -0.440 | 3.06 | 0.200 | 0.020 | 0.222 |
| ketamine | -1.498 | -0.450 | 3.05 | 0.200 | 0.020 | 0.222 |
| propofol | -1.316 | -0.395 | 3.11 | 0.200 | 0.040 | 0.444 |
| sevoflurane | -1.448 | -0.434 | 3.07 | 0.200 | 0.030 | 0.333 |

## Validation (halothane)

Predicted release-p fold-change: 0.333

Target band (Stewart 2000 PMID 11095753 / van Swinderen 1999 PMID 10051668): 0.3-0.7

**Verdict: PASS**
