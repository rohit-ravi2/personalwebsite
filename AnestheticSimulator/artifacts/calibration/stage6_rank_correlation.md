# Stage 6 — Rank correlation analysis

## Method

For each target, compute Spearman ρ between predicted affinity (-ΔG) and clinical-potency proxy (-log10 of clinical aqueous EC50 µM). Strong positive ρ supports the multi-target framing; scrambled / negative ρ suggests pipeline doesn't track clinical potency at the per-target level.

Reference clinical EC50 (µM): etomidate=0.3, halothane=340.0, isoflurane=290.0, ketamine=5000.0, propofol=1.0, sevoflurane=230.0

Implied clinical potency rank (highest→lowest, by aqueous EC50): etomidate, propofol, sevoflurane, isoflurane, halothane, ketamine

## Per-target Spearman ρ

| gene | class | ρ | predicted order | clinical potency order |
|---|---|---|---|---|
| RIC-4 | snare_cooperativity | +0.657 | etomidate,propofol,ketamine,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NUO-3 | complex_i_block | +0.600 | propofol,etomidate,ketamine,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-38 | nachr_antagonism | +0.429 | etomidate,ketamine,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-49 | gaba_potentiation | +0.429 | etomidate,ketamine,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| AVR-14 | glucl_potentiation | +0.314 | propofol,ketamine,etomidate,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| AVR-15 | glucl_potentiation | +0.314 | propofol,ketamine,etomidate,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| SNT-1 | snare_cooperativity | +0.290 | etomidate,ketamine,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| ACR-2 | nachr_antagonism | +0.265 | etomidate,ketamine,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| ACR-16 | nachr_antagonism | +0.174 | propofol,sevoflurane,ketamine,isoflurane,etomidate,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NCA-2 | nca_block | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NLF-1 | nca_block | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NUO-1 | complex_i_block | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NUO-4 | complex_i_block | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-13 | snare_cooperativity | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-64 | snare_cooperativity | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-79 | nca_block | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| GLC-1 | glucl_potentiation | +0.116 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| GLC-2 | glucl_potentiation | +0.116 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| MEV-1 | complex_ii_block | +0.116 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| SNB-1 | snare_cooperativity | +0.116 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| TWK-18 | k2p_potentiation | +0.116 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| TWK-29 | k2p_potentiation | +0.116 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| EXP-1 | gaba_potentiation | +0.086 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| GAS-1 | complex_i_block | +0.086 | ketamine,propofol,etomidate,sevoflurane,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| NUO-2 | complex_i_block | +0.086 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-18 | snare_cooperativity | +0.086 | ketamine,etomidate,sevoflurane,propofol,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| TWK-7 | k2p_potentiation | +0.029 | propofol,ketamine,sevoflurane,isoflurane,etomidate,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-29 | nachr_antagonism | +0.029 | ketamine,propofol,etomidate,isoflurane,sevoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| UNC-63 | nachr_antagonism | -0.029 | ketamine,etomidate,sevoflurane,isoflurane,propofol,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |
| LEV-1 | nachr_antagonism | -0.147 | ketamine,propofol,sevoflurane,etomidate,isoflurane,halothane | etomidate,propofol,sevoflurane,isoflurane,halothane,ketamine |

## Per mechanism-class average

| class | n_targets | n_anesthetics | ρ (avg potency) | predicted avg-potency order |
|---|---|---|---|---|
| complex_i_block | 5 | 6 | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane |
| complex_ii_block | 1 | 6 | +0.116 | ketamine,etomidate,propofol,isoflurane,sevoflurane,halothane |
| gaba_potentiation | 2 | 6 | +0.429 | etomidate,ketamine,propofol,sevoflurane,isoflurane,halothane |
| glucl_potentiation | 4 | 6 | +0.086 | ketamine,propofol,etomidate,sevoflurane,isoflurane,halothane |
| k2p_potentiation | 3 | 6 | +0.314 | propofol,ketamine,etomidate,sevoflurane,isoflurane,halothane |
| nachr_antagonism | 6 | 6 | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane |
| nca_block | 3 | 6 | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane |
| snare_cooperativity | 6 | 6 | +0.143 | ketamine,etomidate,propofol,sevoflurane,isoflurane,halothane |

## Headline

- Per-target ρ > 0: 28/30
- Per-target ρ > 0.5: 2/30
- Median ρ: +0.143

## Caveat

Clinical aqueous EC50 reflects whole-animal behavioral effect — the INTEGRAL of multi-target perturbation. A single target may not order anesthetics the same way as whole-animal potency. Strong rank correlation supports the framing. Weak correlation is interpretable (this target is not a primary potency driver) rather than disqualifying.
