# Stage 4 — dual K_p calibration tables

## Method

Compare Vina-predicted Kd (from `Kd = exp(ΔG/RT)` at 298 K) for mammalian-homolog targets against published experimental EC50/IC50 values.

Frame A — raw: predicted_Kd vs experimental_value, no K_p amplification.
Frame B — with K_p: predicted occupancy at K_p × experimental_concentration.

**Caveat (load-bearing):** All ground-truth values are EC50/IC50 from patch-clamp dose-response, NOT classical equilibrium Kd from radioligand binding. Direct fold-error against EC50/IC50 conflates Vina ΔG bias with Kd-vs-EC50 quantity mismatch. Spearman rank correlation is the more interpretable metric.

## Frame A — raw (predicted Kd vs experimental value)

- N pairs: 24
- log_err mean: +0.26
- log_err median: +0.30
- |log_err| ≤ 0.3 (within 2×): 8/24
- |log_err| ≤ 0.5 (within ~3×): 14/24
- |log_err| ≤ 1.0 (within 10×): 18/24
- Spearman ρ (log_pred_Kd vs log_exp_value): +0.368

## Per-pair table

| target | anesthetic | pred Kd µM | exp µM | type | log err |
|---|---|---|---|---|---|
| KCNK2 | halothane | 702.25 | 700.0 | EC50_activation | +0.00 |
| NDUFS2 | isoflurane | 357.39 | 400.0 | IC50_inhibition | -0.05 |
| NDUFS2 | sevoflurane | 423.14 | 500.0 | IC50_inhibition | -0.07 |
| CHRNA4 | isoflurane | 215.34 | 170.0 | IC50_block | +0.10 |
| CHRNA4 | propofol | 55.77 | 90.0 | IC50_block | -0.21 |
| KCNK2 | isoflurane | 301.86 | 500.0 | EC50_activation | -0.22 |
| NDUFS2 | halothane | 831.44 | 500.0 | IC50_inhibition | +0.22 |
| GLRA1 | propofol | 55.77 | 30.0 | EC50_potentiation | +0.27 |
| GABRA1 | sevoflurane | 423.14 | 210.0 | EC50_potentiation | +0.30 |
| CHRNA4 | sevoflurane | 181.88 | 80.0 | IC50_block | +0.36 |
| GABRA1 | isoflurane | 593.14 | 260.0 | EC50_potentiation | +0.36 |
| GLRA1 | sevoflurane | 500.98 | 200.0 | EC50_potentiation | +0.40 |
| CHRNA4 | etomidate | 109.59 | 300.0 | IC50_block | -0.44 |
| KCNK2 | sevoflurane | 129.75 | 400.0 | EC50_activation | -0.49 |
| GLRA1 | isoflurane | 831.44 | 250.0 | EC50_potentiation | +0.52 |
| CHRNA4 | halothane | 500.98 | 130.0 | IC50_block | +0.59 |
| GLRA1 | halothane | 1934.26 | 250.0 | EC50_potentiation | +0.89 |
| CHRNA4 | ketamine | 47.11 | 5.0 | IC50_block | +0.97 |
| GABRA1 | halothane | 2711.38 | 250.0 | EC50_potentiation | +1.04 |
| GLRA1 | etomidate | 66.03 | 5.0 | EC50_potentiation | +1.12 |
| GLRA1 | ketamine | 92.56 | 1500.0 | EC50_potentiation | -1.21 |
| GABRA1 | ketamine | 47.11 | 1000.0 | EC50_potentiation | -1.33 |
| GABRA1 | etomidate | 47.11 | 2.0 | EC50_potentiation | +1.37 |
| GABRA1 | propofol | 66.03 | 1.5 | EC50_potentiation | +1.64 |
