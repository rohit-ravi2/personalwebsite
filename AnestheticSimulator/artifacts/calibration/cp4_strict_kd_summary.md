# CP4 — Strict-Kd ground-truth subset construction

## Directness tier framework

The current ground-truth table (`ground_truth_Kd_table.csv`) labeled itself as a 'Kd table' but contains **zero strict-Kd entries**. All 30 anchor values are FUNCTIONAL readouts (electrophysiology EC50/IC50, mitochondrial O2 consumption IC50). True strict Kd would require radioligand displacement or photoaffinity binding (e.g., Hall 1994 propofol-azi-octanol, Husain 2003 etomidate-TDBzl photoaffinity, Eckenhoff 1996 halothane photoaffinity).

**Why this matters:** for positive allosteric modulators (PAMs) of GABA-A/GlyR, functional potentiation EC50 is typically **3-10× larger** than direct-binding Kd because the allosteric coupling efficiency is < 1 (Forman & Miller 2016 review *Anesth Analg* 123:1297; Husain 2003 PMID 12707441). For ion-channel blockers (open-channel block), functional IC50 tracks Kd more closely. For mitochondrial O2 assays, the Kd-readout chain is even less direct — multiple coupling steps, possible non-binding effects.

**Directness tier assignments:**

- **T1** — recombinant single-target electrophysiology (HEK/oocyte). Cleanest functional readout; PAM allosteric bias ~3-10×; channel-block bias ~1-3×.
- **T2** — native-tissue electrophysiology. Endogenous modulation noise; additional ~3-10× bias possible.
- **T3** — isolated mitochondrial O2 consumption assay. Multiple intervening steps; ~10-100× bias possible; not a binding measurement.
- **STRICT_KD** — radioligand/photoaffinity displacement Kd. None present.

## Tier distribution

| Tier | Count | Description |
|---|---|---|
| T1 | 18 | recombinant single-target electrophys |
| T2 | 8 | native-tissue / mixed |
| T3 | 4 | mitochondrial O2 consumption |

## Subset statistics (entries with numeric value AND comparison row)

### Strict subset (T1 + STRICT_KD)

- n = 17
- mean |log_err| = 0.629
- median |log_err| = 0.437
- within 10× (|log_err| ≤ 1.0): 13/17 (76%)
- within 3× (|log_err| ≤ 0.477): 9/17 (53%)
- signed mean log_err: +0.527 (pipeline systematically overestimates Kd (weaker than measured))

### T3 (mitochondrial) subset

- n = 3
- mean |log_err| = 0.114
- within 10×: 3/3 (100%)

## Per-entry tiers and errors

| Target | Anesthetic | Tier | EC50/IC50 (µM) | Predicted Kd (µM) | log_err |
|---|---|---|---|---|---|
| GABA-A_α1β2γ2 | halothane | T1 | 250 | 2711.4 | +1.04 |
| GABA-A_α1β2γ2 | isoflurane | T1 | 260 | 593.1 | +0.36 |
| GABA-A_α1β2γ2 | sevoflurane | T1 | 210 | 423.1 | +0.30 |
| GABA-A_α1β2γ2 | propofol | T1 | 1.5 | 66.0 | +1.64 |
| GABA-A_α1β2γ2 | etomidate | T1 | 2 | 47.1 | +1.37 |
| GABA-A_α1β2γ2 | ketamine | T2 | 1000 | 47.1 | -1.33 |
| GlyR_α1 | halothane | T1 | 250 | 1934.3 | +0.89 |
| GlyR_α1 | isoflurane | T1 | 250 | 831.4 | +0.52 |
| GlyR_α1 | sevoflurane | T1 | 200 | 501.0 | +0.40 |
| GlyR_α1 | propofol | T1 | 30 | 55.8 | +0.27 |
| GlyR_α1 | etomidate | T1 | 5 | 66.0 | +1.12 |
| GlyR_α1 | ketamine | T2 | 1500 | 92.6 | -1.21 |
| nAChR_α4β2 | halothane | T1 | 130 | 501.0 | +0.59 |
| nAChR_α4β2 | isoflurane | T2 | 170 | 215.3 | +0.10 |
| nAChR_α4β2 | sevoflurane | T1 | 80 | 181.9 | +0.36 |
| nAChR_α4β2 | propofol | T1 | 90 | 55.8 | -0.21 |
| nAChR_α4β2 | etomidate | T1 | 300 | 109.6 | -0.44 |
| nAChR_α4β2 | ketamine | T1 | 5 | 47.1 | +0.97 |
| TREK-1_KCNK2 | halothane | T1 | 700 | 702.2 | +0.00 |
| TREK-1_KCNK2 | isoflurane | T1 | 500 | 301.9 | -0.22 |
| TREK-1_KCNK2 | sevoflurane | T2 | 400 | 129.8 | -0.49 |
| TREK-1_KCNK2 | propofol | T1 | — | — | — |
| TREK-1_KCNK2 | etomidate | T2 | — | — | — |
| TREK-1_KCNK2 | ketamine | T2 | — | — | — |
| NDUFS2_ComplexI | halothane | T3 | 500 | 831.4 | +0.22 |
| NDUFS2_ComplexI | isoflurane | T3 | 400 | 357.4 | -0.05 |
| NDUFS2_ComplexI | sevoflurane | T3 | 500 | 423.1 | -0.07 |
| NDUFS2_ComplexI | propofol | T3 | — | — | — |
| NDUFS2_ComplexI | etomidate | T2 | — | — | — |
| NDUFS2_ComplexI | ketamine | T2 | — | — | — |

## CP5 recalibration plan

CP5 will recalibrate the pipeline using only the **T1 strict subset** as ground truth. The headline metric `% within 10×` and `% within 3×` will be recomputed; the systematic log_err signed mean will reveal whether the pipeline has a predictable allosteric bias that can be corrected by a multiplicative factor.
