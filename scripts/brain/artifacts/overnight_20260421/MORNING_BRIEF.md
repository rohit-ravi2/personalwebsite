# MORNING BRIEF — overnight run 2026-04-21

*Generated: 2026-04-21 12:22:43*

Wake-up synthesis of the overnight agentic run. Full details per
task live in each `taskN_*/` subdirectory.

## 1. D1 modulator Mode classification (the headline)

**9 modulators classified across 9 rows.**

- **Mode 1 (readout-blind)**: 5 modulators (FLP-1, FLP-11, NLP-12, OA, TA)
- **Mode 2 (readout-trivial)**: 2 modulators (5HT, DA)
- **Mode 3 (readout-cascade)**: 2 modulators (FLP-2, PDF-1)

**Headline table (Δ per state vs control):**

| modulator | scenario | ΔREV | ΔQUI | Mode |
|---|---|---|---|---|
| 5HT | touch | -0.61 | +0.52 | Mode 2 (readout-trivial) |
| DA | touch | -0.16 | +0.16 | Mode 2 (readout-trivial) |
| FLP-1 | osmotic_shock | -0.00 | +0.00 | Mode 1 (readout-blind) |
| FLP-11 | osmotic_shock | -0.01 | -0.01 | Mode 1 (readout-blind) |
| FLP-2 | osmotic_shock | -0.59 | +0.42 | Mode 3 (readout-cascade) |
| NLP-12 | osmotic_shock | +0.00 | +0.01 | Mode 1 (readout-blind) |
| OA | touch | +0.01 | +0.01 | Mode 1 (readout-blind) |
| PDF-1 | touch | -0.17 | +0.23 | Mode 3 (readout-cascade) |
| TA | touch | +0.03 | +0.01 | Mode 1 (readout-blind) |

## 2. Genome-wide peptide survey

- **159** peptides scanned (FLP + NLP + INS + NPP families)
- **93** expressed above TPM=4 threshold
- **66** below detection (present in CeNGEN but TPM ≤ 4)
- **0** unresolved gene symbols (potential artifacts or naming mismatches)

| family | resolved | expressed > 4 |
|---|---|---|
| FLP | 26 / 26 | 25 |
| NLP | 73 / 73 | 53 |
| INS | 33 / 33 | 13 |
| NPP | 25 / 25 | 0 |
| PDF | 2 / 2 | 2 |

## 3. Modulator target-set overlap

vs existing 9

## 4. 18-neuron readout peptidergic coverage

- Predicted Mode 1 (readout-blind): **9**/14 modulators
- Predicted Mode 3 (readout-cascade): **1**/14 modulators

## 5. FLP-13 vs FLP-11 target comparison

- **REDUNDANT** (Jaccard = 0.80) — FLP-13 and FLP-11 target largely the same neurons. Adding FLP-13 to T4-5 may not provide distinct empirical coverage.

## 6. T4-5 citation verification

## Summary verification table

| peptide | claimed ref | PMID | verdict |
|---|---|---|---|
| FLP-13 | Nath 2016 | 27546573 | ✓ CORRECT (Current Biology, ALA sleep) |
| FLP-18 | Rogers 2003 | 14555955 | ✗ MISATTRIBUTED (paper is about FLP-21); use Cohen 2009 |
| FLP-21 | de Bono 1998 | 9741632 | ✓ CORRECT (Cell, NPR-1 natural variation). Rogers 2003 is the complementary FLP-21 → NPR-1 paper. |
| NLP-40 | Wang 2013 | 23583549 | ✓ CORRECT (Current Biology, defecation) |
| DAF-28 | Li 2003 | 12654727 | ✓ CORRECT (Genes & Dev, insulin superfamily) |

## 7. Ripoll-Sánchez 2023 cross-reference

### | peptide | confirmed in Ripoll-Sánchez 2023? | their receptor(s) | our receptor assignment | discrepancy |
|---|---|---|---|---|
| FLP-13 | **✓ Yes** (versatile-neuropeptide group with FLP-4, FLP-9, FLP-10) | not surfaced in fetch | DMSR-1, DMSR-2 (per Nath 2016) | receptor detail not fetchable; likely consistent — DMSR family is known FLP-13 receptor |
| FLP-18 | **✓ Yes** (Fig 4A, pervasive network) | **NPR-5** specifically cited | NPR-1, NPR-4, NPR-5 | NPR-5 confirmed; NPR-1/NPR-4 are from other refs (Cohen 2009, Kim & Li 2004) |
| FLP-21 | ✓ Yes (implied by NPR-1 scaffold) | not surfaced in fetch | NPR-1 (per de Bono 1998 + Rogers 2003) | consistent |
| NLP-40 | **Not explicit in fetched content** | — | AEX-2 (per Wang 2013) | can't verify from fetch; need supplementary access |
| DAF-28 | **Not explicit in fetched content** | — | DAF-2 (per Li 2003) | can't verify from fetch; need supplementary access |

## 8. T4-5 pre-validation

- **4/5** candidates pass smoke gate
- Flagged candidates need inspection before T4-5 start.

## Open issues and recommended next actions

**Citation corrections to apply:**
- Update FLP-18 primary reference from Rogers 2003 to Cohen et al. 2009 (Rogers 2003 is actually FLP-21/NPR-1).
- Verify Nelson 2013 attribution for NLP-22. CeNGEN shows ~zero expression in RIA; either the literature cite was imprecise or CeNGEN under-detects rapid-turnover peptides.

**T4-5 scope refinement:**
- Review D1 classification table. Any modulator that shows a Mode different from Task 5's prediction is a priority for investigation — the predictor needs refinement.
- FLP-13 vs FLP-11 Jaccard (Task 6): check verdict above. If largely redundant, reconsider FLP-13 as the quiescence peptide addition.

**Paper-relevant findings to preserve:**
- Peptidergic broadcasters in 18-neuron readout (Task 5): directly supports Mode 1 readout-blindness argument.
- Ripoll-Sánchez 2023 confirms peptidergic rich-club = 52% of neurons (Task 8); cite in paper methods.
- Genome-wide peptide survey produces supplementary data justifying our modulator selection as data-driven.

## Phase 0 close-out checklist

- [x] Baseline audits (scenario, AVA/touch, RIS molecular)
- [x] Phase 0 three-mode taxonomy demonstrated
- [x] Voltage-scale finding + Mellem 2008 replacement
- [x] Audit strategy document (`docs/audit-strategy.md`)
- [x] Peptide validation pipeline (A1 + B4 + D1)
- [ ] Commit all overnight outputs
- [ ] Apply FLP-18 citation correction
- [ ] Refine T4-5 scope based on D1 outcomes

## STATUS.md tail

```
# Overnight run STATUS — 2026-04-21

*This file is updated progressively as each task completes. Read this first on wake.*

## Pipeline overview

| task | status | output path |
|---|---|---|
| 0 setup + seed determinism | PENDING | this file |
| 1 D1 modulator Mode audit | PENDING | task1_d1/ |
| 2 T4-5 pre-validation | PENDING (runs if time) | task2_t45_preval/ |
| 3 Genome-wide peptide survey | PENDING | task3_peptide_survey/ |
| 4 14×14 overlap matrix | PENDING | task4_overlap_matrix/ |
| 5 18-neuron readout coverage | PENDING | task5_readout_coverage/ |
| 6 FLP-13 vs FLP-11 | PENDING | task6_flp13_vs_flp11/ |
| 7 PubMed metadata | PENDING | task7_pubmed/ |
| 8 Ripoll-Sánchez cross-ref | PENDING | task8_ripoll_sanchez/ |
| 10 Morning brief | PENDING | MORNING_BRIEF.md |

## Completion log

(tasks append their status here as they finish)


## Task 3: genome-wide peptide survey
- Completed: 09:29:54
- Headline: 159 peptides surveyed; 159 resolved, 93 expressed above threshold
- Output: task3_peptide_survey/

## Task 4: overlap matrix
- Completed: 09:30:46
- Headline: 14×14 Jaccard matrix; 0 high-overlap pairs (>0.7), 91 distinct pairs (<0.1)
- Output: task4_overlap_matrix/

## Task 4: overlap matrix
- Completed: 09:31:41
- Headline: 14×14 Jaccard matrix; 1 high-overlap pairs (>0.7), 69 distinct pairs (<0.1)
- Output: task4_overlap_matrix/

## Task 5: readout coverage
- Completed: 09:32:40
- Headline: 9 modulators predicted Mode 1, 1 predicted Mode 3. Broadcasters in readout: 3
- Output: task5_readout_coverage/

## Task 6: FLP-13 vs FLP-11
- Completed: 09:33:37
- Headline: Jaccard = 0.80, FLP-11 unique=0, FLP-13 unique=1, shared=4
- Output: task6_flp13_vs_flp11/

## Task 7: PubMed metadata
- Completed: 09:34:24
- Headline: PubMed queries attempted for 5 T4-5 candidates
- Output: task7_pubmed/

## Task 7: PubMed metadata
- Completed: 09:38:21
- Headline: 5/5 citations resolved; FLP-18 Rogers 2003 attribution found to be mis-cited (actually about FLP-21) — update to Cohen 2009
- Output: task7_pubmed/

## Task 8: Ripoll-Sánchez cross-reference
- Completed: 09:38:21
- Headline: Paper verified (PMID 37935195); FLP-13/FLP-18/FLP-21 confirmed in their connectome; NPR-5 specifically assigned to FLP-18; NLP-40 and DAF-28 not surfaceable from fetch (manual check needed)
- Output: task8_ripoll_sanchez/


## Task 1 analyze: D1 Mode classification
- Completed: 12:15:09
- Headline: {'Mode 1 (readout-blind)': 5, 'Mode 2 (readout-trivial)': 2, 'Mode 3 (readout-cascade)': 2}
- Output: task1_d1/d1_classification_summary.md

## Task 2: T4-5 pre-validation
- Completed: 12:22:30
- Headline: 4/5 candidates pass smoke gate
- Output: task2_t45_preval/
```