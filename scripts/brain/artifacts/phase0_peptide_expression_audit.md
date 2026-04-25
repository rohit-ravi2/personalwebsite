# Phase 0 — Task A1 + B4: Peptide candidate expression audit

Local-data validation filter for T4-5 peptide expansion. Uses 
CeNGEN TPM threshold 4.0 (Taylor 2021 convention) 
and the 18-neuron validated classifier readout to predict each 
candidate's expected audit Mode.

## Existing modulators (baseline)

| modulator | synthesis gene | max TPM | # classes > 4 | top expressing |
|---|---|---|---|---|
| FLP-11 | flp-11 | 985.2 | 1 | RIS |
| FLP-1 | flp-1 | 1285.39 | 1 | AVK |
| FLP-2 | flp-2 | 47.04 | 2 | AIA;RID |
| NLP-12 | nlp-12 | 308.52 | 1 | DVA |
| PDF-1 | pdf-1 | 361.74 | 17 | RID;AIY;AVB;PVT;SDQ |
| 5HT | tph-1 | 2.87 | 0 |  |
| DA | cat-2 | 2.4 | 0 |  |
| TA | tdc-1 | 15.24 | 1 | RIM |
| OA | tbh-1 | 2.79 | 0 |  |

## Candidate peptides

| peptide | category | synth gene | max TPM | #expr | phenotype | predicted Mode |
|---|---|---|---|---|---|---|
| **NLP-22** | quiescence | nlp-22 | 0.04 | 0 | Strong | N/A (not expressed) |
| **NLP-27** | quiescence | nlp-27 | 0.03 | 0 | Weak | N/A (not expressed) |
| **NLP-37** | quiescence | nlp-37 | 0.0 | 0 | Weak | N/A (not expressed) |
| **NLP-50** | quiescence | nlp-50 | 145.05 | 20 | Weak | Mode 1 predicted (readout-blind) |
| **FLP-13** | quiescence | flp-13 | 128.09 | 4 | Strong | Mode 2 predicted (readout-trivial) |
| **FLP-24** | quiescence | flp-24 | 271.72 | 5 | Moderate | Mode 1 predicted (readout-blind) |
| **INS-22** | quiescence | ins-22 | 2.88 | 0 | Weak | N/A (not expressed) |
| **FLP-1** | locomotion | flp-1 | 1285.39 | 1 | Strong | Mode 1 predicted (readout-blind) |
| **FLP-18** | locomotion | flp-18 | 111.23 | 3 | Strong | Mode 1 predicted (readout-blind) |
| **FLP-21** | locomotion | flp-21 | 33.1 | 4 | Strong | Mode 1 predicted (readout-blind) |
| **NLP-12** | locomotion | nlp-12 | 308.52 | 1 | Strong | Mode 1 predicted (readout-blind) |
| **NLP-15** | locomotion | nlp-15 | 22.26 | 12 | Weak | Mode 2 predicted (readout-trivial) |
| **NLP-14** | feeding | nlp-14 | 41.61 | 3 | Weak | Mode 1 predicted (readout-blind) |
| **NLP-24** | feeding | nlp-24 | 0.03 | 0 | Weak | N/A (not expressed) |
| **NLP-40** | feeding | nlp-40 | 165.0 | 6 | Strong | Mode 2 predicted (readout-trivial) |
| **INS-1** | feeding | ins-1 | 3.41 | 0 | Moderate | N/A (not expressed) |
| **INS-6** | feeding | ins-6 | 50.22 | 2 | Moderate | Mode 1 predicted (readout-blind) |
| **INS-7** | feeding | ins-7 | 0.18 | 0 | Moderate | N/A (not expressed) |
| **INS-17** | feeding | ins-17 | 40.07 | 2 | Weak | Mode 1 predicted (readout-blind) |
| **INS-18** | feeding | ins-18 | 15.1 | 6 | Weak | Mode 1 predicted (readout-blind) |
| **DAF-28** | feeding | daf-28 | 14.95 | 3 | Strong | Mode 1 predicted (readout-blind) |
| **NLP-3** | sensory | nlp-3 | 15.65 | 4 | Weak | Mode 2 predicted (readout-trivial) |
| **NLP-9** | sensory | nlp-9 | 13.63 | 3 | Moderate | Mode 1 predicted (readout-blind) |
| **NLP-29** | sensory | nlp-29 | 0.03 | 0 | Moderate | N/A (not expressed) |
| **FLP-33** | sensory | flp-33 | 0.0 | 0 | Weak | N/A (not expressed) |
| **nssp-29** | artifact_check | nssp-29 | 0.0 | 0 | None | N/A (not expressed) |
| **Y51H7C.3** | artifact_check | Y51H7C.3 | 0.0 | 0 | None | N/A (not expressed) |
| **R04A9.1** | artifact_check | R04A9.1 | 7.96 | 1 | None | Mode 1 predicted (readout-blind) |

## A1 filter verdict

- **Passed** (resolved + expressed above threshold): 17/28
- **Failed gene-symbol resolution**: NLP-37, FLP-33, nssp-29, Y51H7C.3
- **Resolved but not expressed above threshold**: NLP-22, NLP-27, INS-22, NLP-24, INS-1, INS-7, NLP-29

## T4-5 inclusion decision per candidate (per research-review criteria)

Inclusion criteria (pre-committed):
1. A1 passes (expressed above TPM=4 in ≥1 class)
2. Phenotype strength = Strong (published quantitative ablation phenotype)
3. Predicted Mode is testable (Mode 2, Mode 3, or Mode 1 with molecular audit design)

| peptide | A1 | phenotype | mode | decision |
|---|---|---|---|---|
| NLP-22 | ✗ | Strong | N/A (not expressed) | exclude (A1 failed) |
| NLP-27 | ✗ | Weak | N/A (not expressed) | exclude (A1 failed) |
| NLP-37 | ✗ | Weak | N/A (not expressed) | exclude (A1 failed) |
| NLP-50 | ✓ | Weak | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| FLP-13 | ✓ | Strong | Mode 2 predicted (readout-triv | **include T4-5** |
| FLP-24 | ✓ | Moderate | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| INS-22 | ✗ | Weak | N/A (not expressed) | exclude (A1 failed) |
| FLP-1 | ✓ | Strong | Mode 1 predicted (readout-blin | **include T4-5** |
| FLP-18 | ✓ | Strong | Mode 1 predicted (readout-blin | **include T4-5** |
| FLP-21 | ✓ | Strong | Mode 1 predicted (readout-blin | **include T4-5** |
| NLP-12 | ✓ | Strong | Mode 1 predicted (readout-blin | **include T4-5** |
| NLP-15 | ✓ | Weak | Mode 2 predicted (readout-triv | hold (phenotype not Strong) |
| NLP-14 | ✓ | Weak | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| NLP-24 | ✗ | Weak | N/A (not expressed) | exclude (A1 failed) |
| NLP-40 | ✓ | Strong | Mode 2 predicted (readout-triv | **include T4-5** |
| INS-1 | ✗ | Moderate | N/A (not expressed) | exclude (A1 failed) |
| INS-6 | ✓ | Moderate | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| INS-7 | ✗ | Moderate | N/A (not expressed) | exclude (A1 failed) |
| INS-17 | ✓ | Weak | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| INS-18 | ✓ | Weak | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| DAF-28 | ✓ | Strong | Mode 1 predicted (readout-blin | **include T4-5** |
| NLP-3 | ✓ | Weak | Mode 2 predicted (readout-triv | hold (phenotype not Strong) |
| NLP-9 | ✓ | Moderate | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |
| NLP-29 | ✗ | Moderate | N/A (not expressed) | exclude (A1 failed) |
| FLP-33 | ✗ | Weak | N/A (not expressed) | exclude (A1 failed) |
| nssp-29 | ✗ | None | N/A (not expressed) | exclude (A1 failed) |
| Y51H7C.3 | ✗ | None | N/A (not expressed) | exclude (A1 failed) |
| R04A9.1 | ✓ | None | Mode 1 predicted (readout-blin | hold (phenotype not Strong) |

## Caveats

- **A3 overlap analysis and full receptor-target coverage NOT yet included** — 
  B4 Mode prediction is based on releaser-in-readout only, not receptor-target coverage. 
  Extending requires pulling receptor expression per peptide and overlapping with readout. 
  Follow-up task.
- **Receptor assignments currently from literature only** — should cross-reference with 
  Ripoll-Sánchez 2023 predicted receptor-ligand pairs for completeness.
- **Single-cell variance within class (Task A4) not computed here** — CeNGEN's aggregated 
  class means may hide cell-to-cell variability.
