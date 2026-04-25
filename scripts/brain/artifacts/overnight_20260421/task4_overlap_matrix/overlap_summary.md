# Task 4 — Modulator target-set overlap matrix

Generated: 2026-04-21 09:31:41

14 modulators (9 existing + 5 T4-5 candidates) × 14. Pairwise 
Jaccard overlap of target neuron classes (TPM > 0.5) 
expressing each modulator's receptor set.

## Target-set sizes

| modulator | receptors | # target classes |
|---|---|---|
| **FLP-11** | npr-1, npr-22, dmsr-1, dmsr-7, npr-11 | 4 |
| **FLP-1** | npr-4, npr-5, npr-11 | 1 |
| **FLP-2** | npr-30, frpr-18 | 0 |
| **NLP-12** | ckr-1, ckr-2 | 0 |
| **PDF-1** | pdfr-1 | 2 |
| **5HT** | mod-1, ser-1, ser-4, ser-5, ser-6, ser-7 | 12 |
| **DA** | dop-1, dop-2, dop-3, dop-4 | 5 |
| **TA** | tyra-2, tyra-3, ser-2, lgc-55 | 7 |
| **OA** | octr-1, ser-3, ser-6 | 2 |
| **FLP-13** | dmsr-1, dmsr-2 | 5 |
| **FLP-18** | npr-1, npr-4, npr-5 | 2 |
| **FLP-21** | npr-1 | 1 |
| **NLP-40** | aex-2 | 0 |
| **DAF-28** | daf-2 | 0 |

## Full overlap matrix (Jaccard)

| | FLP-11 | FLP-1 | FLP-2 | NLP-12 | PDF-1 | 5HT | DA | TA | OA | FLP-13 | FLP-18 | FLP-21 | NLP-40 | DAF-28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **FLP-11** | 1.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.07 | 0.12 | 0.10 | 0.20 | 0.80 | 0.20 | 0.25 | 0.00 | 0.00 |
| **FLP-1** | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| **FLP-2** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **NLP-12** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **PDF-1** | 0.20 | 0.00 | 0.00 | 0.00 | 1.00 | 0.08 | 0.00 | 0.12 | 0.00 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 |
| **5HT** | 0.07 | 0.00 | 0.00 | 0.00 | 0.08 | 1.00 | 0.06 | 0.06 | 0.08 | 0.13 | 0.08 | 0.08 | 0.00 | 0.00 |
| **DA** | 0.12 | 0.20 | 0.00 | 0.00 | 0.00 | 0.06 | 1.00 | 0.00 | 0.17 | 0.11 | 0.40 | 0.20 | 0.00 | 0.00 |
| **TA** | 0.10 | 0.00 | 0.00 | 0.00 | 0.12 | 0.06 | 0.00 | 1.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| **OA** | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.17 | 0.00 | 1.00 | 0.17 | 0.33 | 0.50 | 0.00 | 0.00 |
| **FLP-13** | 0.80 | 0.00 | 0.00 | 0.00 | 0.40 | 0.13 | 0.11 | 0.09 | 0.17 | 1.00 | 0.17 | 0.20 | 0.00 | 0.00 |
| **FLP-18** | 0.20 | 0.50 | 0.00 | 0.00 | 0.00 | 0.08 | 0.40 | 0.00 | 0.33 | 0.17 | 1.00 | 0.50 | 0.00 | 0.00 |
| **FLP-21** | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.20 | 0.00 | 0.50 | 0.20 | 0.50 | 1.00 | 0.00 | 0.00 |
| **NLP-40** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **DAF-28** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## High-overlap pairs (> 0.7 — potential redundancy)

- **FLP-11 ↔ FLP-13**: Jaccard = 0.80

## Distinct pairs (< 0.1 — complementary coverage)

- FLP-11 ↔ FLP-1: Jaccard = 0.00
- FLP-11 ↔ FLP-2: Jaccard = 0.00
- FLP-11 ↔ NLP-12: Jaccard = 0.00
- FLP-11 ↔ NLP-40: Jaccard = 0.00
- FLP-11 ↔ DAF-28: Jaccard = 0.00
- FLP-1 ↔ FLP-2: Jaccard = 0.00
- FLP-1 ↔ NLP-12: Jaccard = 0.00
- FLP-1 ↔ PDF-1: Jaccard = 0.00
- FLP-1 ↔ 5HT: Jaccard = 0.00
- FLP-1 ↔ TA: Jaccard = 0.00

## T4-5 candidate redundancy check vs existing 9

- **FLP-13** → top overlaps: FLP-11=0.80, PDF-1=0.40, OA=0.17 [REDUNDANT]
- **FLP-18** → top overlaps: FLP-1=0.50, DA=0.40, OA=0.33 [PARTIAL]
- **FLP-21** → top overlaps: OA=0.50, FLP-11=0.25, DA=0.20 [PARTIAL]
- **NLP-40** → top overlaps: FLP-11=0.00, FLP-1=0.00, FLP-2=0.00 [DISTINCT]
- **DAF-28** → top overlaps: FLP-11=0.00, FLP-1=0.00, FLP-2=0.00 [DISTINCT]
