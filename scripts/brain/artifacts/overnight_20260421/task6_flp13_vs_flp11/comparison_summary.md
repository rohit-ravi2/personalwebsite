# Task 6 — FLP-13 vs FLP-11 target comparison

Generated: 2026-04-21 09:33:37

FLP-11 receptors: npr-1, npr-22, dmsr-1, dmsr-7, npr-11
FLP-13 receptors: dmsr-1, dmsr-2 (per Nath 2016 ALA sleep pathway)
TPM threshold for target detection: 0.5

## Target-set comparison

- **FLP-11 targets:** 4 classes (M1, PVW, RIG, VA12)
- **FLP-13 targets:** 5 classes (AIM, M1, PVW, RIG, VA12)
- **Shared targets:** 4 classes (M1, PVW, RIG, VA12)
- **FLP-11 only:** 0 classes (none)
- **FLP-13 only:** 1 classes (AIM)
- **Jaccard overlap:** 0.80

## Verdict

- **REDUNDANT** (Jaccard = 0.80) — FLP-13 and FLP-11 target largely the same neurons. Adding FLP-13 to T4-5 may not provide distinct empirical coverage.

## Readout-18 coverage

- FLP-11 target classes in readout: 0 (none)
- FLP-13 target classes in readout: 0 (none)

## Per-class target table

| class | FLP-11 | FLP-13 | FLP-11 rec (TPM) | FLP-13 rec (TPM) | in readout |
|---|---|---|---|---|---|
| AIM | - | ✓ |  | dmsr-2=0.86 | - |
| M1 | ✓ | ✓ | dmsr-1=0.75 | dmsr-1=0.75 | - |
| PVW | ✓ | ✓ | dmsr-1=0.76 | dmsr-1=0.76 | - |
| RIG | ✓ | ✓ | dmsr-1=0.59 | dmsr-1=0.59 | - |
| VA12 | ✓ | ✓ | npr-1=0.66;dmsr-1=0.76 | dmsr-1=0.76 | - |

## T4-5 implication

- FLP-13 is LIKELY REDUNDANT with FLP-11 at current receptor-coverage level. Reconsider inclusion or add only if phenotype evidence distinguishes them.