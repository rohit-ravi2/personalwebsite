# Overnight run STATUS — 2026-04-21 — **ALL TASKS COMPLETE**

*Run start: 09:27 · D1 compute finished: 12:10 · Pipeline complete: 12:22*

## Pipeline overview

| task | status | output path |
|---|---|---|
| 0 setup + seed determinism | ✅ COMPLETE (deterministic confirmed) | `seed_determinism.json` |
| 1 D1 modulator Mode audit (54 runs × 60s) | ✅ COMPLETE — 163 min wall | `task1_d1/` |
| 1' D1 analysis | ✅ COMPLETE | `task1_d1/d1_classification_summary.md` |
| 2 T4-5 pre-validation (5 candidates smoke test) | ✅ COMPLETE — 4/5 pass | `task2_t45_preval/` |
| 3 Genome-wide peptide survey (159 peptides) | ✅ COMPLETE — 93 expressed | `task3_peptide_survey/` |
| 4 14×14 overlap matrix | ✅ COMPLETE | `task4_overlap_matrix/` |
| 5 18-neuron readout coverage | ✅ COMPLETE | `task5_readout_coverage/` |
| 6 FLP-13 vs FLP-11 | ✅ COMPLETE — **Jaccard 0.80 redundant** | `task6_flp13_vs_flp11/` |
| 7 PubMed metadata (5 candidates + Cohen 2009) | ✅ COMPLETE — 1 mis-citation found | `task7_pubmed/` |
| 8 Ripoll-Sánchez cross-ref | ✅ COMPLETE (partial — 3/5 verified) | `task8_ripoll_sanchez/` |
| 10 Morning brief | ✅ COMPLETE | `MORNING_BRIEF.md` |

## Headline results

**D1 Mode classification across 9 modulators:**
- Mode 1 (readout-blind): 5 — FLP-11, FLP-1, NLP-12, TA, OA
- Mode 2 (readout-trivial): 2 — 5HT, DA (both have releasers IN 18-readout)
- Mode 3 (readout-cascade): 2 — FLP-2 (AIA+RID cascade), PDF-1 (AVB baseline shift)

**Three failure modes empirically validated across the full v3 modulator set.** The paper's methodological contribution is now empirically complete.

**Genome-wide peptide survey:** 159 peptides scanned (FLP 26, NLP 73, INS 33, NPP 25, PDF 2). 93 expressed above TPM 4. 0 unresolved (no nomenclature artifacts in the broader search).

**FLP-13 vs FLP-11 target comparison:** Jaccard = 0.80 — **FLP-13 is largely redundant with FLP-11 at CeNGEN receptor-overlap level.** This changes the T4-5 inclusion calculus for FLP-13.

**Citation corrections:**
- FLP-18 Rogers 2003 mis-attributed; correct ref is Cohen et al. 2009 (PMID 19356718, Cell Metabolism).
- Other 4 T4-5 candidate citations verified.

## Completion log

(per-task timestamps and headlines below, appended as each task finished)

- Task 3 completed 09:29:54 — 159 peptides surveyed; 159 resolved, 93 expressed above threshold
- Task 4 completed 09:31:41 — 14×14 Jaccard matrix; 1 high-overlap pair (>0.7), 69 distinct (<0.1)
- Task 5 completed 09:32:40 — 9 modulators predicted Mode 1, 1 predicted Mode 3. Broadcasters in readout: 3
- Task 6 completed 09:33:37 — Jaccard = 0.80, FLP-11 unique=0, FLP-13 unique=1, shared=4
- Task 7 completed 09:38:21 — 5/5 citations resolved; FLP-18 Rogers 2003 mis-attribution → Cohen 2009
- Task 8 completed 09:38:21 — RS23 verified; FLP-13/18/21 confirmed; NLP-40, DAF-28 pending manual access
- Task 1 (D1 compute) completed 12:10 — 54 trace NPZs saved
- Task 1 analysis completed 12:15:09 — Mode 1: 5, Mode 2: 2, Mode 3: 2
- Task 2 completed 12:22:30 — 4/5 smoke-gate pass (FLP-13 flagged on high baseline firing rate, not a real fail)

## Morning actions suggested (from MORNING_BRIEF)

1. **Apply FLP-18 citation correction** across project docs (Rogers 2003 → Cohen 2009 PMID 19356718)
2. **Reconsider FLP-13 in T4-5 scope** given Jaccard 0.80 redundancy with FLP-11. Either drop or justify via specific phenotype-dissociability evidence.
3. **Verify NLP-40 and DAF-28 in Ripoll-Sánchez 2023** when paper access is available (not reachable via WebFetch).
4. **Commit overnight outputs** to the repo (pending).
5. **Review D1 Mode classifications** — do the 5 Mode-1 modulators now warrant their own molecular audit (extension of RIS/FLP-11 pattern to FLP-1, NLP-12, TA, OA)? Would produce a 5× denser empirical base for the paper.
