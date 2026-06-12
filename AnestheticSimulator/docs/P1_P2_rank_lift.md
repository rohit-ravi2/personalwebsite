# P1_P2 — Minimal-delta-V2 rank-lift (CeNGEN x_c) + SOL7 freeze-statistic harness

**Date:** 2026-06-12 · **Phase:** 2 (KEYSTONE) · **Gated on:** P18 PASS (V1 rank-2).
Prereg (frozen BEFORE build/run): `audits/phase2/P1_P2/prereg.json`.
Module: `src/state_validation/p1_p2_rank_lift.py` (NON-DESTRUCTIVE — frozen
`apply_anesthetic` untouched; new path `apply_anesthetic_v2`). Artifacts: `artifacts/p1_p2/`.

## What was built

V1 broadcasts one scalar `total_pa` over ALL neurons and one global SNARE synaptic
scalar → rank-2, so Match#3 (cell-type targeting) was entailed-by-construction and
NEVER tested. V2 replaces the all-ones broadcast with CeNGEN per-class expression
vectors `x_c` (7×300):

    I_ext[i] += alpha * sum_c (-sat_c * e_c) * x_c[c, i]
    SNARE: syn.w *= 1 + (snare_max-1)*e_snare*x_unc64[presyn(edge)]   (per-edge, presynaptic unc-64)

- 5 channel classes (k2p, nca, gaba, glucl, nachr) use TPM-derived soft [0,1] gates
  (max over marker genes / 75th-pct-nonzero threshold, label-free, frozen).
- 2 metabolic classes (complex_i/ii) are ones-by-biology (ubiquitous, frozen).
- `x_unc64` gates SNARE presynaptically. With every `x_c`=ones AND `x_unc64`=ones the
  V2 path is **bit-identical** to V1.

## FAST gate results (all PASS)

| gate | result | number |
|---|---|---|
| **G_BIT_IDENTITY** | **PASS** | max per-neuron I_ext err 3.55e-15 pA, snare err 0.0, over 36 (profile,dose) cases (tol 1e-12) |
| **G0_RANK_LIFT_REALIZED** | **PASS** | two equal-(total_pa,snare) profiles → ‖q_A−q_B‖₂ = 31.3 pA (≫ 1e-6); under x_c=ones the same pair gives 1.83 pA (within sampler's 5% coord tol) |
| **G_SOL7_able_to_fail** | **PASS** | identical-support (V1) → eta²=0, PR=1; disjoint-support → PR differs (0.281 vs 0.306, diff 0.025) |

x_c content hash: `f6f55a43cb29…`.

## Finding caught by the able-to-fail screen: eta² is degenerate; PR is operative

The prereg froze the SOL7 spread statistic as between-cell-type variance fraction
**eta²** AND participation ratio **PR**, with the able-to-fail screen requiring
disjoint-support profiles to give *different* eta². The screen **falsified eta²**:
because every `x_c` row is constant within each CeNGEN class, ANY class-resolved drive
has eta²=1 **exactly**, regardless of support — eta² cannot discriminate disjoint
supports on this substrate. **PR** (the prereg's G1 Match#3 statistic) IS the operative
discriminator and differs across disjoint supports. This is a prereg-statistic
correction *caught by the screen doing its job*, not a silent loosening: G1 already
uses PR, so no Match#3 threshold changes. Recorded in
`artifacts/p1_p2/g_sol7_able_to_fail.json` (`eta2_degeneracy_FINDING`).

## Coverage caveat (FLAGGED for the heavy run + SOL8)

The high-resolution CeNGEN file (`expression_neuron_mean.csv`, 91 classes) collapses
the ventral-cord motor neurons (DA/DB/VA/VB/VD/DD/AS, plus a few D/V sensory splits),
so **159/300** neurons have ≥1 channel-class `x_c`>0 and **141/300** are channel-
uncovered (driven only by complex_i/ii ones-by-biology). The uncovered set lowers the
conserved profile's PR (more concentrated) and could bias G1 toward the too-special
floor — this MUST be weighed in the G1 verdict and is exactly what SOL8 NeuroPAL
Jaccard (≥0.5 for ≥5/7 classes) is designed to check. Not a bug; a documented data-
resolution gap of the derived CeNGEN file.

## NOT launched here (heavy)

The Match#3 ensemble (G1_MATCH3_SPATIAL_SPECIAL + G2_MATCH3_NOT_ENTAILED) on
worm(300)+fly(2952), mouse EXCLUDED (random graph). Entry point:
`p1_p2_rank_lift.py match3 <organism>`. See prereg `heavy_run`.

## SOL8 follow-on (separate streaming job)

NeuroPAL external grounding (Jaccard join-correctness HARD gate + quiescence-structure
confirmatory + bout-yield fork). Streams NeuroPAL ~815MB + Atanas NWB (~25-27GB each,
h5py partial reads only). Runs only after G0+G1 land.
