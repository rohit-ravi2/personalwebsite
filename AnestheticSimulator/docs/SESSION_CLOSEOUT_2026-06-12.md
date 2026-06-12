# AnestheticSimulator — session close-out (2026-06-12)

A full pass from "assert the claims" to "adversarially test the claims," first-principles, with
preregistered accept-either-way gates. Every fix was capable of deflating a claim; several did.

## Foundation (Phase 0) — committed
- **P18 = PASS** — the V1 operator is certified **rank-2** (QF = G(total_pa, snare_factor)); re-run
  independently (max non-uniformity 2.7e-16). Write-path closed, attribution clean.
- **P7 = FAIL (deflation)** — "one free parameter (α)" narrowed: sat_pa ladder UNDECIDABLE_BY_TIMESTAMP,
  3 Hz cutoff redesigned post-prereg. Disclosed in `PROVENANCE.md`.

## Phase 1 / 2 results — committed
- **P8** — corrected two-coordinate Match#2b null. The Sub-Q1 trichotomy was a control-bug artifact
  (phantom SNARE current + unmatched snare_factor). worm 0→28%, fly 4.76→26% (G3 PASS, frozen), mouse 46→38%.
  worm≈fly modest-significant (p<0.002), mouse marginal. "fly uniquely special" + "worm anchor-overfit" fall.
- **P3** — mean-field collapse theorem PASS. CV²∝1/K (slope −1.029, R²=0.999); mouse-at-median is a derived
  corollary, not a null. worm/fly residual = real-connectome signal.
- **P11** — Wave-P K_p reframe. Multi-target ENGAGEMENT robust (M0 29/30) but SATURATION was a Kp double-count
  (M0 0/30 vs M2 26/30 >90%). "saturating occupancy" → moderate partial ~0.25–0.5. Molecular-layer only.
- **P1_P2 (keystone)** — minimal-delta-V2 rank-lift built correctly (bit-identical to V1 at x_c=ones; real
  CeNGEN x_c; rank lift realized; SOL7 able-to-fail screen caught η² degeneracy → PR). **worm Match#3 =
  NULL_DEFLATE** (conserved cell-type spread at the 46th percentile). Lifting the operator rank did NOT reveal
  conserved-target cell-type specificity on worm. **Fly fabrication caught + fixed** (no Drosophila atlas;
  guarded). Fly remains the one untested payoff, gated on data we don't have.

## Phase 1 heavy runs — LAUNCHED (background chain `run_phase1_chain.sh`, ~7hr)
Verdicts land in `artifacts/`; collect + commit when complete. All accept-either-way.
- **P4** — Gate-4 entailment: G4-A pooled R²=0.907 FAILS the literal 0.95 gate → RETAIN_8.4_independent
  (reported, not self-overridden). Heavy: SNARE-orthogonal falsifier (70 sims).
- **P17** — readout-validity vs Kato/NeuroPAL/Atanas. Data-only preview already leans DEMOTE (Atanas Q1
  straddles 0) → would demote the immobilization-readout to "network statistic, not validated behavioral
  quiescence" and BLOCK the Paper-2 bridge. Heavy: full Atanas streaming.
- **P20** — genotype×anesthetic two-block reachability (1608 sims). GATE-A: positive-routing vs
  bookkeeping-deflation.
- **P13-SOL28** — nca magnitude interval sweep (2560 sims). G1 provenance PASS ([75,120] pA; 40 pA legacy below
  floor). G2 quorum-survival is the heavy gate.

## Not completed / deferred (honest residual)
- **P16** — held-out structure→activity exam: fast gates PASS (strict degree-preserving shuffle invariants,
  NWB parse, positive control), but the **heavy path is an unwritten stub** (missing scoring loop +
  non-destructive spike-export). Scaffold committed; needs implementation before it can run. Also gated on P8
  fly (done). Bayesian-likely NULL.
- **P13-SOL27** — ESMFold NCA-1/UNC-80: needs **NYU Greene HPC ≥24 GB GPU** (UNC-80 OOM'd locally). User action.
  The load-bearing MAGNITUDE half is settled locally by SOL28; structure half is secondary (NALCN has no Kd →
  nca stays uncalibratable regardless).
- **P19** — gap-on/off confirmatory leaf: optional, lowest severity, not run.
- **P14** — thermodynamic-necessity / entropy: PARKED (genuinely full-energetic-Tier4 "V8" build).

## Net
The core thesis survives (multi-class quorum; multi-target engagement), but the **headline magnitudes were
corrected down to what's licensed**: the Sub-Q1 trichotomy, the occupancy saturation, the "one free α," and the
cell-type-targeting hope (worm NULL). The most important scientific question — are these the *right* specific
targets, not just plausible ones — is now sharply defined and answered "not established on this substrate, even
after the rank-lift built to test it," with fly the one data-gated open frontier. This is the gated program
working: deflation-capable, accept-either-way, anti-target-engineering throughout.
