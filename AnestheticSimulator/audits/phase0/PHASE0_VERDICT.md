# WF2 Phase 0 — Foundation/Honesty Audits: VERDICT

**Date:** 2026-06-12
**Gate:** WB-P18 (rank-2 foundation) + WB-P7 (provenance). P18 must PASS before any Phase 1/2 build.
**Outcome:** **P18 = PASS** (rank-2 certified). **P7 = FAIL** (deflation, not halt). **Phase 1 unblocked.**

Implemented + run + adversarially verified by workflow `wf_051811b9-91b`; critical results independently
re-checked by the orchestrator before certification. Audit artifacts: `audits/phase0/{P18-A,P18-B,P18-C,P7}/`.

## P18-A — write-path closure (static AST/regex) — PASS
Only pre-`brain.run()` mutators reachable from `run_single` are `apply_genotype` (phase_g_state_validator.py:367)
+ `apply_anesthetic` (:368). All 8 reachable writes are full-slice scalar broadcasts (`I_ext[:] += scalar*pA`,
`w[:] *= scalar`). Zero per-neuron-varying reachable writers, across worm/fly/mouse factories.

## P18-B — dynamic rank certificate (brian2, no brain.run()) — PASS (independently re-run)
Coverage-complete battery; snapshot `I_ext` + synapse weights pre/post the two mutators; SVD of the stacked
delta matrices. **Re-run by orchestrator 2026-06-12:** singular values of D = [7355.39, 0, 0, …] → #SV>τ = 1;
S = [192.86, 0, …] → #SV>τ = 1; max spatial non-uniformity ρ = **2.74e-16** (vs frozen τ=1e-6, ~10 decades of
margin); coverage 7/7 classes + 4/4 genotype branches. The operator provably collapses to two scalars
(`total_pa` current broadcast + `snare_gain` global synaptic scale) → **QF = G(total_pa, snare_gain)**. Rank-2
certified.

## P18-C — artifact provenance ledger (static call-graph) — PASS (per dated amendment)
Rank-2 attribution clean: G0 (n=19 in-scope, hash-locked), G1 (`contaminated_count == 0` by call-graph AND
schema), G1b (`untraceable_in_scope == 0`) — all PASS on the frozen `decision_rule`. The harness's runtime
`overall=FAIL` came from an un-frozen "missing-artifact" clause triggered by `v7_match3_random_50.csv`, the
never-produced Match#3 ensemble (V1 structurally cannot generate it; documented V7-DEVIATION). Disposed as
SUBSTRATE-DEFERRED — NOT_TESTED; see `P18-C/AMENDMENT_2026-06-12.md`. No hidden per-neuron writer; the
catastrophic "rank-2 is wrong → re-scope ledger → revive impossibility claims" branch is **NOT** triggered.

## P7 — provenance git-archaeology — FAIL (deflation, not halt)
- **P7.A (3 Hz quiescence cutoff): RULE_REDESIGNED_VALUE_DEFENSIBLE.** Dated prereg `934725a` (2026-04-28) pins
  only the rule-FORM (mean rate of {AVA,AVB,AVD,AVE,PVC} < threshold) + a derived calibration procedure
  (90th-pct WT, <5% WT-control dwell), NOT a literal value. The shipped hardcoded 3.0 Hz was introduced
  post-prereg in `34ed2da` (2026-05-02). PROVEN_PREREGISTERED branch is structurally unreachable.
- **P7.B (8-value sat_pa ladder): UNDECIDABLE_BY_TIMESTAMP.** The ladder, ALPHA=0.13, and the
  "recalibrated after W_chem bug fix activated SNARE" signature were co-introduced as **43 pure additions / 0
  removals** in a single commit `34ed2da` — git can neither convict nor exonerate; pre-first-commit
  working-tree co-tuning is unobservable. No ACTIVE_TUNING co-modification diff → no halt.
- **Verified by orchestrator:** commit dates + ordering + the additive-only signature all confirmed on disk.

**Claim impact:** the "one free parameter (α)" framing must narrow. Disclosed in `PROVENANCE.md`. This gates
re-promotion of the α/knapsack honesty claims and the Wave-P ladder work (P11→P10); it does NOT block the
Phase 1 empirical core.

## Disposition
- **Phase 1 unblocked.** Launch order (local CPU): **P8** first (corrected two-coordinate Match#2b null — the
  live fly-deflation risk; gates P16), then P3, P11, P4, P20; P17 runs independently (gates the Paper-2 bridge).
  Greene reserved for P13 ESMFold UNC-80 only.
- P14 (entropy/thermodynamic necessity) remains PARKED as full-energetic-Tier4.
