# PROVENANCE — calibration honesty disclosure (P7 audit, 2026-06-12)

This file discloses the outcome of the WF2 Phase 0 **P7 provenance audit** (git-archaeology, no literature),
which narrows the calibration-honesty framing of the v7 network/behavioral validator. It is the precondition
for any re-promotion of the "one free parameter" / knapsack honesty claims and for the Wave-P sat_pa ladder work.

## What was audited

Whether the two non-α calibration inputs were frozen **before** the halothane α was fit:
1. the **8-value `sat_pa` magnitude ladder** (per-mechanism-class saturating current, pA);
2. the **3.0 Hz quiescence cutoff** (command-interneuron mean-rate threshold defining the quiescent state).

## Findings (verified on disk)

- **3 Hz cutoff — RULE_REDESIGNED_VALUE_DEFENSIBLE.** The dated preregistration (commit `934725a`, 2026-04-28)
  pins only the rule **form** — "mean firing rate of {AVA,AVB,AVD,AVE,PVC} < threshold" — and a **derived**
  calibration procedure (90th percentile of WT, <5% WT-control dwell). It does **not** pin a literal value. The
  shipped, hardcoded `3.0 Hz` constant was introduced **after** the prereg, in commit `34ed2da` (2026-05-02).
  The value is biologically defensible (within the WT-calibrated band), but the "preregistered value" branch is
  structurally unreachable: the rule form was preregistered; the numeric value was redesigned post-prereg.

- **`sat_pa` ladder — UNDECIDABLE_BY_TIMESTAMP.** The full 8-value ladder, `ALPHA = 0.13`, and the
  "recalibrated after the W_chem propagation bug fix activated SNARE" signature were all **co-introduced as 43
  pure additions / 0 removals in a single commit** (`34ed2da`). Git therefore can neither convict nor exonerate:
  pre-first-commit working-tree co-tuning of the ladder against α is unobservable from history. No
  co-modification ("active tuning") diff exists, so this is not a proven tuning event — but it is **not a
  vindication** either.

## Corrected claim language

The headline "**one free parameter (α)**" is narrowed to:

> One fitted scalar (**α**, itself re-fit 0.22 ↔ 0.13 after the SNARE-activating bug fix), conditional on
> (i) a `sat_pa` magnitude ladder of **undecidable a-priori provenance** (git can neither convict nor exonerate
> pre-first-commit co-tuning), and (ii) a quiescence cutoff (3.0 Hz) whose **rule-form was preregistered but
> whose numeric value was redesigned/hardcoded post-prereg** (value biologically defensible).

## Scope of impact

- Gates re-promotion of the α / knapsack "one-free-parameter" honesty claims until this disclosure stands.
- Gates the Wave-P `sat_pa` ladder honesty work (roadmap P11 → P10).
- Does **not** affect the rank-2 result (P18 PASS, independently certified) and does **not** halt the Phase 1
  empirical core. It is a deflation/narrowing, not a reversal.

*Source: `audits/phase0/P7/` (prereg.json, harness.py, result.json, run.log) and `audits/phase0/PHASE0_VERDICT.md`.*
