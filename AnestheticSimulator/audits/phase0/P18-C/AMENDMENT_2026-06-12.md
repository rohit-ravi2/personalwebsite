# P18-C dated prereg amendment — Match#3 artifact disposition

**Date:** 2026-06-12
**Block:** P18-C (artifact provenance ledger)
**Status change:** DISPUTED → PASS (and therefore P18 overall → PASS)
**Authority:** ratified by user; harness-vs-frozen-criterion dispute resolved per prereg-freeze-first.

## The dispute

The P18-C harness returned `overall_verdict = FAIL`. Its three FROZEN gates, however, all PASS:
- `G0_denominator_freeze`: PASS (in-scope set n=19, hash-locked predicate, non-empty).
- `G1_rank2_attribution`: PASS (`contaminated_count == 0` by call-graph AND schema; the higher-rank
  `phase_g_network_perturbation.apply_to_brain` / `NetworkPerturbation` path is provably unreachable from
  every in-scope writer; the suspect `phase_g_halothane_dose_response.csv` is correctly out-of-scope).
- `G1b_untraceable`: frozen `pass_condition` is literally `untraceable_in_scope == 0` — **satisfied**
  (`untraceable_in_scope_csv == 0`).

The frozen `decision_rule` is: *"PASS iff G0 frozen-and-nonempty AND G1 contaminated_count==0 AND G1b
untraceable_in_scope==0."* All three hold → the frozen criterion yields **PASS**.

The harness nonetheless forced FAIL on an **un-frozen** clause it added at runtime
(`missing_never_produced_count == 0`, "zero referenced-but-absent in-scope artifacts"), triggered by a single
artifact: `artifacts/v7_random_ensemble/v7_match3_random_50.csv`.

## Why this is not a provenance failure

`v7_match3_random_50.csv` is a **genuinely-never-produced PLANNED artifact**, already recorded in
`artifacts/v7_random_ensemble/v7_random_ensemble_verdict.json` and `docs/v7_final_summary.md` §3.3 as a
**V7-DEVIATION**. The reason it was never produced is structural, not an error: V1's `resolve_target_neurons`
returns `range(brain.N)` — every mechanism class hits every neuron, so cell-type spread is uniform **by
construction** and Match#3 collapses mathematically to Match#2 (independently proven by the rank-2 certificate
in P18-B: the current operator is rank-1 in space, so there is no spatial degree of freedom for a per-cell-type
spread to live in). The file is not deleted, not fabricated, and not an orphan with an unknown writer — it is
the **absence of a capability the substrate provably lacks**.

The harness conflated *untraceable* (orphan with no writer — a real provenance hole) with *missing* (a planned
artifact never produced). Only the former was frozen into G1b. The deviation was in the honest (stricter)
direction, so leak/fake-pass risk is none — but per prereg-freeze-first, evaluation must be against the frozen
criterion.

## Disposition (option b)

`v7_match3_random_50.csv` / Match#3 is recorded as **SUBSTRATE-DEFERRED — NOT_TESTED**: V1's global broadcast
structurally cannot generate it, and producing it is exactly the purpose of the Tier-3 keystone build **P1_P2**
(CeNGEN per-class expression vectors `x_c` lifting operator rank 2 → min(7,N)). Option (a) — formalizing a
"zero referenced-but-absent artifacts" clause into a hard FAIL — was rejected as logically incoherent: it would
HALT P1_P2, which is the only thing that can produce the missing artifact, creating a deadlock.

Under the frozen criterion, P18-C = **PASS**. Combined with P18-A PASS and P18-B PASS (the latter independently
re-run on 2026-06-12: D rank-1 with max ρ = 2.74e-16, S rank-1, 7/7 class + 4/4 genotype coverage), the
**P18 foundation gate is PASS** and the rank-2 premise is certified. The harness's `missing_never_produced`
bookkeeping is retained as informational (it correctly notes the Match#3 deferral) but is NOT a gate.
