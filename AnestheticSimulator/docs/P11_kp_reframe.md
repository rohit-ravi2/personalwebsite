# P11 — Wave-P K_p reference-frame recompute: engagement robust, saturation was inflated

**Date:** 2026-06-12 · **Verdict:** G_P11_1 PASS · G_P11_2 ROBUST · G_P11_3 PASS. Molecular-layer honesty
finding only (no network flip). Prereg `audits/phase1/P11/prereg.json`; numeric `artifacts/p11_kp_reframe/`.

## The frame error

Occupancy `θ = C/(C+Kd)` requires `C` and `Kd` in the same reference frame. Vina's `Kd` is aqueous-1M-
standard-state referenced (`affinity_to_kd_uM`: `Kd = exp(ΔG/RT)·1e6`); the comparator is the clinical
**aqueous** EC50. The shipped `phase_c_occupancy.py:152` computes `conc_eff = Kp·conc_aq` for membrane targets
**without** also scaling `Kd` — a frame mismatch. If both are scaled consistently, Kp cancels:
`(Kp·C)/(Kp·C + Kp·Kd) = C/(C+Kd)`. So the physically correct occupancy is **Kp-free (M0)**; the shipped
full-Kp (M2) over-counts. For halothane, Kp = 250 — a 250× concentration inflation against an unscaled Kd.

## Result (halothane, 1×EC50, frozen Vina Kd / EC50 / Kp; vary only the partition)

| model | partition | engagement >10% | saturation >90% |
|---|---|---|---|
| **M0** (no Kp, correct) | Kp⁰ | **29 / 30** | **0 / 30** |
| M1 (√Kp, bracket) | Kp⁰·⁵ | 30 / 30 | 7 / 30 |
| M2 (full Kp, shipped) | Kp¹ | 30 / 30 | 26 / 30 |

Frame-fragility spread (M2−M0) = **1** → the engagement **count** is frame-robust.

## Honest reading

- **The multi-target engagement claim SURVIVES.** Under the correct frame, 29/30 targets still engage >10% at
  1×EC50 — because the Vina Kd (~357–984 µM) is comparable to the EC50 (340 µM), so occupancy is ~0.25–0.5
  even with no partition enhancement. The "anesthetics engage many targets" / discrimination claim stands.
- **The SATURATION framing was a Kp double-count artifact.** The shipped model put 26/30 targets at >90%
  (near-maximal) occupancy; under the correct frame **zero** are saturated. The corrected occupancies are
  *moderate partial* (~0.25–0.5) — which is, if anything, more biologically sensible for a partial multi-target
  perturbation model.
- **One gene (1/30) clears 10% only via Kp** — the gate is live (G_P11_3), but Kp changes the engagement count
  by a single target.

## Required corrections

- Anywhere the Wave-P page / docs describe **"saturating," "maximal," or "near-100% multi-target occupancy"**,
  correct to **"moderate partial occupancy (~0.25–0.5) at clinical EC50, robust multi-target engagement
  (29/30 > 10%)."** The occupancy_matrix / Gate-C.1 occupancy magnitudes should be regenerated under M0 (or
  reported as an M0/M2 bracket) before any re-promotion.
- **No network impact.** The V1 sat_pa ladder is hand-set and decoupled from Vina (`phase_g_state_validator.py:24`),
  so this correction does NOT flip any quorum, percentile, or EC50 in the network layer. It is a Wave-P
  molecular-layer honesty correction, gated behind the P7 PROVENANCE disclosure for the ladder itself.

## Caveat

M0 designated decision-binding from the standard-state cancellation argument before the recompute. A residual
subtlety — whether a membrane-embedded pocket sees a *local* concentration different from bulk aqueous even
after the Kd-frame correction — is a second-order modeling question that does not change the headline: the
250× full-Kp magnitude is indefensible as written, and the engagement count is robust either way.
