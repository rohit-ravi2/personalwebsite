# P8 — Corrected two-coordinate Match#2b null: result + claim reconciliation

**Date:** 2026-06-12 · **Gate:** G3 fly-survival = **PASS** (frozen rule) · **Bigger finding:** the Sub-Q1
three-organism trichotomy was largely a flawed-control artifact. Prereg: `audits/phase1/P8/prereg.json`
(frozen before the run). Verdict + artifacts: `artifacts/v7_match2b/`.

## What was wrong with the shipped Match#2

The shipped control (`v7_random_ensemble._draw_random_profile_match2` via `_aggregate_pa_at_dose`) matched
only an aggregate-pA scalar that, against the rank-2 operator (P18 PASS), is wrong two ways:
1. it **adds a phantom SNARE 50 pA current** into the aggregate — but the operator routes SNARE to a
   **synaptic-weight multiplier**, never to `I_ext` (`apply_anesthetic` lines 327-335);
2. it **never matches `snare_factor`**, the operator's actual second sufficient-statistic coordinate
   (conserved value ≈ 0.75 — SNARE *cuts* synaptic weight ~25%).

`v7_match2b` rejection-samples random ensembles matching **both** operator coordinates
(`total_pa`, `snare_factor`) of the conserved profile, verified to reproduce the real `apply_anesthetic`
to 7e-15 (G1 fidelity PASS), 50/50 jointly-matched draws per organism (G2 density PASS).

## Result

Conserved-ensemble percentile rank (fraction of jointly-matched randoms with **better/lower** halothane
fold-error; lower percentile = more special), n=50 per organism:

| organism | flawed Match#2 | **corrected Match#2b** | one-sided p (better than matched-random median) |
|---|---|---|---|
| worm | 0.0% | **28.0%** | 0.0013 (significant) |
| fly | 4.76% | **26.0%** | 0.0005 (significant) |
| mouse | 46.0% | **38.0%** | 0.0595 (marginal) |

**G3 (frozen, accept-either-way):** fly 26% ∈ (5%, 30%] → **PASS** — fly's class-identity signal survives the
airtight control. Accepted as the frozen verdict.

## Honest reading (the bigger finding)

The dramatic three-organism trichotomy — "worm anchor-overfit (0%, beats every random) / fly uniquely
cleanest (4.76%) / mouse magnitude-driven (46%)" — **was substantially an artifact of the control bug**, not
biology. Under the airtight two-coordinate null:

1. **Worm's "anchor-overfit 0%" is FALSE.** Worm is 28% (p=0.0013) — a *modest but significant* conserved
   advantage, statistically indistinguishable from fly. The 0% was the flawed control (phantom SNARE current +
   unmatched `snare_factor`) manufacturing artificial specificity, not α-overfitting.
2. **Fly is NOT uniquely special.** Its signal survives the binary gate but the effect collapsed ~5× (4.76% →
   26%), and worm now matches it. "Fly is the cleanest case for class-identity specificity" is not supported.
3. **What actually survives:** a *small but real* (worm + fly significant, ~26-28%, p<0.002; mouse marginal)
   conserved-profile EC50-precision advantage over magnitude+SNARE-matched randoms. The conserved multi-class
   target identity carries a little information beyond the two-coordinate magnitude — in **worm and fly alike**,
   not fly alone — but far less than the original percentiles implied.

## Required claim rewrites (v7_final_summary.md / mdx)

- **§3.1-§3.5, §7.4, §8.5** — replace the worm-0%-anchor-overfit / fly-4.76%-cleanest / mouse-46% narrative
  with the corrected Match#2b percentiles (worm 28%, fly 26%, mouse 38%) and the reading above. Delete the
  "fly is the cleanest / only case for class-identity specificity" sentence; delete "worm anchor-overfit"
  framing (it was a control artifact). State the airtight result: worm≈fly modest-significant, mouse marginal.
- **§8.1 (mouse magnitude-driven)** — unchanged in direction (mouse weakest, now 38% / p=0.06), reinforced by
  the airtight control; defer to P3 mean-field theorem for the a-priori derivation.
- Mark the original `v7_match2_*` artifacts SUPERSEDED-BY-P8 (not deleted) with a pointer to this doc.

## Downstream

- **P16 (held-out structure→activity exam)** is GATED on this fly verdict. Fly PASS (signal survives) ⇒ P16 is
  still motivated, but with a *weaker* prior than the 4.76% headline implied; and worm now also carries signal,
  so the fly-shuffle's expected effect is smaller. Proceed to P16 with calibrated expectations.
- **P3 (mean-field theorem)** should now derive the corrected percentiles' compression (all three toward the
  matched-random median) as the a-priori consequence of the rank-2 + magnitude-matched null, rather than
  explaining only mouse.
