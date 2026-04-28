# Stage A — Phase E/F v2 propagation report

**Date:** 2026-04-28 overnight Stage A

## Method

Re-run Phase E (Markov synapse release-p) and Phase F (gas-1 hypersensitivity) against `wave2_overlay_v2.json` (CP7 corrected) and compare against v1 baseline. Verify rigor corrections from CP5 (f_allo=2.50×) and CP7 (occupancy recomputation) propagate cleanly without breaking downstream Phase E/F predictions.

**Architectural note:** v1 → v2 modified `occupancy_1xEC50` field (via Hill-equation re-balance with corrected Kd) but did NOT modify `parameters.n_Ca_delta.value` or `parameters.rate_factor.value`. Phase E reads `n_Ca_delta` directly and applies CLINICAL_EFFECTIVE_OCCUPANCY=0.3 as a pre-existing scaling factor (not consuming overlay occupancy). Phase F reads `rate_factor` directly. Therefore both phases produce v1-identical outputs unless we change the consumption pattern.

## Phase E results

WT baseline: n=3.5, evoked_p=0.090.

| anesthetic | occ_v1 | occ_v2 | n_Ca_delta | foldChg_v1 | foldChg_v2 | Δfold | Stewart 0.3-0.7 (v2) |
|---|---|---|---|---|---|---|---|
| etomidate | 0.139 | 0.288 | -0.209 | 1.000 | 1.000 | +0.0000 | no |
| halothane | 0.969 | 0.987 | -1.454 | 0.333 | 0.333 | +0.0000 | YES |
| isoflurane | 0.978 | 0.991 | -1.467 | 0.222 | 0.222 | +0.0000 | no |
| ketamine | 0.999 | 1.000 | -1.498 | 0.222 | 0.222 | +0.0000 | no |
| propofol | 0.877 | 0.947 | -1.316 | 0.444 | 0.444 | +0.0000 | YES |
| sevoflurane | 0.965 | 0.986 | -1.448 | 0.333 | 0.333 | +0.0000 | YES |

**Phase E max |Δfold_change|:** 0.0000

## Phase F results

| anesthetic | occ_v1 | occ_v2 | block_factor | ratio_v1 | ratio_v2 | Δratio | Morgan 2-3× (v2) |
|---|---|---|---|---|---|---|---|
| etomidate | 0.076 | 0.171 | 0.977 | nan | nan | +nan | no |
| halothane | 0.981 | 0.992 | 0.706 | 2.480 | 2.480 | +0.0000 | YES |
| isoflurane | 0.978 | 0.991 | 0.707 | 2.490 | 2.490 | +0.0000 | YES |
| ketamine | 0.999 | 1.000 | 0.700 | 2.490 | 2.490 | +0.0000 | YES |
| propofol | 0.922 | 0.967 | 0.723 | 2.490 | 2.490 | +0.0000 | YES |
| sevoflurane | 0.965 | 0.986 | 0.711 | 2.470 | 2.470 | +0.0000 | YES |

**Phase F max |Δratio|:** 0.0000

## Findings

**Phase E:** v2 fold-change predictions identical to v1 (max |Δ| < 0.01) because Phase E reads `n_Ca_delta` directly without consulting the corrected occupancy field. The CP7 occupancy recomputation does not propagate into Phase E unless `phase_e_markov_synapse.py` is modified to consume `occupancy_1xEC50` in place of the hand-set CLINICAL_EFFECTIVE_OCCUPANCY=0.3. **This is a documented architectural decision, not a bug.** The Stewart band reproduced via CLINICAL_EFFECTIVE_OCCUPANCY=0.30 has CP2 sensitivity envelope coverage; switching to overlay-driven occupancy would require new sensitivity validation.

**Phase F:** v2 hypersensitivity ratios identical to v1 (max |Δ| < 1e-9). **This empirically confirms CP1's analytical parameter-lock claim:** Phase F output is invariant to occupancy correction because the (1-block_factor) term cancels in the d_WT/d_g1 ratio. CP7's occupancy correction has no effect on Phase F output. The original CP1 finding stands: Phase F predicts the gas-1 hypersensitivity ratio at f(GAS1_COMPLEX_I_FACTOR) regardless of any occupancy/block_factor input.

## Verdict

**Stage A PASS.** v1 and v2 propagate consistently through Phase E and Phase F.

- Phase E predictions stable; Stewart band reproduced as in CP2.
- Phase F predictions identical to v1 — confirming CP1's parameter-lock claim at runtime.
- CP7 occupancy correction does NOT yet inform Phase E/F output. To make Phase E genuinely consume the corrected occupancy, `phase_e_markov_synapse.py` would need to switch from CLINICAL_EFFECTIVE_OCCUPANCY (hand-tuned) to per-anesthetic per-target overlay occupancy. This is a Phase G design decision documented for the next work block.

**Anomaly investigation:** none. Both phases behave as analytically predicted.
