# V7 Pre-Registration — Minimum Sufficient Subset + Random Ensemble + V5 M3/M4

**Status: PRE-REGISTRATION. No simulations have been run. This document is committed before any V7 code runs and locks the protocol, pass criteria, and predictions for audit.**

Date: 2026-05-05
Author: Rohit Ravi (with Claude Opus 4.7 implementation)
Predecessor: V6 M0 preregistration (commit f140764)

---

## V7 scope and exclusions

**V7 is V6 + four pre-registered controls:**
- Sub-Q2 — minimum sufficient mechanism subset (highest scientific value; new finding)
- Sub-Q1 — random-ensemble controls combined with V5 M1 (null perturbation)
- V5 M3 — sensitivity analysis on perturbation-table parameters
- V5 M4 — calibration cross-validation (anchor swap)

**Explicit exclusions from V7 scope:**
- Sub-Q3 (shared-attractor analysis) — deferred to Paper 2 contingent on richer LIF substrate
- Propofol-vs-sleep peptide-comparison work from parallel chat — deferred to Paper 2
- Cortical-state dynamics, sleep-anesthesia overlap, modulation-layer mechanisms beyond the perturbation table
- Extension to compounds beyond V6 panel (halothane, isoflurane, sevoflurane, desflurane, etomidate, ketamine, propofol, plus Eger cis-DCE / trans-DCE / hexafluoroethane)
- Recalibration of α in response to V7 results — α is FROZEN at V6 values throughout

## Frozen parameters (inherited from V6 M0)

α values per organism, locked for all V7 sub-questions:
```
worm V3:  α = 0.13
fly  V4:  α = 0.060
mouse V6: α = 0.10
```

Per-organism perturbation tables, mutant baseline tables, NT-identity heuristics, and connectome / random-graph substrates are all locked at their V6 M0 hash-stamped state. Any subsequent change to a hashed file invalidates V7 results downstream.

---

## Sub-Q2 — Minimum sufficient mechanism subset

### Protocol

For each organism × each non-empty subset of mechanism classes:
1. Apply only the subset's mechanism classes (zero out others by setting their `target_EC50_uM` to None / DEFERRED in the per-subset profile)
2. Run halothane dose-response at frozen α with 8 doses × 5 seeds (V3-protocol)
3. Fit Hill curve, extract predicted halothane EC50

**Subset enumeration:**
- Worm + fly: 7 mechanism classes (`gaba_potentiation`, `k2p_potentiation`, `complex_i_block`, `snare_cooperativity`, `nca_block`, `nachr_antagonism`, `glucl_potentiation`) → 2^7 − 1 = 127 non-empty subsets
- Mouse: 6 mechanism classes (drop `glucl_potentiation`, no mammalian ortholog) → 2^6 − 1 = 63 non-empty subsets

**Total subsets: 127 + 127 + 63 = 317 subset-organism cells × Stage 1.**

### Stage gating

| stage | test | sims per passing subset | gates carried forward |
|---|---|---|---|
| **Stage 1** | halothane Gate 1 dose-response | 40 (8 doses × 5 seeds) | subsets with predicted halothane EC50 within 2× of published |
| **Stage 2** | isoflurane Gate 2 held-out on Stage 1 passers | 40 | subsets ALSO passing isoflurane within 2× of 290 µM |
| **Stage 3** | Eger Gate 4 (cis-DCE / trans-DCE / hexafluoroethane) on Stage 2 passers | 105 (3 × 7 × 5) | subsets where cis-DCE max qf ≥ 0.5 AND non-immobilizers max qf < 0.5 |

### Pass criteria (frozen)

- Stage 1: `max(predicted_EC50 / published, published / predicted_EC50) ≤ 2.0`
  - worm published 340 µM, fly 340 µM, mouse 350 µM
- Stage 2: same fold-error rule on isoflurane (published 290 µM all organisms)
- Stage 3: `cis_DCE_max_qf ≥ 0.5` AND `trans_DCE_max_qf < 0.5` AND `hexafluoroethane_max_qf < 0.5`

### Pre-registered predictions (deviation thresholds explicit)

**P1.** No 1-class subset passes Stage 1 at frozen α for any organism.
- Threshold for deviation: ANY 1-class subset achieving Stage 1 fold-error ≤ 2× is a deviation.
- Rationale: V6 calibrated full-table α at fold-error 1.06–1.18×; removing 6/7 mechanism classes should make fold-error >> 2×.

**P2.** At least one passing 2-class subset exists per organism at Stage 1.
- Threshold: zero passing 2-class subsets is a deviation.
- Rationale: pairs containing two large-magnitude classes (e.g., GABA-A + Complex I) should sum to enough effective perturbation magnitude.

**P3.** `snare_cooperativity` OR `complex_i_block` appears in ≥ 75% of Stage 1-passing subsets across all organisms.
- Threshold: < 50% appearance is a deviation.
- Rationale: these two classes contribute the largest absolute perturbation magnitude (max_effect 0.5 / 0.7 with high engagement in the table); they're likely necessary.

**P4.** `glucl_potentiation` appears only in worm/fly passing subsets (mechanical for mouse — class doesn't exist there).
- Threshold: not testable for mouse (mechanical exclusion).

**P5.** The smallest passing subset size at Stage 1 is 2 or 3 classes.
- Threshold: smallest passing subset is 1 (would be huge surprise, deviation) or > 4 (would mean redundancy is weaker than expected, also a deviation).
- Rationale: V6 full-table works; V5 M2 showed substrate-agnosticism; minimum sufficient subset is empirically the next natural question.

### Output artifacts (committed after each stage)

- `artifacts/v7_subset_search/v7_stage1_halothane.csv` — 317 rows (subset, organism, qf-vs-dose curves, predicted EC50, fold-error, pass/fail)
- `artifacts/v7_subset_search/v7_stage2_isoflurane.csv` — Stage 1-passers only
- `artifacts/v7_subset_search/v7_stage3_eger.csv` — Stage 2-passers only
- `artifacts/v7_subset_search/v7_subset_verdict.json` — final pass-set per organism + redundancy-structure summary

### Redundancy-structure analysis

After Stage 3, identify:
- **Necessary classes**: classes appearing in 100% of all-stages-passing subsets. These are non-redundant.
- **Redundant pairs**: cases where (A,B) passes AND (A,C) passes but B≠C — A is necessary, B and C are interchangeable.
- **Sufficient classes**: classes that appear in at least one all-stages-passing subset.

This is the "different drugs, different paths, same destination" claim made testable.

---

## Sub-Q1 — Random ensemble controls (combined with V5 M1)

### Protocol

For each organism × each match definition × 50 random ensembles:
1. Generate a random anesthetic perturbation profile matching the conserved ensemble's distribution properties at the level specified by the match definition
2. Run halothane Gate 1 dose-response at frozen α (8 doses × 3 seeds for sweep efficiency)
3. Fit Hill curve, extract predicted EC50 and fold-error vs published
4. Report distribution of fold-errors across the 50 random ensembles
5. Compute conserved ensemble's percentile rank within the random distribution

### Three match definitions (stringency ladder)

**Match #1 — Count only.** For each organism, generate a random "perturbation profile" with:
- Same number of active mechanism classes as the conserved ensemble for halothane (`n_active`; varies per organism: 8 for worm, 8 for fly, 7 for mouse)
- For each randomly-chosen class: random EC50 ∈ Uniform(50, 1000) µM, random max_effect ∈ Uniform(0.3, 3.0), Hill_n = 1.0
- Random class identity is sampled from the available mechanism classes for that organism (no constraint on which classes are picked)

**Match #2 — Count + total perturbation magnitude.** Same as Match #1, plus constrain the random ensemble such that:
- `total_aggregate_pA_at_saturation = Σ over classes (max_effect_pA × class_engagement_at_clinical_EC50)`
- equals the conserved ensemble's value within ±5%
- Operationalization: use the rejection-sampling approach — generate Match #1 candidates, accept only those whose aggregate pA matches within ±5%

**Match #3 — Count + magnitude + cell-type spread.** Worm + fly ONLY (mouse generic graph doesn't support cell-type spread). Same as Match #2, plus:
- For worm: match the spread across CeNGEN-tagged neuron classes that get hit by the perturbation. Specifically, conserved ensemble's GABA-A targets UNC-49-expressing neurons (~75 neurons); K2P targets TWK-*-expressing (~60 neurons); etc. Random ensembles must match the **per-class neuron coverage spread** (i.e., the histogram over neuron classes of "how many neurons get hit by each class") within Wasserstein distance ≤ some threshold.
- For fly: match Winding 2023 cell-type spread similarly using `fly_nt_identity_heuristic.csv` cell-type tags.
- For mouse: NOT TESTABLE (Random graph has only E:I labels). Sub-Q1 caps at Match #2 for mouse.

### Pre-registered predictions

**P6.** Match #1 (count only): conserved ensemble fold-error percentile rank ≤ 50%.
- Translation: the conserved ensemble's halothane-EC50 prediction precision is no better than median random count-matched ensemble.
- Threshold for deviation: percentile rank ≤ 10% would falsify (conserved ensemble is too special at this match level — would surprise me).
- Rationale: V5+ Meyer-Overton showed total magnitude is most of what α calibrates against; randomizing class identity but keeping count should still give comparable fold-error.

**P7.** Match #2 (count + total magnitude): conserved ensemble percentile rank ≤ 30%.
- Translation: even controlling for total magnitude, conserved ensemble is in the top 30% of random ensembles.
- Threshold for deviation: percentile rank ≤ 5% falsifies (would mean total magnitude is the entire story).
- Rationale: classes have different downstream effects (e.g., SNARE scales W_chem globally vs Complex I shifts I_ext globally); class identity matters even at fixed total pA.

**P8.** Match #3 (count + magnitude + cell-type spread, worm + fly only): conserved ensemble percentile rank ≤ 15%.
- Translation: even matched on cell-type targeting distribution, the conserved ensemble is in the top 15% of random ensembles.
- Threshold for deviation: percentile rank > 30% would mean the conserved ensemble is NOT special at this match level → the V6 architecture is fully described by "any reasonable distributed perturbation matches the gates."
- Rationale: this is the strongest test. If conserved ensemble passes here, the conserved targets are doing real specific work beyond aggregate properties.

### Output artifacts

- `artifacts/v7_random_ensemble/v7_match1_random_50.csv` — 50 ensembles × 3 organisms = 150 rows
- `artifacts/v7_random_ensemble/v7_match2_random_50.csv` — same
- `artifacts/v7_random_ensemble/v7_match3_random_50.csv` — worm + fly only = 100 rows
- `artifacts/v7_random_ensemble/v7_random_ensemble_verdict.json` — percentile ranks of conserved ensemble per match level per organism

---

## V5 M3 — Sensitivity analysis

### Protocol

**OAT (one-at-a-time):** For each parameter in the halothane row of each organism's perturbation table:
- Perturb `target_EC50_uM` by ±50% (×0.5, ×1.0 baseline, ×1.5)
- Perturb `max_effect_factor` by ±50% (preserving sign convention: blocking classes stay <1; potentiating classes stay >1)
- Perturb `hill_n` by ±0.5 (around 1.0 baseline; range 0.5 to 1.5)
- For each perturbed value: run halothane Gate 1 at frozen α (8 doses × 3 seeds)
- Compute sensitivity index: `S_param = (ΔEC50 / EC50_baseline) / (Δparam / param_baseline)`

**LHS (Latin Hypercube Sampling):** 100 joint samples in the parameter space (each sample perturbs ALL parameters simultaneously within ±50%):
- Each sample: jointly resample EC50, max_effect, hill_n for all active mechanism classes
- Run halothane Gate 1 at frozen α
- Build distribution of predicted halothane EC50
- Report 95% CI on prediction under joint parameter uncertainty

### Pre-registered predictions

**M3a.** No single ±50% OAT perturbation causes Gate 1 to fail (fold-error > 2×) on the calibration anchor.
- Threshold for deviation: any single OAT perturbation causing fold-error > 3× falsifies (architecture is fragile to single-parameter uncertainty).

**M3b.** At least one parameter has |sensitivity index| > 0.3 (i.e., genuinely load-bearing).
- Threshold for deviation: max sensitivity index < 0.1 across all parameters would mean the architecture is largely insensitive to its inputs (over-determined; result is a fitting artifact).

**M3c.** 95% LHS CI on halothane EC50 prediction under joint ±50% perturbation: range ⊂ [200, 600] µM.
- Threshold for deviation: 95% CI extending beyond [100, 1000] µM falsifies (architecture is brittle under realistic literature uncertainty).

### Output artifacts

- `artifacts/v7_sensitivity/v7_sensitivity_oat.csv` — per-parameter sensitivity indices
- `artifacts/v7_sensitivity/v7_sensitivity_lhs.csv` — 100 joint samples × 3 organisms
- `artifacts/v7_sensitivity/v7_sensitivity_verdict.json` — sensitivity report + 95% CIs

---

## V5 M4 — Calibration cross-validation (anchor swap)

### Protocol

For each organism:
1. Hold out the original halothane MAC anchor; re-calibrate α on isoflurane MAC instead (predicted iso EC50 ≈ 290 µM at re-tuned α)
2. Predict halothane EC50 with the new (iso-calibrated) α
3. Compare new α to original α and compare predicted halothane EC50 to published 340/350 µM

### Pre-registered predictions

**M4a.** New α (calibrated on iso) is within 30% of original α (calibrated on halothane).
- Threshold for deviation: new α differs by > 50% means calibration is highly anchor-specific.

**M4b.** Predicted halothane EC50 using iso-anchored α is within 2× of published (340/350 µM).
- Threshold for deviation: fold-error > 2.5× means the calibration doesn't generalize anchor-to-anchor cleanly.

### Output artifacts

- `artifacts/v7_cross_cal/v7_cross_cal_verdict.json` — per-organism: original α, iso-anchored α, predicted halothane EC50 with iso-anchored α, fold-error vs published

---

## Compute schedule (staged across overnights)

| stage | content | sims | wall (8-12 cores) | overnight |
|---|---|---|---|---|
| **0** | pre-registration (this document) | 0 | now | — |
| **1** | Sub-Q2 Stage 1: halothane Gate 1 across 317 subset-organism cells | ~12,700 | ~14-18 hours | overnight 1 |
| **2** | Sub-Q2 Stage 2 (iso held-out) on passers + Stage 3 (Eger) on Stage 2 passers | ~6,000-12,000 | ~6-12 hours | overnight 2 |
| **3** | Sub-Q1 random ensembles: 3 match levels × 50 ensembles × 3 organisms | ~8,000-12,000 | ~10-14 hours | overnight 3 |
| **4** | V5 M3 sensitivity: OAT (~150 sims) + LHS (~2,400 sims) | ~2,500 | ~3-4 hours | with overnight 3 or solo |
| **5** | V5 M4 cross-cal: 2 alpha sweeps × 3 organisms | ~250 | ~30 min | inline |

Total compute: 4-5 overnights spread across ~3-4 weeks of work.

---

## Stop conditions and deviation reporting

**Stop conditions (V7 work pauses if):**
1. Stage 1 of Sub-Q2 produces results that contradict P1-P5 in ways that imply the architecture has fundamentally different behavior than V6 documented. Specifically: if 1-class subsets PASS Stage 1, that's a major deviation requiring re-analysis of the V6 result before continuing.
2. V5 M3 LHS reveals 95% CI extending beyond [100, 1000] µM — architecture brittleness exceeds publishable threshold.
3. V5 M4 reveals new α differs by > 100% from original α — calibration is anchor-specific in a way that undermines the cross-organism claim.

**Deviation reporting structure:**
For each prediction P1–M4b that is falsified by results:
- Original prediction (verbatim from this document)
- Observed result with effect size
- Whether the deviation is within or outside pre-registered threshold
- Implication for V6/V7 published claims (do they need narrowing?)

The page IS updated post-V7 to reflect deviations honestly, same discipline as V5 M2 (connectome permutation) and V5+ (Meyer-Overton).

---

## Honest scope labels for V7 commits

When V7 results land, commits will use these explicit labels:
- `feat(anesthesia/v7-shipped): ...` — gates passing pre-registered criteria
- `feat(anesthesia/v7-deviation): ...` — gates passing but with pre-registered deviation noted
- `feat(anesthesia/v7-fail): ...` — gates failing; page narrowed accordingly
- `docs(anesthesia/v7-pending): ...` — scaffold or in-progress not yet validated

No commit will be marked "passes" without naming which criterion at what threshold.

---

## Authorship / origin

This pre-registration is committed before any V7 sims run. Hash will be computed and stored in `artifacts/v7_controls/V7_preregistration_hash.txt`. Any subsequent edit to this document invalidates downstream V7 results and creates a `v7.1` superseding registration.
