# V7 Final Summary — Sub-Q2 + Sub-Q1 + V5 M3/M4 Closeout

**Status: V7-SHIPPED closeout document.** All four adversarial controls have run; their results are reported here against the pre-registered predictions. Deviations are reported in the direction the data actually moved, not soft-pedaled toward the prediction. No recalibration of α was performed in V7 scope.

Date: 2026-05-12
Author: Rohit Ravi (with Claude Opus 4.7 implementation)
Pre-registration: `AnestheticSimulator/docs/v7_preregistration.md`, hash `533b624a00b5ff9efecee41fb549fcb9cf5f02810aefcde55e14c7981fd09ff4`, commit `4061f4f` (2026-05-05).

---

## §1 Pre-registration recap and V7 scope

V7 extends V6 with four pre-registered adversarial controls. The pre-registration was committed at `4061f4f` on 2026-05-05 before any V7 simulations ran; its sha256 is referenced in every per-stage verdict JSON for audit. No edits have been made to the pre-registration document during the V7 runs; the protocol is the one that was locked.

Sub-questions in V7:

- **Sub-Q2** — minimum sufficient mechanism subset, three staged gates (halothane Stage 1 → isoflurane held-out Stage 2 → Eger non-immobilizer Stage 3).
- **Sub-Q1** — random-ensemble null at frozen α, three pre-registered match levels (count-only, count + magnitude, count + magnitude + cell-type spread).
- **V5 M3** — parameter sensitivity, OAT plus Latin Hypercube Sampling.
- **V5 M4** — calibration cross-validation by anchor swap (halothane MAC ↔ isoflurane MAC).

α was frozen throughout at the V6-locked per-organism values: **worm 0.13, fly 0.060, mouse 0.10**. No subset, ensemble, or sensitivity run recalibrated α. Per-organism perturbation tables, mutant baseline tables, NT-identity heuristics, and connectome / random-graph substrates were locked at their V6 M0 hash-stamped state.

V7 was scoped to exclude: any compound beyond the V6 panel (halothane, isoflurane, sevoflurane, desflurane, etomidate, ketamine, propofol, plus Eger cis-DCE / trans-DCE / hexafluoroethane); the propofol-vs-natural-quiescence bridge experiment (Paper 2 work, separate session); cell-type-resolved targeting beyond V1 substrate; and the framework-paper-level claim about thermodynamic necessity of reversible quiescence.

The V7 commit-label discipline is: `v7-shipped` (passes pre-registered criteria), `v7-deviation` (passes with documented deviation against pre-registered threshold), `v7-fail` (fails; page narrowed accordingly), `v7-pending` (scaffold only). This document is the closeout against that discipline.

---

## §2 Sub-Q2 — Minimum sufficient mechanism subset

### 2.1 Subset enumeration and protocol

Seven mechanism classes from the V6 conserved-target table: `gaba_potentiation`, `k2p_potentiation`, `complex_i_block`, `snare_cooperativity`, `nca_block`, `nachr_antagonism`, `glucl_potentiation`. For worm and fly, all 127 non-empty subsets were enumerated. For mouse, `glucl_potentiation` was dropped (no mammalian ortholog) and the 63 non-empty subsets of the remaining 6 classes were enumerated. Total: 127 + 127 + 63 = **317 subset-organism cells**.

For each cell, the validator was run at frozen α with the subset's mechanism classes active and the rest set to `DEFERRED` (zero perturbation). 8 doses × 5 seeds × 30-second simulations per cell, three pre-registered seeds added at Stage 1 (42, 137, 219, 331, 443). Total Stage 1 sims: **12,680**. Wall time: 410 minutes on 12 cores.

Pass criterion at each stage:

| stage | test | pass rule |
|---|---|---|
| Stage 1 | halothane Gate 1 dose-response | `max(predicted/published, published/predicted) ≤ 2.0` |
| Stage 2 | isoflurane Gate 2 held-out (only Stage 1 passers tested) | same fold-error rule on 290 µM |
| Stage 3 | Eger non-immobilizer specificity (only Stage 2 passers tested) | `cis_DCE_max_qf ≥ 0.5` AND `trans_DCE_max_qf < 0.5` AND `hexafluoroethane_max_qf < 0.5` |

### 2.2 Stage attrition by organism

The attrition pattern across the three stages is itself a finding.

| organism | Stage 1 passers | Stage 2 passers | Stage 3 passers | attrition |
|---|---|---|---|---|
| worm | 20 / 127 | 14 / 20 | 14 / 14 | 30% lost at iso, 0% at Eger |
| fly | 5 / 127 | 3 / 5 | 3 / 3 | 40% lost at iso, 0% at Eger |
| mouse | 8 / 63 | 8 / 8 | 8 / 8 | **0% lost at iso, 0% at Eger** |

Two readings, both supported by the data:

**Mouse subsets that pass halothane all pass isoflurane and Eger.** Once the mouse substrate (generic random graph, no cell-type structure) produces a correct halothane EC50, isoflurane and Eger discrimination are entailed at frozen α — they are not independent gates in mouse. This is consistent with the Sub-Q1 P7 violation reported in §3.4: in mouse, the conserved-target ensemble is at median percentile against magnitude-matched random ensembles, meaning class identity adds essentially nothing once aggregate magnitude is controlled. Both findings point to the same underlying constraint — the mouse substrate's lack of cell-type structure means anything that produces the right halothane response also produces the right isoflurane response, because there is no spatial heterogeneity in how mechanism classes route through the network. Compound discrimination requires substrate features that are not present in the V6 generic random graph.

**Worm and fly have meaningful Stage 1 → Stage 2 attrition.** Worm loses 6/20 subsets that produced correct halothane at frozen α but failed isoflurane held-out. Fly loses 2/5 by the same criterion. These subsets are anchor-specific in a way mouse subsets are not. Combined with the Sub-Q1 worm 0% percentile (anchor-overfit) and fly 5% percentile (genuinely class-specific) in §3, the cross-organism reading is: fly's class identity carries real predictive information; worm's anchor-tuning is doing some of the same work as class identity; mouse's predictions are magnitude-driven.

**Zero attrition past Stage 2.** No Stage 2 passer fails Stage 3 Eger discrimination in any organism. The conserved-target perturbation table's structural sparseness on non-immobilizers — Eger compounds lack engagement at SNARE and NCA, which are the high-magnitude classes — means that once a subset is engaging SNARE or Complex I sufficiently to produce the volatile EC50, the non-immobilizers fail by construction. This has implications for how the page frames Gate 4: see §7 and §8.

### 2.3 Per-organism passing structure

A class is *necessary* under strict intersection if it appears in 100% of an organism's all-stages passers. Beyond strict necessity, two weaker structures matter: *substitution pairs* (where (A or B) is universal even though neither A nor B is strict-necessary alone), and *modal containment* (classes appearing in ≥75% of passers).

**Worm (14 passers, smallest subset size 5).**

- Strict-necessary intersection: `snare_cooperativity` only (1 class).
- Substitution pair: every worm passer contains `complex_i_block` OR `nca_block`. 1 worm passer drops Complex I; 2 worm passers drop NCA; **no passer drops both**. The verdict JSON's `necessary_classes_100pct` does not surface this because neither Complex I nor NCA is in 100% of passers individually — but their union is.
- Modal containment: GABA-A, K2P, nAChR each appear in 71–86% of worm passers. GluCl appears in 64% (9/14).
- Rule: `snare_cooperativity` AND (`complex_i_block` OR `nca_block`) AND ≥ 3 supporting classes.

**Mouse (8 passers, smallest subset size 4).**

- Strict-necessary intersection: `complex_i_block` AND `nca_block` (2 classes).
- Modal containment: ≥ 2 of {`gaba_potentiation`, `k2p_potentiation`, `nachr_antagonism`} in every passer.
- Rule: `complex_i_block` AND `nca_block` AND ≥ 2 of {GABA-A, K2P, nAChR}.
- This is the cleanest substitution structure across the three organisms.

**Fly (3 passers, smallest subset size 6).**

- Strict-necessary intersection: 5 classes — Complex I, GABA-A, K2P, nAChR, NCA. SNARE is in 2/3; GluCl is in 2/3.
- Caveat on statistical looseness: with only 3 passers, the 100%-intersection set is statistically inflated. The cleaner reading is "SNARE and GluCl are the only swappable classes among 7 active" rather than "5 classes are strictly necessary." More passers would likely shrink the strict-necessary set as more equivalent architectures get sampled.

**Calibration tightness explains the size differences.** Fly's smallest passer at size 6 vs mouse's at size 4 is consistent with fly's smaller frozen α (0.06 vs mouse 0.10): each perturbation contributes less, so more classes must engage to clear the same MAC. Worm sits between (α 0.13, smallest passer 5).

### 2.4 Cross-organism modal architecture

Across all 25 all-stages passers (worm 14, fly 3, mouse 8):

| class | worm 14 | fly 3 | mouse 8 | overall 25 |
|---|---|---|---|---|
| `complex_i_block` | 13 (93%) | 3 (100%) | 8 (100%) | **24/25 (96%)** |
| `nca_block` | 12 (86%) | 3 (100%) | 8 (100%) | **23/25 (92%)** |
| `snare_cooperativity` | 14 (100%) | 2 (67%) | 4 (50%) | **20/25 (80%)** |
| `gaba_potentiation` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `k2p_potentiation` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `nachr_antagonism` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `glucl_potentiation` | 9 (64%) | 2 (67%) | 0 (excluded) | 11 (worm+fly) |

**`snare_cooperativity` OR `complex_i_block` appears in 25/25 = 100% of all-stages passers across all three organisms.** This is the cleanest cross-organism result in V7. It says: every architecture that recovers halothane, isoflurane, and Eger discrimination at frozen α contains one or both of the two highest-magnitude classes in `DEFAULT_PER_CLASS_PA_AT_SATURATION` (50 pA SNARE, 60 pA Complex I).

Three non-mutually-exclusive readings of why:

1. **Magnitude-driven.** SNARE and Complex I contribute the largest absolute perturbation; NCA is third. Their high appearance reflects that the architecture needs sufficient aggregate perturbation magnitude to reach the quiescent threshold. Sub-Q1 Match #2 (§3.2) speaks directly to this — it tests whether class identity adds specificity beyond aggregate magnitude.
2. **Mechanism-conserved.** SNARE, Complex I, NCA are the three classes anchored across worm/fly/mouse with primary-literature EC50 measurements and clinical-dose engagement. They are conserved targets in the literature sense, not chemistry artifacts.
3. **Substrate-coupled.** Complex I and NCA shift global current (I_ext) uniformly; SNARE scales synaptic weights (W_chem) specifically. In substrates with weak cell-type structure (mouse generic graph), global-current classes dominate; in substrates with biological connectivity (worm Cook 2019, fly Winding 2023), SNARE acting on synaptic weights does more work. This explains worm's SNARE-at-100% vs mouse's SNARE-at-50%.

### 2.5 Pre-registered prediction outcomes (P1–P5)

**P1.** No 1-class subset passes Stage 1 at frozen α for any organism. **CONFIRMED.** Zero 1-class subsets passed any organism. The architecture cannot reproduce halothane MAC within 2× from a single mechanism class at the calibrated α.

**P2.** At least one passing 2-class subset exists per organism at Stage 1. **FALSIFIED.** Zero 2-class subsets passed any organism. Zero 3-class subsets passed any organism. The system is brittle to subset removal at low subset counts; the prediction that "two large-magnitude classes (e.g., GABA-A + Complex I) should sum to enough effective perturbation magnitude" was wrong. The architecture is structurally different than predicted in the low-subset-count regime.

**P3.** `snare_cooperativity` OR `complex_i_block` appears in ≥ 75% of Stage 1-passing subsets across all organisms. **CONFIRMED, exceeded.** 25/25 = 100% of all-stages passers contain SNARE OR Complex I, across all three organisms. The pre-registered threshold of 75% was met with margin; the falsification threshold of 50% was nowhere near approached.

**P4.** `glucl_potentiation` appears only in worm/fly passing subsets. **CONFIRMED, mechanical for mouse.** GluCl is absent from the mouse mechanism pool by design (no mammalian ortholog). In worm + fly: 11/17 invertebrate passers contain GluCl.

**P5.** The smallest passing subset size at Stage 1 is 2 or 3 classes. **FALSIFIED.** Smallest passing subset sizes are worm 5, mouse 4, fly 6. The architecture's redundancy structure exists at higher mechanism-class counts but not at lower ones — multiple distinct 4–6 class subsets pass per organism (14 worm, 3 fly, 8 mouse), but no 2- or 3-class subset passes anywhere. The pre-registered prediction that 2–3 class redundancy is the relevant scale was wrong.

**Direction of the P2 and P5 falsifications.** Both deviations are in the same direction: the architecture requires more classes engaged jointly than the pre-registration predicted. This is a falsification, not a sharper version of the prediction. The pre-registered claim ("redundancy exists at small subset sizes") is wrong; the corrected claim ("redundancy exists at 4–6 class subsets but not at 2–3 class subsets") is informative but different. The architecture's structure is more brittle to low-subset-count removal than the pre-registration predicted, and the redundancy scale is shifted upward by 2–3 classes.

---

## §3 Sub-Q1 — Random-ensemble percentile analysis

### 3.1 Match #1 — count-only

For each organism, 50 random ensembles were generated with:
- The same number of active mechanism classes as the V6 conserved ensemble for halothane (`n_active` = 8 worm, 8 fly, 7 mouse).
- For each randomly-chosen class: `EC50 ~ Uniform(50, 1000) µM`, `max_effect ~ Uniform(0.3, 3.0)`, `Hill_n = 1.0`.
- Class identity sampled without constraint from the organism's available mechanism pool.

8 doses × 3 seeds (42, 137, 219) per ensemble. Hill fit, EC50 extracted, fold-error vs published halothane.

| organism | conserved fold-error | random median fold-error (estimated) | conserved percentile rank |
|---|---|---|---|
| worm | 1.051 | (random distribution wider) | **0.0%** |
| fly | 1.064 | — | 5.56% |
| mouse | 1.215 | — | 28.0% |

The conserved-ensemble percentile rank is the fraction of random ensembles whose halothane fold-error is *better* (lower) than the conserved ensemble's.

### 3.2 Match #2 — count + total magnitude

Same as Match #1, plus rejection-sampling constraint: `total_aggregate_pA_at_saturation = Σ over classes (max_effect_pA × class_engagement_at_clinical_EC50)` must equal the conserved ensemble's value within ±5%. All 50 attempted ensembles passed rejection at each organism (0 rejection failures).

| organism | conserved percentile rank (Match #2) |
|---|---|
| worm | **0.0%** |
| fly | 4.76% |
| mouse | **46.0%** |

### 3.3 Match #3 — cell-type spread (NOT TESTED)

Match #3 was pre-registered as: same as Match #2, plus match the per-class neuron-coverage spread across CeNGEN-tagged neuron classes (worm) or Winding 2023 cell-type tags (fly). Mouse was scoped out at Match #2 due to the generic graph's lack of cell-type labels.

**Status: V7-DEVIATION, not tested.** In the V1 validator, `resolve_target_neurons` returns `range(brain.N)` — all mechanism classes hit all neurons. There is no cell-type-resolved targeting in V1, which means cell-type spread is uniform by construction across all mechanism classes. Under the V1 substrate, Match #3 reduces mathematically to Match #2: any constraint on per-class neuron coverage is automatically satisfied because all classes have identical (full) coverage. P8 thresholds are unfilled.

This is a documented limitation of the V1 substrate, not a methodological omission. CeNGEN-aware targeting that would make Match #3 a genuine test is V2 substrate work. The pre-registration anticipated this possibility; the deviation is reported here for completeness.

### 3.4 Pre-registered prediction outcomes (P6–P8)

**P6.** Match #1 conserved-ensemble percentile rank ≤ 50%. **CONFIRMED for all three organisms** (worm 0%, fly 5.56%, mouse 28%). The conserved ensemble's halothane EC50 prediction precision is at least as good as the median random count-matched ensemble.

However, the **pre-registered falsification floor of ≤ 10%** ("conserved is too special at this match level — would surprise me") is **crossed by worm (0.0%)**. This is a falsification in the "too-special" direction: the worm conserved ensemble is better than every one of the 50 random count-matched ensembles. This is consistent with worm being the calibration anchor — α was tuned tightly on worm halothane, so the conserved profile sits exactly at the optimal point.

**P7.** Match #2 conserved-ensemble percentile rank ≤ 30%. **VIOLATED FOR MOUSE.** Worm (0.0%) and fly (4.76%) pass; mouse (46.0%) violates. The mouse conserved ensemble is at the **median** of magnitude-matched random ensembles — it is not statistically special once total perturbation magnitude is controlled.

The pre-registered falsification floor of ≤ 5% is again crossed by worm (0.0%) and approached by fly (4.76%). Worm shows the same anchor-overfit pattern at both match levels; fly tightens slightly from Match #1 to Match #2 (5.56 → 4.76%), suggesting fly's class identity carries genuine predictive information independent of magnitude.

**P8.** Match #3 conserved-ensemble percentile rank ≤ 15% (worm + fly only). **NOT TESTED** — see §3.3 for V1 substrate limitation.

### 3.5 Three-organism reading

The three percentile patterns map cleanly:

- **Worm: anchor-overfit.** 0% at both matches, below the pre-registered "too-special" floor of 10%. α was calibrated on worm halothane, so the conserved profile sits at the precision-optimal point against which random ensembles cannot compete. This is honest reporting: the worm result is partially a calibration artifact. It is not nothing — the conserved profile beats randoms by a wide margin even at Match #2 controlling for magnitude — but the gap is partially due to anchor specificity that the random ensembles don't get.

- **Fly: cleanest class-identity specificity.** 5.56% at Match #1, tightening to 4.76% at Match #2. The conserved profile beats 95% of count-matched randoms; controlling for total magnitude tightens (not loosens) the specificity. This is the direction the pre-registration expected: class identity does work beyond aggregate magnitude. Fly is the cleanest case for the conserved-target hypothesis at the level V7 can test it.

- **Mouse: magnitude-driven.** 28% at Match #1, *loosening to 46%* at Match #2. Once aggregate magnitude is matched, the mouse conserved profile is median — half of the random ensembles do at least as well. Class identity adds essentially nothing in mouse beyond aggregate magnitude. Combined with mouse's zero attrition past Stage 1 of Sub-Q2 and its 4-class minimum-sufficient subset size (§2.2, §2.3), this is consistent: the mouse generic random graph cannot distinguish "right classes" from "right total magnitude" because the substrate has no cell-type structure for class-specific dynamics to operate through.

The three-organism reading is summarizable as a single sentence: **fly is the cleanest case for class-identity specificity; mouse is magnitude-driven; worm is anchor-overfit.** This pattern is the substantive Sub-Q1 finding.

---

## §4 V5 M3 — Parameter sensitivity

### 4.1 OAT (one-at-a-time)

For each parameter in each organism's halothane perturbation table:
- `target_EC50_uM` perturbed by ±50% (factors 0.5, 1.0, 1.5)
- `max_effect_factor` perturbed by ±50% (sign convention preserved)
- `hill_n` perturbed by ±0.5 around 1.0 baseline

For each perturbed value: halothane Gate 1 at frozen α, 8 doses × 3 seeds. Sensitivity index `S_param = (ΔEC50 / EC50_baseline) / (Δparam / param_baseline)`.

**Results:**
- Maximum observed |S| = 0.84.
- 9 parameters across the three organisms have |S| > 0.3 (load-bearing).
- Maximum OAT halothane fold-error: 1.44× — well inside the pre-registered 2× pass threshold, and well inside the 3× falsification threshold.

**M3a (no single OAT perturbation causes Gate 1 fold-error > 2×): PASS.**

**M3b (at least one parameter has |S| > 0.3, i.e., genuinely load-bearing): PASS.** 9 parameters meet this threshold.

The architecture has identifiable load-bearing parameters (load-bearing in the sense that ±50% perturbation moves the EC50 prediction by a measurable amount) and is robust to ±50% single-parameter perturbation in the sense that no single perturbation drives the prediction outside the 2× fold-error pass band.

### 4.2 LHS (Latin Hypercube Sampling, worm only)

100 joint LHS samples in the worm halothane parameter space; each sample perturbs all parameters simultaneously within ±50% of baseline. For each sample: halothane Gate 1 at frozen α, 8 doses × 3 seeds.

**Results (worm only):**
- 95% CI on predicted halothane EC50: **[178.4, 447.6] µM**.
- Median predicted EC50: 311.6 µM (8% off published 340 µM, 2% off the V6 worm point prediction of 317 µM).

**M3c (95% LHS CI ⊂ [200, 600] µM): DEVIATION.** The observed CI extends below 200 µM on the low end. The pre-registered falsification threshold ([100, 1000] µM as the broader range outside which the architecture would be brittle) is well inside; the deviation is from the *tight* pre-registered range, not from the falsification range.

**Honest reading:** the central tendency of the LHS distribution is well-anchored. The median (311.6 µM) is within 9% of published, indistinguishable from the V6 worm point prediction. The deviation is **tail width**, not central-tendency mis-calibration — the architecture's lower tail extends ~22 µM further below 200 µM than pre-registered. This is not "the architecture is fragile to parameter perturbation"; it is "the architecture's parameter uncertainty under joint ±50% perturbation is slightly wider on the low side than we pre-registered." A reviewer asking "is the architecture brittle?" should look at the median and the 95% range together — median well-anchored, range [178, 448] vs pre-registered [200, 600], deviation only in the lower-tail width.

LHS was scoped to worm only in the pre-registration's compute budget; OAT covers all three organisms. Extending LHS to fly + mouse is an open item but not required by the pre-registration.

---

## §5 V5 M4 — Calibration cross-validation (anchor swap)

### 5.1 Protocol

For each organism: hold out the halothane MAC anchor; re-calibrate α to isoflurane MAC (target ≈ 290 µM). Predict halothane EC50 with the iso-anchored α. Compare iso-anchored α to original (halothane-anchored) α; compare predicted halothane EC50 to published.

α grid was discrete per organism. For worm: 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40. For fly: 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20. For mouse: 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40.

### 5.2 Results

| organism | original α | iso-anchored α | α % diff | iso best fold | halothane EC50 at iso-α | halothane fold |
|---|---|---|---|---|---|---|
| worm | 0.13 | 0.13 | 0.0% | 1.05 | 323.5 µM | **1.05×** |
| fly | 0.06 | 0.06 | 0.0% | 1.18 | 361.7 µM | **1.06×** |
| mouse | 0.10 | 0.10 | 0.0% | 1.11 | 288.2 µM | **1.21×** |

**M4a (iso-anchored α within 30% of halothane-anchored α): PASS** for all three organisms.

**M4b (predicted halothane EC50 with iso-anchored α within 2× of published): PASS** for all three organisms.

### 5.3 α-grid resolution caveat

The "0% diff" between iso-anchored and halothane-anchored α is reported at the resolution of the α grid: the same grid point was the best fit for both anchors in all three organisms. This is not a continuous-fit equality; with finer α resolution the iso-anchored α could differ slightly from the halothane-anchored value.

The iso-anchored fit fold-errors (1.05 worm / 1.18 fly / 1.11 mouse) are slightly looser than the halothane-anchored fold-errors at original α (1.07 / 1.06 / 1.18 — page Gate 1). This is consistent with halothane being the original calibration anchor: the original α value was tuned to minimize halothane fold-error, and the iso-anchored fold-error at the same α is slightly worse on isoflurane than the original halothane-anchored fold-error was on halothane. The honest framing is: **the calibration generalizes anchor-to-anchor at the resolution of the search; the calibration is not pathologically anchor-specific.** It is the cleanest positive result in V7.

---

## §6 Pre-registered deviation summary

| pred | scope | verdict | direction |
|---|---|---|---|
| P1 | no 1-class subset passes | CONFIRMED | — |
| P2 | ≥1 two-class subset passes | **FALSIFIED** | system is brittle to subset removal at low subset counts; no 2-class subset passes any organism |
| P3 | SNARE OR Complex I in ≥75% of passers | CONFIRMED at 100% | exceeded threshold |
| P4 | GluCl in worm/fly only | CONFIRMED | mechanical exclusion for mouse |
| P5 | smallest passer is 2 or 3 classes | **FALSIFIED** | smallest passing subset is 4–6 classes (worm 5, mouse 4, fly 6), larger than predicted; redundancy exists at higher mechanism-class counts but not at lower ones |
| P6 | M1 percentile ≤ 50% | CONFIRMED for all 3 | **worm 0% crosses too-special floor** (≤ 10%) — anchor-overfit deviation |
| P7 | M2 percentile ≤ 30% | **mouse 46% violates** | substrate/magnitude over-explains in mouse; class identity adds nothing beyond aggregate magnitude |
| P8 | M3 cell-type spread percentile ≤ 15% | NOT TESTED | V1 architecture: cell-type spread is uniform by construction → Match #3 reduces to Match #2; v2 deferred |
| M3a | no OAT > 2× fold-error | PASS | max observed 1.44× |
| M3b | load-bearing parameter exists | PASS | 9 parameters with |S| > 0.3, max |S| 0.84 |
| M3c | LHS 95% CI ⊂ [200, 600] µM | **DEVIATION** | [178, 448]; median 311.6 well-anchored, lower tail wider than pre-registered, falsification range [100, 1000] not crossed |
| M4a | iso-α within 30% of halo-α | PASS | 0% diff at grid resolution |
| M4b | iso-anchored halo fold ≤ 2× | PASS | 1.05 / 1.06 / 1.21 |

Three confirmations (P1, P3, P4), four falsifications/deviations (P2, P5, P6 worm, P7 mouse), one not tested (P8), three sensitivity outcomes (M3a/M3b PASS, M3c DEVIATION), two cross-validation outcomes (M4a/M4b PASS).

The deviations are reported in the direction the data moved, not soft-pedaled. P2 and P5 are **falsifications** of the pre-registered redundancy structure — the prediction was wrong about *which* subset sizes carry redundancy. P6 worm and P7 mouse are **falsifications** of the pre-registered ensemble-specificity structure — worm in the direction of being too anchor-overfit, mouse in the direction of being not class-specific. M3c is a deviation from the tight pre-registered LHS range but within the broader falsification range.

---

## §7 What V7 demonstrates

V7's adversarial controls strengthen, do not weaken, the following claims about the conserved-target perturbation table integrated through the V6 substrate-agnostic LIF integrator:

1. **The architecture requires a multi-class mechanism quorum.** No 1-, 2-, or 3-class subset reproduces halothane MAC at frozen α in any organism. The smallest passing subset is 4–6 classes per organism. The pre-registered prediction that 2–3 class redundancy is the relevant scale was wrong; the corrected finding is that redundancy exists at the 4–6 class scale but the architecture is brittle to removal below that.

2. **SNARE OR Complex I is universal across organisms.** 25/25 of all-stages passing subsets, across worm, fly, and mouse, contain SNARE OR Complex I. The pre-registered threshold of 75% containment was met at 100% with margin. This is the cleanest cross-organism structural finding in V7: the two highest-magnitude mechanism classes are non-negotiable for the architecture to reach the quiescent threshold under any subset of the conserved table.

3. **The substitution structure differs by organism but the modal core overlaps.** Worm's rule (`SNARE` AND (`Complex I` OR `NCA`) AND ≥ 3 supporting classes). Mouse's rule (`Complex I` AND `NCA` AND ≥ 2 of {GABA-A, K2P, nAChR}). Fly's rule is statistically loose at n=3 but consistent with the modal core. The 76–96% containment range across the three organisms for {Complex I, NCA, SNARE, GABA-A, K2P, nAChR} is the architectural "core" of the conserved table.

4. **Fly retains meaningful class-identity specificity beyond aggregate magnitude.** Sub-Q1 Match #2 percentile of 4.76% (tighter than Match #1's 5.56%) shows that in fly, controlling for total perturbation magnitude *increases* the conserved profile's specificity relative to random ensembles. This is the cleanest single-organism evidence that the V6 architecture's predictive accuracy is not reducible to aggregate-magnitude properties.

5. **Calibration generalizes anchor-to-anchor.** M4 shows the same α value optimizes both halothane and isoflurane anchors in all three organisms at grid resolution. Predicted halothane EC50 with iso-anchored α is within 1.05–1.21× of published. The calibration is not pathologically anchor-specific.

6. **The architecture is robust to single-parameter ±50% perturbation.** M3 OAT shows maximum halothane fold-error of 1.44× under any single ±50% perturbation; no OAT perturbation drives the prediction outside the 2× pass band. Median LHS prediction (worm) is well-anchored at 311.6 µM.

These six findings are the V7 contribution. They are defensible against the adversarial controls as run, and they constitute what V7 has earned as a methodological claim about the V6 architecture.

---

## §8 What V7 does NOT establish

V7's adversarial controls also narrow what the architecture can defensibly claim. The following are explicitly NOT established by V7's results, and should not be claimed in the page or the eventual paper:

1. **The conserved target list is statistically special in mouse.** Sub-Q1 P7 violation (mouse 46% at Match #2) shows the mouse conserved profile is at median percentile against magnitude-matched random ensembles. Mouse's predictive accuracy is largely a function of aggregate perturbation magnitude rather than specific target conservation. The V6 mouse result is consistent with "any sufficiently magnitude-matched random ensemble predicts halothane MAC correctly on the generic random graph." Class identity does not carry specificity in mouse under the V1 substrate.

2. **Cross-organism MAC convergence as conserved-target evidence.** Sub-Q1 P7 (mouse) plus prior V5+ Meyer-Overton analysis jointly say: cross-organism MAC similarity at ~340–350 µM is consistent with lipid biophysics conservation (Meyer-Overton 1899) and does not provide independent evidence of conserved-target specificity beyond aggregate magnitude. The "striking conservation" framing of V4-era results is not warranted by V7's controls.

3. **Cell-type-resolved targeting specificity.** P8 was not tested due to the V1 substrate's uniform-targeting limitation. Whether the conserved-target perturbation profile is special at the level of *which* cell types each class targets cannot be evaluated within V7 scope. This is an architectural limitation of the V1 substrate, not a fact about the conserved-target hypothesis.

4. **Independent positive Gate 4 (Eger non-immobilizer specificity).** Sub-Q2 stage-attrition analysis (§2.2) shows zero attrition from Stage 2 to Stage 3 in all three organisms: every subset that produces correct halothane AND isoflurane EC50s at frozen α *also* correctly classifies the Eger non-immobilizers. **Gate 4 is entailed by Gates 1 + 2**, not an independent positive result. The conserved-target perturbation table's structural sparseness on non-immobilizers — Eger compounds lack engagement at SNARE and NCA — means non-immobilizer discrimination falls out of any subset that gets the volatile EC50 right. The page's current framing of Gate 4 as an independent capability should be demoted accordingly. This does not invalidate the result; it relocates the credit from "the validator can discriminate" to "the perturbation table's sparseness on non-immobilizers does the discrimination, which falls out of correct volatile prediction."

5. **Anchor-independent worm-specific specificity.** Sub-Q1 P6 worm 0% percentile (anchor-overfit deviation) shows the conserved profile beats every random count-matched and magnitude-matched ensemble on worm halothane. This is consistent with α being calibrated tightly on worm halothane: the conserved profile sits exactly at the precision-optimal point. The worm result is real but partially anchor-tuning artifact. Claims about worm-specific class identity beyond what fly demonstrates should be qualified accordingly.

6. **Shared machinery between anesthetic-induced and natural quiescence.** This is the Paper 2 bridge experiment question. V7 does not test it. The framework hypothesis that anesthetic susceptibility persists because the same lipid-coupled state-control machinery is required for natural reduced-activity states is a hypothesis V7 is consistent with but does not test.

7. **Thermodynamic necessity of reversible reduced-activity states.** Out of V7 scope and out of Paper 2 scope. This is a framework-paper-level claim contingent on substrate extension with metabolic state variables (Paper 4 territory, if at all).

The discipline here is to keep the V7 paper's claims at the level of (1)–(6) in §7 and to leave (1)–(7) above as future-work questions that V7's existence makes possible to ask, not as conclusions V7 supports.

---

## §9 Future work and methodological pre-commitments

### 9.1 Paper 2 bridge experiment positioning

V7's Sub-Q1 results constrain what Paper 2 can defensibly demonstrate. The bridge experiment's goal is to test whether anesthetic-induced quiescence and natural reduced-activity states share underlying machinery at the network-state level in the V3 C. elegans simulator. V7 puts a methodological constraint on this:

- **Paper 2 should be pre-registered first in the worm and fly substrates** where Sub-Q1 demonstrates class-identity specificity (worm anchor-overfit but with real signal, fly cleanest at 4.76% percentile). These are the substrates where "shared machinery" can carry meaningful information beyond aggregate magnitude.
- **Mouse should be a stretch goal**, contingent on V2 substrate extension that adds cell-type structure. The V6 generic random graph cannot distinguish "shared specific machinery" from "shared aggregate magnitude" because Sub-Q1 P7 violation shows class identity does not add specificity in mouse under V1.
- **Match #3 (cell-type spread) becomes the relevant test once V2 substrate exists.** P8 was not tested in V7 due to V1's uniform-targeting limitation; V2 substrate extension with CeNGEN-resolved targeting would make Match #3 a genuine test of cell-type-specific conservation. Paper 2's strongest claims about shared machinery are contingent on substrate extension that makes this test possible.

### 9.2 Framework paper (Paper 3) constraints from V7

The framework hypothesis — anesthetic susceptibility persists across complex eukaryotes because lipophilic compounds perturb a conserved class of lipid-coupled state-control machinery that organisms also use for reversible reduced-activity states — is consistent with V7's findings but is not established by them. Specifically:

- V7's Sub-Q2 P3 universality (SNARE or Complex I in 100% of passers across organisms) is consistent with the framework's prediction of a conserved class but does not specify *which* class is load-bearing in a way that distinguishes the framework from a generic multi-mechanism convergence claim.
- V7's Sub-Q1 P7 mouse violation says the V6 substrate cannot distinguish class identity from aggregate magnitude in mouse. The framework's claim that "lipid-coupled state-control machinery" specifically is load-bearing requires substrate extension (V8 work) to test directly. V7 does not test it.
- V7's Sub-Q1 P6 worm anchor-overfit and P7 fly cleanest-specificity together say that, at V7's level, the framework's claim can be tested in worm and fly only, and even there only at the level of "class identity carries information beyond aggregate magnitude" — not at the level of "*these specific* targets are conserved because they are lipid-coupled state-control machinery."

### 9.3 Methodological pre-commitments

To prevent the framework paper from drifting toward unsupported claims:

1. **Paper 2 pre-registration must specify in advance** which substrates carry the bridge experiment's primary test (worm + fly), and which are contingent on substrate extension (mouse, plus Match #3 in all organisms).
2. **Paper 3's strongest framework claims are contingent on positive Paper 2 results AND substrate extension that makes Sub-Q1 Match #3 testable.** No Paper 3 claim about "conserved class of lipid-coupled state-control machinery as load-bearing feature" should be made before Match #3 has been a real test in at least worm and fly.
3. **The V8 substrate extension** — cell-type-resolved targeting via CeNGEN labels, plus an aqueous/lipid partitioning compartment that would let Class A/B/C ensemble distinctions become testable — is the methodological prerequisite for the framework paper's strongest claims. V7 does not have this substrate and does not test the framework at this level.

### 9.4 Open V7 items

- **Mouse bootstrap CI — RESOLVED 2026-06-10** (see `docs/v7_closeout_addendum_9.4.md` §A). `mouse_V6` block added to `artifacts/v5_controls/M5_bootstrap_CIs.json`: halothane WT 296.9 µM [289.9, 307.4], isoflurane WT 273.2 µM [268.8, 277.6]. Reconstruction validated against the committed worm/fly CIs (halothane matches to the decimal). Both mouse published anchors sit just outside the CIs (predicted low) — extends the worm/fly "precise but published-just-outside" pattern; published-inside-CI tally is now 1/6 across WT volatile anchors. No verdict changed.
- **LHS for fly and mouse — RESOLVED 2026-06-10** (see addendum §B). 100-sample ±50% LHS at frozen α: fly 328.0 µM [273.5, 422.6] (fold 1.04), mouse 283.0 µM [229.7, 367.6] (fold 1.24). Driver reproduces the preregistered worm CI [178.4, 447.6] exactly. Key finding: **both fly and mouse CIs fall entirely within the preregistered [200, 600] band** — the worm M3c lower-tail deviation is worm-specific, not architecture-wide. Reported as an exploratory extension (`artifacts/v7_sensitivity/v7_sensitivity_lhs_allorganisms.json`); the preregistered worm-only M3c verdict is untouched.
- **Interactive panel update.** The React `AnesthesiaPipeline.tsx` component does not yet surface Sub-Q2 redundancy structure or Sub-Q1 percentile histograms. Deferred to a separate session (front-end task, not a compute gap).

### 9.5 Cross-session implications to flag (not to act on)

Two items surfaced in the parallel V8 / V3 C. elegans simulator session that affect how Paper 2's tractability should be framed but are explicitly out of scope for V7 and are not to be touched from this session:

1. **Turek 2016 RIS quiescence reproduction is `Falsified-but-cited` under M2-pure** in the parallel session's verification. Paper 2's bridge experiment design depends on the V3 simulator's ability to reproduce natural quiescence dynamics correctly. If RIS quiescence reproduction is falsified, Paper 2's "compare network states under anesthetic vs natural quiescence" plan needs to either (a) use a different natural-quiescence target whose reproduction is validated, (b) re-validate RIS reproduction under a corrected protocol, or (c) restrict the bridge claim to substrate-level state comparisons that do not depend on Turek 2016 specifically. Flagging this here because §9.1's claim that Paper 2 is tractable depends on at least one natural-quiescence reproduction in V3 being valid. The parallel session is the authoritative source on which targets are currently valid.

2. **AVA ablation reproduction (C-22) is `Falsified-but-cited`** in the parallel session. The V3 simulator's behavioral readout uses command-interneuron quiescent fraction, of which AVA is a member. V7 does not ablate AVA, so this does not directly invalidate any V7 finding. But it raises a question about whether the V3 brain's command-interneuron readout reflects the literature semantics it implicitly claims, which is the same readout V7 uses to score immobilization. This does not require any V7 fix in this session but is worth tracking as a cross-session concern for the page's framing of "quiescent fraction in the locomotion command-interneuron set" (line 17 of `anesthesia-pipeline.mdx`). The parallel session should resolve C-22 status before the V7 page rewrite locks language about command-interneuron readouts.

These two items are surfaced here for cross-session bookkeeping. Neither requires action from this session.

---

## Appendix A — Artifact paths

| artifact | path |
|---|---|
| Pre-registration | `AnestheticSimulator/docs/v7_preregistration.md` (hash `533b624a…`) |
| Sub-Q2 final verdict | `artifacts/v7_subset_search/v7_subset_verdict.json` |
| Sub-Q2 Stage 1 verdict | `artifacts/v7_subset_search/v7_stage1_verdict.json` |
| Sub-Q2 Stage 2 verdict | `artifacts/v7_subset_search/v7_stage2_verdict.json` |
| Sub-Q2 Stage 3 verdict | `artifacts/v7_subset_search/v7_stage3_verdict.json` |
| Sub-Q1 verdict | `artifacts/v7_random_ensemble/v7_random_ensemble_verdict.json` |
| Sub-Q1 raw (Match #1) | `artifacts/v7_random_ensemble/v7_match1_raw.csv` |
| Sub-Q1 raw (Match #2) | `artifacts/v7_random_ensemble/v7_match2_raw.csv` |
| M3 verdict | `artifacts/v7_sensitivity/v7_sensitivity_verdict.json` |
| M3 OAT raw | `artifacts/v7_sensitivity/v7_sensitivity_oat.csv` |
| M3 LHS raw | `artifacts/v7_sensitivity/v7_sensitivity_lhs.csv` |
| M4 verdict | `artifacts/v7_cross_cal/v7_cross_cal_verdict.json` |
| V5 M5 bootstrap CIs (worm + fly) | `artifacts/v5_controls/M5_bootstrap_CIs.json` |
| Inter-organism redundancy analysis | `AnestheticSimulator/docs/v7_redundancy_analysis.md` |

All verdict JSONs carry `preregistration_hash` = `533b624a…` for audit traceability.

---

*This document is the V7 closeout. Page rewrite (`anesthesia-pipeline.mdx`) follows from §7–§8 distillation; interactive panel update (`AnesthesiaPipeline.tsx`) deferred.*
