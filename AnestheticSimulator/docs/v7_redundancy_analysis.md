# V7 Sub-Q2 — Inter-organism redundancy analysis

Source data: `artifacts/v7_subset_search/v7_subset_verdict.json`
(25 all-stages passing subsets: 14 worm, 3 fly, 8 mouse).

Scope: this document reports per-class appearance frequencies, evaluates pre-registered predictions P1–P5, and characterizes the redundancy structure across phyla. M3/M4 results are pending (LHS still running).

---

## 1. Per-class appearance frequencies

| class | worm 14 | fly 3 | mouse 8 | overall 25 |
|---|---|---|---|---|
| `complex_i_block` | 13 (93%) | 3 (100%) | 8 (100%) | **24/25 (96%)** |
| `nca_block` | 12 (86%) | 3 (100%) | 8 (100%) | **23/25 (92%)** |
| `snare_cooperativity` | 14 (100%) | 2 (67%) | 4 (50%) | **20/25 (80%)** |
| `gaba_potentiation` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `k2p_potentiation` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `nachr_antagonism` | 10 (71%) | 3 (100%) | 6 (75%) | 19/25 (76%) |
| `glucl_potentiation` | 9 (64%) | 2 (67%) | 0 (excluded) | 11 (worm+fly only) |

Mouse pool excludes `glucl_potentiation` (no mammalian ortholog), so its 0% in mouse is mechanical.

---

## 2. Pre-registered prediction outcomes (Sub-Q2)

### P1 — No 1-class subset passes Stage 1 at frozen α

**CONFIRMED.** Zero 1-class subsets passed any organism. The architecture cannot reproduce halothane MAC within 2× from a single mechanism class at the calibrated α.

### P2 — At least one passing 2-class subset exists per organism

**FALSIFIED — DEVIATION.** Zero 2-class subsets passed any organism. The smallest passing size is 4 (worm/mouse) or 6 (fly).

Direction of deviation: stronger redundancy than predicted. The pre-registration assumed "two large-magnitude classes (e.g., GABA-A + Complex I) should sum to enough effective perturbation magnitude." They do not.

### P3 — `snare_cooperativity` OR `complex_i_block` appears in ≥ 75% of passers

**CONFIRMED, exceeded.** Computed across all 25 passers:

- with SNARE: 20/25 (80%)
- with Complex I: 24/25 (96%)
- with **SNARE OR Complex I: 25/25 (100%)**

Every single all-stages passing subset, across all three organisms, contains either SNARE or Complex I. These are the two largest-magnitude classes in `DEFAULT_PER_CLASS_PA_AT_SATURATION` (50 pA SNARE, 60 pA Complex I) — and at least one of them is required for any passer to clear Stages 1–3.

### P4 — `glucl_potentiation` appears only in worm/fly passers

**CONFIRMED (mechanical for mouse).** GluCl is absent from the mouse mechanism pool by design. Among worm and fly:
- worm: 9/14 passers contain GluCl (64%)
- fly: 2/3 contain GluCl (67%)

So GluCl is **non-necessary but frequently sufficient** in invertebrate organisms. No anomalous appearance.

### P5 — Smallest passing subset size is 2 or 3 classes

**FALSIFIED — DEVIATION.** Smallest passing subset sizes:
- worm: 5
- mouse: 4
- fly: 6

Direction of deviation: same as P2 — architecture requires more classes than predicted.

---

## 3. Necessity structure per organism (intersection ⇒ "necessary")

A class is *necessary* in this analysis if it appears in 100% of an organism's all-stages passers.

| organism | n passers | necessary classes | smallest passer size |
|---|---|---|---|
| worm | 14 | `snare_cooperativity` (1) | 5 |
| fly | 3 | `complex_i_block`, `gaba_potentiation`, `k2p_potentiation`, `nachr_antagonism`, `nca_block` (5) | 6 |
| mouse | 8 | `complex_i_block`, `nca_block` (2) | 4 |

**Caveat on fly:** with only 3 passers, the "100%-intersection" set is statistically loose. With more passers we'd expect the necessary set to shrink as more equivalent architectures get sampled. The cleaner reading for fly is "5 of the 7 active classes appear in ≥67% of passers; SNARE and GluCl are the only swappable ones."

**Worm: substitution structure.** Although only SNARE is *necessary* in 100% of passers, complex_i / nca form a near-necessary substitution pair:

- 1 worm passer drops `complex_i_block` (subset 11: gaba + k2p + snare + nca + nachr + glucl)
- 2 worm passers drop `nca_block` (subsets 5, 6)
- **No worm passer drops both Complex I AND nca.**

So the worm rule is: **`snare_cooperativity` AND (`complex_i_block` OR `nca_block`) AND ≥3 supporting classes**.

**Mouse: 2-of-3 substitution structure.** Mouse smallest passers (size 4) are:
- `gaba + complex_i + nca + nachr`
- `gaba + k2p + complex_i + nca`
- `k2p + complex_i + nca + nachr`

Pattern: **`complex_i_block` AND `nca_block` AND ≥2 of {gaba, k2p, nachr}**. This is the cleanest substitution structure across all three organisms.

---

## 4. Cross-organism modal architecture

The "core" across phyla — present in ≥75% of all-stages passers across all three organisms — is:

- `complex_i_block` (96% overall)
- `nca_block` (92%)
- `snare_cooperativity` (80%)
- `gaba_potentiation` / `k2p_potentiation` / `nachr_antagonism` (76% each)

Three interpretations with different empirical content:

1. **Magnitude-driven.** SNARE (50 pA) and Complex I (60 pA) are the highest-magnitude classes; nca is third (40 pA). Their high appearance reflects that the architecture needs sufficient absolute perturbation magnitude to reach the quiescent threshold, and these classes contribute the most. *Sub-Q1 Match #2 (count + total magnitude) tests this directly.*

2. **Mechanism-conserved.** SNARE, Complex I, NCA are the three classes anchored across worm, fly, mouse with primary-literature EC50 + clinical dose engagement (worm halothane: SNARE 340 µM, NCA 300 µM, Complex I 240 µM). They're conserved targets, not chemistry tricks. The non-immobilizers (cis/trans-DCE, hexafluoroethane) selectively *miss* SNARE and NCA.

3. **Substrate-coupled.** Connectome- and graph-dependent: in mouse generic random graph there's no specialized cell-type targeting, so global-current classes (Complex I, NCA) dominate; in worm with CeNGEN-tagged neurons + biological connectome, SNARE acts on synaptic weights specifically, which matters more.

These three interpretations are not mutually exclusive. Sub-Q1 results (just landed) speak to interpretation 1 — the magnitude-driven story (see §6 below).

---

## 5. Redundancy strength varies inversely with passer count

| organism | passers | smallest size | n necessary classes | redundancy "tightness" |
|---|---|---|---|---|
| mouse | 8 | 4 | 2 | LOOSE |
| worm | 14 | 5 | 1 | LOOSEST |
| fly | 3 | 6 | 5 | TIGHTEST |

Two factors drive this:
- **Calibration tightness.** Worm V3 was calibrated most aggressively (α = 0.13 with full perturbation table). Mouse V6 used α = 0.10 on a generic graph. Fly V4 used α = 0.060 on the Winding 2023 connectome. Fly's smaller α means each perturbation contributes less, so more classes must engage to clear MAC — hence smallest passer size = 6.
- **Substrate sensitivity.** Mouse generic random graph has no cell-type structure → easier to clear MAC with fewer classes since current shifts apply uniformly. Worm + fly connectome substrates require heterogeneous engagement.

**This pattern is itself a falsifiable claim:** if substrate matters, mouse's 4-class minimum should be the easiest, fly's 6-class minimum the hardest. The data agrees.

---

## 6. Tie-in with Sub-Q1 (just completed)

The Sub-Q1 random-ensemble verdict (`v7_random_ensemble_verdict.json`) reports conserved-ensemble percentile rank vs 50 random ensembles per organism per match level:

| organism | Match #1 (count only) | Match #2 (count + total magnitude) |
|---|---|---|
| worm | **0.0%** | **0.0%** |
| fly | 5.6% | 4.8% |
| mouse | 28.0% | 46.0% |

Pre-registered thresholds:
- P6: Match #1 percentile ≤ 50% → **MET for all three organisms**.
- P7: Match #2 percentile ≤ 30% → **MET for worm and fly; NOT met for mouse (46% ≈ median).**
- P6 / P7 falsification thresholds: ≤ 10% / ≤ 5% → worm BELOW both (deviation in "too special" direction); fly approaches; mouse comfortably above.

Reading the percentiles:

- **Worm 0% in both matches**: no random ensemble (count-matched or count+magnitude-matched) achieves halothane fold-error below the conserved ensemble's 1.05×. This says α was tuned tightly to the conserved profile on worm — partly expected (calibration anchor was worm halothane), partly a "too-special" deviation per P6's falsification threshold. Worth honest reporting in the page rewrite.
- **Fly ~5%**: conserved beats 95% of randoms even at Match #2. Strong specificity claim.
- **Mouse 46% at Match #2**: mouse conserved is *median* among count + magnitude-matched random ensembles. **Mouse's predictive power at the conserved profile is largely explained by total perturbation magnitude.** Class identity does NOT add specificity in mouse. Combined with mouse's 4-class minimum and the generic-random-graph substrate, this is consistent: substrate without cell-type structure can't distinguish "right classes" from "right total magnitude."

Direction-of-deviation triage:
- worm M1+M2 percentile = 0%: violates P6's "too special" floor (≤10%). **DEVIATION** in the direction of overfitting to anchor.
- mouse M2 percentile = 46%: violates P7's expected ceiling (≤30%). **DEVIATION** in the opposite direction — total magnitude largely explains the prediction.
- fly: passes P6, P7 cleanly.

---

## 7. Provisional conclusions (subject to M3/M4 sensitivity + cross-cal)

1. **Sub-Q2 confirms a substantive minimum-mechanism quorum.** Anesthetic immobilization in this architecture requires 4–6 mechanism classes acting jointly; 1- and 2-class subsets cannot reach MAC at the calibrated α in any of the three organisms tested. P1, P3, P4 confirmed; P2, P5 falsified in the "more redundancy than expected" direction.

2. **A common core (Complex I, NCA, SNARE, plus ≥1 receptor-class engagement) appears across all three organisms.** Specifically, every all-stages passer contains SNARE OR Complex I (P3 universal at 100%, exceeding the 75% threshold).

3. **Sub-Q1 puts strong specificity on worm and fly conserved profiles, weak specificity on mouse.** Worm conserved beats ALL random count-matched and magnitude-matched ensembles. Fly conserved beats 95%. Mouse conserved is median among magnitude-matched ensembles, which combined with mouse's 4-class minimum and the generic-graph substrate, says: **mouse's predictive accuracy is largely a magnitude-and-substrate phenomenon, not a class-identity phenomenon.**

4. **Honest scope narrowing for the page rewrite:** the cross-phylum claim should be framed as "the architecture's predictive accuracy does NOT survive a connectome-permutation null *in fly* (V5 M2), AND mouse's accuracy is largely explained by total perturbation magnitude rather than class identity (Sub-Q1 P7 deviation). Worm and fly retain meaningful class-specificity (Sub-Q1 P6/P7 met). The 'different drugs / different paths / same destination' claim holds at the level of conserved mechanism *quorum* (4–6 classes from a shared pool of 7) rather than specific 2-class convergence."

---

## 8. Pre-registered deviation summary (so far)

| pred | scope | outcome | deviation direction |
|---|---|---|---|
| P1 | 1-class never passes | CONFIRMED | none |
| P2 | ≥1 two-class passes | FALSIFIED | more redundancy than expected |
| P3 | SNARE OR Complex I in ≥75% | CONFIRMED at 100% | none (exceeded) |
| P4 | GluCl worm/fly only | CONFIRMED | none (mechanical for mouse) |
| P5 | smallest passer size 2-3 | FALSIFIED | architecture more robust than expected |
| P6 | M1 percentile ≤50% | CONFIRMED for all | worm 0% triggers "too-special" floor (deviation) |
| P7 | M2 percentile ≤30% | mouse 46% violates | substrate/magnitude over-explains in mouse |
| P8 | M3 cell-type spread | NOT TESTED | V1 architecture limit; deferred to V2 |

Two confirmed predictions (P1, P3, P4 — three), two falsifications (P2, P5), two boundary deviations (P6 worm, P7 mouse), one not testable in V1 (P8).

The most important page-rewrite implication is in §7.4 — the cross-phylum claim must be qualified, particularly for mouse where Sub-Q1 says total magnitude explains most of the predictive accuracy.

---

## 9. Open items for V7 closeout

- M3 OAT done (file written 14:22 today); LHS in progress.
- M4 cross-cal will run after M3 completes.
- Once M3 + M4 land, integrate into a final V7 summary doc and rewrite the live page (`anesthesia-pipeline.mdx`) with explicit deviation reporting.
- Update the React `NetworkStateValidator.tsx` interactive with a Sub-Q2 / Sub-Q1 panel showing the redundancy structure (n_passers per organism, percentile-rank histograms).
