# Pre-flight pushback — Wave P calibration work block

**Author:** pre-flight (before launching CP1-CP6)
**Status:** PAUSED for cross-session review
**Date:** 2026-04-27

---

## Summary

The work block as written is methodologically sound in skepticism but has three concrete issues that need resolution before launching CP1-CP6. None block calibration entirely; all reshape what calibration *can show*. Surfacing now per the methodology pattern's pre-flight discipline.

The first issue — Kd vs EC50 conflation — is load-bearing because most of the cited "Kd values" do not exist as classical equilibrium binding Kd, and treating functional EC50 as if it were Kd produces calibration artifacts that masquerade as pipeline failures. The second issue — K_p amplification as the actual mechanism behind 30/30 engagement — is also load-bearing because pipeline diagnostics already strongly suggest where the bias lives, and CP4's error decomposition should be designed to surface this specifically. The third — Lu 2007 NALCN — is a smaller scope issue but illustrates the citation-hygiene risk for the proposed targets.

---

## Pipeline state verification (CP0-style sanity check)

Before pushback, confirmed the existing pipeline outputs are real:

- `vina_results.csv`: 540 rows, all parse cleanly with sensible affinity values (-3.6 to -7.7 kcal/mol range)
- `wave2_overlay.json`: parses, 6 anesthetics × 30 targets × 1 parameter shift each = 180 entries, all evidence-graded
- `best_pocket_per_target.csv`: 180 rows, occupancies 0.003 to 1.000

**Occupancy distribution at 1× EC50 across the 180 (anesthetic, target) pairs:**

| range | count |
|---|---|
| occ < 0.50 | 38 |
| 0.50 ≤ occ < 0.90 | 13 |
| 0.90 ≤ occ < 0.99 | 69 |
| occ ≥ 0.99 | 60 |

**Per anesthetic median + (n above 0.9):**

| anesthetic | median occ | n >0.9 / 30 | Kp_oil_water | clinical EC50 (µM) |
|---|---|---|---|---|
| etomidate | 0.212 | **0/30** | 50 | 0.3 |
| halothane | 0.989 | 26/30 | 250 | 340 |
| isoflurane | 0.990 | 26/30 | 90 | 290 |
| ketamine | 0.999 | 30/30 | 10 | 5000 |
| propofol | 0.943 | 22/30 | 1300 | 1 |
| sevoflurane | 0.978 | 25/30 | 50 | 230 |

This distribution is interpretable, not random. Etomidate (low aqueous EC50, low K_p product) shows realistic discrimination. Volatile anesthetics + propofol (high K_p × EC50 product) show near-saturation across most targets. **The pipeline is responding to dose, not just to target.**

The 15 lowest occupancies are dominated by etomidate paired with non-membrane-embedded targets (RIC-4 membrane_interfacial 0.003, SNT-1 0.003, UNC-13 aqueous 0.009, UNC-18 aqueous 0.033) — exactly the cases where K_p amplification doesn't apply.

This pre-flight observation predicts the CP4 finding before CP4 is run: **the high occupancy across most pairs is driven primarily by K_p amplification of effective concentration, not by Vina ΔG predicting unrealistically tight binding.**

---

## Pushback issue 1 — Kd vs EC50 conflation in proposed ground-truth sources

### The problem

Vina docking predicts equilibrium binding free energy → **classical Kd** via `Kd = exp(ΔG/RT)`. The published anesthetic literature for the proposed calibration targets reports several DIFFERENT quantities, most of which are *not* classical Kd:

| Source | What it actually reports | Comparable to Vina Kd? |
|---|---|---|
| Mihic 1997 *Nature* (PMID 9311785) | EC50 of GABA-A potentiation by isoflurane/halothane in α1β2γ2 patch-clamp; mutant rescue (S270I) | NO — functional EC50, not equilibrium Kd |
| Krasowski & Harrison 1999 *J Pharm Exp Ther* | Concentration-response curves for GABA-A potentiation by various anesthetics | NO — functional EC50 |
| Mascia 1996 (PMID 8632881 family) | EC50 for GlyR potentiation by isoflurane/ethanol; mutant scanning | NO — functional EC50 |
| Beckstead 2002 | GlyR α1 potentiation by alcohols; M287 mutation | NO — functional EC50 |
| Forman 1996 | nAChR open-channel block by halothane; quantal current analysis | PARTIAL — block IC50, conditionally Kd-comparable under fast-equilibrium assumption |
| Wachtel 1995 | nAChR + halothane; concentration-response | PARTIAL — IC50 |
| Patel & Honoré 1999 *Nat Neurosci* (PMID 10570488) | TASK / TREK-1 activation EC50 by halothane/isoflurane | NO — activation EC50 |
| Lu 2007 *Cell* (PMID 17288899) | NALCN identification + UNC-79/80 partner; not a ligand-binding paper | NO — no Kd reported |
| Hanley 2002 *J Physiol* | Complex I inhibition IC50 by halothane via mitochondrial respiration | PARTIAL — IC50, comparable to Kd only with explicit substrate-competition assumption |

**Of 9 proposed ground-truth Kd sources, only ~2-3 (the nAChR ones, possibly Hanley) report quantities directly comparable to Vina's predicted Kd.** The rest are functional EC50 from receptor potentiation/activation/block, which differ from binding Kd by:

1. **Whether binding produces functional change at all** (efficacy term — not in Kd)
2. **Cooperative mechanisms** (Hill n in dose-response, not in classical Kd)
3. **Allosteric modulator dynamics** (most volatile-anesthetic targets are allosteric, not orthosteric — Vina doesn't natively distinguish)
4. **Cheng-Prusoff conversion possible only under specific assumptions** (competitive ligand, fast equilibrium)

A well-known example: propofol's EC50 for GABA-A potentiation is ~1 µM (functional), but radioligand displacement Kd against orthosteric muscimol is much weaker because propofol is allosteric. These are different real numbers, both biologically meaningful, but **not interchangeable for pipeline calibration**.

### Implication

Running CP4 with EC50 values labeled as "experimental Kd" would produce:
- Apparent fold-errors that conflate Vina ΔG bias, K_p amplification, and Kd-vs-EC50 quantity mismatch
- A "VERDICT_PIPELINE_NEEDS_CORRECTION" finding when the actual issue is "we compared two different quantities"
- A correction factor that doesn't generalize

### Proposed adjustment

**CP1 should explicitly extract the QUANTITY TYPE per source:**

```
target, mammalian_homolog, anesthetic, value_uM, value_type, ...
GABA-A α1, P14867, isoflurane, 250, EC50_potentiation, ...
nAChR α4β2, P43681, halothane, 130, IC50_block, ...
TREK-1, O95069, halothane, 700, EC50_activation, ...
```

CP4 calibration metrics should then split by value_type:
- For value_type == "Kd_radioligand": direct Vina Kd comparison (Pearson + fold-error)
- For value_type == "EC50_*": rank-order calibration only (Spearman) + Cheng-Prusoff conversion documented as a separate sensitivity analysis
- For value_type == "IC50_*": same as EC50 with caveats

If <3 targets have true equilibrium Kd values (which is likely), absolute calibration is methodologically problematic. The verdict should reflect that limitation rather than treat fold-error against EC50 as pipeline failure.

---

## Pushback issue 2 — K_p amplification is the most likely actual bias source

### The diagnostic

The 60-of-180 pairs at occupancy ≥ 0.99 are concentrated where K_p × clinical_EC50 is large:
- halothane: K_p × EC50 = 250 × 340 µM = **85,000 µM effective**
- isoflurane: 90 × 290 = **26,100 µM**
- propofol: 1300 × 1 = **1,300 µM**
- sevoflurane: 50 × 230 = **11,500 µM**

For a target with Vina-predicted Kd of ~50 µM (typical), occupancy at these effective concentrations is essentially saturated. This is *not a Vina problem*; it's a downstream amplification.

The K_p assumption — that effective concentration at a membrane-embedded binding pocket equals K_p × aqueous concentration — is a simplification:

1. **K_p is bulk lipid:water partitioning** measured in octanol/water systems, not specific binding-pocket microenvironments
2. **Pocket location matters**: a pocket facing the lipid leaflet vs facing the aqueous lumen of an ion channel pore experiences different effective concentrations
3. **Pocket polarity matters**: a polar pocket in a transmembrane protein may not see the lipid-amplified concentration of a hydrophobic anesthetic
4. **Experimental EC50 is typically reported for aqueous bath concentration**, not membrane concentration. The published EC50 is already the value the bath-applied receptor "sees."

If a published patch-clamp EC50 for GABA-A is 250 µM aqueous halothane, comparing to a Vina prediction with K_p × 250 = 62,500 µM is comparing apples-to-oranges: the EC50 is the dose the receptor encountered at bath, while the predicted occupancy uses a multiplied effective concentration.

### Diagnostic test

For mammalian homolog calibration, run CP4 metrics **twice** explicitly:
- **(A) WITHOUT K_p amplification**: compare Vina-predicted Kd directly to experimental Kd/EC50
- **(B) WITH K_p amplification** (current pipeline): compare effective-concentration occupancy to experimental occupancy at the bath EC50

**Prediction**: variant (A) will agree better with experimental data for membrane targets, suggesting K_p amplification is over-applied. Variant (B) will systematically overpredict occupancy.

If prediction holds, the fix is either:
1. Drop K_p uniformly (use raw aqueous EC50 as effective concentration)
2. Apply K_p only to specific compartment-tagged pockets (e.g., pockets explicitly facing lipid vs facing aqueous lumen)
3. Use a more conservative K_p_effective = sqrt(K_p) or similar dampening

### Implication

The original work block's CP4 design — "compare predicted Kd to experimental Kd" — would surface fold-errors that look like pipeline failures but are actually K_p over-application. Without explicit (A)/(B) decomposition, the verdict would mislabel the issue.

**Proposed CP4 adjustment**: explicitly produce two calibration tables, one with K_p, one without. Verdict examines both and reports whether discrepancy is in Vina ΔG, in K_p, or in both.

---

## Pushback issue 3 — Lu 2007 NALCN paper does not report a Kd

### The problem

Lu 2007 *Cell* "The neuronal channel NALCN contributes resting sodium permeability and is required for normal respiratory rhythm" (PMID 17268547, not 17288899 — citation needs verification per yesterday's pattern) is the paper that identified NALCN as a sodium-leak channel. It is **not a ligand-binding study**. The connection between anesthetics and NALCN/UNC-79/UNC-80 in *C. elegans* comes from genetic screens (Sedensky & Meneely 1987) showing halothane resistance in *unc-79*/*unc-80* mutants, not from biochemical binding measurements.

There is no published Kd for halothane against NALCN that I can locate at pre-flight. The functional/pharmacological data on NALCN-anesthetic interaction is largely indirect (genetic, electrophysiological).

### Implication

NCA-1 / NALCN cannot be used as a calibration target with the proposed methodology — there's no ground-truth Kd to compare against. Either drop from the calibration panel or replace with a different mechanism-class target.

### Proposed adjustment

Drop NCA-1 from CP1's panel. Either substitute with a better-characterized target in the same mechanism class (very few exist; NALCN is an outlier), or accept that the `nca_block` mechanism class is not calibratable at this work block.

---

## Pushback issue 4 — Negative controls (CP5) should be the load-bearing test

### The argument

Absolute Kd calibration (CP4) is fraught because of issues 1-3 above. **Discriminative power** (CP5) is much cleaner:

- If the pipeline gives 30/30 engagement for halothane AND 30/30 engagement for n-pentane (an inert solvent at clinical concentrations), the pipeline is producing artifact regardless of absolute calibration.
- If the pipeline gives 30/30 for halothane and ≤ 5/30 for n-pentane, the pipeline has discriminative power even if absolute Kd values are uncalibratable.

Discriminative power is the actual scientific question for the multi-target framing: do anesthetics specifically engage many targets, or does any small-molecule lipophilic compound saturate-bind every druggable pocket?

### Proposed adjustment

**Run CP5 first or in parallel with CP1.** Negative controls are fast (RDKit ligand prep + Vina docking, ~1 hr total) and produce a clean discriminative test that doesn't depend on the Kd-vs-EC50 issue.

Suggested negative controls (substantially less anesthetic-active than halothane at sub-anesthetic concentrations):

| compound | SMILES | rationale |
|---|---|---|
| n-pentane | CCCCC | small-molecule lipophilic, no anesthetic effect at sub-narcotic doses |
| methanol | CO | very weak anesthetic, requires lethal concentrations |
| dimethyl ether | COC | small ether, anesthetic-weak |
| 1,2-dichloroethane | ClCCCl | similar size/polarity to halothane, much weaker anesthetic |
| benzene | c1ccccc1 | aromatic small-molecule, weak anesthetic |
| trans-1,2-dichloroethylene | Cl/C=C/Cl | CONIANT — non-anesthetic isomer of an anesthetic (Eger 2001) |

The Eger 2001 conformational-isomers approach is particularly diagnostic: trans-1,2-dichloroethylene is non-anesthetic while cis-1,2-dichloroethylene is anesthetic (they're conformational isomers of similar lipid solubility). If the pipeline distinguishes them, it's responding to specific target-fitting features beyond raw lipophilicity.

---

## Pushback issue 5 — Verdict labels need refinement

The proposed verdict labels (PIPELINE_CALIBRATED / NEEDS_CORRECTION / UNCALIBRATED) presume absolute Kd calibration is achievable. Given the Kd-vs-EC50 problem, absolute calibration may not be the right axis. Proposed alternative labels:

- **VERDICT_DISCRIMINATIVE_AND_CALIBRATED**: negative controls show low engagement, anesthetics show high; absolute Kd values agree with experimental Kd within 2× for the subset where true Kd is available; pipeline biologically meaningful and ready for downstream phases
- **VERDICT_DISCRIMINATIVE_BUT_BIASED**: negative controls show low engagement (pipeline distinguishes anesthetics from inert), but absolute occupancy values are systematically high due to K_p amplification or Vina ΔG bias; correction factor (likely on K_p) brings into agreement; ship with documented correction
- **VERDICT_DISCRIMINATIVE_RANK_ONLY**: negative controls show low engagement, but absolute Kd cannot be validated because experimental data is mostly EC50 not Kd; rank-order calibration via Spearman is what's available; ship with explicit caveats about absolute occupancy interpretation
- **VERDICT_NON_DISCRIMINATIVE**: negative controls produce comparable engagement to anesthetics; pipeline lacks discriminative power regardless of absolute calibration; pipeline rebuild needed

The first three are all green-light outcomes for downstream Wave P phases (Phase E/F/G/H), with appropriate caveats. The fourth is the hard fail.

---

## Summary of proposed work-block adjustments

1. **Run CP5 (negative controls) first or in parallel with CP1.** Discriminative power is the load-bearing test; absolute Kd calibration is methodologically constrained.

2. **CP1 extracts value-type explicitly**: Kd_radioligand vs EC50_potentiation vs IC50_block vs EC50_activation. Calibration metrics split by type.

3. **CP4 produces two calibration tables**: with and without K_p amplification. Surfaces whether K_p over-application is the actual bias source.

4. **Drop NCA-1 from CP1 panel** unless a different ground-truth source is identified. The mechanism class becomes calibration-deferred.

5. **Verdict labels refined** to four-way categorization that separates discriminative power from absolute calibration.

6. **Time scope adjustment**: with negative controls running early, total time may be similar (~4-6 hours) but produces interpretable findings earlier.

---

## What I'm asking for

Cross-session review of the adjustments above before launching the work block. The CP5-first reordering and CP4 K_p decomposition in particular are load-bearing methodology changes; I'd rather pause and confirm than launch with a design that surfaces "results" that aren't the right comparisons.

If adjustments are accepted, the modified plan is:
1. CP1 (parallel A): primary-source verification of all proposed Kd/EC50 sources, with value-type tagging
2. CP1 (parallel B): negative-control ligand selection (n-pentane, methanol, dimethyl ether, trans-1,2-dichloroethylene)
3. CP2: pull mammalian + negative-control structures (negative controls dock against C. elegans Tier-1 panel; mammalian homologs dock against same anesthetic panel)
4. CP3: docking — both directions
5. CP4: dual calibration tables (with/without K_p) + value-type-split metrics
6. CP5: discriminative test on negative controls — the load-bearing finding
7. CP6: verdict using refined four-way labels

If adjustments are rejected and the original CP1-CP6 should run as written, document why and proceed.

Standing by. Marker file at `artifacts/calibration/PAUSED_FOR_REVIEW.txt` will be created if no response received within the work-block startup window.
