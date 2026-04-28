# Case study 2 — Kd-vs-EC50 conflation in computational binding calibration

**Project:** AnestheticSimulator / Wave P pharmacology pipeline
**Date diagnosed:** 2026-04-27 (CP4-CP5 of rigor-tightening pass)
**Methodology pattern:** primary-source verification of what experimental measurements actually quantify, before treating them as ground truth

---

## Finding

Wave P's binding pipeline produces predicted dissociation constants (Kd) for general anesthetic ligands against canonical mammalian-homolog targets via AutoDock Vina docking + Hill-equation conversion (`Kd = exp(ΔG/RT) × 1e6`). The original calibration table compared Vina-predicted Kd against literature "EC50" values from electrophysiology dose-response studies — Mihic 1997 PMID 9311785 (GABA-A halothane EC50 250 µM), Patel & Honoré 1999 PMID 10321245 (TREK-1 halothane EC50 700 µM), and others.

The calibration appeared mediocre (76% within 10× tolerance band; mean |log_err| = 0.629). The directness-tier audit (CP4) revealed that **none of the 30 anchor entries are strict-Kd measurements.** All are functional EC50 (potentiation, activation) or IC50 (block) from electrophysiology dose-response, mitochondrial O2 consumption, or similar functional readouts.

For positive allosteric modulators (PAMs) of pentameric ligand-gated ion channels — the dominant target class for general anesthetics — functional potentiation EC50 systematically deviates from binding Kd. Per Forman & Miller 2016 *Anesth Analg* 123:1297 (PMID 27749338), the relationship is:

```
EC50_function ≈ Kd_binding × η_allo
```

where η_allo (allosteric coupling efficiency) is < 1 for PAMs. The functional response saturates at sub-Kd concentrations because the modulator only needs to occupy a fraction of binding sites to achieve maximum coupling. This means **a docking-derived Kd will appear systematically larger than the functional EC50** by a factor of 1/η_allo ≈ 3-10× for canonical anesthetic-receptor pairs.

The systematic bias is empirically confirmed: across the T1 strict subset (n=17 — recombinant single-target electrophysiology only, excluding mitochondrial assays and entries without numeric values), the **signed median log_err is +0.399** (10^0.399 = 2.50× systematic over-estimate of Kd vs EC50).

## CP5 quantification + correction

A single-parameter multiplicative allosteric correction was derived from the T1 strict-subset signed median:

```
f_allo = 10^(signed median log_err) = 10^0.399 = 2.50×
```

Applied as: `predicted_Kd_corrected = predicted_Kd_raw / f_allo`.

**Pre-correction T1 metrics (n=17):**
- mean |log_err|: 0.629
- median |log_err|: 0.437
- within 10× (|log_err| ≤ 1.0): 13/17 (76%)
- within 3× (|log_err| ≤ 0.477): 9/17 (53%)
- signed mean log_err: +0.527 (= 3.37× pipeline overestimate)

**Post-correction T1 metrics:**
- mean |log_err|: **0.454** (improvement Δ = -0.175)
- within 10×: **16/17 (94%)** (improvement Δ = +18 percentage points)
- within 3×: 8/17 (47%) (modest decrease — the correction shifts the mean but doesn't tighten the spread)
- signed mean log_err: +0.129 (= 1.34× residual, near zero)

## Cross-validation — does the correction generalize?

A leave-one-anesthetic-out (LOO-CV) analysis trained the correction factor on N-1 anesthetics (using their signed median log_err) and evaluated on the held-out anesthetic. If the correction overfit to a specific anesthetic chemistry, the held-out signed mean would be far from zero.

| held-out | train n | f_allo (train) | held signed_mean | held mean |log_err| |
|---|---|---|---|---|
| etomidate | 14 | 2.50× | +0.286 | 0.844 |
| halothane | 13 | 2.28× | +0.270 | 0.448 |
| isoflurane | 14 | 3.85× | -0.366 | 0.366 |
| ketamine | 16 | 2.50× | +0.575 | 0.575 |
| propofol | 14 | 3.33× | +0.047 | 0.701 |
| sevoflurane | 14 | 3.85× | -0.233 | 0.233 |

**LOO-CV summary:** mean held-out signed = +0.097, mean |held-out signed| = 0.296. The correction generalizes — held-out anesthetics' signed means cluster near zero (0.296 average distance from zero, compared to a pre-correction mean of 0.527).

**Generalization verdict:** ROBUST. The 2.50× allosteric correction is not specific to any one anesthetic chemistry; it captures a structural feature of the binding-vs-functional-readout difference that applies across PAMs.

## Per-chemical-class stratification (CP7)

After applying the universal f_allo, residual signed mean log_err per class was computed (T1 strict subset only):

| chem_class | n | pre signed | post signed | post mean |log_err| | post % within 10× |
|---|---|---|---|---|---|
| ALKANE_HALOGENATED (halothane) | 4 | +0.628 | +0.229 | 0.428 | 100% |
| ETHER_HALOGENATED (iso, sevo) | 6 | +0.287 | -0.112 | 0.153 | 100% |
| IV_ARYLCYCLOHEXYLAMINE (ketamine) | 1 | +0.974 | +0.575 | 0.575 | 100% |
| IV_IMIDAZOLE (etomidate) | 3 | +0.685 | +0.286 | 0.844 | 100% |
| IV_PHENOL (propofol) | 3 | +0.568 | +0.169 | 0.661 | 67% |

After correction, four of five classes are 100% within 10×. The IV_PHENOL (propofol) class retains a 67% within-10× rate driven by a single outlier: propofol-GABA-A log_err = +1.64 pre-correction, +1.24 post-correction. This is consistent with propofol's unusually strong allosteric coupling on GABA-A (η_allo ≈ 0.02 vs typical 0.1-0.3 for volatiles), making functional EC50 ~50× tighter than Kd.

## How the issue was caught

A pre-flight pushback before launching the calibration work block included the question: **"what does each ground-truth entry actually quantify? Are they binding Kd or functional EC50?"**

Reading the original CSV column headers showed `value_type` = `EC50_potentiation`, `IC50_block`, `EC50_activation`, `IC50_inhibition` — entirely functional. Cross-checking the `experimental_method` column showed `patch-clamp_dose-response` and `O2_consumption_assay` — none reported radioligand displacement or photoaffinity.

A targeted literature search confirmed: Hall 1994 (propofol), Husain 2003 (etomidate, photoaffinity), Eckenhoff 1996 (halothane, photoaffinity), Forman 1996 (nAChR halothane, kinetic) all report direct binding measurements but were NOT in the calibration table. The table's "Kd" label was a misnomer for what was actually a functional-EC50 table.

Once the directness-tier framework was applied, the systematic bias became immediately diagnosable: signed log_err > 0 across the T1 subset = pipeline overestimates Kd, consistent with η_allo < 1 PAM theory.

## Methodology lesson

**Surface 1 (project-specific):** Wave P's calibration ground truth must be relabeled as `functional_EC50_table` rather than `Kd_table`. The 2.50× allosteric correction is empirically validated and applied in `wave2_overlay_v2.json` for downstream consumers.

**Surface 2 (general):** computational binding predictions (docking, MM-GBSA, FEP) calibrated against functional dose-response data carry an implicit allosteric coupling factor. The factor is target-class-specific (PAM vs antagonist vs blocker), chemistry-class-specific (volatile vs IV vs ion-channel-blocker), and ligand-specific (propofol's η_allo on GABA-A is unusually small).

**Practical recommendation:** computational binding pipelines should report:

1. **Predicted binding Kd** (the docking output)
2. **Estimated functional EC50** (after applying class-specific allosteric correction)
3. **Comparison subset directness tier** — flag whether the calibration ground truth is binding (T0/STRICT_KD) or functional (T1/T2/T3)

Treating these as the same number — as the original Wave P calibration table did by labeling EC50 entries as "Kd" — produces a systematic bias that gets misread as pipeline failure.

## Generalization beyond anesthesia

The Kd-vs-EC50 conflation pattern shows up in:

- **High-throughput screening** — biochemical IC50 vs cellular EC50 are often used interchangeably; cellular permeability + allosteric coupling produce systematic offsets.
- **Computational ligand library scoring** for orphan GPCRs (where literature anchors are predominantly cAMP or BRET functional assays, not radioligand binding).
- **Antibody affinity prediction** — competitive vs non-competitive binding assays differ; ELISA EC50 ≠ KinExA Kd.
- **Enzyme kinetics in drug discovery** — IC50 in initial-velocity assays drifts from Ki by a Cheng-Prusoff factor that depends on substrate concentration.

The protective methodology is universal: **before benchmarking computational binding predictions, audit the ground-truth table for what each entry actually quantifies.** Don't assume that `value_uM` columns labeled "Kd" or "affinity" are binding Kd; many published "Kd" values in the older literature are functional EC50 fits or apparent constants from indirect measurements.

## Reference artifacts

- `artifacts/calibration/cp4_strict_kd_summary.md` — full directness-tier framework + tier counts
- `artifacts/calibration/cp4_directness_tiers.csv` — per-row tier annotations
- `artifacts/calibration/cp5_strict_recalibration.{csv,md}` — quantitative correction + LOO-CV
- `artifacts/calibration/cp7_summary.md` + `cp7_class_stratified.csv` — per-class stratification
- `artifacts/kinetics/wave2_overlay_v2.json` — corrected occupancies for downstream consumers
