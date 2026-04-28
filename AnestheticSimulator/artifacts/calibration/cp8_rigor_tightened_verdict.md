# CP8 — Rigor-tightened verdict summary (Wave P, post-rigor pass)

**Date:** 2026-04-27
**Status:** Final consolidated verdict after CP1-CP7 rigor-tightening pass.
**Replaces:** the original "5/5 PASS" headline in WAVE_P_PHASE_H_VALIDATION.md.

---

## Executive verdict

> The Wave P binding pipeline is **calibrated within-class** for anesthetic ligands on their canonical mammalian-homolog targets (T1 strict subset, n=17, post-CP5-correction: 94% within 10×, mean |log_err| 0.45). After applying the single-parameter allosteric correction f_allo = 2.50× (CP5), 4/5 chemical classes show class-level mean |log_err| ≤ 0.66 with 100% within-10× rate. The pipeline **does not pass** the Eger 2001 anesthetic-vs-non-immobilizer discrimination test (CP3, CP7) — the binding output cannot distinguish cis-DCE from trans-DCE or hexafluoroethane from clinical alkanes by docking score alone. This is a known boundary; the Eger non-immobilizer puzzle is a network/coupling-side problem, not a binding-side problem.
>
> Phase F's gas-1 hypersensitivity prediction is **structurally parameter-locked** (CP1) — block_factor cancels in the d_WT/d_g1 ratio. The 2.48× value in Morgan's 2-3× band is achieved by tuning GAS1_COMPLEX_I_FACTOR, not by anesthetic-specific binding signal. Verdict downgraded to PASS_PARAMETER_TUNED with documented Phase G follow-up.
>
> Phase E's SNARE release-p reduction is **robust within sensitivity envelope** (CP2) — Stewart 2000's 0.3-0.7 fold-change band is reproduced across CLINICAL_EFFECTIVE_OCCUPANCY ∈ [0.10, 0.30] (3× range). PASS_WITH_SENSITIVITY_ENVELOPE.

---

## Rigor checkpoint summary

| CP | Topic | Verdict | Key number |
|---|---|---|---|
| CP1 | Phase F structural diagnosis | PASS_PARAMETER_TUNED | (1-bf) cancels in ratio; ratio = 2.48 ± 0.05 across 19× block_factor variation at GAS1=0.4 |
| CP2 | Phase E sensitivity | ROBUST | Stewart band reproduced 5/9 occupancy values; range [0.10, 0.30] |
| CP3 | DCE conformational diagnostic | **FAIL** | Max cis−trans gap = 0 across concentration grid 0.1-30 mM |
| CP4 | Strict-Kd subset construction | n=17 T1 | 0 STRICT_KD entries; 17 T1 functional EC50 (recombinant electrophys) |
| CP5 | Strict-subset recalibration | f_allo = 2.50× | 76% → 94% within 10× post-correction; LOO-CV signed mean = +0.097 |
| CP6 | Four-category anchor reframe | 5 verified / 1 homolog / 6 awaiting / 3 uncalibrated | twk-18 direction inverted; corrected per Singaram 2011 |
| CP7 | Class stratification + correction | 4/5 classes 100% within 10× | hexafluoroethane engages 30/30 targets vs cis-DCE 22/30 — non-discriminative |
| CP8 | Final consolidated verdict | (this document) | — |

---

## Anchor-by-anchor table (post-rigor)

| Anchor | Original verdict | Post-rigor category | Confidence | Key calibration metric |
|---|---|---|---|---|
| Multi-target framing (Stage 5 discriminative) | PASS | **VERIFIED** | HIGH | Discriminative gap = 28; no tunable parameters |
| Spearman rank correlation (Stage 6) | PASS | **VERIFIED** | HIGH | ρ = +0.93 over 22 EC50/IC50 entries |
| NDUFS2 / halothane Vina-Kd vs Hanley 2002 | PASS | **VERIFIED** | HIGH | log_err 0.001 pre-correction (canonical) |
| KCNK2 / halothane Vina-Kd vs Patel & Honoré 1999 | PASS | **VERIFIED** | HIGH | log_err 0.001 pre-correction |
| GlyR / propofol Vina-Kd vs Pistis 1997 | PASS | **VERIFIED** | MEDIUM | log_err +0.27 pre-correction |
| nAChR / propofol Vina-Kd vs Flood 1997 | PASS | **VERIFIED** | MEDIUM | log_err −0.21 pre-correction |
| Phase E SNARE release-p reduction (UNC-64 proxy, Stewart 2000) | PASS_PARAMETER_TUNED | **VERIFIED with sensitivity envelope** | MEDIUM | Stewart band reproduced 0.10-0.30 occupancy range |
| GABA-A / propofol Vina-Kd vs Krasowski 1999 | DEFERRED | **STRUCTURALLY_GROUNDED_BY_HOMOLOG** | LOW | log_err +1.64; propofol-GABA-A allosteric outlier |
| WT halothane EC50 (~3% atm = 280 µM aqueous) | PASS_PENDING | **STRUCTURALLY_GROUNDED_AWAITING_WETLAB** | MEDIUM | Phase G dose↔µM mapping required |
| WT isoflurane EC50 (~6% atm) | PASS_PENDING | **STRUCTURALLY_GROUNDED_AWAITING_WETLAB** | MEDIUM | Same as above |
| gas-1(fc21) hypersensitivity behavioral (Morgan 1994) | PASS | **STRUCTURALLY_GROUNDED_AWAITING_WETLAB** | LOW (parameter-tuned) | block_factor cancels; tuned to band via GAS1=0.4 |
| K2P-gain-of-function hypersensitivity (corrected per Singaram 2011) | DEFERRED + INVERTED | **STRUCTURALLY_GROUNDED_AWAITING_WETLAB** | MEDIUM | Direction corrected; binding side already verified |
| Whole-animal *C. elegans* propofol EC50 | DEFERRED | **STRUCTURALLY_GROUNDED_AWAITING_WETLAB** | LOW | Heuer 2014 channel-level IC50 (252 µM) closest published anchor |
| unc-79 / NCA-1 halothane resistance | DEFERRED | **STRUCTURALLY_UNCALIBRATED** | — | No AlphaFold structures; ColabFold deferred per R14 |
| unc-80 ~ unc-79 paralog | DEFERRED | **STRUCTURALLY_UNCALIBRATED** | — | Same as above |
| unc-13 hypersensitivity ratio specifically | PASS_PENDING | **STRUCTURALLY_UNCALIBRATED** | — | UNC-13 not in Tier-1 target list |
| Conformational specificity (CP3 cis/trans-DCE diagnostic) | NOT EVALUATED | **FAIL** | — | Pipeline does not distinguish cis from trans 1,2-DCE; max gap 0 |
| Eger non-immobilizer discrimination (CP7 hexafluoroethane) | NOT EVALUATED | **FAIL** | — | Hexafluoroethane engages more targets than cis-DCE at 1 mM |

## Counts (rigor-tightened)

- **VERIFIED:** 7 (Stage 5 discriminative, Stage 6 rank correlation, NDUFS2-halothane, KCNK2-halothane, GlyR-propofol, nAChR-propofol, Phase E SNARE release-p with envelope)
- **STRUCTURALLY_GROUNDED_BY_HOMOLOG:** 1 (GABA-A propofol with documented +1.64 log_err)
- **STRUCTURALLY_GROUNDED_AWAITING_WETLAB:** 5 (WT halothane EC50, WT iso EC50, gas-1 behavioral, K2P-gf hypersensitivity, whole-animal *C.e.* propofol)
- **STRUCTURALLY_UNCALIBRATED:** 3 (unc-79, unc-80, unc-13 specifically)
- **FAIL:** 2 (CP3 cis/trans-DCE conformational, CP7 hexafluoroethane non-immobilizer)

**Honest headline:** **7 verified + 1 homolog-grounded + 5 falsifiable awaiting wet-lab + 3 uncalibrated + 2 explicit-fail boundary findings.**

This replaces "5/5 PASS" with a textured picture: most within-class predictions are well-calibrated; the boundary failures (CP3, CP7) are documented limits, not hidden defects.

---

## Allosteric correction artifact

`wave2_overlay_v2.json` ships at `artifacts/kinetics/wave2_overlay_v2.json` with:

- All occupancies recomputed using post-CP5-correction Kd (divided by f_allo = 2.50×)
- `_meta.correction_log = +0.399`, `_meta.version = "v2"`, `_meta.source_doc = "cp5_strict_recalibration.md"`
- Per-target `occupancy_1xEC50_v1` retained for trace auditing alongside the corrected `occupancy_1xEC50`

Downstream Phase E/F/G consumers should switch from `wave2_overlay.json` (v1) to `wave2_overlay_v2.json` to use the corrected occupancies.

---

## Boundary findings — where Wave P is honest about limits

### CP3 — Conformational specificity FAIL (cis/trans-1,2-DCE)

**What was tested:** Eger 2001 reports cis-1,2-DCE is anesthetic; trans-1,2-DCE is non-anesthetic. Both have near-identical lipid solubility — so the only feature that could distinguish them at the binding pipeline level is shape. If Vina + the *C. elegans* target ensemble can pick out cis (anesthetic) from trans (non-anesthetic), that's evidence for shape-fitting; if it can't, the pipeline is responding to bulk lipophilicity.

**Result:** Maximum cis−trans engagement gap across concentrations 0.1-30 mM aqueous is **0**. At the Eger anesthetic-range (1-10 mM), trans engages slightly more than cis (gap −3 at 1 mM; tied at 3 mM and above).

**Implication:** The binding pipeline lacks the resolution to discriminate stereoisomers of small halogenated alkanes. Wave P's anesthetic-specific signal is real for distinct chemotypes (alkane vs ether vs phenol vs imidazole vs arylcyclohexylamine) but does not extend to within-chemotype geometric discrimination. This is a known limitation of single-pose Vina docking on small symmetric ligands.

**This finding is reported as a boundary, not a failure of the broader project.** The mechanistic question Eger's cis/trans-DCE poses is "what makes a compound anesthetic at the network/coupling level beyond pharmacokinetic parameters" — and that's a Phase G/H question, not a Phase B/C question.

### CP7 — Eger non-immobilizer discrimination FAIL (hexafluoroethane)

**What was tested:** Hexafluoroethane (CF3CF3) is a halogenated alkane Eger 2001 non-immobilizer. If Wave P's binding profile can distinguish hexafluoroethane from clinical alkanes by docking score alone, that would be a strong test of anesthetic-specific binding. CP7 measured engagement counts at 1 mM aqueous post-CP5-correction.

**Result:** Hexafluoroethane engages **30/30** common *C. elegans* targets ≥10% at 1 mM, while cis-DCE (anesthetic positive control) engages only **22/30**. Hexafluoroethane shows STRONGER binding profile than the anesthetic.

**Implication:** Same as CP3 — the binding pipeline's signal is dominated by lipophilic pocket fit, not anesthetic-specific shape/chemistry. This is consistent with the literature consensus that the Eger non-immobilizer puzzle is not solved by binding affinity alone (Eckenhoff 2007 review *Anesthesiology* PMID 17585226).

**Boundary, not breakage.** Wave P's value is in its quantitative within-class calibration on canonical anesthetic-target pairs. The anesthetic-vs-non-immobilizer classification problem is a separate scientific question that requires network-level simulation (Phase G) and possibly clearance/PK modeling.

---

## Citation hygiene (corrections shipped)

Six citation errors in original preregistration corrected during rigor pass:

| Original cite | Status | Corrected cite |
|---|---|---|
| Crowder 1996 PMID 8855256 | WRONG | **8873562** Anesthesiology 85(4):901-12 |
| Morgan & Sedensky 1995 PMID 7549290 | WRONG | **7943840** Anesthesiology 81(4):888-98 (1994) |
| Sedensky 1992 PMID 1346264 | WRONG | Sedensky & Meneely **1987** PMID **3576211** Genetics 116(3):417-26 |
| van Swinderen 1999 (cited for unc-13) | DOMAIN MIS-CITED | Paper is about unc-64 SNARE; for unc-13 use Nguyen 1995 PMID **7647836** |
| Sedensky 2001 PMID 11756669 (twk-18) | FABRICATED | Singaram 2011 PMID **22137475** Curr Biol 21(24):2070-6 |
| Boddington 2017 (propofol *C. elegans*) | FABRICATED | No primary source for whole-animal; closest is Heuer 2014 PMID **24501356** (channel-level oocyte IC50) |

---

## What this rigor pass establishes

1. **Wave P's within-class calibration is publishable**: 7 verified anchors, including 2 cleanest-in-pipeline log_err 0.001 entries (NDUFS2-halothane, KCNK2-halothane). Stage 5 discriminative gap and Stage 6 Spearman ρ=0.93 are independent of the calibration tuning.

2. **The single-parameter allosteric correction is empirically validated**: CP5's f_allo = 2.50× emerges from the T1 strict-subset signed median log_err and survives leave-one-anesthetic-out cross-validation. This is consistent with Forman & Miller 2016's PAM coupling efficiency theory (η_allo ~ 0.4).

3. **Phase F's structural limit is documented**: the d_WT/d_g1 ratio is parameter-locked to GAS1_COMPLEX_I_FACTOR. Phase G is required to test anesthetic-specific behavioral predictions. Honest verdict: PASS_PARAMETER_TUNED, not "biologically informative as written."

4. **Eger non-immobilizer puzzle remains open at the binding-pipeline level**: CP3 + CP7 explicitly show the binding pipeline cannot solve this. This is a *boundary* finding — Wave P's downstream Phase G/H may pick it up via network/coupling differences, but Phase B/C/D cannot.

5. **Citation chain repaired**: 6 errors corrected, including one direction-inverted anchor (twk-18). No remaining fabricated citations in the verdict table.

---

## Files shipped

- `artifacts/calibration/phase_f_structural_diagnosis.md` (CP1)
- `artifacts/calibration/phase_e_sensitivity.{csv,md}` (CP2)
- `artifacts/calibration/dce_concentration_sweep.csv` + `dce_diagnostic_summary.md` (CP3)
- `artifacts/calibration/cp4_directness_tiers.csv` + `cp4_strict_subset.csv` + `cp4_strict_kd_summary.md` (CP4)
- `artifacts/calibration/cp5_strict_recalibration.{csv,md}` (CP5)
- `artifacts/calibration/cp6_anchor_classification.md` (CP6)
- `artifacts/calibration/cp7_corrected.csv` + `cp7_class_stratified.csv` + `cp7_summary.md` (CP7)
- `artifacts/calibration/cp8_rigor_tightened_verdict.md` (this file)
- `artifacts/kinetics/wave2_overlay_v2.json` (CP7 — corrected occupancies; downstream consumer)

---

## Next work blocks (out of CP1-CP8 scope)

- **Phase G network simulation** — required to test anchors 1, 2, 3-behavioral, 6-genetic, 8-whole-animal. Phase G consumes `wave2_overlay_v2.json` and runs WAVE 2 Brian2 brain perturbation.
- **ColabFold T4 fallback for NCA-1/UNC-80** — required to lift anchors 4, 5 from STRUCTURALLY_UNCALIBRATED to grounded.
- **UNC-13 docking** — adds anchor 7-bis to Tier-1 target list.
- **Phase F reformulation (Option C from CP1)** — make WT_dose absolute and gas-1_dose relative to a fixed behavioral threshold so block_factor doesn't cancel; would allow anesthetic-specific Phase F predictions.
