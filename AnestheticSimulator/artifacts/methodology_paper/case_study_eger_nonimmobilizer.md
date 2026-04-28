# Case study 3 — Eger non-immobilizer puzzle as boundary diagnostic

**Project:** AnestheticSimulator / Wave P pharmacology pipeline
**Date diagnosed:** 2026-04-27 (CP3 + CP7 of rigor-tightening pass)
**Methodology pattern:** explicit boundary tests that distinguish in-class calibration from cross-class discrimination

---

## Finding

Wave P's binding pipeline calibrates well on canonical anesthetic-target pairs (CP5: 94% within 10× tolerance after f_allo correction; CP7: 4/5 chemical classes 100% within 10×). The natural next claim would be: "Wave P discriminates anesthetic from non-anesthetic ligands by binding profile."

Two explicit boundary tests falsify that claim:

### Test 1 — cis vs trans 1,2-dichloroethylene (Eger 2001)

**Background:** Eger and colleagues established a specific class of compounds — "non-immobilizers" — that have lipid solubility comparable to clinical anesthetics but produce no measurable anesthetic effect (Koblin et al. 1994 *Anesth Analg* 79:1043; Eger et al. 2001 *Anesth Analg* 92:1395). Among the cleanest test pairs: cis-1,2-dichloroethylene (anesthetic) vs trans-1,2-dichloroethylene (non-anesthetic). Both have near-identical lipid:water partition coefficients. Their MAC-equivalent values differ because of geometric/conformational features, not bulk lipophilicity.

**Wave P test (CP3):** Vina-dock cis-1,2-DCE and trans-1,2-DCE against the 30 *C. elegans* Tier-1 anesthetic targets. Compute target engagement counts (occupancy ≥ 10%) at concentrations spanning the Eger anesthetic-range (1-10 mM aqueous):

| concentration | cis engaged / 30 | trans engaged / 30 | gap (cis − trans) |
|---|---|---|---|
| 100 µM | 0 | 0 | 0 |
| 300 µM | 0 | 0 | 0 |
| 1000 µM | 9 | 12 | **−3** (trans wins) |
| 3000 µM | 29 | 29 | 0 |
| 10000 µM | 30 | 30 | 0 |
| 30000 µM | 30 | 30 | 0 |

**Result:** Maximum cis−trans engagement gap across the concentration sweep is **0**. At the Eger anesthetic-range (1 mM), trans engages slightly MORE targets than cis (gap −3). At higher concentrations both saturate.

**Verdict:** FAIL — pipeline does not distinguish stereoisomers of small halogenated alkanes. Vina's scoring on small symmetric molecules cannot pick out the geometric features that separate anesthetic cis from non-anesthetic trans.

### Test 2 — hexafluoroethane (CP7)

**Background:** Hexafluoroethane (CF₃CF₃) is a halogenated alkane Eger non-immobilizer — high lipid solubility, no anesthetic activity. If Wave P's binding profile distinguishes it from clinical halogenated alkanes (halothane), that would constitute strong evidence that the binding pipeline picks up anesthetic-specific shape/chemistry beyond bulk lipophilicity.

**Wave P test (CP7):** Compare hexafluoroethane and cis-DCE (positive control — anesthetic) target engagement at 1 mM aqueous, post-CP5-correction:

- **hexafluoroethane:** engages **30/30** common *C. elegans* targets ≥ 10%
- **cis-DCE:** engages **22/30** common targets ≥ 10%

**Result:** Hexafluoroethane (Eger non-immobilizer) engages MORE targets than cis-DCE (anesthetic). The pipeline's binding profile is INVERTED with respect to the Eger anesthetic vs non-immobilizer classification.

**Verdict:** FAIL — pipeline lacks Eger non-immobilizer discrimination at the binding-pipeline level.

## Mechanistic interpretation

The Vina scoring function is dominated by:

1. Lipophilic pocket-fit (van der Waals + hydrophobic burial)
2. Hydrogen-bond donor/acceptor matching
3. Electrostatic complementarity at polar residues

For the Eger non-immobilizer class, all three of these features are present (highly halogenated → strong dispersion + Vdw interactions), and the targets are predominantly hydrophobic membrane-protein pockets (TM-domain GABA-A intersubunit, TREK-1 fenestration, Complex I quinone tunnel). Hexafluoroethane fits these pockets *better* than cis-DCE because its higher fluorine content increases dispersion and electronegativity.

What distinguishes anesthetic from non-immobilizer is therefore NOT pocket-binding affinity. Per Eger 2001's later work (Eger et al. 2008 *Anesth Analg* 107:479) and Eckenhoff 2007 review *Anesthesiology* PMID 17585226, the operative distinguishing features include:

- **Conformational selectivity** at specific receptor sub-states (cis-DCE binds the open-state TREK-1 conformation; trans-DCE does not)
- **Pharmacokinetics** — non-immobilizers fail to reach and maintain CNS concentrations at the predicted EC50
- **Coupling efficiency** — even when bound, the non-immobilizer fails to drive the conformational change that transduces binding into channel modulation
- **Network-level integration** — single-target engagement is insufficient; anesthesia requires coordinated multi-target engagement at specific dose ratios that non-immobilizers fail to achieve

Single-pose Vina docking captures none of these. The pipeline measures **lipophilic-pocket-fit detection**, not anesthetic-specificity.

## Why this is informative, not a Wave P bug

The Eger non-immobilizer puzzle has been a known mystery in anesthesia research for ~30 years. Computational binding pipelines cannot solve it for fundamental reasons (above). What Wave P's CP3 + CP7 tests demonstrate is:

1. **The binding pipeline operates as expected** — it identifies pockets that bind highly-lipophilic small molecules. Hexafluoroethane (highly fluorinated) and cis-DCE (chlorinated) both fit hydrophobic pockets. Vina can't distinguish them by score.
2. **The Eger discrimination problem is correctly localized to the network/coupling level**, not the binding level. Wave P's downstream Phase G perturbation layer + behavioral threshold may produce non-immobilizer-specific phenotypes via differential network coupling, but the binding-pipeline output cannot.
3. **Honest boundary documentation strengthens the rest of the validation table.** The 5 verified anchors (NDUFS2-halothane, KCNK2-halothane, GlyR-propofol, nAChR-propofol, multi-target discriminative) describe what the pipeline DOES. The CP3 + CP7 boundary FAILs describe what the pipeline DOES NOT. Both kinds of statement contribute to scientific honesty.

## Methodology lesson

**Surface 1 (Wave P-specific):** the multi-target discriminative gap (Stage 5 calibration: 28 targets separating anesthetic ligand class from negative-control ligand class) was computed using non-druggable negative controls (benzene, methanol, n-pentane, cyclohexane) — chemically very different from anesthetics. The Eger non-immobilizer test (CP3, CP7) is a **harder** discrimination test that used the well-known Eger negative-control class. Wave P passes the easy test (separating chemicals from drug-like) but fails the hard test (separating anesthetic from non-immobilizer within the lipophilic-halogenated class).

**Surface 2 (general):** when reporting binary "discriminative" claims for computational classifiers, **the choice of negative-control set determines the difficulty of the test.** A classifier that discriminates "drug-like vs random chemicals" is performing a much easier task than one that discriminates "anesthetic vs Eger non-immobilizer." Validation tables should make this distinction explicit.

**Surface 3 (broader):** computational pipelines that compute pocket-fit scores have a structural ceiling on what they can discriminate. Geometric/conformational selectivity, pharmacokinetics, coupling efficiency, and network integration are out of scope for single-pose docking. **Document the ceiling explicitly rather than chasing it with parameter tuning.** The Eger boundary is a real scientific frontier; reporting "pipeline cannot solve it" is more honest than reporting "pipeline solves it after we tune the discriminative threshold."

## Generalization

The "binding profile alone is insufficient" lesson applies to:

- **Computational anesthesia research** — single-target binding scores don't predict whole-animal phenotypes; multi-target integration + network simulation is needed.
- **Polypharmacology** in CNS drugs — the difference between effective antipsychotic (clozapine) and ineffective close analog (olanzapine in some populations) is partially network-coupling, not just receptor binding.
- **Selective vs promiscuous kinase inhibitors** — binding selectivity profiles correlate but don't predict clinical efficacy or off-target liability fully.
- **Allosteric vs orthosteric modulator design** — binding to the allosteric pocket doesn't predict the magnitude of orthosteric coupling effect.

In each case, the lesson is: **binding-pipeline outputs are necessary but not sufficient for phenotype prediction.** Validation tables should report what the pipeline does (binding affinity profile) and what it does not do (mechanism-of-action, dosing window, network-level integration).

## Wave P-specific implications

1. **Phase G is required for non-immobilizer discrimination.** The binding pipeline cannot solve it; the network perturbation layer (Phase G) consuming wave2_overlay_v2.json may capture differential coupling that distinguishes hexafluoroethane from halothane. This is an open hypothesis test for the next work block.

2. **Stage 5 discriminative gap claim narrowed.** The "discriminative gap = 28" is real but only against the easy negative-control set. The Wave P validation table now distinguishes:
   - Discriminative against drug-like vs non-drug-like: **VERIFIED** (Stage 5)
   - Discriminative against Eger anesthetic vs non-immobilizer: **FAIL** (CP3 + CP7)

3. **Ranking-based metrics retain value.** Stage 6 Spearman ρ = +0.93 across 22 EC50/IC50 entries reflects within-anesthetic relative-affinity ranking; this remains robust. Wave P can rank halothane's affinity for GABA-A vs TREK-1 vs nAChR consistently with literature, even if it cannot rank halothane vs hexafluoroethane.

4. **Methodology paper case study contribution.** The Eger-non-immobilizer-as-boundary-test framework can be applied to any computational binding pipeline making cross-class discrimination claims. The diagnostic is cheap (re-dock the published non-immobilizer ligand panel, compare engagement profiles).

## Reference artifacts

- `artifacts/calibration/dce_concentration_sweep.csv` + `dce_diagnostic_summary.md` — CP3 cis/trans-DCE results
- `artifacts/calibration/cp7_summary.md` — CP7 hexafluoroethane test
- `artifacts/calibration/negative_vina_results.csv` — full negative-control docking output (8 negative-control ligands × 30 targets)
- `src/calibration_cp3_dce_diagnostic.py` — reproducible cis/trans-DCE diagnostic
