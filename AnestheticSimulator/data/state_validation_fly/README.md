# V4 cross-species data tables (Drosophila larva, Winding 2023 connectome)

V4 M1 deliverable. Five tables paralleling worm V3 structure with fly-specific anchors and ortholog assignments.

## Files

### `fly_anesthetic_perturbation_table.csv`
Per-(anesthetic, mechanism_class) Hill-curve parameters. **Mirrors the worm V3 table almost verbatim** — the EC50/IC50 anchors are mammalian-receptor electrophysiology data (Mihic 1997 GABA-A, Patel & Honoré 1999 TREK-1, Forman 1996 nAChR, Hanley 2002 Complex I, Stewart 2000 SNARE, Lu 2007 NALCN) that extrapolate to fly orthologs the same way they extrapolate to worm orthologs. Same evidence grades.

The mammalian electrophysiology EC50s are the cross-species transfer mechanism. If the conserved-substrate hypothesis holds, applying these same numbers to a different connectome should produce calibrated predictions in a different organism.

### `fly_immobilization_anchors.csv`
Wild-type fly behavioral EC50 anchors:
- **halothane: 340 µM** (van Swinderen 1999 PMID 10051668) — same as worm calibration anchor
- isoflurane 290 µM, sevoflurane 230 µM (secondary)
- ethanol 40 mM (Wolf & Heberlein 2003)

Fly halothane MAC matches worm halothane MAC. Same anchor for cross-species calibration; if the architecture is right, alpha tuned on fly halothane should give a similar value to worm's α=0.13.

### `fly_directional_mutants.csv`
14-row table of fly anesthesia mutant anchors with directional sign:
- 4 SNARE-axis HYPER (Syx1A, unc-13, syt1, nSyb)
- 2 K2P RESISTANT (Sandman, ORK1)
- 3 NCA-axis HYPER (na, dunc-79, dunc-80)
- 2 Gαo RESISTANT (Goα47A, rdgA)
- 2 Complex I HYPER (ND-49, ND-75)
- 1 GABA-A (Rdl, deferred V1)

Includes literature-ratio estimates and source PMIDs where available. Compared to worm V3 (n=9), fly V4 has more direction diversity (4 RESISTANT vs worm's 2) but fewer total anchors with primary EC50 measurements.

### `fly_mutant_baseline_perturbations.csv`
Per-mutant LIF entry-point parameters for FlyLarvaBrain (mirrors V3's mutant_baseline_perturbations.csv structure). Adds one new field — `k2p_baseline_factor` — to model Sandman/ORK1 K2P loss-of-function as removal of baseline K leak current. Otherwise same fields as worm V3.

### `fly_nt_identity_heuristic.csv`
V1 heuristic for NT-identity assignment by Winding cell type. Used by FlyLarvaBrain (M2 deliverable) when constructing chemical synapse sign assignments. **One important species difference from worm: Drosophila Glu is iGluR-dominant in CNS (excitatory, sign +1), while worm Glu is GluCl-dominant (inhibitory, sign −1).** This is a structural species difference, NOT a calibration error.

V2 refinement: replace heuristic with FlyBase driver-line data + a future per-neuron NT prediction paper for larva (when one publishes). For V1, cell-type-level assignment is sufficient — mechanism classes target cell-type populations, not individual neurons.

## Cross-species transfer logic

1. **Connectome substrate** changes (Cook 2019 → Winding 2023; ~300 → 2952 neurons; chemical+gap → ad/da/aa/dd compartmental)
2. **NT identity assignment** changes (Loer & Rand 2022 worm → cell-type heuristic for fly larva)
3. **Glu sign convention** changes (worm CNS GluCl-dominant inhibitory → fly CNS iGluR-dominant excitatory)
4. **Behavioral readout substrate** changes (worm command interneurons AVA/AVB/etc → fly DN-VNC + pre-DN-VNC descending pool)
5. **Behavioral anchor** stays the same (~340 µM halothane MAC in both organisms — striking)
6. **Perturbation table** stays largely the same (mammalian receptor EC50s extrapolate to both orthologs)
7. **Mutant set** changes (worm WBPhenotype set → fly van Swinderen / Allada / Sandstrom set)

The minimum architecture diff is items 1, 2, 3, 4, 7. Items 5 and 6 hold across species under the conserved-substrate hypothesis. If the hypothesis is right, V4 should pass with a similar α and produce within-2× predictions of fly behavioral EC50s.

## Open issues for M2

1. **NT heuristic confidence.** LN→GABA, KC→ACh, sensory→Glu are HIGH confidence. pre-DN-VNC, DN-VNC, CN are LOW confidence and likely mixed. V1 may show degraded calibration if these populations are critical for the locomotor readout.

2. **Sandman K2P baseline modeling.** The `k2p_baseline_factor=0` for Sandman LoF removes a hypothesized constitutive K2P current. Whether the fly LIF brain needs an explicit baseline K2P contribution depends on whether the WT default firing rate matches biological observations. May need calibration.

3. **na (narrow abdomen) atypical phenotype.** Lear 2005 reports complex dose-dependence (HYPER at low dose, complex at high dose). V1 models as straight HYPER; this may be the row most likely to fail Gate 3 directional accuracy.

4. **Rdl GABA-A modeling.** Loss-of-function would reduce inhibition baseline, predicting RESISTANT — but Schweikert 2014 reports HYPER. Mechanism complex (compensatory upregulation of receptor in mutant?). Deferred from V1.

5. **Compartmental connectivity matrices** (ad/da/aa/dd) are available but not yet used. V1 uses all-all matrix for simplicity; V2 can refine to compartmental wiring (axon-axon synapses behave differently from axon-dendrite — Brian2 modeling for the difference).

## Provenance + commit hooks

These tables are V4 M1 deliverables. Once committed to the repo they're frozen for the V4 ensemble run. Any subsequent edits create a V5.
