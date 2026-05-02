# State validation data tables (V1)

Inputs to `phase_g_state_validator.py`. Built during M1 of the network-state validator
implementation (see `WAVE_P_ARCHITECTURAL_PLAN.md` for sequencing).

## Files

### `anesthetic_perturbation_table.csv`
Per-(anesthetic, mechanism_class) Hill-curve parameters built from primary literature.
Replaces the Vina-occupancy-driven Phase D inputs entirely. ~70 rows across 9 compounds
× 8 mechanism classes. DEFERRED rows are explicit; the validator reads them as no-effect
rather than imputing.

Hill: `occ(dose) = dose^n / (dose^n + EC50^n)`; `effect(dose) = 1 + (max - 1) × occ(dose)`.
For blocking classes, `max < 1` (e.g., 0.7 = 30% block at saturation).

### `worm_immobilization_anchors.csv`
Clean primary-literature worm immobilization EC50s. Boddington 2017 (propofol, fabricated PMID)
and etomidate (no clean worm anchor) DROPPED. Halothane is the M3 calibration anchor;
isoflurane is the M4 held-out test.

### `wb_directional_mutants.csv`
84 rows extracted from WormBase WS297 phenotype association table. Each row: a worm gene
annotated with a specific anesthetic-sensitivity phenotype (WBPhenotype:0001606–0001619)
and its supporting WBPaper. Direction = HYPERSENSITIVE / RESISTANT. Includes 9 dual-annotated
genes (alleles in opposite directions).

### `halothane_directional_test.csv`
Filtered subset of `wb_directional_mutants.csv`: the 13 unambiguous halothane anchors
(no allele-dependent direction conflicts), each tagged with mechanism class and LIF entry
point status. Used directly as the M5 / Gate 3 test set.

## V1 Gate 3 structural finding

Only **7 of 13** unambiguous halothane anchors map to mechanism classes with a current
modulation-layer entry point (Complex I × 5: gas-1, gas-2, nduf-6, ndus-8, nuo-1; NCA × 2:
unc-79, unc-80). All 7 are in the HYPERSENSITIVE direction. The 5 RESISTANT anchors
(dgk-1, eat-16, egl-10, goa-1, ocrl-1) all involve Gαo signaling or PIP2 phosphatase
pathways that aren't in the current modulation layer.

**Implication:** Gate 3 as defined cannot test directional discrimination in V1 because
the testable set is direction-monomorphic. Either:
- Reduce Gate 3 to magnitude prediction (predicted EC50 ratio within ~50% of literature)
  on the 7 testable HYPER anchors, OR
- Extend the modulation layer to include Gαo signaling (W_syn-up multiplier on neurons
  expressing GOA-1, DGK-1) before running M5. This is ~1 extra day of work and gives
  goa-1, dgk-1 RESISTANT predictions for the test.

Decision deferred to M5 kickoff.

## Citation hygiene

All literature anchors carry a PMID (or DOI). Three categories:
- **PRIMARY** — direct measurement at the receptor/channel class
- **HOMOLOG** — mammalian receptor → worm ortholog transfer
- **ANALOGY** — transfer from a related mechanism class with structural conservation
- **CONSERVATIVE** — small-effect default; flagged where no clean primary anchor exists
- **DEFERRED** — left null; not fabricated

The fabricated `Boddington 2017` propofol anchor and the `Sedensky 2001` (PMID 11756669)
twk-18 entry from the original `mutant_panel.csv` (which lists twk-18 as halothane RESISTANT
contradicting CP6's correction to HYPERSENSITIVE) are NOT used here. The new `wb_directional_mutants.csv`
table is sourced directly from WormBase WS297 ontology + per-row WBPaper citations.
