# Wave P — Revision log 2026-04-27 (zero-external-spend revision)

**Date:** 2026-04-27
**Trigger:** User decision to implement Wave P entirely on local hardware (RTX 4060 Ti, 8 GB VRAM) with $0 external spend.
**Scope:** Surgical edits across the kickoff package to propagate the cost-elimination decisions; no rewrites.

---

## Files edited

1. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/SETUP_COMPLETE.md`
2. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/STATUS.md`
3. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/README.md`
4. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/WAVE_P_ARCHITECTURAL_PLAN.md`
5. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/infrastructure/dependencies.md`
6. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/infrastructure/compute_budget.md`
7. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/infrastructure/setup_colab.md`
8. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/preregistration/phase_a_structural_priors.md`
9. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/preregistration/phase_b_binding_pose.md`
10. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/preregistration/phase_c_occupancy_matrix.md`
11. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/preregistration/phase_d_kinetic_shifts.md`
12. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/risk/risk_register.md`
13. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/timeline/timeline.md`
14. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/papers/wave_p_paper_outline.md`
15. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/artifacts/binding/README.md`
16. `/home/rohit/Desktop/website/personalwebsite/AnestheticSimulator/src/phase_b_dock.py`

---

## Cost-elimination decisions applied

1. **Cloud bursts dropped entirely.** No FEP cloud spend ($0 external instead of $200-400). External budget total: $0.
2. **Colab dependency softened to overflow-only.** Default path is RTX 4060 Ti local. Free-tier Colab T4 (~12 hr/day session cap) reserved for pentameric edge cases; cumulative budget ~30 hours = ~3 calendar days. No Colab Pro / A100 quota required.
3. **Open-source structure-prediction substitutes added as load-bearing.** ESMFold (MIT, Lin et al. 2023 *Science* DOI 10.1126/science.ade2574), OpenFold (Apache 2.0, Ahdritz et al. 2024 *Nat Methods*), and Boltz-1 (MIT, Wohlwend et al. 2024) replace AlphaFold-Multimer / RoseTTAFold-AllAtom as the canonical Phase A predictor stack. AF-Multimer / RFAA remain available for academic-use cross-validation but are non-load-bearing.
4. **FEP dropped from Phase B canonical spec.** GNINA (CNN-rescored Vina) is the terminal step of the docking cascade. Multi-target framing needs *relative* per-target occupancy ordering, not absolute ΔG; GNINA's ML-rescored Vina recovers ranking accuracy within ~1 kcal/mol of FEP for ligand series, well inside the noise floor of partition-coefficient and Hill-coefficient assumptions in Phase C. FEP documented as DEFERRED / SPECULATIVE in `phase_b_binding_pose.md` §13. Gate B.1.4 rewritten from FEP-confirmation to GNINA top-10 cross-method-agreement.
5. **Pentameric VRAM constraint** added as risk R14 with mitigation ladder: ESMFold → Boltz-1 → OpenFold → free-tier Colab T4 → subunit-by-subunit pocket modeling.
6. **Complex I full-assembly dropped.** Phase A targets single-subunit-per-anesthetic-binding-site modeling (GAS-1 primary per Morgan & Sedensky 1995 PMID 7549290; NUO-1 through NUO-6 individually). Full ~45-subunit assembly is DEFERRED / SPECULATIVE. Added as risk R21.
7. **Storage** uses pre-existing `/mnt/ssd4tb/` 4 TB SSD; ~120 GB peak allocation; no procurement needed.

---

## Risk-register changes

- **R1 (AlphaFold-Multimer fails on pentamers):** mitigation rewritten to point at the open-source predictor ladder.
- **R2 (Vina/DiffDock/GNINA disagreement):** FEP rescoring removed; deferred-FEP-path escalation noted.
- **R5 (WT EC50 wrong by > 5×):** "FEP rescue burst" replaced with deferred-FEP-path reference.
- **R8 (compute scope creep):** strengthened to "no external spend without explicit user reversal of zero-cost commitment."
- **R13 (4060 Ti VRAM insufficient for MD):** demoted; canonical Phase D fits without cloud burst.
- **R14 (NEW): Pentameric structure prediction fails to fit in 8 GB VRAM.** Likelihood medium-high; mitigation ladder documented.
- **R15 (license terms on AF-Multimer / RFAA):** demoted to Low impact (open-source substitutes available).
- **R21 (NEW): Complex I full-assembly intractable on local hardware.** Canonical scope already routes around it via single-subunit modeling.
- **R22 (DEFERRED, was R14): Lambda Labs cloud burst exceeds $200.** Demoted from blocker to deferred enhancement; not applicable in canonical plan.

---

## Blocking-items list — collapsed from 7 to 4 desk-work tasks

Removed (no longer load-bearing):
- ~~License verification (AF-Multimer / RFAA)~~ — non-load-bearing in revised plan; ESMFold / OpenFold / Boltz-1 are MIT / Apache 2.0.
- ~~ColabFold quota / Colab Pro escalation~~ — overflow-only; free-tier T4 sufficient.
- ~~FEP cloud-burst budget approval~~ — FEP dropped from canonical Phase B.

Remaining (4 desk-work tasks, $0 cost):
1. **PMID pre-flight verification** — ~1-2 hr, free PubMed lookup across 8 blocking PMIDs.
2. **UniProt ID re-verification** — ~1-2 hr, free UniProt + WormBase across 25 Tier-1 targets.
3. **Wave 2 IRK + UNC-103 ship-status check** — ~30 min internal status; doesn't block until Phase G in month 4.
4. **Storage allocation confirmation on `/mnt/ssd4tb/`** — ~5 min `df -h` verification.

---

## New total external spend

**$0.** Documented across `compute_budget.md` §8, `dependencies.md` §0, `SETUP_COMPLETE.md` (declaration block), `STATUS.md`, `README.md`, and `WAVE_P_ARCHITECTURAL_PLAN.md` §6.

Deferred enhancement paths ($200-400 cloud-burst FEP, Colab Pro) are documented in `compute_budget.md` §4 and `phase_b_binding_pose.md` §13 — not authorized by default.

---

## Citation hygiene additions

New citations introduced in this revision:

- **ESMFold** — Lin et al. 2023 *Science* (DOI 10.1126/science.ade2574). Cited in `dependencies.md`, `phase_a_structural_priors.md`, `risk_register.md`.
- **OpenFold** — Ahdritz et al. 2024 *Nature Methods*. Cited in `dependencies.md`, `phase_a_structural_priors.md`, `risk_register.md`.
- **Boltz-1** — Wohlwend et al. 2024 (preprint, github.com/jwohlwend/boltz). Cited in `dependencies.md`, `phase_a_structural_priors.md`, `risk_register.md`.

Existing citations preserved (Eberhardt 2021, McNutt 2021, Corso 2023, Le Guilloux 2009, Lin et al. 2023, Morgan & Sedensky 1995 PMID 7549290).

---

## Verification grep

After edits, the following terms only appear in *deferred / DEFERRED / no-cloud / declared-zero-spend* contexts (not on the canonical path):

- "cloud burst"
- "Colab Pro"
- "A100"
- "$200"
- "$400"
- "FEP" (canonical mentions are now in §13 of `phase_b_binding_pose.md` as DEFERRED / SPECULATIVE; or in Phase C's input-consumption section noting the deferred path)

Verified 2026-04-27.

---

## Ready for Phase A?

**Yes — pending 4 desk-work blocking items, all $0, all completable in ~1 day.** Once PMID pre-flight, UniProt re-verification, Wave 2 ship-status check, and `/mnt/ssd4tb/` storage check are complete, Phase A can activate. The local predictor stack (ESMFold + OpenFold + Boltz-1) installs in ~30 minutes. No external spend authorization is required for any phase in the canonical plan.
