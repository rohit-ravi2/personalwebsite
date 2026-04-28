# Wave P — 6-month timeline

**Status:** SCAFFOLDED. Soft milestones; no calendar deadlines.

The timeline is the planning skeleton; actual cadence depends on user availability, Wave 2 progress, and gate outcomes. Mid-flight surfacing applies — phases may extend or contract based on findings.

---

## Month 1 (~2026-04-27 to 2026-05-26)

**Active phases:** Phase A (structures), tooling setup. Citation pre-flight (blocking).

**Wave 2 in parallel (expected):** IRK and UNC-103 translations land mid-month.

**Soft milestones:**

- All Tier-1 monomer structures predicted via AlphaFold DB pulls (week 1).
- ColabFold pentamer pipeline running (weeks 2-3).
- 12 multimer cases through ColabFold (weeks 2-4).
- RoseTTAFold-AllAtom backup runs on 4-6 low-confidence cases (weeks 3-4).
- Coverage report compiled; Gate A.1 evaluated (week 4).
- Citation verification queue resolved (the 8 blocking PMIDs).

**Risks active:**

- R1 (AF-Multimer fail), R10 (UNC-79/-80 IDR), R15 (license verification).

**End-of-month deliverable:** `artifacts/structures/coverage_report.md` and Phase A verdict (PASS / partial-fail / FAIL).

---

## Month 2 (~2026-05-27 to 2026-06-26)

**Active phases:** Phase B (docking), Phase C (occupancy + Gate C.1).

**Wave 2 in parallel:** Tier-2 channel translation may begin (NMR-1, GLR-1, etc.).

**Soft milestones:**

- Phase A end-of-block report; Gate A.1 verdict locked.
- Vina + DiffDock + GNINA cascade across 150 (target × anesthetic) pairs (weeks 1-3), all local + free-tier Colab T4 overflow.
- GNINA top-10 cross-method-agreement evaluation (week 3, replaces former FEP cloud burst). FEP path is DEFERRED per `preregistration/phase_b_binding_pose.md` §13.
- Photolabel cross-validation report (week 3).
- **Phase C runs (week 3-4).** Hill equation + membrane-partition adjustment.
- **Gate C.1 evaluated (end of month 2). LOAD-BEARING.**

**Risks active:**

- R2 (Vina/DiffDock disagreement), R4 (Gate C.1 falsification), R14 (pentameric structure prediction VRAM fit).

**End-of-month deliverable:** `artifacts/occupancy/gate_c1_evaluation.md`. Either:

- **Gate C.1 PASS** → Wave P proceeds to Phase D + E + F (months 3-4).
- **Gate C.1 FAIL** → Wave P pivots to single-target framework. Negative result is documented for paper.

---

## Month 3 (~2026-06-27 to 2026-07-26)

**Active phases (if C.1 passed):** Phase D (kinetic shifts; literature mining + MD), Phase E (Markov synapses), Phase F (metabolic layer) — all in parallel.

**Wave 2 in parallel:** continued Tier-2 channel work.

**Soft milestones:**

- `literature_shifts.csv` populated for 15+ literature-direct (target, anesthetic) pairs (week 1).
- CHARMM-GUI Membrane Builder setups for 8 MD systems (week 1).
- Mammalian-control MD (TREK-1, GABA-A, NALCN) complete; Gate D.1.2 evaluated (week 2).
- *C. elegans* MD systems complete (TWK-18, NCA-1, AVR-14, UNC-49, GAS-1) — 3-4 weeks.
- Phase E Markov synapse module implemented and validated (mEPSC freq, Ca cooperativity, unc-13 hypomorph) — week 2.
- Phase F metabolic layer implemented and smoke-tested (WT, gas-1, mev-1, atp-2 baselines) — week 2.

**Risks active:**

- R3 (MD divergence from literature), R6 (gas-1 mechanism), R13 (4060 Ti VRAM tight).

**End-of-month deliverable:** Phases D, E, F end-of-block reports. Anesthetic kinetic shift table compiled.

---

## Month 4 (~2026-07-27 to 2026-08-26)

**Active phases:** Phase D continues (final MD systems); Phase G (network runs) starts.

**Wave 2 in parallel:** Tier-2 work; possibly compartmental-fork experiments.

**Soft milestones:**

- All 8 MD systems complete; Gate D.1 evaluated (week 1).
- Phase G integration test (Wave 2 channels + Wave P kinetic shifts + Markov synapses + metabolic layer) — week 1.
- FSM IMMOBILIZED threshold calibration against WT control runs (week 1).
- First 600 of 2,400 main-grid runs complete (weeks 2-4). Throttled around 100-150 runs/week (~2 min/run).
- Lesion sub-grid runs prioritized to surface multi-target falsification early (week 2).

**Risks active:**

- R5 (WT EC50 wrong), R8 (compute scope), R11 (Wave 2 IRK/UNC-103 ship status), R19 (FSM threshold sensitivity).

**End-of-month deliverable:** First 600 grid runs aggregated; preliminary lesion-test verdict.

---

## Month 5 (~2026-08-27 to 2026-09-26)

**Active phases:** Phase G continues (remaining 1,800 runs); Phase H (validation) begins late month.

**Soft milestones:**

- Remaining 1,800 main-grid runs complete (weeks 1-3).
- All 40 lesion runs complete (week 1).
- Full grid aggregation; Hill EC50 fits (week 3).
- **Gate G.1 evaluated** (week 3). 5 sub-criteria (G.1.1-G.1.5).
- Phase H anchor evaluation (week 4). 8 anchors against 4/8 pass criterion.

**Risks active:**

- R5 (WT EC50), R6 (gas-1), R20 (lesion test inconclusive).

**End-of-month deliverable:** `artifacts/validation/program_verdict.md`. Either:

- **>= 4/8 anchors pass + lesion test holds** → program success at proof-of-concept level. Paper draft proceeds.
- **2-3/8 anchors** → partial fail; identify failing anchors; document upstream-phase rebuild plan.
- **0-1/8 anchors** → program fail; reframe paper as negative result.

---

## Month 6 (~2026-09-27 to 2026-10-26)

**Active phases:** Phase H write-up; Phase I (stretch) and Phase J (stretch) optional.

**Soft milestones:**

- Wave P paper draft (week 1-3). Outline at `papers/wave_p_paper_outline.md`.
- (Optional, if H >= 6/8) Phase I JAX inverse design begins — 2-week implementation + 2-week analysis.
- (Optional, if H >= 4/8) Phase J network signature analysis — 1 week.
- Production simulator plug-in implementation (`scripts/brain/anesthetic_overlay.py`) — 1-2 weeks.

**End-of-program deliverable:** Wave P paper submitted to target venue (Cell Systems / Neuron / eLife / PLOS Computational Biology).

---

## Wave 2 expected progress in parallel (informational)

Wave P does not depend on Wave 2 beyond IRK + UNC-103 (month 1-2) and possibly some Tier-2 channels later. Wave 2's expected cadence over 6 months:

- **Month 1:** IRK + UNC-103 translations ship. AVA full Brian2 path active.
- **Months 2-3:** Tier-2 channel translations begin (NMR-1, GLR-1, EGL-19 redux for compartmental work, etc.).
- **Months 3-5:** compartmental-fork experiments (if gate-2b regression surfaces in network testing).
- **Month 6:** documentation and paper preparation for Wave 2.

Wave P treats Wave 2's outputs as upstream dependencies but does not block on them; the hybrid Brian2+NEURON workaround for missing channels is documented in `integration/wave2_handoff.md`.

---

## Soft contingencies

**If Gate A.1 fails (month 1):** Phase A scope expands; multimer pipeline troubleshooting; possible scope reduction to 18-20 Tier-1 targets.

**If Gate C.1 fails (month 2):** Wave P pivots to single-target framework. Months 3-6 reframe around the highest-occupancy target (likely halothane × highest-Kd target). Paper becomes a multi-target falsification study.

**If Phase D MD calibration fails (month 3, controls disagree with literature):** Fall back to literature-only kinetic shifts. Reduced coverage. Phase G runs with caveat.

**If Phase G WT EC50 wrong by > 5× (month 4-5):** phase-by-phase debug starting from Phase B (Vina box, GNINA-to-Kd conversion, photolabel mapping). Deferred FEP path on top hits is available if user authorizes — see `preregistration/phase_b_binding_pose.md` §13.

**If Phase H < 2/8 anchors (month 5-6):** Reframe paper as negative result. Multi-target framing is wrong; document and submit as a falsification study.

---

## Cross-track coordination check-ins

End of each month: Wave P updates `STATUS.md` with:

1. Phase status changes.
2. Gate evaluation results.
3. Wave 2 ship status (IRK, UNC-103, Tier-2).
4. Notebook pipeline artifact freshness.
5. Production simulator status (untouched until month 6).
6. Citation pre-flight queue progress.
7. Compute usage vs budget.

The check-in is a single block at the bottom of `STATUS.md` with date and one-paragraph summary plus inline updates to the phase status tables.
