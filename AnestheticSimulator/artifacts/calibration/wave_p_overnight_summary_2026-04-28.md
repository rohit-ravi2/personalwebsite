# Wave P Session 2 — overnight wake-up summary

**Date:** 2026-04-28 (overnight from 2026-04-27)
**Session:** Wave P Session 2 (post-CP1-CP8 rigor pass)

---

## TL;DR — headline finding

**All four primary stages landed.** Stage A confirmed CP1's Phase F parameter-lock at runtime (Phase E and Phase F bitwise-identical between v1 and v2 overlays). Stage B shipped Phase G architecture + perturbation manager + halothane dose-response demo with an honest gap analysis. Stage C shipped 5 methodology paper case studies (~6700 words). Stage D shipped Wave P × Wave 2 integration scoping with bounded test plan for next work block.

The single most useful finding: Phase G's halothane dose-response on a minimal LIF demo reaches 50% suppression at **0.01× clinical EC50** (100× tighter than Crowder 1996's behavioral EC50 anchor at 1× clinical). This **honest gap** is informative: at clinical concentrations, the binding pipeline reports near-saturating occupancy across all 30 Tier-1 targets, so the dose-response shape is determined by network coupling sensitivity rather than additional binding affinity. Behavioral EC50 sits at the intersection of binding × coupling × threshold — Phase G captures the first two but the third remains uncalibrated.

---

## Stage outcomes

### Stage A — Phase E/F v2 propagation: ✅ COMPLETE

**Result:** Phase E and Phase F predictions are bitwise-identical between v1 and v2 overlays.

- Phase E max |Δfold_change| = 0.0000 across all 6 anesthetics
- Phase F max |Δratio| = 0.0000 across all 6 anesthetics

**Why:** Phase E reads `n_Ca_delta` from the overlay parameters (unchanged in v2) and applies CLINICAL_EFFECTIVE_OCCUPANCY=0.30 as a hand-set scaling factor (unchanged). Phase F reads `rate_factor` (unchanged in v2). CP1's analytical claim that block_factor cancels in d_WT/d_g1 is empirically reconfirmed.

**Implication:** to make Phase E genuinely consume CP7's corrected occupancies, `phase_e_markov_synapse.py` would need to switch from CLINICAL_EFFECTIVE_OCCUPANCY (hand-tuned) to per-target overlay occupancy. This is documented in `artifacts/calibration/phase_ef_v2_propagation.md` as a Phase G design decision for the next work block.

**Artifacts:**
- `src/phase_ef_v2_recompute.py` — recompute driver
- `artifacts/calibration/phase_ef_v2_propagation.{csv,md}` — comparison tables + verdict

### Stage B — Phase G network perturbation: ✅ COMPLETE

**CP B.1 — Architecture document:** `artifacts/phase_g/phase_g_architecture.md` (~10K chars). Defines API, mechanism class → perturbation hook mapping, channel-to-neuron expression mapping (v1 simplified hand-curated), phenotype readouts, integration with LIFBrain, architectural decisions (no brain code modification, perturbation as wrapper, CP7 occupancies consumed directly), failure modes + mitigations, future work.

**CP B.2 — Implementation:** `src/phase_g_network_perturbation.py` ships:
- `AnestheticPerturbation` class with `compute_perturbation_vector()`, `apply_to_brain()`, `revert()`, `predict_phenotype()` methods
- `PerturbationProfile` dataclass for structured per-class output
- `hill_dose_scaling()` for proper Hill-equation dose-occupancy translation
- `CHANNEL_EXPRESSION` v1 dictionary covering 13 canonical *C. elegans* channels mapped to expressing neurons

**CP B.3 — Smoke test (halothane @ 1× EC50):**
- 8 mechanism classes engaged
- 30 targets engaged at occupancy > 10%
- Max class occupancy: 0.998 (pipeline reports near-saturating engagement)
- Mean class occupancy: 0.913
- Per-class: complex_i_block (5 targets), complex_ii_block (1), gaba_potentiation (2), glucl_potentiation (4), k2p_potentiation (3), nachr_antagonism (6), nca_block (3), snare_cooperativity (6)

**CP B.4 — Halothane dose-response sweep (minimal LIF demo, 50 neurons):**

| dose × EC50 | firing rate (Hz) | hyperpol (pA) | max class occ |
|---|---|---|---|
| 0.001 | 51.00 | -58.5 | 0.384 |
| 0.003 | 39.00 | -110.6 | 0.652 |
| 0.010 | 24.00 | -167.0 | 0.862 |
| 0.030 | 0.00 | -201.5 | 0.949 |
| 0.100 | 0.00 | -220.0 | 0.984 |
| 1.000 | 0.00 | -228.9 | 0.998 |

50%-suppression dose ≈ 0.01× clinical EC50 (100× tighter than Crowder 1996 anchor). Honest interpretation documented in `artifacts/phase_g/phase_g_dose_response_summary.md`.

**Artifacts:**
- `artifacts/phase_g/phase_g_architecture.md`
- `src/phase_g_network_perturbation.py`
- `artifacts/phase_g/phase_g_smoke_test.json`
- `artifacts/phase_g/phase_g_halothane_dose_response.csv`
- `artifacts/phase_g/phase_g_dose_response_summary.md`

### Stage C — Methodology paper case studies: ✅ COMPLETE

5 case studies drafted, ~6700 words total:

1. **case_study_phase_f_parameter_lock.md** (1117 words) — sensitivity-sweep methodology surfaces structural parameter-lock; (1-bf) cancellation analytical proof; downgrade verdict to PASS_PARAMETER_TUNED.

2. **case_study_kd_ec50_conflation.md** (1261 words) — directness-tier audit reveals all 30 ground-truth entries are functional EC50, not strict-Kd. f_allo = 2.50× allosteric correction. LOO-CV validates correction generalizes. 76% → 94% within-10× post-correction.

3. **case_study_eger_nonimmobilizer.md** (1379 words) — CP3 cis/trans-DCE FAIL + CP7 hexafluoroethane FAIL → binding pipeline lacks Eger non-immobilizer discrimination. Documented as boundary, not bug. Multi-target discriminative claim narrowed.

4. **case_study_twk18_direction_inversion.md** (1250 words) — original Anchor 6 had fabricated PMID + inverted biological direction (claimed RESISTANT, real is HYPERSENSITIVE per Singaram 2011). Mechanism-trace-vs-empirical-direction methodology surfaces inversions reliably.

5. **case_study_preflight_pushback.md** (1731 words) — umbrella thesis: structured pre-flight pushback is cost-effective methodology for AI-assisted scientific research. Cumulative catch-list: 37+ citation issues + 1 parameter-lock + 1 direction-inversion + Kd/EC50 conflation + saturation collapse. Templates: citation, parameter inventory, sensitivity sweep, boundary test.

**Artifacts:** `artifacts/methodology_paper/case_study_*.md` (5 files)

### Stage D — Wave P × Wave 2 integration scoping: ✅ COMPLETE

**Artifact:** `artifacts/phase_g/wave_p_wave_2_integration_scoping.md`

Covers:
- Substrate landscape (LIFBrain, GradedBrain, Phase δ-projected expansions)
- Mechanism class × Phase δ cell integration matrix
- Touch cascade × anesthesia predictions (halothane vs etomidate qualitative differences)
- Mutant phenotype predictions (gas-1, twk-18(cn110), sup-9, unc-13(s69)) with anchors
- 5-test execution plan (touch cascade discrimination, gas-1 hypersensitivity, twk-18 GoF, hexafluoroethane null, per-anesthetic dose-response)
- Cross-thread coordination requirements (info needed from Phase δ; info Wave P provides)
- Risk register
- Standing follow-ups including methodology paper integration outline

---

## Cross-thread coordination with Session 1's Phase δ overnight

Wave P provides to Session 1:
- `wave2_overlay_v2.json` — kinetic shifts per (anesthetic, target) for downstream consumption
- `AnestheticPerturbation` class API — drop-in perturbation hook for any Brian2-backed brain
- Phase G smoke test + dose-response infrastructure
- Test plan for next bounded work block

Wave P needs from Session 1:
- Phase δ cell roster (which production-grade cells are available post-overnight)
- CeNGEN-derived per-cell channel expression (for Phase G v2)
- Touch cascade closure status (PASS/FAIL/PARTIAL → unblocks Test 1)
- Per-cell biophysics validation status

Both wake-up summaries (Session 1 at `wave2/artifacts/overnight_run_summary_2026-04-28.md`; Session 2 at this file) should be read together for coherent cross-thread story.

---

## Standing followups for next work block

In priority order:

### High priority (advance Phase G against real substrate)

1. **Phase G LIFBrain integration smoke test** — exercise `apply_to_brain()` on the actual 300-neuron LIFBrain. Check W_chem matrix mutation behaves correctly; check NaN/inf handling.
2. **Phase G calibration against LIFBrain command interneuron baseline** — calibrate perturbation magnitudes so halothane @ 1× produces ~50% AVA suppression. Resolves the 0.01× saturation gap from Stage B.
3. **Test 1: halothane vs etomidate touch cascade discrimination** — if Phase δ closed touch cascade.
4. **Test 4: hexafluoroethane null perturbation on Phase G** — most informative test; documents whether network integration adds discrimination beyond binding.

### Medium priority (broaden Phase G)

5. **CeNGEN expression matrix integration** — replace simplified hand-curated CHANNEL_EXPRESSION with full CeNGEN.
6. **Mutant variant infrastructure** — gas-1 (Complex I × 0.4), twk-18(cn110) (K2P leak × 2), sup-9 (K2P leak × 0.3), unc-13(s69) (W_syn × 0.2).
7. **Tests 2, 3, 5** — gas-1 hypersensitivity, twk-18 GoF, per-anesthetic dose-response sweep.

### Lower priority (polish + documentation)

8. **Methodology paper outline draft** — assemble the 5 case studies + Phase G into a coherent methods paper structure.
9. **Phase F reformulation (Option C from CP1)** — make WT_dose absolute and gas-1_dose relative to a fixed behavioral threshold so block_factor doesn't cancel. Allow per-anesthetic Phase F predictions.
10. **ColabFold T4 fallback for NCA-1/UNC-80** — lift Anchor 4-5 from STRUCTURALLY_UNCALIBRATED.
11. **Citation propagation** — apply CP6 anchor corrections through architectural plans + documentation.

---

## What changed this overnight (file-level summary)

### New files created

```
src/phase_ef_v2_recompute.py                                 (Stage A)
src/phase_g_network_perturbation.py                          (Stage B)
artifacts/calibration/phase_ef_v2_propagation.csv            (Stage A)
artifacts/calibration/phase_ef_v2_propagation.md             (Stage A)
artifacts/phase_g/phase_g_architecture.md                    (Stage B CP B.1)
artifacts/phase_g/phase_g_smoke_test.json                    (Stage B CP B.3)
artifacts/phase_g/phase_g_halothane_dose_response.csv        (Stage B CP B.4)
artifacts/phase_g/phase_g_dose_response_summary.md           (Stage B CP B.4)
artifacts/phase_g/wave_p_wave_2_integration_scoping.md       (Stage D)
artifacts/methodology_paper/case_study_phase_f_parameter_lock.md      (Stage C)
artifacts/methodology_paper/case_study_kd_ec50_conflation.md          (Stage C)
artifacts/methodology_paper/case_study_eger_nonimmobilizer.md         (Stage C)
artifacts/methodology_paper/case_study_twk18_direction_inversion.md   (Stage C)
artifacts/methodology_paper/case_study_preflight_pushback.md          (Stage C)
artifacts/calibration/wave_p_overnight_summary_2026-04-28.md          (this file)
```

### No file modifications

- No changes to LIFBrain or Wave 2 brain code (per architecture decision: Phase G as wrapper)
- No changes to existing Phase A-F pipeline scripts (per Stage A decision: phase_e_markov_synapse.py overlay-consumption deferred)
- No changes to wave2_overlay_v2.json (CP7 output retained as-is)

### Tasks tracked

All 5 overnight tasks (Stages A-D + wake-up summary) tracked via TaskCreate/TaskUpdate; all complete.

---

## Verdict landscape (rolled up)

The post-overnight Wave P state combines:

- **CP1-CP8 rigor pass:** 7 verified anchors + 1 homolog-grounded + 5 awaiting wet-lab + 3 uncalibrated + 2 boundary FAIL
- **Stage A:** Phase E/F v2 propagation verified; CP1 parameter-lock empirically confirmed at runtime
- **Stage B:** Phase G architecture + perturbation manager + halothane dose-response demo. Architectural foundation for behavioral phenotype prediction. Behavioral threshold calibration is the gap.
- **Stage C:** 5 methodology paper case studies documenting load-bearing methodology patterns. Ready for paper integration.
- **Stage D:** Bounded test plan for next work block.

**Headline:** Wave P enters the next work block with full architectural infrastructure for Phase G, a documented behavioral threshold gap, 5 methodology paper case studies ready for assembly, and a coordinated test plan against Phase δ-expanded substrate.

The two boundary FAIL findings (CP3 cis/trans-DCE, CP7 hexafluoroethane) remain as the most interesting open scientific frontier — Phase G's hexafluoroethane null perturbation test (Test 4 in Stage D plan) is genuinely the next bet on whether network integration captures something binding alone misses.
