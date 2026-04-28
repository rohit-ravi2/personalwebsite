# Stage II — Wave 2 cellular expansion (overnight 2026-04-27/28)

**Spec reference:** `phase_v_w2_overnight_full_pipeline_prompt.md` Stage II
**Mode:** per-cell validation cycle, target ≥1 production-grade cell
**Outcome:** **AVAR PRODUCTION_GRADE in 69 seconds.** VA5/VD5/VB6 strategic
deferral with full rationale.

---

## Headline finding

**Stage II shipped 1 production-grade cell (AVAR) and identified a
strategic-pivot rationale for VA5/VD5/VB6 deferral that aligns Stage III/IV
work with overnight scientific value.**

The AVAR validation result is exceptional: numerical-precision agreement
(11/11 VC holds with divergence < 0.003; 7/7 CC sweeps with timepoint-level
residual < 3 mV across 35,000 timepoints, with most residuals at 0.000 mV).

---

## CP II.1 — AVAR validation

### Pre-flight

- Primary source: Nicoletti 2024 (`AVAR_simulation.py`,
  `AVAR_simulation_vclamp.py`) — accessed directly from upstream repo at
  `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`.
- Channel inventory: [egl19, leak, irk, nca, unc103]. All 5 channels existed
  in `wave2/channels/` (with leak inline in cell). Zero novel translations
  needed.
- Pre-flight pushback: NONE.

### NMODL translation

No new translations needed. Reused 4 existing channels (egl19, irk, nca,
unc103) plus inline leak. All channels were already validated in earlier
Wave 2 work.

### Cell harness construction

`option_alpha_avar_cell.py` created. Templated from
`option_alpha_ava_cell.py`'s `build_brian2_aval_4channel`, extended with
UNC-103 contribution to `ik_total_mAcm2`. Added `I_ext` aliasing to `I_inj`
in preparation for Phase δ WB2 (modulation layer integration).

AVAR's distinguishing parameters (vs AVAL):
- Different surface area: 1121.79e-8 cm² (vs AVAL's 1123.84e-8)
- Different cm: 0.751761 μF/cm² (vs AVAL's 0.859551)
- Different e_leak: -37 mV (vs AVAL's -39 mV)
- Different gbar values for all 5 channels (per Nicoletti's published
  AVAR_simulation.py line 28)
- NCA non-zero (4.398e-6 S/cm² vs AVAL's 0)
- UNC-103 added (4.294e-6 S/cm²)

Smoke test passed: cell builds and integrates, V settles to ~-25.8 mV after
100 ms passive (consistent with Nicoletti's AVAR rest range).

### Validation

`run_stage_ii_avar.py` executed. Two components:

**Component 2a (voltage clamp, 11 holds at -80..+40 mV):**
- panel_pass: True
- holds passing: **11/11 (100.0%)**
- divergence at most holds: 0.0000 - 0.0021 (well below 0.05 tolerance)

**Component 2b (current clamp, 7 sweeps at -30..+30 pA, 1000 ms protocol):**
- panel_pass: True
- sweeps passing: **7/7 (100.0%)**
- timepoint pass: **35000/35000 (100.0%)**
- All per-feature residuals (peak, plateau, baseline_pre, baseline_post)
  reported as 0.000 mV (i.e., below the 3 mV tolerance — actually well below
  any meaningful threshold)

### Adversarial review

- F18-style channel-conflation check: AVAR has both EGL-19 (Ca channel) and
  no other Ca-using channel. No multi-USEION-ca conflict. PASS.
- Citation-attribution discipline: AVAR provenance is direct upstream
  Nicoletti 2024 (verified `soma.insert` calls in
  `AVAR_simulation_vclamp.py` lines 38-42). PASS.
- F19-style sub-threshold drift check: not applicable (AVAR shows clean
  steady-state at all 7 injection levels).

### Verdict

**PRODUCTION_GRADE.** Wall-clock: 69 seconds.

Output: `wave2/artifacts/avar_validation_results.json`
Code: `wave2/option_alpha_avar_cell.py`,
      `wave2/run_stage_ii_avar.py`

---

## CP II.2 — VA5/VD5/VB6 strategic deferral

### Pre-flight investigation

VA5/VD5/VB6 require 5 novel channel translations to reach production-grade:

| Channel | Source | Complexity |
|---|---|---|
| slo2iso | nicoletti_2024/slo2iso.mod | Low — structurally identical to slo1iso (template ~15 min) |
| slo2egl19 | nicoletti_2024/slo2egl19.mod | High — nanodomain Ca coupling to EGL-19 state, similar to slo1_egl19_coupled but new parameter set (~30-45 min) |
| slo1unc2 | nicoletti_2024/slo1unc2.mod (VB6 only) | Medium — likely similar to slo1_egl19_coupled, but UNC-2 coupling not yet established (~30 min) |
| slo2unc2 | nicoletti_2024/slo2unc2.mod (VB6 only) | Medium — same situation as slo1unc2 (~30 min) |
| cadiff | nicoletti_2024/cadiff.mod | Already implemented (`wave2/calcium_pool.py` with calibration R²=0.984), but per-cell wiring at production-grade has NOT been done (cadiff was used for AIY/RIM via caintra1 path; AIY/RIM cells use caintra1 via `cai_mM_static`, not dynamic cadiff in the cell builder). For VA5/VD5/VB6, cadiff must be wired into the cell with proper depth/beta calibration verification at the new geometry (~60 min including re-validation) |

Estimated total: 3-5 hours for VA5+VD5 (shared channels), additional 1-2
hours for VB6 (unique slo1unc2/slo2unc2). **Total Stage II envelope if
pursued: 4-7 hours.**

### Strategic-value reasoning

Stage IV's central scientific test is **"does the expanded brain reproduce
the touch cascade where pure LIF cannot?"** This depends on:

1. **Cellular Wave 2 layer for AVAL/AVAR.** ✅ Now present (AVAR added).
2. **Connectome wiring** from ALM/AVM (LIF) → AVB/AIB (LIF) → AVAL/AVAR
   (Wave 2). Already in the connectome.
3. **AIY/RIM** cellular layer. ✅ Already production-grade.

**VA5/VD5/VB6 are motor neurons** that receive output from AVA/AVB/PVC and
drive body wall muscle. They are *downstream of* the touch cascade, not part
of it. Replacing them with Wave 2 detail does NOT improve the cascade-cause
test in Stage IV.

For Stage III (Phase δ network integration), VA5/VD5/VB6 are also lower
priority than the AVA pair because:
- Modulation layer reads release events from interneurons primarily
- Connectome edges from VA5/VD5/VB6 go to muscle, not back into the brain
- Behavioral state distributions (Stage III WB5 Layer A check) are driven by
  the interneuron layer; motor neurons are the readout, not the controller

### Decision

**Defer VA5/VD5/VB6 to a future overnight or dedicated Wave 3 work block.**

Rationale:
- Stage II spec acceptance criterion is "≥1 additional cell production-grade"
  (AVAR satisfies this).
- Adding 4-7 more hours of channel translation + per-cell validation would
  consume the entire overnight envelope. With realistic 1-3 hour
  single-invocation envelopes, that means 2-3 more pause-and-resume cycles.
- The Stage III/IV scientific result (touch cascade test) is more valuable
  than additional motor-neuron coverage given the same 12-hour budget.
- Strategic pivot to Stage III with AVAR + AVAL + AIY + RIM panel (4 cells,
  bilateral AVA pair plus interneurons) gives **the best touch cascade
  coverage available** without VA5/VD5/VB6 motor neurons.

### Stage III implication

Stage III now uses 4 cells × 2 (bilateral pairs) = 8 NeuronGroups:
- AVAL, AVAR (Wave 2, 4-channel & 5-channel)
- AIYL, AIYR (Wave 2, 7-channel)
- RIML, RIMR (Wave 2, 7-channel)
- 1 LIF NeuronGroup of 294 cells for the rest

This is the same architecture as Phase δ scoping's Alternative B with the
substitution of AVAR's 5-channel cell for AIY's slot in count.

---

## Stage II acceptance check

- [x] ≥1 additional cell production-grade: **YES (AVAR)**
- [x] All cells attempted have primary source available: YES
- [x] Verdict documented with primary-source justification: YES
- [x] Per-cell hard stops respected: N/A (no debugging required for AVAR;
      VA5/VD5/VB6 deferred BEFORE attempting based on strategic-value
      reasoning, not after debugging stuck)
- [x] Methodology continuity preserved: YES (no F20+ patterns surfaced;
      all existing channels worked in their first instantiation in AVAR
      context)

**Stage II PASSES with 1 cell PRODUCTION_GRADE + 3 strategically deferred.**

---

## State after Stage II

- New cell builder: `wave2/option_alpha_avar_cell.py`
- New validation driver: `wave2/run_stage_ii_avar.py`
- Validation results: `wave2/artifacts/avar_validation_results.json`
- This findings doc: `wave2/artifacts/stage_II_findings.md`
- Status JSON: `wave2/artifacts/checkpoints/stage_II_status.json`

**Proceed to Stage III** with 4-cell panel (AVAL, AVAR, AIY, RIM).
