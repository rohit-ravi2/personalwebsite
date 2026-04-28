# Overnight full-pipeline run — running findings (2026-04-27/28)

**Spec:** `scripts/brain/artifacts/phase_v_w2_overnight_full_pipeline_prompt.md`
**Mode:** 4-stage Wave 2 expansion + Phase δ integration + biological validation
**Start:** 2026-04-27 (orchestrated overnight, Rohit asleep)
**Wake-up summary destination:** `wave2/artifacts/overnight_run_summary_2026-04-28.md`

---

## Pre-flight acknowledgment (Stage 0)

### Spec read
- Full spec at `phase_v_w2_overnight_full_pipeline_prompt.md` read end-to-end.
- 4 stages, hard-stop conditions, wake-up summary structure noted.
- Single-invocation envelope realistically 1-3 h; multi-invocation continuation acceptable.

### Reference docs read
- `wave2/artifacts/mellem_investigation_pushback.md` — Mellem 2008 = RMD (NOT AVA).
  Direct quote: "we never observed action potentials in AVA (n=10)." Plateau in
  RMD only. RMD is therefore high-priority Stage II target if Nicoletti 2019
  has matching channel set.
- `wave2/artifacts/phase_delta_scoping.md` — WB2-WB6 architecture; Alternative
  B (hybrid) with Alternative-C staging is the integration shape.
- `wave2/artifacts/phase_delta_wb1_findings.md` — codegen target unified to
  cython across 22 .py files; I_inj→I_ext rename deferred to WB2 (this work).
- `wave2/translation_patterns.md` — F1-F18 catalog (skimmed; F1-P5 read in
  detail).
- Existing channels confirmed: cca1, egl19, egl2, irk, kqt1, kqt3, nca, shk1,
  shl1, slo1_egl19_coupled, slo1_iso_dynamic_ca, slo1_iso, unc103, unc2 (14).

### Upstream Nicoletti 2024 inventory (huge load-bearing finding)

The orchestrator's pre-flight already located the upstream simulation directory.
Per-cell `soma.insert` calls extracted directly from
`~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/*_simulation_vclamp.py`:

| Cell | Channels | Novel channels needed |
|---|---|---|
| AVAL (existing) | irk, leak, egl19, nca | none (production-grade) |
| AVAR | irk, leak, egl19, nca, unc103 | **none** (all 5 existing) |
| AIY (existing) | egl19, slo1egl19, nca, leak, slo1iso, kqt1, shl1 | none (production-grade) |
| RIM (existing) | shl1, egl2, irk, cca1, unc2, egl19, leak | none (production-grade) |
| VA5 | slo2egl19, slo2iso, egl19, irk, shk1, leak, nca, cadiff | slo2egl19, slo2iso, cadiff (3) |
| VD5 | slo2egl19, slo2iso, egl19, cca1, irk, shk1, leak, nca, cadiff | slo2egl19, slo2iso, cadiff (3) |
| VB6 | slo2egl19, slo1egl19, slo2unc2, slo1unc2, slo2iso, slo1iso, egl19, unc2, cca1, irk, shk1, leak, nca, cadiff | slo2egl19, slo2unc2, slo1unc2, cadiff (4) |

`cadiff.mod` is the calcium pool. We already have `wave2/calcium_pool.py` and
`calibrate_calcium_pool.py` infrastructure but it's not yet a per-cell drop-in.

`leak.mod` is implemented inline in each existing Wave 2 cell as a
constant-conductance leak; not a separate channel module. Same for VA5/VD5/VB6.

### Cython coexistence confirmed (WB1 verified)
- `lif_brain.py` and `graded_brain.py` both use cython per WB1 sed migration.
- I_inj → I_ext rename still needed in 3 Wave 2 cells before integration.

### Three orchestrator pre-flight resolutions confirmed
1. Mellem 2008 reexamination already done (today's earlier work block).
   Direct ground truth in pushback doc. Stage I.2 will READ pushback, not redo.
2. `claude-chat-context.md` located at
   `/home/rohit/Desktop/website/personalwebsite/docs/claude-chat-context.md`
   for §5 falsification baseline at Stage IV.
3. Runtime envelope realistic 1-3 h per invocation; pause-at-stage-boundary
   discipline configured.

### Pre-flight pushback findings: NONE (proceed)

The headline result of pre-flight scoping is highly favorable:

- **AVAR is essentially free** — same channel set as AVAL plus UNC-103 (already
  translated). Estimated cycle time 30-60 min including harness fitness.
- **Touch cascade members (ALM, AVM, AIB, AVB) are NOT in Nicoletti 2024.**
  Nicoletti 2024 specifically covers motor + interneuron families: AVAL/AVAR,
  AIYL/AIYR, RIML/RIMR, VA5, VB6, VD5. ALM/AVM/AIB/AVB are NOT in the upstream
  mod-file repository. Any biophysical Wave 2 modeling of touch cascade members
  requires literature scoping outside of Nicoletti 2024 — and per Mellem
  pushback, citation chain audit precedent, this is a known risk surface.
- **VA5/VD5 share the same novel channel set** (slo2egl19, slo2iso, cadiff).
  If we can translate those three, we get TWO cells for the price of one
  channel-translation cost. VB6 adds 1 more (slo2unc2/slo1unc2 — these may be
  pattern-similar to slo2egl19/slo1egl19 already done).

### Runtime trajectory (revised from spec)

Given AVAR's near-zero novel-channel cost and VA5/VD5's shared channel set,
Stage II realistic envelope:

| Cell | Cycle time estimate | Novel channels |
|---|---|---|
| AVAR | 30-60 min | 0 |
| VA5 | 60-90 min | 3 (slo2egl19, slo2iso, cadiff) |
| VD5 | 30-45 min (shares VA5's translations) | 0 incremental |
| VB6 | 60-90 min | 1-2 incremental (slo2unc2 +/- slo1unc2) |

**Estimated 4-cell Stage II: 3-5 hours**, well within the 4-6 h target in spec.

### Touch cascade Stage IV reality check

Stage IV's "touch cascade biological validation" (§5 falsification baseline)
requires ALM/AVM/AIB/AVB in the cellular layer — but Nicoletti 2024 doesn't
cover them. Stage I.3 needs to honestly assess: is published VC/CC available
for these cells outside Nicoletti's set? If only calcium imaging exists,
Stage IV may need to operate with Wave 2 cells for motor neurons + LIF for
sensory/interneuron touch members. This is acceptable per the spec ("expanded
brain + activity-mode FSM" doesn't require ALL touch cascade cells to be
Wave 2 — just AVAL is the anchor).

### Plan

Proceed to Stage I literature scoping now. Stage I will:
1. Verify the Nicoletti 2024 cell inventory above against the paper directly
   (CP I.1).
2. Read Mellem pushback for RMD candidacy (CP I.2).
3. Quick PubMed scan for ALM/AVM/AIB/AVB primary VC/CC sources — document
   honestly if none exist (CP I.3).
4. Output ranked candidate list + estimated cycle times (CP I.4).

Hard stop at <4 strong candidates; document then halt.

---

## Stage I — running notes (begins below)

Stage I complete — see `literature_scoping_candidates.md`. 4 strong candidates
(AVAR, VA5, VD5, VB6); touch cascade members deferred for primary-source
unavailability outside Nicoletti 2024.

## Stage II — running notes

Stage II complete — see `stage_II_findings.md`. AVAR PRODUCTION_GRADE in 69 s.
VA5/VD5/VB6 strategically deferred (motor neurons not in touch cascade).

## Stage III — running notes

Stage III WB2 PASS, WB3 PAUSED — see `phase_delta_wb2_findings.md`. Hybrid
brain skeleton works in cross_coupling="off" mode. Cross-coupling biology
question (graded synapse model) needs human review per spec's hard-stop.

## Stage IV — running notes

Stage IV complete (reduced scope) — see `stage_IV_touch_cascade_findings.md`.
LIF baseline + per-edge mode reproduces cascade (AVA Δ+7.5 Hz on touch).
Wave 2 AVAL/AVAR plateau dynamics match Nicoletti 2024 across all 7 CC
levels.

## Run complete

Wake-up summary: `overnight_run_summary_2026-04-28.md`. All four stages
have outcome documentation; WB3 paused with three explicit questions for
Rohit's biology-judgment review.
