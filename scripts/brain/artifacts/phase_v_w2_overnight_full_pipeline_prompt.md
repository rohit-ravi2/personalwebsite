# Overnight full-pipeline run — Wave 2 expansion + Phase δ integration + biological validation

**Mode:** overnight bounded multi-stage work block, autonomous execution with stop-and-pause discipline. Rohit asleep. File-based pause-and-wait pattern.

**Budget:** 12 hours wall-clock target; realistic single-agent-invocation envelope is 1-3 hours. Multi-invocation continuation possible via state file resume. Pause cleanly at stage boundaries when envelope expires.

---

## Pre-flight resolved (orchestrator second-order check before launch)

Three items the orchestrator verified before deploying:

1. **`claude-chat-context.md` LOCATED:** `/home/rohit/Desktop/website/personalwebsite/docs/claude-chat-context.md`. Agent should read for §5 falsification baseline in Stage IV.

2. **Mellem 2008 ALREADY REEXAMINED** today by Mellem investigation work block. Findings at `wave2/artifacts/mellem_investigation_pushback.md`. Direct ground truth: Mellem JE, Brockie PJ, Madsen DM, Maricq AV. 2008. *Nat Neurosci* 11:865-867. DOI 10.1038/nn.2131. PMC2697921. **Characterizes RMD plateau, NOT AVA.** Direct quote: "we never observed action potentials in AVA (n=10)." Stage I.2 should READ that pushback doc directly rather than re-investigating from scratch.

3. **Runtime envelope reality:** Single agent invocation typically completes 1-3 hours of work. Today's run #2 was the longest at ~2 hours. The 12-hour overnight design will likely span multiple invocations OR pause at stage boundary with partial completion. Both are acceptable — the pause-with-documentation discipline handles this cleanly.

Per-cell cycle time data from today's work:
- AVAL (option α-1): ~21 min subagent runtime (no new channel translations needed)
- AIY (option B): ~51 min (1 channel translation: KQT-1)
- RIM: ~82 min (3 channel translations: CCA-1, EGL-2, UNC-2)
- Cython migration: ~101 min (3-cell re-validation + benchmark)
- Phase δ scoping: ~9 min

Stage II per-cell estimate: 30 min - 90 min depending on novel channel count, possibly more if new pattern surfaces. 4-cell Stage II target = 2-6 hours single invocation.

---

## Strategic positioning

Wave 2's cellular layer is substantively complete (3 production-grade cells: AVAL, AIY, RIM; 14 channel translations; F1-F18 catalog with F19 standing followup; cython baseline 22.71× on pure-Brian2 benchmark; production-simulator codegen unified to cython per Phase δ WB1 just-completed).

This overnight run extends Wave 2 cellular layer with additional cells (Stage II) and integrates into the production simulator network (Stage III), then tests biological emergent behavior (Stage IV).

---

## Critical methodology continuity (load-bearing)

- **Pre-flight pushback at every stage boundary.** Each stage starts with pre-flight verification. Pre-flight discipline has caught >30 errors across this project's history. Applies overnight as much as in waking work.
- **Primary-source verification before treating any citation as ground truth.** The Mellem 2008 misattribution is the load-bearing case study. Don't propagate similar errors overnight.
- **Don't fabricate when you can't verify.** If primary source is paywalled or inaccessible, document the gap and proceed without that cell.
- **Pause-with-documentation always preferable to push-through.** Stage transitions are pause points. Architectural ambiguity is a pause point.
- **State persistence after every checkpoint.** Status JSON files at each subcheckpoint. Findings files at each stage.

---

## Working environment

- Brain code: `~/Desktop/website/personalwebsite/scripts/brain/`
- Wave 2 cells: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Wave 2 venv: `~/venvs/wave2-neuron/` (NEURON, Brian2, Cython, OpenCV, etc. installed)
- Cython is now production codegen target (WB1 completed; 22 .py files updated)
- Phase δ scoping doc: `wave2/artifacts/phase_delta_scoping.md`
- Phase δ WB1 findings: `wave2/artifacts/phase_delta_wb1_findings.md`

---

## Hard stop conditions (pause overnight if any of these surface)

1. **Stage I returns fewer than 4 strong candidates** beyond current AVAL/AIY/RIM.
2. **Stage II cell can't reach production-grade** after 2-3 hours of debugging on any single cell.
3. **Stage III WB3 release-event rule** surfaces architectural questions needing biological judgment.
4. **Stage IV touch cascade still fails to propagate** even with expanded layer (substantive scientific finding for human interpretation).
5. **Codegen failures, NaN/Inf, memory issues, integration crashes.** Pause and document.
6. **Wall-clock budget exceeded** (>12h, OR realistic single-invocation envelope expires).

**Stop-and-document, don't stop-and-fabricate.** Partial completion with documentation is valuable.

---

## Stage I: Literature scoping (~2-3 hours target, may span multiple invocations)

**Goal:** enumerate C. elegans neurons with sufficient published biophysical data for production-grade Wave 2 validation.

### CP I.1 — Enumerate candidate cells

Strong candidates to investigate:
- **From Nicoletti 2024 already in repo (highest priority — translations & references in hand):** AVAR (bilateral pair to AVAL, has UNC-103), VA5/VB6/VD5 (motor neurons), AWCon (Nicoletti 2019 reference), RMD (Nicoletti 2019 reference)
- **Touch cascade members (high priority for Stage IV):** ALM, AVM (mechanoreceptors); AIB (interneuron); AVB (forward antagonist)
- **Other candidates if data exists:** ASE (ASEL/ASER), AWA, AFD, ASH, PLM, URX/AQR/PQR/BAG, HSN, RIA, RIB

For each cell document:
- Primary source paper(s) found (PMID, year, journal, lab)
- Type of recording (full VC + CC, partial, calcium imaging only)
- Channel coverage (which ion channels characterized)
- Apparent feasibility (channel overlap with our 14 existing translations?)
- Biological importance (touch cascade member? command interneuron? sensory load-bearing?)

**Verification discipline:** verify each candidate paper by reading actual abstract/methods. Don't take agent-found citations at face value.

### CP I.2 — Mellem 2008 reexamination

**Use existing pushback document** at `wave2/artifacts/mellem_investigation_pushback.md` rather than re-investigating from scratch. Confirms: Mellem 2008 has RMD VC/CC data, NOT AVA. RMD is high-priority Stage II target since it's already in the F19 standing followup as DEFERRED in earlier work.

If pushback doc has gaps for Stage I purposes, supplement minimally — don't redo the investigation.

### CP I.3 — Touch cascade prioritization

For ALM, AVM, AIB, AVB specifically: extra scrutiny on whether published VC/CC actually exists. If primary sources paywalled or only have calcium imaging (no VC/CC), document the gap. This determines Stage IV feasibility.

### CP I.4 — Ranked candidate list

Output `wave2/artifacts/literature_scoping_candidates.md` with ranked list, recommended Stage II target list (top 4-8 cells), per-cell estimated cycle time.

### Stage I acceptance

- ≥4 strong candidates beyond AVAL/AIY/RIM
- Each candidate has primary source verified
- Touch cascade members prioritized if data exists
- Per-cell cycle-time estimates documented

### Stage I hard stops

- Fewer than 4 strong candidates: HARD STOP, write `OVERNIGHT_PAUSED.txt` summary, halt
- Mellem 2008 has no usable cell data: document, proceed
- Major paper paywalled: document gap, proceed

### State after Stage I

- `wave2/artifacts/literature_scoping_candidates.md`
- `wave2/artifacts/checkpoints/stage_I_status.json`

---

## Stage II: Targeted Wave 2 expansion (~4-6 hours, multi-invocation likely)

**Goal:** validate top candidate cells from Stage I as production-grade.

### Per-cell CP II.N

For each candidate cell from Stage I ranked list, in priority order:

1. **Pre-flight** for this cell: verify primary source accessible; verify VC/CC protocols; verify channel inventory matches expected; surface concerns BEFORE attempting translation
2. **NMODL translation** using F1-F19 catalog patterns (reuse existing channels from `wave2/channels/`; translate novel channels carefully; document new gotchas as F20+)
3. **Cell harness construction** templated from AVAL/AIY/RIM
4. **Validation:** VC + CC against NEURON reference; target 5-decimal-place agreement
5. **Adversarial review:** F18-style channel-conflation checks; citation-attribution discipline
6. **Verdict:** PRODUCTION_GRADE (5-decimal-place agreement) / DEFERRED (residuals can't be debugged in 2-3 hours)

**Per-cell hard stop conditions** (skip cell, document, continue):
- Primary source inaccessible
- NMODL translation surfaces patterns not covered by F1-F19 (novel pattern needs Rohit's review)
- Validation residuals stuck at >0.5 mV after 2-3 hours
- Brian2 ↔ NEURON setup environment issues

### Stage II hard stops (pause overnight)

- 0 cells validated (systemic harness issue)
- All cells produce significant residuals (methodology issue)
- Time budget exceeded by >50%

### Stage II success

- ≥1 additional cell production-grade: useful progress
- Ideally ≥3 additional cells production-grade: substantial expansion
- Ideal: touch cascade closure cells (ALM, AVM, AIB, AVB) production-grade

### State after Stage II

- New cell builders at `wave2/option_alpha_{cellname}_cell.py`
- New channels at `wave2/channels/` if needed
- `wave2/artifacts/stage_II_findings.md`
- F20+ patterns added to `wave2/translation_patterns.md` if surfaced
- `wave2/artifacts/checkpoints/stage_II_status.json`

---

## Stage III: Phase δ network integration (~3-4 hours)

**Goal:** execute Phase δ WB2-WB6 with expanded cell panel.

**Prerequisites:** Stage II shipped at least 1 additional production-grade cell. If 0, fall back to AVAL/AIY/RIM panel.

Reference: `wave2/artifacts/phase_delta_scoping.md` for architecture.

### CP III.1 — WB2: I_ext rename + first cell drop-in

Rename `I_inj` → `I_ext` across Wave 2 cells (production convention; per Phase δ WB1 findings). Re-run Wave 2 cell validation post-rename to confirm no regression. Drop AVAL into LIFBrain scaffold via Brian2 Network with multiple NeuronGroups + cross-group Synapses. Use placeholder release rule (V-threshold crossing) for now. Smoke test: spontaneous scenario for 10s, verify no errors, AVAL firing rate plausible.

### CP III.2 — WB3: Release-event rule design (PAUSE LIKELY)

Architecturally novel. **Pre-flight specifically for WB3:** if you find yourself making non-trivial choices about graded vs discrete release without primary source guidance, **PAUSE**.

Acceptable autonomous progress:
- V-threshold crossing rule with parameters from Mellem 2008 RMD or available primary source
- Graded Boltzmann release rule with Wicks 1996 parameters
- Run both on AVAL drop-in, document behavioral differences
- Output candidate parameter ranges per rule

NOT acceptable:
- Picking one rule as "the answer" without primary source justification
- Tuning to produce specific downstream phenotypes
- Choosing graded vs discrete based on pipeline convenience rather than biology

### CP III.3 — WB4: Multi-cell drop-in

All Stage II validated cells become NeuronGroups in Brian2 Network. Cross-group Synapses connect Wave 2 cells to remaining LIF cells per Cook 2019. Smoke test: spontaneous scenario for 30s simulated, verify no errors, all cells in plausible firing-rate ranges.

### CP III.4 — WB5: Layer A network validation

Run existing scenarios (spontaneous, touch, osmotic_shock, food, chemotaxis, aerotaxis) under expanded brain. Compare scenario JSON outputs against pre-Phase-δ baseline. Behavioral state distributions should be similar to LIF baseline within reasonable variance. Substantial differences may be Layer B biological enrichment manifesting (informative).

### CP III.5 — WB6: Layer B + Layer C validation

Layer B: touch cascade with expanded cellular layer (overlaps Stage IV). Layer C: namespace sanity at network scale.

### Stage III hard stops

- WB3 needs biological judgment beyond literature: pause for Rohit
- WB4 produces network-scale instabilities unresolvable in 1-2 hours
- WB5 Layer A divergence so large it suggests broken integration mechanics
- Time budget exceeded by >50%

### State after Stage III

- New brain class at `scripts/brain/network_brain.py` (or extension of existing)
- WB findings at `wave2/artifacts/phase_delta_wb{2,3,4,5,6}_findings.md`
- Updated scenario JSONs at `public/data/wormbody-brain-{scenario}-phase-delta.json`
- `wave2/artifacts/checkpoints/stage_III_status.json`

---

## Stage IV: Touch cascade biological validation (~1-2 hours)

**Goal:** test whether expanded cellular layer reproduces touch reversal cascade where pure LIF cannot (§5 falsification test).

**Prerequisites:** Stage III completed with touch cascade cells (ALM, AVM, AIB, AVAL, AVB if data was available) production-grade and integrated.

§5 baseline reference: `personalwebsite/docs/claude-chat-context.md` (orchestrator located it during pre-flight).

### CP IV.1 — Touch cascade firing-rate profiling

Run touch_anterior scenario under expanded brain. Profile per-neuron firing rates pre-touch (t=1-5s) vs peri-touch (t=5-7s):

- ALM/AVM: baseline ~1-2 Hz → peri-touch 50-80 Hz expected
- AIB: baseline ~5-10 Hz → peri-touch 20-30 Hz IF cascade propagating
- AVAL: baseline ~2-5 Hz → peri-touch ≥20 Hz IF cascade propagating
- AVB: baseline ~5-10 Hz → peri-touch decrease expected

Compare to v3 LIF baseline: ALM/AVM 1.7→78 (clean) but AVA decreased 36→28 (broken cascade).

### CP IV.2 — Behavioral state distribution under touch

Run touch scenario for 30s, compute state distribution. Compare:
- Pure LIF v3 + classifier-mode FSM: FWD 7%, REV 48%, PIR 30%, QUI 15%
- Pure LIF v3 + activity-mode FSM: QUI 91% (broken because AVA doesn't fire)
- Expanded brain + activity-mode FSM: ?

If expanded brain + activity-mode produces REV 30-50%, cascade is propagating biologically — major paper-2 result.

### CP IV.3 — Document findings

Output `wave2/artifacts/stage_IV_touch_cascade_findings.md`:
- Per-cascade-neuron firing-rate comparison (LIF baseline vs expanded)
- Behavioral state distribution comparison
- Verdict: cascade reproduced / partially reproduced / still broken
- Implications for §5 methodology finding

**Stage IV doesn't have hard stop conditions** — every outcome is informative.

### State after Stage IV

- `wave2/artifacts/stage_IV_touch_cascade_findings.md`
- `wave2/artifacts/checkpoints/stage_IV_status.json`

---

## Wake-up summary requirement

Single consolidated summary at `wave2/artifacts/overnight_run_summary_2026-04-28.md`:

1. **Final state:** which stages completed, paused, not started
2. **Stage I outcome:** cells identified, verified, ranked
3. **Stage II outcome:** cells attempted, cells validated production-grade, cells deferred with reasons
4. **Stage III outcome:** WB2-WB6 status, release rule decision (or pause), Layer A comparison
5. **Stage IV outcome:** touch cascade verdict, firing-rate comparisons, state distribution
6. **Methodology catches:** new F20+ patterns, citation issues, biological direction inversions
7. **Standing followups:** anything needing Rohit's attention
8. **Rock-solid changed-files list:** new code, artifacts, validations

Readable in 5 minutes when Rohit wakes. Headline finding first.

---

## Final framing

This is bounded multi-stage overnight work with clear pause points. Apply today's discipline:
- Pre-flight pushback at every stage boundary
- Primary-source verification before treating any citation as ground truth
- Don't fabricate when can't verify
- Pause-with-documentation > push-through

Single-agent-invocation envelope is realistically 1-3 hours. Multi-invocation continuation possible. The stage-boundary pause discipline naturally handles whatever envelope Rohit's overnight provides.

Standing by for pre-flight pushback (`OVERNIGHT_PUSHBACK.md` if any) or Stage I launch.
