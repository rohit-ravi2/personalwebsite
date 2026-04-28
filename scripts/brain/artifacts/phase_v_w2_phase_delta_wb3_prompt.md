# Phase δ WB3 — release-event rule design + implementation

**Mode:** Architecturally novel work block. Mandatory CP1 pause-for-review before implementation. Single session for pre-flight + CP1; subsequent invocations for CP2-CP6 after Rohit's release-rule adjudication.

**Strategic load:** WB3 is the gating dependency for both Phase δ multi-cell integration (WB4-WB6) AND Phase G LIFBrain integration (Session 2 thread). Both threads blocked on this decision.

**Critical:** CP1 ends with `PAUSED_FOR_REVIEW.txt`. Do NOT proceed to CP2 implementation work until Rohit explicitly authorizes the chosen rule via subsequent invocation. Even if pre-flight + CP1 complete quickly, halt at CP1's pause gate.

---

## Context: empirical evidence demanding this work block

1. **F20 capacitance mismatch (WB2 finding):** Wave 2 cells have biological cm ~0.86 pF; LIF cells use cm ~100 pF. Naive `v += W_syn * w` cross-group coupling structurally unstable.
2. **Three biology questions paused for review** in `wave2/artifacts/phase_delta_wb2_findings.md`. Agent correctly didn't auto-pick.
3. **Phase G dose-response gap (Session 2):** 100× tighter behavioral EC50; release rule choice has downstream consequences for behavioral calibration.

WB3 resolves the architectural question with biological grounding, implements chosen rule, validates numerical stability, and produces substrate that WB4-WB6 can consume.

---

## Out of scope for WB3

- WB4 multi-cell drop-in (separate work block, post-WB3)
- WB5 Layer A network validation
- WB6 Layer B + Layer C validation
- Phase G LIFBrain integration (Session 2 territory; runs after WB3)
- Bilateral pair completion (AIYR, RIMR)
- Body v3 work
- Path B engineering

If WB3 surfaces architectural questions exceeding scope (e.g., release rule depends on muscle output coupling), document and surface for cross-session decision rather than expanding scope autonomously.

---

## Working environment

- Brain code: `~/Desktop/website/personalwebsite/scripts/brain/`
- Wave 2 work: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Wave2HybridBrain scaffold: `wave2/integration/wave2_hybrid_brain.py` (currently runs `cross_coupling="off"`)
- WB2 findings (load-bearing source): `wave2/artifacts/phase_delta_wb2_findings.md`
- Phase δ scoping doc: `wave2/artifacts/phase_delta_scoping.md`
- Wave 2 venv: `~/venvs/wave2-neuron/`
- Cython is production default (per WB1 unification)

---

## Pre-flight pushback expected (substantial)

Before proposing any release rule, do thorough verification:

1. **Read WB2 findings (`phase_delta_wb2_findings.md`) in full.** The three biology questions are the authoritative scope. Verify what's actually being asked. Don't paraphrase from memory or context summaries.

2. **Read Phase δ scoping doc release-event rule section.** Verify the V-threshold and graded Boltzmann options are still operative or whether WB2's empirical findings changed the option space (capacitance mismatch may make V-threshold non-viable; full conductance-based may need to be a third option).

3. **Verify primary sources for candidate release rules:**
   - Wicks JF, Roehrig CJ, Rankin CH 1996 *J Neurosci* "A dynamic network simulation of the nematode tap withdrawal circuit" — actual sigmoidal release equation, V_half + k values, neuron classes covered (AVA-class via tap-withdrawal circuit; AIY/RIM not in this paper).
   - Goodman MB, Hall DH, Avery L 1998 — graded vs spiking C. elegans evidence.
   - Lockery 2009 / Lindsay 2011 — subsequent characterization if relevant.
   - Faumont 2011, Nicoletti 2019 — plateau dynamics.

4. **Verify capacitance mismatch is what WB2 surfaced.** The 100× scale difference is physical (Wave 2 cells biologically realistic ~0.86 pF; LIF cells use ~100 pF Brian2 default). Confirm not a units bug.

5. **Identify what biological judgment is genuinely needed vs what can be derived from primary sources.** If Wicks specifies V_half + k for AVA-class, that's primary-source-derived. If parameterization needed for cells where Wicks didn't report (AIY, RIM), that's biological judgment. Be explicit.

6. **Verify Brian2 supports the proposed implementation under cython.** Cross-group Synapses with `(summed)` continuous coupling are standard Brian2 idiom but verify the specific pattern compiles cleanly under cython.

If pre-flight surfaces concerns about scope, biological ambiguity, or implementation feasibility, surface to `wave2/artifacts/phase_delta_wb3_pushback.md` and pause for review.

**The pre-flight should produce primary-source-grounded options document.** This is preparation for human decision-making, not autonomous work.

---

## CP1 — Release rule options document (LOAD-BEARING; ENDS IN PAUSE-FOR-REVIEW)

Write `wave2/artifacts/phase_delta_wb3_release_rule_options.md` with structure:

### Section 1 — Empirical constraints from WB2

- Capacitance mismatch quantified (Wave 2 ~0.86 pF, LIF ~100 pF, ratio ~116×)
- Why naive `v += W_syn * w` fails (specific failure mode)
- What numerical-stability requirements the release rule must satisfy

### Section 2 — Biological constraints from primary sources

- C. elegans graded transmission (Goodman/Hall/Avery 1998 evidence summary, with direct quotes)
- Wicks 1996 sigmoidal release equation with reported parameters
- Plateau potential dynamics (Mellem 2008 RMD, Faumont 2011, Nicoletti 2019/2024) — voltage range producing release
- Neuron-class-specific data: AVA plateau properties, AIY graded properties, RIM dynamics
- Citation discipline applies — no fabrication. Direct quotes for load-bearing claims.

### Section 3 — Three candidate release rules with explicit tradeoffs

**Option A: V-threshold crossing (discrete spike-event approximation)**
- Implementation: TimedArray monitoring presynaptic V; threshold crossing emits Brian2 spike
- Biological grounding: weak (graded biology approximated as discrete)
- Numerical stability: depends on per-cell-type calibration of spike-bump magnitude
- Implementation simplicity: highest (existing Brian2 spike-event infrastructure)
- Honest assessment of biological cost

**Option B: Graded Boltzmann release (Wicks 1996 sigmoidal)**
- Implementation: continuous synaptic conductance ∝ σ(V_pre - V_half); Brian2 cross-group Synapses with `(summed)` continuous coupling
- Biological grounding: strong (matches graded biology)
- Numerical stability: handles capacitance mismatch naturally
- Implementation complexity: moderate
- Parameter requirements: V_half, k from Wicks; per-cell-class params where Wicks didn't specify

**Option C: Full conductance-based synaptic dynamics**
- Implementation: HH-style synaptic conductance with kinetics (τ_rise, τ_decay, peak g per receptor type)
- Biological grounding: strongest (matches molecular biology where data exists)
- Numerical stability: handles mismatch + produces realistic synaptic currents
- Implementation complexity: highest
- Parameter requirements: substantial; some receptor types lack published kinetics

### Section 4 — Recommendation with rationale

- Which option pre-flight recommends and why
- Primary-source data supporting recommendation
- Tradeoffs accepted honestly
- What's NOT in recommendation's scope

### Section 5 — Decisions requiring biological judgment vs derivable

- Specific decisions Rohit needs to make
- Per-decision: choice consequences
- Per-decision: default if no strong preference

### MANDATORY PAUSE GATE

After CP1 options document is written:

1. Write `wave2/artifacts/PAUSED_FOR_REVIEW.txt` with content explaining what's awaiting review (release rule choice + parameter decisions)
2. Send `~/bin/notify` notification: "WB3 CP1 options doc complete; awaiting Rohit's release-rule adjudication"
3. **HALT.** Do not proceed to CP2 implementation work even if runtime envelope remains. The release rule choice is not autonomous.

The next agent invocation (after Rohit's authorization) handles CP2-CP6.

### CP1 acceptance criteria

- Options document at `wave2/artifacts/phase_delta_wb3_release_rule_options.md`
- All primary sources verified (not paraphrased from memory)
- Three options with explicit tradeoffs
- Recommendation with rationale
- Decision points flagged for human input
- PAUSED_FOR_REVIEW.txt marker written
- Halt at pause gate

### CP1 failure modes

- Primary sources not accessible: pause, document, surface specific gaps
- Biological ambiguity exceeds primary-source resolution: pause, surface specific questions
- Implementation feasibility uncertain: pause, prototype small Brian2 test before full options doc
- WB2 findings reveal additional constraints: surface and incorporate

---

## CP2-CP6 (NOT executed in this invocation; documented for reference)

These execute in subsequent invocation(s) after Rohit's release rule adjudication.

**CP2 — Implementation of approved release rule:**
- Modify Wave2HybridBrain to support `cross_coupling="<approved_rule>"` mode
- Cross-group Synapses connecting Wave 2 NeuronGroups to LIF NeuronGroups per Cook 2019 connectome
- Apply approved rule equations both directions (Wave 2 → LIF graded; LIF → Wave 2 via chosen rule)
- Reuse Brian2 idioms; cython codegen; per-NeuronGroup clock keyword for dt mismatch (Wave 2 0.025ms, LIF 0.1ms)
- Numerical stability checks; don't tune parameters to produce specific phenotypes

**CP3 — Numerical stability validation:**
- 1s smoke test (spontaneous, no stim): biological voltage range, no NaN/Inf, no runaway firing
- 10s smoke test: stable cross-group propagation
- 30s smoke test: spontaneous + touch stim at t=5s; cascade propagation observed; system remains stable

**CP4 — Touch cascade validation under cross-coupled brain:**
- 30s touch_anterior scenario
- Profile per-cell firing rates pre-touch (1-5s) vs peri-touch (5-7s)
- Cascade neuron checks (ALM/AVM, AIB, AVAL/AVAR, AVB)
- Compare to per-edge LIF baseline AVA Δ+7.5 Hz
- Verify Wave 2 mechanistic resolution (AVAL ≠ AVAR distinguishability)
- Compare behavioral state distribution

**CP5 — Document WB3 outcomes + F20 catalog entry:**
- `wave2/artifacts/phase_delta_wb3_findings.md`: release rule chosen + rationale, implementation, validation outcomes, F20 resolution, what's now ready for WB4 + Phase G
- Update `wave2/translation_patterns.md` with F20 entry

**CP6 — Commit + push:**
- Group A: CP1 options document
- Group B: CP2 implementation
- Group C: CP3+CP4 validation outputs
- Group D: CP5 findings + F20 catalog update
- Honest commit messages with primary-source-disciplined references

---

## Failure modes and recovery

- CP1 surfaces fundamental biological ambiguity primary sources don't resolve: pause, surface for Rohit's input
- Pre-flight implementation feasibility check fails: pause, document, propose alternatives
- Primary source access fails: pause, document gap, propose proceeding without (with caveats) or surface for review

**General principle:** WB3 is load-bearing for both threads' forward work. Pause-with-documentation > push-through. Honest documentation of biological grounding vs engineering judgment > clean-looking implementation hiding ambiguity.

---

## On time scoping

- Pre-flight + CP1 options document: ~2-3 hours (primary source verification time-intensive)
- **MANDATORY PAUSE FOR ROHIT REVIEW AT CP1**

CP1 fits within single-agent-invocation envelope. No multi-invocation chaining needed at this stage — chaining begins at CP2 deployment after release-rule adjudication.

---

## Deployment scope for THIS invocation

This invocation does ONLY:
1. Pre-flight pushback (verify WB2 findings, scoping doc, primary sources, capacitance mismatch, Brian2 implementation feasibility)
2. CP1 options document
3. Pause-for-review marker + notification + halt

This invocation does NOT do CP2 implementation, CP3-CP6 work, or any modification to `wave2_hybrid_brain.py` beyond reading it. Implementation is post-authorization.

Standing by for pre-flight pushback or CP1 options document completion.
