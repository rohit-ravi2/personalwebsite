# Mellem 2008 + Liu 2018 mechanistic investigation

**Mode:** literature investigation, not engineering. Bounded ~1-2 hours focused work. Decision-grade output for whether Wave 2's morphology fork is sufficient or needs additional mechanism translation work.

---

## Context: why this work block exists

Phase β overnight run #2 produced empirical confirmation of condition 6 (single-compartment AVA architecture insufficient for Mellem 2008 plateau dynamics). Density-sensitivity sweep + Ca-pool integration test together robustly excluded "wrong densities" and "missing dynamic Ca-pool" as explanations. Both surfaced the same mechanistic conclusion:

**SLO-1 in single-compartment AVA is hyperpolarizing — the Ca-coupling loop is negative feedback for V. It can only terminate the plateau, never extend it. Across 5 orders of magnitude of Ca-influx scaling, plateau duration stays at 21 ms or shorter.**

The Ca-pool integration agent's mechanistic conclusion: Mellem 2008's 600 ms plateau requires either (a) multi-compartment morphology providing spatial separation, (b) CICR (calcium-induced calcium release) providing positive Ca-feedback, or (c) persistent inward currents with Ca-dependent activation. None of (a-c) are accessible without architectural escalation.

The architectural plan's pre-committed condition 6 response is the morphology fork (path a). But if Mellem identifies the mechanism as (b) or (c), morphology fork alone is insufficient.

This investigation resolves the spatial-vs-mechanistic question by reading the primary literature directly.

---

## Strategic positioning

Bounded literature investigation, not engineering work. Produce decision-grade output that informs the next Wave 2 commitment. Estimated scope ~1-2 hours focused work.

Dual purpose:
1. **Architectural decision input** — determines whether morphology fork is sufficient or additional work needed
2. **Paper 3 background work** — manuscript will need to address what mechanisms were/weren't reproduced and why; this investigation produces the framing material

No additional cost for the dual deliverable.

---

## Working environment

- Project at `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Phase β run #2 outputs at `wave2/artifacts/` for context
- Architectural plan at `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md`
- Run #2 findings catalog at `wave2/artifacts/phase_beta_findings.md` (F1-F17)

Recommended reads before starting:
- `wave2/artifacts/phase_beta_run2_summary.md`
- `wave2/artifacts/gate2_ava_cell_construction.md`
- `wave2/artifacts/density_sensitivity_analysis.md`
- `wave2/artifacts/ca_coupling_test_results.md`

---

## Pre-flight pushback expected

Read this prompt fully. If you find scope concerns or hidden assumptions, surface before starting. If pre-flight surfaces no concerns, proceed.

---

## Investigation scope

### Primary 1: Mellem 2008

Mellem JE, Maricq AV (2008) — verify exact citation. The architectural plan and Phase α/β prompts reference this as the source for AVA plateau dynamics targets (15-25 mV amplitude, 400-800 ms duration; target 20 mV / 600 ms). Verify the citation is correct (the Nicoletti precedent showed citation errors are real risks). The candidate is likely "Action potentials contribute to neuronal signaling in C. elegans" (J Neurophysiol) but verify.

If "Mellem 2008" turns out to be a misattribution similar to the Nicoletti 2019 PCBI/PLOS ONE conflation, the condition-6 target may need reassessment — flag this as a load-bearing finding.

Specific questions:

1. **Experimental protocol:** cell preparation, recording method, current injection parameters, drug conditions, temperature, ionic conditions. Compare against our simulation protocol (50 pA × 100 ms at v_rest = -25 mV).
2. **Proposed mechanism:** Mellem must hypothesize what biological machinery sustains the 600 ms plateau. Document with specificity. Watch for: voltage-gated Ca channels, Ca-induced Ca release from internal stores, persistent Na currents, NMDA-like receptors, dendritic vs axonal compartmentalization, gap junctions to other cells.
3. **Pharmacology:** which channels were blocked or activated to dissect the mechanism? Specifically: did Mellem block SLO-1 to see if plateau persisted? Did they block EGL-19? Did they block intracellular Ca release? The pharmacological dissection identifies which channels are necessary vs sufficient.
4. **Spatial profile:** single-cell recording or compartment-specific? Did they distinguish soma vs neurite contributions? Morphology question depends on this.
5. **Experimental data reported:** voltage traces, IV curves, KO comparisons, dose-response. Document what's in figures vs supplementary materials.

### Primary 2: Liu 2018

Liu et al. 2018 — Nicoletti's experimental source for AVA modeling per F12. Verify exact citation; likely "Postsynaptic current bursts instruct action potential firing at a graded synapse" (the agent's run #2 surfaced this attribution).

Specific questions:

1. **Channel currents characterized:** Liu presumably published voltage-clamp recordings of specific currents (EGL-19 Ca current, SLO-1 K current, etc.). Document which currents and at what protocols.
2. **Plateau dynamics characterized?** Or only voltage-clamp components? Mellem reports plateaus; Liu may report only individual channel currents that compose the plateau.
3. **Relationship to Mellem 2008:** same lab, different lab? Same cell preparation, different conditions? Different worm strain?
4. **Plasma membrane channels in AVA:** what does Liu identify? Candidate set for "essential channels" — what Nicoletti included vs what biology has.

### Primary 3: Architectural plan's referenced cellular biophysics literature

Cross-check Mellem and Liu against:
- Wicks 1996 (referenced in architectural plan as plateau mechanism source)
- Lockery & Goodman 2009 (graded transmission C. elegans review)
- Goodman, Hall & Avery 1998 (worm neurons don't fire vertebrate-style action potentials)
- Faumont et al. 2011 (PVC, RIM characterizations)

Specific questions:

1. **Does Wicks 1996 propose a specific plateau mechanism?** If yes, document.
2. **Field consensus on C. elegans command neuron plateau mechanism?** If yes, document. If no, document the open question.
3. **CICR in C. elegans neurons?** Specifically AVA. Evidence for/against IP3R/RyR-mediated Ca release from internal stores in command neurons. Determines mechanism (b) viability.
4. **Persistent inward currents in C. elegans neurons?** Evidence for CCA-1 or other currents that could sustain depolarization. Determines mechanism (c) viability.

### Supporting literature as needed

If primary papers reference specific mechanisms or mechanisms remain unclear, follow citations to support papers. Don't expand exhaustively — investigate enough to answer the architectural decision question.

Specifically check if relevant:
- Recent (2023-2026) papers on AVA biophysics that may postdate Nicoletti 2024
- Reviews on C. elegans command neuron physiology
- Methods papers describing the specific experimental protocol

---

## Output: classification + decision tree synthesis

Single output document: `wave2/artifacts/mellem_mechanism_investigation.md`

### Structure

**1. Verified citations and access status**

Mellem 2008 full citation, DOI, accessibility (PDF available where, supplementary materials accessible). Same for Liu 2018, Wicks 1996, Lockery & Goodman 2009, Faumont 2011, Goodman/Hall/Avery 1998.

**If "Mellem 2008" citation verification surfaces a misattribution: STOP and surface immediately.** This would invalidate the condition-6 target itself.

**2. Mellem 2008 mechanism analysis**

Per-question answers from Primary 1. Direct quotes for mechanism hypotheses, with figure references.

**3. Liu 2018 framework analysis**

Per-question answers from Primary 2. What Liu characterized, currents involved, relationship to Nicoletti's model.

**4. Cross-literature consensus check**

What does the field agree on about C. elegans command neuron plateau mechanism? Which references support which mechanisms?

**5. CLASSIFICATION VERDICT**

The load-bearing output. One of four categories with explicit reasoning:

**Classification A: SPATIAL_MECHANISM_DOMINANT**

Mellem's mechanism is fundamentally spatial — multi-compartment morphology with separate dendritic/axonal Ca dynamics, somatic-process voltage gradients, similar. Morphology fork directly addresses this.

Evidence pattern: Mellem's pharmacology shows plateau requires intact cellular architecture; spatial pharmacology (e.g., distance-dependent effects, dendritic blockers); imaging shows distributed Ca dynamics.

Recommended Wave 2 response: trigger morphology fork (Phase β-morph + Phase γ-morph) per architectural plan. Plateau dynamics expected to emerge from compartmentalization without additional mechanism translation work.

**Classification B: MECHANISTIC_INGREDIENT_DOMINANT**

Mellem's mechanism is fundamentally mechanistic — CICR, persistent inward currents with Ca-dependent activation, NMDA-like receptors, or other channels/mechanisms not in Nicoletti's 7-channel essential set. Morphology fork alone insufficient.

Evidence pattern: Mellem's pharmacology shows specific blockers (e.g., ryanodine, IP3R blocker, NMDA blocker, persistent Na blocker) that abolish plateau without affecting general cell health; reference to specific channels not in Nicoletti's set.

Recommended Wave 2 response: identify specific mechanism, scope a mechanism-translation work block (similar to Phase β channel translation but for the missing mechanism). Possibly run in parallel with morphology fork, possibly defer morphology fork until mechanism is added and re-validated.

**Classification C: COMBINATION_REQUIRED**

Both spatial and mechanistic ingredients needed. Morphology + missing mechanism translation must both occur.

Evidence pattern: Mellem's pharmacology shows multiple blockers each partially affect plateau; multi-compartment + specific mechanism both implicated; or no single intervention abolishes plateau.

Recommended Wave 2 response: sequence morphology fork first (cheaper relative to introducing new channels we'd need to validate), then evaluate whether plateau dynamics emerge or whether additional mechanism translation is still needed. Iterative gate evaluation.

**Classification D: UNCLEAR_OR_DIFFERENT_QUESTION**

Mellem's mechanism is unclear, contested, or not directly characterized. Or: Mellem's experimental setup differs from our simulated setup in ways that make the comparison itself questionable (different temperature, drug conditions, cell preparation, worm strain).

Evidence pattern: Mellem reports phenomenology without mechanistic dissection; contradictions between Mellem and other papers; experimental differences from simulation that aren't accounted for.

Recommended Wave 2 response: deeper investigation needed before architectural changes. Possibly: re-investigate whether our simulation protocol matches Mellem's experimental conditions; possibly: consult additional literature; possibly: scope down Wave 2 to behavioral closure and defer cellular-level mechanism reproduction.

**6. Decision tree for Wave 2 trajectory**

Based on classification verdict, recommend specific Wave 2 next moves:

- Phase β-morph (morphology fork) authorization or deferral
- Specific additional mechanism translation work blocks if Classification B or C
- Modified validation criteria if Classification D
- Paper 3 manuscript implications

**7. Implications for paper 3 manuscript**

Wave 2 paper will need to address what mechanisms are and aren't reproduced. Document what the literature establishes about Mellem's mechanism so the manuscript discussion section can address this transparently.

If Wave 2's final architecture reproduces Mellem partially but not fully, paper needs to identify which mechanisms are reproduced and which are deferred to future work. This investigation produces the framing material for that discussion.

---

## Scope discipline

What's NOT in this work block:

- Implementation of any architectural changes (investigation only)
- Code modification (no engineering work)
- Decision-making on Wave 2 architectural commitments (this work produces input, doesn't make decision)
- Exhaustive review of C. elegans neurophysiology literature (focus on architectural decision question)
- Investigation of mechanisms unrelated to Mellem 2008

If literature review surfaces interesting findings beyond the architectural decision, document briefly in a "tangential findings" section but don't expand investigation depth.

---

## Methodology continuity

Pre-flight pushback discipline. Stop-and-ask if findings surface architectural questions worth cross-session discussion. Mid-flight surfacing of significant findings to `wave2/artifacts/mellem_investigation_findings.md` if you discover load-bearing items mid-investigation.

If Mellem's paper says something inconsistent with what we've assumed about AVA biology, surface that explicitly. If the literature is contested or the consensus differs from what the architectural plan assumed, surface that.

---

## On time scoping

Estimated ~1-2 hours focused work. Don't compress to fit a felt time budget. If Mellem's paper takes longer to read carefully than expected, take the time. Investigation quality matters more than completion speed.

If you find investigation requires substantially more work than 1-2 hours (e.g., Mellem's mechanism is unclear and requires reading 5 supporting papers), surface and discuss whether to expand scope or accept partial findings.

---

## Output format

Single output: `wave2/artifacts/mellem_mechanism_investigation.md` per the structure above.

If supporting findings warrant separate documentation (e.g., a digitized figure from Mellem's paper for comparison against our simulation), save to `wave2/artifacts/mellem_supporting/` directory.

If pre-flight pushback surfaces concerns: `wave2/artifacts/mellem_investigation_pushback.md` with `PAUSED_FOR_REVIEW.txt` marker.

---

## Final framing

This investigation determines what Wave 2's continuation looks like. The morphology fork commitment is consequential (3-4 weeks of work) and its appropriateness depends on whether Mellem's mechanism is spatial, mechanistic, or combination. Cheap probe before expensive commitment is the methodology pattern that's been load-bearing across today's work.

The investigation also produces background material for paper 3. Doing it now while engineering context is fresh produces better material than retrofitting later.

Other sessions remain idle. This is single-session focused investigation.

Standing by for completion notification or pre-flight pushback.
