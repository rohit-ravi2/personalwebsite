# Phase V Wave 2 — Architectural Plan (Phase 4 synthesis)

**Inputs:** Phase 1 existing-data inventory (`phase_v_w2_existing_data_inventory.md`), Phase 2 simulator landscape (`phase_v_w2_simulator_landscape.md` for 2A+2B; this document for 2C+2D), Phase 3 backend architecture analysis (`phase_v_w2_backend_architecture_analysis.md`).

**Output:** Wave 2 architectural commitment recommendation, sequencing, gates, risks, paper-trajectory connections, and "what would invalidate Path A?" subsection.

**Primary upstream citations (verified, both used in Wave 2):**

- **Nicoletti M, Loppini A, Chiodo L, Folli V, Ruocco G, Filippi S (2019)** — "Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD." *PLOS ONE* 14(7): e0218738. DOI `10.1371/journal.pone.0218738`. PMID 31260485. ModelDB 267187. **Role:** upstream paper introducing the AWCon/RMD channel formulations that Nicoletti 2024 extends. Wave 2 references this paper for channel-library provenance and for AWCon/RMD-specific kinetics.
- **Nicoletti M, Chiodo L, Loppini A, Liu Q, Folli V, Ruocco G, Filippi S (2024)** — "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families." *PLOS ONE* 19(3): e0298105. DOI `10.1371/journal.pone.0298105`. PMID 38551921. ModelDB 2017403. **Role:** primary Wave 2 import target. 22-channel library + 9 validated cell models (AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 + AWCon/RMD from 2019). Wave 2 channel-translation work targets this paper's `.mod` files.

**Citation correction note (v3, 2026-04-26):** prior versions of session prompts and v1 artifacts referenced `10.1371/journal.pcbi.1007611` as "Nicoletti 2019 PLOS Comp Bio." That DOI is incorrect — it resolves to a glioma paper unrelated to C. elegans (Jamous et al., "Self-organization in brain tumors"). Both papers above are real, verified, and by the same group; the spec preamble that originated the wrong DOI conflated them. v3 corrects this across the architectural plan and active session prompts; v1 historical artifacts (`published_traces.json`, `digitize_panels.py`) are preserved unchanged as historical record of the v1→v2 detection of the citation error.

---

## Executive summary

Path A as committed (import + integrate existing implementations) remains the right strategic move based on Session B investigation. The investigation surfaced three findings that sharpen execution:

1. **Backend choice = Path 3A (Brian2 + parameter import).** Translate Nicoletti's 22 NEURON `.mod` channels into Brian2 equation form. Bounded ~44-88 hours of translation. Preserves all existing infrastructure. Brian2's compute advantage matches RTX 4060 Ti constraint. Per-channel additive with rollback.

2. **Acquired packages provide a complete-enough foundation.** c302 framework (cell morphologies + connectome readers + network templates) + ChannelWorm models (4 NeuroML channels) + Nicoletti 2024 (22 NEURON channels + 9 validated neurons + protocol implementations) + BAAIWorm (full multi-compartment 302-cell network in NEURON, integrated body/environment, but uses subset of Nicoletti's channel diversity). Acquisition complete; total disk ~800 MB.

3. **Path A doesn't solve gaps the project must fill itself.** Receptor binding kinetics, CeNGEN-coupled densities, modulator layer refinement, anesthesia-specific framework, per-cell biophysics for non-Nicoletti neurons. These are Wave 2+ work regardless of backend.

The Wave 2 architectural commitment: **Path 3A backend + per-channel translation prioritizing the 6 most load-bearing channels first + preservation of all existing project infrastructure.** Estimated 1-2 months of focused work to first usable deployment.

---

## Sub-phases 2C + 2D summary

### 2C — BAAIWorm (Liang et al. 2024 → corrected: Zhao et al. 2024 Nature Computational Science)

**Lead authors are Zhao/Wang/Jiang at Peking University, NOT Liang** (the previous audit's reference was incorrect). The bioRxiv preprint name was MetaWorm; the Nature Computational Science publication name is BAAIWorm.

**Repository:** `github.com/Jessie940611/BAAIWorm` (Apache 2.0).
**Components:**
- `eworm/` — neural network model (multi-compartment, 302 cells, NEURON `.mod` mechanisms)
- `Metaworm/` — body + environment (96-muscle FEM, 3341 tetrahedra, fluid dynamics)
- `neuronXcore/` — 3D visualization
- `eworm_learn/` — parameter learning (optimization to fit neural data)

**Channel inventory:** 14 NEURON `.mod` files + 6 synaptic mechanisms.
- Ca channels: cca1, egl19, unc2 (3) — same as Nicoletti
- K voltage-gated: egl2, egl36, irk, kqt3, kvs1, shk1, shl1 (7) — Nicoletti has these PLUS kqt1, unc103, exp2 (3 more)
- K Ca-activated: kcnl, slo1_egl19, slo1_unc2, slo2_egl19, slo2_unc2 (5) — Nicoletti has these PLUS slo1iso + slo2iso isolated variants
- Passive: nca, leak (2)

BAAIWorm channel set is a subset of Nicoletti 2024's 22 channels (BAAIWorm has 14, Nicoletti has 22). However BAAIWorm has synaptic mechanisms (`exc_syn_advance.mod`, `inh_syn_advance.mod`, `gapjunction_advance.mod`) that Nicoletti doesn't (since Nicoletti tests bare cells, not networks).

**Compute requirements:** Ubuntu 20.04, **Nvidia 3090, CUDA 11.4, Boost 1.79.** Heavier than this project's RTX 4060 Ti can comfortably run (3090 has more VRAM and SM count). BAAIWorm's full-network compute scale exceeds available compute for this project.

**Validation:** "Faithfully reproduces zigzag movement towards attractors observed in C. elegans." Demonstrates closed-loop integration of brain-body-environment.

**Strategic implication:** BAAIWorm is more like a competing/complementary project than something to import wholesale. It's not architecturally compatible with the project's compute budget. **However its synaptic mechanism `.mod` files are useful reference** for receptor binding kinetics if/when the project implements them. And its parameter-tuning approach (`eworm_learn`) provides methodology reference for fitting parameters to data.

### 2D — Other simulators (bounded landscape)

**PyNN** — multi-backend API (NEURON, NEST, Brian2, neuromorphic hardware). Could provide future portability layer. Adds abstraction overhead. Not immediately necessary for Path 3A; consider if Wave 3 portability becomes a goal.

**NetPyNE** — high-level declarative wrapper around NEURON. Active development. JSON-like model specification. Useful if Path 3C (NEURON backend) is ever pursued; not relevant for Path 3A.

**Brian2GeNN / Brian2CUDA** — Brian2 GPU acceleration paths. **Relevant for Wave 2+ if compute scales beyond CPU comfort.** For 302-cell simulator at 60-second runs, Brian2 + numpy CPU is currently sufficient; Brian2GeNN becomes interesting if running 100s of seeds in parallel or simulating longer durations.

**NEURON benchmark finding:** "BrainPy and Brian2 demonstrate comparable performance, showcasing a remarkable speed advantage of one to two orders of magnitude over NEURON and NEST" on CPU. **This is a load-bearing constraint for backend choice** — Path 3C (NEURON backend) would be substantially slower than current Brian2 setup, and Path 3A's translation work is justified by the resulting Brian2 performance.

**NeuroSimWorm 2025** (Wang/Liang/Tang, Neurocomputing) — newer C. elegans multi-sensory simulation framework with chemical/mechanical/thermal stimuli and closed-loop integration. Less mature than BAAIWorm. Code availability not directly verified. Worth tracking but not Wave 2-relevant.

**modWorm** — modular integration approach. Less mature. Not Wave 2-relevant.

**OpenWorm broadly** — covered in 2A.

---

## Wave 2 primary path (Path 3A execution)

### Week-by-week sketch (no calendar commitments; sequencing logic only)

**Phase α — Setup (week 1)**

- Install NEURON locally (`pip install neuron`) so Nicoletti's `.mod` files can be compiled and run for validation comparison
- Compile Nicoletti's 24 `.mod` files via `nrnivmodl` in `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- Run her 9 simulation scripts (AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 + AWC, RMD from 2019 work) and capture reference voltage-clamp + current-clamp traces
- Verify Brian2 unit handling for the channel parameter space (mV, ms, S/cm², µM)
- Set up validation harness that runs both Nicoletti's NEURON code and the project's Brian2 code with identical protocols, computes trace divergence

**Phase β — Channel translation (weeks 2-6)**

Translate channels in priority order:

1. **EGL-19** (replaces current handcrafted m_Ca + h in `graded_brain_h_kca.py` and `compartmental_neurons_kca.py`). Validate against Nicoletti's voltage-clamp IV curves and current-clamp plateau dynamics.
2. **SLO-1** with EGL-19 coupling (`slo1egl19.mod`). This addresses the plateau termination gap Wave 1 surfaced. Validate against Nicoletti's KO simulations (slo-1 KO should show prolonged plateau).
3. **SLO-1 isolated** (`slo1iso.mod`) — for cells without specific Ca channel coupling.
4. **SHK-1** (Kv1 delayed rectifier; Wang 2001 rich worm-specific data).
5. **SHL-1** (Kv4 A-type; Fawcett 2006 worm-specific data).
6. **NCA** (Na+ leak / NALCN homolog; central to baseline excitability).
7. **KQT-3** (M-current; behavioral state regulation).

After 7 channels, the "essential set" is in place. Each channel translation:
- Read Nicoletti's `.mod` file + understand kinetic scheme
- Translate equations to Brian2 namespace + equation strings
- Match parameter values exactly
- Run identical voltage-clamp / current-clamp protocol in Brian2
- Compare traces: both should overlay within 5% on V trajectories
- If divergence > 5%, debug; usually traces back to unit conversion or an NMODL idiom not handled cleanly

**Phase γ — Cellular validation (week 7)**

With 7 channels in Brian2, run Gate 2's two components (see Acceptance criteria below):

- **2a (voltage-clamp trace correctness):** Brian2 channels match Nicoletti's NEURON reference within 5% across IV-curve protocol — channel kinetics correct
- **2b (current-clamp plateau dynamics):** AVAL reproduces Mellem 2008 plateau (20 mV / 600 ms) and SLO-1-dominated termination on stimulus release — architecture sufficient for full dynamic range

The two components empirically distinguish channel-translation failures (2a fail → per-channel rollback) from architectural insufficiency (2a pass + 2b fail → condition 6 invalidation signature → fork to morphology integration).

This is the gate for declaring Path A's cellular layer "production-grade." Outcome determines Phase δ vs morphology-integration fork.

**Phase δ — Network integration (weeks 8-9)** *(if Phase γ passes both 2a and 2b)*

- Replace `graded_brain_h_kca.py`'s handcrafted Ca-K dynamics with imported channel set for the 14 plateau cells
- Validate that network-level scenarios still run (touch, food, osmotic_shock, etc.)
- Compare phenotypes against current LIFBrain + sample_004 baseline

**Phase ε — Remaining channels (weeks 10-12)**

Add the 15 channels not in the essential set as cell-by-cell calibration requires. Most won't be load-bearing for the cells the project is most interested in; add lazily.

### Conditional fork: morphology integration as Phase β prerequisite

The above sequencing assumes Phase γ passes both 2a (channels correct) and 2b (architecture sufficient). If Phase γ surfaces 2a-pass / 2b-fail (condition 6 signature), Wave 2 sequencing forks:

**Fork branch — morphology integration first:**

- **Pause Phase β/γ channel translation work** at the channels already validated for kinetics in 2a. Their work is preserved; the channel layer is correct, the scaffold is the bottleneck.
- **Phase β-morph (estimated 2-3 weeks):** integrate c302's 607 cell morphologies (already locally cloned at `~/Desktop/C-Elegans/simulation/upstream/c302/`) into the project's compartmental scaffold. Morphologies are NeuroML2 format with full soma + axon segment definitions. Integration touches: scaffold initialization, segment-level state vectors, segment-level channel placement, segment-level coupling (axial resistance, gap-junction-like compartmental coupling).
- **Phase γ-morph (estimated 1 week):** re-run Gate 2 with imported channels in proper compartmental scaffold. The expectation is 2b passes because compartmental architecture provides the dendritic/axonal state separation that single-compartment cannot.
- **Resume Phase β/γ** for the remaining channels in the 7-channel essential set. Resume Phase δ (network integration) and Phase ε (remaining 15 channels) as planned.

**Why the fork is empirically triggered, not discretionary:**

Wave 1 surfaced that single-compartment leak τ ≈ 10 ms overwhelms L-type Ca by 5×. Importing better channel kinetics doesn't change leak τ or the τ_d ≈ 20 ms dendritic ceiling. If 2b fails, the channel work is not the path — the morphology work is. The fork makes this empirically detectable rather than letting "channels validated, plateau still wrong" silently pass through to network integration.

**Why the fork doesn't affect Phase α scope:**

Phase α (NEURON install + Nicoletti compile + voltage-clamp validation harness) proceeds identically regardless. The morphology question only affects Phase β/γ sequencing once Gate 2 results are in hand.

**Estimated fork cost:**

3-4 weeks added to Wave 2 timeline if fork triggers. Assets are in hand (c302 morphologies locally cloned); the work is integration, not acquisition. Per-cell rollback applies if compartmental integration breaks specific cells — keep handcrafted scaffold for those, advance compartmental for cells that benefit.

### Estimated total

3 months focused part-time work, or 4-6 weeks full-time, **assuming no fork**. If Gate 2 triggers the morphology fork, add 3-4 weeks. The project's actual cadence depends on user availability.

---

## Acceptance criteria (gates)

### Gate 1: Channel translation correctness (per channel)

**Pass condition:** Brian2-translated channel produces voltage trajectories within 5% of Nicoletti's NEURON reference under identical voltage-clamp + current-clamp protocols.

**Diagnostic:** if divergence > 5%, check unit conversions, NMODL idiom translations, parameter precision. Most failures are unit issues.

**Rollback:** revert that specific channel; keep handcrafted version. Add channel to "not yet imported" list.

### Gate 2: Cellular validation (after essential 7 channels)

Gate 2 has **two components that must both pass**. Splitting them empirically distinguishes "channels work, architecture doesn't" (condition 6 invalidation signature) from "channels still need calibration" (per-channel rollback territory).

**Component 2a — Voltage-clamp trace correctness (channel kinetics):**

AVAL with imported EGL-19 + SLO-1 + h, held under voltage-clamp protocol matching Nicoletti 2024's published recordings, produces Brian2 traces within 5% of Nicoletti's NEURON reference across the IV-curve test set. This component validates that channel kinetic translation is correct — currents at each holding potential match the source models.

**Component 2b — Current-clamp plateau dynamics (architectural sufficiency):**

AVAL under current-clamp injection reproduces Mellem 2008 voltage-clamp plateau dynamics within tolerance:
- Plateau amplitude 15-25 mV (target 20)
- Plateau duration 400-800 ms (target 600)
- Termination: V settles to baseline ±5 mV by 1500 ms post-injection
- **Release behavior:** plateau holds against simulated leak through full duration; collapse-on-release test (stimulus removed at t = 300 ms) shows V trajectory dominated by SLO-1-mediated termination rather than leak τ_m

This component validates that the cellular architecture (compartmental structure, leak conductance, Ca pool dynamics) is sufficient for the channel set to express its full dynamic range. Wave 1's structural finding (leak τ ≈ 10 ms vs τ_d ≈ 20 ms vs Mellem 600 ms target) means correct channels in an under-resourced scaffold can still fail this component.

**Diagnostic decision tree:**

| 2a result | 2b result | Diagnosis | Action |
|---|---|---|---|
| pass | pass | Path A cellular layer production-grade | Proceed to Gate 3 |
| pass | fail | **Channels work, architecture insufficient** (condition 6) | Pause channel translation; pivot to morphology integration |
| fail | fail | Channel kinetic translation has bugs | Per-channel rollback; debug translation |
| fail | pass | Unlikely; would indicate channel kinetics wrong but compensation cancels in current-clamp | Investigate; suspect numerical artifact |

**Rollback:** for 2a failure, revert specific channel to Wave 1 handcrafted version. For 2b failure, see Wave 2 sequencing conditional fork — pivot to c302 morphology integration before continuing channel translation.

### Gate 3: Network preservation (after compartmental integration)

**Pass condition:** Existing scenario sweep (touch, AVA-ablation phenotype, RIS food rescue) reproduces within tolerance of pre-import baseline.
- ΔREV under AVA / touch ablation: same direction, magnitude within 50% of baseline
- RIS firing under food: same qualitative response (no regression to QUIESCENT-locked)

**Diagnostic:** if network regresses, check classifier readout calibration — imported channels may produce different LIF spike statistics than handcrafted.

**Rollback:** keep imported channels at cellular level; revert network integration to use handcrafted approach for the cells where regression is observed.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **NMODL → Brian2 translation has subtle bugs** | high | side-by-side validation harness in Phase α; per-channel rollback |
| **Brian2 unit handling for nS/(pA·ms)/µM gets confused** | medium | use namespace dict for parameters; explicit unit tests after each channel |
| **Imported channels change network firing statistics, breaking classifier readouts** | medium | Gate 3 catches this; if needed, retune classifier on imported-channel data |
| **Some channels have NMODL features Brian2 doesn't support natively** (POINT_PROCESS for certain dynamics) | low-medium | most channels are SUFFIX type (per-cell mechanisms); POINT_PROCESS is for synapses, mostly. Fall back to Path 3B for problematic cells |
| **License terms on Nicoletti's code don't permit redistribution** | low | use her code internally, don't redistribute; verify before publication. ModelDB convention is academic-use-with-attribution which suffices |
| **Translation work takes 2× estimate** | medium | budget accordingly; no calendar deadline |
| **Wave 1 partial cellular validation work obsoleted by import** | high | accept; Wave 1 was a learning phase. The h-equation insights, σ↔h characterization, etc. inform translation choices |

---

## Resource implications

- **Compute:** Brian2 + numpy on RTX 4060 Ti is sufficient. No GPU acceleration needed for translation phase. Brian2GeNN as future option.
- **Storage:** existing ~184 MB of acquired packages + ~50 MB for translation intermediate states + standard simulator artifacts. Negligible.
- **Engineering time:** 1-2 months full-time equivalent or 3-6 months part-time.
- **External collaboration:** none required. Investigation and translation can proceed independently. Optionally: contact OpenWorm community / Nicoletti's group for license verification before publication.
- **Wet-lab dependency:** none.

---

## Connection to paper trajectories

### Paper 2 (behavioral simulator, NeurIPS GRL or ICLR LMRL workshop)

Closes near-term independent of Wave 2 architectural commitment. Sample_004 + LIF + voltage-domain target framework is sufficient for behavioral closure. Wave 2 work (channel import) doesn't delay paper 2.

**Optional integration:** if paper 2's submission timeline allows, paragraph in methods noting "biophysical channel-level work pending Wave 2" with reference to validated cellular foundation. Not required for behavioral claims.

### Paper 3 (mechanistic simulator)

**Wave 2 work is the foundation for paper 3's mechanistic claims.** Once channels are imported and validated against Mellem 2008 + Nicoletti 2024 traces, paper 3 can claim cellular-level mechanism for cascade dynamics, plateau termination, modulator effects. Paper 3 should:
- Reference Nicoletti's channel implementations (with citation per ModelDB convention)
- Cite c302 framework (MIT license, citation appropriate)
- Document the project's distinguishing contributions (CeNGEN-coupled densities, peptide processing, sensory transduction integration, behavioral closure)
- Acknowledge the partial-coverage limitation (9 of 302 cells have validated biophysics; rest are calibrated by analogy)

### Paper 4 (cross-session methodology)

Independent of Wave 2 architectural commitment. Cross-session adversarial review pattern is the contribution; channel-level architectural decisions are content.

### Long-term research-tool trajectory

Wave 2 establishes the cellular biophysical foundation. Subsequent waves address gaps Path A doesn't solve:
- Wave 3: receptor binding kinetics (Markov state schemes)
- Wave 4: CeNGEN-coupled per-cell channel densities
- Wave 5: peptide processing refinement
- Wave 6: anesthesia-specific allosteric framework (if anesthesia application focus)
- Wave 7: validation against held-out experimental datasets (Atanas 2023, Hallinen 2021, Yemini NeuroPAL)

Each wave is additive; no future wave's success requires reverting earlier waves. The waves correspond to the previous audit's near-term work block recommendations.

---

## What would invalidate Path A?

Per the discipline of pre-flight pushback throughout today, this section commits to honestly identifying conditions under which Path A would not be the right move.

**Path A is invalidated if any of these surface during execution:**

1. **Brian2 has architectural limitations that prevent NMODL-style channel translation cleanly.** If the first 2-3 channel translations produce > 5% trace divergence from Nicoletti and root cause is "Brian2 can't represent this kinetic scheme," Path A is structurally blocked. **Mitigation: Path 3B (multi-framework) becomes primary.**

2. **The project pivots to a primary application that requires NEURON ecosystem.** If anesthesia mechanism research with NetPyNE collaboration becomes the focus, or if drug-discovery partner labs use NEURON exclusively, the architectural commitment shifts. **Mitigation: re-evaluate at the gate where application focus crystallizes.**

3. **Nicoletti's models don't reproduce Mellem 2008 cellular targets in NEURON either.** If running her code locally produces voltage-clamp traces that don't match Mellem's published values, the translation effort is moot — the source models have unresolved gaps. **Mitigation: Phase α validation harness catches this before Phase β translation work.** Re-investigation needed; consider parameter calibration of Nicoletti's set or alternative sources.

4. **Brian2 performance regresses unacceptably with full channel set.** Currently Brian2 is fast for handcrafted ~3-channel cells; 22 channels per cell may be slower. If regression > 5×, Path 3A's compute advantage erodes. **Mitigation: Brian2GeNN GPU acceleration; Path 3B for the high-channel cells.**

5. **Discovery that BAAIWorm or another upstream project provides better-validated Brian2 implementations than translating Nicoletti.** If during translation work a cleaner upstream Brian2 channel library surfaces, switch to importing that. **Mitigation: stay alert to community developments.**

6. **Cellular validation fails not on channel kinetics but on compartmental architecture.** If Phase γ produces correct channel behavior in voltage-clamp but plateau still collapses on stimulus release in current-clamp, the bottleneck is morphology, not channels — Path A is *partial*, and compartmental integration (using c302's 607 cell morphologies) becomes a Phase β/γ prerequisite, not a deferred Wave 3 item. The diagnostic signature is "channels validated, dynamics still wrong" — voltage-clamp trace matching Nicoletti within 5% while current-clamp release plateau collapses within τ_d ≈ 20 ms instead of holding the Mellem 2008 600 ms target. This failure mode is structurally predicted by Wave 1: leak τ ≈ 10 ms overwhelms L-type Ca by 5× in single-compartment graded scaffold, and importing better channel kinetics into the same scaffold cannot fix a leak/architecture mismatch. **Mitigation: Phase γ acceptance gate explicitly tests current-clamp release behavior, not just clamped traces.** If condition 6 surfaces, Wave 2 sequencing forks into morphology-integration-first (see Wave 2 sequencing conditional fork below).

If invalidation surfaces, the response is **mid-flight pause + investigation re-run**, not silent commitment to a known-broken path. Per the cross-session methodology established today.

---

## Decisions the user needs to commit to

To proceed with Wave 2, the following commitments are needed:

1. **Path 3A is primary** (vs Path 3B/3C/3D).
2. **Setup phase (Phase α: install NEURON locally + compile Nicoletti's mods + build validation harness) is the first work block** of Wave 2. ~3-5 days.
3. **The 7-channel essential set is the priority** (EGL-19, SLO-1 + EGL-19 coupled, SLO-1 isolated, SHK-1, SHL-1, NCA, KQT-3) — translates first.
4. **Gate 2 is two-component:** voltage-clamp trace correctness against Nicoletti 2024 (channel kinetics) + current-clamp plateau dynamics against Mellem 2008 (architectural sufficiency). Both must pass to declare Path A's cellular layer production-grade. Network-level changes are Gate 3.
5. **Per-channel rollback is the standard** — no all-or-nothing translation commitment.
6. **License verification on Nicoletti is a prerequisite for publication, not for development.** Treat as production-prep gate.
7. **Morphology integration is empirically triggered, not deferred.** If Gate 2 surfaces "channels work, architecture insufficient" (2a-pass / 2b-fail, the condition 6 signature), Wave 2 sequencing forks into c302 morphology integration as a Phase β prerequisite (~3-4 weeks added) before resuming channel translation. The fork is a commitment to *not* silently advance with a known architectural gap.

If these commitments are accepted, Wave 2 implementation work begins with Phase α setup. The acquired packages on disk are the prerequisite materials.

---

## Connection to today's broader cross-session work

Today's three-session pattern (Sessions 1, 2, 3 working in parallel with adversarial review) produced ~11 substantive methodological corrections across the cellular validation + biophysical audit + sign-convention work. The pattern continues in Wave 2:

- **Session 1** (or equivalent) can run empirical validation as channels are translated
- **Session 3** (or equivalent) can run biophysical consistency audit on each new channel's parameter set against Nernst/equilibrium expectations
- **Session 2** (this work) provides the architectural and integration framing

The pattern of pre-flight pushback before each work block, mid-flight surfacing of findings, and cross-session integration applies to Wave 2 channel translation just as it has applied to today's work.

The strategic positioning unchanged from previous audit: **the project's distinguishing potential is integration above the channel layer.** Path A acquires the channel layer. The layers above (CeNGEN coupling, modulator refinement, behavioral closure, sensory transduction, MuJoCo body integration, Wave 1 cellular work, scenario pipeline, dashboard) remain the project's owned contributions and are preserved fully under Path 3A.

---

## Standing by for Wave 2 implementation commitment

This document, together with the three Phase 1+2+3 documents, comprises the decision-grade output for Wave 2 architectural commitment. If user accepts Path 3A as primary, Phase α setup begins. If user wants to revisit any part of the analysis (choice of Path 3A vs alternatives, sequencing of channel priorities, gate criteria, etc.), discussion before implementation is welcome per the established methodology.

