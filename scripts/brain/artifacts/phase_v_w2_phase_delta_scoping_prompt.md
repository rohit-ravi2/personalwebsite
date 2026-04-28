# Wave 2 Phase δ network integration — scoping work block

**Mode:** investigation + architectural design, not implementation. Single session, file-based output. ~1-2 hours bounded scope.

**Strategic positioning:** Wave 2's cellular layer is substantively complete (3 production-grade cells: AVAL, AIY, RIM; 14 channel translations; F1-F18 catalog; cython baseline established at 22.71× pure-Brian2 speedup). Phase δ network integration is the next major Wave 2 deliverable — connecting the validated cellular layer to the production simulator's existing infrastructure (LIFBrain, GradedBrain, MuJoCo, scenario JSON, dashboard, FSM, classifier, modulator layer, sensory cascades).

This work block produces decision-grade scoping output that informs Phase δ implementation work blocks. **Do not implement Phase δ — scope it.**

---

## Out of scope

- Implementation of Phase δ work (separate work blocks based on this scoping)
- New channel translations (CCA-1/EGL-2/UNC-2 done; RMD-specific deferred)
- Architectural plan revisions
- AVAR upstream issue review
- Methodology paper documentation
- Citation cleanup
- Cython optimization beyond what's done

If scoping surfaces that Phase δ requires substantial new infrastructure (new channel translations, harness extensions beyond what exists), document but don't expand scope into building that infrastructure.

---

## Working environment + context

- Cython now production codegen target across 17 wave2 .py files (just-completed cleanup)
- 3 production-grade cells:
  - `wave2/option_alpha_ava_cell.py` (4-channel AVAL)
  - `wave2/option_alpha_aiy_cell.py` (7-channel AIY, asymmetric USEION ca, eca=127.59)
  - `wave2/option_alpha_rim_cell.py` (7-channel RIM, symmetric USEION ca, eca=60)
- Validation infrastructure: NEURONReference wrapper, voltage_clamp_harness, plateau_harness with Layer A compare
- 14 channel translations at `wave2/channels/`
- F1-F18 NMODL pattern catalog at `wave2/translation_patterns.md`
- Compute baseline: per-cell per-second-simulated cython ~1.0-1.8 s wall-clock at dt=0.025 ms; 3-cell × 10s sim = ~43s wall-clock

**Production simulator infrastructure** (NOT in wave2/, must read for context):
- `~/Desktop/website/personalwebsite/scripts/brain/` — production code root
- `LIFBrain` and `GradedBrain` — existing simulator brain layers
- `compartmental_neurons_kca.py`, `graded_brain_h_kca.py` — existing cellular implementations (Wave 1 era)
- Connectome data: `connectome.npz` referenced in earlier project work; loaded from Cook 2019
- Scenario pipeline: scenario JSON files driving touch/food/osmotic/etc. behavior
- Dashboard, FSM, classifier — observability and behavior layers
- Modulator layer (9 modulators per project memory)
- Sensory transduction cascades (5 cascades)
- MuJoCo body integration

---

## Pre-flight pushback expected

Read this prompt fully. Pre-flight verification points:

1. **Production simulator architecture** — verify the existing brain layer's actual API and integration points by reading the code, not by assuming. Today's pattern of pre-flight catches has shown orchestrator-side claims about external structure are unreliable. Read at minimum: `LIFBrain` definition, `GradedBrain` definition, where they're constructed, how scenarios drive them.

2. **Connectome data availability + format** — verify `connectome.npz` exists and document its actual structure. Phase δ's "connect cells via connectome" needs to know what the connectome data shape is.

3. **Scenario pipeline format** — verify scenario JSON files exist; document one or two scenario specs to understand the integration target.

4. **Wave 2 cellular layer integration shape** — Wave 2 cells (AVAL, AIY, RIM) are Brian2 NeuronGroups built via factory functions. Production simulator probably uses a different cell representation. The integration question is: how do these connect?

If pre-flight surfaces concerns warranting cross-session discussion: write to `wave2/artifacts/phase_delta_scoping_pushback.md` + create `PAUSED_FOR_REVIEW.txt`. Otherwise proceed to investigation.

---

## Methodology continuity

- Mid-flight findings to `wave2/artifacts/phase_delta_scoping_findings.md`
- F1-F18 lessons applied where relevant (especially F18 multi-USEION-ca handling at network scale)
- Document, don't fabricate. If production simulator's actual structure differs from prompt assumptions, primary source wins.
- Pre-flight pushback discipline: this work block scopes; pause if scoping surfaces architectural questions warranting decision before continuing

---

## Five investigation phases

Each phase produces written analysis. Each phase informs the next. The output is decision-grade scoping document.

### Phase 1 — Production simulator architecture survey

Read the production simulator code. Document:

1. **Brain layer architecture:** how `LIFBrain` and `GradedBrain` are structured. Are they classes? Are cells objects within them? How is the connectome consumed?
2. **Cell representation:** what is a "cell" in the production simulator? Compare against Wave 2's Brian2 NeuronGroup representation. What's the structural delta between them?
3. **Time loop:** how does the simulator step forward in time? Single Brian2 `run()`? Custom timestep loop? Coupled with MuJoCo at body cadence?
4. **State variables:** what state does the brain layer maintain? Voltages? Spike times? Per-neuron, per-synapse, per-modulator state?
5. **I/O:** how does the brain layer expose state to dashboard, FSM, classifier? What's the readout interface?

Document in `wave2/artifacts/phase_delta_scoping_findings.md` Section 1.

### Phase 2 — Wave 2 cell integration shape analysis

Given Phase 1's understanding, characterize how Wave 2's Brian2 cells could integrate with the production brain layer. Three architectural alternatives to evaluate:

**Alternative A — Replace existing brain layer entirely.** Wave 2 Brian2 cells become the new brain. Connectome wired via Brian2 Synapses. Existing scenario/dashboard/FSM/classifier connect to Brian2 state via adapter layer.

**Alternative B — Hybrid: Wave 2 cells for the validated subset, existing brain for the rest.** AVAL/AIY/RIM run as Brian2 cells; other 299 cells run via existing infrastructure. Coupling at the connectome layer with adapter passing voltages/currents between systems.

**Alternative C — Wave 2 cells as drop-in replacements for specific cells in existing scaffold.** Existing brain layer remains; Wave 2 cells inject as alternate implementations of AVAL/AIY/RIM only. Existing infrastructure unchanged elsewhere.

For each alternative document:
- Implementation complexity (low / medium / high)
- Coupling points required
- Risk profile (what breaks if this fails)
- Performance implications
- Path to "Phase δ proper" via this alternative

Document in Section 2.

### Phase 3 — Compute envelope and scaling analysis

Given:
- Per-cell per-second-simulated cython baseline: ~1.0-1.8 s wall-clock
- Existing simulator runs scenarios at ~60s simulated worm time
- 302 neurons in full connectome

Estimate compute cost for representative Phase δ workloads under each alternative:

1. Alternative A (full Brian2 brain, 302 cells): how does this scale? Brian2 batches across NeuronGroup, so per-step overhead is fixed regardless of N within group. But channel diversity across 302 cells means many parameter values. What's the compute envelope for 60s simulated time?
2. Alternative B (3 Brian2 + 299 existing): the Brian2 part is bounded (3 cells × 60s × ~1s/s = 180s wall-clock). What does coupling overhead add?
3. Alternative C (3 cells in existing scaffold): minimal compute change vs current; Brian2 cells run for AVAL/AIY/RIM, existing infrastructure unchanged.

Identify compute bottlenecks and where optimization would matter most.

Document in Section 3.

### Phase 4 — Validation strategy for Phase δ

How does Phase δ validate? Two layers:

**Layer A network: Brian2 + connectome integration produces same Phase δ behavior as production simulator on existing scenarios.** Tests that the integration didn't break what worked.

**Layer B network: Brian2 + connectome produces biologically meaningful behavior.** Tests that the integration works toward research goals.

For each layer, identify:
- Reference targets (what does "passing" mean concretely?)
- Test scenarios (touch, food, osmotic_shock, etc.)
- Failure modes (what's plausibly wrong vs implementation bug vs deeper finding?)
- Layer A vs Layer B prioritization

Document in Section 4.

### Phase 5 — Phase δ work block decomposition

Decompose Phase δ implementation into bounded work blocks. Each work block should fit within ~1-3 hours single-session execution. Work blocks should compose: each produces output that the next consumes.

Identify:
- Which alternative (A/B/C) the decomposition assumes
- Sequencing dependencies between work blocks
- Critical path vs parallel-able work
- Gates that determine whether to proceed to next work block
- Estimated total scope (number of work blocks, rough time scale)

Output: ordered work block list with one-paragraph description per work block. This is the implementation roadmap.

Document in Section 5.

---

## Output

Single document: `wave2/artifacts/phase_delta_scoping.md` with the 5 sections above + final synthesis section:

**Section 6 — Recommended Phase δ trajectory:**
- Recommended alternative (A/B/C) with rationale
- First work block to deploy
- Decision gates for subsequent work blocks
- Risk register (what could go wrong, mitigations)
- Success criteria for Phase δ overall

This is the morning-review entry point and the launching pad for Phase δ implementation work blocks.

---

## Failure modes

**Pre-flight surfaces production simulator structure differs substantially from prompt assumptions:** document, surface for review. Most of this prompt's "production simulator infrastructure" claims are best-effort recollections — primary source wins.

**Phase 2 alternatives don't cleanly map onto actual integration constraints:** document, propose actual alternatives, surface for review.

**Phase 5 decomposition surfaces work scope substantially larger than expected (e.g., 20+ work blocks for full Phase δ):** document the actual scope, surface trade-offs (full Phase δ vs reduced scope vs deferred to Wave 3).

**General principle:** scoping work produces decision-grade output. If decisions can't be cleanly made on the available information, document the gap and surface — don't paper over with assumptions.

---

## Output file structure

```
wave2/
└── artifacts/
    ├── phase_delta_scoping.md                           # main output
    ├── phase_delta_scoping_findings.md                  # mid-flight findings
    ├── phase_delta_scoping_pushback.md                  # only if pre-flight concerns
    └── PAUSED_FOR_REVIEW.txt                            # only if paused
```

---

## Final framing

This work block is scoping, not implementation. Output informs Phase δ implementation work blocks. Decision-grade material for: which alternative architecture, how to validate, how to decompose into bounded work blocks.

Apply the discipline. Pre-flight verify production simulator architecture against actual code, not assumptions. Document, don't fabricate. Phase δ is the bigger deliverable — getting scoping right matters.

Standing by for pre-flight pushback or completion notification.
