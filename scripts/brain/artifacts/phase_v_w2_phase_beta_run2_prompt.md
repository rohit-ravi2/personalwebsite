# Phase β overnight run #2 — Wave 2 channel translation: F6 diagnostic + 6-channel pipeline + Gate 2 evaluation, with speculative-architecture fork

**Mode:** overnight, autonomous, file-based pause-and-wait. User is asleep. Cross-session adversarial review remains idle until morning.

**Risk posture:** user has explicitly accepted methodology risk for speculative-architecture fork. If F6 verdict triggers fundamental-issue branch, run forks into speculative work rather than pausing for morning review of options.

---

## Context: Wave 2 status and what this overnight run accomplishes

Phase β overnight run #1 cleared CP1-CP3 cleanly in <1 hour total. Path A empirically vindicated at first channel level: EGL-19 translated, validated against NEURON reference, Gate 2a-cleared in cell context. CP3's CC test landed at 89.9% margin (just over 80% threshold) — the 10.1% failures attributed to onset-transient differences between NEURON capacitive transient and Brian2 force-clamp behavior.

One yellow flag from run #1: F6 finding. cadiff translation required empirical 52,700× scale factor with R²=0.984 (caintra1 R²=1.000). The agent suspected (1e3 mM↔Brian2) × (1e3 ms↔s) × (~52 geometry factor) decomposition but the physical source isn't symbolically understood. Empirical calibration valid for tested parameter ranges but unclear whether it generalizes to:
- Different cell geometries (AIY, RIM, etc.)
- Different Ca injection regimes outside calibration range
- Channels that read [Ca]_i for gating (SLO-1 specifically)

This overnight run accomplishes:

1. **F6 calcium calibration diagnostic** with code-reading depth, geometry analysis, calibration robustness testing, and explicit decision tree for architectural alternatives if F6 surfaces fundamental issues
2. **6 remaining essential-set channels** translated and Gate 2a-validated (SHK-1, SHL-1, NCA, KQT-3, SLO-1 isolated, SLO-1+EGL-19 coupled — order chosen for momentum-building before hardest case)
3. **Gate 2 full evaluation** on AVA cell with **two decoupled cell constructions** (see Phase F clarification below)
4. **Conditional speculative-architecture fork** — if F6 verdict triggers fundamental-issue branch, fork to investigate GNN-based cellular dynamics as Wave 3+ alternative architecture

---

## Phase F clarification — two decoupled cell constructions for Gate 2

**Resolved before launch.** Gate 2 has two components with decoupled validation paths; they do NOT share a cell construction.

**Component 2a (channel kinetics correctness in cell context):**

- **Cell construction:** Brian2 AVA with Nicoletti's actual AVA subset (NCA + EGL-19 + leak; UNC-103/IRK excluded since they're not in our essential set yet)
- **Reference:** Nicoletti's NEURON AVA (her actual model)
- **Comparison:** voltage-clamp Layer A — Brian2 cell vs NEURON cell, same channel set
- **What this tests:** integration of multiple Brian2 channels in cell context matches NEURON reference where Nicoletti has provided one. SHK-1, SHL-1, KQT-3, SLO-1 isolated, SLO-1+EGL-19 coupled don't get cell-context validation here because Nicoletti's AVA doesn't include them — they're per-channel validated in Phases C-E, so Gate 2a's "channel kinetics correctness" is satisfied through per-channel + integration check on Nicoletti subset.

**Component 2b (architectural sufficiency):**

- **Cell construction:** Brian2 AVA with full 7-channel essential set (EGL-19, SLO-1 isolated, SLO-1+EGL-19 coupled, SHK-1, SHL-1, NCA, KQT-3) + leak + Ca-pool, using Nicoletti's published densities where she provides them and reasonable defaults elsewhere (document choices)
- **Reference:** Mellem 2008 plateau targets (20 mV / 600 ms / SLO-1-dominated termination) — NO NEURON reference needed for 2b
- **Comparison:** experimental-target comparison
- **What this tests:** does the full essential-set cell produce Mellem-target plateau dynamics? Condition-6 detection lives here.

**Why decoupled:**

- 2a failure modes are channel-translation issues (caught per-channel in Phases C-E, plus integration verification on Nicoletti subset)
- 2b failure modes are architectural (compartmental sufficiency, leak/Ca balance, dynamics emergence from full channel ensemble)

The two-component design lets the diagnostic decision tree work cleanly: 2a-pass + 2b-fail = condition 6 (architecture insufficient, not channels).

---

## Strategic positioning: methodology + accepted risk on speculative fork

Today's methodology discipline (cross-session adversarial review, pre-flight pushback, mid-flight surfacing of findings, stop-and-pause when uncertainty surfaces) has produced ~15 substantive catches across Phase α, β-pre v1/v2/v3, harness fitness assessment, and Phase β overnight #1. Maintain this discipline through this run.

**On the speculative-architecture fork:** the user has explicitly accepted methodology risk for this run. If F6 surfaces fundamental issue, the run forks into speculative-architecture investigation rather than pausing for morning review of options. Rationale: speculative work has low expected value but plausibly high-impact insight; risk-tolerance is explicitly granted for this overnight. This contradicts default methodology pattern but is intentional and informed.

The speculative fork has its own scope discipline: investigate, document, surface findings — do not commit to architectural pivot, do not modify production simulator, do not abandon Path 3A based on speculative work. Speculative results are inputs to morning review, not autonomous commitments.

---

## Working environment

- Isolated venv at `~/venvs/wave2-neuron/` with NEURON + Brian2 (carry forward from run #1)
- Project code at `~/Desktop/website/personalwebsite/scripts/brain/wave2/` (extend, don't replace)
- Nicoletti 2024 source at `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- c302 source at `~/Desktop/C-Elegans/simulation/upstream/c302/` (relevant for speculative fork morphology data)
- Phase α report at `wave2/phase_alpha_report.md`
- Run #1 summary at `wave2/artifacts/phase_beta_run_summary.md` (read for context)
- Run #1 findings at `wave2/artifacts/phase_beta_findings.md` (F1-F10 catalog)
- Architectural plan at `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` (read fully before starting)

---

## Subagent autonomy bounds

You have full permission for: code creation, code modification within `scripts/brain/wave2/`, file creation, package operations within wave2 venv, running NEURON/Brian2 simulations, comparing outputs, computing metrics, exploratory implementation work in `wave2/speculative/` subdirectory if speculative fork triggers.

You do NOT have permission for: modifying production simulator code outside `scripts/brain/wave2/`, modifying Nicoletti's upstream code (use local patches only), modifying c302 upstream code, filing GitHub issues, expanding scope beyond defined phases.

You have explicit permission to make architectural decisions in speculative fork without pausing for review (per accepted risk), but document reasoning thoroughly. You do NOT have permission to commit speculative architecture to production simulator code — speculative work stays in `wave2/speculative/`.

---

## Pre-flight pushback adapted for autonomous execution

Read the full prompt before starting work. If you find scope concerns, hidden assumptions, or items that warrant cross-session discussion, write them to `wave2/artifacts/phase_beta_run2_pushback.md` and pause via `wave2/artifacts/PAUSED_FOR_REVIEW.txt` marker file. The user will read morning-after and either resolve concerns or authorize proceeding.

If pre-flight surfaces no concerns, proceed to Phase A.

---

## Phase structure overview

7 phases plus conditional speculative fork. Each phase has explicit go-conditions, acceptance criteria, status output. Failure modes documented per phase.

- **Phase A:** F6 calcium calibration diagnostic (gate for Ca-dependent work)
- **Phase B:** NMODL gotcha catalog systematization
- **Phase C:** Non-Ca channels (SHK-1, SHL-1, NCA, KQT-3 — independent of F6)
- **Phase D:** SLO-1 isolated (Ca-dependent, gated on F6)
- **Phase E:** SLO-1+EGL-19 coupled (hardest channel, gated on F6 + Phase D)
- **Phase F:** Gate 2 full evaluation on AVA cell (two decoupled constructions per clarification)
- **Phase G:** Run summary and morning materials

**Conditional Phase X (speculative-architecture fork):** triggered if Phase A F6 verdict is "fudge factor papering over fundamental issue" — investigate GNN-based cellular dynamics as alternative architecture. Triggered AFTER Phase C completes (non-Ca channels independent of F6; complete those before forking), runs in parallel to or instead of Phases D-F depending on time.

---

## Phase A — F6 calcium calibration diagnostic

Code-reading depth investigation. Three sub-investigations + decision tree synthesis.

### A.1 — Symbolic decomposition via NEURON code reading

Read NEURON's source for Ca-handling machinery. Specifically:
- `nicoletti_2024/.../cadiff.mod` — full NMODL source
- `nicoletti_2024/.../caintra1.mod` — full NMODL source
- NEURON's ion-handling internals (USEION, ca pool initialization, surface-area calculations)
- Look for explicit unit conversions in NEURON's compiled mod handling
- Look for geometry parameters (pi, diameter, length, surface area) in the mod files or implicit in NEURON's call protocol

Goal: identify what physical quantities compose the empirical 52,700× factor. Hypothesize candidate decompositions:
- Unit conversion: 1e3 mM↔M × 1e3 ms↔s = 1e6
- Geometry: surface-area-to-volume ratio for cylindrical cell — compute from AVA's geometry (radius, length from Nicoletti's parameters)
- Other: anything in NEURON's ion-handling that the symbolic translation missed

Output: `wave2/artifacts/f6_symbolic_decomposition.md` with attempted decomposition, what was identified, what couldn't be explained.

### A.2 — Geometry parameter check

Compute AVA's surface-area-to-volume ratio from Nicoletti's geometry parameters. Compare to the residual factor after explaining unit conversions.

If unit conversions account for 1e6 of the 52,700, residual ≈ 0.0527 (1/19). Check if 1/19 corresponds to any geometry-derived quantity (e.g., (V_compartment / S_compartment) for AVA's specific dimensions).

If unit conversions account for some other amount, residual changes — track through carefully.

Then repeat for at least 2 other cells (AIY and RIM, since they have Nicoletti reference implementations). Compute their geometry parameters, predict their calibration factors based on the symbolic decomposition.

Output: `wave2/artifacts/f6_geometry_analysis.md` with computed factors for AVA, AIY, RIM and predictions for whether they'd require different empirical factors.

### A.3 — Calibration robustness test

Take the cadiff Brian2 implementation from run #1. Test it across:
- AVA geometry vs AIY geometry vs RIM geometry (use same calibration factor — does it generalize?)
- Multiple Ca injection regimes (low, calibrated middle, high)
- Multiple time scales (short pulses, sustained injection)

For each test condition, compare Brian2 output to NEURON output. Compute R² and divergence metrics.

If calibration factor generalizes cleanly across geometries: F6 verdict trends toward "principled."
If calibration factor needs adjustment per geometry but adjustments follow predictable pattern (e.g., scale linearly with surface-area-to-volume): F6 verdict trends toward "partially principled with documented gaps."
If calibration breaks unpredictably across regimes: F6 verdict trends toward "fudge factor."

Output: `wave2/artifacts/f6_calibration_robustness.md` with full test matrix and results.

### A.4 — Decision tree synthesis

Based on A.1-A.3, write `wave2/artifacts/f6_diagnostic_synthesis.md` with explicit verdict:

- **VERDICT_PRINCIPLED:** symbolic decomposition is clean, geometry analysis predicts cell-specific factors accurately, calibration robustness tests pass across regimes. Implication: Phase A complete. Proceed to Phase B with confidence. Document the principled framework for subsequent Ca-handling work.

- **VERDICT_PARTIALLY_PRINCIPLED:** symbolic decomposition is mostly clean with documented gaps, geometry analysis works for tested cells but has unexplained residuals, calibration robustness passes for tested regimes but uncertain outside them. Implication: Phase A complete with documented limitations. Proceed to Phase B-F with explicit monitoring — flag any calibration anomalies during SLO-1 work and Gate 2 evaluation.

- **VERDICT_FUDGE_FACTOR:** symbolic decomposition fails to explain empirical factor, geometry analysis doesn't predict cell-specific factors, calibration breaks across regimes. Implication: Path A's Ca-handling foundation is uncertain. **TRIGGER SPECULATIVE FORK after Phase C completes.** Phases D-E (SLO-1 work) are gated on Phase X outcome.

The decision tree synthesis is the load-bearing output. Subsequent phases consult it.

### Phase A acceptance criteria

- A.1, A.2, A.3 outputs all written
- A.4 synthesis written with explicit verdict
- Status file `wave2/artifacts/checkpoints/phase_a_status.json` records outcome

### Phase A failure modes

- Code reading reveals NEURON internals are too opaque to decompose: document, set verdict to VERDICT_PARTIALLY_PRINCIPLED with note on opacity
- Geometry analysis can't be performed (geometry parameters not accessible): document, set verdict based on A.3 alone
- Calibration robustness test crashes (numerical instability): document, isolate to specific regimes, set verdict based on partial results

---

## Phase B — NMODL gotcha catalog systematization

Lift F1-F10 from `phase_beta_findings.md` into `wave2/translation_patterns.md` as reusable catalog. Structure each pattern as:

- **Pattern name** (e.g., "Unit conversion 1e-3 trap")
- **Recognition signature** (what symptoms suggest this pattern is occurring)
- **Recommended handling** (what to do when encountered)
- **Cross-channel implications** (which channels likely affected)
- **Source finding** (link back to original F-number)

Add new patterns from F6 diagnostic if any surfaced.

This catalog is institutional knowledge. Subsequent Phase β overnight runs consult and extend.

### Phase B acceptance criteria

- `wave2/translation_patterns.md` exists with all F1-F10 systematized
- New patterns from F6 diagnostic added
- Status file records completion

---

## Phase C — Non-Ca channels (SHK-1, SHL-1, NCA, KQT-3)

Four channels translated sequentially. Each follows the EGL-19 workflow established in run #1: read .mod, translate to Brian2, validate via voltage-clamp harness against NEURON reference. Gate 2a per-channel acceptance: voltage-feature ≤3 mV equivalent in current domain, >80% of holding potentials clear tolerance.

### C.1 — SHK-1 translation

Read `nicoletti_2024/.../shk1.mod`. Translate to Brian2 equation string. Save to `wave2/channels/shk1.py`. Validate via voltage-clamp harness (uses NEURONReference wrapper from run #1).

### C.2 — SHL-1 translation

Same workflow. Save to `wave2/channels/shl1.py`.

### C.3 — NCA translation

Same workflow. Save to `wave2/channels/nca.py`. Note: NCA is NALCN homolog (sodium leak), simpler than typical voltage-gated channels.

### C.4 — KQT-3 translation

Same workflow. Save to `wave2/channels/kqt3.py`. Note: KQT-3 is M-current K channel.

### Phase C acceptance criteria per channel

- Brian2 implementation file exists in `wave2/channels/`
- Voltage-clamp validation passes >80% of holding potentials within tolerance
- IV curve from Brian2 matches NEURON reference within tolerance
- Status file per channel records pass/fail/paused

### Phase C failure modes

- Channel-specific NMODL gotcha not in catalog: document as new finding, attempt resolution, surface if resolution requires architectural decision
- Translation fails tolerance test: investigate (NMODL pattern? unit error? parameter mismatch?), pause if cause unclear after focused debugging
- Numerical instability: document parameters causing instability, do not fudge — pause for review

---

## Conditional Phase X — Speculative architecture fork (triggered ONLY if Phase A verdict is VERDICT_FUDGE_FACTOR)

This phase exists because user explicitly accepted methodology risk on speculative architecture investigation. If Phase A verdict triggers it, Phase X runs after Phase C completes. Phases D-F (SLO-1 work + Gate 2) are gated on Phase X outcome.

### X.1 — GNN hybrid architecture investigation

Investigate whether GNN-based cellular dynamics could replace or augment compartmental cable-equation approach. Specifically:

- **X.1a — Architectural sketch:** what would GNN-based cellular dynamics look like for C. elegans single neurons? Nodes = compartments or channel populations or both? Edges = axial coupling, channel interactions, modulator diffusion? State evolution = learned or specified per node? Document conceptual architecture.

- **X.1b — Training data feasibility:** what data would be needed to train such a GNN? Nicoletti voltage-clamp data + cellular geometry + channel parameters? Is the available data sufficient (in volume, in coverage of regimes) for meaningful training? Or would it be data-starved?

- **X.1c — Comparison framework:** how would GNN dynamics be validated? Same voltage-feature comparison against NEURON reference? Or different metrics (e.g., distributional comparisons, trajectory similarity)? Document validation pathway.

- **X.1d — Prototype attempt (time permitting, bounded effort):** if X.1a-X.1c surface a clear architectural sketch with feasible training data and validation framework, attempt minimal prototype:
  - Construct simple GNN (PyTorch geometric or similar) with 2-3 nodes representing simple cellular compartments
  - Train on Brian2-generated voltage-clamp data from EGL-19 in test cell
  - Evaluate whether trained GNN reproduces voltage trajectories within reasonable tolerance
  - **Bounded effort:** if prototype hits diminishing returns within speculative phase, document state and move to X.2. Negative results documented honestly are valuable; do not over-invest time chasing positive prototype results.

### X.1 outputs

- `wave2/speculative/gnn_architecture_sketch.md`
- `wave2/speculative/training_data_feasibility.md`
- `wave2/speculative/comparison_framework.md`
- `wave2/speculative/prototype/` directory with prototype code if X.1d attempted
- `wave2/speculative/x1_summary.md` with overall assessment

### X.2 — Alternative architectures briefly investigated

If time permits after X.1:

- **Multi-compartment with explicit geometry** (cable equations + explicit segment-level state): document what changes if Ca-pool encoding moves to per-segment computation rather than empirical scaling
- **NeuroML2-native simulation** (use jNeuroML or libNeuroML to consume c302's morphology data directly, bypass Brian2): document what infrastructure changes would be required

These are briefer investigations than X.1 — sketch + feasibility assessment, no prototype.

### X.2 outputs

- `wave2/speculative/multi_compartment_explicit.md`
- `wave2/speculative/neuroml2_native.md`

### Phase X gating decisions for D-F

After Phase X completes, decision tree:

- **Phase X surfaces clear better-than-Path-A architecture:** PAUSE for morning review. Do not proceed to D-F. Morning review evaluates whether Path A continues or pivot is warranted.
- **Phase X surfaces possible alternatives without clear winner:** PAUSE for morning review with options surfaced. D-F deferred to subsequent overnight.
- **Phase X surfaces no compelling alternatives (most likely outcome):** D-F proceed under Path A despite F6 verdict, with F6 fudge-factor risk explicitly carried forward as known uncertainty. Document this explicitly in subsequent work.

### Critical scope discipline for Phase X

- Do NOT modify production simulator code outside `wave2/speculative/`
- Do NOT abandon Path 3A based on Phase X work — speculative results are inputs to morning review, not autonomous commitments
- Do NOT attempt full implementation of any speculative architecture — sketch + minimal prototype only
- Document everything, fabricate nothing
- If prototype work shows "this clearly doesn't work in available time," that's a useful finding documented honestly

---

## Phase D — SLO-1 isolated (gated on Phase A and Phase X)

Translate `nicoletti_2024/.../slo1iso.mod`. SLO-1 is BK channel, voltage- and Ca-dependent. SLO-1 isolated variant doesn't include nanodomain coupling to specific Ca channels.

### Workflow

1. Read slo1iso.mod
2. Translate to Brian2 equation string. Channel reads V from cell, [Ca]_i from cadiff/caintra1 system.
3. Save to `wave2/channels/slo1_iso.py`
4. Validate via voltage-clamp harness with cell containing leak + EGL-19 + Ca-pool + SLO-1iso (need Ca to flow for SLO-1 to gate). Compare against NEURON reference cell with same channel set.

### Phase D acceptance criteria

- Brian2 implementation file exists
- Voltage-clamp Layer A passes >80% of holding potentials within tolerance
- Status file records outcome

### Phase D failure modes

- F6 calibration issue surfaces (SLO-1 reads [Ca]_i, calibration anomalies become observable here): document explicitly, surface as confirmation of F6 fudge-factor risk if Phase A verdict was PARTIALLY_PRINCIPLED
- Standard channel translation issues: per Phase C handling

---

## Phase E — SLO-1+EGL-19 coupled (gated on Phase D)

Hardest channel in the essential set. Translate slo1egl19.mod which encodes nanodomain coupling — SLO-1 BK kinetics depend on local [Ca]_i provided specifically by EGL-19, not bulk [Ca]_i.

**Architectural decision required:** how to encode nanodomain coupling in Brian2.

Options:
- (a) Local [Ca]_i as separate state variable computed from EGL-19's I_Ca with shorter time constant than bulk
- (b) Two coupled compartments (one for bulk, one for sub-membrane near EGL-19)
- (c) Phenomenological: SLO-1 gating depends on EGL-19's m·h product directly rather than [Ca]_i

Examine slo1egl19.mod to see how Nicoletti encoded it. Match her approach if possible.

### Workflow

1. Read slo1egl19.mod, identify nanodomain encoding approach
2. Document architectural decision in `wave2/artifacts/slo1_coupled_architecture.md`
3. Translate to Brian2 matching Nicoletti's approach
4. Save to `wave2/channels/slo1_egl19_coupled.py`
5. Validate via voltage-clamp harness. Compare against NEURON reference.

### Phase E acceptance criteria

- Architectural decision documented
- Brian2 implementation exists
- Voltage-clamp Layer A passes >80% of holding potentials within tolerance

### Phase E failure modes

- Nicoletti's nanodomain encoding doesn't translate cleanly to Brian2: pause, document, surface for review (this is a real architectural question worth cross-session deliberation)
- Nanodomain coupling produces numerical instability: pause, document, do not fudge
- Tolerance fails despite clean translation: investigate whether F6 calibration regime mismatch is contributing (sub-membrane [Ca]_i is different regime than bulk calibration was tested in)

---

## Phase F — Gate 2 full evaluation on AVA cell

**Two decoupled cell constructions per the resolution above.**

### Component 2a — Voltage-clamp Layer A (channel kinetics in cell context)

**Brian2 cell construction:** AVA-like single compartment with Nicoletti's actual AVA channel subset that overlaps our essential set. Specifically: NCA + EGL-19 + leak. (Nicoletti's AVA also uses IRK and UNC-103, but those aren't in our essential set yet — exclude them. The Layer A check is on the subset Brian2 covers that Nicoletti also has.)

**NEURON reference:** Nicoletti's NEURON AVA cell, run with full parameters (Nicoletti's actual model). Note that NEURON has IRK and UNC-103 included; the comparison's interpretation accounts for this — the Brian2 cell with NCA+EGL-19+leak is **not expected to match NEURON's AVA exactly** because the full AVA channel set differs. Instead, run the comparison and document the divergence pattern. If Brian2 matches NEURON within tolerance for the subset-driven dynamics, that's still a positive integration check; if not, the divergence pattern informs whether IRK/UNC-103 are dominant in the protocol regime tested.

**Alternative (if cleaner Layer A is preferable):** construct a NEURON AVA reference with only NCA + EGL-19 + leak (mirror Brian2's subset). This requires modifying Nicoletti's AVA simulation script to suppress IRK/UNC-103 — local patch in `wave2/`, do not modify upstream. If this alternative is taken, Layer A becomes apples-to-apples.

The agent should attempt the alternative (NEURON reference matching Brian2's subset) if feasible — cleaner comparison. If patch is non-trivial, fall back to comparing Brian2 NCA+EGL-19+leak against full NEURON AVA and document the divergence pattern as expected (not a failure).

**Comparison protocol:** voltage-clamp at 11 holding potentials (-110 to +50 mV in 16 steps; standard Nicoletti protocol).

**Pass criterion:** voltage-feature ≤3 mV equivalent in current domain, >80% of holding potentials clear tolerance.

### Component 2b — Current-clamp plateau dynamics (architectural sufficiency)

**Brian2 cell construction:** AVA-like single compartment with **full 7-channel essential set** (EGL-19, SLO-1 isolated, SLO-1+EGL-19 coupled, SHK-1, SHL-1, NCA, KQT-3) + leak + Ca-pool.

**Channel densities:** use Nicoletti's published AVA channel densities for channels she provides (NCA, EGL-19). For channels not in Nicoletti's AVA (SLO-1, SHK-1, SHL-1, KQT-3), use reasonable defaults from Nicoletti's other cells (e.g., AIY's SLO-1+EGL-19 density, RIM's SHK-1 density) scaled by AVA capacitance. Document choices in `wave2/artifacts/gate2_ava_cell_construction.md`.

**Reference:** Mellem 2008 plateau targets — 20 mV plateau amplitude (range 15-25 mV), 600 ms plateau duration (range 400-800 ms), SLO-1-dominated termination (release tau substantially shorter than leak τ_m). NO NEURON reference for 2b.

**Comparison protocol:** Mellem 2008 current injection protocol — 50 pA × 100 ms at v_rest = -25 mV, with 200 ms settle and 1500 ms post-stim window.

**Pass criterion:** plateau amplitude in 15-25 mV range AND plateau duration in 400-800 ms range AND release-dynamics architectural_signature is 'active_termination' (not 'leak_dominated' or 'no_termination').

### Gate 2 outcomes (decision-grade for morning review)

- **2a-pass / 2b-pass:** Path A's cellular layer production-grade. Major Wave 2 milestone. Phase γ complete. Phase δ (network integration) is next work block.
- **2a-pass / 2b-fail:** **Condition 6 surfaces.** Channels work, architecture insufficient. Per architectural plan: **PAUSE for morning review, do NOT auto-trigger morphology fork.** This is the load-bearing decision the cross-session adversarial review pattern is designed for. Speculative fork results from Phase X (if it ran) become inputs to this review.
- **2a-fail:** Per-channel rollback territory. Document which channel(s) caused failure. PAUSE for morning review.

### Phase F acceptance criteria

- Both cell constructions documented
- Component 2a evaluated
- Component 2b evaluated
- Outcome classified per above
- Status file records full Gate 2 results

---

## Phase G — Run summary and morning materials

Aggregate all findings from Phases A-F (and X if triggered). Produce `wave2/artifacts/phase_beta_run2_summary.md` as morning-review entry point.

### Run summary structure

1. **Overall status:** completed-fully / partial-completed / paused-for-review / failed-environment
2. **Per-phase status table:** A through G (and X if triggered) with pass/fail/paused/skipped per phase
3. **F6 verdict** and implications
4. **Channels translated** (SHK-1, SHL-1, NCA, KQT-3, SLO-1 isolated, SLO-1+EGL-19 coupled — pass/fail/paused per)
5. **Gate 2 outcome** (2a/2b results, classified outcome)
6. **If Phase X triggered:** speculative architecture findings, prototype results if attempted, recommendations
7. **Architectural decisions made during run** (documented with reasoning)
8. **Issues requiring user attention** (load-bearing items for morning review)
9. **Recommended next actions** for subsequent Phase β work blocks
10. **Lessons learned** for future overnight runs (execution rate data, what worked, what didn't)

The run summary's structure must clearly distinguish:
- Phases completed cleanly (actionable: proceed)
- Phases paused for review (actionable: review and decide)
- Phases not started (actionable: defer to next overnight)
- Speculative work findings (actionable: review options, decide architectural direction)

---

## Failure modes and recovery (general)

**Environment failures:**
- venv broken, NEURON not loading: abort, document, do not proceed
- Compiled mods missing or corrupted: attempt re-compile, abort if fails

**Translation failures (any channel):**
- Brian2 syntax errors: debug, fix, retry. Document.
- Numerical instability (NaN/Inf): document parameters, pause for review. Do NOT fudge to make it work.
- Comparison divergence beyond tolerance: investigate (NMODL pattern? unit error? parameter mismatch?), pause if cause unclear.

**Validation harness failures:**
- NEURONReference wrapper crashes: investigate, pause, document
- Voltage-clamp harness produces unexpected results: validate against known-good case (run #1 EGL-19), if harness is buggy that's critical issue requiring abort

**Gate 2 specific failures:**
- Numerical issues prevent Gate 2 evaluation: pause, document, surface as infrastructure issue
- Gate 2b fails due to clear non-condition-6 reasons (e.g., channel density wrong): document, suggest fix, do not retry without authorization

**Speculative fork specific failures:**
- GNN prototype produces no meaningful results: document honestly, this is expected outcome, not failure
- Time runs out during speculative work: document state, hand off cleanly to morning review

**General principle:** pause-with-documentation always preferable to fabricate-completion or skip-ahead. Better to have 4 channels cleanly validated and 2 paused for review than 6 channels with hidden uncertainty.

---

## Infrastructure robustness requirements

All simulation calls have explicit timeouts (NEURON > 5 min triggers pause-and-document, Brian2 > 5 min same). Numerical stability checks (NaN/Inf detection) on all traces before computing comparison metrics. Memory monitoring (subagent should periodically check memory usage during long phases). Disk space verification before writing artifacts. Catch and document errors rather than crashing — failure produces diagnostic file, not termination.

State persistence pattern:
- Each subcheckpoint's outputs saved to disk before proceeding to next
- Checkpoint status JSON updated after each subcheckpoint
- If subagent crashes mid-checkpoint, restart can resume from last completed subcheckpoint
- Final overnight summary report aggregates all checkpoint statuses

---

## Methodology continuity items

**Findings file:** Continue extending `wave2/artifacts/phase_beta_findings.md` with new findings (F11+). Reference back to F1-F10 catalog. Use translation_patterns.md for systematized patterns.

**Decision log:** Every architectural choice (Phase A diagnostic interpretations, Phase E nanodomain encoding, Gate 2 cell construction, Phase X architectural sketches) documented with reasoning. Supports cross-session review.

**Pushback file:** `wave2/artifacts/phase_beta_run2_pushback.md` — populate during pre-flight reading if scope concerns surface. If populated, create `PAUSED_FOR_REVIEW.txt` marker file and pause.

**Speculative work isolation:** All speculative architecture work in `wave2/speculative/` subdirectory. Production code unchanged unless Phase F's outcome explicitly authorizes modifications (which doesn't happen autonomously — only after morning review).

---

## On time scoping

No total time estimate. Subcheckpoint-level rough orientation are commitments to scope, not commitments to duration. Run #1 executed roughly 5-10× faster than estimates; this run #2 has more novel work so the multiplier may be smaller.

Realistic outcomes:
- **Best case:** F6 verdict PRINCIPLED, all 6 channels translate cleanly, Gate 2 passes both components. Wave 2 cellular layer production-grade.
- **Expected case:** F6 verdict PARTIALLY_PRINCIPLED, most channels translate, Gate 2 surfaces some issues. Substantial Wave 2 progress with documented gaps for morning review.
- **F6 fudge case:** F6 verdict FUDGE_FACTOR triggers Phase X. Non-Ca channels translate, speculative fork investigates alternatives. Wave 2 cellular layer status uncertain pending morning review.
- **Partial completion:** any phase pauses for review. Run summary documents what's done, what's pending, what needs decision. Acceptable outcome.

Don't compress work to fit a felt time budget. If a phase deserves longer execution to do properly, take the time. If you complete all phases with capacity remaining, stop and surface for morning review rather than expanding scope.

---

## Output file structure

```
wave2/
├── translation_patterns.md                   # Phase B systematized catalog
├── channels/
│   ├── shk1.py                                # Phase C
│   ├── shl1.py                                # Phase C
│   ├── nca.py                                 # Phase C
│   ├── kqt3.py                                # Phase C
│   ├── slo1_iso.py                            # Phase D
│   └── slo1_egl19_coupled.py                  # Phase E
├── speculative/                              # Phase X if triggered
│   ├── gnn_architecture_sketch.md
│   ├── training_data_feasibility.md
│   ├── comparison_framework.md
│   ├── prototype/                            # X.1d if attempted
│   ├── multi_compartment_explicit.md
│   ├── neuroml2_native.md
│   └── x1_summary.md
└── artifacts/
    ├── checkpoints/
    │   ├── phase_a_status.json
    │   ├── phase_b_status.json
    │   ├── phase_c_status.json
    │   ├── phase_d_status.json (if reached)
    │   ├── phase_e_status.json (if reached)
    │   ├── phase_f_status.json (if reached)
    │   └── phase_x_status.json (if triggered)
    ├── f6_symbolic_decomposition.md
    ├── f6_geometry_analysis.md
    ├── f6_calibration_robustness.md
    ├── f6_diagnostic_synthesis.md
    ├── slo1_coupled_architecture.md (if Phase E reached)
    ├── gate2_ava_cell_construction.md (if Phase F reached)
    ├── phase_beta_findings.md                # Extended F1-F10 + new F11+
    ├── phase_beta_run2_pushback.md           # Only if pre-flight concerns
    ├── phase_beta_run2_summary.md            # Final morning-review entry
    └── PAUSED_FOR_REVIEW.txt                 # Only if paused
```

---

## Completion criteria

The overnight run is **fully successful** if:
- Phase A completes with verdict
- Phase B completes
- Phase C completes (all 4 non-Ca channels)
- Either: Phases D-F complete with Gate 2 pass
- Or: Phase X triggers and produces documented speculative findings (F6 fudge case)
- Phase G run summary written

The overnight run is **partial-successful** if:
- Phase A completes
- Phase C completes (or some channels)
- Some subsequent phases pause cleanly for review

The overnight run is **failed** if:
- Phase A fails or aborts (foundation isn't validated)
- Environment issues prevent execution

Partial completion with cleanly documented diagnostics is more valuable than fabricated full completion.

---

## Invocation chaining for multi-invocation overnight execution

Your runtime envelope is ~1-3 hours per invocation. The work scoped here likely exceeds one invocation. To maximize overnight progress, support resume-from-state across multiple invocations.

### Resume protocol

At start of every invocation (including the first), read `wave2/artifacts/checkpoints/run2_state.json` if it exists.

```json
{
  "last_completed_phase": "Phase A" | "Phase B" | "Phase C" | "Phase D" | "Phase E" | "Phase F" | "Phase G" | "none",
  "last_completed_subcheckpoint": "<identifier>",
  "f6_verdict": "PRINCIPLED" | "PARTIALLY_PRINCIPLED" | "FUDGE_FACTOR" | null,
  "phase_x_triggered": true | false,
  "phase_x_completed": true | false,
  "channels_translated": ["egl19", "shk1", "shl1", ...],
  "next_action": "<short description of where to resume>",
  "invocation_count": <int>
}
```

If `run2_state.json` doesn't exist: this is invocation 1, start at Phase A.
If it exists: this is invocation 2+, resume from `next_action`.

### State write protocol

After every subcheckpoint completion, update `run2_state.json` atomically (write to `run2_state.json.tmp`, then rename). This survives crashes and runtime-limit terminations.

### End-of-invocation behavior

The agent doesn't know exactly when its runtime envelope expires, so write state defensively after every subcheckpoint. When runtime expires, current state file represents last completed work — next invocation resumes cleanly from there.

If the agent reaches a natural pause point (Phase A complete, decision tree synthesized, ready to start Phase B) and senses it's been running for a while, it can choose to write a clean handoff state and stop, rather than starting a new phase that won't finish. This is optional — the resume protocol works either way.

### Continuation invocation trigger

After this invocation completes (whether normally or by runtime limit), the user will manually trigger the next invocation by re-running this same prompt. The resume protocol handles continuation automatically — no prompt modifications needed between invocations.

If `run2_state.json` shows all phases complete (`last_completed_phase: "Phase G"`), the agent immediately writes a "run already complete" status and exits without doing duplicate work.

### Invocation completion criteria

Each invocation should complete by:
- Writing final state to `run2_state.json`
- If reaching the end of all phases: writing `phase_beta_run2_summary.md` (final morning-review entry)
- If pausing mid-run: writing brief `phase_beta_run2_invocation_<N>_summary.md` documenting what this invocation accomplished

### On the multi-invocation pattern

This is a methodological pattern worth establishing for Wave 2. Phase β work is multi-night, multi-invocation. Each invocation is bounded but the cumulative pipeline is long. Resume protocol via state file is a clean way to handle this.

If invocation chaining produces issues (state file corruption, resume confusion, state-machine bugs), surface in the run summary and we'll iterate the pattern for Phase β subsequent overnights.

**Important: this invocation chaining only handles within-overnight chaining. Across-overnight is still manual** (user wakes up, reviews, decides what to deploy next). The chaining lets the user trigger the next invocation manually during the overnight if convenient, but doesn't auto-trigger.

---

## Final framing

This overnight run is the most ambitious Phase β work to date. Best case: Wave 2's cellular layer becomes production-grade end-to-end with Gate 2 cleared. Expected case: substantial channel translation progress with F6-related uncertainties documented for morning review. F6-fudge case: speculative architecture investigation surfaces alternatives that inform Wave 2 trajectory.

All three cases produce decision-grade information. The methodology discipline today has produced ~15 substantive catches; this run extends the pattern to autonomous execution at expanded scope.

Apply the discipline. Pause-and-document on architectural questions. Document, don't fabricate. Build infrastructure that will be trustworthy. Speculative work in speculative/ directory, production work in production paths.

Standing by for pre-flight pushback (file-based since user is asleep). If concerns surface in pre-flight that warrant cross-session review, pause and wait for morning. Otherwise execute through Phase A and continue per phase structure.
