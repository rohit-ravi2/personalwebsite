# Phase β overnight run — Wave 2 channel translation foundation + EGL-19

**Scope:** Option A — foundation infrastructure (CP1) + EGL-19 translation (CP2) + EGL-19 Gate 2a in cell context (CP3). CP4-CP6 (SLO-1, SHK-1, SHL-1, NCA, KQT-3) are subsequent overnights.

**Status:** Wave 2 first concrete engineering work block proving Path A operates at the implementation level. Phase α built infrastructure; Phase β-pre v1/v2/v3 cleared condition-3 substantively (Layer B residuals 3-5 mV); this run validates that translation actually works at the channel level.

**Mode:** overnight, autonomous, file-based pause-and-wait. User is asleep. Cross-session adversarial review remains idle until morning.

---

## Context: Wave 2 status and what this overnight run accomplishes

Phase α completed cleanly (NEURON installed, 22 mod files compiled, validation harnesses built, smoke-tested on simple cases). Phase β-pre v1/v2/v3 cleared condition-3 substantively — Nicoletti's NEURON code reproduces her published model figures within 3-5 mV (Layer B comparison), confirming ground-truth correctness for downstream Brian2 translation work. The architectural plan committed to Path 3A (Brian2 + parameter import from Nicoletti 2024) with Gate 2 as two-component (channel kinetics correctness + architectural sufficiency).

This overnight run is Phase β proper, scoped to **Option A: foundation + EGL-19**. It produces:
- Updated harness infrastructure ready for channel-by-channel translation work
- Ca-pool subsystems (cadiff, caintra1) translated and validated
- NEURONReference wrapper enabling Layer A comparisons (Brian2 vs NEURON reference)
- EGL-19 channel translated and Gate 2a-validated as first real channel
- Gate 2a evaluation in cell context (EGL-19 inserted into AVA cell, voltage-clamp protocol comparison vs NEURON reference)

**Out of scope for this overnight run:**
- SLO-1 isolated, SLO-1+EGL-19 coupled, SHK-1, SHL-1, NCA, KQT-3 channels (subsequent Phase β work blocks)
- Gate 2b evaluation (architectural sufficiency, requires full essential set)
- Network integration (Phase δ, after Gate 2 fully cleared)
- Compartmental morphology integration (Wave 3 territory unless condition 6 surfaces)
- Any modifications to production simulator code outside `scripts/brain/wave2/`

**Working environment:**
- Isolated venv at `~/venvs/wave2-neuron/` with NEURON + Brian2 installed
- Project code at `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Nicoletti 2024 source at `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- Compiled mods in Nicoletti's directory, harness infrastructure in wave2/
- Phase α completion report at `wave2/phase_alpha_report.md` for context
- Architectural plan at `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — read fully before starting

---

## Strategic positioning: methodology continuity for overnight execution

Today's pattern of cross-session adversarial review with pre-flight pushback, mid-flight surfacing of findings, and stop-and-ask discipline has produced approximately 15+ substantive methodological catches across Phase α, β-pre v1/v2/v3, and harness fitness assessment. This overnight run continues that pattern, adapted for autonomous execution.

**Adaptation for overnight context:**

- **Pre-flight pushback to file** rather than to user. Read the prompt fully before starting work. If you find scope concerns, hidden assumptions, or items that warrant cross-session discussion, write them to `wave2/artifacts/phase_beta_pushback.md` and pause for go-ahead via a clear marker file (`wave2/artifacts/PAUSED_FOR_REVIEW.txt`). The user will read in the morning and either resolve concerns or authorize proceeding.

- **Mid-flight surfacing via documentation, not interruption.** Findings that would warrant real-time discussion in supervised work go into `wave2/artifacts/phase_beta_findings.md` as they emerge. Continue work if a clear best-path exists. Pause-and-document if architectural questions surface that you can't cleanly resolve.

- **Stop-and-pause vs stop-and-ask.** When uncertainty surfaces:
  - Implementation questions with clear best-path → proceed, document choice in findings file
  - Architectural questions or load-bearing decisions → pause, write decision-needed entry to findings file, stop at next safe checkpoint
  - Validation failures that suggest invalidation conditions from the architectural plan → pause immediately, do not proceed past the failure

- **Pause-and-wait is success.** If you reach CP3 cleanly and CP4 surfaces a load-bearing question, completing CP1-CP3 and pausing at CP4 with documented diagnostics is the correct outcome. The overnight run's success criterion is "as much progress as could be cleanly executed," not "all checkpoints completed."

**Subagent autonomy bounds:**

You have full permission for: code creation, code modification within `scripts/brain/wave2/`, file creation, package operations within the wave2 venv, running NEURON/Brian2 simulations, comparing outputs, computing metrics.

You do NOT have permission for: modifying production simulator code outside `scripts/brain/wave2/`, modifying Nicoletti's upstream code (use local patches only, as established in v3), filing GitHub issues, expanding scope beyond Option A, making architectural decisions on load-bearing matters without pausing.

---

## Checkpoint structure

Six checkpoints. **This overnight run executes CP1, CP2, CP3 only.** CP4-CP6 are listed for context.

Each checkpoint produces:
- Output files (translation code, validation results, status)
- Validation against acceptance criteria
- Status file: `wave2/artifacts/checkpoints/CP{N}_status.json` with pass/fail/paused-for-review classification

If checkpoint passes: proceed to next.
If checkpoint fails: pause with diagnostic.
If checkpoint surfaces architectural question: pause with decision-needed entry.

State checkpointed incrementally. If subagent crashes mid-checkpoint, partial state survives for restart.

---

## CP1 — Foundation: harness iteration + Ca-pool translation

This is the load-bearing checkpoint. Everything else depends on it.

### CP1.A: Harness iteration

**1. Build NEURONReference wrapper class.**

Create `wave2/neuron_reference.py`. Class signature:

```python
class NEURONReference:
    """Programmatic interface to Nicoletti NEURON models for Layer A comparison."""
    
    def __init__(self, cell_name: str, mods_path: str = "..."):
        """Initialize NEURON, load mods, instantiate cell per Nicoletti's wrapper."""
    
    def voltage_clamp(self, holding_potentials: list[float], duration_ms: float, 
                       capacitance_pF: float = 100.0) -> dict:
        """Run voltage-clamp protocol. Return per-protocol-point current traces + features."""
    
    def current_clamp(self, injection_pa: float, injection_duration_ms: float,
                      settle_ms: float, post_ms: float, v_rest_mv: float) -> dict:
        """Run current-clamp protocol. Return voltage trace + plateau features."""
    
    def cleanup(self):
        """Reset NEURON state for next call."""
```

Returned dict structure should mirror what Brian2 harnesses produce so Layer A comparison code reuses both consistently.

Cells the wrapper must support: AVAL, AIY, RIM (existing in Nicoletti repo). AVAR with UNC103 patch (workaround documented in v3, patch at `wave2/avar_unc103_patch.py` if it exists, else port from v3 work).

Validation: wrapper successfully runs Nicoletti's AVAL voltage-clamp protocol, produces output matching what Phase β-pre v3 captured in `comparison_validation_results_v2.json`. If wrapper's output diverges from v3-captured values, surface as harness bug.

**2. Update voltage-clamp harness tolerance metric.**

The existing tolerance metric in `wave2/voltage_clamp_harness.py` is single 5% with 1e-9 floor — inherits the small-denominator pathology that v1/v2/v3 hit three times.

Replace with current-domain analog of v3's voltage-feature gate:
- Per-feature comparison: peak current per holding potential, IV curve points
- Tolerance: 5% relative above 10% of peak current threshold, absolute ≤5% of peak below threshold
- Implementation: `max_divergence = max(|a-b| / max(|a|, |b|, 0.1 * peak_current))`
- Per-panel pass: >80% of holding potentials must clear the threshold
- Document tolerance interpretation in code comments + harness fitness report

**3. Add `current_clamp_layer_a_compare` function to plateau harness.**

The existing `plateau_harness.py` runs Brian2 only and compares against Mellem 2008 targets (Gate 2b verification). For CP3, we need Brian2 cell vs NEURON reference cell comparison (Layer A).

New function signature:

```python
def current_clamp_layer_a_compare(
    brian2_cell,
    neuron_reference: NEURONReference,
    cell_name: str,
    injection_pa: float = 50.0,
    injection_duration_ms: float = 100.0,
    settle_ms: float = 200.0,
    post_ms: float = 1500.0,
    v_rest_mv: float = -25.0,
) -> dict:
    """Run same protocol on Brian2 cell and NEURON reference. Compare voltage-feature."""
```

Tolerance: voltage-feature ≤3 mV residual at peak voltage + plateau amplitude per timepoint. >80% of timepoints must pass per panel. Timing features (time-to-peak, settling time) reported as warn-only diagnostics.

**4. Smoke tests on updated harnesses.**

Carry forward Phase α smoke tests. Add new smoke tests for:
- NEURONReference wrapper (instantiate AVAL, run voltage-clamp at 3 holding potentials, verify output structure)
- Updated voltage-clamp tolerance metric (apply to known-good case, verify metric returns expected pass; apply to known-bad case, verify metric returns expected fail)
- `current_clamp_layer_a_compare` (test on simple leak-only cell against simple NEURON leak-only cell, verify they match within tolerance)

All smoke tests must pass before CP1.A is complete.

### CP1.B: Ca-pool translation

**Architectural commitment: eqs-string encoding for Ca-pool subsystems.**

Nicoletti's models are single-compartment cylindrical. Ca-pool dynamics encoded as eqs-string in Brian2 (matching source structure) minimizes translation artifacts and supports faster validation. If condition 6 surfaces and morphology fork triggers, separate-subsystem encoding becomes part of morphology integration work — addressed then with multi-compartment context, not speculatively.

Document this commitment in `wave2/calcium_pool.py` module docstring.

**5. Translate cadiff.mod to Brian2.**

Read the .mod file. Identify state variables, kinetic equations, parameters. Translate to Brian2 equation string format. Save to `wave2/calcium_pool.py` (eqs-string format, parameter dict).

Validation: voltage-clamp protocol comparing Brian2 cadiff implementation against NEURON cadiff under identical Ca-injection scenarios. Use updated voltage-clamp harness with current-domain tolerance metric. Pass: >80% of test points within 5% relative or absolute tolerance.

**6. Translate caintra1.mod to Brian2.**

Same workflow. Save additional eqs-string + parameters to `wave2/calcium_pool.py`.

Validation: same approach. Pass criterion same.

**7. Combined Ca-pool validation.**

Cells using EGL-19 will use both cadiff and caintra1 together (EGL-19 reads [Ca]_i from caintra1, which is fed by cadiff diffusion). Test combined Ca-pool subsystem with synthetic Ca-injection, verify Brian2 output matches NEURON output within tolerance.

If combined validation passes but individual cadiff/caintra1 validations failed: surface as harness bug or NMODL translation issue.
If individual validations passed but combined fails: surface as architectural issue with how the two subsystems compose. May indicate the eqs-string approach has limits we didn't anticipate.

### CP1 acceptance criteria (all must pass)

- NEURONReference wrapper instantiates and runs all required cells
- Voltage-clamp tolerance metric updated and smoke-tested
- `current_clamp_layer_a_compare` function added and smoke-tested
- All Phase α smoke tests still pass with updated harnesses
- cadiff Brian2 implementation matches NEURON within tolerance
- caintra1 Brian2 implementation matches NEURON within tolerance
- Combined cadiff+caintra1 validation passes against NEURON reference

### CP1 status file output

```json
{
  "checkpoint": "CP1",
  "status": "pass" | "fail" | "paused_for_review",
  "subcheckpoints": {
    "CP1.A.1_neuron_reference": { ... },
    "CP1.A.2_voltage_clamp_tolerance": { ... },
    "CP1.A.3_layer_a_compare": { ... },
    "CP1.A.4_smoke_tests": { ... },
    "CP1.B.5_cadiff_translation": { ... },
    "CP1.B.6_caintra1_translation": { ... },
    "CP1.B.7_combined_capool": { ... }
  },
  "issues_surfaced": [...],
  "next_action": "proceed_to_CP2" | "pause_pending_review" | "abort"
}
```

---

## CP2 — EGL-19 channel translation

**Out of scope reminder:** This is the ONLY ion channel translated in this overnight run. SLO-1, SHK-1, SHL-1, NCA, KQT-3 are subsequent Phase β work blocks.

### Workflow

1. Read `nicoletti_2024/.../egl19.mod` carefully. Identify gating variables (m, h), kinetic parameters, voltage dependence, Ca-dependent inactivation if applicable.
2. Translate to Brian2 equation string format. Channel reads V from cell, reads [Ca]_i from cadiff/caintra1 system.
3. Save to `wave2/channels/egl19.py` (channel module). Module structure:
   - Equation string constant
   - Parameter dictionary
   - Function to attach channel to a NeuronGroup (insert into eqs, register I_EGL19 contribution)
4. Validation: voltage-clamp harness comparing Brian2 EGL-19 in test cell vs NEURON EGL-19 in reference cell. Use updated tolerance metric. Pass: >80% of holding potentials clear voltage-feature ≤3 mV equivalent on current-domain (translate to current-domain tolerance: per-feature residual relative to peak current).

### CP2 acceptance criteria

- EGL-19 Brian2 implementation files exist (channel module + parameters)
- Voltage-clamp validation passes >80% of holding potentials within tolerance
- IV curve from Brian2 implementation matches NEURON reference within tolerance
- Time-to-peak and inactivation kinetics reported (warn-only diagnostics if they don't match — surface in findings file)

---

## CP3 — Gate 2a evaluation: EGL-19 in cell context

The above CP2 validates EGL-19 in isolation. CP3 validates it in cell context — EGL-19 inserted into AVA-like Brian2 cell with leak + Ca-pool, run voltage-clamp protocol matching Nicoletti's, compare against NEURON AVA cell with same channel set.

**Important architectural decision:** AVA's full channel complement is EGL-19 + SLO-1 + multiple K channels + leak + Ca-pool. CP3 tests EGL-19 + leak + Ca-pool only (CP2-validated subset), NOT full AVA. Gate 2a's "channel kinetics correct" is the criterion, not "AVA reproduces Nicoletti's full AVA" — that's downstream after all 7 essential channels are translated.

### Workflow

1. Construct minimal Brian2 cell: leak + cadiff + caintra1 + EGL-19. Same passive parameters as Nicoletti's AVA (capacitance, leak conductance, leak reversal). 
2. Construct equivalent NEURON reference: NEURONReference wrapper with same channel subset (leak + Ca-pool + EGL-19 only — may require local NEURON cell construction since Nicoletti's AVA includes more channels).
3. Run voltage-clamp protocol on both. Compare currents using updated voltage-clamp harness.
4. Run current-clamp protocol on both (Mellem-style 50 pA injection). Compare voltage trajectories using `current_clamp_layer_a_compare`.
5. If both protocols pass tolerance: Gate 2a cleared on EGL-19 in cell context.

### CP3 acceptance criteria

- Voltage-clamp Layer A comparison: >80% of holding potentials clear tolerance
- Current-clamp Layer A comparison: voltage-feature ≤3 mV on peak + plateau amplitude, >80% of timepoints clear
- Timing features reported as diagnostics

**Note:** CP3 will likely show that EGL-19 alone doesn't produce sustained plateau (Mellem amplitude target requires SLO-1 termination dynamics, which we don't have yet). This is expected, NOT a failure. CP3 validates implementation correctness, not phenotype reproduction. Phenotype reproduction is Gate 2b territory after full essential set is translated.

---

## CP4-CP6 (NOT part of this overnight run)

- CP4: SLO-1 isolated translation (next overnight)
- CP5: SLO-1+EGL-19 coupled translation (next overnight)
- CP6: SHK-1, SHL-1, NCA, KQT-3 translations (subsequent overnights)

If CP3 passes cleanly and substantial overnight time remains (e.g., agent estimates 4+ hours of capacity), pause and surface for review rather than starting CP4. Don't expand scope mid-flight.

---

## Failure modes and recovery

**Environment failures (CP1.A.1 setup):**
- NEURON wrapper fails to instantiate cells: investigate, document, pause for review
- Existing harnesses fail smoke tests they previously passed: surface as critical issue, abort

**Translation failures (CP1.B.5/6, CP2):**
- Brian2 equation string syntax errors: debug, fix, retry. Document the issue.
- Brian2 numerical instability (NaN/Inf in traces): document parameters causing instability, pause for review. Do NOT fudge parameters to make it work.
- Comparison divergence beyond tolerance: investigate. Common causes: NMODL gotcha not yet documented, parameter format mismatch, numerical method choice. Document in findings, pause if architectural decision needed.

**Layer A failures (CP3):**
- Brian2 cell vs NEURON cell divergence beyond tolerance: investigate. May indicate:
  - Channel implementation bug (translation incorrect) — debug
  - Cell construction mismatch (different leak parameters, Ca-pool initialization, etc.) — document and align
  - Architectural insufficiency surfacing early (condition 6 territory) — surface immediately, pause

**Infrastructure failures:**
- NEURON simulation timeout (>5 minutes): kill, document, pause
- Brian2 simulation timeout (>5 minutes): kill, document, pause
- Memory exceeded: document, pause
- Disk space issues: document, pause

**General principle:** Document, don't fabricate. If something doesn't work, the honest finding is more valuable than a glossed-over result.

---

## Infrastructure robustness requirements

All simulation calls must have explicit timeouts. Numerical stability checks (NaN/Inf detection) on all traces before computing comparison metrics. Memory monitoring during long runs. Disk space verification before writing artifacts. Catch and document errors rather than crashing — failure produces diagnostic file, not termination.

State persistence pattern:
- Each subcheckpoint's outputs saved to disk before proceeding to next
- Checkpoint status JSON updated after each subcheckpoint
- If subagent crashes mid-checkpoint, restart can resume from last completed subcheckpoint
- Final overnight summary report aggregates all checkpoint statuses

---

## Methodology continuity items

**Findings file:** `wave2/artifacts/phase_beta_findings.md` — running log of:
- NMODL gotchas discovered during translation (extends Phase α's catalog)
- Architectural decisions made and reasoning
- Surprises (positive or negative) about Brian2/NEURON behavior
- Items that should inform Phase β subsequent work blocks

**Pushback file:** `wave2/artifacts/phase_beta_pushback.md` — populate during pre-flight reading if scope concerns surface. If populated, create `PAUSED_FOR_REVIEW.txt` marker file and pause.

**Decision log:** Every architectural choice (eqs-string encoding, tolerance interpretation, cell construction parameters, etc.) documented with reasoning. This supports cross-session review and methodology paper case studies.

---

## Output format and morning review materials

Files produced by this overnight run:

```
wave2/
├── neuron_reference.py                  # NEURONReference wrapper (CP1.A.1)
├── voltage_clamp_harness.py             # Updated tolerance (CP1.A.2)
├── plateau_harness.py                   # Added Layer A compare (CP1.A.3)
├── calcium_pool.py                      # cadiff + caintra1 translations (CP1.B)
├── channels/
│   └── egl19.py                         # EGL-19 translation (CP2)
└── artifacts/
    ├── checkpoints/
    │   ├── CP1_status.json
    │   ├── CP2_status.json
    │   └── CP3_status.json
    ├── phase_beta_findings.md
    ├── phase_beta_pushback.md           # Only if pre-flight concerns
    ├── phase_beta_run_summary.md        # Overall run report (final output)
    └── PAUSED_FOR_REVIEW.txt            # Only if paused
```

**Phase β run summary** (`phase_beta_run_summary.md`) is the morning-review entry point. Should contain:

- Overall status: completed / partial / paused
- Per-checkpoint pass/fail summary
- Key findings surfaced during execution
- Architectural decisions made
- Issues requiring user attention
- Recommended next actions for subsequent Phase β work blocks
- Lessons learned for future overnight runs

---

## On time scoping

No total time estimate. Subcheckpoint-level estimates above are rough orientation, not commitments. The work decomposes into bounded tasks with clear acceptance criteria. Duration depends on session execution patterns we'll learn from this run.

If during execution any subcheckpoint takes substantially longer than its rough estimate (e.g., 3-4× longer), pause and document. May indicate issues we should know about for subsequent overnights.

---

## Completion criteria

The overnight run is **successful** if:
- CP1 fully completes (foundation infrastructure validated) AND CP2 fully completes (EGL-19 translated and validated in isolation) AND CP3 fully completes (EGL-19 Gate 2a-cleared in cell context)

The overnight run is **partial-successful** if:
- CP1 completes but CP2 or CP3 paused for review
- Partial work cleanly executed and documented; subsequent run completes the remaining checkpoints

The overnight run is **failed** if:
- CP1 fails or pauses (foundation isn't validated; no further work meaningful until CP1 cleared)

In all cases, the run summary documents what happened and what should happen next. There is no "wasted" overnight if work was honest and documented — partial completion with diagnostics is more valuable than full completion with hidden bugs.

---

## Final framing

This overnight run is the first concrete engineering work block proving Path A operates at the implementation level. Phase α validated infrastructure could be built. Phase β-pre validated ground-truth correctness existed. This run validates that translation actually works — that Nicoletti's NEURON channels can become Brian2 channels that match within tolerance.

If this run succeeds, Path A is empirically vindicated at the first channel level. Subsequent Phase β work blocks become routine extensions of the same pattern across SLO-1, SHK-1, SHL-1, NCA, KQT-3.

If this run surfaces invalidation conditions from the architectural plan (translation systematically fails, Brian2 can't represent the kinetics, condition 6 surfaces early), that's the kind of decision-grade information Wave 2 was designed to produce — better to find out at first channel than at fifth.

Apply the discipline today's work has established. Pre-flight pushback discipline. Mid-flight surfacing of findings. Stop-and-pause when uncertainty surfaces. Document, don't fabricate. Build infrastructure that will be trustworthy across the rest of Wave 2.

Standing by for pre-flight pushback (file-based since user is asleep). If concerns surface in pre-flight that warrant cross-session review, pause and wait for morning. Otherwise execute through CP1, CP2, CP3 as scoped, producing the foundation + first channel deliverable.

The other sessions remain idle. This is a single-session overnight run.
