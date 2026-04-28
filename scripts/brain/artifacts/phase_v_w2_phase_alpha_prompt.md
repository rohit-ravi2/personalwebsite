# Phase α — Setup work block (Wave 2 first engineering session)

You are the engineering session executing Phase α of Wave 2 of the C. elegans biophysical simulator project. This is the first concrete engineering work block following the Wave 2 architectural commitment. You are operating in the cross-session methodology that has been load-bearing throughout the recent investigation work — pre-flight pushback before substantive moves, mid-flight surfacing of findings rather than batching to the end, stop-and-ask discipline when assumptions stop holding.

You have full user permission for package installs, file creation, and framework setup *for this phase*. Permission does not extend to modifying the production simulator environment (`~/miniconda3/envs/ml/` or equivalent project conda env) — Phase α work is isolated.

---

## Strategic context (read first)

The project committed to **Path 3A** (Brian2 backend + per-channel parameter import from Nicoletti 2024) as the Wave 2 primary architectural path. Path 3A's value proposition: translate Nicoletti's 22 NEURON `.mod` channel implementations into Brian2 equation form, preserving all existing project infrastructure (LIFBrain/GradedBrain, MuJoCo, scenario JSON, dashboard, FSM, classifier, modulator layer, sensory cascades) while gaining worm-validated biophysical channel kinetics for AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 (+ AWC, RMD from 2019).

Phase α is the *setup* work block — it does NOT do channel translation (Phase β), cellular validation against Mellem 2008 (Gate 2), or network integration (Phase δ). Phase α delivers the infrastructure those later phases require.

**Read before starting any substantive work:**

1. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_architectural_plan.md` — project-level commitment document. Especially the "Acceptance criteria (gates)" section and the "What would invalidate Path A?" subsection (condition 6 — architectural-bottleneck failure mode is the diagnostic distinction between Gate 2a and Gate 2b).
2. `~/Desktop/website/personalwebsite/scripts/brain/artifacts/phase_v_w2_backend_architecture_analysis.md` — Path 3A rationale, including per-channel translation pattern verified against `egl19.mod`.
3. `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` — Nicoletti 2024 codebase. Inspect directory structure, identify which `.mod` files are channels vs synapses vs Ca pools, locate her published validation traces (likely in figure-reproduction scripts).

---

## Six concrete deliverables

Each deliverable has explicit acceptance criteria. Do NOT mark a deliverable complete until its criteria pass empirically.

### Deliverable 1: NEURON installed and version-verified in isolated venv

- Create a fresh venv at `~/venvs/wave2-neuron/` (do NOT touch the production brain conda env)
- Install NEURON, Brian2, numpy, scipy, matplotlib in this venv
- Verify NEURON loads: `python -c "from neuron import h; print(h.nrnversion())"`
- Verify Brian2 loads: `python -c "import brian2; print(brian2.__version__)"`
- Document exact installed versions in `phase_alpha_report.md`

**Acceptance:** both libraries import without error, versions logged.

### Deliverable 2: All 22 of Nicoletti's `.mod` files compile cleanly

- Locate Nicoletti's `.mod` files (likely in `nicoletti_2024/` root or a `channels/` subdirectory)
- Run `nrnivmodl` (NEURON's mod compiler) against the directory
- Verify zero compilation errors. Warnings are acceptable; errors are not.
- If a file fails compilation, surface it immediately rather than silently dropping it — this is mid-flight surfacing.

**Acceptance:** `nrnivmodl` exits 0; the resulting compiled mechanism library (`.so` or `.dll`) loads in NEURON; `h.MechanismType(0)` lists all 22 channel mechanisms.

### Deliverable 3: ≥ 2-3 of Nicoletti's published validation traces reproduce in local NEURON within 1% tolerance

This is the **ground-truth correctness verification**. It validates condition 3 of the invalidation list ("Nicoletti's models don't reproduce Mellem 2008 cellular targets in NEURON either"). If we can't reproduce her own published traces using her own code, downstream translation is moot.

- Identify which of her 9 simulation scripts produces published-figure traces (AWC, RMD, AVAL, AVAR, AIY, RIM, VA5, VB6, VD5 are candidates)
- Pick at least 2-3 of these neurons. AVAL is mandatory if available — it is the cell with Mellem 2008 plateau as Gate 2b's reference target.
- Run her unmodified script. Capture voltage traces at the timepoints the published figures show.
- Compare against published-figure values. Tolerance: 1% on V trajectory at sampled timepoints.

**Acceptance:** ≥ 2 neurons reproduce within 1%. Document any discrepancies with hypothesis (parameter drift between published and code? unit confusion? protocol detail in code vs figure?). If discrepancies exceed 5%, **stop and surface to user** — this is the condition-3 invalidation signature and would gate Phase β.

This deliverable is the load-bearing item in Phase α. The other 5 deliverables are infrastructure; this one tests whether the source of truth holds.

### Deliverable 4: Brian2 voltage-clamp validation harness — Gate 2a infrastructure

Reusable harness that takes a Brian2 channel implementation + a NEURON reference + a voltage-clamp protocol and returns a divergence metric.

API design: prototype-first. Start with the simplest case (probably a passive leak channel, then EGL-19 single-channel — even though EGL-19 won't be translated until Phase β, you can construct a Brian2 reference from a known equation form for harness smoke-testing). Don't pre-design abstractions for all 7 essential channels — let Phase β's first real use stabilize the API.

Suggested API skeleton (refine empirically):

```python
def voltage_clamp_compare(
    brian2_model: brian2.NeuronGroup,
    neuron_reference: callable,  # function returning (t, V, I) arrays
    holding_potentials: list[float],  # mV
    duration: float,  # ms
    dt: float = 0.025,  # ms
    tolerance: float = 0.05,  # 5% per Gate 2a
) -> dict:
    """Returns: {'pass': bool, 'max_divergence': float, 'per_step': [...]}"""
```

**Acceptance:** harness runs end-to-end on at least one Brian2 channel + one NEURON reference; returns a diagnostic. Smoke-test passes; failure modes (units mismatch, dt mismatch) produce informative errors rather than silent wrong answers.

### Deliverable 5: Brian2 current-clamp plateau harness — Gate 2b infrastructure

Distinct from Deliverable 4 because Gate 2b tests *plateau dynamics* (amplitude, duration, termination behavior on stimulus release) — not steady-state IV-curve matching. This is the architectural-sufficiency probe per condition 6.

API design: prototype-first, same discipline as #4. Skeleton:

```python
def current_clamp_plateau(
    brian2_cell: brian2.NeuronGroup,
    stim_amp: float,  # pA
    stim_duration: float,  # ms
    total_duration: float,  # ms (must include sufficient post-stim window for termination)
    targets: dict,  # {'amplitude_mv': (15, 25), 'duration_ms': (400, 800), ...}
) -> dict:
    """Returns: {'pass': bool, 'measured': {...}, 'release_dynamics': {...}}"""
```

The release-dynamics block is the load-bearing piece — it tests whether plateau collapse on stimulus removal is dominated by SLO-1-mediated termination (correct) or by leak τ_m (architectural-insufficiency signature). For Phase α smoke-testing, you don't yet have the imported channels — substitute a known-passing analytical case (a deliberately constructed scaffold that should pass) and a known-failing case (deliberately under-resourced leak) to verify the harness distinguishes them.

**Acceptance:** harness runs end-to-end; correctly classifies a known-good case as pass and a known-bad case as fail; release-dynamics diagnostic produces meaningful output.

### Deliverable 6: Both harnesses smoke-tested on simple known cases

- `smoke_tests.py` runs both harnesses on synthetic test cases
- Voltage-clamp smoke test: leak-only cell, expect IV linear, harness reports correct slope match
- Current-clamp smoke test: known-passing scaffold (passes 2b targets), known-failing scaffold (fails 2b — leak-dominated). Harness correctly classifies both.

**Acceptance:** smoke tests pass on first run after harness implementation stabilizes. If they don't pass, treat as harness bug, not user error.

---

## File deliverables (in `~/Desktop/website/personalwebsite/scripts/brain/wave2/`)

Create the directory if it doesn't exist. Files:

- `setup_neuron.py` — venv setup verification, NEURON install check, Nicoletti compilation runner. Idempotent — re-running should be safe.
- `reference_validation.py` — Nicoletti reference reproduction (Deliverable 3). Captures traces, compares against published figures.
- `voltage_clamp_harness.py` — Gate 2a infrastructure (Deliverable 4).
- `plateau_harness.py` — Gate 2b infrastructure (Deliverable 5).
- `smoke_tests.py` — both harnesses' smoke tests (Deliverable 6).
- `phase_alpha_report.md` — completion report (see Output format below).

Code style: match the project's existing style in `scripts/brain/`. Inspect `graded_brain_h_kca.py`, `compartmental_neurons_kca.py` for patterns. Brian2 unit handling, namespace dict conventions, numpy idioms.

---

## Scope discipline — what is NOT in Phase α

Do NOT do any of these. They are explicitly out of scope:

- **No translation of any of Nicoletti's `.mod` files into Brian2.** That is Phase β. Phase α stops at the harness infrastructure.
- **No cellular validation against Mellem 2008 plateau.** That is Gate 2 in Phase γ. Phase α only smoke-tests the harness on synthetic known cases.
- **No integration with the production simulator** (`graded_brain_h_kca.py`, `compartmental_neurons_kca.py`, `LIFBrain`, etc.). Phase δ.
- **No CeNGEN-coupled channel densities, no modulator layer changes, no scenario pipeline modifications.** Phase α is infrastructure; the production simulator is untouched.
- **No license verification publication-prep work.** Production-prep gate, not Phase α.
- **No premature API stabilization** across all 7 essential channels. Prototype-first, document arbitrary decisions for Phase β refactor.
- **Do NOT modify the production brain conda env.** Phase α uses the isolated `~/venvs/wave2-neuron/` venv exclusively.

If you find yourself doing any of the above, stop and surface to user. The scope was set deliberately.

---

## Methodology continuity

The cross-session pattern that's been load-bearing today applies here:

- **Pre-flight pushback before each substantive sub-step.** If a deliverable's plan has a load-bearing assumption, surface it before executing rather than batching findings to the end. Especially: if Nicoletti's directory structure differs from expectation, if her `.mod` files use NMODL features that look unusual, if her validation scripts depend on missing data files — surface these immediately.
- **Mid-flight surfacing.** If Deliverable 3 (reference reproduction) finds discrepancies > 5%, stop and surface. Do not silently proceed to Deliverables 4-6. Discrepancies > 5% on her own published traces is the condition-3 invalidation signature and gates the rest of Wave 2.
- **Stop-and-ask discipline.** If you encounter unexpected state, missing dependencies, license terms that differ from expectation, or any condition where the right move is unclear, ask rather than guess.
- **No time estimates.** Per the established methodology, don't include "this should take 2 hours" or similar. Estimates produce false precision and aren't load-bearing for the work.
- **Empirical grounding over assumed structure.** Trust what you observe in Nicoletti's actual code over what the architectural plan describes. The plan is informed by Session B investigation but the source of truth is the code.

---

## Notifications

Use `~/bin/notify` for milestones:

- Phase α started — one-line summary of plan
- Deliverable 1 complete (venv + installs verified)
- Deliverable 2 complete (mods compiled)
- Deliverable 3 complete (reference reproduction passes) **OR** Deliverable 3 surfaces discrepancy (use `urgent` priority)
- Deliverable 6 complete (smoke tests pass)
- Phase α complete — one-line result
- Blocked / needing user input — use `urgent` priority

Keep notifications under ~150 chars.

---

## Output format

`phase_alpha_report.md` at completion includes:

1. **Versions and environment** — NEURON version, Brian2 version, Python version, venv location, OS info
2. **Compilation status** — list of 22 mod files, compile result for each, any warnings of note
3. **Reference reproduction results** — per-neuron tested, divergence from published figures, any discrepancies with hypothesis
4. **Harness smoke-test results** — voltage-clamp + current-clamp, pass/fail per test case
5. **Harness API observations** — design decisions that felt arbitrary vs grounded, flags for Phase β refactor (per the prototype-first methodology)
6. **Surfaced findings** — anything mid-flight that's load-bearing for Phase β: NMODL idioms that didn't translate cleanly during smoke-test reference construction, Nicoletti directory structure quirks, missing files, license file contents (if encountered), etc.
7. **Wave 2 readiness assessment** — does Phase α's outcome support proceeding to Phase β? Or are there findings that gate Phase β / require user discussion?

---

## Reference packages on disk (already cloned)

- `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/` — Nicoletti 2024 (22 mod files + 9 simulation scripts)
- `~/Desktop/C-Elegans/simulation/upstream/c302/` — c302 (cell morphologies, network templates, MIT license)
- `~/Desktop/C-Elegans/simulation/upstream/ChannelWorm/` — ChannelWorm (4 worm channels, archived 2018)
- `~/Desktop/C-Elegans/simulation/upstream/BAAIWorm/` — BAAIWorm (reference only; not installable on RTX 4060 Ti)

License terms: c302 MIT (verified). BAAIWorm Apache 2.0 (verified). Nicoletti 2024 and ChannelWorm: ModelDB-convention academic-use (presumed; verify by reading any LICENSE / README files in those directories during Phase α inspection — log findings to report but don't gate Phase α on them).

---

## Final instructions

Execute Phase α end-to-end as a single session. Surface findings mid-flight. Stop-and-ask if anything's unclear. Produce the 6 file deliverables and the completion report. Notify at milestones. The other cross-session adversarial review sessions remain idle until Phase α completes — your output is what they review.

Begin with: read the three reference docs, inspect Nicoletti's directory, and produce a one-paragraph plan summary in your first response before touching the venv. That is the pre-flight pushback step.
