# Wave 2 option α implementation — IRK + UNC-103 translation + AVA cellular validation re-evaluation

**Mode:** engineering work block, not investigation. Single session, file-based pause-and-wait pattern. Session 2 redeployment leveraging full Wave 2 implementation context.

**Strategic positioning:** today's work pivoted Wave 2 from "match Mellem 2008's plateau in AVA" (misattributed target) to "match Nicoletti's actual AVAL phenotype" (option α). This work block executes the engineering operationalization.

---

## Context: redeployment, not fresh start

Today you (Session 2) executed substantially all Wave 2 work: Phase α infrastructure, Phase β-pre v1/v2/v3, Phase β overnight runs #1 and #2 (translating all 7 essential channels), F6 misdiagnosis correction, density sensitivity, Ca-coupling integration, speculative GNN architecture, Mellem 2008 misattribution discovery, citation audit. Full implementation context is yours.

This redeployment executes the engineering work that operationalizes option α: translate IRK + UNC-103, reconstruct AVA cell with Nicoletti's actual 5-channel set, re-evaluate Phase F against corrected target.

---

## Out of scope

- Citation audit cleanup of architectural plan (deferred to paper 3 manuscript prep per user directive)
- AVAR upstream issue review and filing
- Translation of channels for non-AVA cells specifically
- Architectural plan revisions
- Paper 2 manuscript work
- Methodology paper documentation
- Speculative architecture work
- New mechanism translations (CICR, persistent inward currents)

If you finish with capacity remaining, stop and surface for user discussion rather than expanding scope.

---

## Working environment

- Isolated venv: `~/venvs/wave2-neuron/`
- Project code: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Nicoletti 2024: `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- Phase α infrastructure (NEURONReference, voltage_clamp_harness, plateau_harness, translation_patterns.md catalog) already built
- F1-F17 NMODL gotcha catalog at `wave2/translation_patterns.md`
- Existing 7 channel translations at `wave2/channels/`
- Phase F cell construction reference at `wave2/artifacts/gate2_ava_cell_construction.md`

---

## Pre-flight pushback expected

Read this prompt fully. If you find scope concerns, hidden assumptions, missing context, or items warranting cross-session discussion, surface to `wave2/artifacts/option_alpha_pushback.md` and pause via `wave2/artifacts/PAUSED_FOR_REVIEW.txt` marker.

If pre-flight surfaces no concerns, proceed to CP1.

---

## Methodology continuity for engineering execution

- Mid-flight findings to `wave2/artifacts/option_alpha_findings.md` as they emerge (extend F1-F17 catalog if new patterns surface)
- Stop-and-pause vs stop-and-ask: implementation questions with clear best-path proceed (document choice in findings); architectural / load-bearing decisions pause for user review
- Numerical stability checks (NaN/Inf detection on all traces); don't fudge parameters
- F16/F17 lessons: caintra1↔slo1iso unit conversion (1000×) and fca scaling. UNC-103 doesn't read [Ca]_i but unit conventions still need verification. IRK is voltage-gated only.
- F2 lesson: UNC-103 specifically flagged for GLOBAL→per-cell state pattern. Address explicitly.
- Document, don't fabricate

---

## CP1 — UNC-103 channel translation

UNC-103 is voltage-gated K channel. F2 from earlier work flagged UNC-103 as having GLOBAL→per-cell state pattern issue. Address explicitly during translation.

**Workflow:**

1. Read `nicoletti_2024/.../unc103.mod`. Identify gating variables, kinetic parameters, voltage dependence.
2. Identify GLOBAL state declarations needing per-cell conversion. Per F2 lesson, GLOBAL state in NMODL becomes problematic when multiple cells share the channel — needs to be RANGE/per-cell in Brian2.
3. Translate to Brian2 equation string format. Save to `wave2/channels/unc103.py`. Module structure mirrors existing channels.
4. Smoke test.
5. Validation: voltage-clamp harness comparing Brian2 UNC-103 vs NEURON UNC-103. Voltage-feature ≤3 mV residual gate at >80% of holding potentials.

**CP1 acceptance:**
- `wave2/channels/unc103.py` exists
- Voltage-clamp validation passes >80% of holding potentials within tolerance
- F2 GLOBAL→per-cell pattern explicitly addressed in module comments
- Status file output

---

## CP2 — IRK channel translation

IRK is inwardly-rectifying K channel. Voltage-gated, no Ca-dependence. Should follow established pattern.

**Workflow:**

1. Read `nicoletti_2024/.../irk.mod`. Identify gating variables, parameters, voltage dependence. IRK channels typically have inward-rectification-specific gating dynamics.
2. Translate to Brian2. Save to `wave2/channels/irk.py`.
3. Smoke test.
4. Validation per voltage-clamp harness.

**CP2 acceptance:**
- `wave2/channels/irk.py` exists
- Voltage-clamp validation passes >80% within tolerance
- Status file output

---

## CP3 — AVA cell construction with Nicoletti's actual 5-channel set

Construct Brian2 AVAL cell matching Nicoletti's actual AVA cell construction.

**Reference (Nicoletti's actual AVA — 5 channels):**

```python
soma.insert('irk')
soma.insert('leak')
soma.insert('egl19')
soma.insert('nca')
soma.insert('unc103')
```

NO SLO-1, no SHK-1, no SHL-1, no KQT-3, no dynamic Ca-pool.

**Densities:** Use Nicoletti's published AVAL densities. If not immediately accessible from .mod files or `AVAL_simulation.py`, surface for review — apples-to-apples comparison requires matched densities.

**Cell parameters:** Match Nicoletti's AVAL geometry (per Phase F's `gate2_ava_cell_construction.md`).

**Save to:** `wave2/option_alpha_ava_cell.py`.

**Document architectural choices** in `wave2/artifacts/option_alpha_cell_construction.md`:
- 5-channel set rationale
- Density choices (sources)
- What's NOT in this cell (explicit list)
- Geometry and initial conditions
- Comparison to Phase F's 7-channel construction

**CP3 acceptance:**
- `wave2/option_alpha_ava_cell.py` exists
- Cell instantiates without errors
- Smoke test produces sensible voltage trajectory under simple current injection
- Architectural choices documented
- Status file output

---

## CP4 — Phase F re-evaluation against option α targets

Re-run Phase F evaluation with new cell construction and corrected targets.

### Component 2a: Voltage-clamp Layer A

Apples-to-apples: Brian2 5-channel AVA vs NEURON 5-channel AVA.

NEURON reference: extend `NEURONReference` if needed to support custom 5-channel AVA construction. May require either patching her script to vary channel set, building minimal NEURON AVA cell with explicit insertions, or suppressing channels via g_max=0. Choose cleanest approach; document in `wave2/neuron_reference.py`.

Use existing voltage-clamp harness with current-domain feature-based tolerance metric.

**Pass:** >80% of holding potentials clear voltage-feature equivalent ≤3 mV in current domain.

### Component 2b: Current-clamp Layer A against Nicoletti's actual AVAL phenotype

**Protocol:** 1000 ms current injection step (Nicoletti's published protocol per audit, NOT 100 ms which was Mellem-misattribution legacy). Multiple injection levels matching Nicoletti.

**Comparison:** Brian2 5-channel vs NEURON 5-channel via `current_clamp_layer_a_compare`.

**Pass:** Voltage-feature ≤3 mV at peak voltage + plateau amplitude during injection + recovery on injection release. >80% of timepoints. Timing features warn-only.

**Expected phenotype per Nicoletti:**
- Slow-rising phase (~200 ms) on injection onset
- Sustained plateau during injection (no spontaneous termination — plateau persists until stimulus removed)
- Decay back to baseline when injection stops (passive RC-like recovery)
- Linear I-V relationship across injection levels
- No regenerative dynamics, no self-sustaining post-injection plateau

**CP4 acceptance:**
- 2a passes
- 2b passes
- Both produce documented results in `wave2/artifacts/option_alpha_phase_f_evaluation.md`
- Status file output

**CP4 failure modes:**
- 2a passes, 2b fails: voltage-clamp matches but not current-clamp. Investigate density/init/numerical method.
- 2a fails: implementation issue with new IRK or UNC-103, or NEURONReference custom mode bug.
- 2a fails, 2b passes: unusual; investigate as numerical artifact.
- Both fail systematically: surface for review — substantive new finding distinct from original condition 6.

---

## CP5 — Document outcome and produce work block summary

Write `wave2/artifacts/option_alpha_summary.md`:

**Section 1: Outcome verdict** (one of):
- VERDICT_AVA_PRODUCTION_GRADE: both 2a and 2b pass
- VERDICT_PARTIAL: 2a passes, 2b fails
- VERDICT_IMPLEMENTATION_BUG: 2a fails (likely IRK/UNC-103 translation or NEURONReference custom mode)
- VERDICT_DEEPER_FINDING: systematic failure suggesting Brian2 single-compartment can't reproduce Nicoletti's actual AVAL phenotype either

**Section 2: What this means for Wave 2 trajectory**

**Section 3: Findings extending F1-F17 catalog**

**Section 4: Artifacts produced** (file-by-file index)

**Section 5: Recommendations for next work blocks**

**CP5 acceptance:**
- Summary complete
- All findings documented
- Final state: `wave2/artifacts/checkpoints/option_alpha_run_state.json`

---

## Failure modes and recovery

Standard pattern. Pause-with-documentation always preferable to fabricate-completion. Better to have CP1-CP3 cleanly completed and CP4 paused for review than CP4 with hidden uncertainty.

---

## Infrastructure robustness

Same as previous overnight runs. Timeouts, NaN/Inf detection, memory monitoring, state persistence after each subcheckpoint.

---

## Output file structure

```
wave2/
├── channels/
│   ├── unc103.py                                    # CP1
│   └── irk.py                                       # CP2
├── option_alpha_ava_cell.py                         # CP3
├── neuron_reference.py                              # potentially extended
└── artifacts/
    ├── checkpoints/
    │   ├── option_alpha_CP1_status.json
    │   ├── option_alpha_CP2_status.json
    │   ├── option_alpha_CP3_status.json
    │   ├── option_alpha_CP4_status.json
    │   ├── option_alpha_CP5_status.json
    │   └── option_alpha_run_state.json
    ├── option_alpha_cell_construction.md            # CP3
    ├── option_alpha_phase_f_evaluation.md           # CP4
    ├── option_alpha_summary.md                      # CP5 — entry point
    ├── option_alpha_pushback.md                     # only if pre-flight concerns
    ├── option_alpha_findings.md                     # mid-flight findings
    └── PAUSED_FOR_REVIEW.txt                        # only if paused
```

---

## Completion criteria

**Successful** if all 5 CPs PASS with VERDICT_AVA_PRODUCTION_GRADE.
**Partial** if CP1-CP3 complete cleanly, CP4 surfaces partial result, CP5 documents what's done and pending.
**Failed** if CP1 or CP2 fails (channel translation blocks further work) or environment issues prevent execution.

Partial completion with cleanly documented diagnostics > fabricated full completion.

---

## Final framing

This work block completes Wave 2's pivot from Mellem-misattributed target to option α. If it succeeds, Wave 2 cellular layer for AVA becomes production-grade — major Wave 2 milestone.

Apply the discipline today's work has established. The pattern caught ~20+ substantive errors today including citation propagation issues at the architectural plan level — continue the pattern.

You have the deepest implementation context for this work. The redeployment leverages that context.

Standing by for pre-flight pushback or completion notification.
