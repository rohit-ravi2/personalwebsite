# Wave 2 cellular validation: AIY + RIM + RMD (with scope discipline for RMD)

**Mode:** engineering work block. Single session, file-based pause-and-wait. Session 2 redeployment leveraging full Wave 2 implementation context.

**Strategic positioning:** AVAL production-grade established under option α-1 (5 decimal place agreement Brian2 vs NEURON across 7 current levels). This work extends to AIY + RIM + RMD to validate channel translations in diverse cellular contexts.

---

## Out of scope

- Citation audit cleanup of architectural plan (deferred to paper 3 manuscript prep)
- AVAR upstream issue review and filing
- Phase δ network integration (separate work block once cellular validation complete)
- Architectural plan revisions
- Paper 2 manuscript work
- Methodology paper documentation
- RMD-specific channel translations beyond what's tractable in available time

If you finish with capacity remaining beyond what's specified, stop and surface for user discussion.

---

## Working environment

- Venv: `~/venvs/wave2-neuron/`
- Code: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Nicoletti source: `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- F1-F17 NMODL gotcha catalog: `wave2/translation_patterns.md` (extend with new findings)
- 9 existing channel translations: `wave2/channels/`
- Option α AVAL precedent: `wave2/option_alpha_ava_cell.py` (template pattern)
- Validation infrastructure: voltage_clamp_harness, plateau_harness with Layer A compare, NEURONReference wrapper with custom mode

---

## Pre-flight pushback expected

Read this prompt fully. **Particular pre-flight verification points** (today caught several propagation errors in similar prompts):

- Verify Nicoletti's actual AIY channel set by reading her AIY simulation script directly. Don't trust prompt's channel claims without primary-source check.
- Same for RIM. Read her RIM simulation script directly.
- Verify Nicoletti has an RMD reference implementation. Mellem investigation noted Nicoletti 2019 had RMD; verify this exists locally. If RMD reference doesn't exist in our local resources, RMD validation has different shape and pre-flight needs to flag.

If pre-flight surfaces concerns: write to `wave2/artifacts/cellular_validation_pushback.md` + create `PAUSED_FOR_REVIEW.txt`. Otherwise proceed to CP1.

---

## Methodology continuity

- Mid-flight findings to `wave2/artifacts/cellular_validation_findings.md` (extend F1-F17 catalog if new patterns)
- Stop-and-pause vs stop-and-ask: implementation questions with clear best-path proceed; architectural / load-bearing decisions pause
- Numerical stability checks; don't fudge parameters
- Document, don't fabricate
- State persistence after each subcheckpoint

**Critical: primary-source channel set verification first.** For each cell: FIRST step is reading Nicoletti's actual cell simulation script and documenting actual channel set inserted. If prompt's claims differ from primary source, primary source wins.

---

## Checkpoint structure

Three cells, each with 5 sub-checkpoints. RMD has additional scope-discipline gate (CP3.0) before any RMD work.

State checkpointed incrementally. Each cell independent — proceed to next regardless of previous cell's outcome (unless infrastructure failure).

---

## Cell 1 (AIY) — CP1.1 through CP1.5

### CP1.1: AIY channel set verification + cell construction

Read `nicoletti_2024/.../AIY_simulation_iclamp.py` (or equivalent). Document Nicoletti's actual channel set. Likely candidates: SHL-1, SLO-1+EGL-19 coupled, KCNL/SK, leak — but verify primary source.

Document:
- Channel set (verified)
- Channel densities (Nicoletti's published g-vector)
- Cell geometry (surf, cm, etc.)
- Initial conditions
- Current-clamp protocol Nicoletti uses

If any channel not in existing translations, surface as scope question.

Construct Brian2 AIY at `wave2/option_alpha_aiy_cell.py`. Smoke test.

Document architectural choices in `wave2/artifacts/aiy_cell_construction.md`.

**CP1.1 acceptance:** primary-source channel set verified; all channels available (or new translation deployed); Brian2 cell instantiates cleanly; smoke test passes; documented.

### CP1.2: AIY voltage-clamp Layer A

Apples-to-apples Brian2 AIY vs NEURON AIY. Use NEURONReference wrapper (extend if needed) or direct upstream invocation (cleanest, per AVAL precedent).

Voltage-clamp harness with current-domain feature-based tolerance.

**Pass:** >80% of holding potentials clear voltage-feature ≤3 mV equivalent in current domain.

### CP1.3: AIY current-clamp Layer A

1000 ms current injection (or whatever AIY-specific protocol Nicoletti uses). Multiple injection levels matching her published recordings.

**Pass:** Voltage-feature ≤3 mV residual at peak voltage + plateau amplitude during injection + recovery on release. >80% of timepoints.

### CP1.4: AIY findings documentation

Document any new findings extending F1-F17 catalog. Patterns observed, NMODL gotchas, harness behavior.

### CP1.5: AIY outcome verdict

VERDICT_AIY_PRODUCTION_GRADE / PARTIAL / IMPLEMENTATION_BUG / DEEPER_FINDING.

Status file output. Proceed to Cell 2 regardless of AIY outcome.

---

## Cell 2 (RIM) — CP2.1 through CP2.5

Same structure as Cell 1, RIM-specific.

### CP2.1: RIM channel set verification + cell construction

Read `nicoletti_2024/.../RIM_simulation_iclamp.py`. Document actual channel set, densities, geometry, IC, protocol.

Construct Brian2 RIM at `wave2/option_alpha_rim_cell.py`. Smoke test.

Architectural choices in `wave2/artifacts/rim_cell_construction.md`.

### CP2.2 - CP2.5

Same pattern as Cell 1, applied to RIM.

VERDICT_RIM_PRODUCTION_GRADE / PARTIAL / IMPLEMENTATION_BUG / DEEPER_FINDING.

Status file output. Proceed to Cell 3 per scope-discipline rules below.

---

## Cell 3 (RMD) — scope-disciplined sub-checkpoints

RMD has different shape than AIY/RIM:

1. **Nicoletti reference availability:** Mellem investigation noted Nicoletti 2019 has RMD model. Verify locally. If no local RMD reference, requires acquisition — pause with scope question.
2. **RMD-specific channels:** UNC-2, EGL-19, CCA-1, NCA-1/NCA-2 candidates per Mellem 2008 pharmacology. Most overlap existing translations. CCA-1 (T-type Ca) most likely missing. Evaluate translation feasibility within work block envelope.
3. **Mellem 2008 figure digitization** (optional, default out of scope). RMD Layer A (Brian2 vs Nicoletti's NEURON RMD) is primary; Mellem-target comparison secondary.

### CP3.0: RMD scope evaluation gate

Before RMD work, evaluate scope. Document in `wave2/artifacts/rmd_scope_evaluation.md`:

- Does Nicoletti have RMD reference implementation locally? Where.
- RMD's verified channel set per primary source?
- Are all RMD channels available in existing translations?
- If channels missing: which, complexity, decision on attempt-vs-defer

**Three scope outcomes:**

**Scope A** (RMD tractable): all needed components available. Proceed to CP3.1.

**Scope B** (minor work): one missing channel translation following established pattern (e.g., CCA-1 like EGL-19). Translate + execute RMD validation if total fits within envelope.

**Scope C** (beyond envelope): multiple missing translations or substantial reference acquisition. PAUSE here. Document scope state. Skip to CP4.

If Scope C, document for future RMD work block:
- Channels needing translation (with complexity)
- Reference data needed
- Estimated work block scope

### CP3.1 - CP3.5 (only if Scope A or B)

Same structure as Cell 1/2.

VERDICT_RMD_PRODUCTION_GRADE / PARTIAL / IMPLEMENTATION_BUG / DEEPER_FINDING / DEFERRED (Scope C).

---

## CP4 — Overall summary

Write `wave2/artifacts/cellular_validation_summary.md`:

**Section 1: Per-cell verdicts table** (AVAL carried from earlier; AIY, RIM, RMD this work block)
**Section 2: Wave 2 cellular layer status overall**
**Section 3: Findings extending F1-F17 catalog**
**Section 4: Implications for Phase δ network integration**
**Section 5: Standing followups** (AVAR upstream, RMD if deferred, citation cleanup, methodology paper)
**Section 6: Recommendations for next work blocks**

---

## Failure modes and recovery

Standard pattern. Cell-translation/construction failures pause that cell; other cells proceed independently. Infrastructure failures (NEURONReference custom mode bug, harness numerical issues) — fix and document like today's UNC-103 bug fix in option α-1.

Pause-with-documentation always preferable to fabricate-completion.

---

## Infrastructure robustness

Same as previous overnight runs: timeouts, NaN/Inf detection, memory monitoring, state persistence after each subcheckpoint.

---

## Output file structure

```
wave2/
├── option_alpha_aiy_cell.py
├── option_alpha_rim_cell.py
├── option_alpha_rmd_cell.py                            # only if Scope A or B
├── channels/
│   └── cca1.py                                          # only if Scope B requires it
├── neuron_reference.py                                  # potentially extended
└── artifacts/
    ├── checkpoints/
    │   ├── cellular_val_aiy_CP1_status.json - CP5
    │   ├── cellular_val_rim_CP2_status.json - CP5
    │   └── cellular_val_rmd_CP3_status.json - CP5+ (or scope_evaluation only)
    ├── aiy_cell_construction.md
    ├── rim_cell_construction.md
    ├── rmd_cell_construction.md                         # only if Scope A or B
    ├── rmd_scope_evaluation.md                          # always
    ├── cellular_validation_summary.md                   # CP4 final entry point
    ├── cellular_validation_findings.md                  # mid-flight findings
    ├── cellular_validation_pushback.md                  # only if pre-flight concerns
    └── PAUSED_FOR_REVIEW.txt                            # only if paused
```

---

## Completion criteria

**Fully successful:** AIY + RIM + RMD all PRODUCTION_GRADE.
**Substantively successful:** AIY + RIM PRODUCTION_GRADE; RMD DEFERRED with documented scope state. (Most likely realistic outcome — acceptable.)
**Partial-successful:** Some cells PARTIAL/BUG/DEEPER with documented diagnostics; others completed cleanly.
**Failed:** AIY translation/construction fails fundamentally OR environment issues prevent execution.

Partial completion with cleanly documented diagnostics > fabricated full completion.

---

## Final framing

This work block extends Wave 2's cellular layer from one production-grade cell (AVAL) to multiple cells, validating channel translations in diverse cellular contexts. AIY and RIM use channel mixes AVAL doesn't, so successful validation establishes broader Path A vindication. RMD attempts to address the Mellem 2008 question — if Nicoletti's RMD model exists and reproduces in Brian2, directly comparable to canonical C. elegans plateau biology literature.

Scope discipline pattern applied to RMD reflects today's lesson: prefer pausing cleanly with documented state over expanding scope autonomously.

Apply the discipline today's work has established. Pre-flight pushback. Primary-source verification (channel set claims especially). Mid-flight surfacing. Stop-and-pause on scope expansion. Document, don't fabricate.

Standing by for pre-flight pushback or completion notification.
