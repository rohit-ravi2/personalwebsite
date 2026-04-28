# Wave 2 cellular validation: RIM

**Mode:** engineering work block. Single session, file-based pause-and-wait. Continuation of Wave 2 cellular validation following AVAL + AIY production-grade outcomes.

**Strategic positioning:** RIM is the third of Nicoletti's published target cells. Substantially de-risked by yesterday's F18 finding (multi-USEION-ca eca handling), RIM g-vector convention fix already in place, UNC-2 GLOBAL pattern anticipated. Outcome target: 3rd production-grade cell + 3 new channel translations (CCA-1, EGL-2, UNC-2).

---

## Out of scope

- Citation cleanup of architectural plan (deferred to paper 3 manuscript prep)
- AVAR upstream issue review and filing
- RMD work (requires Nicoletti 2019 acquisition; separate work block)
- Phase δ network integration (separate work block)
- Architectural plan revisions, paper work, methodology paper

If you finish RIM with substantial capacity remaining, surface for user discussion before expanding scope.

---

## Working environment

- Venv: `~/venvs/wave2-neuron/`
- Code: `~/Desktop/website/personalwebsite/scripts/brain/wave2/`
- Nicoletti source: `~/Desktop/C-Elegans/simulation/upstream/nicoletti_2024/`
- F1-F18 NMODL gotcha catalog: `wave2/translation_patterns.md` (extend with F19+ if new patterns)
- 10+ existing channel translations: `wave2/channels/`
- AVAL precedent: `wave2/option_alpha_ava_cell.py`
- AIY precedent (most relevant template — multi-channel, multi-USEION-ca): `wave2/option_alpha_aiy_cell.py`
- AIY findings: `wave2/artifacts/cellular_validation_findings.md` (read F18 specifically before CP1)

---

## Pre-flight pushback expected

Read this prompt fully. Verify against yesterday's work outcomes. If scope concerns, hidden assumptions, or items warranting cross-session discussion: write to `wave2/artifacts/rim_validation_pushback.md` + create `PAUSED_FOR_REVIEW.txt`. Otherwise proceed to CP1.

**Primary-source verification requirement (load-bearing):**

Yesterday caught four propagation errors at pre-flight. For RIM, FIRST step before any channel translation: read `nicoletti_2024/.../RIM_simulation_iclamp.py` and document actual:

- Channel set Nicoletti inserts (yesterday's pre-flight surfaced `[shl1, egl2, irk, cca1, unc2, egl19, leak]` — reverify)
- Channel densities (already known to be in S/cm² — no `gScm2` rescale)
- Cell geometry parameters
- Initial conditions (especially eca handling — F18 lesson)
- Current-clamp protocol Nicoletti uses for RIM
- USEION ca mechanism count (verify 3: cca1 + unc2 + egl19)

Primary source wins over prompt claims. Document discrepancies and proceed against verified specification.

---

## Methodology continuity

- Mid-flight findings to `wave2/artifacts/cellular_validation_findings.md` (extend yesterday's file)
- Stop-and-pause vs stop-and-ask: implementation questions with clear best-path proceed; architectural / load-bearing decisions pause
- Numerical stability checks; don't fudge parameters
- F1-F18 lessons applied:
  - F2-class GLOBAL handling for UNC-2 (explicit per-cell state)
  - F16/F17 unit conversion lessons
  - **F18 multi-USEION-ca eca handling (explicit eca_mV, don't rely on NEURON's silent override)**
- Document, don't fabricate
- State persistence after each subcheckpoint

---

## CP1 — CCA-1 channel translation (T-type voltage-gated Ca)

CCA-1 is T-type voltage-gated Ca channel. Standard voltage-gated Ca pattern, similar complexity to EGL-19. Reads [Ca]_ext, writes ica.

**Workflow:**

1. Read `nicoletti_2024/.../cca1.mod`. Identify gating variables, kinetic parameters, voltage dependence, ica formulation.
2. Verify USEION ca conventions. Note for F18 awareness — one of three USEION ca mechanisms in RIM.
3. Translate to Brian2. Save to `wave2/channels/cca1.py`. Module structure mirrors existing channels with **eca_mV parameter** for F18-compatible eca handling.
4. Smoke test.
5. Voltage-clamp validation per established pattern.

**CP1 acceptance:** module exists with eca_mV parameter; voltage-clamp validation >80% holding potentials within tolerance; F18-compatible; status output.

---

## CP2 — EGL-2 channel translation (voltage-gated K, ether-a-go-go family)

EGL-2 is voltage-gated K channel from EAG family. Follows established voltage-gated K pattern.

**Workflow:**

1. Read `nicoletti_2024/.../egl2.mod`. Identify gating, kinetics, voltage dependence, ik formulation.
2. EAG channels often have unusual gating kinetics — document any unusual structure.
3. Translate to Brian2. Save to `wave2/channels/egl2.py`.
4. Smoke test + voltage-clamp validation.

**CP2 acceptance:** module exists; validation >80% within tolerance; status output.

---

## CP3 — UNC-2 channel translation (voltage-gated Ca with GLOBAL declarations)

UNC-2 is voltage-gated Ca with GLOBAL state in NMODL — the F2-class pattern that yesterday's pre-flight confirmed exists in UNC-2 (NOT UNC-103, that was a misattribution).

**GLOBAL declarations require special handling:** NMODL GLOBAL state is shared across all instances of a mechanism in NEURON. In Brian2, every instance has its own state by default. If UNC-2's GLOBAL is genuinely shared semantics (e.g., environmental constant), preserve. If accidentally GLOBAL when should be RANGE/per-instance (NMODL pitfall), use per-cell state in Brian2.

**Workflow:**

1. Read `nicoletti_2024/.../unc2.mod`. Identify GLOBAL declarations explicitly:
   - What variable is GLOBAL
   - Shared-semantics state vs per-cell state mistakenly GLOBAL
   - How NEURON treats it at runtime
2. Identify gating, kinetics, voltage dependence, ica formulation.
3. Verify USEION ca conventions. Second of three USEION ca in RIM.
4. Translate to Brian2 with explicit GLOBAL handling decision documented in module comments. Save to `wave2/channels/unc2.py`.
5. Smoke test + voltage-clamp validation.

**CP3 acceptance:** module exists with explicit GLOBAL handling + decision rationale; validation >80% within tolerance; F18-compatible eca; status output.

---

## CP4 — RIM 7-channel cell construction with F18-aware multi-USEION-ca handling

Construct Brian2 RIM cell matching Nicoletti's actual RIM construction.

**Verified channel set:** `[SHL-1, EGL-2, IRK, CCA-1, UNC-2, EGL-19, LEAK]`. 7 channels, 3 USEION ca (CCA-1 + UNC-2 + EGL-19).

**F18 directly applies.** Multi-USEION-ca → NEURON silent eca override. Cell construction MUST set explicit eca_mV. Verify exact value against Nicoletti's RIM script (likely 127.59 mV, but verify against `seg.eca` initialization in RIM_simulation file).

**Densities:** Nicoletti's published RIM g-vector. Already in S/cm². NO `gScm2` (would double-divide).

**Save:** `wave2/option_alpha_rim_cell.py`

**Document architectural choices** in `wave2/artifacts/rim_cell_construction.md`:
- 7-channel set (verified)
- Density choices (S/cm² convention, no gScm2)
- F18-aware eca handling (explicit, value verified)
- UNC-2 GLOBAL handling decision (per CP3)
- Geometry and IC
- Comparison to AIY's construction

**CP4 acceptance:** cell builder exists; instantiates cleanly; smoke test sensible; F18-aware; documented; status output.

---

## CP5 — RIM voltage-clamp Layer A comparison

Apples-to-apples Brian2 vs NEURON RIM. Use NEURONReference custom mode (extend cleanly per AIY precedent if needed).

Voltage-clamp harness with current-domain feature-based tolerance.

**Pass:** >80% of holding potentials clear voltage-feature ≤3 mV equivalent in current domain.

**CP5 failure modes:**
- Systematic Brian2 outward excess → likely F18 pattern; verify eca handling both sides
- Per-channel divergence localization (AIY pattern that surfaced F18) — investigate identified channel
- NEURONReference extension issue — debug per UNC-103 precedent

---

## CP6 — RIM current-clamp Layer A comparison

1000 ms current injection matching Nicoletti's protocol. Multiple injection levels.

`current_clamp_layer_a_compare` from plateau_harness.

**Pass:** Voltage-feature ≤3 mV residual at peak + plateau + recovery. >80% timepoints. Timing features warn-only.

**CP6 failure modes:**
- AIY-class slow integrator drift on slow gates: document as numerical
- Systematic divergence: F18-class or new pattern; investigate
- Per-injection-level fail: classify per established pattern

---

## CP7 — Outcome verdict and summary

Write `wave2/artifacts/rim_validation_summary.md`:

**Verdict:** VERDICT_RIM_PRODUCTION_GRADE / PARTIAL / IMPLEMENTATION_BUG / DEEPER_FINDING

**Wave 2 cellular layer status:** 3 production-grade cells if positive. Channel coverage updated. Unique-to-RIM: CCA-1, EGL-2, UNC-2.

**Findings extending F1-F18:** F19+ if new patterns. Particularly: did F18 apply cleanly with 3 USEION ca? UNC-2 GLOBAL handling outcome? New unit conversions?

**Implications for next work blocks:**
- Phase δ readiness (3 cells, channel diversity)
- RMD scope reduced (CCA-1 now translated)
- AVAR upstream + standing followups
- Methodology paper case studies catalog

**Recommendations** priority-ordered for next work blocks.

**CP7 acceptance:** summary complete; all findings documented; final state file at `wave2/artifacts/checkpoints/rim_run_state.json`.

---

## Failure modes and recovery

Standard pattern. Pause-with-documentation always preferable to fabricate-completion. RIM compounds yesterday's lessons (multi-USEION-ca + GLOBAL handling + S/cm² g-vector + multiple new channels) — better to have CP1-CP4 cleanly completed and pause at CP5/CP6 with diagnostics than skip ahead.

---

## Infrastructure robustness

Same as previous overnight runs: timeouts, NaN/Inf detection, memory monitoring, state persistence after each subcheckpoint.

---

## Output file structure

```
wave2/
├── channels/
│   ├── cca1.py                                          # CP1
│   ├── egl2.py                                          # CP2
│   └── unc2.py                                          # CP3
├── option_alpha_rim_cell.py                             # CP4
├── neuron_reference.py                                  # potentially extended for RIM
└── artifacts/
    ├── checkpoints/
    │   ├── rim_CP1_status.json - CP7
    │   └── rim_run_state.json
    ├── rim_cell_construction.md                         # CP4
    ├── rim_validation_summary.md                        # CP7 entry point
    ├── cellular_validation_findings.md                  # extend yesterday's
    ├── rim_validation_pushback.md                       # only if pre-flight concerns
    └── PAUSED_FOR_REVIEW.txt                            # only if paused
```

---

## Completion criteria

**Successful:** CP1-CP6 all PASS; CP7 with VERDICT_RIM_PRODUCTION_GRADE.
**Substantively successful:** all translations clean; cell construction clean; one of CP5/CP6 PARTIAL with documented diagnostic.
**Failed:** channel translation issue blocks further work; multi-USEION-ca handling unresolvable per F18; environment issues.

Partial completion with diagnostics > fabricated full completion.

---

## Final framing

This work block produces RIM, completing Wave 2's cellular layer for Nicoletti's three published target cells. Three production-grade cells representing channel diversity makes Phase δ network integration meaningfully more substantive.

De-risking from yesterday is real: F18 directly applicable, g-vector handling fixed, UNC-2 GLOBAL anticipated. Apply yesterday's discipline. Pre-flight pushback. Primary-source verification. Mid-flight surfacing. Stop-and-pause on architectural questions. Document, don't fabricate.

You have full implementation context from yesterday's work. Other sessions not in flight; single-session focused work.

Standing by for pre-flight pushback or completion notification.
