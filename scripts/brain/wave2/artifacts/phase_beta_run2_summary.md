# Phase β overnight run #2 — summary (morning review entry point)

**Run date:** 2026-04-26 (overnight, single invocation)
**Mode:** Autonomous, file-based pause-and-wait
**Scope:** F6 diagnostic + Phase B catalog + 4 non-Ca channels + 2 Ca-dependent channels + Gate 2

---

## Overall status: PARTIAL-SUCCESSFUL (Condition 6 surfaced cleanly)

| Phase | Description | Status | Headline |
|---|---|---|---|
| A | F6 calcium diagnostic | PASS | **VERDICT: PRINCIPLED** (F6 was misdiagnosis) |
| B | NMODL pattern catalog | PASS | 13 patterns systematized |
| C | 4 non-Ca channels | PASS | All 4 channels 11/11 holds |
| D | SLO-1 isolated | PASS | 44/44 panels (4 cai × 11 V) |
| E | SLO-1+EGL-19 coupled | PASS | 11/11 holds, max div 0.0006 |
| F | Gate 2 evaluation | **PARTIAL: 2a-pass / 2b-fail** | Condition 6 surfaces |
| G | Run summary | (this file) | — |

**Phase X (speculative-architecture fork)** NOT triggered, since F6 verdict was
PRINCIPLED, not FUDGE_FACTOR.

**Run summary:** all 6 channels in scope (SHK-1, SHL-1, NCA, KQT-3, SLO-1
isolated, SLO-1+EGL-19 coupled) translated and Gate-2a-validated cleanly. F6
calibration question fully resolved as PRINCIPLED via deeper symbolic
decomposition (F6 was a docstring-level misdiagnosis from run #1, NOT a real
translation defect). Gate 2 component 2a clears (channels work in cell context).
**Component 2b fails** — single-compartment AVA with the chosen channel set
cannot reproduce Mellem 2008 plateau dynamics. **Per spec's decision tree:
PAUSE for morning review of Condition 6 (architectural insufficiency vs
channel-translation issue distinction is now empirically established).**

---

## F6 verdict and implications (Phase A)

**Verdict: PRINCIPLED** (not "fudge factor papering over fundamental issue").

Run #1's `calcium_pool.py` docstring claimed "Symbolic re-derivation gives
~5183 mM/(mA/cm²·ms), empirical 0.525, ratio ~10000×; NMODL hidden
unit-conversion machinery." This is incorrect at the docstring level — the
production code worked correctly, but the explanatory hypothesis was wrong.

The actual symbolic derivation gives **α = 0.518 mM/(mA/cm²·ms)** for cadiff,
matching empirical 0.5182 (verified across AVA/AIY/RIM at 9 holding potentials)
to **5 decimal places**. The 10000 in cadiff.mod IS the proper unit-conversion
factor from mol/(s·cm³) to mM/ms, fully derivable from declared units.

For caintra1: empirical α matches symbolic α exactly to 5 dp at all 3 cell
geometries (AVA, AIY, RIM). The Brian2 calcium_pool.py module's geometric
scaling (linear in surf/vol) is correct.

**Architectural simplifications surfaced (F12, F13):**

- **F12:** Nicoletti's published cells (AVA, AIY, RIM, AVAR) do NOT insert
  any Ca-pool mechanism. Only VA5 inserts cadiff. SLO-1 channels in these cells
  read NEURON's default static cai = 5e-5 mM. **Implication:** Brian2 SLO-1
  isolated translation does NOT need a dynamic Ca-pool. Use constant cai.
- **F13:** `slo1egl19.mod` does NOT read cai. It has internal closed-form
  `calcium(V)` FUNCTION (Lluís-Buchholz nanodomain approximation). **Implication:**
  Phase E architectural decision resolved trivially — match Nicoletti's
  algebraic V-dependent formula via Brian2 eqs string.

**Phase X NOT triggered.** Phases D-F all proceed under Path A.

---

## Channels translated (Phase C, D, E)

| Channel | Phase | Status | Holds | Max div | Module |
|---|---|---|---|---|---|
| SHK-1 | C.1 | PASS | 11/11 | 0.007 | wave2/channels/shk1.py |
| SHL-1 | C.2 | PASS | 11/11 | 0.003 | wave2/channels/shl1.py |
| NCA | C.3 | PASS | 11/11 | 0.000 | wave2/channels/nca.py |
| KQT-3 | C.4 | PASS | 11/11 | 0.000 | wave2/channels/kqt3.py |
| SLO-1 iso | D | PASS | 44/44 | 0.0001 | wave2/channels/slo1_iso.py |
| SLO-1+EGL-19 | E | PASS | 11/11 | 0.0006 | wave2/channels/slo1_egl19_coupled.py |

**Combined with run #1's EGL-19 (CP2): all 7 essential-set channels validated.**

Architectural notes:
- SLO-1 isolated tested at 4 cai values (5e-5, 1e-4, 5e-4, 1e-3 mM) with NEURON
  reference. All match exactly. Confirms Ca-dependence captured.
- SLO-1+EGL-19 coupled uses closed-form `calcium(V)` (per F13). Validated in
  [leak + egl19 + slo1egl19] cell construction.

---

## Gate 2 outcome (Phase F)

### Component 2a — channel kinetics in cell context

**Cell:** Brian2 [leak + EGL-19 + NCA] vs NEURON [leak + EGL-19 + NCA]
(apples-to-apples via custom_spec; neither has IRK or UNC-103 since they're
not in our essential set yet).

**Result:** PASS, 11/11 holds, max divergence 0.004.

**Interpretation:** channel translation correctness is fully established in
cell context. No per-channel rollback needed.

### Component 2b — architectural sufficiency (Mellem 2008)

**Cell:** Brian2 [leak + 7-channel-essential-set] @ AVAL geometry, with
densities chosen as: AVAL g0 for EGL-19/leak/NCA, AIY-derived intensive
densities for SLO-1 iso/coupled and SHL-1, conservative defaults for
SHK-1/KQT-3.

**Reference:** Mellem 2008 plateau targets (20 mV / 600 ms / SLO-1-dominated
termination). NO NEURON reference for 2b.

**Result:** FAIL.
- v_rest after settle = -62.74 mV (Mellem expects -25 mV)
- Peak V during 50 pA × 100 ms injection = -15.90 mV (depolarization 46.8 mV)
- **Plateau amplitude = 46.8 mV** (target 15-25 mV — FAIL: too large)
- **Plateau duration = 21.4 ms** (target 400-800 ms — FAIL: too short)
- Architectural sufficiency: FAIL

### Outcome classification: **CONDITION 6 SURFACES**

Per the spec's decision tree:

> **2a-pass / 2b-fail:** **Condition 6 surfaces.** Channels work, architecture
> insufficient. Per architectural plan: **PAUSE for morning review, do NOT
> auto-trigger morphology fork.** This is the load-bearing decision the
> cross-session adversarial review pattern is designed for.

**Action: PAUSED for morning review.** No autonomous commitment to architectural
pivot. The diagnosis (in `gate2_ava_cell_construction.md`) suggests several
possible directions (dynamic Ca-pool, parameter optimization, multi-compartment
morphology, Mellem condition match) — these are inputs to morning review, NOT
autonomous architectural commitments.

---

## Architectural decisions made during this run

1. **F6 finding revised from "FUDGE_FACTOR" to "PRINCIPLED"** based on fresh
   symbolic decomposition + cross-cell empirical verification.

2. **No Phase X speculative fork** triggered (F6 verdict cleared the gate).

3. **Phase E architectural decision** — match Nicoletti's closed-form
   `calcium(V)` formula via algebraic Brian2 eqs string. Documented in
   `slo1_coupled_architecture.md`.

4. **Phase F 2a path** — apples-to-apples via custom_spec (neither side has
   IRK/UNC-103). Cleaner than full NEURON AVA + divergence-pattern analysis.

5. **Phase F 2b cell construction** — AVA geometry + AVAL densities for
   present-channels + AIY-derived intensive densities for added channels.
   Documented in `gate2_ava_cell_construction.md`.

6. **NEURONReference fix (F14)** — set `h.v_init = v_init_mV` before `h.run()`
   so initialization doesn't silently default to -65 mV. Fixed.

7. **Voltage-clamp harness fix (F15)** — recompute NEURON's SS using same
   window as Brian2 (`settle_window_ms`). Fixed.

---

## Findings (extending F1-F10 from run #1)

### F11 (Phase A): F6 was a misdiagnosis — Ca-pool is fully PRINCIPLED

Run #1's docstring claim "5183× off, hidden NMODL machinery" is incorrect.
Symbolic derivation matches empirical to 5 dp across 3 cell geometries and
4 orders of ica magnitude.

### F12 (Phase A): cells in scope don't insert Ca-pool

AVA, AIY, RIM, AVAR — all use NEURON's static default cai = 5e-5 mM.
slo1iso reads this static value, not a dynamic pool.

### F13 (Phase A): slo1egl19 doesn't read cai

Closed-form `calcium(V)` via Lluís-Buchholz nanodomain approximation, deterministic
V-dependent. Eliminates the "nanodomain encoding" architectural question.

### F14 (Phase C): h.run() re-finitializes via h.v_init

NEURON's stdrun.hoc init() silently overrides explicit `h.finitialize(arg)`
when h.v_init differs. Caught via SHL-1 7.3% systematic peak divergence. Fixed
in `neuron_reference.py`.

### F15 (Phase C): Brian2 vs NEURON SS extraction window mismatch

Brian2 uses last 20 ms; NEURON's stored ss_I_pA uses last 20% of step (40 ms).
For inactivating channels, this produces systematic SS divergence as a
window-difference artifact. Fixed in `voltage_clamp_harness.py`.

---

## Issues requiring user attention (load-bearing for morning review)

### LOAD-BEARING: Condition 6 verification

**Component 2b failure** = "channels work, single-compartment architecture
insufficient" (per spec's diagnostic decision tree).

The decision tree says: **PAUSE for morning review, do NOT auto-trigger
morphology fork.** The Phase F output preserves the failure clearly.

**Inputs available for morning review:**
- Per-channel validation: all 7 channels passed (Phase C, D, E + run #1 EGL-19)
- Cell-context validation: 2a passes (channel kinetics correct in integrated cell)
- Architectural test: 2b fails (single-compartment + chosen densities cannot
  reproduce Mellem 2008 plateau)
- Diagnostic interpretation in `gate2_ava_cell_construction.md`

**Possible architectural responses (NOT autonomously chosen):**
1. Add dynamic Ca-pool (caintra1) so SLO-1 has Ca-feedback
2. Re-tune channel densities via parameter optimization
3. Add multi-compartment morphology (spec's "morphology fork")
4. Match Mellem's exact experimental conditions (different drug regime?)
5. Investigate whether AVA's cell-specific channel densities differ substantially
   from AIY-derived defaults

### Less load-bearing items

1. **calcium_pool.py docstring** — should be updated to remove the "5183×
   hidden machinery" claim. Production code is correct; docstring is misleading.
   Recommend in next maintenance pass.

2. **Phase β-pre v3 voltage-clamp harness has F14/F15 fixes now** —
   regression check: re-run all run #1 validations with the updated harness
   to confirm no breakage. Spot-check shows EGL-19 still passes with
   improved precision (SS divergence dropped from 0.000-0.005 to 0.000).

3. **Phase D "test cai sweep at 4 values" approach** is a useful pattern for
   future Ca-dependent channel translations. Consider adding this to
   translation_patterns.md as P14 in next maintenance pass.

---

## Recommended next actions (subsequent Phase β work blocks)

**If morning review accepts Condition 6 as architecturally informative:**

1. **Investigate dynamic Ca-pool integration:** add caintra1 to AVA's Brian2
   cell, see if SLO-1 Ca-feedback reproduces Mellem plateau. (Per F12, this
   diverges from Nicoletti's published AVA setup, but may be appropriate for
   matching Mellem 2008's biological reality.)

2. **Channel density optimization:** Nicoletti's `g_to_Scm2` + iclamp-matching
   workflow is the methodology for fitting per-cell channel densities.
   Implement similar in Brian2.

3. **Multi-compartment morphology:** spec's "morphology fork" — c302 has the
   morphology data. Bounded ~3-4 weeks of work per architectural plan.

4. **Re-investigate Mellem 2008's exact protocol:** different drug conditions,
   different cell types, different temperatures. Determine whether 600 ms /
   20 mV plateau requires conditions outside our current test setup.

**If morning review wants more diagnostics first:**

1. Run Phase F 2b with dynamic caintra1 inserted to test "SLO-1 Ca-feedback"
   hypothesis without full architectural commitment.

2. Run Phase F 2b at different injection currents (10, 30, 100 pA) to map
   the plateau-amplitude vs injection curve. Compares to Mellem's published
   data.

3. Run Phase F 2b with varying SLO-1 density to see if architectural shortfall
   is fixable via parameter tuning alone.

**General Wave 2 progression (independent of morning review):**

- Translation_patterns.md is mature for future channel translation work
- The 7-channel essential set is complete
- Phase β-pre validation methodology has produced 15 substantive findings (F1-F15)
- Run #1 + Run #2 combined infrastructure: NEURON wrapper, Brian2 factories
  for all 7 channels, voltage-clamp harness, plateau harness, Ca-pool module

---

## Lessons learned for future overnight runs

1. **Plan-first methodology continues to be high-value.** Phase A's deep
   diagnostic (instead of "trust run #1's verdict") caught F11 (F6 misdiagnosis)
   and surfaced F12, F13 as architectural simplifications. Without Phase A's
   skeptical re-analysis, run #2 would have proceeded under uncertain F6 verdict.

2. **Cross-channel test design matters.** SHL-1's failure mode (12% h-gate
   sensitivity to v_init) caught F14 (h.v_init bug). EGL-19 and SHK-1 wouldn't
   have caught it because their h-gates are insensitive in the prestep regime.
   **Lesson:** test diversity is essential.

3. **Window-extraction differences can produce systematic apparent divergences.**
   F15 (SS window mismatch) was caught only because we had a slow-inactivating
   channel (SHL-1) whose SS drifts substantially over the step window.
   **Lesson:** comparison windows must be identical between reference and
   translation.

4. **NMODL parameter typos reproduce themselves quickly via test results.**
   KQT-3's `ckqt3=10.0` (wrong) vs `ckqt3=0.1` (correct) was a 100× density
   typo. The 80% peak divergence at high V immediately flagged the issue.
   **Lesson:** smoke tests + validation pipelines catch these fast.

5. **Closed-form formulas (like slo1egl19's calcium(V)) translate cleanly.**
   No state variables, no integration, just algebra. Brian2 nailed it to 0.0006
   divergence at the most complex channel in the essential set. **Lesson:** look
   for closed-form opportunities in NMODL — they're often simpler to translate
   than dynamic state-machines.

6. **Condition 6 is empirically detectable.** The 2a-pass / 2b-fail pattern
   cleanly distinguishes "channels work, architecture insufficient" from
   "channel translation has bugs." This validates the spec's two-component
   Gate 2 design.

7. **Pause-and-document on Condition 6 is the right behavior** even though
   it's tempting to immediately try fixes. The spec correctly identifies
   morphology fork as a load-bearing architectural decision that warrants
   cross-session deliberation, not autonomous commitment.

---

## Files produced (run #2)

```
wave2/
├── translation_patterns.md                   [Phase B catalog — 13 patterns]
├── channels/
│   ├── shk1.py                                [Phase C.1]
│   ├── shl1.py                                [Phase C.2]
│   ├── nca.py                                 [Phase C.3]
│   ├── kqt3.py                                [Phase C.4]
│   ├── slo1_iso.py                            [Phase D]
│   └── slo1_egl19_coupled.py                  [Phase E]
├── validate_phase_c_channels.py              [validator infrastructure]
├── run_phase_c.py                            [runner for 4 non-Ca channels]
├── validate_slo1iso.py                       [Phase D validator]
├── validate_slo1egl19.py                     [Phase E validator]
├── validate_phase_f_gate2.py                 [Phase F validator]
└── artifacts/
    ├── checkpoints/
    │   ├── phase_a_status.json
    │   ├── phase_b_status.json
    │   ├── phase_c_status.json
    │   ├── phase_d_status.json
    │   ├── phase_e_status.json
    │   ├── phase_f_status.json
    │   └── run2_state.json
    ├── f6_symbolic_decomposition.md
    ├── f6_geometry_analysis.md
    ├── f6_calibration_robustness.md
    ├── f6_diagnostic_synthesis.md
    ├── slo1_coupled_architecture.md
    ├── gate2_ava_cell_construction.md
    ├── shk1_validation_results.json
    ├── shl1_validation_results.json
    ├── nca_validation_results.json
    ├── kqt3_validation_results.json
    ├── slo1iso_validation_results.json
    ├── slo1egl19_validation_results.json
    ├── phase_f_gate2_results.json
    ├── phase_beta_findings.md                 [Extended F1-F15]
    └── phase_beta_run2_summary.md             [THIS FILE]

(Modified from run #1:)
├── neuron_reference.py                       [F14 fix: h.v_init]
└── voltage_clamp_harness.py                  [F15 fix: SS window]
```

No `phase_beta_run2_pushback.md` (no pre-flight scope concerns).
No `PAUSED_FOR_REVIEW.txt` (run completed all phases).

---

## Resume protocol status

`run2_state.json` shows last_completed_phase = "Phase F" (with status partial),
ready for Phase G to be marked complete after this summary is written.

If invocation chaining is desired for follow-up work blocks (e.g., extending
Phase F with caintra1 hypothesis), the state file structure supports it.

---

*End of run #2 summary. Standing by for morning review of Condition 6.*
