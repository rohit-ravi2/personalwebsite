# Phase δ WB3 — CP2 through CP6 (post-adjudication continuation)

**Mode:** WB3 implementation arc. Single agent invocation; multi-CP execution. Rohit has authorized all 7 defaults from CP1's options document with two caveats specified below.

**Predecessor:** WB3 CP1 produced `wave2/artifacts/phase_delta_wb3_release_rule_options.md` — primary-source-grounded options doc + 7 decisions with defaults. Pre-flight surfaced WB2 capacitance arithmetic correction (specific vs total capacitance conflation).

---

## Authorized decisions (all 7 defaults accepted)

| # | Decision | Authorized value |
|---|---|---|
| 1 | Release rule choice | **Option B (graded Boltzmann release, Wicks 1996 sigmoidal)** |
| 2 | LIF→Wave2 sub-pattern | **B2 (per-Synapses g_syn(t), τ_syn = 10 ms)** |
| 3 | AIY/RIM sigmoidal params | **(c) cellular-anchored V_half from Wave 2 cell-builder validation, k = 6 mV** + **Caveat 1 below** |
| 4 | W_graded_I calibration | **0.3 pA (Mellem-calibrated starting point)** + **Caveat 2 below** |
| 5 | E_rev / τ_syn | **E_exc = 0 mV, E_inh = -70 mV, τ_syn = 10 ms** |
| 6 | Stability safety net | **(ii) soft cap at ±100 pA + log warning** |
| 7 | Spike emission from Wave 2 | **(a) σ > 0.5 rising-threshold pseudo-spikes** |

---

## Caveat 1 (Decision 3 — AIY/RIM extrapolation)

**F20 catalog entry in CP5 must explicitly note** that AIY/RIM V_half values are anchored extrapolation from Wave 2 cell-builder validation responses to graded current injection, NOT direct synaptic release measurements. The honest framing matters: Wicks 1996 doesn't cover these neurons, parameters are derived from cell-builder validation data rather than primary-source synaptic release literature.

**Add to CP3 validation suite:** sensitivity analysis on V_half ± 5 mV for AIY and RIM specifically. Document how much downstream behavior depends on this parameter choice. If sensitivity is high (network behavior changes substantially across the ±5 mV range), that's a finding worth surfacing — it would mean AIY/RIM coupling is parameter-dependent in ways the cell-builder validation doesn't constrain.

---

## Caveat 2 (Decision 4 — W_graded_I)

Start with 0.3 pA per CP1 default — primary-source-grounded value (Mellem 2008 ±30 pA injection range / ~100 unit-weight saturated coupling = 0.3 pA per unit weight).

**But this parameter genuinely needs CP3 empirical calibration.** If CP3 touch cascade validation shows AVA Δ peri-touch <+5 Hz under cross-coupling (regression from per-edge LIF baseline established in Stage IV), retune empirically with documented rationale.

**The retune is not fabrication if driven by empirical network behavior and documented honestly** — it's the difference between "primary-source-grounded starting point" and "empirically calibrated for network propagation."

**Document the retune trajectory if it happens:**
- Starting value (0.3 pA)
- Test outcome (AVA peri-touch firing rate observed)
- Retune target (e.g., 1.0 pA, 2.0 pA)
- Retune rationale (network-level coupling requires X-fold stronger drive than per-cell injection scale; specifically: per-cell Mellem injection scale doesn't account for cumulative summation across many active inputs OR for the fact that not all weights saturate to σ=1)

Don't pre-commit to 0.3 pA staying static if CP3 evidence demands adjustment.

---

## WB2 capacitance correction propagation

**F20 catalog entry in CP5 must record:**
- Corrected per-cell totals: AVAL 9.66 pF, AVAR 8.43 pF, AIY 1.05 pF, RIM 1.55 pF
- Corrected ratios vs LIF (~100 pF): AVA-class 10-12×, AIY/RIM 65-95×
- Methodology lesson: specific capacitance (μF/cm²) vs total capacitance (pF) conflation as a translation pattern; primary-source re-derivation catches arithmetic propagation that downstream documentation may have inherited

**Also amend `wave2/artifacts/phase_delta_wb2_findings.md`** with a correction note pointing to the F20 entry — preserve the original WB2 findings as historical record of what was observed, but flag the corrected magnitudes inline so future readers don't propagate the original numbers.

---

## CP2 — Implementation of approved release rule

Implement Option B (graded Boltzmann, B2 sub-pattern) in `wave2/integration/wave2_hybrid_brain.py`:

1. Add `cross_coupling="graded_b2"` mode (in addition to existing `"off"`)
2. Cross-group Synapses: Wave 2 NeuronGroups ↔ LIF NeuronGroups per Cook 2019 connectivity
3. Implement both directions:
   - **Wave 2 → LIF:** continuous σ(V_pre) = 1/(1+exp(-(V_pre - V_half)/k)); LIF receives `(summed)` current `I_post += W_graded_I * w * σ`
   - **LIF → Wave 2:** per-Synapses g_syn(t) state with `τ_syn = 10 ms`; `on_pre`: `g_syn += W_g * w`; current to post: `I_post += g_syn * (E_rev - v_post)` where E_rev = 0 mV (excitatory) or -70 mV (inhibitory) per per-edge sign mode
4. Per-NeuronGroup `clock` keyword for dt mismatch (Wave 2 0.025 ms, LIF 0.1 ms)
5. Soft cap safety net: log warnings when |I_total per Wave 2| > 100 pA; do NOT truncate
6. Pseudo-spike emission from Wave 2 cells: σ > 0.5 rising-threshold per `graded_brain.py:269`'s `_poll_sigma` pattern; preserve `firing_rates()` API

**Implementation principles:**
- Reuse existing Brian2 idioms (`graded_brain.py` precedent for `(summed)` cross-group continuous coupling)
- Cython codegen consistently
- Don't tune parameters to produce specific phenotypes; implement biology first

**CP2 acceptance criteria:**
- Wave2HybridBrain supports `cross_coupling="graded_b2"` mode
- Cross-group Synapses connect Wave 2 cells to LIF cells per connectome
- Compiles cleanly under cython
- Soft cap + pseudo-spike emission implemented per spec

---

## CP3 — Numerical stability validation + V_half sensitivity analysis

### CP3.1 — Numerical stability smoke tests

1. **1s smoke** (spontaneous, no stim): all cells in biological V range (-80 to +20 mV); no NaN/Inf; no runaway firing (mean <100 Hz)
2. **10s smoke** (spontaneous): stable cross-group propagation; LIF spikes drive Wave 2 g_syn kicks; Wave 2 σ values modulate LIF currents
3. **30s smoke** (spontaneous + touch stim at t=5s): cascade propagates; system stable through and after stim; soft-cap log warnings counted (if frequent → calibration issue)

### CP3.2 — V_half sensitivity analysis (Caveat 1)

Specifically for AIY and RIM (NOT AVAL/AVAR — those have Mellem 2008 anchor):
- Run touch_anterior 30s scenario at three V_half values for each cell: cellular-anchored default (D), D − 5 mV, D + 5 mV
- Measure: AIY/RIM firing rate change peri-touch vs spontaneous; downstream effect on LIF cells receiving from AIY/RIM
- If sensitivity high (network behavior shifts substantially across ±5 mV range): document as finding worth surfacing in F20 catalog

### CP3.3 — Soft-cap warning analysis

Count log warnings across the 1s/10s/30s tests. Decision tree:
- 0 warnings → can downgrade safety net to (i) no cap in future, but keep (ii) for now
- <10 warnings (rare excursions): keep (ii); investigate which cells/edges caused excursions
- ≥10 warnings: parameter calibration issue; pause + investigate before CP4

**CP3 acceptance criteria:**
- All three smoke tests pass without numerical instability
- V_half sensitivity analysis run for AIY + RIM with ±5 mV variation
- Soft-cap warning count documented
- Voltage ranges remain biological throughout

---

## CP4 — Touch cascade validation + W_graded_I retune (if needed)

### CP4.1 — Cascade validation under cross-coupled brain

Run touch_anterior 30s with `cross_coupling="graded_b2"`:
- ALM/AVM (LIF, sensory): expect 1-2 Hz → 50-80 Hz peri-touch
- AIB (LIF, interneuron): relay if cascade propagates
- AVAL/AVAR (Wave 2, command): plateau response → drive downstream via graded release rule
- AVB (LIF, forward antagonist): suppression expected
- Compare AVA Δ peri-touch to per-edge LIF baseline (+7.5 Hz from Stage IV)

### CP4.2 — W_graded_I retune (Caveat 2 — only if AVA Δ <+5 Hz)

If CP4.1 shows AVA Δ peri-touch <+5 Hz under cross-coupling (regression from per-edge LIF baseline):
- Document starting value (0.3 pA) + observed AVA response
- Retune: try 1.0 pA, then 3.0 pA if needed; document each test
- Final value: smallest W_graded_I that brings AVA Δ peri-touch ≥+5 Hz (matches or exceeds per-edge LIF baseline)
- Retune rationale to document: per-cell Mellem injection scale doesn't account for cumulative summation across many active inputs OR not all weights saturate to σ=1 in physiological regime
- Do NOT exceed 10 pA without surfacing — if 10 pA doesn't fix it, deeper architectural issue may be at play

### CP4.3 — Wave 2 mechanistic resolution check

Verify Wave 2 cellular detail adds value beyond per-edge LIF:
- AVAL vs AVAR distinguishability under cross-coupling (per Stage IV: AVAL rest -40 mV / +80 mV plateau at +10 pA; AVAR rest -24 mV / +40 mV plateau at +10 pA)
- Plateau dynamics realistic
- Compare behavioral state distribution to per-edge LIF baseline; document differences (Wave 2 enrichment vs implementation issue)

**CP4 acceptance criteria:**
- Touch cascade reproduces under cross-coupled brain (AVA Δ peri-touch ≥+5 Hz, possibly post-retune)
- AVAL/AVAR distinguishability visible
- Behavioral state distribution preserved or differences documented
- W_graded_I retune trajectory documented if applied

---

## CP5 — Findings + F20 catalog entry

Write `wave2/artifacts/phase_delta_wb3_findings.md` (extend pre-flight section already present from CP1):

1. Release rule chosen + rationale (Option B with B2 sub-pattern)
2. Implementation approach (cross-group Synapses, `(summed)` continuous coupling, per-Synapses g_syn(t), soft cap, σ pseudo-spikes)
3. CP3 numerical stability outcomes
4. CP3.2 AIY/RIM V_half sensitivity analysis findings
5. CP4 touch cascade validation outcomes (including W_graded_I retune trajectory if applied)
6. F20 capacitance mismatch resolution
7. Methodology catches surfaced during implementation
8. WB4 readiness (multi-cell drop-in)
9. Phase G LIFBrain integration unblocked status (Session 2 dependency)
10. Per-cell-class confidence ratings: AVAL/AVAR primary-source-anchored (Mellem 2008 + Lockery & Goodman 2009); AIY/RIM anchored extrapolation (cell-builder validation, no direct synaptic release primary source); both biologically reasonable; one more rigorously grounded

### Update `wave2/translation_patterns.md` with F20 entry

Pattern: "Cross-group coupling under heterogeneous capacitance scales requires conductance-based synaptic models."

Required content:
- Recognition signature: Wave 2 (single-compartment biological cm) coupled with LIF (default Brian2 cm); naive `v += W_syn * w` produces unphysiological voltage excursions
- Recommended handling: graded Boltzmann release (Wicks 1996 sigmoidal) with `(summed)` continuous coupling
- Cross-channel implications: any future Wave 2 cell with biological cm needs same handling pattern
- Source finding: WB3 (this work block)
- **WB2 capacitance arithmetic correction (load-bearing methodology lesson):**
  - Original WB2 conflated specific capacitance (μF/cm²) with total capacitance (pF); claimed Wave 2 cm ~0.86 pF and ~116× ratio
  - Corrected per-cell totals: AVAL 9.66 pF, AVAR 8.43 pF, AIY 1.05 pF, RIM 1.55 pF
  - Corrected ratios vs LIF (~100 pF): AVA-class 10-12×, AIY/RIM 65-95×
  - Methodology lesson: primary-source re-derivation (here: re-deriving from Brian2 cell-builder code) catches arithmetic propagation that downstream documentation may have inherited
  - Same pattern as Mellem misattribution catch — primary source over downstream paraphrase

### Amend `wave2/artifacts/phase_delta_wb2_findings.md`

Add inline correction note (preserve original findings as historical record):
- Brief note pointing to F20 entry for corrected magnitudes
- "Original WB2 findings (cm ~0.86 pF, ~116× ratio) overstated magnitude due to specific-vs-total capacitance conflation; corrected values in F20 entry of `wave2/translation_patterns.md`. Mismatch direction + design conclusion (naive `v += W_syn*w` structurally unstable) preserved; only magnitude correction."

**CP5 acceptance criteria:**
- Findings document complete with all 10 sections
- F20 catalog entry written with corrected capacitance values + methodology lesson
- WB2 findings doc amended with correction note
- AIY/RIM extrapolation framing explicit per Caveat 1
- Per-cell-class confidence ratings documented honestly

---

## CP6 — Commit + push

Logical commit groupings:

**Group A — CP1 options doc** (already on disk, uncommitted):
```
docs(wave-v-w2): Phase δ WB3 release-event rule options analysis

Primary-source-grounded analysis of 3 candidate release rules
(V-threshold, graded Boltzmann/Wicks 1996, full conductance-based)
against F20 capacitance mismatch + biological grounding +
numerical stability constraints. Pre-flight surfaced WB2
capacitance arithmetic correction (specific μF/cm² vs total pF
conflation). 7 decisions documented with defaults. Recommended:
Option B with B2 sub-pattern (graded Boltzmann + per-Synapses
g_syn(t), τ_syn = 10 ms).
```

**Group B — CP2 implementation** (`wave2_hybrid_brain.py` + dependencies):
```
feat(wave-v-w2): Phase δ WB3 cross-coupling implementation

Wave2HybridBrain supports cross_coupling="graded_b2" mode.
Cross-group Synapses connect Wave 2 NeuronGroups to LIF
NeuronGroups per Cook 2019. Wave 2 → LIF via continuous
σ(V_pre) summed coupling; LIF → Wave 2 via per-Synapses
g_syn(t) with τ_syn=10ms, E_exc=0mV, E_inh=-70mV. Soft cap
±100 pA + log warnings. σ > 0.5 rising-threshold pseudo-spikes
preserve firing_rates() API. Per-NeuronGroup clock keyword
for dt mismatch (Wave 2 0.025ms, LIF 0.1ms).
```

**Group C — CP3+CP4 validation outputs:**
```
test(wave-v-w2): Phase δ WB3 numerical stability + cascade validation

CP3: 1s/10s/30s smoke tests pass; AIY/RIM V_half ± 5 mV
sensitivity analysis [findings]; soft-cap warning count
[count]. CP4: touch cascade reproduces under cross-coupled
brain; AVA Δ peri-touch [value] vs per-edge LIF baseline
+7.5 Hz; W_graded_I [retune trajectory if applied];
AVAL/AVAR distinguishability [confirmed/issue].
```

**Group D — CP5 findings + F20 catalog + WB2 amendment:**
```
docs(wave-v-w2): Phase δ WB3 findings + F20 catalog (capacitance mismatch resolution)

WB3 findings: release rule chosen, implementation, validation
outcomes, per-cell-class confidence ratings (AVAL/AVAR
primary-source-anchored; AIY/RIM anchored extrapolation).
F20 catalog entry: cross-group coupling under heterogeneous
capacitance scales pattern + WB2 arithmetic correction
(corrected per-cell totals AVAL 9.66 pF, AVAR 8.43 pF, AIY
1.05 pF, RIM 1.55 pF; methodology lesson on specific-vs-total
capacitance conflation). WB2 findings doc amended with
correction note pointing to F20.
```

Push to remote.

**CP6 acceptance criteria:**
- All work committed in 4 logical groups
- Commit messages honest + primary-source-disciplined
- Push successful

---

## Failure modes and recovery

- CP2 implementation fails under cython: diagnose, may indicate codegen edge case for `(summed)` coupling pattern; document
- CP3 numerical instability in 1s smoke: diagnose immediately; may need calibration adjustment beyond W_graded_I (parameter-search methodology distinct from "fudging to make it work")
- CP4 cascade regression beyond W_graded_I retune (10 pA doesn't fix): pause; deeper architectural issue; document and surface
- AIY/RIM sensitivity analysis reveals high parameter dependence: not a failure, an honest finding for F20

**General principle:** WB3 lands as the gating milestone. Pause-with-documentation > push-through. Honest documentation of biology vs engineering judgment > clean-looking implementation.

---

## On time scoping

CP2-CP6 in single invocation if possible (~2-3 hours). Multi-invocation continuation acceptable if envelope expires; state persistence after each CP.

Begin with CP2 implementation. Reference `wave2/artifacts/phase_delta_wb3_release_rule_options.md` for full decision context; `wave2/artifacts/phase_delta_wb3_findings.md` for pre-flight findings already documented.
