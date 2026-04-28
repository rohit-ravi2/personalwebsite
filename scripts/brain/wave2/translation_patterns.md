# NMODL → Brian2 translation pattern catalog

**Purpose:** institutional knowledge for Wave 2 channel translation. Reusable
catalog of recurring patterns (gotchas, idioms, calibration strategies) lifted
from Phase β overnight runs. Each pattern: recognition signature, recommended
handling, cross-channel implications, source finding.

**Maintained across overnight runs.** Phase β run #1 produced F1-F10. Phase β
run #2 added F11-F13 (and revised F6's interpretation). Future runs extend.

---

## P1: Distinguish cai-writers from cai-non-writers (from F1)

**Recognition signature:** NMODL declares `USEION ca READ ica WRITE cai` (writer)
vs `USEION ca READ ica` (non-writer). Channels reading `cai` see the WRITTEN
value if a writer is inserted, otherwise the static default.

**Examples in Nicoletti's set:**
- cadiff: writes cai
- caintra1: does NOT write cai (private STATE caintra)
- cca1, unc2, egl19: read ica (don't read cai for gating)
- slo1iso, kcnl: read cai for gating

**Recommended handling:**
- Brian2 translation: encode cai as a NeuronGroup state variable. If a writer
  pool is in the cell, dynamic; if not, constant (use NEURON's default 5e-5 mM).
- Recording: for cadiff cells, record `_ref_cai`. For caintra1 cells, record
  `mech._ref_caintra` (the private state).
- Multi-pool: cadiff and caintra1 CAN coexist without multi-writer error
  (caintra1 doesn't write cai). But functionally tracking different variables.

**Cross-channel implications:**
- SLO-1 isolated reads cai. In AIY (which uses caintra1), cai stays at default
  (5e-5 mM) since caintra1 doesn't write cai. So SLO-1 sees static cai.
- kcnl reads cai. Same situation as SLO-1 isolated.

**Source:** F1 in phase_beta_findings.md.

---

## P2: caintra1 absolute trajectory matches NEURON (from F2)

**Recognition signature:** caintra1's caintra trajectory peaks at 1e-5 - 1e-4 M
range with ica peaks of -1 to -10 mA/cm² (after reaching SS in ~150 ms).

**Recommended handling:** Don't argue with the formula's absolute correctness.
Match Nicoletti's parameters exactly. If translation produces different absolute
values, the bug is in translation, not in the formula.

**Cross-channel implications:** Channels reading caintra (via mechanism's
private state) see this dynamic. Channels reading cai (without a writer pool)
see the static default.

**Source:** F2 in phase_beta_findings.md.

---

## P3: Ca-pool unit-conversion factor — NOT hidden (REVISED, from F6 → F11)

**Recognition signature:** large numerical constant (e.g., 10000 in cadiff.mod's
BREAKPOINT) appearing as a multiplier on dt in a discrete-time update formula.

**Common misdiagnosis (run #1's F6):** "This is a hidden NMODL unit-conversion
factor that we can't symbolically derive." → False. The factor is a deliberate
unit-conversion factor compensating for the formula's terms (mol/(s·cm³)) →
output unit (mM/ms).

**Recommended handling:**
1. Trace through the formula's units explicitly:
   - For cadiff: `(10000) * dt * (-1/(2F)*ica/depth)` evaluates to 0.518 mM/ms
     per (-1 mA/cm² ica). The 10000 IS the proper unit-conversion factor.
2. Verify symbolic derivation against fresh NEURON IV sweep across multiple
   regimes and cell geometries.
3. If symbolic and empirical disagree: the symbolic derivation is the bug; revisit
   unit accounting before resorting to "hidden machinery" hypotheses.

**Cross-channel implications:**
- All channels with explicit `(N) * dt * (...)` patterns in BREAKPOINT need
  symbolic-derivation verification (cadiff is the only one in Nicoletti's set
  with this pattern).
- Channels using DERIVATIVE blocks (most voltage-gated channels) don't have this
  issue — NEURON's solver handles unit consistency.

**Source:** F11 in phase_beta_findings.md (revising F6).

---

## P4: Use mM scale for cai internally to match NEURON (from F3, F4)

**Recognition signature:** Brian2 translation produces cai values 10× or 1000×
off from NEURON. Likely a unit-scale mismatch (mM vs M) at the boundary.

**Recommended handling:**
- Internally store cai in mM (matches NEURON's cai for cadiff, and approximately
  matches caintra1's M-scale numerics if you accept the 1000× scale offset).
- Channels needing μM-scale formulas (e.g., slo1iso has `ca*1e3` to convert to
  μM) apply the conversion at the read site, NOT at the pool boundary.

**Source:** F3, F4 in phase_beta_findings.md.

---

## P5: caintra1's NMODL default vol/surf is AIY-specific

**Recognition signature:** caintra1.mod has `vol = 7.42e-12, surf = 65.89e-8`
as PARAMETER defaults (RANGE so per-section overridable). These match AIY's
actual cell geometry from neuromorpho.

**Recommended handling:**
- For matching Nicoletti's NEURON exactly: use NMODL defaults regardless of
  cell geometry (none of her cell scripts override these — even when caintra1
  is inserted, which is rare).
- For matching the Brian2 cell's actual stub-cylinder geometry: compute
  vol = (π/4)·d³, surf = π·d² per cell. This diverges from Nicoletti's NEURON
  setup but is more "physically correct."
- **Default choice:** match Nicoletti (option 1) for apples-to-apples comparison
  in Gate 2 evaluations. Document the divergence as a known approximation.

**Source:** F12 in phase_beta_findings.md (architectural finding from Phase A).

---

## P6: slo1egl19 has internal calcium(V) — no nanodomain encoding question

**Recognition signature:** SLO-1 channel coupled to a Ca channel (EGL-19 in
Nicoletti's case) — appearance of needing sub-membrane vs bulk Ca compartments.

**Reality (Nicoletti's approach):** slo1egl19.mod has a closed-form `calcium(V)`
FUNCTION computing nanodomain Ca purely from V via Lluís-Buchholz/Alvarez-style
formula:
```
calcium(V) = |gsc·(V-eca)·1e-3| / (8·π·r·d·FARADAY) × exp(-r/√(d/(kb·b))) × 1e6 × 1e-3 + fondo
```
with fixed parameters (gsc=40 pS, r=13 nm, d=250 μm²/s, kb=500e6/M-s, b=30 μM,
eca=60 mV, fondo=0.05 μM, FARADAY=96485).

**Recommended handling:** match Nicoletti exactly. Translate to Brian2 as
algebraic equation in eqs string. No state variable for nanodomain Ca needed.

**Cross-channel implications:**
- slo2egl19 likely follows same pattern (closed-form calcium(V))
- slo1unc2, slo2unc2 likely follow same pattern with UNC-2 instead of EGL-19

**Source:** F13 in phase_beta_findings.md.

---

## P7: NEURONReference instance reuse for multi-hold sweeps

**Recognition signature:** validating a channel against NEURON requires running
multiple holding potentials. Naive approach creates new section per hold (slow,
risk of cross-hold contamination).

**Recommended handling:** reuse one section across all holds. Between holds,
only `h.finitialize` to reset state; don't recreate section.

**Source:** CP1.A.1 architecture in phase_beta_findings.md (run #1).

---

## P8: Voltage-clamp tolerance: current-domain v3-analog metric

**Recognition signature:** comparing Brian2 vs NEURON ica per voltage hold;
some holds have near-zero current, breaking relative-tolerance metrics.

**Recommended handling:** use the divergence metric:
```
divergence(a, b, peak) = |a - b| / max(|a|, |b|, 0.1 * peak)
```
- Per-feature: ≤5% within tolerance
- Per-panel: >80% holds clear

This avoids small-denominator pathologies. The 0.1 * peak floor handles the
low-current regime cleanly.

**Source:** CP1.A.2 in phase_beta_findings.md.

---

## P9: Voltage-clamp protocol matters — NEURON capacitive transient

**Recognition signature:** Brian2 vs NEURON ica trajectories agree at SS but
disagree at step onset. Likely the NEURON capacitive transient (Cm·dV/dt at
the V-clamp step) that Brian2's force-clamp suppresses.

**Recommended handling:**
- Add `skip_initial_transient_ms = 2.0` to feature extraction (skip first 2 ms
  after step onset).
- Add `brian2_prestep_ms = 50.0` to Brian2 simulation (-60 mV pre-step before
  the test step) to settle initial conditions before the active comparison
  window.
- Both adjustments together align Brian2 and NEURON cleanly.

**Source:** F9 in phase_beta_findings.md.

---

## P10: Unit handling check — leak-relative scale (CRITICAL, from F10)

**Recognition signature:** Brian2 channel produces ~1000× too small (or large)
ica/ik. Brian2's "1" (dimensionless) units approach hides this from intrinsic
unit checking.

**Recommended handling:** add a "leak-relative scale check" smoke test for
each new channel. At V = 0 mV, channel's I should be within an order of
magnitude of `gbar × m_ss × h_ss × |0 - erev|`. Catches order-of-magnitude
unit errors fast.

**Common bug:** writing `I = gbar * m * h * (v_mV - erev) * 1e-3` when the
correct expression is `I = gbar * m * h * (v_mV - erev)` (factor of 1e-3
already absorbed in unit consistency between V·S → mA when V is in mV and S
is in S/cm² producing mA/cm²).

**Source:** F10 in phase_beta_findings.md.

---

## P11: VClamp via SEClamp with low rs

**Recognition signature:** for cell context validation (vs isolated channel),
need a clamp that adapts to cell impedance.

**Recommended handling:**
- NEURON: `h.SEClamp(soma(0.5))` with `rs = 0.001` (effectively low-impedance,
  approximating ideal V-clamp without the high-frequency artifacts of point-
  process VClamp).
- Brian2: `network_operation` callback that force-resets V at every dt.
- Document each side's clamp implementation; they differ in high-frequency
  response but match at SS.

**Source:** Phase α + CP1 setup in run #1.

---

## P12: Symbolic derivation FIRST, empirical calibration as confirmation

**Recognition signature:** running NMODL through Brian2 gives wrong numerical
values. Tempting to "calibrate empirically" without understanding why.

**Recommended handling:**
1. Trace through symbolic derivation explicitly with full unit accounting.
2. Predict Brian2 output value.
3. Compare to NEURON at multiple regimes/cells.
4. If symbolic == empirical: translation is principled, no calibration needed.
5. If symbolic ≠ empirical: bug in symbolic derivation OR actual hidden NMODL
   behavior. Investigate before falling back to calibration.

Run #1 fell into a "calibrate first, document later" trap. The empirical
calibration converged to the right answer, but the documented hypothesis ("hidden
NMODL machinery") was wrong. Phase A's deeper diagnostic caught this.

**Lesson:** empirical calibration is a backstop, not a primary tool. Symbolic
derivation should always be attempted first.

**Source:** F11 in phase_beta_findings.md.

---

## P13: cai default is 5e-5 mM — channels reading cai see this absent a pool

**Recognition signature:** channel reads cai (via `USEION ca READ cai`). Cell
doesn't insert any pool that writes cai. What value does the channel see?

**Answer:** NEURON's default `cai0_ca_ion = 5e-5 mM` (= 50 nM, biological
resting [Ca]_i). This is the Hodgkin-Huxley convention default.

**Recommended handling:**
- For Brian2 cells matching Nicoletti's actual setup (no pool inserted):
  set `cai_mM = 5e-5` as constant in the eqs string. Channels reading
  cai_mM see this constant.
- For Brian2 cells with pool inserted (cadiff or caintra1): cai_mM is a state
  variable evolved per the pool's dynamics.

**Source:** F12 in phase_beta_findings.md.

---

## P14: NEURON ion_style override is asymmetry-triggered, not count-triggered (from F18 refinement)

**Recognition signature:** cell with multiple `USEION ca` mechanisms, user
sets `seg.eca = X` (e.g. 60 mV) before `h.run()`, but observed runtime
`seg.eca` ≠ X (e.g. 127.59 mV from Nernst computation).

**Trigger condition (corrected):** the override fires when at least one
channel declares `USEION ca READ eca` WITHOUT `WRITE ica` (a "READ-only"
Ca reader, e.g. slo1egl19, slo2egl19, slo1unc2, slo2unc2, kcnl), AND
coexists with channels that DO write ica (cca1, unc2, egl19). The
asymmetry in USEION declarations promotes NEURON's ion_style to
Nernst-computed eca.

**Non-trigger:** all USEION ca channels have identical READ eca + WRITE ica
declarations (e.g. RIM with cca1+unc2+egl19 all ica-writers). User-set
seg.eca is preserved.

**Recommended handling:**
1. **Always probe** `seg.eca` after a brief `h.run()` when validating a
   new cell — don't predict from channel-count heuristic.
2. If overridden: pass the runtime eca explicitly to all Ca-using channels
   in the Brian2 cell builder (via `eca_mV=` parameter on
   `<chan>_apply_params`).
3. If preserved: use the user-set value (typically 60 mV per Nicoletti's
   convention).

**Cross-cell verification:**
- AVAL: single USEION ca → no multi → eca preserved at 60 mV ✓
- AIY: egl19 (READ+WRITE) + slo1egl19 (READ only) → asymmetric → eca = 127.59 ✓
- RIM: cca1+unc2+egl19 (all READ+WRITE) → symmetric → eca preserved at 60 ✓

**Source:** F18 refinement entry in cellular_validation_findings.md
(2026-04-26, Wave 2 RIM CP4 pre-flight discovery).

---

## P15: NMODL GLOBAL pitfall is automatically handled by Brian2 per-cell semantics (from UNC-2 CP3)

**Recognition signature:** NMODL declares GLOBAL on derived assignments
(minf, hinf, mtau, htau, etc.) or on diagnostic copies of state vars
(munc2, hunc2). This is a NMODL pitfall — these should be RANGE for
per-instance correctness in NEURON multi-cell models.

**NEURON behavior:** functionally harmless in single-cell models because
rates(v) is called at every DERIVATIVE step before m', h' are evaluated;
multi-cell models would have the GLOBAL state corrupted by the last-touched
cell's values. The actual STATE {m, h} is RANGE-by-default per NMODL
convention, so per-instance.

**Brian2 translation handling:** translate the GLOBAL-declared derived
quantities as per-cell `: 1` declarations in the EQS string. Brian2's
per-cell-by-default semantics give correct functional behavior with no
special handling. The NMODL GLOBAL pitfall surprise is NEVER inherited
by Brian2 code.

**Examples in Nicoletti's set:**
- unc2.mod: GLOBAL minf, hinf, mtau, htau, munc2, hunc2

**Recommended handling:** standard per-cell translation. Document the
GLOBAL handling decision in the channel module docstring per CP3-style
acceptance criterion.

**Source:** UNC-2 channel translation, Wave 2 RIM CP3 (2026-04-26).

---

## Summary of pattern usage so far

| Pattern | Used in run #1 | Should be checked in run #2+ |
|---|---|---|
| P1 cai-writers vs non-writers | EGL-19 cell construction (CP3) | All Ca-related cells |
| P2 caintra1 matches NEURON | CP1.B.6 caintra1 validation | CP1.B.6 |
| P3 unit-conversion factor | cadiff (was: misdiagnosed; now: correct) | Any pool with explicit dt-multiplier |
| P4 mM scale for cai | calcium_pool.py | All channels reading cai |
| P5 caintra1 vol/surf default | CP1.B.7 combined Ca-pool | Any cell inserting caintra1 |
| P6 slo1egl19 internal calcium(V) | (not yet used) | Phase E translation |
| P7 NEURONReference reuse | CP1.A.1 + all subsequent | All channel validations |
| P8 current-domain tolerance | CP1.A.2 + all VC validations | All channels |
| P9 VC protocol with prestep | CP2 EGL-19 | All channels |
| P10 leak-relative scale check | CP2 (caught EGL-19 1e-3 bug) | All new channels (smoke test) |
| P11 SEClamp with low rs | All cell-context validations | All Phase F work |
| P12 symbolic-first methodology | (not used in run #1) | All future translations |
| P13 cai default 5e-5 mM | (implicit) | All cells without explicit pool |
| P14 ion_style asymmetry trigger | AIY (override) + RIM (preserved) | All multi-USEION-ca cells |
| P15 GLOBAL → Brian2 per-cell auto | UNC-2 (RIM CP3) | Any channel with GLOBAL derived assignments |
