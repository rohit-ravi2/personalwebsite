# Phase β findings — running log

**Run start:** 2026-04-26
**Mode:** Overnight autonomous, file-based pause-and-wait
**Scope:** Option A — CP1 (foundation infrastructure) + CP2 (EGL-19 isolated) + CP3 (EGL-19 Gate 2a in cell context)

---

## Pre-flight acknowledgment

Spec read fully. Reference docs (architectural plan, phase α report, phase β-pre validation, harness fitness report, comparison_validation_results_v2.json, voltage_clamp_harness.py, plateau_harness.py, avar_unc103_patch.py) all reviewed. No load-bearing scope concerns surface — proceeding with CP1.A.1.

Key settled commitments locked in:

1. **eqs-string encoding** for Ca-pool subsystems (cadiff + caintra1 living in the same NeuronGroup eqs as the channels that need [Ca]_i). Documented in `wave2/calcium_pool.py` module docstring.

2. **Voltage-clamp tolerance metric:** current-domain analog of v3's voltage-feature gate.
   - Per-feature: `divergence = |a-b| / max(|a|, |b|, 0.1 * peak_current)`
   - Pass threshold: ≤5% relative above 10% of peak threshold; ≤5% of peak absolute below threshold
   - Per-panel: >80% of holding potentials clear

3. **Current-clamp Layer A tolerance:** voltage-feature gate
   - ≤3 mV residual at peak voltage + plateau amplitude per timepoint
   - >80% of timepoints clear per panel
   - Timing features (time-to-peak, settling) reported as warn-only diagnostics

4. **Cell construction strategy for CP3:** Build a custom NEURON cell (leak + cadiff + caintra1 + EGL-19 only) by re-using the AVAL geometry (surf=1123.84e-8 cm², matching capacitance/leak parameters) but with subset channel set. Equivalent Brian2 cell uses same passive parameters.

5. **AVAR UNC103 patch:** Existing at `wave2/avar_unc103_patch.py` — verified. Will reuse `_activate_nicoletti_env` / `_restore_env` pattern to manage cwd + sys.path for NEURON imports.

---

## Decision log

### CP1.A.1: NEURONReference wrapper design

- **Decision:** Single class wrapping NEURON section + mechanism handles, with per-call protocol methods (`voltage_clamp`, `current_clamp`). State management: explicit `cleanup()` method to delete the section between protocols if needed. Avoid creating a new section per holding potential — instead reuse one section across all holds in a sweep, only `h.finitialize` between holds.
- **Reasoning:** Matches the harness fitness report's recommendation (refactor flag #3 in voltage_clamp_harness.py). Minimizes section-creation overhead for multi-hold sweeps.
- **Cell support:** AVAL, AIY, RIM (existing in Nicoletti repo) via cell-name dispatch to upstream wrapper functions. AVAR via the existing `avar_unc103_patch.py`. For CP3, an additional minimal-cell mode that takes an explicit channel list (for the leak+cadiff+caintra1+EGL19 subset).

### CP1.B: Ca-pool encoding

- **Decision:** eqs-string encoding (option a from harness fitness report).
- **Reasoning:**
  - Matches Nicoletti's single-compartment model structure
  - Simpler validation pathway
  - Defers separate-subsystem encoding to morphology integration (Wave 3 territory if condition 6 surfaces)
  - From cadiff.mod: `ca = ca + 10000 * dt * (-1/(2*F)*ica/depth - 0.0001*beta*ca)` with floor at 1e-4 mM
  - From caintra1.mod: `dca/dt = fca*(-((1/(2*vol*Fc))*(ica*surf*1e-3))) - ((ca-ca_eq)/tca)` (only when ica < 0; else just decay term)

### CP1.A.2: Voltage-clamp tolerance metric

- The current `voltage_clamp_compare` uses single 5% relative tolerance with 1e-9 floor. Pathology: when channels are inactive at some holds and current is near zero, relative tolerance becomes useless (small denominator).
- **New approach:** Add `current_domain_tolerance` function and use it inside `voltage_clamp_compare`. Keep old behavior reachable via a `legacy_tolerance` flag for backward compat with existing smoke tests if needed.

---

## Findings during execution

### F1: cadiff and caintra1 do NOT write NEURON's `cai` ion variable

**Discovery:** During CP1.B validation harness construction, recording `_ref_cai`
on a section with caintra1 inserted shows cai stays pinned at 5e-5 (the default
NEURON ion value, equal to `ca_eq` in mM). caintra1 stores its dynamic Ca state
in a PRIVATE STATE variable `caintra` (and writes it to GLOBAL `calcium`), NOT
to the standard ion `cai`.

Verified by NMODL source:
- `caintra1.mod`: `USEION ca READ ica, eca` — no `WRITE cai`. STATE { caintra }.
- `cadiff.mod`: `USEION ca READ ica, cai WRITE cai` — DOES write cai.

So the two pools have very different ion-machinery contracts:
- cadiff: writes to `cai`, channels reading `cai` see updates
- caintra1: doesn't write `cai`; channels read `cai` (= ca_eq pinned default)
  unless they directly access `caintra` via the mechanism, OR read GLOBAL
  `calcium`.

**Implication for translation validation:** We must compare Brian2's `cai_M`
trajectory against:
- For cadiff: NEURON's `_ref_cai` (the standard ion variable)
- For caintra1: NEURON's `_ref_caintra` via the mechanism (the private state)

The initial validation v1 used `_ref_cai` for both and found caintra1 "broken"
(cai never moved). Revised validation v2 uses the correct recording variable
per pool.

### F2: caintra1's Ca trajectory has questionable absolute values

Upstream caintra1 with ca_eq = 5e-5 mM (50 nM), ica peak = 2.77 mA/cm², over
200 ms produces caintra peak = 1.6e-5 mM = 16 nM, with caintra final =
4.87e-7 mM = 0.49 nM. That's BELOW ca_eq, suggesting the formula's
`fca * (-((1/(2*vol*Fc))*(ica*surf*1e-3)))` term is sometimes adding NEGATIVE
[Ca] when ica is positive (the conditional handles this — only inward
ica<=0 contributes). But the decay-only branch `-((caintra-ca_eq)/tca)`
should drive caintra back to ca_eq, not below it. If caintra is decaying
toward ca_eq from inflated values then dropping below, that suggests the
inward branch IS contributing during the simulation.

Actually re-reading the NMODL:
```
if (ica<=0):
   rs = fca * (-((1/(2*vol*Fc))*(ica*surf*1e-3))) - ((caintra-ca_eq)/tca)
```
With ica negative (inward), `-ica` is positive, so the first term contributes
positive ΔCa per ms. With ica = -2.77 mA/cm², surf = 65.89e-8 cm²,
vol = 7.42e-12 cm³, Fc = 96485 coul/mol:
  term1 = fca * (-1) * (1/(2*vol*Fc)) * ica * surf * 1e-3
        = 0.001 * 1 / (2 * 7.42e-12 * 96485) * 2.77 * 65.89e-8 * 1e-3
        = 0.001 / 1.432e-6 * 1.825e-9
        = 698.3 * 1.825e-9
        = 1.275e-6 mol/(cm³·ms?)
This is suspicious — the unit is mol/cm³ if treated as per-second. NEURON
then advances caintra by this rate × dt(ms) treating it as per-ms, giving
a step of 1.275e-6 per ms = 0.255e-3 over 200 ms = 2.55e-4 (very large).

The fact that NEURON empirically produces caintra peaks of ~1e-5 mM
suggests there's substantial smoothing/numerical handling we aren't seeing
on the surface. **For validation purposes: we match whatever NEURON
produces via direct cai_M-vs-caintra comparison. We don't argue with the
formula's absolute correctness — Nicoletti's parameter fits absorb this.**

### F3: cadiff produces dramatic Ca transients (1e-1 mM range)

Upstream cadiff under cca1 vclamp produces cai peaks of ~6e-1 mM = 600 μM.
This is unphysiological (real intracellular Ca peaks at 1-10 μM during
spikes), but it's the upstream behavior — channels reading from cadiff
are calibrated to these inflated values. Any cadiff-using channel
parameterization assumes this scale.

Brian2 cadiff translation produces Ca peaks 10× LARGER than NEURON's
(brian2_final ~3.6, nrn_final ~3.58e-1) — exactly a factor of 10
mismatch. Likely the `cai_unit_M` conversion in calcium_pool.py is being
applied incorrectly (treating cadiff's mM-scale output as M-scale).

**Action:** Revise calcium_pool.py to keep cadiff in mM (NEURON's native
output) and convert at the boundary if/when channels need M-scale.

### F4: Decision — use cai_mM internally to match NEURON

Revising calcium_pool.py:
- `cai_mM` is the Brian2 state variable (matches NEURON's `cai` for cadiff
  and `caintra` for caintra1, both stored in mM-scale).
- Channels needing M-scale Ca apply ×1e-3 at their use site.
- Eliminates the unit-conversion factor mismatch in F3.

### F5: caintra1's `cai` doesn't update — channels reading cai must use mechanism state

Implication for CP3: When we build a Brian2 cell with EGL-19 + caintra1 + leak,
the EGL-19 inactivation (h-gate) reads cai for Ca-dependent inactivation in
some implementations. Nicoletti's egl19.mod does NOT have Ca-dependent
inactivation — it only has voltage-dependent m, h. Verified by reading
egl19.mod.

So the CP3 cell construction is simpler than feared:
- Brian2: leak + EGL-19 (m,h voltage-only) + caintra1 pool tracking. Pool is
  bookkeeping; nothing reads from it for CP3.
- NEURON: leak + EGL-19 + caintra1.

Both should produce identical V(t) and ica(t) because nothing depends on
cai for EGL-19 dynamics. caintra1 is bookkeeping.

**This simplifies CP3 substantially.** Translation correctness reduces to:
match leak-only V trajectories AND EGL-19 isolated kinetics → both
already individually validated. CP3 becomes a multi-channel composition test.

### F6: NMODL hidden unit-conversion factor (CRITICAL, ~52,700×)

**Discovery during cadiff calibration:** Naive Python translation of
cadiff.mod's BREAKPOINT formula `ca = ca + 10000*dt*(-1/(2F)*ica/depth)`
produces Δcai per ms that is ~52,700× larger than NEURON's empirical
output. For ica = -1.22 mA/cm², NEURON gives Δcai/Δt ≈ +0.12 mM/ms;
naive Python gives ~6324 mM/ms.

This is the classic NMODL hidden-unit-conversion gotcha. NEURON's NMODL
compiler implements implicit unit conversions during code generation that
we don't replicate when reading the source as plain Python arithmetic.
The factor 10000 in the source is part of NMODL's unit-conversion
machinery, not a free user-tunable parameter.

**Implication:** Symbolic re-derivation of the cadiff formula produces
incorrect numerical behavior. Translation correctness can only be
established by **direct empirical calibration** against NEURON's output.

**Strategy for CP1.B.5/6:** Calibrate a Brian2 coefficient empirically
by running known-ica trajectories through NEURON, observing the resulting
cai trajectory, and tuning the Brian2 coefficient to match. Document the
calibration explicitly in calcium_pool.py.

**Implication for CP3:** EGL-19 in Nicoletti's parameterization has NO
Ca-dependent inactivation (verified by reading egl19.mod). The CP3 cell
[leak + EGL-19 + caintra1] does NOT consume cai for any dynamic — caintra1
is bookkeeping only. So Ca-pool calibration accuracy is NOT load-bearing
for CP3 V(t) match. We can document the calibration gap, validate cadiff
and caintra1 against NEURON empirically (calibrated), and proceed to CP3
with full transparency about which validation succeeded by direct match
vs. by empirical calibration.

### F7: Calibration strategy for cadiff/caintra1

Recipe:
1. Drive NEURON cadiff with constant ica via cca1 vclamp at known V
2. Measure NEURON's incremental Δcai per Δt at multiple ica values
3. Fit linear: Δcai/Δt = α × ica + β × (cai - floor)
4. α is the empirical coefficient in mM/(mA/cm²·ms); β is decay rate.
5. Substitute into Brian2 eqs.

This produces a phenomenological match valid in the ica regime sampled.
For broader validity we'd need a deeper NMODL inspection (or direct
NEURON-→Brian2 channel-port via `mod2c` intermediate). Out of scope for
this overnight run.

### F8: Decision — calibrate empirically, document, proceed

Per spec "document, don't fabricate" and "honest finding more valuable
than glossed-over result": we will:
1. Calibrate cadiff Brian2 coefficient empirically against NEURON.
2. Calibrate caintra1 similarly.
3. Run combined Ca-pool validation.
4. Document calibration drift (residuals after calibration) as the
   honest measure of translation fidelity.
5. Proceed to CP2 (EGL-19 voltage-only, no Ca dependency) and CP3
   (where Ca-pool is bookkeeping).

If calibration produces > 20% residual error after fitting, this is a
deeper architectural issue and invalidates Path A's assumption that
NMODL → Brian2 translation is feasible at the phenomenological level.
For now, attempt calibration and report residuals.

### F9: Voltage-clamp protocol mismatch (capacitive transient + V step prestep)

**Discovery during CP2:** initial Brian2 vs NEURON comparison failed because:
1. NEURON's voltage-clamp using prestep_mV=-30 (Nicoletti default) sets the
   cell at -30 mV, then steps to test V. The V→V transition produces a
   capacitive transient (Cm·dV/dt) that dominates the first ~1 ms.
2. Brian2's voltage clamp via network_operation force-resets v at every
   timestep, suppressing dV/dt → no capacitive transient.
3. The default `peak_I_pA` finder (signed extremum across step window)
   picks the capacitive transient on NEURON side and the EGL-19 m·h
   activation peak on Brian2 side — they're not the same physical feature.

**Fix:** Added `skip_initial_transient_ms` and `brian2_prestep_ms` parameters
to `voltage_clamp_compare_v2`. With prestep on Brian2 (-60 mV for 50 ms) +
2 ms skip on both sides, peak/SS comparisons align cleanly.

## F14 (run #2 Phase C): h.run() re-finitializes via h.v_init (default -65 mV)

**Discovery during SHL-1 validation.** SHL-1 systematically showed 7.3% peak
divergence Brian2 vs NEURON. Trajectory inspection revealed the cause: NEURON's
`h.run()` (from stdrun.hoc's `init` procedure) re-finitializes the cell using
`h.v_init` (default -65 mV), **silently overriding any explicit `h.finitialize(v_arg)`
call**.

For channels whose `hinf` is voltage-sensitive in the prestep regime (SHL-1's
hinf at -60=0.77 vs at -65=0.86, a 12% difference), this caused initial
inactivation to be at hinf(-65) not hinf(v_init_mV).

**Fix:** in `neuron_reference.py` `voltage_clamp` and `current_clamp` blocks, set
`h.v_init = v_init_mV` BEFORE `h.run()`. After fix, SHL-1 peak divergence dropped
to 0.3% (from 7.3%).

**EGL-19 wasn't affected because:** EGL-19's hinf at -60 vs -65 are both very
close to 1 (de-inactivated at hyperpolarized V), so the offset doesn't propagate
to peak current. SHK-1 similarly unaffected.

**Methodology lesson:** voltage-feature comparison would not catch this if all
tested channels were equally insensitive in the prestep regime. SHL-1 caught it
because its h-gate has a sharper voltage dependence.

## F15 (run #2 Phase C): Brian2 vs NEURON SS extraction window mismatch

**Discovery during SHL-1 validation post-F14.** After the v_init fix, peak matched
to 0.3% but SS still showed 8% divergence at high V. Cause: Brian2 computes SS
as mean of last 20 ms; NEURON's stored ss_I_pA uses last 20% of step (= 40 ms
for 200 ms step). For inactivating channels (SHL-1 at high V), the current
declines monotonically over the step, so different windows give different
SS values.

**Fix:** in `voltage_clamp_compare_v2`, recompute NEURON's SS from raw
trajectory using the same `settle_window_ms` as Brian2. After fix, SS divergence
dropped from 8% to 0.0%.

**Methodology lesson:** any feature extractor must use identical window/algorithm
on both reference and translation. Stored features from the reference should be
recomputed at comparison time when window-sensitivity is plausible.

## F11 (run #2 Phase A): F6 was a misdiagnosis — Ca-pool translation is fully PRINCIPLED

**Discovery during Phase A diagnostic.** Run #1's calcium_pool.py docstring claimed
"Symbolic re-derivation gives ~5183 mM/(mA/cm²·ms), empirical 0.525, ratio ~10000×;
NMODL hidden unit-conversion machinery." This is incorrect.

The proper symbolic derivation gives **0.518 mM/(mA/cm²·ms)** for cadiff. Empirical
NEURON sweep across AVA/AIY/RIM at 9 holding potentials gives **0.5182** (5 dp match).
The 10000 in cadiff.mod is the proper unit-conversion factor from mol/(s·cm³) to mM/ms,
fully derivable from declared units.

For caintra1, symbolic α(geometry) matches empirical α(geometry) to 5 dp at AVA, AIY, RIM.

**No hidden NMODL machinery exists.** Run #1's empirical calibration converged to the
symbolically-correct value because the symbolic derivation IS correct. The docstring's
"5183" claim is internally inconsistent with the production code (which uses 0.525).

**Phase A verdict: PRINCIPLED.** Speculative-architecture fork NOT triggered.
See `f6_symbolic_decomposition.md`, `f6_geometry_analysis.md`, `f6_calibration_robustness.md`,
`f6_diagnostic_synthesis.md` for details.

## F12 (run #2 Phase A): Nicoletti's published cells don't insert Ca-pool

Of 5 cell scripts inspected (AVAL, AVAR, AIY, RIM, VA5), only VA5 inserts cadiff.
AVA, AIY, RIM, AVAR rely on NEURON's default static cai = 5e-5 mM (= ca_eq default).

**Implication:** SLO-1 isolated (which reads cai) sees a constant value in
Nicoletti's actual cells. Brian2 translation does NOT need a dynamic Ca-pool for
Phase D. Use constant cai matching NEURON's default.

## F13 (run #2 Phase A): slo1egl19 has internal calcium(V), doesn't read cai

slo1egl19.mod has a closed-form `calcium(V)` FUNCTION computing nanodomain Ca purely
from V (Lluís-Buchholz/Alvarez-style nanodomain approximation). It reads only `eca`,
not `cai`.

**Implication:** Phase E "nanodomain coupling encoding" architectural decision is
resolved trivially. Match Nicoletti's deterministic V-dependent formula via
algebraic equation in Brian2 eqs string. No sub-membrane vs bulk compartments
needed.

---

## Run #2 plan acknowledgment (invocation 1, 2026-04-26 evening)

Spec read fully (`phase_v_w2_phase_beta_run2_prompt.md`, ~30k chars). Run #1
summary + findings catalog (F1–F10) re-read. Architectural plan and Nicoletti
NMODL sources (`cadiff.mod`, `caintra1.mod`) inspected. Cell-construction
geometry verified in `AVAL_simulations.py`, `AIY_simulation.py`,
`RIM_simulation.py`: stub cylinder L=diam=rsoma=1e4·sqrt(surf/π) μm.

`run2_state.json` did not exist → invocation 1, starting at Phase A.

**No pre-flight scope concerns surface.** The two-cell-construction Phase F
clarification is clean. Speculative-fork risk-acceptance is documented and
isolated to `wave2/speculative/`. Channels in scope all exist as `.mod` files
in Nicoletti's repo. Calibration json + run #1 deliverables are in place.

Proceeding to Phase A. State file written; will update atomically after
each subcheckpoint. Findings will be extended with F11+ as discovered.

---

### F10 (CRITICAL): Unit-conversion bug in initial EGL-19 eqs (1e-3 factor)

**Discovery during CP2:** initial Brian2 EGL-19 SS currents matched leak-only
IV (essentially zero EGL-19 contribution). Investigation showed Brian2's
`ica_egl19_mAcm2` was ~1000× too small — m,h gates were correct but the
output current was off by 1e3.

**Root cause:** my eqs had:
```
ica_egl19_mAcm2 = egl19_gbar * m_egl19 * h_egl19 * (v_mV - egl19_eca) * 1e-3
```
The 1e-3 factor was wrong. Correct unit derivation:
- gbar (S/cm²) × V (V) = A/cm². With V in mV, multiply by 1e-3 to get V.
- A/cm² = mA/cm² × 1e-3. So gbar × v_mV × 1e-3 = mA/cm² × 1e-3 × 1e-3 = mA/cm² × 1e-6 (wrong).
- Correct: gbar × v_mV directly = (A/cm²)·mV = mA·V/(cm²·V) = mA/cm². No factor needed.

**Fix:** removed the 1e-3 factor from EGL19_EQS. Also verified leak side
already had no factor (correct).

**After fix:** EGL-19 11/11 holds pass, max divergence 0.004 (vs 5% tolerance).
IV curve essentially exact match to NEURON. CP2 cleared.

This is an example of the kind of unit-handling error that could destroy
later channel translations if not caught early. It went undetected with
Brian2's "1" (dimensionless) units approach because `1 × 1e-3` is still
dimensionless. Brian2's intrinsic unit checking would have caught it if
we'd used SI units throughout. **Lesson:** when next channel is translated
(SLO-1, SHK-1, etc.), explicitly cross-check leak-relative scale.
