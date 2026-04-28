# Harness fitness report — pre-Phase-β check

**Read-only inspection.** No code modified. Output informs whether Phase β overnight scope expands to include harness iteration before CP2.

---

## Summary verdict

**Mixed — minor iteration on voltage-clamp harness; substantive iteration on plateau harness. Recommend folding harness work into Phase β CP1 scope (~2-4 hours) rather than running Phase β overnight on as-built infrastructure or spawning a separate work block.**

The voltage-clamp harness is close to fit-for-Phase-β with two well-bounded additions (NEURONReference wrapper + metric clarification). The plateau harness has a structural gap — it tests Brian2 against Mellem 2008 target ranges (Gate 2b), but does **not** compare Brian2 cell against NEURON reference cell (Layer A in cell context). CP6 explicitly requires the latter.

Both harnesses' authors flagged the Phase β refactor items in module-bottom comments — the gaps below are partly anticipated by Phase α work, not surprises.

---

## Voltage-clamp harness assessment

| Phase β requirement | Status | Note |
|---|---|---|
| Brian2 channel + NEURON reference, matching VC protocols, comparison output | **ready** | Factory pattern + reference-callable signature is generic enough |
| Ca-pool dependency support (EGL-19 inactivation, SLO-1 gating reading [Ca]_i) | **minor iteration** | Current factory returns a single NeuronGroup with `v` + `I_total`. Ca-pool channels require initializing [Ca]_i to steady state for the held voltage, OR running long enough for the pool to equilibrate before measuring SS current. Encodable inside the eqs string of the factory; doesn't require harness API change. Decision needed: encode Ca-pool inside eqs string (cleaner) vs treat as separate subsystem (more architectural). |
| Voltage-feature ≤3 mV / >80% gate | **needs clarification** | The Phase β-pre v3 metric (voltage-feature, ≤3 mV abs, >80% steps) is a current-clamp criterion. Voltage-clamp inherently forces V — there is no "peak voltage" feature. The natural VC analog is **current divergence per hold, X% tolerance, >80% of holds pass**. Current harness uses 5% tolerance on SS current; this may inherit the "small-denominator pathology" v1/v2/v3 surfaced (currents are near-zero at activation thresholds). Recommend defining a current-domain analog of the v3 voltage-feature gate before CP5: e.g., `≤ max(5% relative, 0.5 pA absolute) per hold, >80% pass`. |
| Timing features as warn-only | **N/A** | Not relevant for VC; V is forced. |
| NEURONReference wrapper integration | **minor iteration** | Wrapper not yet built. Reference callable signature `(hold_mV, dur_ms, dt_ms) → (t, V, I_pA)` is sufficient — wrapper just needs to expose `__call__` matching it. Wrap `h.Section` + `h.VClamp` + persistent mech handles. ~30-50 lines. |
| Multiple cells with different protocol params | **ready** | Per-call params already cover this. |

**Subtotal:** voltage-clamp harness is **ready with minor iteration** — NEURONReference wrapper class + metric clarification. ~1-2 hours.

---

## Plateau harness assessment (Phase β CP6 only — CP1-CP5 don't use this)

| Phase β CP6 requirement | Status | Note |
|---|---|---|
| Brian2 cell with imported channels, Mellem-style current injection | **ready** | Factory pattern accepts arbitrary cell construction. |
| **Compare voltage trajectory against NEURON reference cell** | **substantive iteration** | **Structural gap.** Current `current_clamp_plateau` runs ONLY the Brian2 cell and compares against Mellem 2008 target ranges (Gate 2b verifier). There is no NEURON reference path. CP6 is Layer A (Brian2 cell vs NEURON cell), which requires a parallel NEURON simulation + per-feature comparison. Cleanest path: add a `current_clamp_layer_a_compare(brian2_factory, neuron_factory, ...)` function that runs both, extracts features from both, computes per-feature divergence with v3's voltage-feature ≤3 mV gate. Existing `current_clamp_plateau` stays for Gate 2b. ~80-150 lines. |
| Report plateau amplitude, duration, termination | **ready** | Computes amplitude_mV, duration_ms, baseline_post_mV, settle_offset_mV, plus release-dynamics block. |
| Distinguish ignite-fail / terminate-too-fast / held-correctly | **ready (coarse)** | Partition via pass_amp / pass_dur / pass_settle flags. Coarser than spec implies; for CP6 sufficient. |

**Subtotal:** plateau harness has a **structural gap for CP6 Layer A**. ~1.5-2 hours work.

---

## Specific concerns flagged

1. **Comparison metric clarification is upstream load-bearing.** The Phase β-pre v3 voltage-feature gate (≤ 3 mV absolute, > 80% steps) was adopted for current-clamp comparison. Voltage-clamp needs an analogous current-domain gate. Without this, Phase β CP5 (EGL-19 voltage-clamp Layer A check) inherits the same "small-denominator-on-relative-tolerance" pathology that v1/v2/v3 hit three times. The current harness's 5%-with-1e-9-floor tolerance is naive; needs the v3-style relative-with-absolute-floor formulation. **Resolve before CP5 launches.**

2. **No Ca-pool encoding decision yet.** EGL-19 reads [Ca]_i for inactivation. SLO-1 reads [Ca]_i for gating. cadiff/caintra1 are the Ca-pool subsystems. The harness factory pattern doesn't have a first-class concept of cross-channel state. Two encoding options:
   - (a) **Inside eqs string**: factory returns one NeuronGroup whose equations include both channel currents and [Ca]_i dynamics (`d[Ca]_i/dt = ...`). Simplest, matches single-compartment Nicoletti. Recommended for CP1-CP4.
   - (b) **Separate subsystem**: factory returns multiple linked groups via Brian2 `Synapses`-style coupling. More complex, generalizes if multi-compartment work happens later.
   - Decision should be made at start of CP1 and applied consistently. (a) is cheaper and matches Nicoletti's single-compartment models.

3. **Smoke tests validated synthetic phenomenological cells, not channel kinetics.** Phase α was prototype-first by design. The voltage-clamp smoke test ran a leak-only cell (no gating dynamics). The plateau smoke tests ran phenomenological "Ca-like + SLO-like" scaffolds with arbitrary equations — not real EGL-19 m²h kinetics or real SLO-1 Ca-gating. **Both harnesses' actual behavior on real channel kinetics is unverified.** First Phase β channel translation (cadiff or EGL-19) doubles as harness validation. Expect 1-2 unexpected harness bugs surfacing during CP1-CP5; budget for them.

4. **Missing NEURONReference wrapper.** Voltage-clamp harness's reference-callable interface is generic, but no wrapper exists yet. Refactor flag #3 in `voltage_clamp_harness.py` documents this: "wrapping h.VClamp requires section setup and persistent state — define a NEURONReference class that holds the section + mechanism handles and exposes a matching call signature." Build at start of CP1; needed for any cross-validation.

5. **Plateau harness's release-dynamics classifier uses scaffold-tuned thresholds.** Lines 245-250: `architectural_signature` classification uses ratio thresholds (0.6 / 1.4) chosen to cleanly distinguish the synthetic scaffolds. Real EGL-19 + SLO-1 cells may produce ratios that don't fit these thresholds. Refactor flag #4 in the file. Empirical recalibration needed once real cells are running. **Not blocking for CP1-CP5; resolve before CP6 Gate 2b lock-in.**

6. **Factory pattern creates a fresh Brian2 Network per holding step.** Refactor flag #1. Pays scope-init cost on every `voltage_clamp_compare` per-hold iteration. For Phase β with 7-channel essential set × 16 holds × multiple cells, cumulative overhead is non-trivial. Defer optimization unless it becomes problematic — premature optimization here is real risk.

---

## Iteration scope estimate (if user accepts the verdict)

**Fold into Phase β CP1 scope (cadiff/caintra1 translation), executed before CP2 launches:**

1. **NEURONReference wrapper class** (`scripts/brain/wave2/neuron_reference.py`, ~50 lines): wrap `h.Section` + `h.VClamp` + mechanism handles; expose `__call__(hold_mV, dur_ms, dt_ms) → (t, V, I_pA)`. Reusable across all CP5+ work. ~30-45 minutes.

2. **Voltage-clamp metric refinement** (modify `voltage_clamp_compare` tolerance handling): replace single `tolerance: float = 0.05` with explicit relative-tolerance + absolute-floor params. Define current-domain v3-analog gate. ~30 minutes.

3. **Plateau harness Layer A extension** (new function `current_clamp_layer_a_compare` in `plateau_harness.py` or a new `plateau_layer_a.py`): runs both Brian2 and NEURON cells under matching CC protocol, extracts {peak_v, plateau_amp, plateau_dur, termination_tau} from each, applies v3 voltage-feature gate (≤ 3 mV absolute, > 80% pass). ~1-2 hours.

4. **Ca-pool encoding decision** (architectural, not coding): pick (a) eqs-string encoding or (b) subsystem; document choice; implement in CP1's cadiff factory. Decision time, not coding time. ~15 minutes of explicit thought.

**Total: ~2-4 hours folded into CP1.** Phase β CP1 produces (a) cadiff translation + (b) harness fitness for the rest of Phase β. Phase β CP2 onwards then runs against properly-fit infrastructure.

**Alternative: separate harness-fitness work block before Phase β.** Cleaner separation but spawns another session before any channel translation work begins. Not recommended unless the "smoke tests validated synthetic, not kinetics" concern surfaces a deeper structural bug during CP1 — in which case escalate.

---

## Documented harness behaviors worth knowing for Phase β

- Both harnesses' module-bottom "Phase β refactor flags" comment blocks anticipate ~half of these gaps. The harness authors knew prototype-first meant some items deferred.
- Voltage-clamp harness implements clamp via `network_operation` that resets `v` to target each timestep — works for SS, may produce numerical artifacts on transient capture. Phase β CP5 likely needs to switch to a virtual-electrode high-conductance clamp current for proper transient measurement.
- Plateau harness's release-dynamics fits a single exponential to decay. Real EGL-19 + SLO-1 + leak cells likely show biexponential decay (fast leak + slow K_Ca). Refactor flag #3 anticipates this. Watch for poor R² on the exponential fit during CP6.
- Both harnesses use Brian2 numpy code-generation (`prefs.codegen.target = "numpy"`). Cython would be faster for production runs but compiler-toolchain-sensitive on Linux. Numpy is the safer default for now.

---

## Recommendation

Adopt the verdict: **minor iteration on VC harness, substantive iteration on plateau harness, fold both into Phase β CP1 scope**. Total ~2-4 hours of harness work executed alongside cadiff/caintra1 translation, both completed before CP2 launches. Phase β overnight does not launch as-built; Phase β overnight launches with harness iteration as the first ~half-day of CP1 work.

If user prefers separate harness-fitness work block before any Phase β work begins, that's defensible but heavier-process — the iteration list above is concrete enough that a single CP1 session can handle both translations and harness fixes without scope creep.
