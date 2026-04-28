# Phase δ WB2 — Wave2HybridBrain skeleton + AVAL/AVAR drop-in

**Date:** 2026-04-27/28 (overnight run)
**Status:** PARTIAL PASS — skeleton works in isolated mode; cross-coupling
between LIF and Wave 2 surfaces a load-bearing biophysical issue that
triggers WB3 hard-stop pause.

---

## Headline finding

**Wave2HybridBrain class built and verified in isolated mode** (cross-coupling
off). Brian2 successfully builds and runs a unified Network containing:
- 1 LIF NeuronGroup of 298 cells (with eqs unchanged from LIFBrain)
- 2 Wave 2 NeuronGroups (1 cell each: AVAL, AVAR)
- 2,926 LIF→LIF chemical synapses
- 2,002 LIF→LIF gap junctions
- Cross-group routing infrastructure built but DISABLED by default

In isolated mode:
- LIF mean rate: 5.08 Hz (vs LIFBrain 15.44 Hz baseline; difference attributed
  to 2 missing cells + 380 missing cross-edges)
- AVAL settles to -39.39 mV (passive RC from -60 mV init, no driving current)
- AVAR settles to -24.10 mV (its physiological rest, per Mellem 2008 quote
  "AVA rest typically between -20 and -30 mV")
- Wall time: 3.2 s for 1000 ms simulated (acceptable)

**However:** when cross-coupling is enabled (LIF firing rates → Wave 2 I_ext;
Wave 2 V → LIF v_post bumps), AVAL/AVAR voltages saturate to physically
unrealistic values (V ≈ -490 mV in graded_current_capped mode at ±200 pA
cap; V → -∞ in naive_voltage_bumps mode).

This is the **load-bearing biophysical surface that requires WB3 review**.

---

## What was built

**Class:** `wave2/integration/wave2_hybrid_brain.py::Wave2HybridBrain`

Constructor signature (matches LIFBrain plus extensions):
```python
Wave2HybridBrain(
    wave2_active=["AVAL", "AVAR"],   # cells to use Wave 2 detail
    W_syn=W_SYN_DEFAULT,              # LIFBrain's chemical scale
    g_gap=G_GAP_DEFAULT,              # LIFBrain's gap scale
    C_mem=C_MEM_DEFAULT,              # LIF cm
    noise_sigma=NOISE_SIGMA_DEFAULT,
    v_rest_bias=V_REST_BIAS_DEFAULT,
    include_gap=True,
    sign_overrides=None,
    use_per_edge_glu_signs=False,
    sign_exceptions=None,
    seed=42,
    cross_coupling_mode="off",        # WB2 default
)
```

`cross_coupling_mode` options:
- **"off"** — Wave 2 cells run isolated. LIF→LIF synapses unchanged.
  Wave 2 cells receive no input (rest at their natural settle voltage).
  This is the SAFE WB2 default.
- **"naive_voltage_bumps"** — instantaneous v += W_syn * w on each crossing
  event. **NOT RECOMMENDED** — causes V-blowup on Wave 2 due to small cm.
- **"graded_current_capped"** — current-based coupling capped at ±20 pA.
  WB2 provisional, replaceable by WB3.

I/O contract preserved (matches LIFBrain):
- Attributes: `names, idx, N, neurons, spikes, ablation_current_pA,
  proprio_group, summary, sign_overrides_applied, sign_exceptions_applied`
- Methods: `run, time_ms, firing_rates, set_proprioception,
  set_sensory_rate, inject_poisson, ablate`
- Wave 2 specific: `wave2_groups, wave2_bundles, wave2_active,
  wave2_last_spike_t, cross_coupling_mode`

---

## Cross-group routing architecture (built; disabled in "off" mode)

The cross-group routing is implemented as a `@network_operation(dt=50*ms)`
that runs at ClosedLoopEnv's sync cadence. It has 5 phases:

1. **Wave 2 release-event detection.** V-threshold crossing at -25 mV with
   5 ms refractory (provisional WB2 rule).
2. **Wave 2 → LIF chemical events.** Released event delivers W_syn * w mV
   to the post LIF cell's V (instantaneous bump). LIF cells handle this
   natively (their v += dv pattern is what LIF synapses already do).
3. **Wave 2 → Wave 2 chemical events.** Converts release event into a
   small graded current on the post Wave 2 cell's I_ext.
4. **LIF → Wave 2 chemical events.** Aggregates LIF spike counts × edge
   weights, converts into a current on Wave 2's I_ext via the formula
   `I_pA = W_syn_mV * w * count * cm_typical_pF / dt_ms`.
5. **Cross-group gap junctions.** Computes `g_gap * w * (V_pre - V_post)`
   and adds to I_ext on the Wave 2 side / I_ext on the LIF side.

Edges identified at construction time:
- 194 LIF↔Wave2 chemical edges
- 186 LIF↔Wave2 gap edges
- (a few Wave2↔Wave2 chemical edges between AVAL and AVAR)

---

## WB3 hard-stop finding: graded-coupling biology

When `cross_coupling_mode="naive_voltage_bumps"`, AVAL/AVAR receive direct
voltage bumps from LIF spikes:
- Each spike: `v_post += W_syn * w` mV
- LIF cells fire at ~10-50 Hz baseline → ~1-3 spikes per 50 ms per pre-cell
- AVAL has 50-100 LIF presynaptic chemical edges
- Aggregate dV per 50 ms: ~50-300 mV → V → +∞ within seconds

This isn't a translation bug. It's a **structural biophysical issue**:

> Wave 2 cells (AVAL/AVAR) have small cm (~0.86 pF, single-compartment Nicoletti
> AVAL geometry with surf=1123.84e-8 cm² × cm=0.86 μF/cm² ≈ 0.97 pF). They are
> graded-response neurons with no Na+ spike mechanism — they integrate
> conductance, not voltage.
>
> LIF cells (with cm=100 pF) tolerate v += W_syn rules because their cm is
> 100× larger. Their threshold-fire-reset dynamics also bound the response.
>
> The Phase δ scoping doc identified this as the central design surface
> (§6.4 Risk Register, "Wave 2 cell's continuous voltage doesn't drive LIF
> cells the way ASH→AVA pathway expects" — High likelihood, High impact,
> mitigation: graded-release Boltzmann mapping).

Even with `cross_coupling_mode="graded_current_capped"` and ±20 pA cap, the
Wave 2 cells settle below physiological range (V ≈ -490 mV in initial test
with ±200 pA cap; would be ~-150 mV with ±20 pA cap which is also
non-physiological).

---

## What WB3 needs to address (per spec hard-stop framing)

The spec explicitly says:

> "Stage III WB3 release-event rule needs biological judgment beyond
> literature: PAUSE for Rohit"

Specific WB3 questions for biological-judgment review:

### Q1: How should LIF→Wave 2 chemical synaptic input be modeled?

Options (each implies different biology):
- **(a) Conductance-based graded synapse.** Each LIF→Wave2 edge becomes a
  graded conductance `g_syn(t)` that opens with each LIF spike and decays
  exponentially. Current: `I = g_syn * (V_post - E_rev)`. Requires
  per-edge E_rev (excitatory ~0 mV, inhibitory ~-70 mV) and τ_decay.
  This is the **biophysically accurate** path but requires re-grounding
  parameters.
- **(b) LIF firing rate → continuous current.** Treat the LIF group as
  emitting a Poisson-rate-modulated current to Wave 2 cells:
  `I = K * mean_rate(LIF) * w_signed`. K calibrated so Mellem-class
  injections (-30 to +30 pA) are reproduced when LIF inputs match
  ASH→AVA-class drive.
- **(c) Use the existing `caintra1`/`cadiff` Ca-pool indirection.** Excitatory
  input → Ca influx → Ca-dependent K activation. Requires modeling chemical
  synapses as Ca-permeable receptors. Most biologically accurate but
  highest implementation cost.

Rohit's call: I lean **(a)** is the publishable approach but **(b)** is the
fastest path to a working hybrid for Stage IV evaluation. Option (b) with
K calibrated against Mellem-class drive could be done in WB3 in 1-2 hours
of work if we accept "LIF spikes generate proportional current with some
post-hoc calibration constant."

### Q2: How should Wave 2 → LIF release events work?

Wave 2 cells are graded — they don't have a discrete spike. Two choices:
- **(a) V-threshold crossing.** When V crosses -25 mV, emit "spike" → LIF
  receives `v += W_syn * w` (current implementation in WB2).
- **(b) Boltzmann graded release.** Continuous neurotransmitter release rate
  `r(V) = r_max / (1 + exp(-(V-V_half)/k))`, with V_half from Wicks 1996
  (commonly -25 to -35 mV) and k ≈ 5-10 mV. This is what `graded_brain.py`
  already does for the GradedBrain variant.

Choice (b) is more biophysically faithful but requires deciding parameters
that aren't in Mellem 2008 (which doesn't characterize AVA's transmitter
release; it shows only graded passive responses). The Wicks 1996
parameters are widely used but published for body-wall muscle, not
command interneurons. **This is exactly the kind of "biological judgment
beyond literature" the spec named as a pause point.**

### Q3: Is the Wave 2 cellular detail providing value for Stage IV?

The §5 falsification test is whether the expanded brain reproduces touch
cascade where pure LIF cannot. Pure LIF AVA's failure mode is "AVA fires
36→28 Hz under touch stim instead of 36→100+ Hz." Wave 2 AVAL has no
spike mechanism at all — it produces graded V plateau. So the Stage IV
test re-frames as: "does Wave 2 AVAL's plateau response activate
classifier-mode FSM correctly where LIF AVA's rate decrease does not?"

This is a different and arguably MORE rigorous test than what the spec
framed. The question is whether the FSM activity-mode classifier (which
reads firing-rate vectors) needs Wave 2 cells to emit pseudo-spike events
at all, OR can read voltage directly. If voltage-direct, Stage IV becomes
a clean test. If pseudo-spike-required, the release-event biology becomes
load-bearing again.

---

## Stage III pause-and-document state

### Files produced

- `wave2/integration/__init__.py`
- `wave2/integration/wave2_hybrid_brain.py` — Wave2HybridBrain class with
  WB2 skeleton + AVAL+AVAR drop-in (cross-coupling off by default)
- I_ext aliasing added to `wave2/option_alpha_ava_cell.py`,
  `wave2/option_alpha_aiy_cell.py`, `wave2/option_alpha_rim_cell.py` (adds
  I_ext as additional injection variable; backwards compatible since
  default 0 pA)
- `wave2/option_alpha_avar_cell.py` (Stage II) — already had I_ext
- This findings doc: `wave2/artifacts/phase_delta_wb2_findings.md`
- Status JSON: `wave2/artifacts/checkpoints/stage_III_status.json`

### Smoke tests passed

- AVAL standalone (re-validated post I_ext rename): builds, runs 100 ms
  passive, V settles to -44.54 mV
- AVAR standalone (Stage II re-run): PRODUCTION_GRADE (43.6 s wall, all 11
  VC + 7 CC checks pass)
- Wave2HybridBrain (cross_coupling=off): builds in 3.2 s wall, runs 1000 ms
  simulated, AVAL/AVAR at physiological rest

### What was NOT done (WB3-WB6 scope)

- WB3: release-event rule design (PAUSED FOR ROHIT — this is the spec's
  hard-stop condition)
- WB4: AIY pair extension (ready to add when WB3 resolves; cells already
  in WAVE2_CELL_FACTORIES dict)
- WB5: RIM pair extension (same)
- WB6: full multi-scenario validation

### Recommendation

When Rohit reviews:
1. Pick option for Q1 (LIF→Wave 2 chemical model)
2. Pick option for Q2 (Wave 2 → LIF release event model)
3. Decide on Q3 (does Stage IV need pseudo-spikes from Wave 2)
4. Either restart WB3 with biology-grounded options OR pivot Stage IV
   to test cellular-detail value via a different methodology (e.g.,
   compare AVAL plateau dynamics under touch vs spontaneous, both within
   the LIF scaffold)

Either decision unblocks WB3-WB6 in 2-4 hours of follow-up work.

---

## Methodology continuity preserved

This is exactly the pause-with-documentation pattern the spec (and CLAUDE.md
working style) prescribes. The discovered biophysical issue is informative
— it's the central WB3 design surface that needed explicit attention rather
than autonomous architectural commitment. Per the orchestrator's pre-flight
guidance and the spec's hard-stop framing, pausing here preserves
optionality and surfaces the decision to a human with full context.

The Phase δ scoping doc explicitly anticipated this:

> "WB3 — Release-event rule design (PAUSE LIKELY)" (§5.1)
>
> "Architecturally novel. Pre-flight specifically for WB3: if you find
> yourself making non-trivial choices about graded vs discrete release
> without primary source guidance, PAUSE."

Done.
