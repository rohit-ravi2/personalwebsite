# Phase δ WB3 CP1 — Release-event rule options

**Status:** options document. **Awaits Rohit's adjudication before CP2
implementation.** Spec hard-stop pause-for-review per
`phase_v_w2_phase_delta_wb3_prompt.md`.

**Date:** 2026-04-26.

**Purpose.** Wave2HybridBrain's cross-group coupling between Wave 2
biophysical cells (AVAL/AVAR; future AIY/RIM) and the LIF scaffold
needs a release-event rule. WB2 demonstrated that the naive
voltage-bump and capped-current approximations are not viable
(`phase_delta_wb2_findings.md`). This document presents three
candidate rules, primary-source-grounded constraints, recommendation
with rationale, and the explicit decision points needing Rohit's
biological judgment.

**Scope of this document.** Section 1 quantifies WB2's empirical
constraints (with a corrected capacitance arithmetic). Section 2
extracts biological constraints from primary sources. Section 3
presents the three options with explicit tradeoffs. Section 4 gives
the pre-flight recommendation and why. Section 5 enumerates decisions
that require biological judgment (with consequences and defaults).

---

## Section 1 — Empirical constraints from WB2

### 1.1 Capacitance mismatch — quantified

WB2 findings claimed Wave 2 cells have C ≈ 0.86 pF (ratio LIF/Wave2
≈ 116×). Re-derivation from the Brian2 cell builders yields
order-of-magnitude different values; the WB2 quote appears to confuse
*specific* capacitance (μF/cm²) with *total* capacitance (pF).

The Brian2 equation for AVAL (option_alpha_ava_cell.py, line 160) is

```
dv/dt = -I_total / (cm_uFcm2_param * surf_cm2_param * 1e6 * pF)
```

so the effective total capacitance is the product `cm_uFcm2_param ×
surf_cm2_param × 1e6 [pF]`. For the four Wave 2 cells:

| Cell | surf (cm²) | specific cm (μF/cm²) | C_total (pF) | ratio LIF/Wave2 |
|------|-----------:|---------------------:|-------------:|----------------:|
| AVAL | 1123.84e-8 | 0.859551             | **9.66 pF**  | **10.35×**      |
| AVAR | 1121.79e-8 | 0.751761             | **8.43 pF**  | **11.86×**      |
| AIY  | 65.89e-8   | 1.6                  | **1.05 pF**  | **94.86×**      |
| RIM  | 103.34e-8  | 1.5                  | **1.55 pF**  | **64.51×**      |
| LIF  | (lumped)   | n/a                  | **100 pF**   | 1×              |

`C_MEM_DEFAULT = 100 * pF` per `lif_brain.py:106`.

**The mismatch is real. The magnitude was overstated.** For the
WB2-active cells (AVAL/AVAR), the ratio is ~10×, not 116×. For the
yet-to-drop-in AIY/RIM, the ratio rises to 65-95× because their
cells are physically smaller. For F20 catalog entry (CP5 deliverable),
record the corrected values.

### 1.2 Failure mode of `v += W_syn * w`

Charge delivered per LIF spike to a Wave 2 cell, in `v_post += W_syn
* w` mode:

`ΔQ = C_post · W_syn · w = 8-10 pF · 0.8 mV · w_signed ≈ 6-8 fC · w`

per LIF spike per edge. For a single-edge case, ΔV per spike is exactly
W_syn · w = 0.8 · w mV (the bump amount itself, regardless of C).
This is the "bug": the rule is a *voltage* bump, not a *charge*
delivery. Voltage bumps DO NOT scale with C; they accumulate
unboundedly across spikes-and-edges-per-step.

WB2 quantified the aggregate (`phase_delta_wb2_findings.md` line 109-111):

> "AVAL has 50-100 LIF presynaptic chemical edges. LIF cells fire at
> ~10-50 Hz baseline → ~1-3 spikes per 50 ms per pre-cell. Aggregate
> dV per 50 ms: ~50-300 mV → V → +∞ within seconds."

The 50-300 mV figure is invariant under capacitance correction: it's
arithmetic on `W_syn × Σ w_signed × spike_count`. The corrected
capacitance value does not rescue the rule; it remains structurally
unstable.

### 1.3 Numerical-stability requirements for the release rule

The release rule must satisfy:

1. **Bounded V trajectory.** Wave 2 cells must remain within their
   physiological V range (~ -70 to +5 mV for AVA-class cells per
   Mellem 2008; ~ -50 to -10 mV for AIY/RIM per Nicoletti 2024
   passive simulation). No rule may produce V → +∞ or V < -150 mV
   under any combination of feasible LIF input rates.
2. **Charge-conservation semantics.** Cross-group coupling must
   deliver *charge* (or current), not voltage bumps, so the Wave 2
   cell's small C correctly limits the V excursion per input event.
3. **Polarity preserved.** `sign_overrides` and per-edge sign
   decisions made in `lif_brain.py` (`DEFAULT_SIGN_OVERRIDES`,
   `DOCUMENTED_SIGN_EXCEPTIONS`) must propagate transparently — i.e.,
   exc/inh classification of an edge must produce
   depolarizing/hyperpolarizing input regardless of the
   release-rule's mathematical form.
4. **Stable across dt mismatch.** Wave 2 cells run at dt=0.025 ms,
   LIF runs at dt=0.1 ms (per scoping doc §3.2). The release rule
   must produce identical aggregate-current expectations under
   per-NeuronGroup `clock` keyword scheduling, with no race
   conditions on (summed) variables.
5. **Cython-compilable.** Eqs must use Brian2 idioms that codegen
   under cython without fallback to numpy (verified for
   `(summed)` cross-group; see `cython_migration_summary.md`).

### 1.4 What WB2 *did* establish (corrected)

- Cross-group routing infrastructure (cross_chem_edges,
  cross_gap_edges) is built and verified.
- 194 LIF↔Wave2 chemical edges, 186 LIF↔Wave2 gap edges identified.
- Isolated mode (cross_coupling_mode="off") works:
  AVAL/AVAR settle to physiological rest (-39.4, -24.1 mV).
- Naive voltage bumps (cross_coupling_mode="naive_voltage_bumps")
  blow up. **Corrected reason: not capacitance scale per se but
  voltage-bump semantics.**
- Capped current mode (cross_coupling_mode="graded_current_capped")
  with ±20 pA hard cap is non-physiological — caps mask the input
  signal rather than fix the rule.

---

## Section 2 — Biological constraints from primary sources

### 2.1 Graded transmission is the empirical default in C. elegans

**Lockery & Goodman 2009** (Nat Neurosci 12:377-378, PMC3951993) is
the canonical review. Direct quotes (via repo cross-ref +
fetched PMC summary):

> "The small size and high resistance of C. elegans neurons makes
> them sensitive to the random opening of single ion channels,
> probably rendering codes that are based on classical, all-or-none
> action potentials unworkable."

> "Classical voltage-gated sodium channels are absent in the
> C. elegans genome."

The four regenerative-event types they enumerate: action potentials,
graded potentials, intrinsic oscillations, **plateau potentials**.
The 2009 review explicitly reframes the C. elegans regenerative
event as the *plateau potential* (referencing Mellem 2008 RMD
recordings), not the classical AP.

**Goodman MB, Hall DH, Avery L, Lockery SR (1998)** (Neuron
20:763-772, PMID 9581767) is the foundational electrophysiology
paper — first patch-clamp recordings from a C. elegans neuron (ASE).
Key finding (per fetched summary):

> "ASER is nearly isopotential and fails to generate classical Na+
> action potentials. Rather, ASER displays a high sensitivity to
> input currents coupled to a depolarization-dependent reduction
> in sensitivity that may endow ASER with a wide dynamic range."

> Most C. elegans neurons "conduct excitation passively and do not
> need regenerative action potentials."

**Liu Q, Hollopeter G, Jorgensen EM (2009)** (PNAS
106:10823-10828, PMC2705609) verified graded transmission at the
C. elegans neuromuscular junction:

> "Caenorhabditis elegans neuromuscular junctions release
> neurotransmitter in a graded fashion."

> "graded presynaptic depolarization of the motor neurons by
> photo-stimulation elicits graded postsynaptic currents."

(This study covered NMJ — ACh and GABA motor neurons. Not AVA/AIY/RIM.)

**Mellem 2008** (Nat Neurosci 11:865-867, PMC2697921), per the
in-repo verified quotes (`mellem_investigation_pushback.md`):

> "In contrast, we never observed action potentials in AVA (n=10;
> Fig. 1b). The resting potential of AVA was typically between −20
> and −30 mV and we did not observe action potentials (Fig. 1d),
> even when we changed the resting potential to more hyperpolarized
> levels."

> "In contrast to what was observed in RMD, glutamate application
> caused short-lived, modest changes in AVA membrane potential with
> no switch to a new steady-state potential (n = 5; Fig. 3i)."

**Implication for release rule:** every primary source converges on
"graded, not spike-based." A discrete-spike approximation (Option A)
is not biologically faithful for AVA/AIY/RIM. Whether this matters
*for the WB3 deliverable* depends on whether the LIF scaffold
reading the Wave 2 cell's output needs spike-arrival semantics
(downstream LIF cells *do* fire, and their on_pre rules expect
events). See Section 3.

### 2.2 Wicks 1996 sigmoidal release equation — verbatim

**Wicks JF, Roehrig CJ, Rankin CH (1996)** J Neurosci
16(12):4017-4031, PMID 8753865, DOI
10.1523/JNEUROSCI.16-12-04017.1996. Per fetched PMC text
(PMC6578605):

The synaptic conductance is a sigmoidal Boltzmann of presynaptic V
(eq. 6 in paper):

> "g∞(VPRE) = ḡ / [1 + e^(K(VPRE − VEQ)/VRANGE)]"

with parameters (eq. 7 + caption):

> "we used a value of K = 2ln(0.9/0.1) = −4.3944"
>
> "an average of the two VRANGE values...to estimate the activation
> range (−35 mV)."

Where `VEQ` (≈ V_half) is "set near presynaptic resting potential"
per Davis & Stretton 1989 *Ascaris* data. Wicks does not give a
single universal V_half — it's per-cell-type, anchored to that cell's
V_rest.

Substituting the K and VRANGE values, the slope parameter γ in the
common form `1/(1+exp(-γ(V-V_half)))` is

`γ = -K / VRANGE = -(-4.3944) / -35 mV = -0.126 mV⁻¹`

(negative because Wicks's sign convention has g∞ → 0 at high V; the
common downstream form flips this.) Equivalently, slope `k = 1/γ ≈
7.94 mV` — the depolarization required to traverse from 10% to 90%
activation is ~35 mV / (2 ln 9) ≈ 7.97 mV per *e*-fold.

The widely-cited "γ = 0.15 mV⁻¹" value in subsequent literature
appears to be a slightly tightened slope; it corresponds to
`k ≈ 6.67 mV`. `graded_brain.py` uses `k_half = 6 mV`.

**Critical scope statement from Wicks 1996 (verbatim):**

> "Electrophysiology on C. elegans cells is still in its infancy"
>
> "In the absence of detailed physiological data from C. elegans, it
> was necessary to make a number of extrapolations from the related
> nematode Ascaris lumbricoides."

The Ascaris source (Davis & Stretton 1989) recorded *commissural
motorneurons*, not interneurons. Wicks's extrapolation to AVA, AVB,
AVD, etc., is itself a biological-judgment leap not validated in
either species against C. elegans interneuron data.

**Cells covered by Wicks 1996** (per fetched Figure 1 caption):

> "The circuit consists of seven sensory neurons (shaded circles),
> nine interneurons (unshaded circles)."

Sensory: PLM, ALM, AVM, PVD. Interneurons: AVA, AVB, AVD, PVC, DVA
(and a few more). **NOT covered: AIY, RIM, RIS, AIB.**

### 2.3 AVA-specific release biology — Mellem 2008 + Faumont 2011

Mellem 2008 (in-repo verified):
- AVA rest -20 to -30 mV.
- AVA shows no AP (n=10), even at hyperpolarized initial conditions.
- Glutamate produces "short-lived, modest changes" — graded, not
  bistable.

Per Faumont 2011 (PLoS ONE 6:e24666), AVA shows "large calcium
transients that begin with the onset of backward locomotion, peak
around the end of backward locomotion during the onset of forward
locomotion, and then slowly decay." This is calcium-imaging, not
voltage; the time scale is seconds (slower than transient
release-event biology).

**Implication:** AVA's V dynamics are graded (no AP, no plateau in
AVA per Mellem). Release from AVA is graded transmitter release on
the time scale of V changes; the V_half is plausibly near AVA's rest
(-20 to -30 mV) per the Wicks-extrapolated convention. Release-rule
candidate parameters (where AVA is the pre-cell):

- **V_half (AVA→post):** ≈ -25 mV (midpoint of Mellem's reported
  rest range; aligns with WB2's ad hoc threshold value).
- **k (AVA→post):** 6-8 mV (Wicks-derived, no AVA-specific
  measurement).
- **g_max (AVA→post):** **NOT SET BY ANY PRIMARY SOURCE.**
  Calibration parameter; choices in Section 5 (Decision 4).

### 2.4 AIY and RIM release biology — primary-source gap

AIY:
- Faumont 2011, Clark 2006, Beverly 2011 — Ca imaging of AIY
  in food / temperature contexts; voltage range not directly
  reported.
- Clark et al. 2006 (PLoS Biol) reports AIY "graded response
  to food signal" — qualitative, not quantitative V_half / k.
- **No AIY V-clamp recordings establishing V_half or k in primary
  literature**. AIY's V_rest is reported in Nicoletti 2024
  simulations as ~-65 mV (in our cellular validation results).

RIM:
- RIM is tyramine-releasing; modulator-layer biology is captured
  in `modulation_layer.py`, separate from cellular V-driven
  release.
- Lindsay et al. 2011, Liu et al. 2017 — RIM characterization
  in reversal context; Ca-imaging with V correlates inferred.
- **No RIM V-clamp recordings establishing V_half or k in primary
  literature**.

**Implication for AIY and RIM release rules:** any sigmoidal-release
parameters used will be extrapolations. The choices are:
1. Use Wicks-1996 Ascaris-derived parameters universally (V_half =
   V_rest, k ≈ 6-8 mV) — explicit extrapolation acknowledgment.
2. Tune V_half per-cell-type against cellular V_rest from Wave 2
   validation traces — same pattern as Wicks's "anchored to V_rest"
   convention, but with the C. elegans cell's actual rest.
3. Defer AIY/RIM coupling to CP2-CP4 follow-up (only AVAL/AVAR for
   WB3 — narrows scope, but breaks Phase G + Wave2 multi-cell
   timeline).

This is a **decision point** (Section 5, Decision 3). It cannot be
resolved by primary sources alone.

### 2.5 What primary sources do NOT resolve

For complete transparency, the following questions are NOT answered
by the verified corpus:

1. **AVA-specific V_half and k** — Wicks gives Ascaris values; no
   C. elegans direct recording.
2. **Maximum synaptic conductance ḡ** — Wicks reports ḡ as a
   per-cell-pair calibration parameter; no normative value.
3. **Synaptic time constants** (τ_rise, τ_decay) — none of Wicks,
   Mellem, Lockery/Goodman report these for C. elegans interneurons.
   Liu 2009 reports kinetics for NMJ only.
4. **Reversal potentials (E_rev)** for excitatory vs inhibitory in
   C. elegans interneurons — generic values from vertebrate
   electrophysiology (E_glu ≈ 0 mV, E_GABA ≈ -70 mV) used in
   `graded_brain.py` and most literature; not directly measured
   for AVA/AIY/RIM.

These gaps drive the Section 5 decision points.

---

## Section 3 — Three candidate release rules with explicit tradeoffs

The three rules are presented in increasing order of biological
fidelity and implementation complexity.

For all three, the **directionality** is split:

- **Wave2 → LIF (forward)**: how does a Wave 2 cell's V drive a LIF
  postsynaptic cell?
- **LIF → Wave2 (reverse)**: how does a LIF spike-arrival drive a
  Wave 2 postsynaptic cell?
- **Wave2 → Wave2 (intra)**: AVAL ↔ AVAR (and future AIY/RIM)
  cross-cell.
- **Gap junctions (bidirectional)**: orthogonal to the chemical
  release-rule choice; handled identically across all three options
  via Brian2 (summed) `g_gap * w_gap * (v_pre - v_post)` (already
  works in WB2 LIF↔LIF; extension to cross-group is mechanical).

### Option A — V-threshold crossing (discrete spike-event)

**Implementation summary.** Treat each Wave 2 cell's V trajectory as
emitting a discrete "spike event" when V crosses a threshold (e.g.,
-25 mV) with refractory. LIF→Wave2 reverses: each LIF spike-arrival
delivers a fixed current bump to the Wave 2 cell's I_ext.

```python
# Wave2 NeuronGroup eqs add:
# (no change — V is continuous)

# Wave2-to-LIF Synapses (one per pre Wave 2 cell):
syn_w2_to_lif = Synapses(
    wave2_groups[name], lif_neurons,
    model="w : 1",
    on_pre="v_post += W_syn * w",           # exc
    # or on_pre="v_post -= W_syn * w" for inh
    method="euler",
)
# Threshold detection on Wave 2 cell:
wave2_groups[name].thresholder = "v > -25*mV"
wave2_groups[name].refractory  = 5*ms
# Brian2 emits a spike event natively when condition fires.

# LIF-to-Wave2 Synapses:
syn_lif_to_w2 = Synapses(
    lif_neurons, wave2_groups[name],
    model="w : 1",
    on_pre="I_ext_post += W_syn_pA * w",   # bump current, not voltage
    method="euler",
)
```

**Biological grounding.** Weak. Mellem 2008 explicitly reports AVA
does NOT show APs ("we never observed action potentials in AVA").
The threshold-crossing approximation is in tension with Lockery &
Goodman 2009's framing of C. elegans signaling as graded. However:
WB2's existing scaffolding uses this rule as the provisional default.
Threshold-crossing IS the LIF group's native idiom — same on_pre
pattern as LIF→LIF.

**Numerical stability.** Bounded if (a) refractory ≥ 5 ms is set,
(b) per-edge w magnitudes are typical of LIF→LIF (max ~5-10 in
connectome edges), and (c) `I_ext_post += W_syn_pA * w` (LIF→Wave2)
delivers current, not voltage — picks a per-spike charge of
`W_syn_pA * w * dt` (charge over a brief pulse rather than instant
voltage delta). With W_syn_pA chosen to deliver, e.g., 1-5 pA per
spike per unit weight, and Wave 2 cells responding via their own
`dv/dt = (-I_total + I_ext) / C`, the V trajectory is bounded by
the ratio (input rate × charge per spike) / leak. Calibrate
W_syn_pA so that Mellem-class injections (-30 to +30 pA total)
correspond to LIF input rates seen in baseline (10-50 Hz × 50-100
edges = 500-5000 spikes/s × ~20 fC per spike ≈ 10-100 pA — in
range). Calibration is empirical.

**Implementation simplicity.** Highest. Reuses Brian2's built-in
threshold and refractory machinery. No new state variables, no
(summed) variables for chemical synapses (only for gap junctions,
which are unchanged). Drop-in compatible with Brian2 SpikeMonitor
on Wave 2 cells, so spikes record naturally.

**Parameter requirements.**
- Threshold V_thr (one per Wave 2 cell type): default -25 mV.
- Refractory (one per Wave 2 cell type): default 5 ms.
- W_syn (mV) for Wave2→LIF: reuse LIFBrain default 0.8 mV.
- W_syn_pA (pA) for LIF→Wave2: NEW; calibrate ~1-5 pA per
  unit weight (per spike); tune so AVA receives Mellem-class
  drive at expected LIF rate.

**Honest assessment of biological cost.** The graded biology
(Lockery & Goodman 2009; Mellem 2008 AVA) is approximated as
threshold crossing. AVA "fires" a discrete event when V crosses
-25 mV — a category mismatch with Mellem's "no APs in AVA" finding.
Downstream LIF cells receive event-arrival semantics, which they
expect (their on_pre rule wants events), so the pipeline is
internally consistent. **The biological inaccuracy is one-sided:
the Wave 2 cell pretends to spike for the benefit of downstream LIF
cells that themselves spike.**

### Option B — Graded Boltzmann release (Wicks 1996 sigmoidal)

**Implementation summary.** Use a continuous (summed) variable
`sigma_pre` on each Wave 2 NeuronGroup (the σ output, Boltzmann of V).
Cross-group Synapses deliver continuous current proportional to
`σ_pre · w · W_graded_I` to the postsynaptic cell's I_ext (LIF) or
I_ext (Wave 2). LIF→Wave2 reverses: convert LIF spikes to a
post-synaptic conductance / current via the same continuous
`σ_pre`-style aggregation, with σ_pre derived from a low-pass-filtered
firing-rate proxy or from the LIF cell's V (since LIF V *is*
continuous between spikes).

```python
# Wave2 NeuronGroup eqs add:
sigma = 1 / (1 + exp(-(v - v_half_w2)/k_w2)) : 1
# v_half_w2, k_w2 per cell type.

# Wave2-to-LIF Synapses (cross-group):
syn_w2_to_lif = Synapses(
    wave2_groups[name], lif_neurons,
    model="""
    w : 1
    I_syn_w2lif_post = W_graded_I * w * sigma_pre : amp (summed)
    """,
    method="euler",
)
# LIF eqs grow: dv/dt = ... + I_syn_w2lif/C_mem + ...

# LIF-to-Wave2 Synapses (the reverse direction):
# Two viable patterns:
#  (B1) Use LIF's V directly (LIF is continuous between spikes):
syn_lif_to_w2_B1 = Synapses(
    lif_neurons, wave2_groups[name],
    model="""
    w : 1
    sigma_lif = 1 / (1 + exp(-(v_pre - v_half_lif)/k_lif)) : 1
    I_syn_lifw2_post = W_graded_I * w * sigma_lif : amp (summed)
    """,
)
#  (B2) Convert LIF spikes to a post-synaptic g(t) state on Wave 2:
syn_lif_to_w2_B2 = Synapses(
    lif_neurons, wave2_groups[name],
    model="""
    w : 1
    dg_syn/dt = -g_syn / tau_syn : siemens (clock-driven)
    I_syn_lifw2_post = g_syn * (E_rev - v_post) : amp (summed)
    """,
    on_pre="g_syn += W_g * w",
)
```

**Biological grounding.** Strong. Wicks 1996 is the canonical
reference; matches Davis & Stretton 1989 *Ascaris* data; aligns with
Lockery & Goodman 2009's graded-transmission framing. The
`graded_brain.py` reference implementation already uses this pattern
for an all-graded brain (line 183: `sigma = 1 / (1 + exp(-(v -
v_half)/k_half))`). Wave2HybridBrain effectively swaps in Wicks-style
release for the cross-group edges that cross between LIF and
Wave 2 cells.

**Numerical stability.** Strong. Continuous coupling delivers current
proportional to a bounded sigmoid σ ∈ [0, 1]. Maximum current per
edge is `W_graded_I * w * 1.0` (when σ saturates at 1). Across all
edges to a given Wave 2 cell, total current is bounded by `Σ w *
W_graded_I` (no spike summation). For AVA, with ~50-100 edges and
W_graded_I = 5 pA, max I ≈ 500 pA worst-case. Within physiological
range; calibrate to Mellem-class -30/+30 pA via W_graded_I.

**Implementation complexity.** Moderate. The Wave2-to-LIF direction
adds a per-NeuronGroup `sigma : 1` derived variable on Wave 2 cells
(one line in the eqs string). Cross-group Synapses use (summed) — a
standard idiom verified working in graded_brain.py. The LIF-to-Wave2
direction is the design choice point: B1 (continuous σ_lif from LIF
V) is simpler but unrealistic (LIF V is reset to v_reset on spike,
discarding the time-integral of release); B2 (g_syn(t) state per
synapse with on_pre kick) is more realistic but adds per-Synapses
state and an exp decay — moderate cython codegen cost.

**Parameter requirements.**
- Per Wave 2 cell type: V_half_w2, k_w2.
  AVA: V_half = -25 mV, k = 6-8 mV (Wicks-derived; consistent with
  Mellem's reported AVA rest range).
  AIY, RIM: **NO PRIMARY SOURCE.** Default to V_half = V_rest_cell,
  k = 6 mV (Wicks-style anchor); flag as extrapolation (Section 5
  Decision 3).
- W_graded_I (amp): single scale for all cross-edges; calibrate so
  Mellem-class -30/+30 pA at saturated σ.
- For B2 only: τ_syn (ms), W_g (nS).

**Honest assessment.** Option B inherits Wicks 1996's Ascaris
extrapolation. For AVA, the choice is reasonable (Mellem 2008's
direct AVA recordings agree on graded response and approximate rest
range). For AIY and RIM, the parameters are pure extrapolation —
acknowledge this in F20 documentation.

### Option C — Full conductance-based synaptic dynamics

**Implementation summary.** Each cross-group chemical edge has its
own kinetic state (g_syn(t) governed by τ_rise, τ_decay), driven by
release events from the pre-cell, with current = g_syn * (V_post -
E_rev). For Wave2→LIF and LIF→Wave2, the release event is either
threshold-crossing (Option A-style) or σ-driven (Option B-style),
but the current delivered to V_post obeys a kinetic shape function
matching molecular biology (e.g., NMDA-style slow rise-and-decay,
GABA-A-style fast).

```python
# Per cross-group Synapses:
syn_cross = Synapses(
    pre_group, post_group,
    model="""
    w : 1
    dg_syn/dt = -g_syn / tau_decay : siemens (clock-driven)
    I_syn_post = g_syn * (E_rev - v_post) : amp (summed)
    """,
    on_pre="g_syn += g_peak * w",    # event-driven (or sigma-modulated)
    method="euler",
)
```

For graded coupling (Wave2→LIF and Wave2→Wave2 with no discrete
event), replace `on_pre` with continuous σ-modulated current:

```python
syn_cross_graded = Synapses(
    pre_group, post_group,
    model="""
    w : 1
    dg_syn/dt = (g_max * sigma_pre - g_syn) / tau_decay : siemens (clock-driven)
    I_syn_post = g_syn * (E_rev - v_post) : amp (summed)
    """,
)
```

**Biological grounding.** Strongest. Matches molecular receptor
kinetics where data exists (e.g., GLR-1 activation kinetics from
*Mellem 2002* paired-cell recordings; UNC-49 GABA-A kinetics).
Captures the (V_post - E_rev) driving force, so postsynaptic V
limits its own depolarization — built-in inactivation of the
input.

**Numerical stability.** Strong. Driving force `(E_rev - V_post)`
goes to zero when V_post → E_rev, providing physical clamping.
Compatible with Wave 2 cell's small C because current is graded by
the postsynaptic V.

**Implementation complexity.** Highest. Adds 2 state variables per
Synapses (g_syn, possibly tau_kinetic), two parameters per receptor
type (E_rev, tau_decay), and per-Synapses initialization. Brian2
handles this idiom (see Brian2 `examples/synapses.continuous_interaction`)
under cython, but the per-Synapses state increases Synapses memory
footprint linearly. For 194 cross chem edges this is negligible.
For full Phase δ multi-cell integration (~600+ cross-group edges if
all Wave 2 cells active), still under 10K-state-vars per Synapses
— well within Brian2 capability.

**Parameter requirements.**
- Per receptor type: E_rev, tau_decay (and optionally tau_rise).
  - **GLR-1 / iGluR (excitatory glutamate)**: E_rev ≈ 0 mV, τ_decay
    ≈ 5-20 ms. Mellem 2002 reports glutamate-evoked currents in
    AVA/AVD with τ ≈ 30 ms decay (specific value from paired-cell
    paper, uncited here for verification).
  - **GABA-A (inhibitory)**: E_rev ≈ -70 mV, τ_decay ≈ 10-30 ms.
    UNC-49 kinetics from *Bamber 1999* (uncited here).
  - **Glutamate sign exceptions** for inhibitory glutamate (e.g.,
    AVR-15-mediated): E_rev ≈ -70 mV, τ_decay ≈ 30 ms.
- g_peak (or g_max): peak conductance per unit edge weight.
  Calibrate against Mellem-class drive.
- Plus: Wicks-style V_half / k for the sigma-modulated branches.

**Honest assessment.** Option C is the publication-grade approach. It
demands the most parameter sourcing — many of which require
acquiring papers we don't have direct quotes for in the WB3
pre-flight (Mellem 2002, Bamber 1999, etc.). Implementation is
larger but mechanical. The biggest cost is parameter-sourcing
discipline: each receptor-type / cell-type pairing needs a
literature line.

---

## Section 4 — Recommendation with rationale

**Pre-flight recommends Option B (graded Boltzmann release).**

### Rationale

1. **Biological grounding sufficient where it matters.** AVA's
   graded response is the only release-side biology actually
   measured for the cells WB3 directly handles (AVAL, AVAR via
   Mellem 2008). For these cells, Option B's σ-Boltzmann formulation
   is the canonical formalism and the field-standard reference
   (Wicks 1996, with its acknowledged Ascaris-extrapolation caveat).
   Lockery & Goodman 2009 reframes the C. elegans regenerative
   regime as graded-and-plateau — Option B's continuous σ(V) is
   the natural representation.

2. **Numerical stability via construction.** σ ∈ [0, 1] gives
   bounded per-edge current `W_graded_I · w · σ`. No threshold
   tuning, no refractory tuning, no per-spike calibration needed.

3. **Reuses existing Brian2 idioms.** `graded_brain.py` already
   implements this pattern (lines 183, 216, 229) under cython
   codegen, in production. Wave2HybridBrain's cross-group
   Synapses can use the same `(summed)` pattern with no new
   verification overhead.

4. **Capacitance-mismatch-immune.** Because current (not voltage)
   couples across, the ratio LIF/Wave2 ~ 10-95× becomes a feature
   (the Wave 2 cell's small C correctly amplifies a small input
   current into a larger voltage response, exactly as in
   biology) instead of a bug.

5. **Lower implementation cost than Option C.** No per-Synapses
   kinetic state variable; only one new derived variable on the
   Wave 2 NeuronGroup (`sigma : 1`) and one (summed) cross-group
   Synapses per direction.

6. **Phase G compatibility.** The downstream Phase G LIFBrain
   integration thread (Session 2) needs LIF→Wave2 coupling to
   feed into a body-output classifier. Option B's continuous
   coupling delivers a smooth V trajectory on Wave 2 cells →
   smoother readout.

### Tradeoffs accepted

- **AIY and RIM σ-parameters are extrapolations.** Use Wicks-style
  V_half = V_rest_cell + 5-10 mV, k = 6 mV. Document in F20 catalog
  entry.
- **No receptor-type fidelity.** Option B's W_graded_I is a single
  scale, not GLR-1-vs-UNC-49-specific. Inhibitory edges use sign
  bit but identical magnitude. Acceptable for Phase δ; refine in
  later phase if a falsification test demands it.
- **No explicit τ_decay for chemical synapses.** Option B is
  instantaneous σ-coupling, not a kinetic state. For LIF→Wave2,
  this means each LIF spike's effect on Wave 2 is felt only
  during the spike's brief V excursion above v_reset. In LIF
  terms, V is at v_thr for ~dt = 0.1 ms before reset to v_reset,
  so each spike's σ_lif is ≈ 1 for ~0.1 ms then drops back to
  baseline. **This is a known approximation issue** — adopt
  pattern B2 (per-Synapses g_syn(t) with τ_syn) for LIF→Wave2 to
  preserve postsynaptic effect duration realistically. See
  Section 5 Decision 2.

### What is NOT in scope

- Option C-level kinetic detail (NMDA τ_rise, etc.).
- Full receptor-type pharmacology (GLR-1 vs GLR-2 vs UNC-49 vs
  GAB-1 etc.).
- Plateau-generation kinetics on AVA itself (the cell already
  implements its 4-channel biophysical model; release rule is
  about *what AVA's V drives downstream*, not about how AVA
  computes its V).
- Bilateral pair completion (AIYR, RIMR) — handled in WB4-WB5.

### If pre-flight is wrong

If Rohit prefers Option A: defensible per "internal consistency
with LIF on_pre semantics; threshold-crossing on Wave 2 cells is
the WB2 ad hoc default; minimum departure from existing scaffold."
Cost: biological inaccuracy (AVA does not spike) and per-spike
calibration effort.

If Rohit prefers Option C: defensible per "publication-grade
cellular model with full kinetic detail." Cost: substantial
parameter-sourcing work (per-receptor τ, E_rev) and longer CP2.
Recommended only if Phase δ Layer B falsification (AVA plateau
under sustained ASH stim per Mellem 2008) demands kinetic detail
to reproduce the plateau-then-inactivation phenotype — which is
plausible since Mellem reports inactivation, but Mellem also
characterized ASH→AVA as graded with no plateau in AVA, so this
is unlikely to be load-bearing.

---

## Section 5 — Decisions requiring biological judgment

The following decisions cannot be derived from primary sources alone.
Each is a Rohit-call. For each: choice space, consequences, and
default if no strong preference.

### Decision 1 — Release rule choice (the meta-decision)

**Choice space:** A (V-threshold), B (graded Boltzmann), or C (full
conductance-based).

**Consequence:** Determines CP2-CP6 implementation arc. Option A
fastest; Option C most rigorous; Option B balances.

**Default if no preference:** **Option B** (per Section 4).

### Decision 2 — LIF→Wave2 coupling sub-pattern

If Option B chosen: which sub-pattern?

**Choice space:**
- B1: continuous σ_lif from LIF's instantaneous V (LIF V is
  continuous between spikes, V_post = v_pre).
- B2: per-Synapses g_syn(t) state, exp-decay with τ_syn, kicked
  by on_pre on each LIF spike: `g_syn += W_g * w`.

**Consequence:**
- B1 is simpler (no per-Synapses state). However, LIF V in
  resting LIF cells is at v_rest = -25 mV typically, putting σ_lif
  near baseline (similar to AVA's rest). LIF V briefly hits v_thr
  = -10 mV at spike time (0.1 ms), then resets to v_reset = -30 mV
  (5 ms refractory), then drifts back to v_rest. So σ_lif is
  *quasi-tonic* at ~σ(V_rest) ≈ 0.5 with brief excursions during
  spikes. **This means LIF→Wave2 input is approximately constant**,
  modulated by the LIF cell's overall V trajectory (rest level vs
  spike rate), which actually does encode firing rate — but
  conflates rate and resting V.
- B2 adds per-Synapses g_syn state; each LIF spike kicks g_syn
  upward by W_g * w; g_syn decays with τ_syn (e.g., 10 ms);
  current to post = g_syn * (E_rev - v_post). This is the
  biologically faithful pattern. Slight cython codegen overhead
  (one new state per cross-edge ≈ 200 vars total — negligible).

**Default if no preference:** **B2** (per-Synapses g_syn(t),
τ_syn = 10 ms). Negligible overhead, biologically faithful,
matches the Brian2 `synapses.continuous_interaction` example.

### Decision 3 — AIY and RIM sigmoidal parameters

**Choice space:**
- (a) Apply Wicks-1996 Ascaris-derived defaults universally: V_half
  = V_rest_cell + 5 mV, k = 6 mV, W_graded_I = 5 pA. Document as
  extrapolation.
- (b) Defer AIY/RIM cross-coupling to Phase δ post-WB3
  (WB4/WB5 deliverables); only AVAL/AVAR cross-couple in WB3.
- (c) Tune AIY/RIM V_half against their cellular V_rest from the
  Wave 2 cell-builder validation results (`AIY_validation_results`,
  `RIM_validation_results`); treat as cellular-anchored defaults
  with explicit "no primary-source validation" caveat.

**Consequence:**
- (a) cleanest; AIY/RIM behavior may be off in nonspecific ways.
- (b) extends Phase δ timeline; WB3 is no longer load-bearing
  for Phase G.
- (c) intermediate; fewer biological commitments than (a) but
  requires per-cell tuning step in CP2.

**Default if no preference:** **(c) for AIY/RIM with cellular-anchored
V_half**, paired with default (a)-style k = 6 mV. Document explicitly
in F20.

### Decision 4 — W_graded_I calibration

**Choice space:**
- (i) Match `graded_brain.py`'s value: W_graded_I = 5 pA.
- (ii) Calibrate against Mellem 2008 -30/+30 pA injection range:
  pick W_graded_I such that summed `Σ w * σ` over a typical AVA's
  pre-edges, at saturated σ = 1, equals ±30 pA.
  Worked example: AVA has ~50 chemical pre-edges with mean |w| ~
  2-3 (per `lif_brain.py` connectome stats). Σ w ≈ 100-150. To hit
  30 pA at σ = 1: W_graded_I = 30 pA / 100 = 0.3 pA per unit weight.
- (iii) Both (i) and (ii) are too far apart? Pick mid-range: 1 pA.

**Consequence:**
- (i) matches the existing graded brain reference; cross-comparison
  cleaner. But Wave 2 cells have different downstream Synapses
  patterns; this isn't directly portable.
- (ii) is calibrated against the only direct AVA-V experiment in
  the literature.
- (iii) hedge.

**Default if no preference:** **(ii), W_graded_I = 0.3 pA** —
empirically calibrated against Mellem 2008 injection range. Confirm
via spontaneous-mode 1 s smoke test in CP3.

### Decision 5 — Sigma definition for LIF cells (if B1) or g_syn / E_rev (if B2)

**If B1:** what V_half and k for LIF cells?
- Reuse Wicks-Ascaris: V_half = LIF v_rest = -25 mV, k = 6 mV.
- Default if no preference: **as above** (V_half = -25 mV, k = 6 mV).

**If B2:** what E_rev (excitatory and inhibitory) and τ_syn?
- E_rev_exc = 0 mV (vertebrate iGluR convention); E_rev_inh = -70 mV
  (vertebrate GABA convention). C. elegans-direct values not
  measured for AVA/AIY/RIM.
- τ_syn: 10 ms canonical.
- Default if no preference: **E_rev_exc = 0 mV, E_rev_inh = -70
  mV, τ_syn = 10 ms.**

### Decision 6 — Numerical-stability safety net

**Choice space:**
- (i) No cap. Trust the rule.
- (ii) Soft cap: log when |I_total per Wave 2| > 100 pA, but don't
  truncate.
- (iii) Hard cap (current WB2 default): clip at ±20 pA. Mask input.

**Consequence:**
- (i) cleanest if the rule is correct. Risk: if a calibration is
  off, the simulator silently produces non-physiological V.
- (ii) reveals miscalibrations without masking them.
- (iii) is the WB2 status quo; physically incorrect but tells you
  nothing about whether the rule is wrong.

**Default if no preference:** **(ii) — soft cap with log warning at
±100 pA**. CP3 numerical-stability validation should drive the
decision: if 1 s spontaneous smoke test shows currents bounded
naturally, downgrade to (i); if not, keep (ii) and investigate.

### Decision 7 — Spike emission from Wave 2 cells (downstream readout)

Independent of A/B/C choice for the *coupling* rule, the
ClosedLoopEnv readout layer (FSM activity-mode classifier) consumes
firing-rate vectors. Wave 2 cells under Option B are graded — they
don't emit discrete spikes natively. Two routes for the readout:
- (a) Emit pseudo-spikes via threshold crossing on σ (same as
  `graded_brain.py` line 269: `_poll_sigma`). When σ crosses 0.5
  rising, register a spike event for the cell.
- (b) Read σ continuously via `firing_rates(window_ms)` returning
  σ * (some_rate_max) for Wave 2 cells; FSM classifier needs to
  handle this calibration.

**Default if no preference:** **(a) σ-threshold-crossing** (matches
`graded_brain.py`'s production approach, preserves
`firing_rates()` API).

---

## Summary table — recommendation

| Decision | Pre-flight default | Rohit confirms? |
|----------|--------------------|-----------------|
| 1. Release rule | Option B (graded Boltzmann) |  |
| 2. LIF→Wave2 sub-pattern | B2 (g_syn(t), τ_syn = 10 ms) |  |
| 3. AIY/RIM sigmoidal params | (c) cellular-anchored V_half, k = 6 mV |  |
| 4. W_graded_I | 0.3 pA (Mellem-calibrated) |  |
| 5. E_rev / τ_syn | E_exc = 0 mV, E_inh = -70 mV, τ_syn = 10 ms |  |
| 6. Stability safety net | (ii) soft cap at ±100 pA + log |  |
| 7. Readout pseudo-spikes | (a) σ > 0.5 rising threshold |  |

---

## What CP2-CP6 will do (after Rohit's authorization)

CP2: implement chosen rule in `wave2_hybrid_brain.py`. Add cross-group
Synapses, replace `_step_route` Python callback with native Brian2
(summed) coupling. Per-NeuronGroup `clock` keyword for dt mismatch
(Wave 2 0.025 ms, LIF 0.1 ms).

CP3: 1s / 10s / 30s numerical-stability smoke tests.

CP4: 30s touch_anterior cascade validation. Compare to per-edge LIF
baseline AVA Δ+7.5 Hz cascade.

CP5: F20 catalog entry in `wave2/translation_patterns.md`. Update
`phase_delta_wb3_findings.md` with implementation details.

CP6: commit + push grouped (A: this options doc; B: implementation;
C: validation; D: findings).

---

**End of CP1 deliverable. PAUSED FOR ROHIT REVIEW. See
`PAUSED_FOR_REVIEW.txt`.**
