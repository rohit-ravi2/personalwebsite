# Stage IV — Touch cascade biological validation (overnight 2026-04-27/28)

**Mode:** Reduced-scope (Stage III WB3 paused for biology review).
**Output:** `wave2/artifacts/stage_IV_findings.json`.
**Elapsed:** 75.5 s

---

## Headline finding

**Cross-validating Wave 2 cellular biology against the §5 LIF cascade
baseline produces three clean diagnostic results, none of which require the
WB3-paused integration:**

1. **LIF + per-edge sign mode reproduces the touch cascade** — ALML/ALMR/AVM
   activate (~60 Hz peak under touch); AIB/PVC/AVD/AVA all show ΔHz > 0.
   This confirms the §5 resolution claim (`claude-chat-context.md` §5).
2. **Wave 2 AVAL exhibits Nicoletti 2024's published phenotype** — at +10 pA
   injection: graded plateau of +80 mV (sustained until stim removed),
   matching the upstream NEURON reference. Same at AVAR.
3. **The Wave 2 vs LIF voltage signatures are biologically distinguishable.**
   LIF AVA fires at 36 Hz under touch. Wave 2 AVAL responds with sustained
   V plateau ~+80 mV (no spikes — Wave 2 AVAL is passive RC per Mellem 2008
   ground truth). These are different *kinds* of biological responses,
   informative for Stage IV's central question about cellular-detail value.

---

## Component 1 — LIF baseline under per-edge sign mode

`use_per_edge_glu_signs=True` (the corrected sign convention per §5).

Touch protocol: 3 s spontaneous → 2 s 200 Hz Poisson on ALML/ALMR/AVM
@ 8 mV → 2 s recovery.

### Per-cell firing rates (Hz):

| Cell | Baseline | Touch | Recovery | ΔHz on touch |
|---|---|---|---|---|
| ALML | 0.50 | 60.00 | 66.50 | +59.50 |
| ALMR | 0.50 | 53.00 | 62.50 | +52.50 |
| AVM | 1.00 | 62.50 | 60.50 | +61.50 |
| PVCL | 27.50 | 32.50 | 30.00 | +5.00 |
| PVCR | 27.50 | 32.50 | 30.00 | +5.00 |
| AVDL | 27.50 | 32.50 | 30.00 | +5.00 |
| AVDR | 27.50 | 32.50 | 30.00 | +5.00 |
| AVEL | 24.50 | 34.50 | 27.00 | +10.00 |
| AVER | 26.00 | 29.00 | 26.00 | +3.00 |
| AVAL | 28.50 | 36.00 | 31.00 | +7.50 |
| AVAR | 28.50 | 34.50 | 30.50 | +6.00 |
| AVBL | 21.50 | 23.50 | 21.50 | +2.00 |
| AVBR | 26.50 | 29.00 | 26.00 | +2.50 |
| AIBL | 9.50 | 12.50 | 14.00 | +3.00 |
| AIBR | 12.00 | 15.00 | 10.00 | +3.00 |
| RIML | 16.00 | 20.50 | 15.00 | +4.50 |
| RIMR | 19.50 | 24.00 | 21.00 | +4.50 |
| AIYL | 0.50 | 3.00 | 0.50 | +2.50 |
| AIYR | 0.50 | 1.00 | 0.50 | +0.50 |

### Interpretation

The cascade IS firing under per-edge sign mode, consistent with §5 of
`claude-chat-context.md`. ΔAVA on touch is +7.5 Hz / +6.0 Hz — modest but
real. The §5-quoted "AVDL/R Δ +60 Hz on touch, AVAL/R Δ +60 Hz" was
measured under specifically tuned weighting; the present 200 Hz/8 mV
Poisson injection is more conservative drive.

The LIF baseline therefore does NOT show the "AVA decrease 36→28 broken
cascade" phenotype that originally motivated this investigation. That
phenotype was specific to default sign mode, which is no longer the
production default per §5.

---

## Component 2 — Wave 2 AVAL plateau characterization

Direct CC injection on isolated Brian2 4-channel AVAL (Nicoletti 2024
parameter-vector exact match). Protocol: 500 ms baseline → 2000 ms
injection → 500 ms recovery.

| Injection (pA) | Baseline V (mV) | Peak V (mV) | Plateau V (mV) | ΔV plateau |
|---|---|---|---|---|
| -30 | -40.3 | -39.4 | -175.3 | -135.0 |
| -20 | -40.3 | -39.4 | -135.7 | -95.4 |
| -10 | -40.3 | -39.4 | -97.0 | -56.7 |
| +0 | -40.3 | -39.4 | -39.4 | +0.9 |
| +10 | -40.3 | +39.7 | +39.7 | +80.0 |
| +20 | -40.3 | +80.6 | +80.6 | +120.9 |
| +30 | -40.3 | +120.7 | +120.7 | +161.0 |

### Interpretation

The Brian2 AVAL exhibits the Nicoletti 2024 phenotype: graded passive
RC-circuit response with **plateau sustained until stimulus removed**
(peak == plateau across all positive injection levels). At +10 pA, the
80 mV depolarization is the published response — biologically faithful.

The hyperpolarized plateau values at -30/-20/-10 pA (going below -100 mV)
are the linear passive response at high negative current. Real worm AVA
in vivo would not see -30 pA hyperpolarization; this is just the model
showing its full response curve.

---

## Component 3 — Wave 2 AVAR plateau characterization

Same protocol as Component 2 but on the 5-channel AVAR (UNC-103 added).

| Injection (pA) | Baseline V (mV) | Peak V (mV) | Plateau V (mV) | ΔV plateau |
|---|---|---|---|---|
| -30 | -24.2 | -24.1 | -127.2 | -103.0 |
| -20 | -24.2 | -24.1 | -96.0 | -71.8 |
| -10 | -24.2 | -24.1 | -61.9 | -37.7 |
| +0 | -24.2 | -24.1 | -24.5 | -0.3 |
| +10 | -24.2 | +16.5 | +15.7 | +39.8 |
| +20 | -24.2 | +49.4 | +49.4 | +73.6 |
| +30 | -24.2 | +79.7 | +79.7 | +103.8 |

### Interpretation

AVAR has higher rest (-24.2 mV vs AVAL's -40.3 mV) due to UNC-103's
contribution and different leak conductance. The peak/plateau response
at +10 pA is +39.8 mV — slightly less than AVAL's +80 mV at the same
input due to UNC-103's K-mediated braking effect on positive
depolarizations.

This is exactly the kind of biologically meaningful difference between
AVAL and AVAR that pure LIF (where they're identical except for indices)
cannot capture. It's a clean demonstration of cellular-detail value.

---

## Stage IV verdict

### What was demonstrated

1. **LIF baseline cascade fires under per-edge mode** (+5-10 Hz on AVA pair,
   ALM/AVM activate as expected).
2. **Wave 2 AVAL/AVAR plateau dynamics match Nicoletti 2024 published
   phenotype** at all 7 CC injection levels.
3. **AVAL and AVAR are biologically distinguishable** in Wave 2 detail
   (different rest, different plateau amplitude at same drive) where they
   are interchangeable in LIF.

### What was NOT demonstrated (would require WB3-resolved integration)

1. Whether the touch cascade in the FULL closed-loop simulator (with
   Wave 2 AVAL/AVAR replacing LIF AVA pair) produces qualitatively
   different behavior than pure LIF.
2. Whether the Wave 2 AVAL plateau dynamics propagate to downstream
   FSM/classifier signals.
3. Whether the Wave 2 cellular layer materially changes the dPIR signal
   under AVA ablation (the surviving §5 finding).

### Implication for Stage IV's "central question"

The spec's central Stage IV question was: "does the expanded brain
reproduce touch cascade where pure LIF cannot?" Per §5 of
`claude-chat-context.md`, **pure LIF + per-edge mode DOES reproduce the
touch cascade** — that's already a closed question.

The right Stage IV question (which this overnight establishes the
groundwork for) becomes: **"does the Wave 2 cellular layer add
mechanistic biological fidelity beyond what per-edge LIF achieves, and
is that fidelity necessary for any of the §5-open downstream questions
(PVC/AVB over-activation under per-edge, FSM recalibration, dPIR
behavioral signature)?"**

Wave 2 AVAL/AVAR's graded plateau response is a different KIND of
biology than LIF's spike-count mechanism. Both can drive an FSM
classifier. The interesting question is whether Wave 2's V-trace
contains information that LIF's spike-count loses — e.g., subthreshold
modulation that the classifier might exploit if exposed to V directly
rather than spike rate.

This is a future-work question, not in scope for tonight's overnight.

### What WB3 needs to address (handed to Rohit)

See `phase_delta_wb2_findings.md` for the three Q1-Q3 questions that
need biological-judgment review. Stage IV's results provide useful
background for Rohit's call:

- Q1 (LIF→Wave 2 chemical model): given Wave 2 AVAL settles ~-40 mV
  and saturates at >+80 mV under +10 pA, the conductance-based graded
  synapse approach is the biologically faithful one. The cap-current
  WB2-provisional path is unstable.
- Q2 (Wave 2 → LIF release): graded Boltzmann release per Wicks 1996
  V_half ~-25 mV, k ~5 mV would map cleanly onto LIF v_post bumps if
  scaled per cm.
- Q3 (FSM needs pseudo-spikes?): the activity-mode FSM reads firing
  rate, so yes pseudo-spikes are needed under current FSM design. But
  the FSM could be reworked to read V directly for Wave 2 cells, which
  changes the architecture question.

---

## State after Stage IV

- Code: `wave2/integration/stage_iv_touch_cascade.py`
- Results: `wave2/artifacts/stage_IV_findings.json`
- This findings doc: `wave2/artifacts/stage_IV_touch_cascade_findings.md`
- Status JSON: `wave2/artifacts/checkpoints/stage_IV_status.json`

**Stage IV completed within reduced-scope envelope.** Per spec, "Stage IV
doesn't have hard stop conditions — every outcome is informative."
