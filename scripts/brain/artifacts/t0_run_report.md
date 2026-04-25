# Tier 0 run report — P0+P1 end-to-end validation (2026-04-21)

Ran all 6 scenarios (classifier default + one activity-mode variant)
on v3 LIF brain after landing the P0/P1 commits. Goal: convert
"committed code" into "measured behaviour" and identify any
bugs blocking real-site deployment.

## What ran

| scenario | duration | wall time | size | state distribution |
|---|---|---|---|---|
| spontaneous | 30 s | 79 s | 898 KB | FWD 4%, REV 76%, PIR 20% |
| touch | 30 s | 90 s | 890 KB | FWD 7%, REV 48%, PIR 30%, QUI 15% |
| osmotic_shock | 30 s | 91 s | 987 KB | FWD 3%, REV 87%, PIR 10% |
| food | 30 s | 107 s | 939 KB | FWD 3%, REV 87%, PIR 10% |
| chemotaxis | 60 s | 213 s | 1919 KB | FWD 1%, REV 93%, PIR 5% |
| aerotaxis | 60 s | 211 s | 1979 KB | FWD 2%, REV 88%, PIR 10% |

File sizes grew ~40-50% from v3.3 era because P0 #1 added full-
network raster. Acceptable on Vercel with gzip.

## Bugs found and fixed

**T0-bug-1 (aerotaxis, fixed):** `Environment.inject_into_brain()`
had an unguarded `peak / sigma` division. Aerotaxis scenarios use a
dummy chemotaxis gradient (peak=0), which hit `ZeroDivisionError`.
Patched with a `peak > 1e-6 and sigma > 1e-4` guard before the
chemotaxis injection path. Aerotaxis now runs cleanly.

**T0-bug-2 (activity-FSM baseline, fixed):** My original
BASELINE_TAU_S=4 s + WARMUP_S=0 caused the EMA baseline to overshoot
during initial convergence (prior = 2 Hz, actual tonic = 30 Hz →
z-score huge for first ~2 seconds of simulation, triggering spurious
REVERSE/OMEGA transitions at t=0-1s). Fixes:
- BASELINE_TAU_S: 4 → 20 s (prevents stim-driven excursions from
  washing into the baseline)
- WARMUP_S: 0 → 2 (no transitions allowed during initial baseline
  estimation; during warmup, baseline uses a fast τ=0.5 s EMA to
  converge quickly from prior)
- ROLE_Z_THRESHOLD bumped up across the board because the v3
  brain's tonic firing is higher than I originally assumed; z=1.5
  was too permissive and fired spuriously from natural rate noise

## Circuit-level finding (v3 LIF brain)

**Touch does NOT propagate to AVA in the current v3 LIF network.**

Profiling confirmed by computing per-neuron firing rate
differentials between pre-touch (1–5 s) and peri-touch (5–7 s)
windows in a classifier-mode run with a t=5 s `touch_anterior` stim:

Sensory side (as expected):
- ALM/AVM: 1.7 Hz → 78.2 Hz  (+76.5, clean touch response)

Command interneurons (**not** as expected):
- AVEL: 29.2 Hz → 26.0 Hz (−3.2) — *decreased* on touch
- AVER: 36.0 Hz → 28.5 Hz (−7.5) — *decreased* on touch
- AIBL: 7.8 Hz → 7.5 Hz (−0.3) — flat

Top-responder set is dominated by head motor neurons
(SIBVL/RIVR/SMDVL/RMDR) rising 2-3 Hz, not the expected
command-interneuron reversal cascade.

**Interpretation:** The previously-documented ΔREV = −0.57 ± 0.37
"AVA ablation abolishes reversal" reproduction (v3.0 audit) runs
through the classifier's multi-neuron correlation pattern, **not**
via biologically-correct AVA plateau drive. The classifier has
learned to infer reversal from whatever firing pattern is
correlated with reversal labels in Atanas training, which isn't
necessarily the anatomical circuit. My P1 #4 ActivityFSM reads
AVA directly and therefore cannot detect touch-driven reversals
on this brain.

Activity-mode touch run empirically confirms: FORWARD 9%,
QUIESCENT 91%. The brain never fires hard enough through the
ActivityFSM's literature-role neurons to trigger REVERSE. Instead,
residual NSM firing noise crosses the (already-raised-to-3.0)
quiescent threshold and the FSM locks in QUIESCENT.

## What this means for the project

**Good news:**
- All 6 scenarios ship with 300-neuron full raster + validated-18
  tag. P0 #1 is live on the site after this commit.
- Aerotaxis scenario loads cleanly. P0 #3 is live.
- CeNGEN expression ring JSON built; needs visual QA.
- No regressions in classifier-mode scenario output.

**The honest headline:**
- ActivityFSM is architecturally correct but ships unusable on
  the current v3 LIF brain, because v3 LIF's synaptic tuning
  doesn't reproduce the touch → AVA cascade.
- This is a brain-calibration issue (tuning the chemical+gap
  synapse weights so ALM → AIB → AVA produces a real reversal
  burst), not an FSM-design issue.

**Publishable methodological finding:**

> *"Connectome-constrained LIF simulators that reproduce Atanas
> perturbation phenotypes via trained classifier readouts do so
> through distributed pattern recognition rather than
> biologically-correct command-neuron cascades. Directly reading
> command-neuron activity (as in our ActivityFSM) exposes this gap
> and can serve as a falsification test for whether the simulator
> has captured circuit-level dynamics vs. only readout-level
> statistics."*

This is the kind of result that belongs in the methods section of
the eLife/PLOS CB paper. The simulator's perturbation-phenotype
reproduction is real but mediated by a trained classifier; the
next-generation goal is to have the circuit itself produce the
right dynamics.

## Next steps

**Tier 0 successes to ship:**
1. Commit the 6 regenerated JSONs so rohitravi.com serves the
   upgraded format.
2. Run visual QA in browser — P0 #2 CeNGEN ring, P0 #3 arena O2
   gradient, P1 #4 activity-FSM pill, P1 #6 diffusion field on
   modulator hover, P1 #6 synthetic Ca trace.

**Not ready to ship yet:**
3. ActivityFSM remains opt-in via env var. Don't default it.
4. Activity-FSM ensemble audit (T0c) is **not useful on v3 LIF**
   — every ablation would just produce QUIESCENT=91%, zero signal.
   Defer until the brain is recalibrated OR we validate on graded
   brain (T1a).

**To unblock ActivityFSM productively (v3.5 work):**
5. Synaptic weight calibration pass: tune W_syn so ALM→AIB→AVA
   actually produces an AVA burst when ALM fires at 78 Hz. Target:
   AVA baseline 2-5 Hz, AVA-during-touch ≥ 20 Hz. This is a 1-2
   week focused fit.
6. Alternative: run the T1a graded brain (biologically more
   correct σ(V) continuous release) with ActivityFSM, to test
   whether graded dynamics produce the right cascade even where
   LIF does not. If yes, ActivityFSM + graded is the v3.6 default.

## File artefacts

New JSONs written to `public/data/`:
- wormbody-brain-spontaneous.json (898 KB)
- wormbody-brain-touch.json (890 KB)
- wormbody-brain-osmotic_shock.json (987 KB)
- wormbody-brain-food.json (939 KB)
- wormbody-brain-chemotaxis.json (1919 KB)
- wormbody-brain-aerotaxis.json (1979 KB)
- wormbody-brain-touch-activity.json (886 KB) — demo only

All include P0 #1 full_raster + validated_readout_set metadata.

---

## T0 Resolution Update — 2026-04-25

This postscript supersedes the framing in the original April 21
report above. The original content is preserved as a historical
record of what was understood at that point. For the canonical
record of how T0 was resolved, see `docs/t0_resolution_report.md`.

### What the original framing got wrong

The April 21 report framed the T0 cascade-failure as a synaptic
weight calibration problem, with the proposed fix being a 1-2 week
focused tuning of W_syn so ALM → AIB → AVA produces an AVA burst
(see "Next steps" #5 above). Two things in that framing are now
known to be wrong:

1. **Wrong cascade.** ALM and AVM have zero direct chemical
   synapses to AIB in this connectome (Cook 2019 hermaphrodite +
   Loer & Rand 2022 NT identity). AIB also has zero chemical edges
   to AVD. The "ALM → AIB → AVA" pathway never existed in the
   simulator's wiring; it was a textbook description that didn't
   match what the connectome encodes. AIB is in the chemotaxis
   pirouette circuit, not the touch reversal circuit.

2. **Wrong fix category.** The cascade failure was not a synaptic
   weight calibration problem. It was a sign-assignment problem.
   The simulator's default glutamate-sign convention treated
   glutamate edges to iGluR-dominant postsynaptic neurons (AVA,
   AVD, AVE, PVC) as inhibitory because the per-presynaptic-neuron
   sign field defaults to −1 for Glu. The connectome already
   contained a precomputed alternative (`W_chem_per_edge`) using
   CeNGEN-derived postsynaptic-receptor signs, but it was
   off-by-default behind a constructor flag.

### What the actual cascade is

The operative touch-reversal cascade in this connectome is:

> **ALM/AVM → PVC → AVD/AVE → AVA**

PVC is the load-bearing first-stage relay. ALM/AVM glutamate
inputs to PVC sign as inhibitory under default convention (PVC
drops on touch, removing its dominant cholinergic excitation of
AVD, which is why AVD also drops on touch — the original
"command interneurons decreased" finding above). Under per-edge
sign convention (PVC iGluR-dominant per CeNGEN, ratio 9.6×), the
same glutamate edges sign excitatory and PVC fires UP on touch.
PVC then drives AVD/AVE cholinergically (~5× more drive to AVD
than direct ALM/AVM glutamate to AVD); AVD/AVE drive AVA
cholinergically; recurrent positive-feedback within the command
pair amplifies all command neurons to ~97 Hz coherently.

### What today's diagnostic block established

- **Voltage regime is not the bottleneck** (FALSIFIED). Patching
  v_rest from −65 to −25 mV per Mellem 2008 was a coordinate
  translation; LIF dynamics unchanged because rest-to-threshold
  gap was preserved. Voltage fix kept in place for biological
  documentation.
- **Gap conductance is not the bottleneck** (FALSIFIED).
  Increasing g_gap monotonically silenced the network because
  gap junctions average noise across 2188 coupled cells.
- **Per-edge sign mode resolves the cascade firing**
  (CONFIRMED). Under `use_per_edge_glu_signs=True`, AVDL/R fire
  Δ +60 Hz on touch, AVAL/R Δ +60 Hz, AVEL/R Δ +47 Hz, n=10 with
  seed-to-seed variance under 1.5 Hz.
- **The original ΔREV reproduction was a sign-convention
  artifact** (CONFIRMED). Under per-edge mode the ΔREV regresses
  to +0.04 (2/10 negative seeds). The Chalfie 1985 direction was
  being reproduced via Mode 3 tonic-shift on AVA's broken-sign
  baseline rate, not via cascade firing.
- **Behavioral effect persists in dPIR channel** (NEW FINDING).
  Under per-edge mode, AVA-ablation produces a clean dPIR effect
  (mean −0.117, 9/10 negative seeds). The simulator's circuit-
  level response to AVA loss is preserved; the FSM/classifier
  was calibrated to read it through dREV under default-mode
  dynamics.

### What this means for "Next steps" in the original report

Items 5 and 6 of the original report's "Next steps" are now
revised:

- **Item 5 (synaptic weight calibration as ActivityFSM unblock)
  — obsolete in its original form.** The cascade fires at +60 Hz
  under per-edge mode without weight tuning. Specific weights
  (notably ALM/AVM → PVC) may still need fine-tuning if PVC
  over-activation under per-edge is judged a real bug; that's a
  different, narrower question.
- **Item 6 (T1a graded brain alternative) — possibly still
  relevant.** The same sign-convention question may exist in the
  graded brain. Per-edge sign mode is glutamate-receptor-specific
  and applies to LIF directly; whether GradedBrain has analogous
  per-edge handling needs separate verification.

### What's open after the resolution

1. **PVC/AVB over-activation under per-edge mode.** PVC fires
   Δ +60-70 Hz and AVB fires Δ +51-57 Hz on touch — biologically
   questionable since canonical biology has anterior touch
   suppressing forward locomotion. Two interpretations remain
   open: (A) CeNGEN expression-vs-function mismatch at specific
   synapses, or (B) canonical biology more nuanced than textbook.
   Neither yet falsified.
2. **FSM/classifier recalibration question.** Refined by dPIR
   finding: not just "retrain to recover dREV" but characterize
   what behavioral signature AVA-ablation produces under correct
   cascade dynamics, and which FSM channels best reflect the
   circuit-level Chalfie phenotype.
3. **RIS silencing under per-edge.** RIS goes from 21.8 Hz tonic
   (default) to 0.8 Hz (per-edge) — a network-equilibrium effect,
   not a direct sign flip. RIS molecular audit findings from
   earlier in the project don't transfer to per-edge mode without
   re-running.

### File references

Full details, sweep CSVs, verification scripts:
- `docs/t0_resolution_report.md` (canonical record of resolution)
- `docs/current-state-summary.md` (updated state summary)
- `scripts/brain/artifacts/phase0_postvolt_*.csv` (sweep data)
- `scripts/brain/phase0_postvolt_compare.py` (comparison harness)
- `scripts/brain/phase0_avd_drive_decomp.py` (drive decomposition)
