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
