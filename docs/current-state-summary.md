# Current state summary

*Last updated: 2026-04-25 (T0 resolution work block).*
*Supersedes: 2026-04-21 version (Phase 0 close-out).*

Narrative layer on top of `scripts/brain/artifacts/phase0_baseline_report.md`
and `docs/t0_resolution_report.md`. Updated at phase boundaries and
when load-bearing findings change.

## What changed today (2026-04-25)

The T0 cascade-failure question was resolved at the architectural
level. Two suspects were falsified (voltage regime, gap conductance);
the actual mechanism turned out to be the simulator's default sign
convention. Full diagnostic record in
`docs/t0_resolution_report.md`. Headline findings:

- **T0 mechanism settled.** The v3 LIF brain's failure to propagate
  touch through to AVA was caused by the per-neuron NT-sign
  convention treating glutamate→iGluR-dominant edges as inhibitory.
  Switching to per-edge CeNGEN-derived signs (constructor flag
  `use_per_edge_glu_signs=True`, already in the codebase but off by
  default) makes the cascade fire: AVDL/R Δ +60 Hz on touch, AVAL/R
  Δ +60 Hz, AVEL/R Δ +47 Hz, n=10 with seed-to-seed variance under
  1.5 Hz.
- **Operative cascade is ALM/AVM → PVC → AVD/AVE → AVA**, not the
  long-assumed ALM → AIB → AVA. AIB has zero chemical edges to
  AVD in either sign mode and is not in the touch-reversal
  pathway in this connectome. PVC is the load-bearing first-stage
  relay (ALM/AVM glutamate → PVC, flips excitatory under per-edge
  because PVC is iGluR-dominant); PVC then drives AVD/AVE
  cholinergically (~5× more drive to AVD than direct ALM/AVM
  glutamate inputs); AVD/AVE drive AVA cholinergically; recurrent
  positive-feedback in the command pair amplifies all command
  neurons to ~97 Hz coherently. Replace "ALM → AIB → AVA" framing
  throughout any remaining docs.
- **The April 21 phenotype reproduction was a sign-convention
  artifact on dREV.** AVA-ablation under default mode produced
  ΔREV = −0.49 (10/10 negative seeds at n=10 × 60s). Under per-edge
  mode (with cascade firing correctly) ΔREV regresses to +0.04
  (2/10 negative) — but **dPIR retains a clean AVA-ablation effect
  under per-edge** (mean −0.117, 9/10 negative seeds). The Chalfie
  1985 direction was being reproduced via a Mode 3 tonic-shift
  mechanism on the broken sign convention through the dREV
  channel; under correct cascade dynamics the AVA-ablation effect
  shifts FSM channels rather than vanishing. Audit-quality
  improvement (10/10 at n=10) was real but was measuring a
  wrong-mechanism reproduction in a single channel.
- **Per-edge mode introduces a new open question.** PVC fires up
  Δ +60-70 Hz and AVB fires up Δ +51-57 Hz on touch under per-edge
  — biologically questionable since canonical biology has anterior
  touch suppressing forward locomotion. Mechanism: PVC is iGluR-
  dominant (CeNGEN ratio 9.6×), so ALM glutamate flips to
  excitatory under per-edge. Two interpretations both consistent
  with data: CeNGEN tells receptor presence not functional
  dominance, OR canonical biology is more nuanced than the textbook
  story. Not yet decided.
- **GABA + peptide release are structurally clean.** GABA uniformly
  signed −1 across all 26 GABA neurons via per-presynaptic-neuron
  sign field; per-edge mechanism is glutamate-specific by design;
  135 GABA edges byte-identical across modes. Peptide release is
  pure linear rate-coupling (release = releaser_weights @
  spike_counts, capped at 10). Both verified by direct measurement.
- **RIS silenced under per-edge mode.** RIS goes from 21.8 Hz tonic
  (default) to 0.8 Hz (per-edge). Not a direct sign-flip effect
  (RIS is GluCl-dominant, sign unchanged) — a network-equilibrium
  consequence. Implications for the RIS molecular audit and any
  RIS/sleep work; needs re-running under per-edge.
- **Voltage fix in place but a no-op for LIF dynamics.** Patched
  `v_rest=−25, v_thr=−10, v_reset=−30` per Mellem 2008 (PMID
  18587393). Preserves 15 mV rest-to-threshold gap so LIF dynamics
  are coordinate-translated unchanged. Kept for biological
  documentation; will matter when SK/BK and compartmental work
  starts.

## Simulator execution profile

Brian2 2.9.0 on CPU (numpy codegen target). Measured wall/simulated
ratio on the shipped v3 LIF brain: ~1.66× under post-volt config (was
3.06× in earlier sessions; difference likely due to environmental
factors and lower spike count at slightly different parameters). Full
phenotype audit at n=10 × 60s × 1 ablation: ~66 min wall.

## Phase roadmap status

- **Phase 0** — closed. Three-mode taxonomy, voltage diagnostic,
  RIS molecular audit, sweep falsifications all complete.
- **T0 cascade question** — resolved at architectural level by
  per-edge sign convention. See `docs/t0_resolution_report.md`.
  PVC/AVB handling and FSM recalibration are the two open questions
  that succeed it.
- **T2-#4 sensory cascade calibration** — pending. Baseline
  captured; digitisation pending for Frechet eval. Not affected by
  T0 work; could proceed independently.
- **T4-2 compartmental plateau calibration** — pending. The
  original target numbers (AVAL peri ≥ 20 Hz, Δ ≥ +15 Hz) are
  **superseded** by per-edge mode achieving Δ +60 Hz, BUT this is
  firing-rate response not plateau dynamics. Plateau dynamics are
  an active-conductance phenomenon (L-type Ca + SK/BK termination)
  that per-edge sign mode does not address. Plateau target stays a
  separate question pending compartmental + SK/BK work.
- **T4-1 motor coupling** — pending. CPG baseline captured for
  curvature-ρ comparison.
- **T4-3 synaptic calibration** — **lower priority now.** Original
  framing was that this would unblock the touch cascade. Cascade now
  fires correctly under per-edge mode without weight tuning. May
  still be useful for fine-tuning specific edges (PVC over-
  activation may be one) but is not load-bearing.
- **T4-4 CeNGEN-conductance coupling** — pending. End of sequence;
  architectural overlay.
- **T4-5 INS-family peptide expansion** — pending. 5-peptide scope
  refinement noted (FLP-13 0.80 Jaccard with FLP-11 redundancy
  flagged before T0 work).
- **T4-6 trajectory correlation** — pending. Baseline ρ
  distribution captured; capstone.

## Ratified thresholds (status under per-edge findings)

- **T4-3 synaptic calibration** — original target AVAL peri ≥ 20 Hz
  AND Δ ≥ +15 Hz on ≥ 8/10 seeds. **Achieved at Δ +60 Hz under
  per-edge mode** (n=10, all seeds). Threshold superseded; new
  question is the readout's calibration to that firing regime.
- **T4-5 RIS/Turek phenotype** — ΔQUI ≤ −0.30 with 95% CI
  excluding zero. **Status under per-edge mode unknown** — RIS
  silenced at 0.8 Hz, original audit ran under default mode.
- **T4-2 AVA plateau (Mellem 2008, PMID 18587393)** — plateau
  duration ∈ [480, 720] ms; amplitude ∈ [18, 22] mV above rest.
  Stays — this is a plateau-dynamics question, separate from
  cascade firing.
- **T2-#4 sensory cascade calibration** — Each cascade's rate
  trace ≤ 10% Frechet distance to digitised published ΔF/F.
  Unchanged.
- **T4-1 motor coupling (curvature ρ)** — median ρ vs Tierpsy
  pool ≥ max(0.6, CPG_baseline + 0.15). Unchanged.
- **T4-6 trajectory correlation** — median ρ increases ≥ +0.10
  relative to baseline. Will need re-baselining under per-edge mode
  if per-edge becomes production default.

## Pending decisions

All decisions from the 2026-04-21 list are now contingent on
resolving the per-edge architecture question. New decisions added in
priority order:

**New, post-T0 resolution:**
1. **PVC/AVB handling under per-edge mode.** Two interpretations
   open and not yet falsified: (A) CeNGEN expression-vs-function
   mismatch (receptor presence ≠ functional dominance at specific
   synapses; PVC may need targeted override), or (B) canonical
   biology more nuanced than textbook (PVC excitation on anterior
   touch may be defensible under specific conditions). Document
   both; do not commit to either yet.
2. **FSM/classifier recalibration against per-edge firing
   distributions.** Refined by dPIR finding: question is not just
   "retrain to recover dREV" — AVA-ablation effect persists in
   dPIR channel under per-edge. Three sub-questions: (a) is the
   biological Chalfie phenotype now being measured through dPIR
   instead of dREV? (b) would recalibrated thresholds re-route the
   signal to dREV? (c) is the existing 18-readout architecture
   fundamentally incompatible with per-edge dynamics? Bank
   retraining is the technical prerequisite (deferred during
   overnight v2 Track B as LOGISTICAL_FAILURE).
3. **Whether per-edge becomes production default or stays opt-in.**
   Depends on (1) and (2). Currently `use_per_edge_glu_signs=False`
   is the default; per-edge is the better cascade biology but
   different readout dynamics.
4. **RIS molecular audit re-run under per-edge mode.** RIS silenced
   at 0.8 Hz under per-edge; April 21 audit transferability is
   unknown.

**Carried forward, contingent:**
5. **LIF-vs-graded head-to-head** decision. Now contingent on
   per-edge mode resolution; the same sign-convention question may
   exist in graded brain.
6. **Apply citation corrections** (Gao-Hobert→Mellem; Rogers→Cohen;
   etc.) across project docs. Unrelated to T0 work; could happen
   any time.
7. **Reconsider FLP-13 in T4-5** given 0.80 Jaccard with FLP-11.
   Independent of sign mode.
8. **Verify NLP-22 inclusion** (Nelson 2013 paper real, but CeNGEN
   shows zero NLP-22 in RIA). Genuine data-method mismatch;
   independent of T0.

## Audit-trail items

- **Sign-flip count discrepancy** between sessions: Session 1
  reports 518 hard sign flips across 5 counting methods; Session 2
  reports 415 across 4 methods. Probably from different
  `connectome.npz` builds or override list contents. Not blocking;
  worth reconciling.
- **535 GB `data/external/` inventory** revealed AVA in 100% of
  Atanas worms (10/10), AVD in 100%, AIZ in 90%. The strict cross-
  worm intersection filter that produced the 18-readout set
  excluded canonical command interneurons that ARE present in the
  data. Readout expansion is now clearly possible from existing
  data; the 18-readout set was a methodology choice, not a data
  limitation. Deferred until T0 work block fully closes.
- **Voltage fix in place but no-op for LIF.** Stays for biological
  documentation; matters for future compartmental + SK/BK work.

## References

- Primary measurement: `scripts/brain/artifacts/phase0_baseline_report.md`
- T0 resolution: `docs/t0_resolution_report.md` (canonical record
  of 2026-04-25 diagnostic block)
- T0 historical: `scripts/brain/artifacts/t0_run_report.md` (April
  21 framing, with postscript pointing to resolution report)
- Per-subsystem: `phase0_plateau_baseline.md`,
  `phase0_cascade_baseline.md`, `phase0_swap_jitter.md`
- Sweep artifacts: `scripts/brain/artifacts/phase0_postvolt*.csv`
- Comparison harness: `scripts/brain/phase0_postvolt_compare.py`
- Drive decomposition: `scripts/brain/phase0_avd_drive_decomp.py`
- Digitised reference traces: `scripts/brain/references/`
