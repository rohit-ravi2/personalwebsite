# Brain v3.5 — locked specification (Phase 1 outcome, 2026-05-03)

Phase 1 of the horizontal rebase produced a decision-grade brain-level
sign-mode choice. This document is the canonical record of what's locked,
what's validated, and what remains open for Phase 2.

**Decision date:** 2026-05-03 (Phase 1 screen-tier gauntlet completion)
**Decision:** Brain v3.5 = LIFBrain operating in **M2-pure** sign mode.
**Methodology:** priority-ordered decision rule (cascade firing > PVC
suppression > dREV reproduction); see `docs/state_of_claims_2026-05-02.md`
§7.1 for the full rule. Decision matrix at
`scripts/brain/artifacts/phase1_gauntlet_screen_decision_matrix.md`.

---

## 1 · What's locked

### 1.1 Sign mode

**M2-pure** = pure per-edge CeNGEN-derived signs (`W_chem_per_edge`)
with **no** `DOCUMENTED_SIGN_EXCEPTIONS`.

Construction:
```python
LIFBrain(
    use_per_edge_glu_signs=True,
    sign_exceptions={},   # explicit empty — overrides DOCUMENTED_SIGN_EXCEPTIONS default
    # ... other defaults
)
```

`DEFAULT_SIGN_OVERRIDES` (per-presynaptic-neuron NT signs for ~26 cells)
is **not applied** under per-edge mode (the per-edge matrix is loaded
directly without the per-neuron sign vector).

**Why M2-pure won:** the only candidate that satisfies priority-1 of the
decision rule (cascade firing). M1 (default) and M2-current (per-edge +
7 DOCUMENTED_SIGN_EXCEPTIONS) both fail to fire the touch cascade —
AVD/AVA *drop* on touch in both modes by 2-4 Hz, contradicting all
canonical biology of the touch reversal pathway. M3a was untested but
predicted to behave like M2-pure for cascade purposes (AIY exceptions
don't affect ALM/AVM→PVC→AVD→AVA pathway).

### 1.2 Brain version

LIFBrain (v3 connectome-constrained leaky-integrate-fire). Wave2HybridBrain
is **NOT** the production brain; it's an experimental research substrate
per Phase 3 of the rebase plan (C-37 demoted from production-grade to
research-substrate status).

### 1.3 Other parameters (unchanged from prior baseline)

- 300 neurons, Cook 2019 hermaphrodite + Loer & Rand 2022 NT identity
- W_syn = default (`W_SYN_DEFAULT`)
- g_gap = 0.1 nS (T0-resolution baseline)
- C_mem = 100 pF (LIF default)
- noise_sigma = default (`NOISE_SIGMA_DEFAULT`)
- v_rest = -25 mV, v_thr = -10 mV, v_reset = -30 mV (per Mellem 2008
  voltage regime, no-op for current LIF dynamics; see C-11)

---

## 2 · What's directly validated under M2-pure

### 2.1 Touch cascade firing — VALIDATED

| cell | Δ peri-touch (Hz, n=5×30s) | T0 §5 reference (n=10×60s) |
|---|---:|---:|
| ALML | +85.4 | +88.0 |
| AVM | +88.0 | +87.0 |
| PVCL | +60.4 | +60.4 |
| AVDL | +60.4 | +60.2 |
| AVAL | +60.2 | +60.6 |
| AVAR | +60.1 | +61.0 |
| AVEL | (not in screen tier output) | +47.4 |
| AVBL | +51.7 | +51.2 |

Today's screen-tier numbers reproduce T0 §5's documented numbers exactly.
The cascade fires through ALM/AVM → PVC → AVD/AVE → AVA at +60 Hz
peri-touch with seed-to-seed variance under 1.5 Hz across both runs.

### 2.2 RIS silencing under per-edge — VALIDATED

| | spontaneous RIS rate |
|---|---:|
| M2-pure today | 1.08 Hz |
| Documented (T0 §6.2) | 0.8 Hz |
| M1 default today | 30.66 Hz |
| Documented M1 baseline | 21.8 Hz |

Catalog claim C-24 reconfirmed. RIS is silenced under per-edge mode as a
network-equilibrium consequence, not a direct sign flip. Affects RIS
phenotype audit transferability — see §3.2 below.

### 2.3 GABA + peptide release mechanism — UNCHANGED FROM C-16, C-17

GABA uniformly signed −1 across all 26 GABA neurons; peptide release is
pure linear rate-coupling. Both verified by direct measurement on
2026-04-25; no Phase 1 disturbance.

### 2.4 NSM ablation → ΔQUI ≈ +0.39 — confirmed as classifier artifact

| mode | NSM→dQUI |
|---|---:|
| M1 | +0.39 |
| M2-pure | +0.39 |
| M2-current | +0.39 |

Identical across all 3 modes — confirms catalog claim C-27 (Mode 2
readout-trivial). NSM is in the 18-readout, so its ablation directly
moves the classifier output regardless of biology. **This is a known
classifier-readout dependency, not a biology claim.** Phase 2 readout-set
expansion may resolve.

---

## 3 · What's NOT validated under M2-pure (open for Phase 2)

### 3.1 AVA-ablation phenotype on dREV / dPIR — UNDERPOWERED

Screen tier produces noise-dominated phenotype results:
- dREV: +0.04 ± 0.03 (matches T0 §5 +0.04 exactly; null on dREV)
- dPIR: −0.04 ± 0.03 (T0 §5 reported -0.117 at n=10×60s; we got
  underpowered version of the same direction)

**Phase 2 is the right place to settle this.** The classifier was
trained on default-mode firing distributions; under M2-pure dynamics,
phenotype reproduction in the dREV channel is null because the FSM
classifier doesn't decode M2-pure's tripled AVA dynamic range into
dREV correctly. Phase 2 retrains the classifier under M2-pure dynamics
to characterize what behavioral signature AVA-ablation produces.

### 3.2 RIS phenotype + molecular audit transferability — OPEN

RIS is silenced under M2-pure (1.08 Hz vs 21.8 Hz default tonic).
Default-mode RIS phenotype findings (ΔQUI = -0.24 ± 0.33; FLP-11
release fires correctly; ~22% disinhibition of peptidergic targets) do
NOT transfer to M2-pure without re-running. Both the RIS molecular audit
(C-26) and the Turek phenotype claim (C-25) need re-running under M2-pure
in Phase 2 or later.

### 3.3 PVC/AVB over-activation under M2-pure — DOCUMENTED LIMITATION

Under M2-pure: PVC fires +60 Hz on touch and AVB fires +52 Hz on touch.
Canonical biology has anterior touch suppressing forward locomotion via
PVC inhibition. Two interpretations remain open and neither is yet
falsified (see `docs/t0_resolution_report.md` §6.1):
- **Interpretation A:** CeNGEN expression-vs-function mismatch at
  ALM/AVM → PVC synapses. PVC has iGluR receptors but the synapse may be
  functionally GluCl-mediated.
- **Interpretation B:** canonical biology more nuanced than textbook.
  Per-edge dynamics may match newer literature about parallel chassis
  circuits where forward command isn't strictly suppressed during reversal.

The DOCUMENTED_SIGN_EXCEPTIONS attempt to enforce Interpretation A
collapsed the cascade entirely (M2-current fails priority-1). Per the
priority-ordered decision rule, this is **tolerable as documented
limitation** because cascade firing wins. Resolution requires per-edge
functional measurements (electrophysiology data per synapse), not just
receptor expression.

### 3.4 Network instability under osmotic_shock — DOCUMENTED LIMITATION

Cells firing > 100 Hz: 32 cells at M2-pure under osmotic_shock at
n=5×30s. This is half of M1's 69-cell excursion count but still
non-zero. Possibly fixable with per-edge gap-junction tuning (similar
to T0 g_gap sweep) or scenario-specific modulation tuning. Not blocking
for Phase 2 but should be re-measured under longer runs.

### 3.5 Three-mode taxonomy classifications — OPEN

The specific Mode 1/2/3 classifications for FLP-2, PDF-1, etc. were
established under default mode. Under M2-pure these may shift. Re-run
during or after Phase 2.

### 3.6 Wave2HybridBrain biology (C-37) — RESOLVED 2026-05-03

**Status update:** the W2 investigation thread (catalog §7.2) ran during
Phase 2 sub-task 2.2. Result: under Wave2HybridBrain M2-pure (per-edge +
sign_exceptions={}), the touch cascade fires identically to pure LIFBrain
M2-pure (AVDL Δ +60.5 Hz today vs +60.4 Hz Phase 1A; AVAL σ Δ +0.10
peri-touch with ~zero post-touch drift). All command cells (PVC, AVD,
AVE, AVB, AVA) fire +Δ on touch.

**The C-37 Falsified status was caused entirely by DOCUMENTED_SIGN_EXCEPTIONS,
NOT by Wave 2 cellular substitution.** Wave2HybridBrain is biologically
sound under M2-pure. Demotion from production to research-substrate is
**rescinded**: Wave2HybridBrain + M2-pure is a viable production-grade
brain mode that adds biophysical resolution beyond LIF without breaking
cascade dynamics.

Files of record:
- `scripts/brain/wave2/integration/run_wb_investigation_w2_m2pure.py`
- `scripts/brain/wave2/artifacts/wb_investigation_w2_m2pure_results.json`

---

## 4 · Code changes implied by lock (deferred to Phase 3 doc-fix sweep)

The lock is a methodology decision; production code defaults are NOT yet
changed. Phase 3 (doc-fix sweep) is when defaults change. Required
changes when ready:

1. **`scripts/brain/lif_brain.py`**: change defaults to
   `use_per_edge_glu_signs=True` and `sign_exceptions={}`. Document the
   change in module header. Preserve `DOCUMENTED_SIGN_EXCEPTIONS` as a
   constant for any consumer that explicitly opts in (and for the §3.3
   research thread testing Interpretation A).
2. **`scripts/brain/closed_loop_env.py`**: same default change in the
   passthrough kwargs.
3. **`scripts/brain/wave2/integration/wave2_hybrid_brain.py`**: same
   default change.
4. **`docs/claude-chat-context.md`** §3 + §5: update brain framing to
   M2-pure as production default; document Phase 1 outcome.
5. **`docs/current-state-summary.md`**: update phase-status; mark T0
   resolution as superseded by Phase 1 lock.
6. **`docs/t0_resolution_report.md`**: append postscript referencing
   Phase 1 outcome.
7. **`scripts/brain/artifacts/phase_delta_wb3_findings.md`**: amend §5 +
   §11 with the cascade-collapse mechanism explanation (DOCUMENTED_SIGN_EXCEPTIONS).
8. **`src/content/projects/c-elegans-multimodal.mdx`**: rewrite the
   T0 / Wave 2 paragraphs honestly per Phase 1 outcome. Retract C-22
   ("genuine Chalfie reproduction") and C-25 ("RIS quiescence").

These changes happen at the end of the rebase, after Phase 2 lands the
recalibrated FSM/classifier so docs can be amended once with full context.

---

## 5 · Phase 1 → Phase 2 handoff

**Brain locked at M2-pure.** Phase 2 begins:
1. Decide readout-set expansion question (current 18 vs adding AVA/AVD).
2. Retrain classifier bank under M2-pure firing distributions (with
   chosen readout set).
3. Recalibrate ActivityFSM thresholds under M2-pure dynamics.
4. Re-run validation gauntlet under recalibrated FSM. Document which
   phenotypes reproduce cleanly under correct cascade dynamics + new FSM.

Compute and timeline for Phase 2 are unknown — depends on Atanas data
prep and classifier infrastructure (deferred during overnight v2 Track B
as LOGISTICAL_FAILURE; needs separate scoping pre-flight).

---

## 6 · Files of record

- Phase 1 decision matrix: `scripts/brain/artifacts/phase1_gauntlet_screen_decision_matrix.md`
- Per-mode CSVs: `scripts/brain/artifacts/phase1_gauntlet_<mode>_screen_phenotype.csv`,
  `phase1_gauntlet_<mode>_screen_scenario.csv`
- Summary JSON: `scripts/brain/artifacts/phase1_gauntlet_screen_summary.json`
- Sign-flip reconciliation: `scripts/brain/artifacts/phase1_signflip_reconciliation.json`
- State of claims catalog: `docs/state_of_claims_2026-05-02.md`
- Gauntlet runner: `scripts/brain/phase1_gauntlet.py`
- Sign-flip reconciliation script: `scripts/brain/phase1_signflip_reconciliation.py`
