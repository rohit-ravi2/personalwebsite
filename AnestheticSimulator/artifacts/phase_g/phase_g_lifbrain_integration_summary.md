# Phase G LIFBrain integration — work-block summary

**Started:** 2026-05-12 (post horizontal-rebase commit ce9c7c9)
**Status:** **PAUSED at CP2 (hard stop)** — calibration gap diagnosis at
`CALIBRATION_GAP.md`. CP3-CP5 blocked pending decision on
fix-in-place vs Phase G v2 re-architecture.

---

## Checkpoint status

| CP | Description | Status |
|---|---|---|
| Pre-flight | Verification of stack plumbing + biology decisions | ✅ COMPLETE (`phase_g_lifbrain_preflight.md`) |
| CP1 | Substrate switch demo → LIFBrain | ✅ COMPLETE — infrastructure landed cleanly, smoke test PASS |
| CP2 | Behavioral threshold calibration vs Crowder 1996 | ❌ **HARD STOP** — 0% suppression across 5 OOM in dose; implementation gaps diagnosed |
| CP3 | Cross-anesthetic verification | ⏸ blocked behind CP2 |
| CP4 | Ablation harness consumption verification | ⏸ blocked behind CP2 |
| CP5 | Documentation + commits | ⏸ blocked behind CP2 |

---

## Headline outcomes

**CP1 infrastructure is sound and committed-quality:**
- `make_lifbrain_substrate(seed)` wraps ClosedLoopEnv with M2-pure config
  + Phase 2 recalibrated stack
- `lifbrain_behavioral_readout()` extracts FWD state fraction (primary
  anchor per Crowder swimming-behavior analog) + AVA/AVB command-cell
  rates (secondary diagnostic) + FSM state distribution
- W_chem sync helper writes Phase G's `_W_chem_runtime` modifications back
  to Brian2 `syn_exc.w` / `syn_inh.w` post-construction
- Halothane @ 1× clinical EC50 smoke test passed: env loads, 8 mechanism
  classes engaged at 0.998 max occupancy, FSM produces interpretable
  distribution, command cells at biological rates, cascade fires under
  touch + perturbation

**CP2 surfaced a load-bearing implementation gap:**
- Phase G perturbation produces ZERO detectable behavioral effect across
  5 orders of magnitude in halothane dose (0.001× → 10× clinical EC50)
- Per-seed FWD fractions byte-identical across all doses including baseline
- Diagnosis identifies 3 implementation issues, NOT a biology gap:
  1. ModulationLayer overwrites Phase G's I_ext modifications every step
  2. NT string equality check fails (LIFBrain stores 'Acetylcholine (ACh)'
     not 'ACh') → nAChR antagonism silently no-ops
  3. glucl_potentiation / complex_ii_block / nca_block hooks are missing
     entirely from apply_to_brain

Only **gaba_potentiation** is functional under production substrate. The
original Phase G demo's apparent dose-response worked because the demo
network used additive-I_ext approximation that papered over the named-hook
breakdowns.

---

## Why this is implementation-side, not biology-side

If Phase G's hooks worked correctly, halothane at 1× clinical EC50 would
hyperpolarize all 300 neurons via complex_i + k2p (≈ −250 to −400 pA),
zero out 159 cholinergic-→-nAChR-receptor edges, enhance ~150 GABA + GluCl
inhibitory edges. The combined effect is substantial; the binding-side
saturation flagged in the original Phase G dose-response (occupancy 0.998
at clinical EC50) would re-emerge as the "behavioral suppression is too
strong because we engage too many targets at once" question.

The current 0% suppression result tells us nothing about biology — only
about implementation. Biology calibration is blocked behind the
implementation fixes.

---

## Two paths forward

**(A) Fix bugs 1+2+3 in-place, re-run CP2.** Estimated ~2-3 hours code +
35 min compute. Achievable in continuation of this work block.

**(B) Pause for Rohit's review.** Substantial implementation gap that
wasn't caught in Phase G shipping. Better to surface the architectural
question (Phase G v1 fix vs Phase G v2 re-spec) for ~1 hour discussion
before producing semi-working code.

**Recommended: B.** The glucl / complex_ii / nca hooks need biological
input. complex_ii_block — does it have the same K-ATP-opening consequence
as complex_i_block? nca_block (NCA-1 sodium leak) — what's the LIFBrain
substrate analog?

---

## Standing followups (post-CP2-resolution)

When CP2 calibration eventually closes:
- CP3 cross-anesthetic verification on production substrate
- CP4 ablation harness consumption verification
- CP5 documentation + commits + push
- AVA → dFWD literature precedent check (task #15, deferred from Phase 3
  doc-fix sweep)
- Phase G architecture v2 update if Option B chosen

---

## Files produced in this work block

**New code (Phase 2 — LIFBrain integration):**
- `src/phase_g_lifbrain_substrate.py` — factory + behavioral readout
- `src/phase_g_lifbrain_calibration.py` — CP2 runner

**Modified code:**
- `src/phase_g_network_perturbation.py` — added `_sync_wchem_to_brian2`
  helper for W_chem mod propagation to Brian2 Synapses

**Artifacts:**
- `artifacts/phase_g/phase_g_lifbrain_preflight.md`
- `artifacts/phase_g/phase_g_lifbrain_cp1_smoke.json`
- `artifacts/phase_g/phase_g_lifbrain_cp2_calibration.json` (gap data)
- `artifacts/phase_g/phase_g_lifbrain_cp2_calibration.md` (gap data)
- `artifacts/phase_g/phase_g_lifbrain_cp2_run.log`
- `artifacts/phase_g/CALIBRATION_GAP.md` (this hard-stop diagnosis)
- `artifacts/phase_g/phase_g_lifbrain_integration_summary.md` (this file)

No commits yet — all work uncommitted pending decision on path forward.

---

## Methodology continuity

Per the horizontal rebase's pause-with-documentation principle: this is
the methodology working as intended. The CP2 calibration question was
flagged as the natural pause point in the work-block spec. The result
surfaces an implementation gap rather than a biology limit; the response
is to document the gap honestly and pause for review rather than
push-through-to-publishable-claim.

The diagnosis is rigorous because the rebase methodology (catalog-driven,
direct-measurement-grounded, honest about Falsified-but-cited) sets the
bar for what counts as "calibration succeeded." 0% suppression across 5
OOM in dose is not 50% suppression at any dose, regardless of how the
fitting is framed.
