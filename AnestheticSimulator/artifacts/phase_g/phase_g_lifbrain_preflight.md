# Phase G LIFBrain integration — pre-flight findings

**Date:** 2026-05-12
**Status:** Pre-flight complete; proceeding to CP1 (no hard-stop pushback)
**Predecessor:** Horizontal rebase Phase 0-3 (5 commits 9505d23..ce9c7c9)

---

## Verification scope

Read in full:
- `docs/brain_v3.5_locked.md` — canonical M2-pure brain spec
- `docs/state_of_claims_2026-05-02.md` — catalog (29 Direct / 12 Direct-narrow / 7 Inferred / 5 Falsified-but-cited as of doc-fix sweep)
- `docs/phase2_preflight.md` — architecture sign-off
- `AnestheticSimulator/artifacts/phase_g/phase_g_architecture.md` — Phase G architecture v1
- `AnestheticSimulator/src/phase_g_network_perturbation.py` — perturbation manager (483 lines)
- `AnestheticSimulator/src/ablation_harness.py` — harness (997 lines)
- `scripts/brain/closed_loop_env.py` — recalibrated stack entry point
- `scripts/brain/lif_brain.py` — Brian2 Synapses construction pattern
- `AnestheticSimulator/artifacts/kinetics/wave2_overlay_v2.json` — schema verification

---

## 1 · Substrate-switch scope

**Phase G perturbation manager is already substrate-agnostic.** The
`AnestheticPerturbation.apply_to_brain()` API expects a duck-typed object
with: `.names`, `.neurons.I_ext`, `._W_chem_runtime`, `.W_syn`,
`.nt_primary`, `.idx`, `.spikes`. **LIFBrain has all of these** (verified
in `scripts/brain/lif_brain.py` — `_W_chem_runtime` at line 310, syn_exc at
346, nt_primary at 341 via per-pre NT vector, idx is constructed alongside
names).

**Demo-network references are localized to:**
- `phase_g_network_perturbation.py:dose_response_sweep()` — the 50-neuron
  Brian2 demo, only invoked by the module's own `main()` smoke-test path
- `ablation_harness.py:make_phase_g_demo_substrate()` — the default
  substrate provider; ablation harness `AblationHarness(substrate=...)` is
  already a callable kwarg
- `ablation_harness.py:make_lifbrain_substrate_TODO()` — explicit
  placeholder raising NotImplementedError, awaiting this work block

So the substrate switch is mostly *additive infrastructure* (new
`make_lifbrain_substrate(seed)` factory + LIFBrain dose-response runner)
rather than refactoring existing code. The demo substrate stays as legacy
(useful for fast smoke tests).

## 2 · CRITICAL design issue (tractable) — W_chem mod timing

**Problem.** Phase G's `apply_to_brain()` modifies `brain._W_chem_runtime`
in-place via numpy mutations. But LIFBrain builds Brian2 `Synapses(...)`
objects at construction time with weights bound via `self.syn_exc.w =
exc_w.tolist()` (line 354). **NumPy mutations to `_W_chem_runtime` after
construction do not propagate to the running Brian2 simulation** because
the Synapses objects already have their `.w` arrays initialized from the
pre-mutation matrix.

The demo network avoids this because it uses additive I_ext only (no
W_chem modifications in the dose-response path).

**Fix.** Phase G's `apply_to_brain()` for LIFBrain needs to write modified
weights BACK to the Brian2 Synapses objects after numpy mutations:
```python
# After W_chem_runtime mutations:
if hasattr(brain, "syn_exc") and hasattr(brain, "syn_inh"):
    # Rebuild w arrays from modified _W_chem_runtime
    exc_w_new = brain._extract_exc_weights_from_runtime()  # helper to add
    inh_w_new = brain._extract_inh_weights_from_runtime()
    brain.syn_exc.w[:] = exc_w_new
    brain.syn_inh.w[:] = inh_w_new
```

OR (simpler): construct a fresh LIFBrain per dose (what the ablation
harness already does per seed). Construction overhead is ~5-10 sec for
LIFBrain; for 25 calibration runs this is ~3 minutes of overhead — well
within budget. **This is the chosen approach** for CP1 simplicity.

**Tractable, not blocking.** No pushback needed.

## 3 · Recalibrated stack plumbing verification

`closed_loop_env.py` supports opt-in `bank_path`, `cal_path`, and
`fsm_thresholds_path` kwargs (added 2026-05-03, committed in
`51b4b58`). Phase G's LIFBrain substrate factory will pass:

```python
env = ClosedLoopEnv(
    seed=seed,
    use_per_edge_glu_signs=True,          # M2-pure
    sign_exceptions={},                     # M2-pure: no DOCUMENTED_SIGN_EXCEPTIONS
    bank_path=ART / "classifier_bank_v2_a2balanced.npz",
    cal_path=ART / "calibration_m2pure.npz",
    fsm_thresholds_path=ART / "phase2_fsm_thresholds_behavioral_m2pure.json",
    fsm_mode="classifier",                  # behavioral_fsm (vs activity_fsm)
    enable_modulation=True,                 # production substrate has modulation
)
brain = env.brain   # the LIFBrain instance
```

The env also runs the FSM internally; behavioral readout via
`env.fsm_states` (list of state IDs over time).

## 4 · Ablation harness consumption — API contract verification

`ablation_harness.py` already has:
- `make_lifbrain_substrate_TODO()` — placeholder
- `lifbrain_readout()` — full implementation at lines 61-104, returns
  `firing_rate_Hz`, `n_spikes`, `fsm_state_fractions`,
  `command_interneuron_rates`. Reads from `brain.spikes`, `brain.idx`,
  `brain.fsm_state_history`.
- `_do_run` already branches on `substrate_label`: if not
  `phase_g_demo_50neuron`, uses `self.pert.apply_to_brain(...)` directly.

**Harness API contract is preserved by the substrate switch.** No
modifications to the harness needed — only need to:
1. Implement `make_lifbrain_substrate(seed)` to replace the TODO placeholder
2. Verify `lifbrain_readout` works on the actual LIFBrain (currently has
   TODO comment but the function body is implemented; needs smoke test)

## 5 · Behavioral threshold calibration — biology decision

**Crowder 1996 operationalized "behavioral suppression" via swimming
(locomotor) behavior.** Production substrate options:

| Option | Readout | Pros | Cons |
|---|---|---|---|
| **(A) FWD state fraction** | env.fsm_states → count(FWD) / len(states) | Direct locomotor analog (FWD = forward locomotion behavioral state); matches Phase 2.5's strongest AVA-ablation signal | Coarse; depends on FSM threshold calibration |
| (B) Command interneuron firing rates | AVA / AVB rates (averaged) | Mechanistic; directly bridges Phase G perturbation to physiology | Removes one inferential layer (FSM); but Crowder didn't measure firing rates |
| (C) Mean network firing rate | aggregate suppression | Closest to original Phase G demo readout | Aggregate; loses behavioral specificity |
| (D) Body coupling (locomotor velocity) | requires MuJoCo body | Most direct Crowder analog | Out of scope for this work block |

**Decision: A (FWD state fraction) as primary; B (command interneuron
rates) as secondary diagnostic.**

Rationale:
1. FWD state is the explicit behavioral classification — Crowder's swimming
   measurement closest analog without body coupling.
2. Phase 2.5 default tier established AVA-ablation produces dFWD = -0.302
   (Cohen's d ≈ 0.93) — the strongest behavioral signal in the recalibrated
   stack. Anesthesia should produce a similar FWD suppression at clinical
   EC50 if it's hitting the same effective pathway.
3. Command interneuron rates as secondary diagnostic: if FWD fraction
   suppresses but command rates don't (or vice versa), that surfaces
   mechanism vs readout-stack questions.

**Calibration target: 50% behavioral suppression at clinical EC50** =
FWD fraction at dose=1.0 is 0.5 × FWD fraction at dose=0.

If calibration closes within 2× of clinical: success.
If within 5×: partial calibration, document gap honestly.
Beyond 5×: hard stop per spec.

## 6 · Compute budget

| Step | Per-run wall | Runs | Total |
|---|---|---|---|
| CP1 smoke test | ~3 min | 1 | 3 min |
| CP2 calibration (halothane × 5 doses × 5 seeds) | ~3 min | 25 | ~75 min |
| CP3 cross-anesthetic (6 anesthetics × 3 seeds) | ~3 min | 18 | ~54 min |
| CP4 harness mini-ablation (halothane × UNC-49 × 3 seeds × 2 conditions) | ~3 min | 6 | ~18 min |

**Total compute: ~150 min (2.5 hr).** Fits easily within overnight + morning
review.

Per-run estimate of ~3 min is conservative based on Phase 2.5 default tier
timing (60s sim with full closed-loop env, modulation, FSM, classifier
prediction = ~2.5-4 min/run, with most cost in brain construction +
spike-loop). At 30s sim per Phase G dose-response convention, ~2 min/run is
realistic — total may be closer to 100 min.

## 7 · Schema verification

`wave2_overlay_v2.json` schema confirmed consumable as-is:
- Top-level: `by_anesthetic`, `_meta`
- 6 anesthetics (halothane, isoflurane, sevoflurane, propofol, ketamine,
  etomidate)
- Per-target: `mechanism_class`, `occupancy_1xEC50`, `parameters`,
  `occupancy_1xEC50_v1`, `correction_applied`

Phase G's `compute_perturbation_vector` reads `mechanism_class` and
`occupancy_1xEC50`. No schema changes needed.

## 8 · Decisions for CP1

| Decision | Choice | Rationale |
|---|---|---|
| Substrate factory entry point | `ClosedLoopEnv` with M2-pure config | Provides LIFBrain + modulation + FSM in one wrapper |
| FSM mode | `classifier` (behavioral_fsm) | Tested at Phase 2.5; recalibrated thresholds available |
| W_chem mod timing | Fresh substrate per dose | Avoids Brian2 Synapses re-init complexity; ~10s overhead per dose acceptable |
| Demo substrate | Preserved as legacy via `substrate_label` flag | Useful for fast smoke tests + harness back-compat |
| Calibration readout (primary) | FWD state fraction from env.fsm_states | Direct locomotor analog; strongest Phase 2.5 signal |
| Calibration readout (secondary diagnostic) | AVA + AVB command interneuron firing rates | Surfaces mechanism vs readout-stack questions |
| Modulation layer | Enabled (`enable_modulation=True`) | Production-substrate convention |
| Per-run sim duration | 30 s | Matches Phase 2.5 / Phase G dose-response convention |
| Behavioral threshold target | Within 2× of clinical EC50 = success; within 5× = partial | Per spec |

## 9 · No pushback document needed

Pre-flight surfaced one design issue (W_chem mod timing) that is tractable
via fresh-substrate-per-dose approach. Pre-flight surfaced one biological
judgment call (calibration readout) that is decidable from existing
catalog data (FWD state fraction primary).

**Proceeding to CP1.**

Status persistence: this doc + checkpoint at end of each CP per spec.
