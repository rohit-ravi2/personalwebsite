# Production simulator → Wave P handoff

**Status:** SCAFFOLDED. Activates **only after Phase H lands** (i.e., months 5-6 of Wave P).

---

## Intent

The production simulator at `/home/rohit/Desktop/website/personalwebsite/scripts/brain/` (LIFBrain, GradedBrain, closed_loop_env, FSM, modulator layer) is the user-facing C. elegans simulator. Wave P's outputs feed back into the production simulator as a documented anesthetic-mode plugin **only after** Wave P's empirical validation (Phase H) confirms the predictions.

The production simulator is **not** modified during Wave P scaffolding or the first 5 months of Wave P execution. Modification happens in month 6 if Phase H >= 4/8 anchors.

---

## Plug-in protocol (when Wave P passes Phase H)

### File-level plug-in

A single new module is added to the production simulator:

```
scripts/brain/anesthetic_overlay.py
```

This module:

1. Reads `AnestheticSimulator/artifacts/kinetics/anesthetic_kinetic_shifts.npz` (Phase D output).
2. Reads `AnestheticSimulator/artifacts/occupancy/occupancy_matrix.npz` (Phase C output).
3. Provides a function `apply_anesthetic(brain_object, anesthetic="halothane", dose_mult=1.0)` that:
   - Fetches per-target occupancy at the given dose.
   - Looks up per-target kinetic shifts.
   - Applies multiplicative overlays to channel `g_max` and `tau_decay` parameters.
   - Substitutes the LIF synapse with the Markov synapse module (if invoked with `--markov-synapses`).
   - Activates the metabolic layer module (if invoked with `--metabolic`).

### Production-simulator API surface change

```python
# Existing (unchanged)
brain = LIFBrain(...)
brain.run(scenario="touch")

# New (Wave P plugin)
import anesthetic_overlay
anesthetic_overlay.apply_anesthetic(brain, anesthetic="halothane", dose_mult=1.0)
brain.run(scenario="touch")
# Output reflects anesthetic-perturbed dynamics
```

### Backward compatibility

The plug-in is **opt-in**. Existing production-simulator code paths run unchanged. Anesthetic overlay activates only when `anesthetic_overlay.apply_anesthetic` is called. The default state (no anesthetic) is bit-identical to the pre-Wave-P production simulator.

---

## Validation gate before plug-in activates

Before `anesthetic_overlay.py` lands in `scripts/brain/`, the following must hold:

1. **Phase H >= 4/8 anchors pass** — Wave P validation confirms the predictions are at proof-of-concept level.
2. **Lesion test (G.1.5) passes** — multi-target framing confirmed at network level.
3. **Production-simulator integration test passes** — applying the overlay to a known production-simulator scenario produces qualitatively expected results (locomotion reduces under anesthetic; cell-firing rates shift).
4. **No-overlay regression test passes** — existing production-simulator runs without `apply_anesthetic` produce identical output to pre-overlay code.

If any of these fail, the plug-in does **not** land in production; Wave P remains a standalone research artifact and the production simulator is untouched.

---

## License considerations

Wave P uses some non-commercial-restricted tools (AlphaFold-Multimer, RoseTTAFold-AllAtom). The **outputs** of Wave P (kinetic shift table, occupancy matrix) are derived numerical values, not the source structures. Numerical values fall outside the non-commercial-use clauses on the structure prediction tools.

However, if the plug-in references or distributes the predicted PDB structures (e.g., bundling them with the production simulator for offline use), the non-commercial clause re-applies. The plug-in design avoids this:

- The plug-in references the kinetic-shift NPZ and occupancy-matrix NPZ.
- It does **not** include any predicted PDB structures.
- The structures live in `AnestheticSimulator/artifacts/structures/` and are referenced by file path.

If the production simulator ever ships commercially:

- The kinetic-shift NPZ + occupancy-matrix NPZ can be redistributed (numerical values, derived through academic-licensed tools).
- The PDB structures cannot be redistributed (would require AlphaFold license re-evaluation).
- The user's standing instruction is to flag this if commercial deployment becomes a goal.

---

## Reverse-direction handoff (production simulator → Wave P)

Wave P consumes some inputs from the production simulator's data:

- Atanas 2023 calcium recordings at `scripts/brain/artifacts/atanas_worm_*.npz` — used in Phase I (inverse design, stretch).
- Production simulator's connectome.npz, neuron_positions.npz, motor_interface.npz — already overlapping with notebook pipeline data; same content.

These reads are read-only and do not modify the production simulator.

---

## Plug-in implementation timeline

The plug-in is **not** implemented during Wave P kickoff. The implementation is a month-6 deliverable, contingent on Phase H success.

Estimated effort once Phase H passes: 1-2 weeks of integration work.

Activities:

1. Author `anesthetic_overlay.py` per the API above.
2. Add unit tests against `AnestheticSimulator/artifacts/kinetics/` outputs.
3. Add integration tests against existing production-simulator scenarios.
4. Document in production simulator's CLAUDE.md / README.
5. Land via PR.

---

## What does not change in the production simulator

- The 9-modulator layer (FLP-11, FLP-1, etc.) — Wave P's anesthetic effects on peptide processing (Tier 2: EGL-3, EGL-21) would integrate as additional overlays, but only after Tier 2 ships.
- The FSM (BehavioralFSM, ActivityFSM) — Wave P's IMMOBILIZED state is an FSM addition, but the addition happens in `anesthetic_overlay.py`, not in core FSM code.
- The MuJoCo body — Wave P's locomotion readout uses the existing body integration; no body-level changes.
- The classifier bank — Wave P does not retrain the classifier; if classifier readouts shift under anesthetic-perturbed dynamics, Wave P documents but does not modify the classifier.

---

## What does change

- One new file: `scripts/brain/anesthetic_overlay.py`.
- One new optional FSM state: `IMMOBILIZED`.
- Per-cell ATP / K-ATP state variables (added to NeuronGroup if metabolic layer is active).
- Markov synapse module (added to Synapses object if `--markov-synapses` is active).

All additions are opt-in. The default production-simulator behavior is unchanged.

---

## STATUS

This document is for reference only. The plug-in lands **only after** Phase H >= 4/8 anchors. Until then, no production-simulator code changes for Wave P.
