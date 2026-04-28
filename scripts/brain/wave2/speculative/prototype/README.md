# X.1d — GNN prototype (bounded effort, sanity-check scope)

**Outcome: FAIL (informative-negative)**

- Init test MAE: 52.9 mV
- Final test MAE (after 40 epochs): 46.2 mV
- Pass threshold: < 5 mV
- Verdict: prototype did not converge to a useful approximation of single-compartment ground truth.

## What was attempted

A minimal Variant A architecture (per `gnn_architecture_sketch.md`):
- 2-node graph: node 0 = "soma" (EGL-19 + leak), node 1 = "axon-stub" (leak only).
- Edge: axial conductance, learnable.
- Channel kinetics mechanistic (Nicoletti 2024 EGL-19 + leak), unrolled forward Euler with PyTorch autograd.
- Trainable: log-axial-conductance, log-gbar-EGL-19 (node 0), log-gleak (both nodes). 4 trainable scalars.
- Loss: MSE on V_0 trajectory.
- Training data: 64 single-compartment NumPy ground-truth traces (200 ms, dt 0.025 ms), under random injection currents in {-3 to +5 pA}, voltage range -60 to -17 mV. 51 train / 13 test split.
- Optimizer: Adam, lr 5e-3, 40 epochs, batch size 8, gradient clip 1.0.

## What happened

- Training reduced loss by ~13% (3434 → 2985) and test MAE by ~13% (52.9 → 46.2 mV).
- Axial conductance grew from initial 1e-3 S to 1.8e-3 S (model trying to tighten 2-node coupling).
- gbar_egl19 fell from 9.3e-6 to 6.4e-6 S/cm² (model trying to dampen node-0 EGL-19 to fit slower compromise dynamics).
- gleak0 fell from 1.34e-5 to 8.1e-6 S/cm² (further dampening of node-0 dynamics).
- Final state: model has dampened node 0's intrinsic dynamics in an attempt to match the slower-effective-dynamics that single-compartment ground truth has — but this isn't what the GNN should be doing structurally.

## Why this matters for the architectural roadmap

The prototype's failure mode is **structurally informative** about Variant A:

1. **A 2-node model with finite axial coupling cannot trivially reduce to single-compartment dynamics.** Adding a charge-sink axon stub fundamentally changes the cell's effective dynamics, even when channel kinetics are mechanistically anchored. The "obvious" expectation that "axial_g → ∞ recovers single-compartment" is true only in the asymptotic limit; at any finite axial_g the dynamics differ.

2. **Mechanistic anchoring is not free.** Even with channel kinetics fixed and only 4 trainable params, training landscape is non-trivial. The parameter drift toward dampened-channel + tighter-coupling regime suggests the model is fitting in a way that contradicts the architectural premise (we wanted axial_g + per-segment gbar to be tunable, but training is using gbar_egl19 reduction as the dominant gradient signal, which is a degenerate path).

3. **Forward-pass instability constraints are real.** dt = 0.125 ms was the largest stable timestep tested; finer dt + longer trajectories increase backprop memory burden. For 57-segment AVAL with 8 channels, this scaling is concerning.

4. **The prototype does not invalidate Variant A as a Wave 3 path** — only confirms that:
   - Per-segment density fitting in Variant A is *the* load-bearing degree of freedom (more important than learnable axial coupling).
   - A larger, better-prepared training run with proper regularization and biology-informed initial conditions could plausibly converge.
   - But the bounded prototype effort here does not reach that demonstration.

## Comparison framing for `x1_summary.md`

This prototype does NOT establish that GNN Variant A is a viable Wave 3 approach. It also does not establish that it is a dead-end — the negative result here is consistent with "needs more careful setup" as much as with "structural inefficiency." Per the spec's bounded-effort discipline:

> Document state honestly; negative results documented honestly are valuable; do NOT over-invest time chasing positive prototype results.

The honest characterization: **prototype attempted, training pipeline functional, training converged to a stable but non-useful regime**. Further investigation (better init from Brian2 single-compartment params, regularization toward biology priors, deeper channel coverage, bigger graph) is plausible but deferred.

## Files

- `data.py` — generates 64 voltage traces from a NumPy single-compartment EGL-19+leak integrator.
- `gnn_prototype.py` — `TwoNodeGNN` PyTorch module.
- `train.py` — training loop, evaluation, results dump.
- `train.log` — full training output.
- `train_data.npz` — generated dataset.
- `results.json` — final metrics + history.
- `trained_model.pt` — model checkpoint.

## Reproduce

```bash
cd /home/rohit/Desktop/website/personalwebsite/scripts/brain/wave2/speculative/prototype
~/venvs/ds/bin/python data.py     # ~5 s; generates train_data.npz
~/venvs/ds/bin/python -u train.py # ~13 min on RTX 4060 Ti; produces results.json + trained_model.pt
```

## Honest limits

- Single-compartment ground truth was **synthetic NumPy** rather than Brian2 / NEURON validated traces. The Nicoletti EGL-19 parameters match `wave2/channels/egl19.py` and have been validated to <5% in Phase β, but the explicit-Euler integration here may have small numerical differences from Brian2's integrator.
- Only EGL-19 + leak; no SLO-1, no Ca-pool. So this prototype does not test Mellem-style plateau dynamics, which are the actual condition-6 failure mode.
- Only 2 nodes, 4 trainable parameters; not the 57-segment AVAL with hundreds of trainable parameters that would be the actual Variant A architecture.
- Training did not include early-stopping, hyperparameter search, or biology-informed priors. A real Variant A study would do all three.
- Test MAE is computed on out-of-distribution random injection currents but the train and test set come from the same distribution, so this isn't a generalization test in the strong sense.
