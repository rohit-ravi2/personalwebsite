# Phase I — Differentiable simulator + inverse design (stretch)

**Phase letter:** I
**Status:** SCAFFOLDED, DEFERRED. Activates only if Phase H ≥ 6/8 anchors.
**Predecessor:** Phase H verdict.
**Successor:** none (deliverable feeds the paper directly as a methodological extension).
**Compute:** local + Colab, ~40 GPU-hours estimated.

---

## 1. Goal

Build a **differentiable** version of the simulator (JAX backend) and use gradient-based optimization to solve the **inverse problem**: given Atanas 2023's wet-lab calcium recordings under anesthetic-equivalent conditions (or, if Atanas does not include direct anesthetic data, a published Calcium-imaging anesthesia dataset), find the per-target occupancy vector that best reproduces the observed neural dynamics.

The inverse-design occupancy vector is then compared against Phase C's structural-prediction occupancy. Agreement = independent confirmation of the structural-prediction approach. Disagreement = evidence that either Phase B/C miscalibration or the inverse problem is ill-posed at the data resolution available.

---

## 2. Background

### 2.1 Inverse problem statement

Forward problem (Phases A through H):

```
occupancy_vec  →  kinetic shifts  →  network simulation  →  V(t), Ca(t), behavior
```

Inverse problem (Phase I):

```
empirical V(t), Ca(t)  →  occupancy_vec
```

via:

```
minimize  ||simulator(occupancy_vec) − empirical_data||²
        over occupancy_vec ∈ [0, 1]^N_targets
```

The minimization requires gradients ∂loss/∂occupancy_vec, which requires a differentiable simulator.

### 2.2 JAX implementation strategy

Two options:

- **A: Manual JAX reimplementation** of LIF + channels + connectome. ~2 weeks of work. Clean, fully differentiable, full control. Limited to the channel set we manually translate.
- **B: Brian2-to-JAX bridge.** Use Brian2's existing simulation infrastructure but checkpoint intermediate states for JAX autodiff. More fragile; not all Brian2 features differentiable.

Phase I default: **option A, manual JAX reimplementation** of the essential set: LIF dynamics + 7 Wave 2 channels + Markov synapse + metabolic layer. The inverse problem is solved on this minimal differentiable simulator and validated against the full simulator from Phase G.

### 2.3 Empirical data source

The simulator's training data layer in Wave 2 already uses Atanas 2023 calcium recordings (`atanas_worm_*.npz` files in `scripts/brain/artifacts/`). Atanas 2023 includes WT recordings during locomotion. Wave P checks whether Atanas or a related dataset (Hallinen 2021, Yemini NeuroPAL) includes anesthetic-condition recordings. If yes, Phase I targets that data directly. If no, Phase I uses a published mammalian / non-Atanas anesthesia calcium dataset and maps it via cell-type homology.

### 2.4 Inverse-occupancy validation

Phase I produces `inverse_occupancy_vec` (length N_targets, range [0,1]). Compare to Phase C's `occupancy_central` matrix at the relevant anesthetic and dose:

- Spearman ρ between the two vectors.
- Per-target absolute difference; flag targets where Phase I says "high occupancy" but Phase C says "low" (or vice versa).
- Calibrated Kp scenario: re-run inverse with Kp set to half / double; assess whether disagreements track Kp uncertainty.

---

## 3. Method

### 3.1 JAX simulator skeleton

```python
# src/phase_i_inverse_jax.py
import jax
import jax.numpy as jnp
import optax

@jax.jit
def simulate(occupancy_vec, params, duration_steps):
    """Run differentiable simulator. Returns V(t), Ca(t)."""
    # LIF dynamics, channel currents, synapse currents, metabolic
    state = init_state()
    def step(state, t):
        kinetic_shifts = apply_occupancy(occupancy_vec, params)
        state = step_dynamics(state, kinetic_shifts)
        return state, (state.V, state.Ca)
    state, (V_traj, Ca_traj) = jax.lax.scan(step, state, jnp.arange(duration_steps))
    return V_traj, Ca_traj

@jax.jit
def loss(occupancy_vec, empirical_V, empirical_Ca, params):
    sim_V, sim_Ca = simulate(occupancy_vec, params, len(empirical_V))
    return jnp.mean((sim_V - empirical_V)**2) + jnp.mean((sim_Ca - empirical_Ca)**2)

grad_fn = jax.grad(loss)
optimizer = optax.adam(learning_rate=1e-3)
```

### 3.2 Optimization loop

```python
occupancy_vec = jnp.full((N_targets,), 0.1)  # init
opt_state = optimizer.init(occupancy_vec)
for iteration in range(2000):
    grads = grad_fn(occupancy_vec, empirical_V, empirical_Ca, params)
    updates, opt_state = optimizer.update(grads, opt_state)
    occupancy_vec = optax.apply_updates(occupancy_vec, updates)
    occupancy_vec = jnp.clip(occupancy_vec, 0.0, 1.0)
    if iteration % 100 == 0:
        print(f"iter {iteration}: loss = {loss(occupancy_vec, ...)}")
```

### 3.3 Validation against Phase G

After the inverse solver converges, plug `inverse_occupancy_vec` back into Phase G's full Brian2 simulator and check whether the network-level behavior matches Phase G's WT-halothane-1× run. Ideally, the inverse solver finds a near-Phase-C occupancy and Phase G reproduces. Disagreement = either (a) inverse is fitting noise; (b) Phase C is mis-calibrated; (c) the differentiable simulator's simplifications matter.

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| JAX manual reimplementation | local CPU | 80 (2 weeks engineering) | $0 |
| Smoke tests against Phase G | local CPU + GPU | 4 | $0 |
| Inverse optimization (2000 iter × 5 datasets) | local GPU | 20 | $0 |
| Cross-validation against Phase C | local CPU | 4 | $0 |
| **Total Phase I** | | **~110 hours** | **$0** |

Phase I is the most engineering-heavy phase but compute-light. The 2-week JAX reimplementation is the dominant cost.

---

## 5. Preregistered success criteria (Gate I.1)

1. **I.1.1 — Differentiable simulator validates against Phase G:** for a known occupancy_vec (e.g., the Phase C central values), the JAX simulator produces V(t), Ca(t) traces within 5% of Phase G's Brian2 simulator output.
2. **I.1.2 — Inverse converges:** loss decreases monotonically over 2000 iterations; final loss is at least 50× lower than initial.
3. **I.1.3 — Inverse-occupancy matches Phase C:** Spearman ρ between inverse and Phase C occupancy ≥ 0.5. Per-target absolute differences within ±0.3 occupancy on ≥ 80% of targets.
4. **I.1.4 — Identifiability sanity:** with synthetic data generated from a known occupancy, the inverse recovers the occupancy within ±0.1 on ≥ 80% of targets. (This is the identifiability test.)

---

## 6. Halting rules

**Pause and surface:**

- I.1.4 fails: the inverse problem is structurally non-identifiable on synthetic data. Document; do not proceed to real-data fits.
- I.1.1 fails: JAX simulator is not faithful to Brian2; inverse occupancy is meaningless. Re-implement.
- Loss does not decrease: optimization is broken; debug.

**Document and continue:**

- I.1.3 fails on a few targets: per-target identifiability issues; flag, do not block phase.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `src/phase_i_inverse_jax.py` | JAX differentiable simulator |
| `artifacts/runs/inverse_occupancy.npz` | Inferred occupancy vector |
| `artifacts/runs/inverse_validation.md` | Identifiability + Phase C comparison |
| `artifacts/runs/phase_i_completion.md` | end-of-block report |

---

## 8. Falsifiability checks

Premise: **"A differentiable JAX simulator can solve the inverse occupancy problem and produce an empirically grounded occupancy vector that converges with the structural-prediction occupancy."**

Falsified if:

- I.1.4 fails (non-identifiable).
- Inverse occupancy disagrees catastrophically with Phase C (Spearman ρ < 0.2) AND identifiability is fine — implies either Phase C or the inverse is wrong; deeper investigation needed.

---

## 9. Integration points

**Inputs:**

- Phase G `artifacts/runs/<config>.npz` — Brian2 simulation traces for cross-validation.
- Phase C `artifacts/occupancy/occupancy_matrix.npz` — comparison target.
- External: Atanas 2023 / Hallinen 2021 calcium recordings.

**Outputs:** documented as a methodological extension in the paper.

---

## 10. Citation hygiene declaration

- Atanas 2023 — (PMID lookup needed; reference is in `scripts/brain/`).
- Hallinen 2021 — (PMID lookup needed).
- Yemini NeuroPAL — (PMID lookup needed).
- JAX paper — (no canonical PMID; cite Bradbury 2018 or current).
- optax — (no PMID; cite as software).

---

## 11. Risk register (Phase I)

| Risk | Likelihood | Mitigation |
|---|---|---|
| 2-week JAX reimplementation runs over | High | Bound; defer Phase I if Wave P month 6 is closing |
| Identifiability fails (inverse non-unique) | High (small N_data, large N_targets) | Use regularization (sparsity prior); reduce target panel |
| JAX-Brian2 mismatch creates phantom solutions | Medium | I.1.1 cross-validation; iterate on JAX implementation |
| No empirical anesthesia calcium data exists | Medium | Use synthetic data with Phase C occupancy as ground truth; reframe as validation rather than discovery |

---

## 12. Phase I execution plan

(Activates only if Phase H ≥ 6/8.)

1. JAX reimplementation of essential simulator.
2. Cross-validation against Phase G Brian2.
3. Identifiability test on synthetic data.
4. Inverse fit on real data.
5. Phase C comparison.
6. Gate I.1 evaluation; end-of-block report.
