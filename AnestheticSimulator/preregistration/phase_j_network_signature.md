# Phase J — Network signature analysis (stretch)

**Phase letter:** J
**Status:** SCAFFOLDED, DEFERRED. Activates only if Phase H ≥ 4/8.
**Predecessor:** Phase H verdict; Phase G traces.
**Successor:** none (deliverable feeds the paper as a network-theoretic extension).
**Compute:** local CPU, ~20 hours.

---

## 1. Goal

Characterize the **network-level signatures** of anesthesia in the simulator, comparing pre- and post-anesthetic states across multiple complementary metrics:

- **Φ (integrated information)** via PyPhi, on a reduced subnetwork (full network's state space is too large for PyPhi's exponential cost).
- **Lyapunov spectrum** approximated via numerical perturbation, characterizing dynamical-system stability.
- **Modularity** (Newman) on the effective functional connectivity matrix.
- **Spectral entropy** of population firing rates.
- **Manifold embedding** (UMAP / t-SNE / diffusion maps) of network state-space trajectories.

Cross-validate the simulator's signatures against published mammalian EEG / fMRI anesthesia signatures: reduced Φ, increased modularity, decreased complexity, altered spectral content. Agreement = the simulator captures the network-level phenomenology of anesthesia, even if specific molecular details differ.

---

## 2. Background

### 2.1 Mammalian anesthesia signatures (literature)

| Signature | Direction under anesthesia | Mammalian source |
|---|---|---|
| Φ (integrated information) | Decreased | Casali 2013 *Sci Transl Med*; Tononi 2016 review |
| EEG modularity | Increased (network "fragments") | Lewis 2012 *PNAS* |
| Spectral complexity (Lempel-Ziv) | Decreased | Schartner 2015 *PLOS ONE* |
| Spectral power | Shifted to low frequencies (delta) | classical |
| Effective connectivity | Decreased long-range | Boly 2012 |

Wave P tests whether the *C. elegans* simulator under anesthetic shows the same directional signatures. *C. elegans* is small enough that Φ may be computable on subsets that mammalian brains preclude.

### 2.2 Φ computability constraint

PyPhi's complexity is exponential in network size (bounded by 2^N for N nodes). For full 300-cell network, PyPhi is infeasible. Wave P uses:

- **Subnetwork analysis:** select the command-neuron subset (AVA, AVB, AVD, AVE, PVC) — 5 cells, tractable.
- **Hierarchical Φ:** approximate Φ on overlapping 5-cell subsets and aggregate.

### 2.3 Lyapunov spectrum

For LIF networks, exact Lyapunov exponents are subtle (LIF dynamics are not smooth). Wave P uses **numerical perturbation Lyapunov**: perturb V(t) by ε at time t_0, integrate forward, measure ||δV(t)|| growth rate. Average over multiple perturbation seeds. This is the largest Lyapunov exponent (LLE).

Under anesthetic:

- LLE should decrease (dynamics become more dissipative; network is closer to fixed-point attractor).
- Lyapunov spectrum (top-K) should shift downward.

### 2.4 Manifold embedding

State-space trajectory: at each time step, the network state is a 300-dimensional vector V(t). Trajectory over 60 s × 1000 Hz = 60,000 points in 300-D. Reduce to 2-3D via UMAP / diffusion maps. Compare WT pre- and post-anesthetic embeddings:

- WT pre-anesthetic: trajectory explores extended manifold (multiple FSM states, complex transitions).
- WT post-anesthetic at clinical EC50: trajectory contracts to a smaller region (fewer states, less complexity).

This is the visual / qualitative version of the complexity-decrease finding.

---

## 3. Method

### 3.1 PyPhi on command-neuron subset

```python
# src/phase_j_signature.py --phi
import pyphi

# Reduce trace to 5 command neurons; binarize firing rate (above/below median)
binarized = (firing_rate[command_neurons, :] > median).astype(int)

# Empirical TPM from binarized data
tpm = compute_empirical_tpm(binarized)

# PyPhi system
network = pyphi.Network(tpm)
state = tuple(binarized[:, t_sample])
subsystem = pyphi.Subsystem(network, state, range(5))
phi = pyphi.compute.phi(subsystem)
```

Repeat across pre- and post-anesthetic conditions; bootstrap over time samples; compute mean Φ ± std.

### 3.2 Lyapunov via perturbation

```python
def lyapunov_perturbation(V_baseline, dt, eps=1e-6, T_reset=10):
    V_perturbed = V_baseline + eps * random_unit_vector()
    delta = []
    for t in range(T_reset):
        V_baseline_next = step_simulator(V_baseline, dt)
        V_perturbed_next = step_simulator(V_perturbed, dt)
        delta.append(jnp.linalg.norm(V_perturbed_next - V_baseline_next))
        V_baseline, V_perturbed = V_baseline_next, V_perturbed_next
    return jnp.log(jnp.mean(delta[-3:]) / eps) / (T_reset * dt)
```

### 3.3 Modularity

Use the time-averaged effective connectivity matrix (cross-correlation of V traces, thresholded):

```python
W_eff = abs(np.corrcoef(V_traces)) > 0.3
modularity_pre = nx.community.modularity(nx.from_numpy_array(W_eff_pre), partition)
modularity_post = nx.community.modularity(nx.from_numpy_array(W_eff_post), partition)
```

### 3.4 Spectral entropy

Welch power spectrum of population firing rate; entropy of normalized power distribution.

### 3.5 Manifold embedding

```python
import umap
embedding_pre = umap.UMAP(n_components=2).fit_transform(V_traces_pre.T)
embedding_post = umap.UMAP(n_components=2).fit_transform(V_traces_post.T)
```

---

## 4. Compute budget

| Sub-task | Resource | Hours | Cost |
|---|---|---|---|
| PyPhi on 5-cell subnets | local CPU | 4 | $0 |
| Lyapunov perturbation | local CPU | 6 | $0 |
| Modularity + spectral entropy | local CPU | 2 | $0 |
| Manifold embeddings | local CPU | 3 | $0 |
| Comparison + writeup | local CPU | 5 | $0 |
| **Total Phase J** | | **~20 hours** | **$0** |

---

## 5. Preregistered success criteria (Gate J.1)

1. **J.1.1 — Φ direction:** Φ on command-neuron subset decreases under WT-halothane-1×-EC50 vs WT-no-anesthetic, paired t-test across seeds, p < 0.05.
2. **J.1.2 — Lyapunov direction:** LLE decreases under anesthetic, paired t-test p < 0.05.
3. **J.1.3 — Modularity direction:** Modularity increases under anesthetic, paired t-test p < 0.05.
4. **J.1.4 — Manifold contraction:** UMAP-embedded state-space variance under anesthetic is < 50% of variance pre-anesthetic.

J.1.1-J.1.4 do not need to all pass; the paper reports each independently.

---

## 6. Halting rules

**Document and continue:** Phase J is exploratory; failures are reported, not gated.

---

## 7. Output deliverables

| File | Contents |
|---|---|
| `artifacts/runs/signatures.npz` | Per-config Φ, LLE, modularity, spectral entropy |
| `artifacts/runs/manifold_embeddings.npz` | UMAP coordinates pre/post anesthetic |
| `artifacts/runs/signature_report.md` | Direction-of-change table; mammalian comparison |
| `artifacts/runs/phase_j_completion.md` | end-of-block report |

---

## 8. Falsifiability checks

Premise: **"The simulator under anesthetic exhibits network-level signatures (decreased Φ, decreased Lyapunov, increased modularity, manifold contraction) consistent with mammalian anesthesia signatures."**

Falsified if: all 4 signatures show the wrong direction. Three of four wrong = inconsistent with mammalian phenomenology, requires explanation.

---

## 9. Integration points

**Inputs:** Phase G traces.

**Outputs:** paper section on network signatures.

---

## 10. Citation hygiene declaration

- Casali 2013, perturbational complexity, *Sci Transl Med* — (PMID lookup needed).
- Tononi 2016, IIT review, *Nat Rev Neurosci* — (PMID lookup needed).
- Lewis 2012, *PNAS* — (PMID lookup needed).
- Schartner 2015, *PLOS ONE* — (PMID lookup needed).
- Boly 2012 — (PMID lookup needed).
- PyPhi — Mayner 2018, *PLOS Comp Bio* — (PMID lookup needed).
- UMAP — McInnes 2018, arXiv 1802.03426.

---

## 11. Risk register (Phase J)

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyPhi too slow even on 5-cell subset | Medium | Use approximation; reduce to 4 cells |
| Lyapunov perturbation unstable | Medium | Multiple eps; document the eps-dependence |
| Modularity partition is arbitrary | Medium | Use Louvain or Leiden; report multiple |
| All 4 signatures fail | Low | Document; explain why simulator differs from mammalian phenomenology |

---

## 12. Phase J execution plan

(Activates only if Phase H ≥ 4/8.)

1. PyPhi on command-neuron subset, pre/post anesthetic.
2. Lyapunov perturbation on full network.
3. Modularity + spectral entropy on effective connectivity.
4. UMAP embeddings.
5. Direction-of-change comparison against mammalian.
6. End-of-block report.
