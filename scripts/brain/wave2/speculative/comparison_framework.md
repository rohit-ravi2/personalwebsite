# X.1c — Comparison framework for GNN dynamics validation

**Status:** Speculative architectural investigation. Inputs to morning review.

**Goal:** Specify what gates would govern accepting/rejecting a GNN-based cellular model, mapped onto the existing two-component Gate 2 framework so it integrates cleanly with the architectural plan's empirical methodology.

---

## 1. The existing two-component Gate 2 framework (recap)

From `phase_v_w2_architectural_plan.md` §"Gate 2: Cellular validation":

- **Component 2a — voltage-clamp trace correctness (channel kinetics):** Brian2 traces match Nicoletti's NEURON reference within 5% across IV-curve test set.
- **Component 2b — current-clamp plateau dynamics (architectural sufficiency):** Mellem-style plateau (15-25 mV / 400-800 ms / SLO-1-dominated termination on stim release).

The two components empirically distinguish channel-translation failure (2a fail) from architectural insufficiency (2a pass + 2b fail = condition 6).

A GNN must be evaluated against the same conceptual gates so its outputs are comparable to Path A and to the architectural-plan's morphology-fork outputs.

---

## 2. Component 2a-equivalent for GNN: per-channel kinetic correctness

If channel kinetics are mechanistic (Variants A and B), 2a is satisfied by construction — the gating equations are unchanged Brian2 eqs strings, already validated in Phase C/D/E. Per-channel rollback property is preserved.

If channel kinetics are learned (Variant C, or Variant B with learned gating), 2a requires:

- **Per-channel surrogate validation:** isolate the channel in the trained model (zero-out other channels' gbar), apply voltage-clamp, check that resulting current matches Nicoletti's NEURON reference within 5%.
- **Failure mode:** GNN's "channel" is a distributed representation across multiple latent dims; voltage-clamp probes the integrated cell. **Mechanistic interpretation may not survive.**

**Verdict:** Variant A passes 2a trivially. Variants B and C have non-trivial 2a interpretation work.

---

## 3. Component 2b-equivalent for GNN: architectural sufficiency

This is the load-bearing test — it is the failure surface that triggered Phase X in the first place.

### 2b primary test (preserved from architectural plan)

- AVAL with current-clamp protocol matching Mellem 2008.
- Plateau amplitude in 15-25 mV range.
- Plateau duration 400-800 ms.
- Termination: V settles to baseline ±5 mV by 1500 ms post-injection.
- Release behavior: collapse-on-release shows SLO-1-mediated termination dominates, not leak τ_m.

### 2b extensions specific to GNN

Because a GNN can in principle learn to fit a single trace (overfitting concern raised in X.1b §5):

- **Multi-protocol robustness:** the same trained GNN must reproduce voltage trajectories at 3-5 different injection currents (e.g., 10, 30, 50, 100 pA) and at multiple stimulus durations. Mellem-only fitting is rejected.
- **Out-of-distribution generalization:** train on injection currents in {30, 50, 100} pA, test on 10 and 200 pA. Loss above some threshold = overfitting flag.
- **Mechanistic perturbation correctness:** SLO-1 KO (gbar_slo1 → 0) should produce prolonged plateau (per Mellem's published manipulation). EGL-19 reduction should reduce plateau amplitude monotonically.

### 2b failure → action mapping

| 2a (channel) | 2b (architecture) | Diagnosis | Action |
|---|---|---|---|
| pass | pass | GNN cellular layer production-grade | Accept; consider for Wave 3 |
| pass | fail | GNN doesn't fix the gap that motivated it; same condition-6 signature as Path A | Reject GNN as a condition-6 response; consider X.2a or morphology fork |
| fail | fail | GNN broken at channel level | Variant suspect; rollback or Variant restructure |
| fail | pass | Channel kinetics distorted but compositional dynamics happen to fit | Suspect overfitting; reject |

---

## 4. Distributional comparison gates (additional GNN-specific)

A GNN producing voltage trajectories has more degrees of freedom in failure mode than a Brian2 cell — it can fit on average but mismatch in distributional ways. So beyond pointwise voltage-feature comparison:

### 4.1 MMD on trace ensembles

- Run trained GNN at, say, 50 different injection currents.
- Run Brian2 (or NEURON multi-compartment) ground truth at the same 50 currents.
- Compute Maximum Mean Discrepancy (MMD) between the two ensembles in voltage-feature space.
- Pass: MMD < 0.05 (or comparable to MMD between two NEURON runs with different RNG).

### 4.2 Wasserstein distance on plateau-amplitude distribution

- For 100 randomized injection-current parameters, GNN plateau amplitudes form distribution P_GNN; ground truth forms P_GT.
- W_2(P_GNN, P_GT) < threshold = pass.

### 4.3 Trajectory similarity (DTW / soft-DTW)

- Per-trace DTW alignment cost between GNN trajectory and ground truth, normalized by trace length.
- Threshold: median < some calibrated value, 95th percentile < 2× median.

These are useful as **additional diagnostic gates** but not load-bearing for Gate 2 acceptance — voltage-feature comparison + multi-protocol robustness is sufficient. DTW etc. are reported as warn-only diagnostics.

---

## 5. Mellem-style architectural-sufficiency test (the 2b core)

The primary test is **identical to the architectural plan's 2b**:

```
def gate_2b_mellem(model, geometry='AVAL'):
    """Gate 2b: architectural sufficiency."""
    # Mellem 2008 protocol
    settle_ms = 200
    inject_ms = 100
    inject_pA = 50
    recover_ms = 1500
    
    # Run model
    t, V = model.run_current_clamp(
        settle_ms=settle_ms,
        inject_ms=inject_ms,
        inject_pA=inject_pA,
        recover_ms=recover_ms,
        geometry=geometry,
    )
    
    # Extract features
    v_rest = V[t < settle_ms].mean()
    plateau_window = (t >= settle_ms) & (t < settle_ms + inject_ms)
    plateau_V = V[plateau_window]
    plateau_amp = plateau_V.max() - v_rest
    
    # Plateau duration: time from stim onset to V returning to v_rest + 5 mV
    post_stim = (t >= settle_ms + inject_ms)
    settle_idx = np.argmax(V[post_stim] - v_rest < 5.0) if any(...) else len(V[post_stim])
    plateau_duration_ms = ... # implementation per existing plateau_harness.py
    
    # Pass conditions
    amp_pass = 15 <= plateau_amp <= 25
    dur_pass = 400 <= plateau_duration_ms <= 800
    settle_pass = abs(V[-1] - v_rest) < 5.0
    
    return {'amp_pass': amp_pass, 'dur_pass': dur_pass, 'settle_pass': settle_pass}
```

(The existing `plateau_harness.py` implements this — a GNN-aware version would just replace the Brian2 call with a GNN forward pass.)

---

## 6. Per-channel ablation correctness (mechanistic survival test)

Because Variant A retains mechanistic channel kinetics, ablation tests should preserve mechanistic predictions:

```
# Ablation: SLO-1 KO
model_KO = copy(model)
model_KO.gbar_slo1 = 0
result_KO = gate_2b_mellem(model_KO)

# Expected (per Mellem 2008): prolonged plateau (slo-1 KO removes termination)
# Pass: plateau_duration_KO > plateau_duration_WT × 2
```

This is **the test that Variant A passes by construction** but Variant C would have to learn to satisfy. It is a key falsifiability anchor for any GNN claim.

---

## 7. Wave 1 — Wave 2 — Wave 3 gate alignment

Mapping these proposed GNN gates onto the existing Wave 2 gate hierarchy:

| Wave 2 gate | GNN Variant A equivalent | Falsifiability signal |
|---|---|---|
| Gate 1 (per-channel translation) | Per-channel Brian2 modules unchanged; GNN nodes use them as-is | 5% trace divergence vs Nicoletti |
| Gate 2a (kinetic correctness in cell context) | GNN node's per-channel current under voltage-clamp matches Nicoletti | Same as Path A |
| Gate 2b (architectural sufficiency) | GNN AVAL produces Mellem plateau (15-25 mV / 400-800 ms) | New: load-bearing for GNN architectural claim |
| Gate 2b-ablation | KO simulations match Mellem published manipulations | New: mechanistic survival test |
| Gate 2b-multiprotocol | Multi-current-injection robustness | New: overfitting check |
| Gate 3 (network preservation) | Existing scenario sweep with GNN-cellular layer reproduces baselines | Same as Path A |

---

## 8. Quantitative thresholds (proposed)

| Test | Threshold |
|---|---|
| 2a per-channel divergence | ≤ 5% (matches Path A) |
| 2b plateau amplitude | 15-25 mV |
| 2b plateau duration | 400-800 ms |
| 2b settle | abs(V_final - V_rest) < 5 mV |
| 2b multi-protocol robustness (5 protocols) | ≥ 4 / 5 within tolerance |
| 2b OOD (train: 30/50/100 pA, test: 10/200 pA) | trajectory MAE < 5 mV |
| 2b ablation (SLO-1 KO, EGL-19 reduction) | direction and magnitude match Mellem published |
| MMD on ensemble | < 0.05 (or < ground-truth-vs-ground-truth MMD × 2) |

---

## 9. Validation runbook (sketch)

If a GNN model is submitted for Wave 3 acceptance:

1. **Smoke test** — model loads, forward pass at known input gives sensible output shape.
2. **Gate 2a** — per-channel surrogate validation. Reject if any channel deviates > 5% from Nicoletti.
3. **Gate 2b primary** — Mellem plateau test. If FAIL, reject (no condition 6 distinction since GNN was meant to fix that).
4. **Gate 2b multi-protocol** — 5 injection currents, 5 stimulus durations.
5. **Gate 2b ablation** — SLO-1 KO, EGL-19 KO/reduction.
6. **Gate 2b OOD** — held-out injection currents.
7. **Distributional gates (warn-only)** — MMD, Wasserstein, DTW ensemble metrics.
8. **Gate 3** — network preservation when GNN cellular layer is dropped into the existing scenario pipeline.

If all gates pass, the GNN is acceptable as a cellular layer. If gates 2b or 3 fail, GNN is rejected as an alternative to morphology fork.

---

## 10. Summary

| Question | Answer |
|---|---|
| Can the existing Gate 2 framework accommodate GNN validation? | **Yes**, with modest extensions. |
| What new tests are needed? | Per-channel ablation correctness (mechanistic survival); multi-protocol robustness; OOD generalization. |
| What distributional metrics help? | MMD on ensembles, Wasserstein on amplitude distributions, DTW for trajectory similarity — all warn-only. |
| Is the framework strict enough to detect overfitting? | Yes if multi-protocol + OOD + ablation tests are enforced. Mellem-only fitting is detectable. |
| Does Variant A pass 2a by construction? | Yes (mechanistic channels preserved). |
| Does Variant C pass 2a by construction? | No (channels are distributed in latent representation). |

**Net comparison-framework recommendation:** the existing two-component Gate 2 generalizes cleanly to GNN evaluation. **The mechanistic-channel ablation test (SLO-1 KO produces prolonged plateau) is the load-bearing falsifiability anchor** — it's the test that makes Variant A defensible and Variant C suspicious.
