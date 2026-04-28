# X.1 — GNN hybrid investigation summary

**Status:** Speculative architectural investigation, completed bounded prototype.

**Position:** Inputs to morning review of Wave 3 architectural roadmap; alternative for comparison if density-sensitivity work confirms condition 6.

---

## 1. Verdict per investigation axis

| Axis | Verdict | Detail |
|---|---|---|
| Architectural feasibility | **Yes — Variant A specifically** | See `gnn_architecture_sketch.md`. Variant A (mechanistic-anchored, learned coupling) is conceptually sound and reduces to multi-compartment-explicit (X.2a) when learned coupling is replaced by analytic axial Ra. Variants B (learned membrane integration) and C (fully learned) are speculative beyond Wave 3 horizon. |
| Training data feasibility | **Sufficient at prototype scale; marginal at single-cell scale; data-starved at full-worm scale** | Per `training_data_feasibility.md`. Simulator-self-distillation training data is unlimited. Biological data (Mellem 2008, Nicoletti 2024 panels) is dozens of panels, sufficient for single-trace fitting (with overfitting risk) but not for full-cell density-fitting without strong CeNGEN priors. |
| Comparison framework feasibility | **Yes** | Per `comparison_framework.md`. Existing Gate 2 generalizes cleanly with mechanistic-channel ablation tests as the load-bearing falsifiability anchor (SLO-1 KO → prolonged plateau). |
| Prototype outcome | **Failed (informative-negative)** | Per `prototype/README.md`. 2-node Variant A with 4 trainable params reduced test MAE from 52.9 → 46.2 mV (13%) over 40 epochs; did not reach the < 5 mV pass threshold. Training pipeline works; bounded effort exhausted before convergence to useful regime. |

---

## 2. Recommendation

**Worth further investigation BUT NOT as a primary condition-6 response.**

The honest framing is:

- **As a primary alternative to the morphology fork (X.2a) under condition 6:** **NO.** Variant A is a special case of X.2a with optional learned augmentation. There's no architectural value-add over X.2a that justifies the additional Variant-A engineering until X.2a has been executed and its limits are known.

- **As a Wave 4 enhancement on top of an X.2a foundation:** **YES, plausible.** Variant A's distinguishing capability is data-driven per-segment density fitting. Per-segment channel densities are sparse in C. elegans literature, so Wave 4-style density-fitting (which the project will do regardless of Wave 2 architectural commitment) is the natural place for Variant A. At that point, learnable per-segment gbars + learnable axial coupling become tools for unifying the density-fitting workflow.

- **As a Wave 5+ research direction (Variants B / C):** **NO, not now.** Data-starved at the scales they imply; Variant C in particular is competing-project territory (BAAIWorm `eworm_learn`).

The prototype's failure is consistent with both "more careful setup needed" and "structural inefficiency at small graph sizes." It does not falsify Variant A; it does not validate it either. The bounded effort was insufficient for definitive verdict.

---

## 3. Wave 3 implications if GNN approach is adopted

If a future Wave 3+ work block elects to pursue GNN:

1. **Variant A as the only candidate.** Variants B and C are not currently feasible.
2. **Layered on top of X.2a, not in place of it.** The morphology + axial-coupling structure comes from c302 NeuroML2 + Brian2 SpatialNeuron (X.2a foundation). The GNN extension is added per-segment density learning.
3. **Training data strategy:** simulator-self-distillation (Brian2 multi-compartment) for prototype; Atanas / Hallinen calcium imaging for cross-cell generalization at Wave 4+.
4. **Validation: extended Gate 2.** Standard 2a/2b plus mechanistic-channel-ablation tests, multi-protocol robustness, OOD generalization. Mellem-only fitting is rejected.
5. **Compute scale:** single-cell training tractable on RTX 4060 Ti. Full-worm training may need cloud or longer training time.
6. **Backwards compatibility:** X.2a = Variant A with axial coupling fixed and densities hand-tuned. So adopting Variant A doesn't invalidate X.2a investment.

---

## 4. What the prototype actually tested vs what would be needed for definitive verdict

| What was tested | What's needed |
|---|---|
| 2-node graph | 57-segment AVAL graph (per c302 morphology) |
| EGL-19 + leak only | Full 7-channel essential set + Ca-pool |
| Synthetic NumPy ground truth | Brian2-validated multi-compartment ground truth (or NEURON multi-compartment via Nicoletti's setup) |
| 4 trainable params | 100s of trainable params (per-segment gbars + axial Ra + Ca diffusion) |
| Single-compartment target | Multi-compartment + Mellem plateau target |
| MSE on V trajectory | MSE + voltage-feature loss + Mellem-feature gates |
| 40 epochs, 1 random seed | Multiple seeds + early stopping + hyperparameter sweep |
| ~13 min training | Hours-to-days of focused training |

**Bounded prototype exhausted at one or two of these axes.** A Wave 3 work block scoping ~1-2 weeks of focused effort would address the rest.

---

## 5. Prototype-specific findings (worth retaining)

1. **Mechanistic anchoring + autograd through unrolled time loop is functional.** PyTorch handles ~1600 unrolled steps × 51 traces × 4 trainable params × 40 epochs in ~13 min on RTX 4060 Ti. Doesn't OOM. Doesn't NaN at dt = 0.125 ms.

2. **Forward-Euler stability constrains dt < ~0.2 ms.** EGL-19's m-gate has tau as small as 0.06 ms in the published parameterization; explicit Euler at dt > 0.2 ms is unstable. Brian2's `exponential_euler` would handle this without dt restriction; PyTorch implementations would benefit from semi-implicit integration. **Implementation note for future Variant A work.**

3. **At 4 trainable params with random injection-current data, gradient signal favors gbar reduction over axial coupling tightening.** This is a training-dynamics finding: the model finds easier local minima by suppressing channel currents than by tightening axial coupling. **Suggests biology-anchored param priors (e.g., L2 to AVAL g0) are needed to keep gbars near biological values.**

4. **Test MAE 46 mV is far from acceptable.** This is not a near-miss; it's an order of magnitude off. Indicates the prototype's 2-node geometry + small param set is structurally insufficient, not just under-trained.

5. **A useful next step (for any future investigation) is to fix gbars and only train axial Ra.** This isolates whether axial coupling alone can recover single-compartment dynamics. (Skipped here per bounded effort.)

---

## 6. What this means for morning review

The morning-review framing should be:

> "GNN Variant A is conceptually sound, training-data-feasible at prototype scale, validation-gateable via existing Gate 2 framework, but the bounded prototype did not pass its own sanity check. Variant A is **not** the recommended condition-6 response — the architectural plan's morphology fork (X.2a) remains primary. Variant A is a plausible Wave 4 enhancement (per-segment density fitting on top of multi-compartment foundation) but not a Wave 2 alternative."

If the parallel density-sensitivity work confirms condition 6 (architectural insufficiency, not parameter-tunable), the recommended response is X.2a (morphology fork) per architectural plan, NOT GNN Variant A. The GNN work here is **a characterized alternative for documentation**, not a competing primary recommendation.

If the parallel density-sensitivity work returns DENSITY_TUNABLE (condition 6 false alarm), this GNN work is Wave 3+ long-term roadmap material, not immediate-impact.

---

## 7. Notes on Variant choice and risk posture

**Variant A is the lowest-risk variant** in the sense that mechanistic-channel anchoring preserves all of Wave 2's investment in channel translation (Phase β's 7 channels) and all of paper 3's mechanistic-claims foundation. Adopting Variant A does not require abandoning Path 3A.

**Variants B and C are higher-risk** in the sense of breaking mechanistic interpretability and requiring fundamentally larger training-data regimes than the project currently has access to. They become viable only if the project's research direction shifts to research questions that learned latent dynamics can address better than mechanistic models — which is not the current focus.

The honest read: **GNN work, if it happens, should be Variant A only, layered on top of X.2a, with carefully scoped Wave 4 effort.** This investigation's prototype is consistent with that read but does not establish it definitively.
