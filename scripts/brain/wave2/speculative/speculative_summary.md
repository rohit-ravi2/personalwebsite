# Speculative architecture summary — entry point for morning review

**Status:** Wave 2 speculative-architecture work block complete. Outputs are inputs to morning review of Wave 3 architectural roadmap and to condition-6 response selection (if density-sensitivity work confirms it).

**Scope:** Three speculative paths characterized — GNN hybrid (X.1), multi-compartment-explicit (X.2a), NeuroML2-native (X.2b) — and compared against the architectural plan's recommended condition-6 response (morphology fork).

**Top-line recommendation:** **Architectural plan's morphology fork (= X.2a) remains the strongest condition-6 response.** GNN (X.1 Variant A) is a plausible Wave 4 enhancement; Variants B/C are not currently feasible. NeuroML2-native (X.2b) is rejected as it was previously.

---

## 1. One-paragraph framing

The Phase F 2b failure (single-compartment AVA Brian2 produces 46.8 mV / 21.4 ms vs Mellem 15-25 mV / 400-800 ms targets) surfaced condition 6 (architectural insufficiency). The architectural plan's pre-committed response is the **morphology fork** — integrate c302's NeuroML2 multi-compartment cell descriptions into the project's Brian2 scaffold. This speculative-architecture investigation, run in parallel with the density-sensitivity analysis subagent, characterized three alternative responses to compare against. None of them displaces the morphology fork as primary; all three are documented for completeness and future Wave 3+ use.

---

## 2. Comparison matrix

| Aspect | X.1 GNN Variant A | X.2a Multi-compartment-explicit | X.2b NeuroML2-native | Morphology fork (arch plan) |
|---|---|---|---|---|
| Backend | Brian2 + PyTorch | Brian2 SpatialNeuron | jNeuroML/pyNeuroML → NEURON | Brian2 SpatialNeuron |
| Effort | ~2-3 months for production-grade | ~3-4 weeks | ~3 months | ~3-4 weeks |
| Channel kinetics | Mechanistic (preserved from Phase β) | Mechanistic (preserved) | Translated to ChannelML (more work) | Mechanistic (preserved) |
| Spatial coupling | Learnable axial conductance | Analytical from morphology | Native NeuroML2 | Analytical from morphology |
| Per-segment densities | Learnable | Hand-tuned by group, optimized | Hand-tuned by group | Hand-tuned by group |
| Mechanistic interpretability | Full (Variant A specifically) | Full | Full | Full |
| Compute on RTX 4060 Ti | Acceptable (tested at 2-node) | Acceptable | Slow (NEURON-backend) | Acceptable |
| Wins existing Wave 1 + Phase β investment | Yes | Yes | No (would discard channel translation) | Yes |
| Addresses condition 6 (Mellem 2b failure) | Plausibly | Plausibly (architectural plan's primary hypothesis) | No (same morphology work needed regardless of backend) | Plausibly |
| Distinguishing value-add | Per-segment density fitting | Standard multi-compartment | Community / OpenWorm collaboration | Standard multi-compartment |
| X.2a relationship | X.1 Variant A is X.2a + learned coupling/densities | (this is the primary) | Same architecture, different backend | Identical |
| Validation framework | Extended Gate 2 (mechanistic ablation tests) | Standard Gate 2 | Standard Gate 2 | Standard Gate 2 |
| Risk if pursued | Training dynamics non-trivial; per-segment density data sparse | Per-segment density data sparse | Performance penalty, infrastructure cost | Per-segment density data sparse |
| Prototype outcome | Bounded prototype FAIL (52.9 → 46.2 mV, threshold 5 mV) | No prototype attempted | No prototype attempted | (already implicitly endorsed by arch plan) |

---

## 3. Trigger conditions per speculative path

### X.1 GNN Variant A becomes attractive when:

- X.2a (morphology fork) has been executed and Gate 2b fails OR Mellem-style dynamics are not fully reproduced.
- AND data-driven per-segment density fitting is needed (Wave 4 territory anyway).
- AND CeNGEN-coupling work surfaces tractable density priors that constrain Variant A's training.

If those preconditions are met, Variant A becomes a Wave 4 enhancement layer on top of X.2a, focused on density fitting rather than learned dynamics.

### X.1 Variants B / C become attractive when:

- Project pivots to fully-learned cellular dynamics research direction (not currently planned).
- AND large biological calcium imaging datasets become available beyond Atanas (e.g., Hallinen, Yemini, future work).
- AND mechanistic interpretability is no longer a paper-3-style requirement.

This is Wave 5+ speculation, not actionable now.

### X.2a Multi-compartment-explicit becomes primary when:

- Density-sensitivity analysis confirms condition 6 (architectural insufficiency, not parameter-tunable).
- This is the architectural plan's pre-committed condition-6 response and remains the default.

### X.2b NeuroML2-native becomes attractive when:

- Project pivots to OpenWorm-community-substrate as a primary integration goal.
- OR NetPyNE-based anesthesia-mechanism collaboration becomes a research focus.
- Neither is currently planned; X.2b stays in the rejected-but-documented bucket.

### Morphology fork (arch plan) is recommended when:

- Density-sensitivity analysis confirms condition 6 (the primary trigger).
- Same conditions as X.2a (morphology fork ≡ X.2a, just framed differently).

---

## 4. Cross-path observations

### 4.1 X.2a and morphology fork are essentially identical

The architectural plan's morphology fork **is** X.2a in different words. This investigation's X.2a sketch makes the morphology fork concrete enough to compare apples-to-apples against speculative variants, but doesn't propose a different architecture. So the speculative investigation **confirms** the architectural plan's recommended condition-6 response rather than challenging it.

### 4.2 GNN Variant A's prototype failure is informative but not load-bearing

The 2-node prototype's 46 mV final test MAE (vs <5 mV target) shows that:
- The training pipeline is functional (autograd through unrolled time loop, gradient flow, parameter updates).
- A 2-node mechanistic-anchored model with finite axial coupling does not trivially reduce to single-compartment dynamics.
- Bounded effort was insufficient to determine whether more careful setup would converge or whether the architecture is structurally inefficient at small scales.

This is consistent with **"Variant A is interesting at scale but small prototypes don't validate it"** — a finding to report, not a load-bearing roadblock for any path.

### 4.3 NeuroML2-native is consistently rejected

X.2b's analysis confirms the architectural plan's prior rejection. Path 3D's value proposition under condition 6 is weakest — same morphology work needed, but with worse backend performance and higher infrastructure cost.

### 4.4 The four paths are not mutually exclusive in the long run

- Wave 2 commits to Path 3A (Brian2 + parameter import). Done.
- Wave 2 condition-6 response = morphology fork = X.2a.
- Wave 3 may run X.2a to completion.
- Wave 4 may layer X.1 Variant A on top of X.2a for density fitting + CeNGEN coupling.
- Wave 5+ may revisit X.2b if community-collaboration goals shift.

This sequencing is **additive, not exclusive**, consistent with the architectural plan's "no future wave's success requires reverting earlier waves."

---

## 5. Key load-bearing conclusions

1. **The architectural plan's morphology fork (X.2a) is the right primary response to condition 6.** Speculative investigation does not displace it.

2. **GNN Variant A is plausible as a Wave 4 enhancement, not a Wave 2 alternative.** The prototype's bounded outcome is consistent with this read.

3. **GNN Variants B and C are not currently feasible** (data-starved regime); these are Wave 5+ research directions if at all.

4. **NeuroML2-native (X.2b) remains rejected** for Wave 2/3 work; revisit only if community/collaboration goals shift.

5. **All Variant A work, if pursued, should be mechanistic-anchored** (preserves Phase β channel translation investment + paper-3 mechanistic claims).

6. **Per-segment channel density data is sparse in C. elegans literature.** This is a load-bearing risk shared by X.2a, X.1 Variant A, and morphology fork — not unique to any one. Wave 4 CeNGEN-coupling work + parameter-fitting methodology (à la Nicoletti's `g_to_Scm2` workflow) is the unblocking path for all three.

---

## 6. Decision-grade summary for morning review

If density-sensitivity analysis confirms condition 6:
> **Recommend morphology fork = X.2a as primary response. Bounded ~3-4 weeks for Phase β-morph + Phase γ-morph per architectural plan. Speculative X.1 Variant A and X.2b documented as alternatives but not recommended for Wave 2.**

If density-sensitivity analysis returns DENSITY_TUNABLE:
> **Continue Path 3A as planned (Phase γ → Phase δ). X.1, X.2a, X.2b remain documented-but-unactioned long-term-roadmap material for Wave 3+.**

Either way, **the GNN prototype's failure is reportable and informative but does not require any architectural action right now.**

---

## 7. Files in this work block

```
wave2/speculative/
├── gnn_architecture_sketch.md           # X.1a — Variant A/B/C with ASCII diagrams
├── training_data_feasibility.md          # X.1b — data + parameter-count analysis per variant
├── comparison_framework.md               # X.1c — Gate 2 extension for GNN validation
├── prototype/                            # X.1d — minimal Variant A PyTorch prototype
│   ├── data.py                            # synthetic single-compartment ground truth
│   ├── gnn_prototype.py                   # 2-node TwoNodeGNN module
│   ├── train.py                           # training driver
│   ├── train.log                          # ~794s training log
│   ├── train_data.npz                     # 64 traces × 8000 steps
│   ├── results.json                       # final metrics + history
│   ├── trained_model.pt                   # final state dict
│   └── README.md                          # prototype outcome (FAIL, informative-negative)
├── x1_summary.md                         # X.1 GNN summary
├── multi_compartment_explicit.md         # X.2a — sketch of arch plan's morphology fork
├── neuroml2_native.md                    # X.2b — sketch + previous-rejection confirmation
└── speculative_summary.md                # This file (entry point for morning review)
```

No `SPECULATIVE_PAUSE.txt` was created — no load-bearing concern surfaced during the investigation that warranted halting.
