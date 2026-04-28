# X.1b — Training data feasibility for GNN hybrid

**Status:** Speculative architectural investigation. Inputs to morning review of Wave 3 architectural roadmap.

**Companion to:** `gnn_architecture_sketch.md` (Variant A is the lead candidate). This document evaluates whether sufficient training data exists to fit each variant's trainable parameter set.

---

## 1. Available data inventory

### 1.1 Brian2 + NEURON simulator-generated data (the cleanest source)

From run #1 + run #2 Phase β infrastructure:

| Source | Coverage | Volume |
|---|---|---|
| EGL-19 voltage-clamp validation (`egl19_validation_results.json`) | 11 holding potentials × ~100ms each, dt=25µs ≈ 4000 samples/trace, V/I traces | ~44k samples |
| SLO-1 isolated voltage-clamp (`slo1iso_validation_results.json`) | 4 cai × 11 holds | ~176k samples |
| SLO-1+EGL-19 coupled (`slo1egl19_validation_results.json`) | 11 holds | ~44k samples |
| SHK-1, SHL-1, NCA, KQT-3 (Phase C) | 11 holds each | ~176k samples total |
| Phase F gate2 plateau trace (`phase_f_gate2_results.json`) | single failed plateau trace at AVA single-compartment | ~70k samples |
| EGL-19 isolated regen (re-runnable via `validate_egl19.py`) | re-runnable for any voltage-clamp protocol or current-clamp protocol against AVAL geometry | unbounded by re-run |

**Implication:** for Variant A's prototype-scale training (2-node EGL-19+leak), simulator-generated training data is essentially unlimited. Re-running `validate_egl19.py` with diverse holding potentials and injection currents is cheap (seconds per trace).

### 1.2 Cellular electrophysiology (Mellem 2008 + Nicoletti 2024 + Mellem-cited recordings)

| Source | Coverage | Volume |
|---|---|---|
| Mellem 2008 published plateau trace | 1 representative trace (digitized at `published_traces.json`/`published_traces_v2.json`) | ~hundreds of points |
| Nicoletti 2024 published voltage-clamp panels (per published cell) | ~5 cells × ~10 panels each | ~50 panels |
| Nicoletti 2019 AWCon/RMD published traces | ~10 panels | ~10 panels |
| Mellem 2008 supplementary (if accessible) | unknown — would need to acquire | unknown |

**Implication:** for full architectural validation against published biology, we have **dozens of validated panels, not thousands**. This is the data-starved regime for any model with substantial parameter count.

### 1.3 In vivo calcium imaging (Atanas 2023 etc.)

| Source | Coverage | Volume |
|---|---|---|
| Atanas 2023 worm 01-10 calcium traces (locally cached in `artifacts/atanas_worm_*.npz`) | 10 worms × ~100s × multiple cells × ~10Hz sampling | ~10k samples × cells per worm |
| Hallinen 2021 calcium imaging | not yet acquired | TBD |

**Caveats:**
- Calcium signal is a low-pass-filtered proxy for V; recovering V-trajectories requires deconvolution with cell-specific GCaMP kinetics. The Atanas data is at ~10 Hz; Mellem-scale plateau (~600 ms) is barely resolvable, and per-segment dynamics absolutely are not.
- Single-cell-resolution exists for some cells, but compartment-level resolution does not.
- **Useful for Variant C (full-cell latent dynamics) and possibly Variant B (cell-level V dynamics).** Not useful for Variant A (per-segment).

### 1.4 c302 morphology data (NeuroML2 — locally cloned)

`~/Desktop/C-Elegans/simulation/upstream/c302/c302/NeuroML2/` contains 305 cell.nml files including `AVAL.cell.nml` (57 segments, 1 soma + ~30 axon + branches). Per architectural plan, c302 has 607 cell morphologies total.

**Use:** structural priors for graph topology in any variant. Not "training data" in the loss-function sense, but constrains the architecture.

---

## 2. Parameter-counts per variant

### Variant A — mechanistic-anchored, learned coupling + densities

For a single cell (e.g., AVAL with 57 segments, 7 channels):

| Parameter group | Count |
|---|---|
| Per-segment, per-channel gbar | 57 × 7 = 399 |
| Axial conductance per edge | ~56 edges (tree) → 56 |
| Ca diffusion coefficient | 1 |
| Per-segment cm, surf | 57 × 2 = 114 (mostly fixed from morphology) |
| (Optional) residual MLP per node | ~256-1024 |

**Trainable count, conservative (gbar + axial only, MLP off):** ~455 per cell.
**Trainable count, with residual MLP:** ~700-1500 per cell.

For 302 cells × ~455 = ~137,000 across the worm. Heavy regularization needed unless cells share gbar parameters via cell-type priors (CeNGEN density coupling, Wave 4-style).

### Variant B — mechanistic channels, learned membrane integration

| Parameter group | Count |
|---|---|
| Channel gating params (mechanistic, fixed) | 0 |
| Per-segment gbar (mechanistic, free) | 399 |
| Edge-message learned MLP | ~1k-10k |
| Node-update learned MLP | ~10k-100k |

**Trainable count:** ~10k-100k per cell. Dramatically higher than Variant A.

### Variant C — fully learned per-cell graph

| Parameter group | Count |
|---|---|
| Per-cell encoder MLP | ~10k |
| Per-cell decoder MLP | ~10k |
| Edge-message GNN layers | ~100k-1M |
| Latent dim ~32 → ~64 across cells | absorbed in MLPs |

**Trainable count:** ~100k-1M total for a 302-cell network. Requires Atanas-scale or larger training data for any hope of generalization.

---

## 3. Data-vs-parameter regime per variant

The classical rule-of-thumb for ML: trainable parameters << ~10× number of effective independent training examples.

### Variant A

- **Trainable per cell:** ~500.
- **Available simulator-generated training samples:** unlimited (re-runnable Brian2 at any protocol).
- **Available biological training samples:** dozens of published Nicoletti panels.

**Verdict:** **sufficient for simulator-self-distillation training** (target: a GNN that reproduces Brian2 cellular dynamics). **Marginal but feasible for Mellem-style architectural fitting** if treated as a single-target optimization rather than a generalization regime — the ~50 published panels constrain the gbar prior, and the morphology constrains topology, leaving axial Ra and a small number of free parameters.

For full per-cell Wave 4-style density-fitting across 302 cells, the data regime is harder; Atanas calcium imaging is the natural anchor but requires the deconvolution work to extract V-trajectory targets.

### Variant B

- **Trainable per cell:** ~10k-100k.
- **Available biological training:** dozens of panels.

**Verdict:** **data-starved unless trained primarily on simulator-generated data.** A learned membrane-integration MLP that's distilled from Brian2 dynamics is feasible but mostly tautological (you've reproduced what Brian2 does). The interesting variant — learned dynamics that go beyond Brian2's expressive limits — needs biological data Brian2 doesn't capture. We don't have that at the volume needed.

### Variant C

- **Trainable total:** ~100k-1M.
- **Available training:** Atanas (~10k V-equivalent samples per worm × 10 worms = ~100k samples, but at compartment-coarse resolution).

**Verdict:** **data-starved at full-network scale.** BAAIWorm-style fully-learned approaches use orders of magnitude more compute and longer training to fit similar parameter counts. Without Hallinen / Yemini / synthetic-augmentation, Variant C is mostly speculation about a future Wave 5+ project, not a Wave 2/3 work item.

---

## 4. Recommended training data strategy (if X.1d prototype proceeds)

For the X.1d prototype (2-3 node Variant A on EGL-19+leak), the right training data is:

1. **Re-run `validate_egl19.py` with a sweep of injection currents and step durations** (already-existing infrastructure, just call it with different protocol params). Generates Brian2-validated voltage traces in known-mechanistic regime.
2. **Use Brian2 single-compartment simulation as ground truth** — the GNN's job is to learn that 2-3 nodes + axial coupling can approximate a single-compartment (degenerate target). This is sanity-check level; pass = pipeline works, fail = pipeline broken.
3. Optionally: **run NEURON multi-compartment AVAL with the same channels** (using c302 morphology + Nicoletti channels in NEURON). This generates target traces for the genuinely interesting test: can a 2-3 node GNN approximate a 57-segment NEURON simulation?

The optional step 3 is **the actual research question for Wave 3** but is out of scope for the X.1d bounded prototype.

---

## 5. Caveats specific to architecture+data unification

The honest concern: **even Variant A training data may not be sufficient to validate architectural claims about Mellem dynamics.**

Reason: Mellem 2008 reports a single representative plateau under specific experimental conditions. We have a single trace (digitized) as our 2b ground truth. Fitting a GNN to a single trace is **overfitting territory regardless of parameter count** — many architectures could match a single trace.

The architectural-plan's morphology fork has the same data limitation (see X.2a). The right path through this for either approach is:

1. Validate the channel kinetics (already done — Phases C/D/E).
2. Validate the morphology-aware integration against **multiple target traces from Mellem and other AVA recordings** (Lockery, Goodman lab traces, ChannelWorm reference).
3. If the model fits all targets simultaneously, the architectural commitment is defensible. If it can fit Mellem only by overfitting to that single trace, the model is suspect.

This is a **Wave 3 work-block-level concern**, not a Wave 2 prototype concern, but should inform the morning-review framing.

---

## 6. Summary

| Variant | Sufficient data for prototype | Sufficient data for full-cell fitting | Sufficient data for cross-cell generalization |
|---|---|---|---|
| A — mech-anchored + learned coupling | **Yes** (simulator-self-distillation) | Marginal (Mellem trace + Nicoletti panels + CeNGEN priors) | Marginal (needs Wave 4 CeNGEN coupling) |
| B — learned membrane integration | Yes (simulator-distillation) | No (data-starved) | No |
| C — fully learned cell-graph | No (parameter count too high) | No | No |

**Net data-feasibility recommendation:** Variant A is feasible for prototyping and for Wave 3 single-cell research. Variants B and C are data-starved at the scales they imply.

---

## 7. Order-of-magnitude estimates summary

- **Variant A prototype (2-3 nodes, EGL-19 + leak):** ~50-100 trainable params, unlimited simulator data → strong over-determined regime.
- **Variant A single-cell AVAL (57 seg, 7 channels):** ~500 trainable params, unlimited simulator + dozens biological → marginal but feasible with strong priors.
- **Variant A full worm:** ~137k trainable across 302 cells, biology limited → needs CeNGEN priors + parameter sharing across cells of same class. Wave 4-territory.
- **Variant B / C:** orders of magnitude more parameters than data; not currently feasible.
