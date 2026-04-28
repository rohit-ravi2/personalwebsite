# X.2a — Multi-compartment explicit architecture (sketch)

**Status:** Speculative architectural investigation. Sketch only — no prototype, no implementation.

**Frame:** This sketch characterizes the architectural plan's already-recommended condition-6 response (the morphology fork) at slightly more concrete depth, so it can be compared apples-to-apples against the GNN and NeuroML2-native variants in `speculative_summary.md`.

---

## 1. The core idea

Move from single-compartment Brian2 cell (current production architecture) to **per-segment Brian2 cell** using c302's NeuroML2 morphologies. Each segment carries its own state vector (V, [Ca], gating); segments are coupled axially.

This is the architectural plan's **morphology fork** (`phase_v_w2_architectural_plan.md` §"Conditional fork: morphology integration as Phase β prerequisite"). It is the plan's recommended response if condition 6 surfaces (which it now has, per Phase F 2b).

---

## 2. What changes versus current single-compartment

### 2.1 Ca-pool encoding

**Current (Phase β):**
- `cadiff` and `caintra1` per-cell (single bulk pool).
- SLO-1 isolated reads static cai = 5e-5 mM (per F12 — Nicoletti's published cells don't insert dynamic Ca-pool).
- SLO-1+EGL-19 uses closed-form `calcium(V)` (per F13 — nanodomain approximation).

**Proposed (multi-compartment):**
- Per-segment cadiff (or caintra1) instance.
- Each segment has its own [Ca]_i.
- Ca-entry through EGL-19 in segment i raises [Ca]_i locally; diffusion to neighbors via axial Ca current; decay via cadiff's standard β·Ca term.
- SLO-1 isolated in segment i reads [Ca]_i (locally — different across segments now).
- SLO-1+EGL-19 still uses closed-form `calcium(V)` (the nanodomain-approximation closed form is per-segment too, evaluated at local V).

### 2.2 Channel placement

c302 morphology has segment groups: `Soma`, `Axon`, `Dendrite`. Real biology has differential channel distribution (e.g., somatic Ca-channels, axonal K-channels). Initial sketch:

| Channel | Soma | Axon | Dendrite |
|---|---|---|---|
| leak | yes | yes | yes |
| EGL-19 | dominant | low | low |
| SLO-1 isolated | yes | low | yes |
| SLO-1+EGL-19 | yes (paired with EGL-19) | low | low |
| SHK-1 | low | dominant | low |
| SHL-1 | low | dominant | low |
| NCA | yes | yes | yes |
| KQT-3 | yes | yes | yes |

These distributions are **starting hypotheses** based on general neuroscience patterns and would need calibration against Nicoletti or alternate sources. Mellem 2008 paper itself does not specify per-segment distribution. **This is itself a significant scientific gap** — the segment-level channel distribution for AVAL is not well-characterized in the C. elegans literature, unlike for e.g. mammalian pyramidal cells.

### 2.3 Brian2 SpatialNeuron

Brian2 supports multi-compartment cells via `SpatialNeuron`:

```python
from brian2 import SpatialNeuron, Morphology

morpho = Morphology.from_file('AVAL.cell.nml')  # or build manually from c302 segments
cell_eqs = """
Im = i_leak + i_egl19 + i_slo1iso + i_slo1egl19 + i_shk1 + i_shl1 + i_nca + i_kqt3 : amp/meter**2
... per-channel equations ...
"""
neuron = SpatialNeuron(
    morphology=morpho,
    model=cell_eqs,
    Cm=1.0*ufarad/cm**2,  # capacitance density
    Ri=100*ohm*cm,         # axial resistivity
    method='exponential_euler',
)
neuron.gbar_egl19 = 9.288e-6 * 1e-4 * siemens / cm**2  # AVAL g0 in Soma
neuron.gbar_slo1iso = ... # per-segment
```

Brian2 handles axial coupling automatically through SpatialNeuron's discretization. Per-segment Ca pools require eqs-string state per-segment, which is straightforward because SpatialNeuron extends per-compartment.

### 2.4 Time-step / numerical considerations

- Brian2's `exponential_euler` method handles HH-style stiff dynamics on SpatialNeuron.
- AVAL has 57 segments. With 8 channels × ~3 state variables each + 2 Ca-pool variables = ~26 state variables × 57 segments = ~1500 state vars. Tractable.
- dt = 25 µs (current production) probably stable; may need to reduce to 10 µs for SpatialNeuron + EGL-19. Empirical check needed.

---

## 3. Expected effect on plateau dynamics

The single-compartment failure modes from `gate2_ava_cell_construction.md`:

- **Plateau amplitude too large (46 mV):** in single-compartment, 50 pA injection acts over the whole capacitance simultaneously. In multi-compartment, the soma's V depolarization is buffered by the long axon — current spreads, charge redistributes. Expected effect: amplitude reduced by ~2-5× depending on geometry. Hits 15-25 mV target plausibly.

- **Plateau duration too short (21 ms):** in single-compartment, no [Ca] buffering means SLO-1 K-current doesn't develop the slow termination dynamics characteristic of Mellem's 600 ms plateau. In multi-compartment with per-segment cadiff, [Ca]_soma rises with EGL-19 entry, [Ca] diffuses out along axon (slow), SLO-1 in soma sees gradually rising [Ca], K-current ramps up, plateau terminates on the slow [Ca] timescale. Expected effect: plateau duration extends by 10-20×. Hits 400-800 ms plausibly.

- **Termination dynamics:** SLO-1 dominates because the [Ca] decay timescale (cadiff τ ≈ 50-200 ms × spatial diffusion) exceeds leak τ. Matches Mellem.

These are **structural predictions**, not numerical guarantees. The fork's premise is that they hold; failure of multi-compartment to reproduce Mellem would invalidate not just multi-compartment but the architectural plan's hypothesis about the failure mechanism.

---

## 4. Implementation sketch

### Phase β-morph (per architectural plan: estimated 2-3 weeks of effort)

Conceptual subphases:

1. **Morphology import:** parse AVAL.cell.nml into Brian2 Morphology object. Brian2 has limited NeuroML2 support; may need custom parser or a libNeuroML→Brian2 bridge.
2. **Per-segment channel placement:** apply gbar by segment group. Initial uniform per-segment-group; reserve calibration for Phase γ-morph.
3. **Per-segment Ca-pool:** modify `calcium_pool.py` to support per-compartment instances; integrate with SpatialNeuron eqs.
4. **Validation harness extension:** modify `voltage_clamp_harness.py` to query Brian2 SpatialNeuron's segment-resolved V and currents; modify `plateau_harness.py` to extract Mellem features from soma segment.
5. **NEURON multi-compartment reference:** Nicoletti 2024 already provides multi-compartment AVAL via c302's morphology + her .mod files. Use this as reference for 2a apples-to-apples multi-compartment validation.

### Phase γ-morph

- Run Gate 2 with imported channels in proper compartmental scaffold.
- Spec expectation: 2b passes because compartmental architecture provides dendritic/axonal state separation that single-compartment cannot.

### Per-segment channel-density calibration

Critical sub-question: **what densities go in each segment group?** Initial hypothesis is uniform within Soma/Axon/Dendrite. If 2b fails under uniform-by-group, the next layer is per-segment density-tuning (which is where GNN Variant A's learned-density extension would naturally enter — see `gnn_architecture_sketch.md` §3).

---

## 5. Risks

| Risk | Likelihood | Notes |
|---|---|---|
| Brian2 NeuroML2 import not robust | medium | Brian2's NeuroML support is partial; fallback to custom parser is feasible (the .nml format is XML-segments+groups, simple). |
| Per-segment density calibration is ill-posed (no biological data per segment) | high | The literature mostly reports cell-level gbar, not per-segment. Calibration relies on assumption that group-uniform is approximately correct. |
| SpatialNeuron + many channels is slow on RTX 4060 Ti | medium | 57 seg × 26 state × 8 cells = ~12k state vars per cell-batch; should be tractable on CPU. GPU advantage probably small at this scale. |
| Mellem plateau emerges in NEURON multi-compartment but not Brian2 multi-compartment | medium | Possible if SpatialNeuron's discretization differs from NEURON's. Validation against NEURON multi-compartment is mandatory. |
| 57-segment scale + Mellem 600 ms = 24k timesteps × backprop is impractical for any density-fitting | high | Important for X.1d Variant A integration but not for pure forward simulation. |
| Per-segment Ca-pool stoichiometry calibration repeats Phase β-pre F6/F7 work | medium | Per-segment cadiff parameters need empirical calibration against NEURON multi-compartment, similar to single-compartment F7 calibration. Tractable but adds ~1 week. |

---

## 6. Comparison to architectural plan's morphology fork

| Aspect | Architectural plan | This sketch |
|---|---|---|
| Backend | Brian2 SpatialNeuron | same |
| Morphology source | c302 NeuroML2 | same |
| Channel set | Same as Phase β (7 essential + 15 lazy) | same |
| Ca-pool | Per-segment cadiff/caintra1 | same |
| Validation | Gate 2 re-run | same + per-segment calibration step |
| Estimated cost | 2-3 weeks Phase β-morph + 1 week Phase γ-morph | ~3-4 weeks (matches plan) |
| Distinguishing risk vs single-compartment | Per-segment density calibration is ill-posed | (same; sketch makes it explicit) |

**No substantive divergence from the architectural plan.** This sketch's role is to make the morphology fork concrete enough to compare against GNN and NeuroML2-native variants.

---

## 7. Comparison to GNN Variant A

| Aspect | X.2a multi-compartment explicit | GNN Variant A |
|---|---|---|
| Spatial coupling | Analytic axial resistance from morphology | Learned axial conductance per edge |
| Per-segment density | Hand-tuned by group, then optimized | Trainable per-segment |
| Ca-pool | Per-segment cadiff (mechanistic) | Per-segment cadiff (mechanistic) |
| Channel kinetics | Mechanistic (Brian2 eqs) | Mechanistic (Brian2 eqs) |
| Trainable params per cell | ~50-200 (gbars + axial) | ~500 (per-seg gbars + axial + diffusion) |
| Mechanistic interpretability | Full | Full |
| Falsifiability | Standard | Standard + ablation tests required |

**Key insight: X.2a is a special case of GNN Variant A** with axial coupling computed from morphology rather than learned. If X.2a doesn't fit Mellem due to incorrect axial-coupling assumptions, the GNN extension is the natural escalation. Otherwise, X.2a is sufficient and simpler.

The recommendation: **execute X.2a (the morphology fork) first**; if it passes 2b, GNN Variant A is unnecessary at the cellular level. If it fails 2b, GNN Variant A becomes a natural Wave 4 extension.

---

## 8. Summary

| Question | Answer |
|---|---|
| Architectural feasibility | Yes — Brian2 SpatialNeuron + c302 morphologies + per-segment channels is well-trodden territory in computational neuroscience. |
| Implementation cost | ~3-4 weeks (matches architectural plan's estimate). |
| Mechanistic interpretability | Fully preserved. |
| Expected effect on Mellem 2b failure | Plausibly fixes both amplitude (charge spread across compartments) and duration (Ca-buffering in spatial geometry). Subject to per-segment density calibration. |
| Distinguishing risk | Per-segment channel density data is sparse in the C. elegans literature; calibration is partly speculative within Soma/Axon/Dendrite groups. |
| Comparison vs GNN Variant A | X.2a is a special case (analytic axial coupling). Execute X.2a first; GNN extension is Wave 4 if needed. |
| Comparison vs morphology fork in arch plan | Identical in substance. |
| Recommendation | **This is the architectural plan's already-committed condition-6 response and remains the strongest candidate for primary action under condition 6.** |
