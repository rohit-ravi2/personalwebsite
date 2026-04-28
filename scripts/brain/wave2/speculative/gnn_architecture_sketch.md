# X.1a — GNN hybrid architecture sketch

**Status:** Speculative architectural investigation. Not a commitment. Inputs to morning review of Wave 2 architectural roadmap (Wave 3 long-term thinking and condition-6 alternative-comparison).

**Frame:** The Phase F 2b failure (single-compartment AVA Brian2 produces 46.8 mV / 21.4 ms vs Mellem 15-25 mV / 400-800 ms targets) is the architectural insufficiency this sketch exists to address. A GNN hybrid is one of three speculative responses being characterized in parallel with the architectural plan's recommended morphology fork.

---

## 1. Why a GNN at all? What problem would it solve?

The 2b failure is structurally diagnosed in `gate2_ava_cell_construction.md` as:

1. **Plateau amplitude too large:** input resistance ~6.7 GΩ × 50 pA over 100 ms over-depolarizes a single-compartment cell. Real AVA's 30+ axon segments distribute injected charge across spatially separated compartments before the soma's V is over-driven.
2. **Plateau duration too short:** in single-compartment, leak τ ≈ 64 ms but observed termination is ~21 ms — active K dominates termination. Mellem's 400-800 ms plateau likely needs Ca-induced-Ca-release nanodomains and SLO-1 spatial buffering with diffusion-mediated termination.
3. **No dynamic Ca-pool:** SLO-1 isolated reads static cai = 5e-5 mM (per F12); SLO-1+EGL-19 uses closed-form `calcium(V)` (per F13). Neither reads spatially-resolved Ca.

A GNN's promise here is **learnable spatial dynamics**: the message-passing operator can in principle absorb the per-compartment Ca-buffering and inter-compartment coupling that single-compartment Brian2 cannot represent and that explicit multi-compartment NEURON pays for in compute and morphology-data dependence.

The honest worry: a GNN may also absorb (and obscure) the physics in a way that makes it un-mechanistic — losing the property that channel-level conclusions transfer back to the biological literature. So the sketch below leans toward **mechanistic-anchored hybrid** rather than fully learned dynamics.

---

## 2. Conceptual architectures (three variants)

### Variant A — "GNN-as-spatial-coupling, channels stay mechanistic"

```
                Per-segment node state                 Edge messages = axial coupling
              ┌────────────────────────┐              ┌──────────────────────┐
   Segment i  │ V_i, [Ca]_i, m_egl19_i,│   ←——————→   │ axial current        │
              │ h_egl19_i, m_slo1_i,...│              │   I_axial_ij = (V_i-V_j)/Ra_ij
              └────────────────────────┘              │ Ca diffusion         │
                       │   ▲                          │   J_Ca_ij = D·([Ca]_i-[Ca]_j)/dx
                       │   │                          └──────────────────────┘
              Mechanistic per-node update:
              dV_i/dt   = -I_channels_i / C_i + ΣI_axial_ji / C_i
              d[Ca]_i/dt = -ica_i/(2F·vol_i) + ΣJ_Ca_ji - decay
              dm_egl19_i/dt = (m∞(V_i) - m_egl19_i)/τ_m(V_i)
              ... etc per channel
```

- **Nodes:** compartments (one per c302 segment, or coarsened into 3-6 super-segments).
- **Edges:** axial coupling between adjacent segments + (optional) chemical/peptidergic edges to other cells.
- **Node state:** V, [Ca], gating variables for each channel.
- **State evolution:** **mechanistic** inside the node (Nicoletti's NMODL-derived eqs), GNN provides only the message-passing operator for inter-compartment coupling. The "GNN" here is degenerate-mechanistic — closer to MeshGraphNet with mechanistic node updates than a learned dynamics model.
- **Trainable parameters:** axial conductance, Ca diffusion coefficient, per-segment channel densities. Possibly a small learned correction term Δ on the mechanistic update for systematic-error absorption (à la "neural ODE residual").

This is essentially **multi-compartment Brian2/NEURON with a learned coupling kernel and learned per-segment densities**. It reduces to the X.2a multi-compartment-explicit architecture if the learned coupling kernel is replaced by analytical axial resistance.

**Pros:** falsifiable per-channel; channel-level claims still transfer to literature; small trainable surface area; data-efficient.
**Cons:** marginal value over X.2a unless densities or coupling are genuinely better learned than computed. Likely the right choice if data-driven density-fitting (Wave 4 anyway) is unified with the architecture decision.

### Variant B — "Hybrid: mechanistic channels, learned membrane integration"

```
              Per-segment node state                  Edge messages = learned coupling
    Segment i ┌────────────────────────┐              ┌──────────────────────┐
              │ V_i, [Ca]_i, gating_i  │   ←——————→   │ learned m_ij(V_i, V_j)│
              └────────────────────────┘              │ Ca-, mod- edges       │
                       │   ▲                          └──────────────────────┘
                Mechanistic ICHANNEL_i (Nicoletti eqs)  +  GNN(messages_to_i, state_i) → dV_i/dt, d[Ca]_i/dt
                                                       (learned residual or full V/Ca dynamics)
```

- **Nodes:** compartments.
- **Edges:** axial + chemical synaptic + peptidergic, with **learned edge functions**.
- **Node state:** V, [Ca], gating.
- **State evolution:** channel kinetics (gating variables) **mechanistic**; V and [Ca] dynamics **learned** (through MLP or Neural-ODE that takes channel currents + edge messages as input and outputs dV/dt, d[Ca]/dt).
- The GNN absorbs: capacitance distribution, Ca-pool stoichiometry, spatial buffering, axial resistance, possibly nanodomain channel-channel coupling.

**Pros:** can in principle absorb the bookkeeping (Ca-pool unit-conversion, axial coupling) without requiring explicit physical formulas; potentially the right level of abstraction if Mellem-style plateau emerges from many-channel many-compartment cooperation that single-formula Brian2 can't easily express.
**Cons:** learned dV/dt loses the mechanistic-falsifiability property — what does it mean to say "EGL-19 is mostly responsible for the plateau" if dV/dt is a learned MLP? Validation has to demonstrate that mechanistic interpretation survives the learned operator.

### Variant C — "Fully learned cellular dynamics on a per-cell graph"

```
                 Cell graph (each cell = 1 node)
                 Edges = synapses, gap junctions, peptide signaling
    
    Cell i  ┌─────────────────────────────────────┐
            │ Latent state z_i ∈ R^d              │
            │ Encoder: (sensory inputs, graph context) → z_i
            │ Decoder: z_i → V_i, firing rate, Ca │
            └─────────────────────────────────────┘
              GNN message passing over connectome
              Learned dynamics: z_{t+1} = f_θ(z_t, messages)
```

- **Nodes:** whole cells (302).
- **Edges:** chemical + electrical connectome.
- **State:** opaque latent z_i.
- **State evolution:** fully learned via GraphCast/MeshGraphNet pattern.
- Outputs (V, firing, behavior) decoded from z.

**Pros:** the "obvious" deep-learning approach; if it works, fastest by far at inference; matches existing classifier-readout abstractions in production simulator.
**Cons:** complete loss of mechanistic interpretation. Channel-level claims (slo-1 KO, egl-19 mutant, modulator sensitivity) have to be reproduced as input perturbations to a learned model — possible but expensive to validate. Most data-hungry of the three. Loses the architectural-plan property that paper 3's mechanistic claims are anchored to per-channel biophysics.

---

## 3. Recommended variant for further investigation: Variant A

Variant A is the closest to the architectural-plan's already-committed direction and the most defensible if it surfaces in Wave 3:

1. **Mechanistic anchoring preserved:** per-channel claims still hold; Wave 1 + run #1/#2 channel translation work is not invalidated.
2. **Falsifiability preserved:** Gate 2 component 2a still applies per-channel; component 2b becomes "Mellem plateau is/isn't reproduced under learned spatial coupling" — testable.
3. **Comparison framework with X.2a multi-compartment-explicit is clean:** Variant A reduces to X.2a if the learned coupling is replaced by analytical Ra. So Variant A isn't an alternative to X.2a so much as **X.2a with optional learned augmentation**.
4. **Smallest trainable surface area:** axial conductance (~30 segment edges in AVAL), Ca diffusion coefficient, per-segment channel densities. Probably 100-1000 trainable parameters per cell. Tractable with available data.

Variant B is more speculative but worth considering if Variant A + analytical multi-compartment together still under-resolve Mellem dynamics — at that point a learned-residual augmentation is the natural escalation.

Variant C is a Wave 5+ direction (or competing-project territory like BAAIWorm's `eworm_learn`) and not the right move now.

---

## 4. ASCII diagram: Variant A applied to AVAL with c302 morphology

```
                                                           AVAL (c302 morphology, 57 segments,
                                                           1 soma + ~30 axon segments + branches)
                                                           
       Mechanistic channel updates per segment       ←———— Brian2/NEURON-style per-segment integration
       (EGL-19, SLO-1 iso, SLO-1+EGL-19, NCA, leak,
        SHK-1, SHL-1, KQT-3 + cadiff Ca-pool)
                       │
                       │ V_i, [Ca]_i, gating
                       ▼
                ┌─────────────┐
                │ Soma (s=0)  │
                └──────┬──────┘
                       │  Ra_axial (learned/analytic)
                       │  D_Ca_axial (learned/analytic)
                       ▼
                ┌─────────────┐
                │ Axon seg 16 │
                └──────┬──────┘
                       │
                       ▼
                       …
                       │
                       ▼
                ┌─────────────┐
                │ Axon seg 60 │
                └─────────────┘
       
       Inter-segment messages:
         m_ij = (V_i - V_j)/Ra_ij    (axial current)
         q_ij = D_Ca · ([Ca]_i - [Ca]_j) / dx_ij
       
       Per-segment update:
         dV_i/dt   = (- Σ I_channel_i + Σ m_ij_in - Σ m_ij_out) / C_i
         d[Ca]_i/dt = -(I_Ca_i)/(2 F vol_i) + Σ q_ji - Σ q_ij - ([Ca]_i - [Ca]_eq)/τ_Ca
       
       Trainable (Variant A, conservative):
         {Ra_ij, D_Ca, gbar_channel_i for each channel × each segment}
       
       Trainable (Variant A + residual augmentation):
         + small MLP δ_θ(V_i, [Ca]_i, gating_i, neighbor_state_i) added to dV_i/dt
         (interpretation: data-driven correction for systematic Brian2/NEURON gaps)
```

---

## 5. Reference patterns from literature

**Not exhaustive — pattern references for downstream readers.**

- **Neural ODEs** (Chen et al. 2018) — continuous-time dynamics with autograd through ODE solver. Variant A's mechanistic core is already a Neural ODE in this sense; the "neural" part is just the channel kinetics expressed in Brian2 eqs strings.
- **MeshGraphNet** (Pfaff et al. 2021) — mesh-based GNN for fluid/solid dynamics. Pattern: node = mesh element, edge = mesh adjacency, learned edge function for dynamics. Most directly analogous to Variant A.
- **GraphCast** (Lam et al. 2023) — global weather forecast on encoder-processor-decoder GNN. Pattern: multi-resolution grid, learned temporal evolution. Closer to Variant C if applied here.
- **HodgkinHuxley-style learnable-channel models** (Beniaguev et al. 2021, "Single cortical neurons as deep artificial neural networks" — Neuron 2021) — fits cell-level deep network to cellular electrophysiology. Demonstrates that single neurons require multi-layer GRU-equivalent expressivity to capture full dynamics. **Cited here as evidence that single-compartment models are structurally insufficient and that learned spatial structure can absorb the gap** — though their model is fully learned (Variant C-style) rather than hybrid.
- **Geneva & Zabaras 2022** ("Transformers for modeling physical systems") — learned operators for PDE-like systems. Variant B-style for V dynamics.
- **PINN / Physics-informed NN** patterns — embed conservation laws as soft constraints in loss. Useful for Variant B if learned dV/dt should respect KCL.
- **OpenWorm c302 + NetPyNE** — declarative cell descriptions in NeuroML2, runnable on multiple backends. Not GNN-based, but the data substrate (per-segment state, axial coupling) matches Variant A.

---

## 6. Open architectural questions (to resolve before any prototype scales beyond X.1d)

1. **Coarsening:** AVAL has 57 segments. For 302 cells × ~30 segments = ~9000 nodes per worm. Is this tractable on RTX 4060 Ti? Pre-flight estimate: yes for forward pass (small MLPs at each node), maybe-no for training with autograd through long time series.
2. **Time-step regime:** Brian2 production uses dt = 25 µs. Mellem plateau is 600 ms = 24,000 steps. Backprop through 24k unrolled steps is the data-hungriest part. Adjoint-method Neural ODE training is the standard answer; need to validate it works for this stiff dynamics.
3. **Per-cell vs per-segment trainable parameters:** if every cell has 30 segments × 8 channels = 240 gbar values that are trainable per cell × 302 cells = 72,480 parameters. Even with strong CeNGEN priors, this needs careful regularization to avoid overfitting.
4. **Mechanistic-vs-learned interface for Ca-pool:** Variant A's analytic Ca-buffer formula has parameters (depth, β, fca, vol) that are conservative-anchored. Should these be free-trainable or fixed? Recommendation: fix to Nicoletti values, trainable only at "doesn't fit" diagnostic flag.
5. **Loss function:** voltage trajectory loss (L2 on V_t)? Voltage-feature loss (current production pattern)? Mellem-style discrete-feature loss (plateau amplitude, duration)? Likely **all three combined** with weights — feature loss dominates, trajectory loss as regularizer.
6. **Validation regime:** does the trained GNN generalize across cells, or is each cell trained separately? Cross-cell generalization is the more interesting research question (related to CeNGEN-coupling work).

---

## 7. Reasonable scope of a prototype (X.1d gate)

Given bounded effort:
- 2-3 nodes (toy soma + 1-2 axon segments), not the full 57-segment AVAL.
- 1-2 channels (EGL-19 + leak; data already available from run #1's `validate_egl19.py`).
- Variant A architecture with trainable axial Ra and trainable per-segment gbar.
- Train against Brian2-generated voltage traces (which are themselves NEURON-validated to <5% per Phase β results).
- Question to answer: **"can a 2-node GNN approximate the integration of a 1-node Brian2 cell, when the inter-node coupling is the only learnable degree of freedom?"** This is mostly a sanity check on the training pipeline, not a Mellem-style architectural claim. Mellem-style validation is far out of scope for X.1d.

If even this prototype-scale doesn't reproduce 1-node Brian2 dynamics within tolerance, the approach is structurally suspect. If it does, the path forward to morphology-scale validation is open.

---

## 8. Summary

| Aspect | Verdict |
|---|---|
| Architectural feasibility | **Yes, conceptually, especially Variant A.** Variant A reduces toward X.2a if learned coupling is replaced by analytic axial Ra; the GNN framing is mostly justified by per-segment density-fitting rather than learned spatial dynamics. |
| Distinguishing value over X.2a (multi-compartment explicit) | **Marginal unless density-fitting is unified.** A pure spatial-coupling GNN is a worse engineering-effort/validation-effort trade than X.2a. The clear value-add is data-driven per-segment density estimation (Wave 4 anyway). |
| Mechanistic interpretability | **Preserved in Variant A, lost in Variant C, partial in Variant B.** Variant A is the only variant that doesn't break paper 3's mechanistic-claim foundation. |
| Wave 3 implication if adopted | **Variant A is a natural Wave 4 enhancement on top of X.2a.** Not a Wave 2 architectural commitment; rather a Wave 4 density-learning extension. |

**Net architectural recommendation (preview, fully developed in `x1_summary.md`):** Variant A is interesting as a Wave 4 density-fitting tool layered on top of an X.2a / morphology-fork foundation. It is **not** a primary alternative to X.2a or to the architectural plan's morphology fork. The strongest case for GNN work is unification with per-cell density estimation (CeNGEN-coupling, modulator-density-tuning).
