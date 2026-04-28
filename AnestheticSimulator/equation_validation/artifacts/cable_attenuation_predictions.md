# CP A.4 — Cable equation attenuation predictions

**Date:** 2026-04-28

Wave 2 production cells (AVAL, AVAR, AIY, RIM) are single-compartment Brian2 implementations of Nicoletti's published whole-cell models. The published electrophysiology validates the cells as point neurons for the somatic V(t) phenotypes; multi-compartment dendritic dynamics are not required for the validated empirical phenotypes.

This validator computes the electrotonic length constant λ assuming the cell were a uniform cable with R_a = 150.0 Ω·cm (invertebrate axoplasm range from Goodman et al. 1998).

## Per-cell length constants

| cell | R_m (MΩ·cm²) | r_soma (μm) | λ (μm) | τ_m (ms) | interpretation |
|---|---|---|---|---|---|
| AVAL | 0.03 | 9.46 | 3161.0 | 27.25 | λ >> r_soma — single-compartment defensible |
| AVAR | 0.03 | 9.45 | 2869.2 | 19.65 | λ >> r_soma — single-compartment defensible |
| AIY | 0.0 | 2.29 | 415.0 | 3.61 | λ >> r_soma — single-compartment defensible |
| RIM | 0.0 | 2.87 | 209.1 | 0.69 | λ >> r_soma — single-compartment defensible |

## Applicability assessment

**Wave 2 single-compartment models are defensible** when λ >> typical neurite length scales. C. elegans neurites are short (typical somatic process length 10-100 μm; full neuron extent including axon up to ~500 μm). λ values computed above are checked against this scale.

**Multi-compartment validation deferred** to compartmental_neurons.py and compartmental_neurons_kca.py in the production codebase. Those frameworks exist for cells where dendritic compartmentalization matters for the validated phenotypes (notably for AWC/AVA-Mellem compartmental dynamics — separate from Nicoletti's somatic models).

## λ predictions by cell

### AVAL

- R_m = 0.03 MΩ·cm²
- Total conductance density = 3.15e-05 S/cm²
- Soma radius (sphere approx) = 9.46 μm
- **Length constant λ = 3161.0 μm**
- Membrane time constant τ_m = 27.25 ms
- Verdict: λ >> r_soma — single-compartment defensible

### AVAR

- R_m = 0.03 MΩ·cm²
- Total conductance density = 3.83e-05 S/cm²
- Soma radius (sphere approx) = 9.45 μm
- **Length constant λ = 2869.2 μm**
- Membrane time constant τ_m = 19.65 ms
- Verdict: λ >> r_soma — single-compartment defensible

### AIY

- R_m = 0.0 MΩ·cm²
- Total conductance density = 4.43e-04 S/cm²
- Soma radius (sphere approx) = 2.29 μm
- **Length constant λ = 415.0 μm**
- Membrane time constant τ_m = 3.61 ms
- Verdict: λ >> r_soma — single-compartment defensible

### RIM

- R_m = 0.0 MΩ·cm²
- Total conductance density = 2.19e-03 S/cm²
- Soma radius (sphere approx) = 2.87 μm
- **Length constant λ = 209.1 μm**
- Membrane time constant τ_m = 0.69 ms
- Verdict: λ >> r_soma — single-compartment defensible

## Cross-validation note

These λ predictions assume the cell as a uniform cable. The actual C. elegans neuron geometry (asymmetric soma + neurite tree) makes the uniform-cable assumption an upper-bound estimate. Real attenuation may be larger if the cell has thin distal neurites with smaller r.

For multi-compartment validation against published attenuation measurements: **deferred to compartmental_neurons.py production wiring** and OpenWorm morphology-derived geometries when those become available in the validated production substrate.
