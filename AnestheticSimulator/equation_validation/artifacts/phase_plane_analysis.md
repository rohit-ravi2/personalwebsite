# CP B.1 — Phase plane analysis for production-grade Wave 2 cells

**Date:** 2026-04-28

Phase plane in (V, slow_gate) space for each Nicoletti-validated cell. Slow gate identified per cell from the channel suite; nullclines + fixed points computed via ohmic-channel approximation with the slow gate as the only dynamic variable.

## AVAL

- Slow gating variable: **h_egl19** (Ca_inactivation on egl19)
- Boltzmann: V_half = -25.0 mV, k = 5.0 mV, τ = 50.0 ms
- Rationale: EGL-19 L-type Ca channel inactivation drives plateau decay; Wicks 1996 + Mellem 2008.

### Fixed points

| I_inj (pA) | fixed points (V_mV, gate) |
|---|---|
| 0 | [{'V_mV': -29.27, 'gate': 0.7013}] |
| -5 | [{'V_mV': -37.64, 'gate': 0.9261}] |
| +5 | [{'V_mV': -22.47, 'gate': 0.3759}] |
| +20 | [{'V_mV': 24.56, 'gate': 0.0}] |

### Interpretation

Monostable at rest. Single attracting fixed point — biologically expected for graded interneuron.

**Wicks 1996 plateau check:** ✓ plateau-state FP at V = -29.27 mV (matches Mellem 2008 depolarized AVA regime).

## AVAR

- Slow gating variable: **h_egl19** (Ca_inactivation on egl19)
- Boltzmann: V_half = -25.0 mV, k = 5.0 mV, τ = 50.0 ms
- Rationale: Same as AVAL; UNC-103 (ERG-like K) provides additional slow current but EGL-19 inactivation dominates plateau.

### Fixed points

| I_inj (pA) | fixed points (V_mV, gate) |
|---|---|
| 0 | [{'V_mV': -26.83, 'gate': 0.5905}] |
| -5 | [{'V_mV': -34.88, 'gate': 0.8783}] |
| +5 | [{'V_mV': -18.96, 'gate': 0.2301}] |
| +20 | [{'V_mV': 18.95, 'gate': 0.0002}] |

### Interpretation

Monostable at rest. Single attracting fixed point — biologically expected for graded interneuron.

**Wicks 1996 plateau check:** ✓ plateau-state FP at V = -26.83 mV (matches Mellem 2008 depolarized AVA regime).

## AIY

- Slow gating variable: **n_slo1egl19** (K_activation_via_Ca_coupling on slo1egl19)
- Boltzmann: V_half = -30.0 mV, k = 10.0 mV, τ = 100.0 ms
- Rationale: SLO-1 BK channel activation coupled to EGL-19 Ca influx; slowest dynamic in AIY's channel suite; CP B.1 extrapolated parameters per WB3 caveat.

### Fixed points

| I_inj (pA) | fixed points (V_mV, gate) |
|---|---|
| 0 | [{'V_mV': -66.55, 'gate': 0.0252}] |
| -5 | [{'V_mV': -91.38, 'gate': 0.0022}] |
| +5 | [{'V_mV': -44.49, 'gate': 0.1901}] |
| +20 | [{'V_mV': -0.9, 'gate': 0.9483}] |

### Interpretation

Monostable at rest. Single attracting fixed point — biologically expected for graded interneuron.

**WB3 caveat note:** AIY parameters extrapolated from Wave 2 cell-builder validation. Phase plane structure here reflects extrapolated parameters; not a primary-source-anchored prediction. Sensitivity to V_half ± 5 mV would be informative.

## RIM

- Slow gating variable: **h_unc2** (Ca_inactivation on unc2)
- Boltzmann: V_half = -35.0 mV, k = 6.0 mV, τ = 80.0 ms
- Rationale: UNC-2 P/Q-type Ca inactivation; RIM's plateau/burst dynamics via Ca-dependent rebound. CP B.1 extrapolated parameters per WB3 caveat.

### Fixed points

| I_inj (pA) | fixed points (V_mV, gate) |
|---|---|
| 0 | [{'V_mV': -71.25, 'gate': 0.9976}] |
| -5 | [{'V_mV': -73.46, 'gate': 0.9984}] |
| +5 | [{'V_mV': -69.04, 'gate': 0.9966}] |
| +20 | [{'V_mV': -62.44, 'gate': 0.9898}] |

### Interpretation

Monostable at rest. Single attracting fixed point — biologically expected for graded interneuron.

**WB3 caveat note:** RIM parameters extrapolated from Wave 2 cell-builder validation. Phase plane structure here reflects extrapolated parameters; not a primary-source-anchored prediction. Sensitivity to V_half ± 5 mV would be informative.

## Cross-cell synthesis

AVA-class cells (AVAL, AVAR): tested for Wicks 1996 plateau structure. AIY/RIM: phase-plane structure documented under explicit WB3 extrapolation caveat — slow-gate Boltzmann parameters are biologically reasonable defaults but not primary-source-anchored.

**Sensitivity analysis caveat (per WB3 Decision 3 caveat):** AIY and RIM phase-plane fixed points should be re-evaluated under V_half ± 5 mV perturbation; if FP topology changes substantially across that range, the cell-builder extrapolation produces parameter-dependent dynamics that may not be robust. Sensitivity sweep deferred to a separate analysis.
