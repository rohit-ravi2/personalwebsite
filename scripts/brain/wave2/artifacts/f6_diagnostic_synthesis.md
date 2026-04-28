# Phase A.4 — F6 diagnostic synthesis

**Date:** 2026-04-26 run #2 invocation 1
**Inputs:** A.1 (symbolic decomposition), A.2 (geometry analysis), A.3 (robustness test)

---

## VERDICT: PRINCIPLED

The F6 finding from run #1 ("hidden 52,700× NMODL unit-conversion factor") was a
misdiagnosis. Both Ca-pool translations (cadiff, caintra1) are fully principled:
empirical calibration matches symbolic derivation exactly, and the calibration
generalizes cleanly across cell geometries and ica regimes.

**Phase A complete. Proceed to Phase B with confidence. No speculative-architecture
fork is triggered.**

---

## Decision tree result

Per the spec's three-way verdict structure:

> - **VERDICT_PRINCIPLED:** symbolic decomposition is clean, geometry analysis predicts
>   cell-specific factors accurately, calibration robustness tests pass across regimes.
>   Implication: Phase A complete. Proceed to Phase B with confidence.

All three conditions are satisfied:

1. **Symbolic decomposition clean** ✓
   - cadiff: 0.518 mM/(mA/cm²·ms) symbolic = 0.518 NEURON empirical (4 dp match,
     across 4 orders of ica magnitude across 3 cell geometries)
   - caintra1: α_sym(geometry) = α_NEURON(geometry) to 5 dp at AVA, AIY, RIM

2. **Geometry analysis predicts cell-specific factors accurately** ✓
   - cadiff is geometry-independent (verified)
   - caintra1 scales linearly with surf/vol (verified, predicted vs empirical match
     to 5 dp at AVA, AIY, RIM)

3. **Calibration robustness across regimes** ✓
   - 4 orders of magnitude of ica
   - Linear α holds throughout
   - No numerical instability or regime breakdown observed

---

## Summary of supporting findings

### F11 (this run): F6 was a misdiagnosis

Run #1's calcium_pool.py docstring claim "Symbolic re-derivation gives ~5183
mM/(mA/cm²·ms), empirical 0.525, ratio ~10000" is incorrect. The proper symbolic
derivation gives 0.518, matching empirical 0.525 within fit noise. The 10000 in
cadiff.mod is the proper unit-conversion factor from mol/(s·cm³) to mM/ms.

**The "hidden NMODL unit-conversion machinery" hypothesis is wrong.** No hidden
machinery exists. The 10000 in cadiff.mod IS the unit-conversion factor, fully
derivable from declared units. The empirical calibration converged to the
symbolically-correct value because the symbolic derivation IS the correct derivation
of the formula.

### F12 (this run): cells in scope don't insert Ca-pool

Of the 5 cell scripts inspected (AVAL, AVAR, AIY, RIM, VA5), only VA5 inserts
cadiff. AVA, AIY, RIM, AVAR rely on NEURON's default static cai (5e-5 mM = ca_eq
default). This means SLO-1 isolated (which reads cai) sees a constant value in
Nicoletti's actual cells, NOT a dynamic pool.

**Architectural implication for Phase D:** The Brian2 SLO-1 isolated translation
does NOT need a dynamic Ca-pool. It can use a constant cai matching NEURON's default,
keeping behavior consistent with Nicoletti's published cells.

### F13 (this run): slo1egl19 doesn't read cai

slo1egl19.mod has an internal closed-form `calcium(V)` FUNCTION computing nanodomain
Ca purely from V via:
```
calcium(V) = |gsc·(V-eca)·1e-3| / (8·π·r·d·FARADAY) × exp(-r/√(d/(kb·b))) × 1e6 × 1e-3 + fondo
```
with fixed parameters (gsc=40 pS, r=13 nm, d=250 μm²/s, kb=500e6/M-s, b=30 μM,
eca=60 mV, fondo=0.05 μM, FARADAY=96485).

**This is a deterministic V-dependent formula, not a dynamic pool.** Eliminates
the Phase E "nanodomain coupling encoding" architectural question; matches Nicoletti
exactly via algebraic equation in Brian2 eqs string.

---

## Implications for Phases B-F

### Phase B (NMODL pattern catalog)

The F6 entry in the pattern catalog should be **revised** to reflect the corrected
finding:
- Pattern name: "Ca-pool unit-conversion factor (the 10000 in cadiff.mod)"
- Recognition signature: large numerical constant in BREAKPOINT formula that scales
  rate quantities to match declared output units
- Recommended handling: trace through unit derivation; verify symbolic prediction
  matches empirical NEURON output at multiple regimes
- The "hidden machinery" framing should be retracted

### Phase C (4 non-Ca channels)

Unaffected by F6 verdict. SHK-1, SHL-1, NCA, KQT-3 are all voltage-gated K
(or Na-leak) channels that don't read cai. Standard symbolic translation expected
to work cleanly per the EGL-19 precedent.

### Phase D (SLO-1 isolated)

**Simplification:** since AIY's actual NEURON cell doesn't insert caintra1 or cadiff,
SLO-1 isolated reads NEURON's default static cai = 5e-5 mM. The Brian2 translation
can use a constant cai value matching NEURON's default. **No dynamic Ca-pool needed
for Phase D.**

### Phase E (SLO-1+EGL-19 coupled)

**Simplification:** slo1egl19 has internal `calcium(V)` formula. Phase E translation
matches Nicoletti exactly via algebraic equation. No nanodomain coupling architectural
decision required (the decision is "match Nicoletti's algebraic formula").

### Phase F (Gate 2)

AVA's actual cell uses irk + leak + egl19 + nca (no Ca-pool inserted). The Phase F
construction can match this exactly without any Ca-pool issues. **EGL-19 in AVA
doesn't read cai for gating** (verified in run #1: egl19.mod has only voltage-dependent
m, h gates), so the absence of a dynamic Ca-pool doesn't affect AVA's V dynamics.

For Component 2b (Mellem 2008 plateau dynamics with full essential set), we'd want
SLO-1 to terminate the plateau. SLO-1 isolated reads cai which is static at default.
SLO-1+EGL-19 reads its own internal calcium(V). Either way, Ca-pool is not load-bearing
for AVA's plateau dynamics.

**This is a meaningful Phase F simplification:** the cell construction's cellular
dynamics are determined by 7 channels + leak, with cai treated as constant default.
Brian2 vs NEURON comparison should be apples-to-apples without Ca-pool calibration
concerns.

---

## What we learned about the methodology

The Phase A diagnostic produced 3 substantive new findings (F11, F12, F13) and
substantially de-risked Phases D-F. The investment was worth it.

**Key methodology insight:** cross-checking docstring claims against fresh empirical
NEURON sims caught a translation-pattern misdiagnosis from run #1. The empirical
calibration was correct; the explanatory docstring was wrong. **Production code
empiricism > documented hypothesis.** Run #1's calibration converged to the right
answer for the right reason, but the docstring stated the wrong reason.

This is exactly the kind of finding the cross-session methodology is designed to
catch. Run #1 validated the calibration empirically (Brian2 vs NEURON match within
tolerance) but didn't cross-check the symbolic interpretation. Run #2's deeper
diagnostic confirmed the calibration is principled while exposing the misdiagnosed
explanation.

---

## Next actions

1. Update `run2_state.json`: last_completed_phase = "Phase A", f6_verdict = "PRINCIPLED"
2. Proceed to Phase B (NMODL pattern catalog systematization)
3. Phase X (speculative-architecture fork) is **NOT triggered** since verdict is
   PRINCIPLED, not FUDGE_FACTOR
4. Phases C-F all proceed under Path A as planned
