# Phase E architectural decision — SLO-1+EGL-19 nanodomain encoding

**Date:** 2026-04-26 run #2 invocation 1
**Source:** `nicoletti_2024/slo1egl19.mod`

---

## Decision: match Nicoletti's closed-form calcium(V) formula

Per F13 (Phase A finding), `slo1egl19.mod` does NOT read `cai` and does NOT
have a sub-membrane state variable. Instead it uses a closed-form
deterministic formula:

```
calcium(V) = |gsc·(V-eca)·1e-3| / (8·π·r·d·FARADAY) × exp(-r/√(d/(kb·b))) × 1e6 × 1e-3 + fondo
```

This is the Lluís-Buchholz / Alvarez nanodomain approximation: the Ca
concentration in a nanodomain at distance r from a single open channel,
in the steady-state limit where buffer capture and diffusion balance.

**Brian2 translation strategy:**

- Encode `calcium(V)` as an algebraic equation in the eqs string.
- All formula parameters are constants (no fitting, no calibration).
- No state variable for nanodomain Ca needed.
- The slo1egl19 m gate's `mminf` and `tslo1` then depend on this V-derived
  calcium value, plus `megl19_egl19` and `hegl19_egl19` (the EGL-19 m and h
  gates passed via NMODL EXTERNAL declarations).

**EXTERNAL hooking:** in NMODL, slo1egl19 declares
```
EXTERNAL megl19_egl19, hegl19_egl19
```
which means it reads EGL-19's m and h gates from a co-inserted egl19
mechanism. In Brian2, this is automatic — both channels' state variables
live in the same NeuronGroup eqs and are accessible to each other by name
(`m_egl19`, `h_egl19` for our EGL-19 module).

The slo1egl19 BREAKPOINT uses `m * hegl19_egl19` for current rather than just
`m` (it picks up EGL-19's h gate as part of its current calculation). This
encodes the 1:1 stoichiometry (one slo1egl19 channel per egl19 channel,
modulated by egl19's inactivation).

---

## Brian2 translation skeleton

```python
SLO1_EGL19_EQS = """
# SLO-1+EGL-19 coupled BK channel: nanodomain Ca from V (Lluís-Buchholz formula).
# slo1egl19_caCALC is the deterministic V-dependent nanodomain Ca in μM.
# fabs(...) handled by abs() in Brian2.
slo1egl19_caCALC = (
    abs(slo1egl19_gsc * (v_mV - slo1egl19_eca) * 1e-3)
    / (8.0 * 3.14 * slo1egl19_r * slo1egl19_d * slo1egl19_FARADAY)
    * exp(-slo1egl19_r / sqrt(slo1egl19_d / (slo1egl19_kb * slo1egl19_b)))
    * 1e6 * 1e-3
) + slo1egl19_fondo : 1
# kcm, kop, kom rate functions with caCALC + V dependence:
slo1egl19_kcm  = slo1egl19_wom * exp(-slo1egl19_wyx * v_mV) / (1.0 + (slo1egl19_fondo / slo1egl19_kyx)**slo1egl19_nyx) : 1
slo1egl19_kom  = slo1egl19_wom * exp(-slo1egl19_wyx * v_mV) / (1.0 + (slo1egl19_caCALC / slo1egl19_kyx)**slo1egl19_nyx) : 1
slo1egl19_kop  = slo1egl19_wop * exp(-slo1egl19_wxy * v_mV) / (1.0 + (slo1egl19_kxy / slo1egl19_caCALC)**slo1egl19_nxy) : 1
# alpha1, beta1 from EGL-19 actegl19/tactegl19. We approximate by reading
# directly from EGL-19's Brian2 state (already present in the cell eqs):
# alpha1 = egl19_minf / egl19_mtau ; beta1 = (1/egl19_mtau) - alpha1
slo1egl19_alpha1 = egl19_minf / egl19_mtau : 1
slo1egl19_beta1  = (1.0 / egl19_mtau) - slo1egl19_alpha1 : 1
# mminf and tslo1 from the kop/kom/kcm rates:
slo1egl19_mminf = (m_egl19 * slo1egl19_kop * (slo1egl19_alpha1 + slo1egl19_beta1 + slo1egl19_kcm)) / (
    (slo1egl19_kop + slo1egl19_kom) * (slo1egl19_kcm + slo1egl19_alpha1)
    + slo1egl19_beta1 * slo1egl19_kcm
) : 1
slo1egl19_tslo1 = (slo1egl19_alpha1 + slo1egl19_beta1 + slo1egl19_kcm) / (
    (slo1egl19_kop + slo1egl19_kom) * (slo1egl19_kcm + slo1egl19_alpha1)
    + slo1egl19_beta1 * slo1egl19_kcm
) : 1
# State variable:
dm_slo1egl19/dt = (slo1egl19_mminf - m_slo1egl19) / (slo1egl19_tslo1 * ms) : 1
# Channel current density (mA/cm²): coupled to EGL-19 h gate.
ik_slo1egl19_mAcm2 = slo1egl19_gbar * m_slo1egl19 * h_egl19 * (v_mV - slo1egl19_ek) : 1
# Parameters:
slo1egl19_gsc : 1
slo1egl19_eca : 1
slo1egl19_r : 1
slo1egl19_d : 1
slo1egl19_FARADAY : 1
slo1egl19_kb : 1
slo1egl19_b : 1
slo1egl19_fondo : 1
slo1egl19_wom : 1
slo1egl19_wyx : 1
slo1egl19_kyx : 1
slo1egl19_nyx : 1
slo1egl19_wop : 1
slo1egl19_wxy : 1
slo1egl19_kxy : 1
slo1egl19_nxy : 1
slo1egl19_gbar : 1
slo1egl19_ek : 1
"""
```

**Key constraint:** the slo1egl19 module REQUIRES `egl19_minf`, `egl19_mtau`,
`m_egl19`, `h_egl19` to be present in the same NeuronGroup eqs. So
slo1egl19 cannot be used standalone — it needs EGL-19 in the same cell.

Validation will require [leak + egl19 + slo1egl19] cell construction.

---

## Validation strategy

Build [leak + egl19 + slo1egl19] cell at AVAL geometry. Voltage-clamp at 11
holds (-80 to +40 mV). Compare ik_total trajectories to NEURON.

Note: NEURON's slo1egl19 EXTERNAL declarations require both modules inserted
into same section. The cell construction is straightforward.

Expected behavior: at depolarized V (where egl19 m·h product gives high ica
and thus high nanodomain ca), slo1egl19 activates substantially. At rest,
slo1egl19 is essentially closed.

---

## Risks

1. The closed-form calcium(V) formula uses several constants in non-standard
   units (gsc in S, r in m, d in μm²/s — NOT SI). NMODL handles unit conversion
   via the `1e-3` and `1e6 * 1e-3` factors in the formula. Brian2 must use
   identical scales. Verified by direct reading of source.

2. NMODL's `pi=3.14` is a low-precision constant. We match exactly (not π).

3. The mminf/tslo1 formulas have potential numerical issues if denominators
   approach zero. At physiological V/Ca, the rate constants are bounded
   away from zero, so this should be safe. Watch for NaN/Inf during validation.

4. Brian2's `**` exponent operator with non-integer exponents may have
   numerical issues. Switch to exp(n*log(x)) form if needed.
