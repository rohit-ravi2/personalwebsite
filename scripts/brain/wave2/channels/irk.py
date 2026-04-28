"""
IRK inwardly-rectifying K channel — Brian2 translation of irk.mod.

Wave 2 option α-1 CP2 deliverable.

Source: nicoletti_2024/irk.mod
Citations:
  - Nicoletti et al. PLoS ONE 2019, https://doi.org/10.1371/journal.pone.0218738
  - Nicoletti et al. PLoS ONE 2024, https://doi.org/10.1371/journal.pone.0298105

Channel structure
-----------------

IRK is a Kir-family inwardly-rectifying K channel with **a single gate**:
  m : activation only (no inactivation gate)
Current: ik = gbar * m * (v - ek)

Steady-state and tau functions (note +30 offset in minf — this is the
hyperpolarization-shift characteristic of inward rectifiers):

  minf = 1 / (1 + exp((v - va_kir + 30) / ka_kir))
  mtau = p1tmkir / (exp(-(v - p2tmkir) / p3tmkir)
                  + exp((v - p4tmkir) / p5tmkir)) + p6tmkir

Parameters (PARAMETER block, default values):
  va_kir   = -52 mV         (activation V_half, before +30 shift → effective -22 mV center)
  ka_kir   = 13 mV          (slope; positive = activates as V hyperpolarizes)
  p1tmkir  = 17.0752        (tau scale)
  p2tmkir  = -17.8258       (tau center A)
  p3tmkir  = 20.3154        (tau slope A)
  p4tmkir  = -43.4414       (tau center B)
  p5tmkir  = 11.1691        (tau slope B)
  p6tmkir  = 3.8329         (tau offset)
  gbar     = 0.65 (annotated nS/cm² in NMODL; **value-treated as S/cm²** by
             upstream gScm2() wrapper that scales nS → S/cm² uniformly)
  ek       = (ion default; cells set externally)

NMODL gbar units note
---------------------

irk.mod declares `gbar=.65 (nS/cm2)` while shk1/shl1/kqt3/etc declare
`(S/cm2)`. The unit annotation is misleading — Nicoletti's `gScm2()` wrapper
treats all gbar values uniformly as nS at the cell level, scaling to S/cm²
via `g[i] * 1e-9 / surf`. The `(nS/cm2)` annotation in irk.mod appears to be
a documentation typo; the runtime value is set by the wrapper to a S/cm²-scale
number. Empirical confirmation: NEURON simulations using `seg.irk.gbar = X`
produce currents matching the S/cm² interpretation, not nS/cm².

Translation pattern
-------------------

IRK follows the established voltage-gated K pattern (P10 leak-relative scale
check). Single-gate (`m` only), no Ca dependence, no GLOBAL state. The +30 mV
shift in minf and the U-shaped tau (exp(-) + exp(+)) are mechanism-specific
quirks but pose no translation challenge.

Use-case context
----------------

IRK is part of Nicoletti's AVAL cell (4-channel set: IRK + LEAK + EGL19 + NCA),
contributing the inward-rectifier component that makes AVA's I-V approximately
linear at hyperpolarized holds. Required for option α-1 CP3 AVA cell construction.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter defaults from irk.mod
# ---------------------------------------------------------------------------

IRK_PARAMS = {
    "va_kir":  -52.0,
    "ka_kir":  13.0,
    "p1tmkir": 17.0752,
    "p2tmkir": -17.8258,
    "p3tmkir": 20.3154,
    "p4tmkir": -43.4414,
    "p5tmkir": 11.1691,
    "p6tmkir": 3.8329,
    "gbar_irk_Scm2": 0.65,
    "ek_mV":   -80.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

IRK_EQS = """
# IRK Kir-family inwardly-rectifying K channel: single m gate.
# Note: minf has a +30 mV shift inside the exp argument — characteristic of
# inward rectifiers (activates as V hyperpolarizes below va_kir-30).
irk_minf = 1.0 / (1.0 + exp((v_mV - irk_va + 30.0) / irk_ka)) : 1
# Tau: U-shape (sum of two exponentials, one positive one negative slope).
irk_mtau = (
    irk_p1tmkir
    / (exp(-(v_mV - irk_p2tmkir) / irk_p3tmkir)
       + exp((v_mV - irk_p4tmkir) / irk_p5tmkir))
    + irk_p6tmkir
) : 1
# State variable (gating fraction 0..1):
dm_irk/dt = (irk_minf - m_irk) / (irk_mtau * ms) : 1
# Channel current density (mA/cm²): I = g * m * (V_mV - E_mV).
ik_irk_mAcm2 = irk_gbar * m_irk * (v_mV - irk_ek) : 1
# Parameters:
irk_va : 1
irk_ka : 1
irk_p1tmkir : 1
irk_p2tmkir : 1
irk_p3tmkir : 1
irk_p4tmkir : 1
irk_p5tmkir : 1
irk_p6tmkir : 1
irk_gbar : 1
irk_ek : 1
"""


def irk_apply_params(group, gbar_Scm2: float | None = None,
                     ek_mV: float | None = None,
                     params_override: dict | None = None) -> None:
    p = dict(IRK_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_irk_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va_kir":  "irk_va",
        "ka_kir":  "irk_ka",
        "p1tmkir": "irk_p1tmkir",
        "p2tmkir": "irk_p2tmkir",
        "p3tmkir": "irk_p3tmkir",
        "p4tmkir": "irk_p4tmkir",
        "p5tmkir": "irk_p5tmkir",
        "p6tmkir": "irk_p6tmkir",
        "gbar_irk_Scm2": "irk_gbar",
        "ek_mV":   "irk_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def irk_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_irk to voltage-clamped SS at v_mV."""
    import numpy as np
    p = IRK_PARAMS
    minf = 1.0 / (1.0 + np.exp((v_mV - p["va_kir"] + 30.0) / p["ka_kir"]))
    group.m_irk = float(minf)


# Standard interface for validate_phase_c_channels.validate_channel:
NAME = "irk"
EQS = IRK_EQS
apply_params = irk_apply_params
init_states = irk_init_states
