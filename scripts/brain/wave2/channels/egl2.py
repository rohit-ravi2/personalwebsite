"""
EGL-2 voltage-gated K channel (EAG family) — Brian2 translation of egl2.mod.

Wave 2 cellular extension Wave 2/RIM CP2 deliverable.

Source: nicoletti_2024/egl2.mod
"egl2 currents model — From Nicoletti et al. PloS One 2019"

Channel structure
-----------------

Single state variable: m (activation only — EGL-2 has no inactivation in this
parameterization).
Current: ik = gbar * m * (v - ek)

Steady-state and tau:
  minf = 1 / (1 + exp(-(v - va_egl2 + stmegl2)/(ka_egl2 * fegl2)))
  mtau = (p1tmegl2 / (1 + exp((v - p2tmegl2 + stmegl2)/p3tmegl2)) + p4tmegl2) * cegl2

Parameters from egl2.mod (PARAMETER block):
  va_egl2  = -6.8594 mV
  ka_egl2  = 14.9131 mV
  stmegl2  = 0
  cegl2    = 0.5
  p1tmegl2 = 16.7800 ms
  p2tmegl2 = -122.5682 mV
  p3tmegl2 = 13.7976 mV
  p4tmegl2 = 8.0969 ms
  fegl2    = 1
  gbar     = 0.85 S/cm² (default; cell-specific value comes from RIM g vector)

EAG (ether-a-go-go) family kinetics note
----------------------------------------
EAG channels typically have:
- relatively shallow activation (ka ~15 mV here — moderate),
- voltage-dependent activation requiring depolarization beyond -7 mV
  for half-activation (va_egl2 = -6.86 mV),
- slow tau peak around v = +123 mV (p2 = -123 mV in the formula's negative-shifted
  Boltzmann term, so the sigmoid centers at v = -p2 = +123 mV when stmegl2=0).
For the operating range of -80 to +40 mV, mtau evolves smoothly from baseline
(p4tmegl2 + low contribution from p1tmegl2/(1+exp(huge)) ≈ 8 ms) at low voltages,
to peak (p4tmegl2 + p1tmegl2/(1+exp(0)) = 8.10 + 16.78/2 = 16.5 ms) around v = +123 mV.
Thus tau increases mildly from ~8/2 = 4 ms (×cegl2=0.5) at -80 mV to ~16.5/2 ≈ 8 ms
near depolarization extreme. **No unusual extreme-tau pattern** for EGL-2
within RIM's operating range — distinct from KQT-1's 186 s s-gate.
Translation is a clean voltage-gated K channel: no GLOBAL state, no inactivation,
no Ca dependence.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter dictionary (defaults from egl2.mod)
# ---------------------------------------------------------------------------

EGL2_PARAMS = {
    "va_egl2":   -6.8594,
    "ka_egl2":   14.9131,
    "stmegl2":   0.0,
    "cegl2":     0.5,
    "p1tmegl2":  16.7800,
    "p2tmegl2":  -122.5682,
    "p3tmegl2":  13.7976,
    "p4tmegl2":  8.0969,
    "fegl2":     1.0,
    "gbar_egl2_Scm2": 0.85,
    "ek_mV":     -80.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

EGL2_EQS = """
# EGL-2 voltage-gated K channel (EAG family): m gate only.
egl2_minf = 1.0 / (1.0 + exp(-(v_mV - egl2_va + egl2_stm) / (egl2_ka * egl2_f))) : 1
egl2_mtau = (
    egl2_p1tm / (1.0 + exp((v_mV - egl2_p2tm + egl2_stm) / egl2_p3tm))
    + egl2_p4tm
) * egl2_c : 1
# State variable:
dm_egl2/dt = (egl2_minf - m_egl2) / (egl2_mtau * ms) : 1
# Channel current density (mA/cm²):
ik_egl2_mAcm2 = egl2_gbar * m_egl2 * (v_mV - egl2_ek) : 1
# Parameters:
egl2_va : 1
egl2_ka : 1
egl2_stm : 1
egl2_c : 1
egl2_p1tm : 1
egl2_p2tm : 1
egl2_p3tm : 1
egl2_p4tm : 1
egl2_f : 1
egl2_gbar : 1
egl2_ek : 1
"""


def egl2_apply_params(group, gbar_Scm2: float | None = None,
                      ek_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    p = dict(EGL2_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_egl2_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va_egl2":         "egl2_va",
        "ka_egl2":         "egl2_ka",
        "stmegl2":         "egl2_stm",
        "cegl2":           "egl2_c",
        "p1tmegl2":        "egl2_p1tm",
        "p2tmegl2":        "egl2_p2tm",
        "p3tmegl2":        "egl2_p3tm",
        "p4tmegl2":        "egl2_p4tm",
        "fegl2":           "egl2_f",
        "gbar_egl2_Scm2":  "egl2_gbar",
        "ek_mV":           "egl2_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def egl2_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = EGL2_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_egl2"] + p["stmegl2"]) / (p["ka_egl2"] * p["fegl2"])))
    group.m_egl2 = float(minf)


# Standard interface
NAME = "egl2"
EQS = EGL2_EQS
apply_params = egl2_apply_params
init_states = egl2_init_states
