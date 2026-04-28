"""
SHK-1 voltage-gated K channel — Brian2 translation of shk1.mod.

Phase β run #2 Phase C.1 deliverable.

Source: nicoletti_2024/shk1.mod
Citations:
  - Fawcett 2006 (inactivation kinetics)
  - Liu et al 2018 (activation kinetics)
  - Nicoletti et al 2024 (parameter integration)

Channel structure
-----------------

State variables: m (activation), h (inactivation).
Current: ik = gbar * m * h * (v - ek)

Voltage-only gates (no Ca dependence). Inactivation tau is constant
(htau = pthshak = 1400 ms — very slow).

Steady-state functions (from shk1.mod):
  minf = 1 / (1 + exp(-(v - vashak) / kashak))
  hinf = 1 / (1 + exp((v - vishak) / kishak))
  mtau = ptmshak1 / (exp(-(v - (ptmshak2+shiftV05))/ptmshak4)
                   + exp((v - (ptmshak2+shiftV05))/ptmshak3)) + ptmshak5
  htau = pthshak (constant)

Parameters (PARAMETER block, default values):
  vashak  = 2 mV
  kashak  = 10 mV
  vishak  = -6.95 mV
  kishak  = 5.8 mV
  ptmshak1 = 26.5715 ms
  ptmshak2 = -33.7416 mV
  ptmshak3 = 15.7579 mV
  ptmshak4 = 15.3649 mV
  ptmshak5 = 1.9900 ms
  shiftV05 = 0 mV
  pthshak  = 1400 ms
  gbar     = 0.1 S/cm² (default)
  ek       = (ion default; cells set externally)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameter defaults from shk1.mod
# ---------------------------------------------------------------------------

SHK1_PARAMS = {
    "vashak":   2.0,
    "kashak":   10.0,
    "vishak":   -6.95,
    "kishak":   5.8,
    "ptmshak1": 26.571450568169027,
    "ptmshak2": -33.741611800716130,
    "ptmshak3": 15.757936311607475,
    "ptmshak4": 15.364937728953288,
    "ptmshak5": 1.990037272604829,
    "shiftV05": 0.0,
    "pthshak":  1400.0,
    "gbar_shk1_Scm2": 0.1,
    "ek_mV":    -80.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

SHK1_EQS = """
# SHK-1 K channel: m, h gates (voltage-only).
shk1_minf = 1.0 / (1.0 + exp(-(v_mV - shk1_vashak) / shk1_kashak)) : 1
shk1_hinf = 1.0 / (1.0 + exp((v_mV - shk1_vishak) / shk1_kishak)) : 1
shk1_mtau = (
    shk1_ptmshak1
    / (exp(-(v_mV - (shk1_ptmshak2 + shk1_shiftV05)) / shk1_ptmshak4)
       + exp((v_mV - (shk1_ptmshak2 + shk1_shiftV05)) / shk1_ptmshak3))
    + shk1_ptmshak5
) : 1
shk1_htau = shk1_pthshak : 1
# State variables (gating fractions 0..1):
dm_shk1/dt = (shk1_minf - m_shk1) / (shk1_mtau * ms) : 1
dh_shk1/dt = (shk1_hinf - h_shk1) / (shk1_htau * ms) : 1
# Channel current density (mA/cm²): I = g * (V_mV - E_mV) — see P10 leak-relative scale.
ik_shk1_mAcm2 = shk1_gbar * m_shk1 * h_shk1 * (v_mV - shk1_ek) : 1
# Parameters:
shk1_vashak : 1
shk1_kashak : 1
shk1_vishak : 1
shk1_kishak : 1
shk1_ptmshak1 : 1
shk1_ptmshak2 : 1
shk1_ptmshak3 : 1
shk1_ptmshak4 : 1
shk1_ptmshak5 : 1
shk1_shiftV05 : 1
shk1_pthshak : 1
shk1_gbar : 1
shk1_ek : 1
"""


def shk1_apply_params(group, gbar_Scm2: float | None = None,
                       ek_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    p = dict(SHK1_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_shk1_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "vashak":   "shk1_vashak",
        "kashak":   "shk1_kashak",
        "vishak":   "shk1_vishak",
        "kishak":   "shk1_kishak",
        "ptmshak1": "shk1_ptmshak1",
        "ptmshak2": "shk1_ptmshak2",
        "ptmshak3": "shk1_ptmshak3",
        "ptmshak4": "shk1_ptmshak4",
        "ptmshak5": "shk1_ptmshak5",
        "shiftV05": "shk1_shiftV05",
        "pthshak":  "shk1_pthshak",
        "gbar_shk1_Scm2": "shk1_gbar",
        "ek_mV":    "shk1_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def shk1_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_shk1 and h_shk1 to voltage-clamped SS at v_mV."""
    import numpy as np
    p = SHK1_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["vashak"]) / p["kashak"]))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vishak"]) / p["kishak"]))
    group.m_shk1 = float(minf)
    group.h_shk1 = float(hinf)


# Standard interface for validate_phase_c_channels.validate_channel:
NAME = "shk1"
EQS = SHK1_EQS
apply_params = shk1_apply_params
init_states = shk1_init_states
