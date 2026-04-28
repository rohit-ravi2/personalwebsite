"""
UNC-103 voltage-gated K channel — Brian2 translation of unc103.mod.

Wave 2 option α-1 CP1 deliverable.

Source: nicoletti_2024/unc103.mod
Citations:
  - Nicoletti et al. PLoS ONE 2024, 19(3): e0298105.
  - https://doi.org/10.1371/journal.pone.0298105

Channel structure
-----------------

UNC-103 is a voltage-gated K channel (ERG-family-like) with **two gates**:
  m : activation
  h : inactivation
Current: ik = gbar * m * h * (v - ek)

Steady-state functions:
  minf = 1 / (1 + exp(-(v - va) / ka))
  hinf = 1 / (1 + exp((v - vi) / ki))

Time constants — note the PRODUCT structure (NOT sum):
  mtau = ((tm1 / (1 + exp((v - tm2) / tm3))) + tm4)
        * ((tm1 / (1 + exp(-(v - tm2) / tm3))) + tm4)
  htau = ((th1 / (1 + exp((v - th2) / th3))) + th4)
        * ((th1 / (1 + exp(-(v - th2) / th3))) + th4)

This is the standard Nicoletti "double-sigmoid product" tau form, also seen
in egl2.mod and a few others. The product of (1/(1+exp(+))) and (1/(1+exp(-)))
gives a peaked tau profile centered near tm2 / th2 with magnitude ~tm1²/4
at the peak (when exp arguments are 0).

Parameters (PARAMETER block, default values):
  va  = -15.1 mV     (activation V_half)
  ka  = 7.85 mV      (activation slope)
  vi  = -48 mV       (inactivation V_half)
  ki  = 28 mV        (inactivation slope)
  tm1 = 87.4088 ms   (m peak tau scale)
  tm2 = -28.3339 mV  (m tau center)
  tm3 = 13.0998 mV   (m tau slope)
  tm4 = 0.2562 ms    (m tau offset)
  th1 = 8.1559 ms    (h peak tau scale)
  th2 = -25.2890 mV  (h tau center)
  th3 = 29.5074 mV   (h tau slope)
  th4 = 0.2300 ms    (h tau offset)
  gbar = 2.9 S/cm² (default)
  ek = (ion default; cells set externally)

Translation pattern
-------------------

UNC-103 follows the **same pattern as SHK-1, SHL-1, KQT-3** — a clean
voltage-gated K channel with no GLOBAL state in NMODL. STATE block is
standard `m h` per-cell (NMODL implicit RANGE-default). The pre-flight
α prompt's "F2 GLOBAL→per-cell" framing was a misattribution — F2 is
about caintra1's Ca trajectory (P2 in translation_patterns.md), and
UNC-103's NMODL declares no GLOBAL variables. No special handling needed
beyond the established voltage-gated K pattern.

Use-case context
----------------

UNC-103 is part of Nicoletti's AVAR cell (5-channel set), NOT AVAL
(4-channel set). Translated here for completeness of Brian2 channel
coverage (future AVAR work). Not used in Wave 2 option α-1 CP3/CP4
since CP3 targets the true 4-channel AVAL.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter defaults from unc103.mod
# ---------------------------------------------------------------------------

UNC103_PARAMS = {
    "va":   -15.1,
    "ka":   7.85,
    "vi":   -48.0,
    "ki":   28.0,
    "tm1":  87.4088,
    "tm2":  -28.3339,
    "tm3":  13.0998,
    "tm4":  0.2562,
    "th1":  8.1559,
    "th2":  -25.2890,
    "th3":  29.5074,
    "th4":  0.2300,
    "gbar_unc103_Scm2": 2.9,
    "ek_mV": -80.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

UNC103_EQS = """
# UNC-103 voltage-gated K channel: m, h gates (voltage-only, no Ca dependence).
unc103_minf = 1.0 / (1.0 + exp(-(v_mV - unc103_va) / unc103_ka)) : 1
unc103_hinf = 1.0 / (1.0 + exp((v_mV - unc103_vi) / unc103_ki)) : 1
# Tau functions — PRODUCT of two sigmoids (NOT sum). Peak tau ~tm1²/4 near tm2.
unc103_mtau = (
    (unc103_tm1 / (1.0 + exp((v_mV - unc103_tm2) / unc103_tm3)) + unc103_tm4)
    * (unc103_tm1 / (1.0 + exp(-(v_mV - unc103_tm2) / unc103_tm3)) + unc103_tm4)
) : 1
unc103_htau = (
    (unc103_th1 / (1.0 + exp((v_mV - unc103_th2) / unc103_th3)) + unc103_th4)
    * (unc103_th1 / (1.0 + exp(-(v_mV - unc103_th2) / unc103_th3)) + unc103_th4)
) : 1
# State variables (gating fractions 0..1):
dm_unc103/dt = (unc103_minf - m_unc103) / (unc103_mtau * ms) : 1
dh_unc103/dt = (unc103_hinf - h_unc103) / (unc103_htau * ms) : 1
# Channel current density (mA/cm²): I = g * m * h * (V_mV - E_mV).
# Per P10 leak-relative scale: V in mV, gbar in S/cm² → product in mA/cm² directly.
ik_unc103_mAcm2 = unc103_gbar * m_unc103 * h_unc103 * (v_mV - unc103_ek) : 1
# Parameters:
unc103_va : 1
unc103_ka : 1
unc103_vi : 1
unc103_ki : 1
unc103_tm1 : 1
unc103_tm2 : 1
unc103_tm3 : 1
unc103_tm4 : 1
unc103_th1 : 1
unc103_th2 : 1
unc103_th3 : 1
unc103_th4 : 1
unc103_gbar : 1
unc103_ek : 1
"""


def unc103_apply_params(group, gbar_Scm2: float | None = None,
                        ek_mV: float | None = None,
                        params_override: dict | None = None) -> None:
    p = dict(UNC103_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_unc103_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va":   "unc103_va",
        "ka":   "unc103_ka",
        "vi":   "unc103_vi",
        "ki":   "unc103_ki",
        "tm1":  "unc103_tm1",
        "tm2":  "unc103_tm2",
        "tm3":  "unc103_tm3",
        "tm4":  "unc103_tm4",
        "th1":  "unc103_th1",
        "th2":  "unc103_th2",
        "th3":  "unc103_th3",
        "th4":  "unc103_th4",
        "gbar_unc103_Scm2": "unc103_gbar",
        "ek_mV": "unc103_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def unc103_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_unc103 and h_unc103 to voltage-clamped SS at v_mV."""
    import numpy as np
    p = UNC103_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va"]) / p["ka"]))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vi"]) / p["ki"]))
    group.m_unc103 = float(minf)
    group.h_unc103 = float(hinf)


# Standard interface for validate_phase_c_channels.validate_channel:
NAME = "unc103"
EQS = UNC103_EQS
apply_params = unc103_apply_params
init_states = unc103_init_states
