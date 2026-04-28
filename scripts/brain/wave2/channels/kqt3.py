"""
KQT-3 M-current K channel — Brian2 translation of kqt3.mod.

Phase β run #2 Phase C.4 deliverable.

Source: nicoletti_2024/kqt3.mod
Citation: Nicoletti et al. 2019/2024

Channel structure
-----------------

KQT-3 is the M-current K channel with **four state variables**:
  mf : fast activation
  ms : slow activation (same minf as mf, different tau)
  s  : slow inactivation
  w  : voltage-dependent gating modifier

Current: ik = gbar * (0.3*mf + 0.7*ms) * s * w * (v - ek)

Steady-state functions:
  minf = 1 / (1 + exp(-(v - va_kqt3 + constkqt3) / ka_kqt3))
  winf = w1 + w2 / (1 + exp((v + w3) / w4))
  sinf = sq1 + sq2 / (1 + exp((v + sq3) / sq4))

Time constants:
  mtauf = (p1tmfkqt3 / (1 + ((v + p2tmfkqt3) / p3tmfkqt3)**2)) * ckqt3
  mtaus = (p1tmskqt3 - p2tmskqt3 / (1 + 10**(p3tmskqt3 * (p4tmskqt3 - v)))
                       - p5tmskqt3 / (1 + 10**(p6tmskqt3 * (v + p7tmskqt3)))) * ckqt3
  s tau = 500 ms (constant)
  tw    = (tw1 + tw2 / (1 + ((v + tw3) / tw4)**2)) * ckqt3

Default parameters from PARAMETER block (kqt3.mod).
"""
from __future__ import annotations


KQT3_PARAMS = {
    "va_kqt3":     -12.6726,
    "ka_kqt3":     15.8008,
    "constkqt3":   10.0,
    "p1tmskqt3":   5503.0,
    "p2tmskqt3":   5345.4,
    "p3tmskqt3":   -0.02827,
    "p4tmskqt3":   -23.9,
    "p5tmskqt3":   4590.6,
    "p6tmskqt3":   -0.0357,
    "p7tmskqt3":   14.15,
    "p1tmfkqt3":   395.3,
    "p2tmfkqt3":   38.1,
    "p3tmfkqt3":   33.59,
    "ckqt3":       0.1,
    "w1":          0.49,
    "w2":          0.51,
    "w3":          1.084,
    "w4":          28.78,
    "tw1":         5.44,
    "tw2":         29.2,
    "tw3":         48.09,
    "tw4":         48.83,
    "sq1":         0.34,
    "sq2":         0.66,
    "sq3":         45.3,
    "sq4":         12.3,
    "tsq1":        5000.0,
    "gbar_kqt3_Scm2": 0.55,
    "ek_mV":       -80.0,
}


# Note: NMODL `10^x` is `10**x` in Brian2/Python — but Brian2's eqs string parser
# may not handle `**` cleanly with negative exponents. Use the equivalent
# `exp(x * log(10))` for safety.
KQT3_EQS = """
# KQT-3 M-current K channel: mf, ms (activation, fast/slow); s (slow inact); w (modifier).
# Steady-state functions:
kqt3_minf = 1.0 / (1.0 + exp(-(v_mV - kqt3_va + kqt3_const) / kqt3_ka)) : 1
kqt3_winf = kqt3_w1 + kqt3_w2 / (1.0 + exp((v_mV + kqt3_w3) / kqt3_w4)) : 1
kqt3_sinf = kqt3_sq1 + kqt3_sq2 / (1.0 + exp((v_mV + kqt3_sq3) / kqt3_sq4)) : 1
# Time constants:
kqt3_mtauf = (kqt3_p1tmfkqt3 / (1.0 + ((v_mV + kqt3_p2tmfkqt3) / kqt3_p3tmfkqt3)**2)) * kqt3_ck : 1
kqt3_mtaus = (
    kqt3_p1tmskqt3
    - kqt3_p2tmskqt3 / (1.0 + exp((kqt3_p3tmskqt3 * (kqt3_p4tmskqt3 - v_mV)) * 2.302585093))
    - kqt3_p5tmskqt3 / (1.0 + exp((kqt3_p6tmskqt3 * (v_mV + kqt3_p7tmskqt3)) * 2.302585093))
) * kqt3_ck : 1
kqt3_tw = (kqt3_tw1 + kqt3_tw2 / (1.0 + ((v_mV + kqt3_tw3) / kqt3_tw4)**2)) * kqt3_ck : 1
# State variables (gating fractions 0..1):
dmf_kqt3/dt = (kqt3_minf - mf_kqt3) / (kqt3_mtauf * ms) : 1
dms_kqt3/dt = (kqt3_minf - ms_kqt3) / (kqt3_mtaus * ms) : 1
ds_kqt3/dt = (kqt3_sinf - s_kqt3) / (500.0 * ms) : 1
dw_kqt3/dt = (kqt3_winf - w_kqt3) / (kqt3_tw * ms) : 1
# Channel current density (mA/cm²):
ik_kqt3_mAcm2 = kqt3_gbar * (0.3 * mf_kqt3 + 0.7 * ms_kqt3) * s_kqt3 * w_kqt3 * (v_mV - kqt3_ek) : 1
# Parameters:
kqt3_va : 1
kqt3_ka : 1
kqt3_const : 1
kqt3_p1tmskqt3 : 1
kqt3_p2tmskqt3 : 1
kqt3_p3tmskqt3 : 1
kqt3_p4tmskqt3 : 1
kqt3_p5tmskqt3 : 1
kqt3_p6tmskqt3 : 1
kqt3_p7tmskqt3 : 1
kqt3_p1tmfkqt3 : 1
kqt3_p2tmfkqt3 : 1
kqt3_p3tmfkqt3 : 1
kqt3_ck : 1
kqt3_w1 : 1
kqt3_w2 : 1
kqt3_w3 : 1
kqt3_w4 : 1
kqt3_tw1 : 1
kqt3_tw2 : 1
kqt3_tw3 : 1
kqt3_tw4 : 1
kqt3_sq1 : 1
kqt3_sq2 : 1
kqt3_sq3 : 1
kqt3_sq4 : 1
kqt3_gbar : 1
kqt3_ek : 1
"""


def kqt3_apply_params(group, gbar_Scm2: float | None = None,
                      ek_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    p = dict(KQT3_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_kqt3_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va_kqt3":     "kqt3_va",
        "ka_kqt3":     "kqt3_ka",
        "constkqt3":   "kqt3_const",
        "p1tmskqt3":   "kqt3_p1tmskqt3",
        "p2tmskqt3":   "kqt3_p2tmskqt3",
        "p3tmskqt3":   "kqt3_p3tmskqt3",
        "p4tmskqt3":   "kqt3_p4tmskqt3",
        "p5tmskqt3":   "kqt3_p5tmskqt3",
        "p6tmskqt3":   "kqt3_p6tmskqt3",
        "p7tmskqt3":   "kqt3_p7tmskqt3",
        "p1tmfkqt3":   "kqt3_p1tmfkqt3",
        "p2tmfkqt3":   "kqt3_p2tmfkqt3",
        "p3tmfkqt3":   "kqt3_p3tmfkqt3",
        "ckqt3":       "kqt3_ck",
        "w1":          "kqt3_w1",
        "w2":          "kqt3_w2",
        "w3":          "kqt3_w3",
        "w4":          "kqt3_w4",
        "tw1":         "kqt3_tw1",
        "tw2":         "kqt3_tw2",
        "tw3":         "kqt3_tw3",
        "tw4":         "kqt3_tw4",
        "sq1":         "kqt3_sq1",
        "sq2":         "kqt3_sq2",
        "sq3":         "kqt3_sq3",
        "sq4":         "kqt3_sq4",
        "gbar_kqt3_Scm2": "kqt3_gbar",
        "ek_mV":       "kqt3_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def kqt3_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = KQT3_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_kqt3"] + p["constkqt3"]) / p["ka_kqt3"]))
    winf = p["w1"] + p["w2"] / (1.0 + np.exp((v_mV + p["w3"]) / p["w4"]))
    sinf = p["sq1"] + p["sq2"] / (1.0 + np.exp((v_mV + p["sq3"]) / p["sq4"]))
    group.mf_kqt3 = float(minf)
    group.ms_kqt3 = float(minf)
    group.w_kqt3 = float(winf)
    group.s_kqt3 = float(sinf)


# Standard interface
NAME = "kqt3"
EQS = KQT3_EQS
apply_params = kqt3_apply_params
init_states = kqt3_init_states
