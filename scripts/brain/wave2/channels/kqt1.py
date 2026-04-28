"""
KQT-1 voltage-gated K channel — Brian2 translation of kqt1.mod.

Wave 2 cellular extension Option B CP1 deliverable.

Source: nicoletti_2024/kqt1.mod
Citation: Nicoletti et al. 2024 (PLoS ONE 19(3): e0298105)

Channel structure
-----------------

KQT-1 is a KCNQ-family K channel with **2 state variables** (m, s).
This is **distinct** from KQT-3 (4-state: mf, ms, s, w with extra w gate
modifier and fast/slow activation pair).

Current: ik = gbar * m * s * (v - ek)

Steady-state functions (from kqt1.mod):
  minf = 1 / (1 + exp(-(v - va) / ka))
  sinf = (s1 / (1 + exp((v - s2) / s3)))
       + (s4 / (1 + exp((v - s5) / s6)))   # double-Boltzmann inactivation

Time constants:
  mtau = (p2tmkqt1 / (1 + exp((p3tmkqt1 - v) / p4tmkqt1))) + p1tmkqt1
  stau = p1tskqt1 + (p2tskqt1 / (1 + ((v - p3tskqt1) / p4tskqt1)^2))

Parameters (PARAMETER block defaults):
  va        = -17.6053 mV
  ka        =   9.5843 mV
  p1tmkqt1  =  10      ms
  p2tmkqt1  = 895.9    ms
  p3tmkqt1  = -18.01   mV
  p4tmkqt1  =  31.04   mV
  s1        =   0.41
  s2        = -86.84   mV
  s3        =  15.05   mV
  s4        =   0.59
  s5        =  70.13   mV
  s6        =  13.37   mV
  p1tskqt1  = 1077     ms
  p2tskqt1  = 185845   ms (note: very slow)
  p3tskqt1  =  39.44   mV
  p4tskqt1  =   7.34   mV
  gbar      =   2.9    S/cm² (default in mod; AIY uses parameter-vector value)
  ek        = (ion default; cells set externally)

Notes
-----
- The s-gate's stau has an extremely slow component (185845 ms ≈ 186 s).
  Within typical 200-ms voltage-clamp windows this s-gate barely changes;
  it dominates only on very long simulation horizons. Steady-state
  initialization (init_states with v_mV=-60 mV by default) lets us bypass
  the long settle.
- The mod file has **no GLOBAL declarations** — all parameters are PARAMETER
  block (effectively per-segment RANGE in NEURON). Standard F1/F3 pattern,
  not F2 (no GLOBAL state propagation issue).
- `^2` in NMODL → `**2` in Python — Brian2's eqs string parser handles this.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter defaults from kqt1.mod
# ---------------------------------------------------------------------------

KQT1_PARAMS = {
    "va":         -17.6053,
    "ka":          9.5843,
    "p1tmkqt1":   10.0,
    "p2tmkqt1":  895.9,
    "p3tmkqt1":  -18.01,
    "p4tmkqt1":   31.04,
    "s1":          0.41,
    "s2":        -86.84,
    "s3":         15.05,
    "s4":          0.59,
    "s5":         70.13,
    "s6":         13.37,
    "p1tskqt1": 1077.0,
    "p2tskqt1": 185845.0,
    "p3tskqt1":   39.44,
    "p4tskqt1":    7.34,
    "gbar_kqt1_Scm2": 2.9,
    "ek_mV":     -80.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

KQT1_EQS = """
# KQT-1 K channel: m (activation), s (slow inactivation, double-Boltzmann).
kqt1_minf = 1.0 / (1.0 + exp(-(v_mV - kqt1_va) / kqt1_ka)) : 1
kqt1_sinf = (kqt1_s1 / (1.0 + exp((v_mV - kqt1_s2) / kqt1_s3))) + (kqt1_s4 / (1.0 + exp((v_mV - kqt1_s5) / kqt1_s6))) : 1
kqt1_mtau = (kqt1_p2tmkqt1 / (1.0 + exp((kqt1_p3tmkqt1 - v_mV) / kqt1_p4tmkqt1))) + kqt1_p1tmkqt1 : 1
kqt1_stau = kqt1_p1tskqt1 + (kqt1_p2tskqt1 / (1.0 + ((v_mV - kqt1_p3tskqt1) / kqt1_p4tskqt1)**2)) : 1
# State variables (gating fractions 0..1):
dm_kqt1/dt = (kqt1_minf - m_kqt1) / (kqt1_mtau * ms) : 1
ds_kqt1/dt = (kqt1_sinf - s_kqt1) / (kqt1_stau * ms) : 1
# Channel current density (mA/cm²):
ik_kqt1_mAcm2 = kqt1_gbar * m_kqt1 * s_kqt1 * (v_mV - kqt1_ek) : 1
# Parameters:
kqt1_va : 1
kqt1_ka : 1
kqt1_p1tmkqt1 : 1
kqt1_p2tmkqt1 : 1
kqt1_p3tmkqt1 : 1
kqt1_p4tmkqt1 : 1
kqt1_s1 : 1
kqt1_s2 : 1
kqt1_s3 : 1
kqt1_s4 : 1
kqt1_s5 : 1
kqt1_s6 : 1
kqt1_p1tskqt1 : 1
kqt1_p2tskqt1 : 1
kqt1_p3tskqt1 : 1
kqt1_p4tskqt1 : 1
kqt1_gbar : 1
kqt1_ek : 1
"""


def kqt1_apply_params(group, gbar_Scm2: float | None = None,
                      ek_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    p = dict(KQT1_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_kqt1_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va":             "kqt1_va",
        "ka":             "kqt1_ka",
        "p1tmkqt1":       "kqt1_p1tmkqt1",
        "p2tmkqt1":       "kqt1_p2tmkqt1",
        "p3tmkqt1":       "kqt1_p3tmkqt1",
        "p4tmkqt1":       "kqt1_p4tmkqt1",
        "s1":             "kqt1_s1",
        "s2":             "kqt1_s2",
        "s3":             "kqt1_s3",
        "s4":             "kqt1_s4",
        "s5":             "kqt1_s5",
        "s6":             "kqt1_s6",
        "p1tskqt1":       "kqt1_p1tskqt1",
        "p2tskqt1":       "kqt1_p2tskqt1",
        "p3tskqt1":       "kqt1_p3tskqt1",
        "p4tskqt1":       "kqt1_p4tskqt1",
        "gbar_kqt1_Scm2": "kqt1_gbar",
        "ek_mV":          "kqt1_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def kqt1_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_kqt1 and s_kqt1 to voltage-clamped SS at v_mV.

    The s-gate's slow component (p2tskqt1 ≈ 186 s) makes SS-init essential —
    otherwise short voltage-clamp windows can't capture the gate's true
    holding-potential equilibrium.
    """
    import numpy as np
    p = KQT1_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va"]) / p["ka"]))
    sinf = (
        (p["s1"] / (1.0 + np.exp((v_mV - p["s2"]) / p["s3"])))
        + (p["s4"] / (1.0 + np.exp((v_mV - p["s5"]) / p["s6"])))
    )
    group.m_kqt1 = float(minf)
    group.s_kqt1 = float(sinf)


# Standard interface for validate_phase_c_channels.validate_channel:
NAME = "kqt1"
EQS = KQT1_EQS
apply_params = kqt1_apply_params
init_states = kqt1_init_states
