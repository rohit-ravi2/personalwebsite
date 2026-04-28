"""
SHL-1 voltage-gated K channel — Brian2 translation of shl1.mod.

Phase β run #2 Phase C.2 deliverable.

Source: nicoletti_2024/shl1.mod
Citations:
  - Fawcett et al. 2006
  - Nicoletti 2019/2024

Channel structure
-----------------

SHL-1 is a Kv4-family A-type K channel with **two inactivation states**: hs (slow)
and hf (fast). Activation m is voltage-dependent. Inactivation gates share the
same hinf (steady-state) but differ in tau.

Current: ik = gbar * m * (a*hf + (1-a)*hs) * (v - ek)
where a = 0.8 (fast inactivation fraction).

Steady-state and tau functions (UNITSOFF in source):
  minf  = 1 / (1 + exp(-(v - vashal + shalsfhit) / kashal))
  hinf  = 1 / (1 + exp((v - vishal + shalsfhit) / kishal))
  mtau  = (ptmshal1/(exp(-(v-ptmshal2)/ptmshal3) + exp((v-ptmshal4)/ptmshal5)) + ptmshal6) / 2
  htauf = (pthfshal1/(1 + exp((v-pthfshal2)/pthfshal3)) + pthfshal4) / 3
  htaus = pthsshal1/(1 + exp((v-pthsshal2)/pthsshal3)) + pthsshal4

Parameters (PARAMETER block, default values):
  vashal=10.26 mV, kashal=16.25 mV, vishal=-40 mV, kishal=8.3 mV
  shalsfhit=10 mV
  ptmshal1=13.8 ms, ptmshal2=-40 mV, ptmshal3=12.9213 mV
  ptmshal4=-40 mV, ptmshal5=6.4876 mV, ptmshal6=1.8849 ms
  pthfshal1=539.1584 ms, pthfshal2=-60 mV, pthfshal3=4.9199 mV, pthfshal4=27.2811 ms
  pthsshal1=8422 ms, pthsshal2=-60 mV, pthsshal3=6.3785 mV, pthsshal4=118.8983 ms
  a=0.8
  gbar=2.9 S/cm² (default)
"""
from __future__ import annotations


SHL1_PARAMS = {
    "vashal":      10.26,
    "kashal":      16.250,
    "vishal":      -40.0,
    "kishal":      8.3,
    "shalsfhit":   10.0,
    "ptmshal1":    13.8,
    "ptmshal2":    -40.0,
    "ptmshal3":    12.9213,
    "ptmshal4":    -40.0,
    "ptmshal5":    6.4876,
    "ptmshal6":    1.8849,
    "pthfshal1":   539.1584,
    "pthfshal2":   -60.0,
    "pthfshal3":   4.9199,
    "pthfshal4":   27.2811,
    "pthsshal1":   8422.0,
    "pthsshal2":   -60.0,
    "pthsshal3":   6.3785,
    "pthsshal4":   118.8983,
    "a":           0.8,
    "gbar_shl1_Scm2": 2.9,
    "ek_mV":       -80.0,
}


SHL1_EQS = """
# SHL-1 Kv4 A-type K channel: m activation + hs (slow) + hf (fast) inactivation.
shl1_minf = 1.0 / (1.0 + exp(-(v_mV - shl1_vashal + shl1_shalsfhit) / shl1_kashal)) : 1
shl1_hinf = 1.0 / (1.0 + exp((v_mV - shl1_vishal + shl1_shalsfhit) / shl1_kishal)) : 1
shl1_mtau = (
    shl1_ptmshal1
    / (exp(-(v_mV - shl1_ptmshal2) / shl1_ptmshal3)
       + exp((v_mV - shl1_ptmshal4) / shl1_ptmshal5))
    + shl1_ptmshal6
) / 2.0 : 1
shl1_htauf = (
    shl1_pthfshal1 / (1.0 + exp((v_mV - shl1_pthfshal2) / shl1_pthfshal3))
    + shl1_pthfshal4
) / 3.0 : 1
shl1_htaus = (
    shl1_pthsshal1 / (1.0 + exp((v_mV - shl1_pthsshal2) / shl1_pthsshal3))
    + shl1_pthsshal4
) : 1
# State variables (gating fractions 0..1):
dm_shl1/dt = (shl1_minf - m_shl1) / (shl1_mtau * ms) : 1
dhf_shl1/dt = (shl1_hinf - hf_shl1) / (shl1_htauf * ms) : 1
dhs_shl1/dt = (shl1_hinf - hs_shl1) / (shl1_htaus * ms) : 1
# Channel current density (mA/cm²):
ik_shl1_mAcm2 = shl1_gbar * m_shl1 * (shl1_a * hf_shl1 + (1.0 - shl1_a) * hs_shl1) * (v_mV - shl1_ek) : 1
# Parameters:
shl1_vashal : 1
shl1_kashal : 1
shl1_vishal : 1
shl1_kishal : 1
shl1_shalsfhit : 1
shl1_ptmshal1 : 1
shl1_ptmshal2 : 1
shl1_ptmshal3 : 1
shl1_ptmshal4 : 1
shl1_ptmshal5 : 1
shl1_ptmshal6 : 1
shl1_pthfshal1 : 1
shl1_pthfshal2 : 1
shl1_pthfshal3 : 1
shl1_pthfshal4 : 1
shl1_pthsshal1 : 1
shl1_pthsshal2 : 1
shl1_pthsshal3 : 1
shl1_pthsshal4 : 1
shl1_a : 1
shl1_gbar : 1
shl1_ek : 1
"""


def shl1_apply_params(group, gbar_Scm2: float | None = None,
                       ek_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    p = dict(SHL1_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_shl1_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "vashal":      "shl1_vashal",
        "kashal":      "shl1_kashal",
        "vishal":      "shl1_vishal",
        "kishal":      "shl1_kishal",
        "shalsfhit":   "shl1_shalsfhit",
        "ptmshal1":    "shl1_ptmshal1",
        "ptmshal2":    "shl1_ptmshal2",
        "ptmshal3":    "shl1_ptmshal3",
        "ptmshal4":    "shl1_ptmshal4",
        "ptmshal5":    "shl1_ptmshal5",
        "ptmshal6":    "shl1_ptmshal6",
        "pthfshal1":   "shl1_pthfshal1",
        "pthfshal2":   "shl1_pthfshal2",
        "pthfshal3":   "shl1_pthfshal3",
        "pthfshal4":   "shl1_pthfshal4",
        "pthsshal1":   "shl1_pthsshal1",
        "pthsshal2":   "shl1_pthsshal2",
        "pthsshal3":   "shl1_pthsshal3",
        "pthsshal4":   "shl1_pthsshal4",
        "a":           "shl1_a",
        "gbar_shl1_Scm2": "shl1_gbar",
        "ek_mV":       "shl1_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def shl1_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = SHL1_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["vashal"] + p["shalsfhit"]) / p["kashal"]))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vishal"] + p["shalsfhit"]) / p["kishal"]))
    group.m_shl1 = float(minf)
    group.hf_shl1 = float(hinf)
    group.hs_shl1 = float(hinf)


# Standard interface
NAME = "shl1"
EQS = SHL1_EQS
apply_params = shl1_apply_params
init_states = shl1_init_states
