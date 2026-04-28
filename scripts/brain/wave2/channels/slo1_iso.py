"""
SLO-1 isolated BK channel — Brian2 translation of slo1iso.mod.

Phase β run #2 Phase D deliverable.

Source: nicoletti_2024/slo1iso.mod
Citation: Nicoletti et al. 2024

Channel structure
-----------------

SLO-1 isolated is a Ca²⁺-and-voltage-gated K channel (BK family) WITHOUT
nanodomain coupling to a specific Ca channel. It reads bulk `cai` (which in
Nicoletti's actual cells is NEURON's static default 5e-5 mM = 50 nM, since
neither cadiff nor caintra1 is inserted in cells using slo1iso — verified by
F12).

State variable: m (single gate).
Current: ik = gbar * m * (v - ek)

The activation kinetic scheme is parameterized via voltage- and [Ca]-dependent
rate constants kop and kom (and a v-dependent kcm). Steady-state minf and
time constant tslo1 are derived from these:

  v0 = (1/(wyx-wxy)) * (log(wom/wop) + log(1+(kxy/(ca·1e3))^nxy)
                                       - log(1+((ca·1e3)/kyx)^nyx))
  s0 = 1/(wyx-wxy)
  minf = 1 / (1 + exp(-(v - v0)/s0))
  mtau = (exp(wxy*v)/wop) * (1 + (kxy/(ca·1e3))^nxy) * minf * c1

(The `ca·1e3` in the formula converts cai from mM to μM, since formula
constants kxy, kyx are in μM units. The c1 is a global scaling factor for
mtau, default 1.)

Parameters from PARAMETER block:
  fondo  = 0.05 μM (resting [Ca] reference)
  ek     = -80 mV
  eca    = 60 mV
  gbar   = 0.11 S/cm² (default)
  wom    = 3.152961 /ms
  wyx    = 0.012643 /mV
  kyx    = 34.338784 μM
  nyx    = 0.000100 (1)
  wop    = 0.156217 /ms
  wxy    = -0.027527 /mV
  kxy    = 55.726186 /ms
  nxy    = 1.299198 (1)
  c1     = 1 (1)

NOTE on nxy: NMODL parameter `nxy=0.000100` is essentially zero, making
`(kxy/(ca·1e3))^nxy ≈ 1.0` always. Practically the (kxy/ca)^nxy term is
constant ≈ 1.0; the Ca dependence enters only through `((ca·1e3)/kyx)^nyx`
in v0 and through `wxy*v` in mtau prefactor.

Default static cai
------------------
In Nicoletti's AIY cell (the primary slo1iso user), no Ca-pool is inserted,
so cai stays at NEURON's default `cai0_ca_ion = 5e-5 mM`. Brian2 translation
uses a constant `cai_mM = 5e-5` parameter by default. Future Phase F+ work
may replace with a dynamic pool when cells warrant it.
"""
from __future__ import annotations


SLO1_ISO_PARAMS = {
    "fondo_uM":   0.05,
    "wom":        3.152961,
    "wyx":        0.012643,
    "kyx":        34.338784,
    "nyx":        0.000100,
    "wop":        0.156217,
    "wxy":        -0.027527,
    "kxy":        55.726186,
    "nxy":        1.299198,
    "c1":         1.0,
    "gbar_slo1iso_Scm2": 0.11,
    "ek_mV":      -80.0,
    "cai_mM_static": 5e-5,  # NEURON default cai0_ca_ion
}


# slo1iso eqs. The NMODL formula uses `ca` in mM as the parameter cai (the channel
# reads cai via USEION ca READ cai). The `ca*1e3` conversion to μM happens inside
# the formula. We replicate this faithfully.
#
# Brian2 has an issue with non-integer exponents (^nxy where nxy=0.000100). We use
# `exp(nxy * log(x))` for numerical stability. Brian2's eqs parser uses ** for
# exponent — but with non-integer power, it reduces to exp/log internally anyway.
# We use exp/log explicitly for clarity and safety.
#
# Also note: the formula has nested `log(...)` which expects the inner argument
# to be positive. (ca*1e3) at cai=5e-5 mM = 5e-2 μM is positive. (kyx/(ca*1e3))
# at cai=5e-5 mM = 34.34/0.05 = 686.8, also positive. Numerical safety OK.

SLO1_ISO_EQS = """
# SLO-1 isolated BK channel: m gate, voltage- and Ca-dependent.
# cai_mM is the bulk intracellular Ca (parameter or pool-state).
# Convert to μM-numeric for formulas:
slo1iso_ca_uM = cai_mM * 1000.0 : 1
# Steady-state v0 and s0:
slo1iso_s0 = 1.0 / (slo1iso_wyx - slo1iso_wxy) : 1
slo1iso_v0 = slo1iso_s0 * (
    log(slo1iso_wom / slo1iso_wop)
    + log(1.0 + exp(slo1iso_nxy * log(slo1iso_kxy / slo1iso_ca_uM)))
    - log(1.0 + exp(slo1iso_nyx * log(slo1iso_ca_uM / slo1iso_kyx)))
) : 1
slo1iso_minf = 1.0 / (1.0 + exp(-(v_mV - slo1iso_v0) / slo1iso_s0)) : 1
slo1iso_mtau = (
    (exp(slo1iso_wxy * v_mV) / slo1iso_wop)
    * (1.0 + exp(slo1iso_nxy * log(slo1iso_kxy / slo1iso_ca_uM)))
    * (1.0 / (1.0 + exp(-(v_mV - slo1iso_v0) / slo1iso_s0)))
) * slo1iso_c1 : 1
# State variable:
dm_slo1iso/dt = (slo1iso_minf - m_slo1iso) / (slo1iso_mtau * ms) : 1
# Channel current density (mA/cm²):
ik_slo1iso_mAcm2 = slo1iso_gbar * m_slo1iso * (v_mV - slo1iso_ek) : 1
# Parameters:
slo1iso_wom : 1
slo1iso_wyx : 1
slo1iso_kyx : 1
slo1iso_nyx : 1
slo1iso_wop : 1
slo1iso_wxy : 1
slo1iso_kxy : 1
slo1iso_nxy : 1
slo1iso_c1 : 1
slo1iso_gbar : 1
slo1iso_ek : 1
cai_mM : 1
"""


def slo1iso_apply_params(group, gbar_Scm2: float | None = None,
                         ek_mV: float | None = None,
                         cai_mM: float | None = None,
                         params_override: dict | None = None) -> None:
    p = dict(SLO1_ISO_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_slo1iso_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV
    if cai_mM is not None:
        p["cai_mM_static"] = cai_mM

    name_map = {
        "wom":        "slo1iso_wom",
        "wyx":        "slo1iso_wyx",
        "kyx":        "slo1iso_kyx",
        "nyx":        "slo1iso_nyx",
        "wop":        "slo1iso_wop",
        "wxy":        "slo1iso_wxy",
        "kxy":        "slo1iso_kxy",
        "nxy":        "slo1iso_nxy",
        "c1":         "slo1iso_c1",
        "gbar_slo1iso_Scm2": "slo1iso_gbar",
        "ek_mV":      "slo1iso_ek",
        "cai_mM_static": "cai_mM",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def slo1iso_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_slo1iso to its voltage- and Ca-dependent SS at v_mV.

    Uses the parameter values currently set on the group (cai_mM, slo1iso_*).
    """
    import numpy as np
    p = SLO1_ISO_PARAMS
    cai_mM = float(group.cai_mM[0]) if hasattr(group.cai_mM, "__getitem__") else float(group.cai_mM)
    ca_uM = cai_mM * 1000.0
    s0 = 1.0 / (p["wyx"] - p["wxy"])
    v0 = s0 * (
        np.log(p["wom"] / p["wop"])
        + np.log(1.0 + (p["kxy"] / ca_uM) ** p["nxy"])
        - np.log(1.0 + (ca_uM / p["kyx"]) ** p["nyx"])
    )
    minf = 1.0 / (1.0 + np.exp(-(v_mV - v0) / s0))
    group.m_slo1iso = float(minf)


# Standard interface (renames for run_phase_c style validator)
NAME = "slo1iso"
EQS = SLO1_ISO_EQS
apply_params = slo1iso_apply_params
init_states = slo1iso_init_states
