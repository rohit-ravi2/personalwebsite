"""
SLO-1+EGL-19 coupled BK channel — Brian2 translation of slo1egl19.mod.

Phase β run #2 Phase E deliverable.

Source: nicoletti_2024/slo1egl19.mod
Citations: Nicoletti et al. 2019/2024

Architectural decision (see wave2/artifacts/slo1_coupled_architecture.md):
match Nicoletti's closed-form `calcium(V)` formula exactly. No sub-membrane
state variable; deterministic V-dependent nanodomain Ca.

Channel structure
-----------------

State variable: m (single gate, complex Markov-derived kinetic scheme).
Current: ik = gbar * m * hegl19_egl19 * (v - ek)

The 1:1 stoichiometry encodes that each slo1egl19 channel is coupled to one
egl19 channel, modulated by egl19's inactivation (h gate). Hence the
multiplication by `hegl19_egl19` in the current.

Internal calcium calculation (Lluís-Buchholz / Alvarez nanodomain formula):

  calcium(V) = |gsc·(V-eca)·1e-3| / (8·π·r·d·FARADAY)
             × exp(-r/√(d/(kb·b)))
             × 1e6 × 1e-3
             + fondo

with gsc=40 pS, r=13 nm, d=250 μm²/s, kb=500e6/M-s, b=30 μM, FARADAY=96485,
eca=60 mV, fondo=0.05 μM, π=3.14 (NMODL constant, not real π).

Rate-constant scheme:
  kcm(V) = wom * exp(-wyx*V) / (1 + (fondo/kyx)^nyx)
  kom(V) = wom * exp(-wyx*V) / (1 + (calcium(V)/kyx)^nyx)
  kop(V) = wop * exp(-wxy*V) / (1 + (kxy/calcium(V))^nxy)

α1, β1 from EGL-19 m gate (rate transitions in/out of activated state):
  α1 = egl19_minf / egl19_mtau
  β1 = (1/egl19_mtau) - α1

m steady-state and time constant:
  mminf = (egl19_m · kop · (α1+β1+kcm)) / ((kop+kom)·(kcm+α1) + β1·kcm)
  tslo1 = (α1+β1+kcm) / ((kop+kom)·(kcm+α1) + β1·kcm)

Default parameters from PARAMETER block (slo1egl19.mod, identical kop/kom/kcm
constants to slo1iso since both are SLO-1 channels):
  fondo  = 0.05 μM
  ek     = (set by cell)
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
  r      = 13 nm = 13e-9 m
  d      = 250e-12 m²/s (NMODL says 250e-12 (um2/s) but treat as raw numeric for unit consistency w/ NMODL output)
           Actually NMODL declaration: d=250e-12 (um2/s). Numerically 250e-12.
           [Phase E note: matching NMODL's raw numerics is what matters for translation correctness.]
  kb     = 500e6 /M-s
  b      = 30e-6 M
  gsc    = 40e-12 S
  pi     = 3.14 (NMODL constant)
  FARADAY = 96485 coul/mol

This module REQUIRES `egl19_minf`, `egl19_mtau`, `m_egl19`, `h_egl19` to be
present in the same NeuronGroup eqs (typically by also inserting EGL-19).

Note on egl19's mminf in the coupled formula
---------------------------------------------
NMODL formula uses `megl19_egl19` (the EGL-19 m gate state) in mminf calculation.
Brian2 reads the same state variable as `m_egl19` from the EGL-19 module.
"""
from __future__ import annotations


SLO1_EGL19_PARAMS = {
    "fondo_uM":  0.05,
    "wom":       3.152961,
    "wyx":       0.012643,
    "kyx":       34.338784,
    "nyx":       0.000100,
    "wop":       0.156217,
    "wxy":       -0.027527,
    "kxy":       55.726186,
    "nxy":       1.299198,
    "r":         13e-9,       # m (nanodomain radius)
    "d":         250e-12,     # NMODL: declared (um2/s); numerically 250e-12
    "kb":        500e6,       # /M-s
    "b":         30e-6,       # M
    "gsc":       40e-12,      # S (single-channel conductance)
    "pi":        3.14,        # NMODL constant (not real π)
    "FARADAY":   96485.0,     # coul/mol
    "eca_mV":    60.0,
    "gbar_slo1egl19_Scm2": 0.11,
    "ek_mV":     -80.0,
}


SLO1_EGL19_EQS = """
# SLO-1+EGL-19 coupled BK channel: nanodomain Ca from V (closed form), kinetic Markov scheme.
# REQUIRES egl19_minf, egl19_mtau, m_egl19, h_egl19 in the same eqs (EGL-19 module).
# Internal nanodomain calcium (μM):
slo1egl19_caCALC = (
    abs(slo1egl19_gsc * (v_mV - slo1egl19_eca) * 1e-3)
    / (8.0 * slo1egl19_pi * slo1egl19_r * slo1egl19_d * slo1egl19_FARADAY)
    * exp(-slo1egl19_r / sqrt(slo1egl19_d / (slo1egl19_kb * slo1egl19_b)))
    * 1e6 * 1e-3
) + slo1egl19_fondo : 1
# Rate constants:
slo1egl19_kcm = slo1egl19_wom * exp(-slo1egl19_wyx * v_mV) / (
    1.0 + exp(slo1egl19_nyx * log(slo1egl19_fondo / slo1egl19_kyx))
) : 1
slo1egl19_kom = slo1egl19_wom * exp(-slo1egl19_wyx * v_mV) / (
    1.0 + exp(slo1egl19_nyx * log(slo1egl19_caCALC / slo1egl19_kyx))
) : 1
slo1egl19_kop = slo1egl19_wop * exp(-slo1egl19_wxy * v_mV) / (
    1.0 + exp(slo1egl19_nxy * log(slo1egl19_kxy / slo1egl19_caCALC))
) : 1
# α1, β1 from EGL-19's actegl19/tactegl19 (= egl19_minf / egl19_mtau and its complement):
slo1egl19_alpha1 = egl19_minf / egl19_mtau : 1
slo1egl19_beta1  = (1.0 / egl19_mtau) - slo1egl19_alpha1 : 1
# mminf and tslo1:
slo1egl19_mminf_denom = (
    (slo1egl19_kop + slo1egl19_kom) * (slo1egl19_kcm + slo1egl19_alpha1)
    + slo1egl19_beta1 * slo1egl19_kcm
) : 1
slo1egl19_mminf = (
    m_egl19 * slo1egl19_kop * (slo1egl19_alpha1 + slo1egl19_beta1 + slo1egl19_kcm)
) / slo1egl19_mminf_denom : 1
slo1egl19_tslo1 = (
    slo1egl19_alpha1 + slo1egl19_beta1 + slo1egl19_kcm
) / slo1egl19_mminf_denom : 1
# State variable:
dm_slo1egl19/dt = (slo1egl19_mminf - m_slo1egl19) / (slo1egl19_tslo1 * ms) : 1
# Channel current density (mA/cm²): coupled via h_egl19 (1:1 stoichiometry).
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
slo1egl19_pi : 1
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


def slo1egl19_apply_params(group, gbar_Scm2: float | None = None,
                            ek_mV: float | None = None,
                            eca_mV: float | None = None,
                            params_override: dict | None = None) -> None:
    p = dict(SLO1_EGL19_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_slo1egl19_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV
    if eca_mV is not None:
        # F18 finding: in cells with multiple USEION ca mechanisms (e.g. AIY's
        # egl19 + slo1egl19), NEURON's ion_style overrides user-set seg.eca with
        # Nernst-computed eca. The Brian2 reproduction must match by passing
        # the runtime eca explicitly. Default 60 mV remains the published-script
        # nominal value for cells where ion_style preserves user-set eca.
        p["eca_mV"] = eca_mV

    name_map = {
        "gsc":      "slo1egl19_gsc",
        "eca_mV":   "slo1egl19_eca",
        "r":        "slo1egl19_r",
        "d":        "slo1egl19_d",
        "FARADAY":  "slo1egl19_FARADAY",
        "kb":       "slo1egl19_kb",
        "b":        "slo1egl19_b",
        "fondo_uM": "slo1egl19_fondo",
        "pi":       "slo1egl19_pi",
        "wom":      "slo1egl19_wom",
        "wyx":      "slo1egl19_wyx",
        "kyx":      "slo1egl19_kyx",
        "nyx":      "slo1egl19_nyx",
        "wop":      "slo1egl19_wop",
        "wxy":      "slo1egl19_wxy",
        "kxy":      "slo1egl19_kxy",
        "nxy":      "slo1egl19_nxy",
        "gbar_slo1egl19_Scm2": "slo1egl19_gbar",
        "ek_mV":    "slo1egl19_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def slo1egl19_init_states(group, v_mV: float = -60.0,
                           egl19_minf_v: float | None = None,
                           egl19_mtau_v: float | None = None,
                           egl19_m_init: float | None = None) -> None:
    """Initialize m_slo1egl19 to NMODL INITIAL block: m = mminf at v_mV.

    mminf requires:
      - calcium(v_mV) from the closed-form formula
      - α1 = egl19_minf(v_mV) / egl19_mtau(v_mV)
      - β1 = (1/egl19_mtau(v_mV)) - α1
      - egl19_m at v_mV (= egl19_minf(v_mV) since EGL-19 init = minf)
      - kop, kom, kcm at v_mV with calcium(v_mV)

    If egl19_minf_v / egl19_mtau_v / egl19_m_init are not provided, we compute
    them from EGL-19 defaults (assumes egl19 is the standard module).
    """
    import numpy as np
    from channels.egl19 import EGL19_PARAMS

    if egl19_minf_v is None or egl19_mtau_v is None:
        ep = EGL19_PARAMS
        egl19_minf_v = 1.0 / (1.0 + np.exp(-(v_mV - ep["va_egl19"] + ep["shift"]) / ep["ka_egl19"]))
        egl19_mtau_v = (
            ep["pdg1"]
            + ep["pdg2"] * np.exp(-(v_mV - ep["pdg3"] + ep["shift"])**2 / ep["pdg4"]**2)
            + ep["pdg5"] * np.exp(-(v_mV - ep["pdg6"] + ep["shift"])**2 / ep["pdg7"]**2)
        ) * ep["ctm19"]
    if egl19_m_init is None:
        egl19_m_init = egl19_minf_v

    p = SLO1_EGL19_PARAMS
    # Calcium(V) at v_mV
    ca = (
        abs(p["gsc"] * (v_mV - p["eca_mV"]) * 1e-3)
        / (8.0 * p["pi"] * p["r"] * p["d"] * p["FARADAY"])
        * np.exp(-p["r"] / np.sqrt(p["d"] / (p["kb"] * p["b"])))
        * 1e6 * 1e-3
    ) + p["fondo_uM"]

    # Rate constants
    kcm = p["wom"] * np.exp(-p["wyx"] * v_mV) / (1.0 + (p["fondo_uM"] / p["kyx"]) ** p["nyx"])
    kom = p["wom"] * np.exp(-p["wyx"] * v_mV) / (1.0 + (ca / p["kyx"]) ** p["nyx"])
    kop = p["wop"] * np.exp(-p["wxy"] * v_mV) / (1.0 + (p["kxy"] / ca) ** p["nxy"])

    alpha1 = egl19_minf_v / egl19_mtau_v
    beta1 = (1.0 / egl19_mtau_v) - alpha1

    denom = (kop + kom) * (kcm + alpha1) + beta1 * kcm
    mminf = egl19_m_init * kop * (alpha1 + beta1 + kcm) / denom

    group.m_slo1egl19 = float(mminf)


# Standard interface
NAME = "slo1egl19"
EQS = SLO1_EGL19_EQS
apply_params = slo1egl19_apply_params
init_states = slo1egl19_init_states
