"""
EGL-19 L-type calcium channel — Brian2 translation of egl19.mod.

Phase β CP2 deliverable.

Source: nicoletti_2024/egl19.mod
"L-type channels — From Nicoletti et al. PloS One 2019"

Channel structure
-----------------

State variables: m (activation), h (inactivation).
Current: ica = gbar * m * h * (v - eca)
NO Ca-dependent inactivation in this parameterization (m, h are
voltage-only). Verified by reading egl19.mod DERIVATIVE block:
  m' = (minf - m) / mtau
  h' = (hinf - h) / htau
  rates(v) computes minf, hinf, mtau, htau from voltage only.

Steady-state functions (`shift = 10` mV applied uniformly):
  minf = 1 / (1 + exp(-(v - va_egl19 + shift)/ka_egl19))
  hinf = (p1/(1+exp(-(v-p2+shift)/p3)) + p4) *
         (p5/(1+exp((v-p6+shift)/p7)) + p8)
  mtau = (pdg1 + pdg2*exp(-(v-pdg3+shift)^2/pdg4^2)
          + pdg5*exp(-(v-pdg6+shift)^2/pdg7^2)) * ctm19
  htau = pds1 * (pds2*pds3/(1+exp((v-pds4+shift)/pds5))
                  + pds6 + pds7*pds8/(1+exp((v-pds9+shift)/pds10))
                  + pds11)

Parameters from egl19.mod (PARAMETER block):
  va_egl19 = 5.6 mV
  ka_egl19 = 7.50 mV
  shift = 10 mV
  p1hegl19 = 1.4314, p2hegl19 = 24.8573 mV, p3hegl19 = 11.9541 mV
  p4hegl19 = 0.1427, p5hegl19 = 5.9589, p6hegl19 = -10.5428 mV
  p7hegl19 = 8.0552 mV, p8hegl19 = 0.6038
  pdg1 = 2.3359 ms, pdg2 = 2.9324 ms, pdg3 = 5.2357 mV, pdg4 = 6.0 mV
  pdg5 = 1.8739 ms, pdg6 = 1.3930 mV, pdg7 = 30.0 mV, ctm19 = 1
  pds1 = 0.4, pds2 = 0.55, pds3 = 81.1179 ms, pds4 = -22.9723 mV
  pds5 = 5 mV, pds6 = 43.0937 ms, pds7 = 0.9, pds8 = 40.4885 ms
  pds9 = 28.7251 mV, pds10 = 3.7125 mV, pds11 = 0
  gbar = 1.55 S/cm² (default; cell-specific value comes from g0 vector)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Parameter dictionary (defaults from egl19.mod)
# ---------------------------------------------------------------------------

EGL19_PARAMS = {
    # Activation
    "va_egl19": 5.6,
    "ka_egl19": 7.50,
    "shift": 10.0,
    # Inactivation steady-state
    "p1hegl19": 1.4314,
    "p2hegl19": 24.8573,
    "p3hegl19": 11.9541,
    "p4hegl19": 0.1427,
    "p5hegl19": 5.9589,
    "p6hegl19": -10.5428,
    "p7hegl19": 8.0552,
    "p8hegl19": 0.6038,
    # Activation tau
    "pdg1": 2.3359,
    "pdg2": 2.9324,
    "pdg3": 5.2357,
    "pdg4": 6.0,
    "pdg5": 1.8739,
    "pdg6": 1.3930,
    "pdg7": 30.0,
    "ctm19": 1.0,
    # Inactivation tau
    "pds1": 0.4,
    "pds2": 0.55,
    "pds3": 81.1179,
    "pds4": -22.9723,
    "pds5": 5.0,
    "pds6": 43.0937,
    "pds7": 0.9,
    "pds8": 40.4885,
    "pds9": 28.7251,
    "pds10": 3.7125,
    "pds11": 0.0,
    # Conductance + reversal (defaults; cells override)
    "gbar_egl19_Scm2": 1.55,
    "eca_mV": 60.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

# Notes on Brian2 expression:
# - v_mV is the membrane potential as a numeric mV value (extracted from `v`).
# - All voltage parameters are dimensionless mV-numerics.
# - Time parameters (mtau, htau) are dimensionless ms-numerics; we convert
#   to per-second for Brian2 ODE integration via /ms scaling.
# - ica_egl19_Acm2 is the contribution of EGL-19 to the cell's ica in A/cm²
#   (NEURON convention is mA/cm²; we use A/cm² internally to keep Brian2 SI
#   units, then convert at the cell-current summation site).
#
# Actually for consistency with calcium_pool.py (which uses ica_mAcm2), we
# expose ica_mAcm2 here too. Channels write to a per-channel ica_<name>_mAcm2,
# and the parent eqs sums them:
#   ica_mAcm2 = ica_egl19_mAcm2 + ica_other_channels_mAcm2 + ...

EGL19_EQS = """
# EGL-19 L-type Ca channel: m, h gates (voltage-only).
# Steady-state functions (with shift):
egl19_minf = 1.0 / (1.0 + exp(-(v_mV - egl19_va + egl19_shift) / egl19_ka)) : 1
egl19_hinf = (
    (egl19_p1 / (1.0 + exp(-(v_mV - egl19_p2 + egl19_shift) / egl19_p3)) + egl19_p4)
    * (egl19_p5 / (1.0 + exp((v_mV - egl19_p6 + egl19_shift) / egl19_p7)) + egl19_p8)
) : 1
# Time constants (in ms-numeric):
egl19_mtau = (
    egl19_pdg1
    + egl19_pdg2 * exp(-(v_mV - egl19_pdg3 + egl19_shift)**2 / egl19_pdg4**2)
    + egl19_pdg5 * exp(-(v_mV - egl19_pdg6 + egl19_shift)**2 / egl19_pdg7**2)
) * egl19_ctm19 : 1
egl19_htau = egl19_pds1 * (
    (egl19_pds2 * egl19_pds3 / (1.0 + exp((v_mV - egl19_pds4 + egl19_shift) / egl19_pds5)))
    + egl19_pds6
    + (egl19_pds7 * egl19_pds8 / (1.0 + exp((v_mV - egl19_pds9 + egl19_shift) / egl19_pds10)))
    + egl19_pds11
) : 1
# State variables (dimensionless gating fractions 0..1):
dm_egl19/dt = (egl19_minf - m_egl19) / (egl19_mtau * ms) : 1
dh_egl19/dt = (egl19_hinf - h_egl19) / (egl19_htau * ms) : 1
# Channel current density (mA/cm²): I = g * (V_mV - E_mV) with units
#   (S/cm²) × mV = (A/cm²)·(V/V) × 1e-3 V = mA/cm² (factor 1e-3 from V→mV cancels A→mA).
ica_egl19_mAcm2 = egl19_gbar * m_egl19 * h_egl19 * (v_mV - egl19_eca) : 1
# Parameters (set at NeuronGroup construction):
egl19_va : 1
egl19_ka : 1
egl19_shift : 1
egl19_p1 : 1
egl19_p2 : 1
egl19_p3 : 1
egl19_p4 : 1
egl19_p5 : 1
egl19_p6 : 1
egl19_p7 : 1
egl19_p8 : 1
egl19_pdg1 : 1
egl19_pdg2 : 1
egl19_pdg3 : 1
egl19_pdg4 : 1
egl19_pdg5 : 1
egl19_pdg6 : 1
egl19_pdg7 : 1
egl19_ctm19 : 1
egl19_pds1 : 1
egl19_pds2 : 1
egl19_pds3 : 1
egl19_pds4 : 1
egl19_pds5 : 1
egl19_pds6 : 1
egl19_pds7 : 1
egl19_pds8 : 1
egl19_pds9 : 1
egl19_pds10 : 1
egl19_pds11 : 1
egl19_gbar : 1
egl19_eca : 1
"""


def egl19_apply_params(group, gbar_Scm2: float | None = None,
                       eca_mV: float | None = None,
                       shift_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    """Apply EGL-19 parameters to a Brian2 NeuronGroup whose eqs include EGL19_EQS.

    Parameters
    ----------
    group : NeuronGroup
        Must have all egl19_* parameters in its eqs.
    gbar_Scm2 : float, optional
        Conductance density. Defaults to EGL19_PARAMS['gbar_egl19_Scm2'].
    eca_mV : float, optional
        Ca reversal. Defaults to 60 mV.
    shift_mV : float, optional
        Override shift parameter (default 10 mV).
    params_override : dict, optional
        Map of {egl19_<name>: value} to override any default.
    """
    p = dict(EGL19_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_egl19_Scm2"] = gbar_Scm2
    if eca_mV is not None:
        p["eca_mV"] = eca_mV
    if shift_mV is not None:
        p["shift"] = shift_mV

    # Map parameter dict keys → eqs variable names
    name_map = {
        "va_egl19": "egl19_va",
        "ka_egl19": "egl19_ka",
        "shift": "egl19_shift",
        "p1hegl19": "egl19_p1",
        "p2hegl19": "egl19_p2",
        "p3hegl19": "egl19_p3",
        "p4hegl19": "egl19_p4",
        "p5hegl19": "egl19_p5",
        "p6hegl19": "egl19_p6",
        "p7hegl19": "egl19_p7",
        "p8hegl19": "egl19_p8",
        "pdg1": "egl19_pdg1",
        "pdg2": "egl19_pdg2",
        "pdg3": "egl19_pdg3",
        "pdg4": "egl19_pdg4",
        "pdg5": "egl19_pdg5",
        "pdg6": "egl19_pdg6",
        "pdg7": "egl19_pdg7",
        "ctm19": "egl19_ctm19",
        "pds1": "egl19_pds1",
        "pds2": "egl19_pds2",
        "pds3": "egl19_pds3",
        "pds4": "egl19_pds4",
        "pds5": "egl19_pds5",
        "pds6": "egl19_pds6",
        "pds7": "egl19_pds7",
        "pds8": "egl19_pds8",
        "pds9": "egl19_pds9",
        "pds10": "egl19_pds10",
        "pds11": "egl19_pds11",
        "gbar_egl19_Scm2": "egl19_gbar",
        "eca_mV": "egl19_eca",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def egl19_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_egl19 and h_egl19 to their voltage-clamped steady states.

    Mirrors NEURON's INITIAL block: m=minf, h=hinf at the holding potential.
    """
    import numpy as np
    p = EGL19_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_egl19"] + p["shift"]) / p["ka_egl19"]))
    hinf = ((p["p1hegl19"] / (1.0 + np.exp(-(v_mV - p["p2hegl19"] + p["shift"]) / p["p3hegl19"])) + p["p4hegl19"])
            * (p["p5hegl19"] / (1.0 + np.exp((v_mV - p["p6hegl19"] + p["shift"]) / p["p7hegl19"])) + p["p8hegl19"]))
    group.m_egl19 = float(minf)
    group.h_egl19 = float(hinf)
