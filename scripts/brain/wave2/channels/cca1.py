"""
CCA-1 T-type voltage-gated calcium channel — Brian2 translation of cca1.mod.

Wave 2 cellular extension Wave 2/RIM CP1 deliverable.

Source: nicoletti_2024/cca1.mod
"T-type, CCA-1, channels — From Nicoletti et al. PloS One 2019"

Channel structure
-----------------

State variables: m (activation), h (inactivation).
Current: ica = gbar * m * m * h * (v - eca)   [note m^2 power]
NO Ca-dependent inactivation in this parameterization (m, h are
voltage-only). Verified by reading cca1.mod DERIVATIVE block:
  m' = (minf(v) - m) / mtau(v)
  h' = (hinf(v) - h) / htau(v)

Steady-state functions:
  minf = 1 / (1 + exp(-(v - va_cca1 + sscca1)/(ka_cca1 * fcca)))
  hinf = 1 / (1 + exp((v - vi_cca1 + sshcca1)/(ki_cca1 * f2cca1)))
Time constants:
  mtau = (p1tmcca1 / (1 + exp(-(v - p2tmcca1 + stmcca1)/(p3tmcca1 * f3ca)))
          + p4tmcca1) * constmcca1
  htau = (p1thcca1 / (1 + exp((v - p2thcca1 + sthcca1)/(p3thcca1 * f4ca)))
          + p4thcca1) * consthcca1

Parameters from cca1.mod (PARAMETER block):
  va_cca1=-42.65 mV, ka_cca1=1.7 mV
  sscca1=15, sthcca1=15, sshcca1=15, stmcca1=30
  vi_cca1=-58 mV, ki_cca1=7 mV
  fcca=1.4, f2cca1=1.15, f3ca=1.7, f4ca=1.1
  constmcca1=0.5, consthcca1=0.08
  p1tmcca1=40 ms, p2tmcca1=-62.5393 mV, p3tmcca1=-12.4758 mV, p4tmcca1=0.6947 ms
  p1thcca1=280 ms, p2thcca1=-60.7312 mV, p3thcca1=8.5224 mV, p4thcca1=19.7456 ms
  gbar = 0.7 S/cm² (default; cell-specific value comes from g vector)
  eca = 60 mV (default; cell may override per F18 with multi-USEION-ca cells)

F18 awareness
-------------
This channel reads eca via USEION. In RIM (3 USEION ca: cca1+unc2+egl19) NEURON's
ion_style silently overrides any user-set seg.eca. cell builder MUST pass
explicit eca_mV (e.g. 127.59 for celsius=6.3, cai=5e-5, cao=2). See
cellular_validation_findings.md F18 entry.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter dictionary (defaults from cca1.mod)
# ---------------------------------------------------------------------------

CCA1_PARAMS = {
    # Activation steady-state
    "va_cca1":   -42.65,
    "ka_cca1":   1.7,
    "sscca1":    15.0,
    "fcca":      1.4,
    # Inactivation steady-state
    "vi_cca1":   -58.0,
    "ki_cca1":   7.0,
    "sshcca1":   15.0,
    "f2cca1":    1.15,
    # Activation tau
    "p1tmcca1":  40.0,
    "p2tmcca1":  -62.5393,
    "p3tmcca1":  -12.4758,
    "p4tmcca1":  0.6947,
    "stmcca1":   30.0,
    "f3ca":      1.7,
    "constmcca1": 0.5,
    # Inactivation tau
    "p1thcca1":  280.0,
    "p2thcca1":  -60.7312,
    "p3thcca1":  8.5224,
    "p4thcca1":  19.7456,
    "sthcca1":   15.0,
    "f4ca":      1.1,
    "consthcca1": 0.08,
    # Conductance + reversal
    "gbar_cca1_Scm2": 0.7,
    "eca_mV":    60.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

CCA1_EQS = """
# CCA-1 T-type Ca channel: m, h gates (voltage-only).
# Steady-state:
cca1_minf = 1.0 / (1.0 + exp(-(v_mV - cca1_va + cca1_ssm) / (cca1_ka * cca1_fcca))) : 1
cca1_hinf = 1.0 / (1.0 + exp((v_mV - cca1_vi + cca1_ssh) / (cca1_ki * cca1_f2))) : 1
# Time constants (ms):
cca1_mtau = (
    cca1_p1tm / (1.0 + exp(-(v_mV - cca1_p2tm + cca1_stm) / (cca1_p3tm * cca1_f3)))
    + cca1_p4tm
) * cca1_constm : 1
cca1_htau = (
    cca1_p1th / (1.0 + exp((v_mV - cca1_p2th + cca1_sth) / (cca1_p3th * cca1_f4)))
    + cca1_p4th
) * cca1_consth : 1
# State variables (gating fractions 0..1):
dm_cca1/dt = (cca1_minf - m_cca1) / (cca1_mtau * ms) : 1
dh_cca1/dt = (cca1_hinf - h_cca1) / (cca1_htau * ms) : 1
# Channel current density (mA/cm²): ica = gbar * m^2 * h * (v - eca)
ica_cca1_mAcm2 = cca1_gbar * m_cca1 * m_cca1 * h_cca1 * (v_mV - cca1_eca) : 1
# Parameters:
cca1_va : 1
cca1_ka : 1
cca1_ssm : 1
cca1_fcca : 1
cca1_vi : 1
cca1_ki : 1
cca1_ssh : 1
cca1_f2 : 1
cca1_p1tm : 1
cca1_p2tm : 1
cca1_p3tm : 1
cca1_p4tm : 1
cca1_stm : 1
cca1_f3 : 1
cca1_constm : 1
cca1_p1th : 1
cca1_p2th : 1
cca1_p3th : 1
cca1_p4th : 1
cca1_sth : 1
cca1_f4 : 1
cca1_consth : 1
cca1_gbar : 1
cca1_eca : 1
"""


def cca1_apply_params(group, gbar_Scm2: float | None = None,
                      eca_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    """Apply CCA-1 parameters to a Brian2 NeuronGroup whose eqs include CCA1_EQS."""
    p = dict(CCA1_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_cca1_Scm2"] = gbar_Scm2
    if eca_mV is not None:
        p["eca_mV"] = eca_mV

    name_map = {
        "va_cca1":         "cca1_va",
        "ka_cca1":         "cca1_ka",
        "sscca1":          "cca1_ssm",
        "fcca":            "cca1_fcca",
        "vi_cca1":         "cca1_vi",
        "ki_cca1":         "cca1_ki",
        "sshcca1":         "cca1_ssh",
        "f2cca1":          "cca1_f2",
        "p1tmcca1":        "cca1_p1tm",
        "p2tmcca1":        "cca1_p2tm",
        "p3tmcca1":        "cca1_p3tm",
        "p4tmcca1":        "cca1_p4tm",
        "stmcca1":         "cca1_stm",
        "f3ca":            "cca1_f3",
        "constmcca1":      "cca1_constm",
        "p1thcca1":        "cca1_p1th",
        "p2thcca1":        "cca1_p2th",
        "p3thcca1":        "cca1_p3th",
        "p4thcca1":        "cca1_p4th",
        "sthcca1":         "cca1_sth",
        "f4ca":            "cca1_f4",
        "consthcca1":      "cca1_consth",
        "gbar_cca1_Scm2":  "cca1_gbar",
        "eca_mV":          "cca1_eca",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def cca1_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_cca1, h_cca1 to voltage-clamped steady states."""
    import numpy as np
    p = CCA1_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_cca1"] + p["sscca1"]) / (p["ka_cca1"] * p["fcca"])))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vi_cca1"] + p["sshcca1"]) / (p["ki_cca1"] * p["f2cca1"])))
    group.m_cca1 = float(minf)
    group.h_cca1 = float(hinf)


# Standard interface
NAME = "cca1"
EQS = CCA1_EQS
apply_params = cca1_apply_params
init_states = cca1_init_states
