"""
UNC-2 P/Q-type voltage-gated calcium channel — Brian2 translation of unc2.mod.

Wave 2 cellular extension Wave 2/RIM CP3 deliverable.

Source: nicoletti_2024/unc2.mod
"P/Q-type channels — From Nicoletti et al. PloS One 2019"

Channel structure
-----------------

State variables: m (activation), h (inactivation).
Current: ica = gbar * m * h * (v - eca)
NO Ca-dependent inactivation (m, h are voltage-only). Verified by reading
unc2.mod DERIVATIVE block:
  m' = (minf - m) / mtau
  h' = (hinf - h) / htau
where rates(v) computes minf, hinf, mtau, htau from voltage only.

Steady-state functions:
  minf = 1 / (1 + exp(-(v - va_unc2 + stm2)/ka_unc2))
  hinf = 1 / (1 + exp((v - vi_unc2 + sth2)/ki_unc2))

Time constants (note unusual mtau form — sum of two exponentials in denominator):
  mtau = (p1tmunc2 / (exp(-(v - p2tmunc2 + shiftmunc2)/(p3tmunc2*fp3))
                    + exp((v - p2tmunc2 + shiftmunc2)/(p4tmunc2*fp4)))
          + p5tmunc2) * constmunc2
  htau = (p1thunc2 / (1 + exp((v - p2thunc2 + shifthunc2)/(p3thunc2*fp5)))
          + p4thunc2 / (1 + exp(-(v - p5thunc2 + shifthunc2)/(p6thunc2*fp5)))) * consthunc2

Parameters from unc2.mod (PARAMETER block):
  va_unc2=-12.17 mV, ka_unc2=3.97 mV, vi_unc2=-52.47 mV, ki_unc2=5.6 mV
  stm2=25 mV, sth2=25 mV
  p1tmunc2=1.4969 ms, p2tmunc2=-8.1761 mV, p3tmunc2=9.0753 mV
  p4tmunc2=15.3456 mV, p5tmunc2=0.1029 ms
  p1thunc2=83.8037 ms, p2thunc2=52.8997 mV, p3thunc2=3.4557 mV
  p4thunc2=72.0995 ms, p5thunc2=23.9009 mV, p6thunc2=3.5903 mV
  fp3=1, fp4=1, fp5=1
  shifthunc2=30, shiftmunc2=30
  consthunc2=1.7, constmunc2=3
  func2=1, f2unc2=1
  gbar=1 S/cm² (default; cell-specific value comes from g vector)
  eca=60 mV (default; F18 may apply in multi-USEION-ca cells)

GLOBAL declarations handling decision (CP3 acceptance criterion)
-----------------------------------------------------------------

unc2.mod NEURON block (line 19) declares:
  GLOBAL minf, hinf, mtau, htau, munc2, hunc2

This is a NMODL pitfall pattern. Of these six variables:

  minf, hinf, mtau, htau    derived assignments computed by rates(v) from
                            the cell's voltage. In NEURON, GLOBAL means
                            "shared across all instances" — so if multiple
                            sections insert unc2, only the LAST-touched
                            cell's rates(v) values would be visible until
                            the next call. However, rates(v) is called at
                            every DERIVATIVE timestep before m', h' are
                            evaluated, so each instance's per-step
                            integration sees the correct (just-computed)
                            values. **Functionally harmless in NEURON.**

  munc2, hunc2              copies of m, h written in BREAKPOINT after
                            integration, evidently for diagnostic export.
                            Same single-cell-functional-correctness story.

The actual STATE block declares `STATE { m h }` — these are RANGE-by-default
per NMODL convention, so per-instance per-section. The GLOBAL declarations
on the derived assignments and exposed copies are a NMODL pitfall that
doesn't affect single-cell correctness but would corrupt multi-cell readouts
of these specific assigned variables. Within Nicoletti's published model
(which only inserts unc2 in single-section cells), the GLOBAL declarations
are inert.

Brian2 handling
~~~~~~~~~~~~~~~

In Brian2, every NeuronGroup variable is per-cell by default. We treat
unc2_minf, unc2_hinf, unc2_mtau, unc2_htau as per-cell assigned variables
(`: 1` declarations in EQS). munc2, hunc2 don't need explicit handling —
m_unc2 and h_unc2 are the equivalent per-cell state vars; we expose them
directly via the StateMonitor instead of via the NMODL "munc2/hunc2"
diagnostic copy convention.

For our **single-cell-per-NeuronGroup** Layer A validation harness, this
matches NEURON exactly. For future **multi-cell-per-Brian2-NeuronGroup**
deployment, Brian2 provides correct per-cell semantics for free — we never
inherit the NMODL GLOBAL pitfall surprise in our translation. **Decision:
no special handling needed.** Document explicitly here for posterity.

F18 awareness
-------------
This channel reads eca via USEION. In RIM (3 USEION ca: cca1+unc2+egl19)
NEURON's ion_style silently overrides any user-set seg.eca. cell builder
MUST pass explicit eca_mV (e.g. 127.59 for celsius=6.3, cai=5e-5, cao=2).
See cellular_validation_findings.md F18 entry.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Parameter dictionary (defaults from unc2.mod)
# ---------------------------------------------------------------------------

UNC2_PARAMS = {
    # Activation steady-state
    "va_unc2":     -12.17,
    "ka_unc2":     3.97,
    "stm2":        25.0,
    # Inactivation steady-state
    "vi_unc2":     -52.47,
    "ki_unc2":     5.6,
    "sth2":        25.0,
    # Activation tau (sum-of-exp denominator)
    "p1tmunc2":    1.4969,
    "p2tmunc2":    -8.1761,
    "p3tmunc2":    9.0753,
    "p4tmunc2":    15.3456,
    "p5tmunc2":    0.1029,
    "shiftmunc2":  30.0,
    "fp3":         1.0,
    "fp4":         1.0,
    "constmunc2":  3.0,
    # Inactivation tau (two-Boltzmann sum)
    "p1thunc2":    83.8037,
    "p2thunc2":    52.8997,
    "p3thunc2":    3.4557,
    "p4thunc2":    72.0995,
    "p5thunc2":    23.9009,
    "p6thunc2":    3.5903,
    "shifthunc2":  30.0,
    "fp5":         1.0,
    "consthunc2":  1.7,
    # Conductance + reversal
    "gbar_unc2_Scm2": 1.0,
    "eca_mV":      60.0,
}


# ---------------------------------------------------------------------------
# Brian2 equation block
# ---------------------------------------------------------------------------

UNC2_EQS = """
# UNC-2 P/Q-type Ca channel: m, h gates (voltage-only).
# Steady-state:
unc2_minf = 1.0 / (1.0 + exp(-(v_mV - unc2_va + unc2_stm2) / unc2_ka)) : 1
unc2_hinf = 1.0 / (1.0 + exp((v_mV - unc2_vi + unc2_sth2) / unc2_ki)) : 1
# Activation tau (sum-of-exponentials in denominator):
unc2_mtau = (
    unc2_p1tm
    / (
        exp(-(v_mV - unc2_p2tm + unc2_shiftm) / (unc2_p3tm * unc2_fp3))
        + exp((v_mV - unc2_p2tm + unc2_shiftm) / (unc2_p4tm * unc2_fp4))
    )
    + unc2_p5tm
) * unc2_constm : 1
# Inactivation tau (two-Boltzmann sum):
unc2_htau = (
    unc2_p1th / (1.0 + exp((v_mV - unc2_p2th + unc2_shifth) / (unc2_p3th * unc2_fp5)))
    + unc2_p4th / (1.0 + exp(-(v_mV - unc2_p5th + unc2_shifth) / (unc2_p6th * unc2_fp5)))
) * unc2_consth : 1
# State variables (gating fractions 0..1):
dm_unc2/dt = (unc2_minf - m_unc2) / (unc2_mtau * ms) : 1
dh_unc2/dt = (unc2_hinf - h_unc2) / (unc2_htau * ms) : 1
# Channel current density (mA/cm²): ica = gbar * m * h * (v - eca)
ica_unc2_mAcm2 = unc2_gbar * m_unc2 * h_unc2 * (v_mV - unc2_eca) : 1
# Parameters:
unc2_va : 1
unc2_ka : 1
unc2_stm2 : 1
unc2_vi : 1
unc2_ki : 1
unc2_sth2 : 1
unc2_p1tm : 1
unc2_p2tm : 1
unc2_p3tm : 1
unc2_p4tm : 1
unc2_p5tm : 1
unc2_shiftm : 1
unc2_fp3 : 1
unc2_fp4 : 1
unc2_constm : 1
unc2_p1th : 1
unc2_p2th : 1
unc2_p3th : 1
unc2_p4th : 1
unc2_p5th : 1
unc2_p6th : 1
unc2_shifth : 1
unc2_fp5 : 1
unc2_consth : 1
unc2_gbar : 1
unc2_eca : 1
"""


def unc2_apply_params(group, gbar_Scm2: float | None = None,
                      eca_mV: float | None = None,
                      params_override: dict | None = None) -> None:
    """Apply UNC-2 parameters to a Brian2 NeuronGroup whose eqs include UNC2_EQS."""
    p = dict(UNC2_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_unc2_Scm2"] = gbar_Scm2
    if eca_mV is not None:
        p["eca_mV"] = eca_mV

    name_map = {
        "va_unc2":         "unc2_va",
        "ka_unc2":         "unc2_ka",
        "stm2":            "unc2_stm2",
        "vi_unc2":         "unc2_vi",
        "ki_unc2":         "unc2_ki",
        "sth2":            "unc2_sth2",
        "p1tmunc2":        "unc2_p1tm",
        "p2tmunc2":        "unc2_p2tm",
        "p3tmunc2":        "unc2_p3tm",
        "p4tmunc2":        "unc2_p4tm",
        "p5tmunc2":        "unc2_p5tm",
        "shiftmunc2":      "unc2_shiftm",
        "fp3":             "unc2_fp3",
        "fp4":             "unc2_fp4",
        "constmunc2":      "unc2_constm",
        "p1thunc2":        "unc2_p1th",
        "p2thunc2":        "unc2_p2th",
        "p3thunc2":        "unc2_p3th",
        "p4thunc2":        "unc2_p4th",
        "p5thunc2":        "unc2_p5th",
        "p6thunc2":        "unc2_p6th",
        "shifthunc2":      "unc2_shifth",
        "fp5":             "unc2_fp5",
        "consthunc2":      "unc2_consth",
        "gbar_unc2_Scm2":  "unc2_gbar",
        "eca_mV":          "unc2_eca",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def unc2_init_states(group, v_mV: float = -60.0) -> None:
    """Initialize m_unc2, h_unc2 to voltage-clamped steady states."""
    import numpy as np
    p = UNC2_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_unc2"] + p["stm2"]) / p["ka_unc2"]))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vi_unc2"] + p["sth2"]) / p["ki_unc2"]))
    group.m_unc2 = float(minf)
    group.h_unc2 = float(hinf)


# Standard interface
NAME = "unc2"
EQS = UNC2_EQS
apply_params = unc2_apply_params
init_states = unc2_init_states
