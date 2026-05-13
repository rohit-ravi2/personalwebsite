"""
EXP-2 voltage-gated K channel — Brian2 module.

EXP-2 is a C. elegans Kv-family K channel critical for repolarization
of pharyngeal + neuronal action potentials. Heavily expressed in HSN,
RIB, AVE, ALA, and several other neuron classes per CeNGEN T2.

Distinctive features:
  - Direct C. elegans single-channel measurement: γ = 67 ± 2 pS
    (Davis et al. 2006, Genetics 174:1399-1410). Only channel in our
    Layer 1 inventory with direct C. elegans single-channel γ.
  - N-type fast inactivation: behaves inward-rectifier-like in macroscopic
    currents because both activation and inactivation are fast.
    Shtonda + Avery 2005, Davis 2006.

Kinetic model (m·h Hodgkin-Huxley):
  ik = gbar · m · h · (v - ek)
  minf  = 1 / (1 + exp(-(v - va) / ka))
  hinf  = 1 / (1 + exp((v - vi) / ki))
  mtau, htau: fast (ms-scale per Davis 2006 single-channel)

Default parameters (approximation from C. elegans literature; γ direct):
  va = -20 mV, ka = 8 mV
  vi = -35 mV, ki = 6 mV
  mtau = 3 ms, htau = 5 ms
  ek = -80 mV (overridden via dynamic Nernst bridge in cell builder)
  gbar = derived per-cell from γ × TPM × C_global

Epistemic label: kinetics are approximation; γ is direct. Audit 4
applies: refine kinetics when cell-specific data emerges.
"""
from __future__ import annotations


EXP2_PARAMS = {
    "va_exp2":          -20.0,
    "ka_exp2":           8.0,
    "vi_exp2":          -35.0,
    "ki_exp2":           6.0,
    "mtau_exp2":         3.0,
    "htau_exp2":         5.0,
    "gbar_exp2_Scm2":    1.0e-4,   # overridden per-cell
    "ek_mV":           -80.0,
}


EXP2_EQS = """
# EXP-2 Kv-family K channel with fast N-type inactivation.
exp2_minf = 1.0 / (1.0 + exp(-(v_mV - exp2_va) / exp2_ka)) : 1
exp2_hinf = 1.0 / (1.0 + exp((v_mV - exp2_vi) / exp2_ki)) : 1
dm_exp2/dt = (exp2_minf - m_exp2) / (exp2_mtau * ms) : 1
dh_exp2/dt = (exp2_hinf - h_exp2) / (exp2_htau * ms) : 1
ik_exp2_mAcm2 = exp2_gbar * m_exp2 * h_exp2 * (v_mV - exp2_ek) : 1
# Parameters:
exp2_va : 1
exp2_ka : 1
exp2_vi : 1
exp2_ki : 1
exp2_mtau : 1
exp2_htau : 1
exp2_gbar : 1
exp2_ek : 1
"""


def exp2_apply_params(group, gbar_Scm2: float | None = None,
                       ek_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    p = dict(EXP2_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_exp2_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va_exp2":          "exp2_va",
        "ka_exp2":          "exp2_ka",
        "vi_exp2":          "exp2_vi",
        "ki_exp2":          "exp2_ki",
        "mtau_exp2":        "exp2_mtau",
        "htau_exp2":        "exp2_htau",
        "gbar_exp2_Scm2":   "exp2_gbar",
        "ek_mV":            "exp2_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def exp2_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = EXP2_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_exp2"]) / p["ka_exp2"]))
    hinf = 1.0 / (1.0 + np.exp((v_mV - p["vi_exp2"]) / p["ki_exp2"]))
    group.m_exp2 = float(minf)
    group.h_exp2 = float(hinf)


NAME = "exp2"
EQS = EXP2_EQS
apply_params = exp2_apply_params
init_states = exp2_init_states
