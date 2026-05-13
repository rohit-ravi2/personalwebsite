"""
EGL-36 voltage-gated K channel — Brian2 module.

EGL-36 is the C. elegans Kv3 (Shaw) family ortholog. Heavily expressed in
pharyngeal neurons (I3, M3, MC), motor circuitry (PQR, ALN, PDB, AVL),
AVE, and ~29 total CeNGEN classes above T2.

Kv3 family properties:
  - Fast activation, depolarization-shifted (V_half ~ +10 mV)
  - Important for high-frequency repolarization (Kv3 enables fast firing)
  - Largely non-inactivating in C. elegans (egl-36 mutants show
    hyperexcitability per Johnstone 1997)

Kinetic model (m^4 Hodgkin-Huxley, non-inactivating Kv3):
  ik = gbar · m^4 · (v - ek)
  minf = 1 / (1 + exp(-(v - va) / ka))
  mtau: fast, ~1-3 ms

References:
  - Johnstone et al. 1997, Cell 88:147-156 — EGL-36 cloning + mutant phenotype
  - Wei et al. 1996 — Kv3 family in C. elegans
  - Rudy & McBain 2001 — mammalian Kv3 V_half + slope

Default parameters (Kv3 canonical):
  va = 10 mV (depolarization-activated)
  ka = 8 mV (slope)
  mtau = 1.5 ms (fast Kv3 activation)
  ek = -80 mV (overridden via dynamic Nernst bridge)
  gbar = derived per-cell from γ × TPM × C_global (γ = 16 pS)

EGL-36 contributes little at resting V; primary role is action-potential
repolarization.
"""
from __future__ import annotations


EGL36_PARAMS = {
    "va_egl36":         10.0,
    "ka_egl36":          8.0,
    "mtau_egl36":        1.5,
    "gbar_egl36_Scm2":   1.0e-4,
    "ek_mV":           -80.0,
}


EGL36_EQS = """
# EGL-36 Kv3 Shaw-family K channel, m^4 non-inactivating.
egl36_minf = 1.0 / (1.0 + exp(-(v_mV - egl36_va) / egl36_ka)) : 1
dm_egl36/dt = (egl36_minf - m_egl36) / (egl36_mtau * ms) : 1
ik_egl36_mAcm2 = egl36_gbar * m_egl36 * m_egl36 * m_egl36 * m_egl36 * (v_mV - egl36_ek) : 1
# Parameters:
egl36_va : 1
egl36_ka : 1
egl36_mtau : 1
egl36_gbar : 1
egl36_ek : 1
"""


def egl36_apply_params(group, gbar_Scm2: float | None = None,
                        ek_mV: float | None = None,
                        params_override: dict | None = None) -> None:
    p = dict(EGL36_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_egl36_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "va_egl36":         "egl36_va",
        "ka_egl36":         "egl36_ka",
        "mtau_egl36":       "egl36_mtau",
        "gbar_egl36_Scm2":  "egl36_gbar",
        "ek_mV":            "egl36_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def egl36_init_states(group, v_mV: float = -60.0) -> None:
    import numpy as np
    p = EGL36_PARAMS
    minf = 1.0 / (1.0 + np.exp(-(v_mV - p["va_egl36"]) / p["ka_egl36"]))
    group.m_egl36 = float(minf)


NAME = "egl36"
EQS = EGL36_EQS
apply_params = egl36_apply_params
init_states = egl36_init_states
