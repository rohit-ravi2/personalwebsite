"""
SLO-2 Ca-activated K channel — Brian2 module.

SLO-2 is the C. elegans Slack/Slick (KCNT) family ortholog: a K+ channel
activated by intracellular Ca²⁺ (and Na⁺, but Ca dominant for our purposes).
Voltage-independent (unlike SLO-1/BK which is voltage + Ca dependent).

Critical functional role in our substrate: SLO-2 provides negative feedback
on Ca accumulation. When [Ca]_in rises, SLO-2 opens, providing outward K
current → hyperpolarizes → closes voltage-gated Ca channels → Ca is cleared.

Kinetic model (Hill-form Ca-activation, voltage-independent):
  ik = gbar · m_Ca · (v - ek)
  m_Ca = (Ca_in / K_Ca)^n / (1 + (Ca_in / K_Ca)^n)

We use a fast equilibrium approximation (no separate τ for Ca-gating, since
SLO-2 Ca-binding is fast relative to substrate timescales).

References:
  - Yuan et al. 2003, Nature 426:570 — SLO-2 cloning, Ca-activation
  - Liu et al. 2014 — gating mechanism in C. elegans
  - Canonical SK family K_Ca ~ 0.5-2 μM; Hill coefficient n ~ 2-4

Default parameters (SK-family conservative):
  K_Ca = 1 μM = 1.0e-3 mM (typical SK Ca half-activation)
  n_Ca = 4 (Hill coefficient — typical for SK family)
  ek = -80 mV (overridden via dynamic Nernst bridge)
  gbar = derived per-cell from γ × TPM × C_global (γ = 20 pS)
"""
from __future__ import annotations


SLO2_PARAMS = {
    "K_Ca_slo2_mM":    1.0e-3,    # 1 μM Hill K_d
    "n_Ca_slo2":       4.0,        # Hill coefficient
    "gbar_slo2_Scm2":  1.0e-4,
    "ek_mV":          -80.0,
}


SLO2_EQS = """
# SLO-2 Ca-activated K channel — Hill Ca-activation, voltage-independent.
# Ca_in is the cell-state intracellular Ca concentration (mM).
slo2_ratio = Ca_in / slo2_K_Ca : 1
slo2_m = slo2_ratio**slo2_n_Ca / (1.0 + slo2_ratio**slo2_n_Ca) : 1
ik_slo2_mAcm2 = slo2_gbar * slo2_m * (v_mV - slo2_ek) : 1
# Parameters:
slo2_K_Ca : 1
slo2_n_Ca : 1
slo2_gbar : 1
slo2_ek : 1
"""


def slo2_apply_params(group, gbar_Scm2: float | None = None,
                       ek_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    p = dict(SLO2_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_slo2_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "K_Ca_slo2_mM":    "slo2_K_Ca",
        "n_Ca_slo2":       "slo2_n_Ca",
        "gbar_slo2_Scm2":  "slo2_gbar",
        "ek_mV":           "slo2_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def slo2_init_states(group, v_mV: float = -60.0) -> None:
    # No state variables (m is computed instantaneously from Ca_in).
    pass


NAME = "slo2"
EQS = SLO2_EQS
apply_params = slo2_apply_params
init_states = slo2_init_states
