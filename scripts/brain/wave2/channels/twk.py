"""
TWK K2P (two-pore domain) background K channel — Brian2 module.

C. elegans has a large TWK family (~46 genes, twk-1 through twk-50ish);
the four most-relevant per CeNGEN T2 expression are TWK-7, TWK-18, TWK-30,
TWK-40. We aggregate their TPMs into a single TWK channel module
(family-level, like IRK paralogs in path2_scale).

K2P channels are constitutive K leak: no voltage-dependent gating, no
inactivation. Open at all voltages, providing a steady outward K current
proportional to (V - E_K).

Kinetic model: pure passive K leak.
  ik = gbar · (v - ek)

This is the simplest possible channel — no state variables, no gating.

References:
  - Salkoff et al. 2005, J Exp Biol 208:2317 — C. elegans TWK family review
  - Buckingham + Sattelle 2009 — invertebrate K2P channels
  - Canonical K2P γ ≈ 40 pS (matches our extended_gamma inventory)

Default parameter:
  ek = -80 mV (overridden via dynamic Nernst bridge in cell builder)
  gbar = derived per-cell from γ × ΣTPM × C_global (γ = 40 pS)
"""
from __future__ import annotations


TWK_PARAMS = {
    "gbar_twk_Scm2":   1.0e-4,   # overridden per-cell
    "ek_mV":          -80.0,
}


TWK_EQS = """
# TWK K2P background K channel — no gating.
ik_twk_mAcm2 = twk_gbar * (v_mV - twk_ek) : 1
# Parameters:
twk_gbar : 1
twk_ek : 1
"""


def twk_apply_params(group, gbar_Scm2: float | None = None,
                       ek_mV: float | None = None,
                       params_override: dict | None = None) -> None:
    p = dict(TWK_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_twk_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "gbar_twk_Scm2":   "twk_gbar",
        "ek_mV":           "twk_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def twk_init_states(group, v_mV: float = -60.0) -> None:
    # No state variables; nothing to initialize.
    pass


NAME = "twk"
EQS = TWK_EQS
apply_params = twk_apply_params
init_states = twk_init_states
