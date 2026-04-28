"""
SLO-1 isolated with DYNAMIC [Ca]_i — variant of slo1_iso.py.

Created as a Phase F follow-on after density-sensitivity sweep produced
VERDICT_AMPLITUDE_TUNABLE_DURATION_FAILS (see
`artifacts/density_sensitivity_analysis.md`). The mechanism diagnosis from F12
is that SLO-1 isolated reads bulk `cai_static = 5e-5 mM` — a constant — so its
gating cannot mediate Ca-feedback even though the kinetic formula has Ca-
dependence built in.

This module is identical to `slo1_iso.py` EXCEPT that the eqs does NOT declare
`cai_mM : 1` as a parameter. Instead, `cai_mM` is expected to be defined as a
*state variable* by an attached Ca-pool subsystem (e.g., caintra1 from
`calcium_pool.py`). The kinetic formula then sees the dynamically-evolving
[Ca]_i rather than a static parameter.

The original `slo1_iso.py` is preserved untouched so prior validations and
Phase F's published 2b run continue to behave identically.

Usage
-----
The cell builder must:
  1. Splice `caintra1_eqs()["eqs"]` (or any other pool exposing `cai_mM` state)
     into the parent NeuronGroup eqs.
  2. Splice `SLO1_ISO_DYNAMIC_CA_EQS` (this module) instead of
     `slo1_iso.SLO1_ISO_EQS`.
  3. Apply parameters via `slo1iso_dynca_apply_params` (which differs from the
     static variant by NOT setting `cai_mM` — the pool's init handles that).
  4. Initialize `m_slo1iso` to its SS at the *initial* [Ca]_i (= ca_eq for
     caintra1, = 5e-8 mM per its NMODL default).

This is THE load-bearing edit for the Ca-coupling hypothesis.
"""
from __future__ import annotations


# Reuse the same parameter dict as the static variant — only the eqs differ.
# Import patterns in this codebase: callers add `wave2/` to sys.path, so
# `from channels.slo1_iso import ...` works whether this file is run as a
# script, imported as a module, or imported via the `channels` package.
try:
    from channels.slo1_iso import SLO1_ISO_PARAMS  # noqa: F401
except ImportError:  # fall back to relative import when used as package
    from .slo1_iso import SLO1_ISO_PARAMS  # noqa: F401


# Eqs identical to SLO1_ISO_EQS except `cai_mM : 1` is removed (cai_mM is
# defined by the pool subsystem as a state variable).
SLO1_ISO_DYNAMIC_CA_EQS = """
# SLO-1 isolated BK channel — DYNAMIC [Ca]_i variant.
# cai_mM is provided as a STATE by an attached Ca-pool (e.g., caintra1).
# The kinetic formula now sees cai evolving with EGL-19's Ca current.
slo1iso_ca_uM = cai_mM * 1000.0 : 1
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
dm_slo1iso/dt = (slo1iso_minf - m_slo1iso) / (slo1iso_mtau * ms) : 1
ik_slo1iso_mAcm2 = slo1iso_gbar * m_slo1iso * (v_mV - slo1iso_ek) : 1
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
"""


def slo1iso_dynca_apply_params(group, gbar_Scm2: float | None = None,
                                ek_mV: float | None = None,
                                params_override: dict | None = None) -> None:
    """Apply SLO-1 isolated parameters when [Ca]_i is supplied dynamically.

    Differs from `slo1_iso.slo1iso_apply_params` only by NOT setting `cai_mM` —
    the pool subsystem owns that state.
    """
    p = dict(SLO1_ISO_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_slo1iso_Scm2"] = gbar_Scm2
    if ek_mV is not None:
        p["ek_mV"] = ek_mV

    name_map = {
        "wom":               "slo1iso_wom",
        "wyx":               "slo1iso_wyx",
        "kyx":               "slo1iso_kyx",
        "nyx":               "slo1iso_nyx",
        "wop":               "slo1iso_wop",
        "wxy":               "slo1iso_wxy",
        "kxy":               "slo1iso_kxy",
        "nxy":               "slo1iso_nxy",
        "c1":                "slo1iso_c1",
        "gbar_slo1iso_Scm2": "slo1iso_gbar",
        "ek_mV":             "slo1iso_ek",
    }
    for src, dst in name_map.items():
        setattr(group, dst, p[src])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def slo1iso_dynca_init_states(group, v_mV: float = -60.0,
                              cai_mM_init: float = 5e-8) -> None:
    """Initialize m_slo1iso to its SS at v_mV using cai_mM_init.

    Pass cai_mM_init = ca_eq of the Ca-pool (caintra1 default 5e-8 mM).
    """
    import numpy as np
    p = SLO1_ISO_PARAMS
    ca_uM = cai_mM_init * 1000.0
    s0 = 1.0 / (p["wyx"] - p["wxy"])
    v0 = s0 * (
        np.log(p["wom"] / p["wop"])
        + np.log(1.0 + (p["kxy"] / ca_uM) ** p["nxy"])
        - np.log(1.0 + (ca_uM / p["kyx"]) ** p["nyx"])
    )
    minf = 1.0 / (1.0 + np.exp(-(v_mV - v0) / s0))
    group.m_slo1iso = float(minf)


# Standard interface
NAME = "slo1iso_dynamic_ca"
EQS = SLO1_ISO_DYNAMIC_CA_EQS
apply_params = slo1iso_dynca_apply_params
init_states = slo1iso_dynca_init_states
