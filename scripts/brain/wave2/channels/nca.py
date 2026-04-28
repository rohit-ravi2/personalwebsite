"""
NCA NALCN-homolog non-specific leak channel — Brian2 translation of nca.mod.

Phase β run #2 Phase C.3 deliverable.

Source: nicoletti_2024/nca.mod
Citation: Nicoletti et al. 2019/2024

Channel structure
-----------------

NCA is a "passive leak" with no gates — a constant-conductance non-specific
current with reversal e=30 mV. NMODL: `i = gbar * (v - e)`.

This is the simplest possible channel model. There are no state variables.

Parameters:
  gbar = 0.055 S/cm² (default)
  e    = 30 mV
"""
from __future__ import annotations


NCA_PARAMS = {
    "gbar_nca_Scm2": 0.055,
    "e_nca_mV":      30.0,
}


# NCA produces ik_nca_mAcm2 (we treat it as a "K-channel-like" current variable
# from the validate_phase_c_channels harness POV — the validator sums to ik_total_mAcm2).
# Strictly NCA is non-specific (sodium-leak); naming as ik_nca_mAcm2 is purely
# convention to match the validator's expected interface.

NCA_EQS = """
# NCA non-specific leak: i = gbar * (v - e). No gates.
ik_nca_mAcm2 = nca_gbar * (v_mV - nca_e) : 1
nca_gbar : 1
nca_e : 1
"""


def nca_apply_params(group, gbar_Scm2: float | None = None,
                     ek_mV: float | None = None,  # ignored; NCA uses its own e
                     params_override: dict | None = None) -> None:
    p = dict(NCA_PARAMS)
    if gbar_Scm2 is not None:
        p["gbar_nca_Scm2"] = gbar_Scm2
    # ek_mV is for the validator's standard interface; NCA has its own e (=30 mV)

    setattr(group, "nca_gbar", p["gbar_nca_Scm2"])
    setattr(group, "nca_e", p["e_nca_mV"])

    if params_override:
        for k, v in params_override.items():
            setattr(group, k, v)


def nca_init_states(group, v_mV: float = -60.0) -> None:
    # No state variables; nothing to init.
    pass


# Standard interface
NAME = "nca"
EQS = NCA_EQS
apply_params = nca_apply_params
init_states = nca_init_states
